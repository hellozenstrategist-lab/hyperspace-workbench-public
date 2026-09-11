"""Reproducible five-mode Coordinator benchmark and read-only evidence metrics.

Only run_benchmark starts models. Aggregation and markdown rendering do not.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import asyncio
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics

from .coordinator import Coordinator, default_context_budget
from .evidence_audit import audit_run
from .hyperbolic_index import distance
from .router import MODES
from .schema import AGENTS, worker_ids, atomic_json
from .runtime_factory import resolve_model
from .concurrency import concurrency_report

MODEL = "example/default-model"
CONTEXT_BUDGET = 3000
REJECTION_LIMIT = 3
VALIDATION_POLICY_VERSION = "model_tool_validation_v1"
MAX_MODE_ATTEMPTS = 3


def _object(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def _seconds(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ValueError("nonfinite time")
        return float(value)
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timestamps require a timezone")
    return stamp.timestamp()


def _distribution(values):
    values = list(values)
    if not values:
        return {"status": "NOT_MEASURED", "count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {"status": "MEASURED", "count": len(values), "min": min(values), "median": statistics.median(values), "mean": statistics.mean(values), "max": max(values)}


def _usage(ledger, agents=AGENTS):
    snapshots = {agent: [] for agent in agents}
    for row in ledger:
        if row["kind"] == "token_usage" and row["data"].get("agent") in snapshots:
            usage = row["data"].get("usage")
            if isinstance(usage, dict):
                snapshots[row["data"]["agent"]].append({"at": row["at"], "usage": usage})
    agents = {}
    def valid_counts(value):
        return isinstance(value, dict) and isinstance(value.get("totalTokens"), int) and not isinstance(value.get("totalTokens"), bool) and value["totalTokens"] >= 0
    for agent, rows in snapshots.items():
        final = rows[-1] if rows else {}
        usage = final.get("usage", {})
        valid = valid_counts(usage.get("total")) and valid_counts(usage.get("last"))
        agents[agent] = {"status": "MEASURED" if valid else "INVALID" if rows else "NOT_MEASURED", "update_count": len(rows), "latest_at": final.get("at"), "total": usage.get("total"), "last": usage.get("last"), "model_context_window": usage.get("modelContextWindow")}
    complete = all(row["status"] == "MEASURED" for row in agents.values())
    sums = {}
    for kind in ("total", "last"):
        rows = [row[kind] for row in agents.values() if valid_counts(row[kind])]
        keys = set.intersection(*(set(row) for row in rows)) if rows else set()
        sums[kind] = {key: sum(row[key] for row in rows) for key in sorted(keys) if all(isinstance(row[key], int) and not isinstance(row[key], bool) and row[key] >= 0 for row in rows)}
    return {"status": "MEASURED" if complete else "PARTIAL" if any(snapshots.values()) else "NOT_MEASURED", "per_agent": agents, "aggregate_final_totals": sums["total"] if complete else None, "aggregate_latest_requests": sums["last"] if complete else None, "available_agent_totals": sums["total"], "method": "Use the last cumulative total snapshot per worker, then sum workers. Never sum cumulative update notifications. 'last' means each worker's latest request, not its whole run."}


def _correction(fixture, ledger, final_checks, agents=AGENTS):
    assumption = fixture.get("planted_assumption")
    if not isinstance(assumption, dict) or not assumption.get("claim") or not assumption.get("candidate"):
        return {"status": "NOT_MEASURED", "reason": "This fixture has no explicitly recorded planted false assumption.", "per_agent": {}, "all_workers_corrected_after_start_seconds": None}
    expected = assumption.get("expected_candidate", fixture.get("expected_candidate"))
    prompt_matches = all(assumption["claim"] in fixture.get("private_prompts", {}).get(a, "") for a in agents)
    if not prompt_matches or expected == assumption["candidate"]:
        return {"status": "NOT_MEASURED", "reason": "Planted claim is not recorded in all initial prompts, or its expected correction is ambiguous.", "per_agent": {}, "all_workers_corrected_after_start_seconds": None}
    starts = {r["data"].get("agent"): _seconds(r["at"]) for r in ledger if r["kind"] == "worker_started"}
    corrected = {}
    for row in ledger:
        agent = row["data"].get("agent")
        if row["kind"] != "agent_message" or row["data"].get("phase") not in ("final", "final_answer") or agent not in starts or agent in corrected:
            continue
        try:
            output = _object(row["data"]["text"])
        except (ValueError, TypeError, KeyError):
            continue
        if output.get("chosen_candidate") == expected and all(final_checks.get(agent, {}).values()) and final_checks.get(agent):
            elapsed = _seconds(row["at"]) - starts[agent]
            if elapsed >= 0:
                corrected[agent] = {"seconds_from_own_worker_start": elapsed, "observed_at": row["at"], "candidate": expected}
    all_corrected = set(corrected) == set(agents)
    return {"status": "MEASURED_FINAL_BASED" if all_corrected else "PARTIAL_FINAL_BASED", "planted_claim": assumption["claim"], "per_agent": corrected, "all_workers_corrected_after_start_seconds": max(_seconds(row["observed_at"]) for row in corrected.values()) - min(starts.values()) if all_corrected else None, "scope": "Time until externally observed correct final answers. This does not establish that a model ever believed the planted claim or when an internal belief changed."}


def _validation_policy(manifest):
    runtime = manifest.get("runtime", manifest.get("runtime_manifest", {}))
    runtime = runtime if isinstance(runtime, dict) else {}
    limit = manifest.get("tool_input_rejection_limit", runtime.get("tool_input_rejection_limit"))
    version = manifest.get("tool_validation_policy_version", runtime.get("tool_validation_policy_version"))
    details = manifest.get("tool_input_rejection_policy", runtime.get("tool_input_rejection_policy"))
    return {"status": "RECORDED" if limit is not None and version else "NOT_RECORDED", "limit_per_worker": limit, "version": version, "details": details}


def _tool_validation(ledger, manifest, task_success):
    policy = _validation_policy(manifest)
    attempts = {(r["data"].get("agent"), r["data"].get("call_id")) for r in ledger if r["kind"] in ("tool_call", "tool_rejected") and r["data"].get("call_id")}
    rejected_ids = {(r["data"].get("agent"), r["data"].get("call_id")) for r in ledger if r["kind"] == "tool_rejected"}
    seen, pending, episodes = set(), {}, []
    per_agent = {a: {"rejections": 0, "recovered_rejections": 0, "successful_retry_calls": 0} for a in worker_ids(manifest.get("worker_count", 3))}
    for row in ledger:
        data = row["data"]
        key = (data.get("agent"), data.get("tool"))
        call = (data.get("agent"), data.get("call_id"))
        if row["kind"] == "tool_rejected" and call not in seen:
            seen.add(call)
            episode = pending.setdefault(key, {"agent": key[0], "tool": key[1], "rejected_call_ids": [], "first_rejected_at": row["at"], "recovered": False})
            episode["rejected_call_ids"].append(data.get("call_id"))
            if key[0] in per_agent:
                per_agent[key[0]]["rejections"] += 1
        elif row["kind"] == "tool_completed" and key in pending and call not in rejected_ids:
            episode = pending.pop(key)
            episode.update(recovered=True, successful_retry_call_id=data.get("call_id"), successful_retry_at=row["at"], seconds_to_successful_retry=_seconds(row["at"])-_seconds(episode["first_rejected_at"]))
            episodes.append(episode)
            if key[0] in per_agent:
                per_agent[key[0]]["recovered_rejections"] += len(episode["rejected_call_ids"])
                per_agent[key[0]]["successful_retry_calls"] += 1
    episodes.extend(pending.values())
    recovered = sum(len(episode["rejected_call_ids"]) for episode in episodes if episode["recovered"])
    instrumented = policy["status"] == "RECORDED" or bool(seen)
    return {"instrumentation": "RECORDED" if instrumented else "LEGACY_NO_REJECTION_EVENT_POLICY", "policy": policy, "unique_tool_attempts": len(attempts), "rejected_calls": len(seen), "rejection_rate": len(seen)/len(attempts) if instrumented and attempts else None, "recovered_rejected_calls": recovered, "recovery_rate_per_rejected_call": recovered/len(seen) if seen else None, "recovered_retry_episodes": sum(episode["recovered"] for episode in episodes), "unrecovered_retry_episodes": sum(not episode["recovered"] for episode in episodes), "episodes": episodes, "per_agent": per_agent, "task_succeeded_after_rejections": bool(seen) and bool(task_success), "scope": "A recovery is a later successful call of the same tool by the same worker, with a different call ID. It is separate from final task success. Same-ID rejection replay is not another rejection. Legacy fatal validation failures are reported separately, never silently treated as zero rejected inputs."}


def _failure_trace(ledger):
    errors = [row for row in ledger if row["kind"] == "error"]
    if not errors:
        return {"status": "NO_FATAL_ERROR_RECORDED"}
    first = errors[0]
    earlier = []
    for row in ledger:
        if row is first:
            break
        if row["kind"] == "tool_call" and row["data"].get("agent") == first["data"].get("agent"):
            earlier.append(row)
    tool = earlier[-1] if earlier else None
    return {"status": "FATAL_ERROR_RECORDED", "first_error": {"at": first["at"], **first["data"]}, "nearest_preceding_same_worker_tool": {"at": tool["at"], **tool["data"]} if tool else None, "association_is_sequence_inference": bool(tool), "interrupted_workers": sorted({row["data"].get("agent") for row in ledger if row["kind"] == "worker_completed" and row["data"].get("status") == "interrupted"}), "submitted_arguments": "NOT_RECORDED_IN_LEDGER", "scope": "The error identifies the failed validation stage. An argument digest cannot identify the exact submitted code or which receipt failed."}


def _aggregate_core(snapshot: dict, fixture: dict, ledger: list[dict], audit: dict) -> dict:
    """Pure deterministic aggregation. Audit is computed independently by caller."""
    run = snapshot["run"]
    manifest = _object(run["manifest"])
    agents = worker_ids(manifest.get("worker_count", 3))
    ledger = [{**row, "data": _object(row["data"])} for row in ledger if row.get("run_id") == run["run_id"]]
    events = {event["event_id"]: event for event in snapshot.get("events", [])}
    deliveries = snapshot.get("deliveries", [])
    required = fixture.get("required_event_ids", {})
    required_ids = set(required.values())
    expected_pairs = {(ident, recipient) for sender, ident in required.items() for recipient in agents if recipient != sender}
    by_pair = {(row["event_id"], row["recipient"]): row for row in deliveries}
    checks = dict(audit.get("checks", {}))
    final_key = "all_finals_prove_correct_incorporation"
    ack_key = "all_actual_model_code_acknowledgements"
    checks.setdefault(final_key, checks.get("all_three_finals_prove_correct_incorporation", False))
    checks.setdefault(ack_key, checks.get("six_actual_model_code_acknowledgements", False))
    final_checks = audit.get("details", {}).get(final_key, audit.get("details", {}).get("all_three_finals_prove_correct_incorporation", {}))
    publications = {row["data"]["event"]["event_id"]: row for row in ledger if row["kind"] == "published" and isinstance(row["data"].get("event"), dict)}
    latencies, invalid_latency = [], 0
    for row in deliveries:
        if row["event_id"] not in required_ids or not row.get("incorporated_at"):
            continue
        try:
            elapsed = _seconds(row["incorporated_at"]) - _seconds(publications[row["event_id"]]["at"])
            if elapsed < 0:
                raise ValueError("negative latency")
            latencies.append({"event_id": row["event_id"], "recipient": row["recipient"], "seconds": elapsed})
        except (ValueError, TypeError, KeyError):
            invalid_latency += 1
    required_paths = {tuple(events[ident].get("scope_path", [])) for ident in required_ids if ident in events}
    categories = Counter()
    category_costs = Counter()
    per_agent_branches = {agent: Counter() for agent in agents}
    accepted = [r for r in deliveries if r.get("accepted_at") or r.get("delivered_at")]
    for row in accepted:
        event = events.get(row["event_id"], {})
        path = tuple(event.get("scope_path", []))
        if row["event_id"] in required_ids:
            category = "required_useful"
        elif path and any(path[:len(p)] == p or p[:len(path)] == path for p in required_paths):
            category = "same_branch_optional_unproven_usefulness"
        elif path and required_paths and not any(path[:len(p)] == p or p[:len(path)] == path for p in required_paths):
            category = "wrong_branch_irrelevant_to_fixture"
        else:
            category = "unclassified"
        categories[category] += 1
        category_costs[category] += row.get("token_cost", 0)
        if row["recipient"] in per_agent_branches and path:
            per_agent_branches[row["recipient"]]["/".join(path)] += 1
    anchors = {}
    for row in ledger:
        if row["kind"] == "attention_region" and row["data"].get("agent") in agents:
            anchors[row["data"]["agent"]] = row["data"]
    pairs = []
    for left, right in itertools.combinations(sorted(anchors), 2):
        pairs.append({"left": left, "right": right, "distance": distance(anchors[left]["position"], anchors[right]["position"])})
    branch_diversity = {}
    for agent, counts in per_agent_branches.items():
        total = sum(counts.values())
        entropy = -sum((n/total) * math.log2(n/total) for n in counts.values()) if total else None
        branch_diversity[agent] = {"unique_scope_paths": len(counts), "scope_path_counts": dict(sorted(counts.items())), "entropy_bits": entropy}
    normalized = []
    for ident in sorted(required_ids):
        if ident in events:
            claim = events[ident].get("claim", "").casefold()
            for code in fixture.get("nonces", {}).values():
                claim = claim.replace(str(code).casefold(), "<private-code>")
            normalized.append(" ".join(claim.split()))
    duplicates = sum(n-1 for n in Counter(normalized).values() if n > 1)
    transitions = Counter((row["kind"], row["data"].get("delivery_id")) for row in ledger if row["kind"] in ("accepted", "delivered", "acknowledged", "incorporated"))
    failures = {"missing_durable_delivery_pairs": sum(pair not in by_pair for pair in expected_pairs)}
    for stage in ("accepted", "delivered", "acknowledged", "incorporated"):
        failures["missing_required_" + stage] = sum(not by_pair.get(pair, {}).get(stage + "_at") for pair in expected_pairs)
    failures.update(required_terminal_failed_or_expired=sum(by_pair.get(pair, {}).get("state") in ("failed", "expired") for pair in expected_pairs), optional_terminal_failed_or_expired=sum(r["event_id"] not in required_ids and r.get("state") in ("failed", "expired") for r in deliveries), unresolved_uncertain=sum(r.get("state") == "uncertain" for r in deliveries))
    finals = {}
    for agent, row in snapshot.get("workers", {}).items():
        try:
            finals[agent] = _object(row.get("final", "")).get("chosen_candidate")
        except (ValueError, TypeError, KeyError):
            finals[agent] = None
    correct = {agent: bool(facts.get("correct_candidate")) for agent, facts in final_checks.items()}
    return {
        "run_id": run["run_id"], "mode": manifest.get("routing_mode", manifest.get("mode")),
        "worker_count": len(agents), "required_message_pairs": len(expected_pairs), "run_status": run.get("status"),
        "independent_audit_pass": bool(audit.get("core_pass")), "audit_failures": audit.get("failures", []),
        "metric_integrity_verified": bool(checks.get("ledger_hash_chain") and checks.get("snapshot_events_match_hashed_publications")),
        "task_success": bool(checks[final_key]), "correct_final_workers": sum(correct.values()), "worker_final_count": len(finals),
        "peer_only_evidence_code_incorporation": bool(checks.get("no_initial_peer_code_preknowledge") and checks.get("private_prompts_match_runtime_inputs") and checks[ack_key] and checks[final_key]),
        "common_active_turn_overlap_seconds": audit.get("active_turn_overlap_seconds"),
        "concurrency": concurrency_report(ledger, agents),
        "publication_to_incorporation_seconds": {**_distribution(row["seconds"] for row in latencies), "per_delivery": latencies, "invalid_or_missing_timestamps": invalid_latency, "scope": "Required peer publication to coordinator-validated incorporation of actual final answers; includes model/tool waits."},
        "accepted_context": {"unique_claim_recipient_pairs": len({(r["event_id"], r["recipient"]) for r in accepted}), "runtime_accepted_pairs": sum(bool(r.get("accepted_at")) for r in deliveries), "tool_delivered_pairs": sum(bool(r.get("delivered_at")) for r in deliveries), "categories": dict(categories), "estimated_cost_by_category": dict(category_costs), "estimated_cost_total": sum(category_costs.values()), "scope": "A claim enters this count after runtime acceptance or inbox delivery; repeated stages count once. Required claims are useful by fixture construction. Same-branch optional claims have unproven usefulness. Costs are the harness byte-based context estimate, not provider tokens."},
        "duplicate_work_proxies": {"repeated_required_claim_wording_after_code_removal": duplicates, "repeated_delivery_rows": len(deliveries)-len(by_pair), "repeated_state_transitions": sum(n-1 for n in transitions.values() if n > 1), "scope": "Wording and transport duplication only; semantic duplicate work and compute avoided are NOT_MEASURED."},
        "runtime_usage": _usage(ledger, agents), "planted_assumption_correction": _correction(fixture, ledger, final_checks, agents),
        "attention_diversity": {"pairwise_anchor_distances": pairs, "pairwise_distance_summary": _distribution(pair["distance"] for pair in pairs), "distance_source": "exact_poincare_from_recorded_worker_anchors", "accepted_scope_diversity": branch_diversity, "distinct_final_candidates": len({candidate for candidate in finals.values() if candidate is not None}), "scope": "Structural/input and final-output proxies; hidden reasoning diversity is NOT_MEASURED. More branch entropy may mean more irrelevant context."},
        "failures": failures, "controls": {key: manifest.get(key) for key in ("model", "provider", "worker_count", "reasoning_effort", "context_budget_per_worker", "fixture", "index_backend", "photon_enabled", "coordinate_method", "semantic_method")}}


def aggregate_run(snapshot: dict, fixture: dict, ledger: list[dict], audit: dict) -> dict:
    """Pure metrics including validation recovery and retained failure evidence."""
    metrics = _aggregate_core(snapshot, fixture, ledger, audit)
    manifest = _object(snapshot["run"]["manifest"])
    rows = [{**row, "data": _object(row["data"])} for row in ledger if row.get("run_id") == snapshot["run"]["run_id"]]
    metrics["tool_validation"] = _tool_validation(rows, manifest, metrics["task_success"])
    metrics["failure_trace"] = _failure_trace(rows)
    policy = _validation_policy(manifest)
    metrics["controls"].update(tool_input_rejection_limit=policy["limit_per_worker"], tool_validation_policy_version=policy["version"])
    metrics["controls"]["receipt_policy_version"] = manifest.get("receipt_policy_version", "NOT_RECORDED")
    return metrics


def summarize_run(run_dir) -> dict:
    directory = Path(run_dir).resolve()
    audit = audit_run(directory)
    try:
        snapshot = json.loads((directory / "snapshot.json").read_text())
        fixture = json.loads((directory / "fixture.json").read_text())
        ledger = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines() if line.strip()]
        metrics = aggregate_run(snapshot, fixture, ledger, audit)
    except Exception as exc:
        metrics = {"independent_audit_pass": False, "task_success": False, "aggregation_status": "FAILED", "aggregation_error": f"{type(exc).__name__}: {exc}"}
    return {"directory": str(directory), "audit": audit, "metrics": metrics}


def _fixture_signature(fixture, manifest):
    replacements = [str(value) for key in ("nonces", "evidence_refs", "required_event_ids") for value in fixture.get(key, {}).values()]
    if fixture.get("run_id"):
        replacements.append(fixture["run_id"])
    prompts = {}
    for agent, prompt in fixture.get("private_prompts", {}).items():
        for value in sorted(replacements, key=len, reverse=True):
            prompt = prompt.replace(value, "<run-specific-value>")
        prompts[agent] = prompt
    comparable = {"fixture_version": manifest.get("fixture"), "measurements": fixture.get("measurements"), "expected_candidate": fixture.get("expected_candidate"), "planted_assumption": fixture.get("planted_assumption"), "normalized_private_prompts": prompts}
    return hashlib.sha256(json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_controls(manifest, mode, expected_model=MODEL, *, require_policy=False, expected_policy=None,
                       worker_count=3, provider="codex"):
    expected = {"routing_mode": mode, "model": expected_model, "reasoning_effort": "low" if provider == "codex" else "provider_default", "context_budget_per_worker": default_context_budget(worker_count), "index_backend": "hyperspace"}
    mismatches = {key: {"expected": value, "actual": manifest.get(key)} for key, value in expected.items() if manifest.get(key) != value}
    for key, expected, default in (("worker_count", worker_count, 3), ("provider", provider, "codex")):
        if manifest.get(key, default) != expected:
            mismatches[key] = {"expected": expected, "actual": manifest.get(key, default)}
    if require_policy:
        policy = _validation_policy(manifest)
        for key, value in (("limit_per_worker", REJECTION_LIMIT),):
            if policy.get(key) != value:
                mismatches["tool_validation_policy." + key] = {"expected": value, "actual": policy.get(key)}
        if not policy.get("version"):
            mismatches["tool_validation_policy.version"] = {"expected": "an explicitly recorded policy version", "actual": None}
        if expected_policy is not None and _policy_key(policy) != _policy_key(expected_policy):
            mismatches["tool_validation_policy.scope"] = {"expected": expected_policy, "actual": policy}
    if mismatches:
        raise ValueError("benchmark controls differ: " + json.dumps(mismatches, sort_keys=True))


def render_markdown(report: dict) -> str:
    model = report.get("controls", {}).get("model", "NOT_RECORDED")
    count = report.get("controls", {}).get("workers_per_group", 3)
    allowance = report.get("controls", {}).get("context_allowance_per_worker", default_context_budget(count))
    effort = report.get("controls", {}).get("reasoning_effort", "low")
    lines = ["# Five-mode harness benchmark", "", "Status: **" + report["status"] + f"**. First attempts and any explicit retrials are listed separately; each run requests {count} concurrent workers and modes execute sequentially. Measured overlap is recorded in each run's concurrency metrics.", "", f"All modes use `{model}`, `{effort}` reasoning, and a {allowance}-unit per-worker harness context allowance. Equal allowance does not mean identical consumed tokens. The flat baseline uses deterministic hashed lexical vectors, not learned semantic embeddings.", "", "| Mode | Independent audit | Correct workers | Peer-code incorporation | Median publication → incorporation (s) | Useful / irrelevant context pairs | Runtime total tokens | Final-based correction (s) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in MODES:
        entry = report.get("runs", {}).get(mode)
        if not entry:
            lines.append(f"| {mode} | NOT_RUN | — | — | — | — | — | — |")
            continue
        m = entry["metrics"]
        latency = m.get("publication_to_incorporation_seconds", {}).get("median")
        counts = m.get("accepted_context", {}).get("categories", {})
        tokens = (m.get("runtime_usage", {}).get("aggregate_final_totals") or {}).get("totalTokens")
        correction = m.get("planted_assumption_correction", {}).get("all_workers_corrected_after_start_seconds")
        def value(number):
            return "NOT_MEASURED" if number is None else f"{number:.3f}" if isinstance(number, float) else str(number)
        lines.append(f"| {mode} | {'PASS' if m.get('independent_audit_pass') else 'FAIL'} | {m.get('correct_final_workers', 0)}/{count} | {m.get('peer_only_evidence_code_incorporation', False)} | {value(latency)} | {counts.get('required_useful', 0)} / {counts.get('wrong_branch_irrelevant_to_fixture', 0)} | {value(tokens)} | {value(correction)} |")
    lines += ["", "Required messages are explicit dependencies in every mode; their success cannot by itself establish a geometry benefit. Optional branch probes measure selectivity separately. Same-branch optional claims are reported separately from proven-useful private constraints in the JSON.", "", "## Evidence and limits", "", "- Every row is recomputed from exported evidence with the independent auditor; saved success flags are not trusted.", "- Runtime token totals use each worker's latest cumulative report once. Per-worker cumulative and latest-request snapshots remain in the JSON; missing usage is not filled with zero.", "- Correction time means time to an observed correct final answer, not time of internal belief revision. It is NOT_MEASURED when no false assumption was explicitly planted.", "- Duplicate wording, repeated delivery states, anchor distances and branch entropy are proxies. They do not establish compute saved or hidden reasoning diversity.", "- Single runs, sequential order, random private codes and provider variation prevent causal quality or efficiency claims. Active-turn overlap includes tool waits; simultaneous provider inference is not established.", "", "## Structural distractor fixture", ""]
    geometry = report.get("adversarial_geometry_fixture", {})
    if geometry.get("status") == "AVAILABLE":
        lines.append("The separate constructed one-fixture benchmark used the actual Hyperspace index. Graph, hyperbolic and hybrid selected the relevant branch; flat lexical selected the repeated-word distractor; broadcast selected both. This illustrates a structural signal, not general superiority. See the embedded source artifact and receipt in benchmark.json.")
    else:
        lines.append("NOT_MEASURED: no existing adversarial geometry result was available.")
    lines += ["", "## Hypotheses for repeated evaluation", "", "1. Structural routing may reduce irrelevant accepted context relative to broadcast/flat lexical on tasks with reliable branch information.", "2. Hybrid routing may retain required task success while reducing optional context; explicit dependencies remain necessary across branches.", "3. Hyperbolic routing may differ from the graph baseline at larger scale, but this small benchmark does not establish an advantage.", "", "Full per-run audits, timing samples, failures, usage snapshots and structural diversity metrics are in `benchmark.json`.", ""]
    lines += ["## Tool input validation and recovery", "", "Each attempt records its own validation policy version and scope. The allowance stays three explicit input-validation rejections per worker; the fourth unique rejected call is fatal. Changed retrial scope is listed separately from the first-attempt comparison. The model must supply a corrected call; no evidence code is silently substituted.", "", "| Mode | Tool attempts | Rejected calls | Rejection rate | Recovered / unrecovered retry episodes |", "| --- | --- | --- | --- | --- |"]
    for mode, entry in report.get("runs", {}).items():
        validation = entry.get("metrics", {}).get("tool_validation", {})
        rate = validation.get("rejection_rate")
        rendered_rate = "NOT_MEASURED" if rate is None else f"{rate:.1%}"
        lines.append(f"| {mode} | {validation.get('unique_tool_attempts', 'NOT_MEASURED')} | {validation.get('rejected_calls', 'NOT_MEASURED')} | {rendered_rate} | {validation.get('recovered_retry_episodes', 'NOT_MEASURED')} / {validation.get('unrecovered_retry_episodes', 'NOT_MEASURED')} |")
    lines += ["", "A successful retry is a later successful call of the same tool by that worker with a new call ID. It is counted separately from final task success. Replayed rejection IDs do not increase the rate. Historical runs without rejection-event instrumentation do not receive a fabricated zero rejection rate.", "", "## Previous attempts retained", ""]
    history = report.get("previous_attempts", [])
    if not history:
        lines.append("No earlier attempts were supplied to this comparison.")
    for attempt in history:
        lines.append("Prior evidence: `" + attempt["directory"] + "`. It is separate from the five current mode results and is not included in their pass counts.")
        past_modes = attempt.get("mode_attempts") or {mode: [entry] for mode, entry in attempt.get("runs", {}).items()}
        for mode, entries in past_modes.items():
            for index, entry in enumerate(entries, 1):
                metrics = entry.get("metrics", {})
                error = metrics.get("failure_trace", {}).get("first_error", {}).get("error")
                message = str(error).replace("\n", " ")[:220] if error else "No fatal error recorded."
                lines.append(f"- {mode}, attempt {entry.get('attempt_number', index)}: independent audit {'PASS' if entry.get('audit', {}).get('core_pass') else 'FAIL'}. {message}")
        for ancestor in attempt.get("ancestor_attempt_reports", []):
            outcomes = [f"{mode} attempt {entry['attempt_number']}: {'PASS' if entry['passed'] else 'FAIL'}" for mode, entries in ancestor.get("recorded_outcomes", {}).items() for entry in entries]
            lines.append("- Earlier referenced evidence: `" + str(ancestor.get("directory")) + "` (" + "; ".join(outcomes) + "). Its source hash is retained without recursively embedding the report.")
    lines.append("")
    if "attempt_summary" in report:
        summary = report["attempt_summary"]
        primary = ["# Attempt outcomes", "", f"First attempts: **{summary['first_attempt_modes_passed']}/{summary['first_attempt_modes_run']} modes passed**. Eventual selected outcomes: **{summary['eventually_passed_modes']}/5 passed**, using **{summary['total_attempts']} total attempts** ({summary['failed_attempts']} failed).", "", "Each retrial has another full allowance and may use revised validation scope. It is separate from the original equal-controls first attempt; it is not evidence of an equal-total-budget efficiency gain.", "", "| Mode | Attempt | Independent outcome | Validation policy | Tool rejections | Role |", "| --- | --- | --- | --- | --- | --- |"]
        for mode in MODES:
            for attempt in report.get("mode_attempts", {}).get(mode, []):
                validation = attempt.get("metrics", {}).get("tool_validation", {})
                version = validation.get("policy", {}).get("version", "NOT_RECORDED")
                number = attempt["attempt_number"]
                primary.append(f"| {mode} | {number} | {'PASS' if attempt['audit'].get('core_pass') else 'FAIL'} | {version} | {validation.get('rejected_calls', 'NOT_MEASURED')} | {'first attempt' if number == 1 else 'explicit retrial'} |")
        primary += ["", f"First-attempt validation policies equal: **{summary['first_attempt_validation_policies_equal']}**. Selected-result validation policies equal: **{summary['selected_result_validation_policies_equal']}**. Source hashes captured for new attempts and all prior failed evidence remain in the JSON.", ""]
        lines = primary + lines
    return "\n".join(lines)


def _previous_attempt(path):
    directory = Path(path).resolve()
    report_path = directory / "benchmark.json"
    if report_path.is_file():
        raw = report_path.read_bytes()
        previous = json.loads(raw)
        runs, attempts = {}, {}
        for mode in sorted(set(previous.get("runs", {})) | set(previous.get("mode_attempts", {}))):
            selected = previous.get("runs", {}).get(mode, {})
            recorded = previous.get("mode_attempts", {}).get(mode) or [selected]
            attempts[mode] = []
            by_path = {}
            for number, entry in enumerate(recorded, 1):
                candidate = Path(entry.get("directory", directory / mode))
                if not candidate.is_absolute():
                    candidate = directory / candidate
                candidate = candidate.resolve()
                summary = _summarize_attempt(candidate, entry.get("attempt_number", number))
                attempts[mode].append(summary)
                by_path[str(candidate)] = summary
            selected_path = Path(selected.get("directory", directory / mode))
            if not selected_path.is_absolute():
                selected_path = directory / selected_path
            runs[mode] = by_path.get(str(selected_path.resolve()), attempts[mode][-1])
        # Retain original sources and their recorded outcomes without recursively
        # embedding entire already-embedded reports. Growth is linear in sources.
        ancestors = {}
        for ancestor in previous.get("previous_attempts", []):
            key = (ancestor.get("directory"), ancestor.get("source_report_sha256"))
            old_modes = ancestor.get("mode_attempts") or {mode: [entry] for mode, entry in ancestor.get("runs", {}).items()}
            ancestors[key] = {"directory": ancestor.get("directory"), "source_report_sha256": ancestor.get("source_report_sha256"), "recorded_status": ancestor.get("recorded_status"), "recorded_outcomes": {mode: [{"attempt_number": entry.get("attempt_number", index), "passed": bool(entry.get("audit", {}).get("core_pass")), "directory": entry.get("directory")} for index, entry in enumerate(entries, 1)] for mode, entries in old_modes.items()}, "scope": "Lightweight reference to original preserved evidence, not recursive report embedding."}
            for older in ancestor.get("ancestor_attempt_reports", []):
                ancestors[(older.get("directory"), older.get("source_report_sha256"))] = older
        return {"directory": str(directory), "source_report_sha256": hashlib.sha256(raw).hexdigest(), "recorded_status": previous.get("status"), "recorded_execution_error": previous.get("execution_error"), "runs": runs, "mode_attempts": attempts, "ancestor_attempt_reports": list(ancestors.values()), "included_in_current_mode_pass_counts": False, "scope": "Every directly supplied attempt is re-audited, including failed originals and successful retrials. Older sources are flattened references. Historical attempts do not enter current mode pass counts."}
    if (directory / "snapshot.json").is_file():
        summary = summarize_run(directory)
        return {"directory": str(directory), "runs": {summary["metrics"].get("mode", "unknown"): summary}, "included_in_current_mode_pass_counts": False}
    raise ValueError("previous attempt has no benchmark or run evidence: " + str(directory))


def _policy_key(policy):
    return json.dumps({key: policy.get(key) for key in ("limit_per_worker", "version", "details")}, sort_keys=True)


def _attempt_paths(directory, mode, previous):
    found = {}
    base = directory / mode
    if base.exists():
        found[1] = base
    for path in directory.glob(mode + "-attempt-*"):
        match = re.fullmatch(re.escape(mode) + r"-attempt-([2-9][0-9]*)", path.name)
        if match and path.is_dir():
            found[int(match[1])] = path
    recorded = previous.get("mode_attempts", {}).get(mode, [])
    if not recorded and previous.get("runs", {}).get(mode):
        recorded = [previous["runs"][mode]]
    for entry in recorded:
        path = Path(entry["directory"]).resolve()
        number = entry.get("attempt_number", 1)
        if number not in found:
            found[number] = path
        elif found[number].resolve() != path:
            raise ValueError("conflicting recorded mode attempt paths")
    if found and (sorted(found) != list(range(1, max(found) + 1)) or max(found) > MAX_MODE_ATTEMPTS):
        raise ValueError("mode attempt history must be contiguous and within the fixed bound")
    return sorted(found.items())


def _summarize_attempt(path, number):
    summary = summarize_run(path)
    summary["attempt_number"] = number
    summary["outcome"] = "PASSED" if summary["audit"].get("core_pass") else "FAILED"
    control_file = Path(path) / "attempt_control.json"
    summary["source_control"] = json.loads(control_file.read_text()) if control_file.exists() else {"status": "NOT_RECORDED_FOR_HISTORICAL_ATTEMPT"}
    return summary


def _record_attempt(report, mode, entry):
    attempts = report.setdefault("mode_attempts", {}).setdefault(mode, [])
    attempts[:] = [item for item in attempts if item["attempt_number"] != entry["attempt_number"]]
    attempts.append(entry)
    attempts.sort(key=lambda item: item["attempt_number"])
    successful = [item for item in attempts if item["audit"].get("core_pass")]
    report["runs"][mode] = successful[0] if successful else attempts[-1]


def _refresh_attempt_summary(report):
    modes = report.get("mode_attempts", {})
    all_attempts = [entry for attempts in modes.values() for entry in attempts]
    first = {mode: attempts[0] for mode, attempts in modes.items() if attempts}
    first_policies = {mode: entry.get("metrics", {}).get("tool_validation", {}).get("policy", {}) for mode, entry in first.items()}
    selected_policies = {mode: entry.get("metrics", {}).get("tool_validation", {}).get("policy", {}) for mode, entry in report.get("runs", {}).items()}
    report["attempt_summary"] = {"total_attempts": len(all_attempts), "passed_attempts": sum(bool(entry["audit"].get("core_pass")) for entry in all_attempts), "failed_attempts": sum(not entry["audit"].get("core_pass") for entry in all_attempts), "first_attempt_modes_run": len(first), "first_attempt_modes_passed": sum(bool(entry["audit"].get("core_pass")) for entry in first.values()), "eventually_passed_modes": sum(bool(entry["audit"].get("core_pass")) for entry in report.get("runs", {}).values()), "per_mode_attempt_counts": {mode: len(attempts) for mode, attempts in modes.items()}, "all_attempts_passed": bool(all_attempts) and all(entry["audit"].get("core_pass") for entry in all_attempts), "first_attempt_validation_policies_equal": len({_policy_key(policy) for policy in first_policies.values()}) <= 1, "selected_result_validation_policies_equal": len({_policy_key(policy) for policy in selected_policies.values()}) <= 1, "first_attempt_validation_policies": first_policies, "selected_result_validation_policies": selected_policies, "scope": "First attempts remain the primary equal-allowance comparison. Successful retrials are separate eventual outcomes with additional model calls and possibly revised validation scope; they are not equal-total-budget wins."}
    receipts = {mode: entry.get("metrics", {}).get("controls", {}).get("receipt_policy_version", "NOT_RECORDED") for mode, entry in first.items()}
    report["attempt_summary"]["first_attempt_receipt_policies"] = receipts
    receipt_complete = bool(receipts) and all(value != "NOT_RECORDED" for value in receipts.values())
    report["attempt_summary"]["first_attempt_receipt_policies_equal"] = len(set(receipts.values())) <= 1 if receipt_complete else None
    report["attempt_summary"]["receipt_policy_comparison_complete"] = receipt_complete


def _write_report(directory, report):
    _refresh_attempt_summary(report)
    atomic_json(directory / "benchmark.json", report)
    temporary = directory / ".benchmark.md.tmp"
    temporary.write_text(render_markdown(report))
    temporary.replace(directory / "benchmark.md")


async def _run_benchmark(output_dir, endpoint, hybrid_run=None, *, model=MODEL, previous_attempts=None, retry_failed=False,
                         worker_count=3, provider="codex") -> dict:
    """Execute at most one explicitly requested new attempt per failed mode."""
    directory = Path(output_dir).resolve()
    old_path = directory / "benchmark.json"
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    for key, expected, default in (("workers_per_group", worker_count, 3), ("provider", provider, "codex"), ("model", model, MODEL)):
        if old and old.get("controls", {}).get(key, default) != expected:
            raise ValueError(f"Cannot change benchmark {key} when resuming; use a new output directory")
    context_budget = default_context_budget(worker_count)
    report_receipts = list(old.get("prior_report_receipts", []))
    if old_path.exists():
        raw = old_path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        archive = directory / "report_history" / (checksum + ".json")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_bytes(raw)
        report_receipts.append({"sha256": checksum, "path": str(archive)})
    report = {"schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(), "status": "RUNNING", "runs": {}, "mode_attempts": {}, "prior_report_receipts": report_receipts,
        "controls": {"model": model, "provider": provider, "reasoning_effort": "low" if provider == "codex" else "provider_default", "context_allowance_per_worker": context_budget,
            "workers_per_group": worker_count, "groups_sequential": True, "new_runs_photon_enabled": False,
            "initial_attempts_per_mode": 1, "maximum_total_attempts_per_mode": MAX_MODE_ATTEMPTS,
            "explicit_retry_requested": bool(retry_failed), "new_attempts_per_failed_mode_this_invocation": 1 if retry_failed else 0,
            "tool_input_rejection_limit": REJECTION_LIMIT,
            "token_budget_interpretation": "equal allowance per attempt, never equal total tokens after a retrial",
            "flat_baseline": "deterministic local hashed lexical vectors; not learned semantics"},
        "hypotheses": ["Structure may filter wrong-branch lexical distractors.", "Required dependencies may retain success while optional context decreases.", "A hyperbolic advantage over graph routing remains unproven."],
        "adversarial_geometry_fixture": {"status": "NOT_MEASURED"}}
    history = {entry["directory"]: entry for entry in old.get("previous_attempts", [])}
    for path in previous_attempts or []:
        entry = _previous_attempt(path)
        history[entry["directory"]] = entry
    report["previous_attempts"] = list(history.values())
    geometry = Path(__file__).resolve().parents[1] / "artifacts" / "geometry_benchmark.json"
    if geometry.exists():
        raw = geometry.read_bytes()
        report["adversarial_geometry_fixture"] = {"status": "AVAILABLE", "source": str(geometry), "sha256": hashlib.sha256(raw).hexdigest(), "result": json.loads(raw)}
    signature = None
    first_policy = None
    for mode in MODES:
        coordinator = None
        run_dir, attempt_number = directory / mode, 1
        try:
            paths = _attempt_paths(directory, mode, old)
            if mode == "hybrid" and hybrid_run and not paths:
                paths = [(1, Path(hybrid_run).resolve())]
            unfinished = None
            for number, path in paths:
                if not (path / "snapshot.json").exists():
                    unfinished = (number, path)
                    continue
                snapshot = json.loads((path / "snapshot.json").read_text())
                manifest = _object(snapshot["run"]["manifest"])
                fixture = json.loads((path / "fixture.json").read_text())
                _validate_controls(manifest, mode, model, require_policy=True, worker_count=worker_count, provider=provider)
                if number == 1:
                    policy = _validation_policy(manifest)
                    if first_policy is None:
                        first_policy = policy
                    elif _policy_key(policy) != _policy_key(first_policy):
                        raise ValueError("first-attempt validation policies differ; record the changed scope as an explicit retrial")
                current = _fixture_signature(fixture, manifest)
                if signature is None:
                    signature = current
                elif current != signature:
                    raise ValueError("attempt fixture differs; preserve it as separate historical evidence")
                entry = _summarize_attempt(path, number)
                entry["execution"] = "retained_existing_attempt"
                _record_attempt(report, mode, entry)
                run_dir, attempt_number = path, number
            selected = report["runs"].get(mode)
            if selected and selected["audit"].get("core_pass"):
                selected["execution"] = "reused_completed_run"
                _write_report(directory, report)
                continue
            if selected:
                if not retry_failed:
                    raise RuntimeError("failed attempt preserved; explicitly use --retry-failed for one bounded new trial")
                if attempt_number >= MAX_MODE_ATTEMPTS:
                    raise RuntimeError("fixed maximum of three attempts for this mode has been reached")
                state_path = run_dir / "runtime_state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    terminal = {"completed", "interrupted", "failed", "cancelled", "canceled"}
                    if any(worker.get("status") not in terminal for worker in state.get("workers", {}).values()):
                        raise RuntimeError("prior worker state is not terminal; recover/cancel it before a new trial")
                elif selected["metrics"].get("run_status") not in {"failed", "completed"}:
                    raise RuntimeError("prior attempt terminal state cannot be verified")
                attempt_number += 1
                run_dir = directory / (mode + "-attempt-" + str(attempt_number))
                if run_dir.exists():
                    raise RuntimeError("retry directory already exists; its state must be reviewed without overwrite")
            elif unfinished:
                attempt_number, run_dir = unfinished
            coordinator = Coordinator(run_dir, mode=mode, endpoint=endpoint, notifier=None, context_budget=context_budget,
                                      worker_count=worker_count, provider=provider, model=model)
            _validate_controls(coordinator.manifest, mode, model, worker_count=worker_count, provider=provider)
            current = _fixture_signature(coordinator.fixture, coordinator.manifest)
            if signature is None:
                signature = current
            elif current != signature:
                raise ValueError("attempt fixture differs; use separate history for a changed task")
            report["fixture_signature"] = signature
            controls = run_dir / "attempt_control.json"
            if not controls.exists() and not (run_dir / "snapshot.json").exists():
                source = Path(__file__).parent
                atomic_json(controls, {"created_at": datetime.now(timezone.utc).isoformat(), "attempt_number": attempt_number,
                    "model": model, "provider": provider, "worker_count": worker_count, "context_allowance_per_worker": context_budget,
                    "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in ("benchmark.py", "coordinator.py", "openrouter_runtime.py" if provider == "openrouter" else "codex_runtime.py", "router.py", "hyperbolic_index.py")},
                    "scope": "Source hashes captured before this attempt; historical unrecorded sources are not inferred."})
            _write_report(directory, report)
            await coordinator.run()
            _validate_controls(coordinator.manifest, mode, model, require_policy=True, worker_count=worker_count, provider=provider)
            if attempt_number == 1:
                policy = _validation_policy(coordinator.manifest)
                if first_policy is None:
                    first_policy = policy
                elif _policy_key(policy) != _policy_key(first_policy):
                    raise ValueError("first-attempt validation policies differ; this result cannot join the primary comparison")
            entry = _summarize_attempt(run_dir, attempt_number)
            entry["execution"] = "explicit_retrial" if attempt_number > 1 else "executed_or_resumed_coordinator"
            _record_attempt(report, mode, entry)
            if not entry["audit"].get("core_pass"):
                raise RuntimeError("attempt failed independent core audit; no automatic retry")
        except asyncio.CancelledError:
            report["status"] = "CANCELLED"
            report["stopped_at_mode"] = mode
            if (run_dir / "snapshot.json").exists():
                _record_attempt(report, mode, _summarize_attempt(run_dir, attempt_number))
            _write_report(directory, report)
            raise
        except Exception as exc:
            if (run_dir / "snapshot.json").exists():
                _record_attempt(report, mode, _summarize_attempt(run_dir, attempt_number))
            report.update(status="INCOMPLETE", stopped_at_mode=mode,
                execution_error={"type": type(exc).__name__, "message": str(exc)})
            _write_report(directory, report)
            return report
        finally:
            if coordinator:
                coordinator.close()
        _write_report(directory, report)
    _refresh_attempt_summary(report)
    report["status"] = "COMPLETED_WITH_RETRIALS" if report["attempt_summary"]["total_attempts"] > len(MODES) else "COMPLETED"
    report["all_modes_independently_passed"] = all(report["runs"][mode]["audit"].get("core_pass") for mode in MODES)
    report["all_core_pass"] = report["all_modes_independently_passed"]
    report["all_core_pass_scope"] = "At least one independently successful attempt per mode; inspect first attempts and retrials separately."
    _write_report(directory, report)
    return report


async def run_benchmark(output_dir, endpoint, hybrid_run=None, *, model=None, previous_attempts=None, retry_failed=False,
                        worker_count=3, provider="codex") -> dict:
    """One-command entry point; the model must match every Coordinator manifest.

    Worker count, provider, model and per-worker allowance remain constant
    across all modes and are checked before an existing run is reused.
    """
    if not endpoint or endpoint == "local":
        raise ValueError("benchmark requires the explicit actual HyperspaceDB endpoint")
    worker_ids(worker_count)
    model = resolve_model(provider, model)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (directory / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("benchmark output directory already has an active owner") from exc
        return await _run_benchmark(directory, endpoint, hybrid_run, model=model, previous_attempts=previous_attempts,
                                    retry_failed=retry_failed, worker_count=worker_count, provider=provider)
