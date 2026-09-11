"""Independent, read-only audit of exported harness evidence; no model calls.

Saved pass flags are ignored. The ledger hash chain is recomputed here rather
than delegated to the writer. Hash chains detect alteration, not a privileged
operator rewriting and rehashing an entire history.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

AGENTS = {"agent-a", "agent-b", "agent-c"}
ALL_AGENTS = tuple("agent-" + letter for letter in "abcde")
MODEL = "example/default-model"
SUPPORTED_MODELS = frozenset({MODEL, "example/alternate-model"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _time(value):
    if isinstance(value, (float, int)):
        return float(value)
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Audit timestamps must have a timezone")
    return stamp.timestamp()


def _json_object(value):
    if isinstance(value, str):
        text = re.sub(r"^```(?:json)?\s*", "", value.strip())
        text = re.sub(r"\s*```$", "", text)
        value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def _chain(rows):
    previous = "0" * 64
    previous_seq = 0
    failures = []
    decoded = []
    for row in rows:
        data = _json_object(row["data"])
        body = {name: row[name] for name in ("run_id", "kind", "at", "prev_hash", "idem_key")}
        body["data"] = data
        expected = _hash(_canonical(body).encode())
        if row["prev_hash"] != previous or row["hash"] != expected or row["seq"] != previous_seq + 1:
            failures.append(row["seq"])
        previous, previous_seq = row["hash"], row["seq"]
        decoded.append({**row, "data": data})
    return {"valid": bool(rows) and not failures, "invalid_sequences": failures,
            "head": previous, "entries": len(rows)}, decoded


def _http_requests(groups, agents, requested_model):
    """Recompute request provenance and overlap independently of the runtime."""
    starts = groups.get("request_started", [])
    completed = groups.get("request_completed", [])
    by_id = {}
    valid = bool(starts)
    for row in starts:
        data = row["data"]
        ident = data.get("request_id")
        if not isinstance(ident, str) or not ident or ident in by_id or data.get("agent") not in agents:
            valid = False
            continue
        by_id[ident] = row
    closed, successful_agents, changes, response_ids = set(), set(), [], set()
    successful = rate_limited = 0
    for row in completed:
        data = row["data"]
        ident = data.get("request_id")
        first = by_id.get(ident)
        if not first or ident in closed or data.get("agent") != first["data"].get("agent"):
            valid = False
            continue
        closed.add(ident)
        start = _time(first["data"].get("started_at", first["at"]))
        end = _time(data.get("completed_at", row["at"]))
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            valid = False
            continue
        agent = data["agent"]
        if end > start:
            changes.extend([(start, 1, agent), (end, -1, agent)])
        if data.get("status") == "completed":
            successful += 1
            successful_agents.add(agent)
            response_id = data.get("response_id")
            valid = valid and data.get("http_status") == 200 and data.get("model") == requested_model
            valid = valid and data.get("requested_model") == requested_model
            valid = valid and isinstance(response_id, str) and bool(response_id) and response_id not in response_ids
            response_ids.add(response_id)
        elif data.get("status") == "rate_limited" and data.get("http_status") == 429:
            rate_limited += 1
        else:
            valid = False
    active, peak, peak_workers, overlap, previous = Counter(), 0, 0, 0.0, None
    for stamp, delta, agent in sorted(changes):
        if previous is not None and all(active[a] > 0 for a in agents):
            overlap += stamp - previous
        active[agent] += delta
        peak = max(peak, sum(active.values()))
        peak_workers = max(peak_workers, sum(value > 0 for value in active.values()))
        previous = stamp
    valid = valid and closed == set(by_id) and successful_agents == agents
    return {"valid": bool(valid), "status": "MEASURED" if starts else "NOT_MEASURED",
            "request_count": len(starts), "completed_requests": len(closed), "successful_requests": successful,
            "rate_limited_requests": rate_limited, "incomplete_requests": len(set(by_id) - closed),
            "peak_in_flight_requests": peak, "peak_workers_with_requests": peak_workers,
            "all_workers_requests_overlap_seconds": round(overlap, 6),
            "scope": "Recorded client HTTP intervals and returned model identifiers; provider GPU inference overlap is not established."}


def _evidence_files(directory, references, manifest):
    if isinstance(manifest, list):
        manifest = {entry.get("hash", entry.get("sha256")): entry for entry in manifest}
    if isinstance(manifest, dict) and "evidence" in manifest:
        return _evidence_files(directory, references, manifest["evidence"])
    records = {}
    for checksum in references:
        entry = manifest.get(checksum) if isinstance(manifest, dict) else None
        candidates = []
        if entry:
            path_value = entry if isinstance(entry, str) else entry.get("path")
            if path_value:
                path = Path(path_value)
                candidates.append(path if path.is_absolute() else directory / path)
        candidates.extend([directory / "evidence" / checksum, directory.parent / "data/evidence" / checksum,
                           directory.parent.parent / "data/evidence" / checksum])
        path = next((path for path in candidates if path.is_file()), None)
        if path is None:
            records[checksum] = {"valid": False, "error": "content-addressed evidence file missing"}
            continue
        size = path.stat().st_size
        if size > 10_000_000:
            records[checksum] = {"valid": False, "error": "evidence exceeds declared schema bound"}
            continue
        actual = _hash(path.read_bytes())
        size_ok = not isinstance(entry, dict) or entry.get("size", size) == size
        records[checksum] = {"valid": actual == checksum and size_ok, "actual_sha256": actual,
                             "bytes": size, "path": str(path)}
    return records


def _photon(directory, manifest, run_id):
    setting = manifest.get("photon_enabled", manifest.get("photon", {}).get("enabled") if isinstance(manifest.get("photon"), dict) else None)
    photon_mode = manifest.get("photon_mode", manifest.get("photon"))
    enabled = setting is True or photon_mode in ("live", "enabled")
    path = directory / "photon_receipts.json"
    receipts = json.loads(path.read_text()) if path.exists() else []
    if isinstance(receipts, dict):
        receipts = receipts.get("receipts", [])
    receipts = [r for r in receipts if r.get("run_id", run_id) == run_id]
    sent = [r for r in receipts if r.get("status") in ("sent", "delivered")]
    if not enabled:
        return {"tested": False, "status": "NOT_TESTED", "pass": None,
                "disabled_has_no_sends": len(sent) == 0, "sent_count": len(sent),
                "limitation": "Photon was disabled; core harness success is separate from notification transport validation."}
    expected_kinds = manifest.get("photon_expected_kinds", ["start", "verified_important", "completed"])
    statuses = Counter(r.get("kind") for r in sent)
    keys = [r.get("key") for r in sent]
    valid = len(sent) == 3 and set(statuses) == set(expected_kinds) and all(n == 1 for n in statuses.values())
    valid = valid and len(set(keys)) == 3 and all(keys)
    return {"tested": True, "status": "PASSED" if valid else "FAILED", "pass": bool(valid),
            "sent_count": len(sent), "kinds": dict(statuses), "receipts": sent,
            "evidence_scope": "Recorded Photon provider acknowledgement; recipient-device receipt is not independently verified."}


def _replay(directory):
    path = directory / "replay.json"
    if not path.exists():
        return {"tested": False, "status": "NOT_TESTED", "pass": None}
    replay = json.loads(path.read_text())
    before, after = replay.get("before", {}), replay.get("after", {})
    aliases = {"turn_starts": ("turn_starts", "turns", "worker_started"),
               "deliveries": ("deliveries", "delivery_count", "message_published"),
               "notifications": ("notifications", "notification_count", "photon_sent")}
    checks = {}
    for name, options in aliases.items():
        key = next((k for k in options if k in before and k in after), None)
        checks[name] = key is not None and isinstance(before[key], int) and not isinstance(before[key], bool)
        checks[name] = checks[name] and before[key] >= 0 and before[key] == after[key]
    return {"tested": True, "status": "PASSED" if all(checks.values()) else "FAILED",
            "pass": all(checks.values()), "checks": checks, "before": before, "after": after,
            "evidence_scope": "Exported before/after counters; inspect lifecycle ledger and pinned source for the no-replay policy."}


def audit_run(run_dir):
    """Return audit findings for one exported run. Does not alter its artifacts."""
    directory = Path(run_dir).resolve()
    source_receipts = {}
    def load(name, required=True):
        path = directory / name
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        raw = path.read_bytes()
        source_receipts[name] = {"bytes": len(raw), "sha256": _hash(raw)}
        return raw
    try:
        snapshot = json.loads(load("snapshot.json"))
        fixture = json.loads(load("fixture.json"))
        rows = [json.loads(line) for line in load("ledger.jsonl").decode().splitlines() if line.strip()]
        evidence_raw = load("evidence_manifest.json", False)
        evidence_manifest = json.loads(evidence_raw) if evidence_raw else {}
        chain, all_rows = _chain(rows)
        run = snapshot["run"]
        run_id, manifest = run["run_id"], _json_object(run["manifest"])
        ledger = [row for row in all_rows if row["run_id"] == run_id]
        groups = {}
        for row in ledger:
            groups.setdefault(row["kind"], []).append(row)
        count = manifest.get("worker_count", 3)
        if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 5:
            raise ValueError("worker_count must be an integer from 2 through 5")
        agents = set(ALL_AGENTS[:count])
        pairs = count * (count - 1)
        names = {
            "exactly_three_workers": "exactly_configured_workers",
            "three_distinct_started_turns": "all_distinct_started_turns",
            "three_successful_completed_turns": "all_successful_completed_turns",
            "positive_three_worker_overlap": "positive_all_worker_overlap",
            "private_fixture_has_three_unique_codes": "private_fixture_has_unique_worker_codes",
            "three_required_publications_have_own_codes": "all_required_publications_have_own_codes",
            "six_required_directed_publications": "all_required_directed_publications",
            "all_six_message_publications_during_common_overlap": "all_message_publications_during_common_overlap",
            "six_required_durable_delivery_rows": "all_required_durable_delivery_rows",
            "six_actual_model_code_acknowledgements": "all_actual_model_code_acknowledgements",
            "all_three_finals_prove_correct_incorporation": "all_finals_prove_correct_incorporation",
        }
        checks, details = {}, {}
        def check(name, valid, detail=None):
            for key in ({name, names.get(name, name)} if count == 3 else {names.get(name, name)}):
                checks[key] = bool(valid)
                if detail is not None:
                    details[key] = detail
        check("ledger_hash_chain", chain["valid"], chain)
        check("run_completed", run.get("status") == "completed")
        workers = snapshot["workers"]
        check("exactly_three_workers", set(workers) == agents and len({w.get("thread_id") for w in workers.values()}) == count
              and all(w.get("thread_id") for w in workers.values())
              and set(manifest.get("worker_ids", agents)) == agents)
        runtime_manifest = manifest.get("runtime", manifest.get("runtime_manifest", manifest))
        requested_model = manifest.get("requested_model", manifest.get("model",
            runtime_manifest.get("requested_model", runtime_manifest.get("model", MODEL))))
        declarations = [value for value in (manifest.get("requested_model"), manifest.get("model"),
            runtime_manifest.get("requested_model"), runtime_manifest.get("model")) if value is not None]
        provider = manifest.get("provider", runtime_manifest.get("provider", "codex"))
        provider_valid = (provider == "codex" and runtime_manifest.get("auth") == "chatgpt" and requested_model in SUPPORTED_MODELS
                          or provider == "openrouter" and runtime_manifest.get("provider") == "openrouter"
                          and runtime_manifest.get("auth") == "openrouter_api_key" and isinstance(requested_model, str) and "/" in requested_model)
        check("chatgpt_requested_model_manifest" if provider == "codex" else "openrouter_requested_model_manifest", provider_valid
              and runtime_manifest.get("model") == requested_model
              and all(value == requested_model for value in declarations)
              and runtime_manifest.get("model_fallback", False) is False,
              {"requested_model": requested_model, "runtime_model": runtime_manifest.get("model"),
               "default_for_new_runs": MODEL, "historical_explicit_model": requested_model != MODEL})
        reroutes = groups.get("model_rerouted", [])
        fatal_errors = groups.get("error", []) + groups.get("runtime_error", [])
        check("no_reroutes_or_runtime_errors", not reroutes and not fatal_errors)
        http = _http_requests(groups, agents, requested_model) if provider == "openrouter" else {
            "status": "NOT_MEASURED", "scope": "Codex active turns do not expose HTTP request intervals."}
        if provider == "openrouter":
            check("openrouter_request_provenance", http["valid"], http)

        starts, completes = groups.get("worker_started", []), groups.get("worker_completed", [])
        start_by = {row["data"].get("agent"): row for row in starts}
        complete_by = {row["data"].get("agent"): row for row in completes}
        check("three_distinct_started_turns", len(starts) == count and set(start_by) == agents
              and len({row["data"].get("turn_id") for row in starts}) == count
              and all(row["data"].get("turn_id") for row in starts))
        check("three_successful_completed_turns", len(completes) == count and set(complete_by) == agents
              and all(row["data"].get("status") == "completed" and not row["data"].get("error") for row in completes)
              and all(complete_by[a]["data"].get("turn_id") == start_by.get(a, {}).get("data", {}).get("turn_id") for a in complete_by))
        windows = set(start_by) == agents and set(complete_by) == agents
        start_t = max(_time(row["at"]) for row in starts) if windows else 0
        end_t = min(_time(row["at"]) for row in completes) if windows else 0
        overlap = max(0, end_t - start_t)
        check("positive_three_worker_overlap", overlap > 0)

        nonces, private = fixture["nonces"], fixture["private_prompts"]
        required = fixture["required_event_ids"]
        check("private_fixture_has_three_unique_codes", set(nonces) == agents and len(set(nonces.values())) == count
              and set(private) == agents and set(required) == agents and len(set(required.values())) == count)
        check("no_initial_peer_code_preknowledge", all(nonces.get(a, "MISSING") in private.get(a, "") and
              all(code not in private.get(a, "") for b, code in nonces.items() if b != a) for a in agents))
        prompt_rows = groups.get("worker_prompt", [])
        check("private_prompts_match_runtime_inputs", len(prompt_rows) == count and {row["data"].get("agent") for row in prompt_rows} == agents and all(
              row["data"].get("prompt") == private.get(row["data"].get("agent")) for row in prompt_rows))

        events = {event["event_id"]: event for event in snapshot["events"]}
        publications = groups.get("published", [])
        publication_by = {row["data"]["event"]["event_id"]: row for row in publications}
        event_hashes = {}
        for event_id, event in events.items():
            written = publication_by.get(event_id, {}).get("data", {}).get("event")
            event_hashes[event_id] = {"snapshot_sha256": _hash(_canonical(event).encode()),
                "ledger_sha256": _hash(_canonical(written).encode()) if written else None,
                "valid": written == event}
        check("snapshot_events_match_hashed_publications", bool(events) and len(events) == len(snapshot["events"])
              and set(events) == set(publication_by) and all(v["valid"] for v in event_hashes.values()), event_hashes)
        check("three_required_publications_have_own_codes", all(required[a] in events
              and events[required[a]].get("author_agent") == a
              and nonces[a] in _canonical(events[required[a]]) for a in agents)
              and len([row for row in publications if row["data"]["event"]["event_id"] in required.values()]) == count)
        refs = {reference for event in events.values() for reference in event.get("evidence_refs", [])}
        evidence = _evidence_files(directory, refs, evidence_manifest)
        check("content_addressed_evidence_hashes", bool(refs) and all(record["valid"] for record in evidence.values()), evidence)

        required_ids = set(required.values())
        expected = {(required[sender], recipient) for sender in agents for recipient in agents if sender != recipient}
        messages = [row for row in groups.get("message_published", []) if row["data"].get("event_id") in required_ids]
        message_pairs = {(row["data"].get("event_id"), row["data"].get("recipient")) for row in messages}
        check("six_required_directed_publications", len(messages) == pairs and message_pairs == expected)
        check("all_six_message_publications_during_common_overlap", len(messages) == pairs
              and all(start_t <= _time(row["at"]) < end_t for row in messages))
        delivery_rows = [row for row in snapshot["deliveries"] if row.get("event_id") in required_ids]
        delivery_map = {row["delivery_id"]: row for row in delivery_rows}
        check("six_required_durable_delivery_rows", len(delivery_rows) == pairs and len(delivery_map) == pairs
              and {(row["event_id"], row["recipient"]) for row in delivery_rows} == expected
              and {row["data"].get("delivery_id") for row in messages} == set(delivery_map))
        transition_details = {}
        for delivery_id, delivery in delivery_map.items():
            recipient, event_id = delivery["recipient"], delivery["event_id"]
            transition_rows = {kind: [row for row in groups.get(kind, []) if row["data"].get("delivery_id") == delivery_id]
                               for kind in ("accepted", "delivered", "acknowledged", "incorporated")}
            valid = all(len(items) == 1 for items in transition_rows.values())
            times = []
            for kind, items in transition_rows.items():
                if len(items) != 1:
                    continue
                row = items[0]
                valid = valid and row["data"].get("event_id") == event_id and row["data"].get("recipient") == recipient
                timestamp = delivery.get(kind + "_at")
                valid = valid and bool(timestamp)
                if timestamp:
                    # Store timestamps and ledger timestamps are adjacent writes
                    # in one transaction, rather than guaranteed byte-identical.
                    valid = valid and abs(_time(timestamp) - _time(row["at"])) < 2.0
                times.append(_time(row["at"]))
            valid = valid and len(times) == 4 and times == sorted(times)
            if transition_rows["delivered"] and recipient in start_by and recipient in complete_by:
                delivered_at = _time(transition_rows["delivered"][0]["at"])
                valid = valid and _time(start_by[recipient]["at"]) <= delivered_at < _time(complete_by[recipient]["at"])
            else:
                valid = False
            acknowledgement = transition_rows["acknowledged"][0]["data"] if transition_rows["acknowledged"] else {}
            sender = next((a for a, ident in required.items() if ident == event_id), None)
            code = acknowledgement.get("evidence_code", acknowledgement.get("code", acknowledgement.get("nonce")))
            matching_tool = [row for row in groups.get("tool_call", []) if row["data"].get("agent") == recipient
                and row["data"].get("call_id") == acknowledgement.get("call_id")]
            model_ack = acknowledgement.get("source") == "model_tool" and bool(acknowledgement.get("call_id")) and len(matching_tool) == 1
            if matching_tool and transition_rows["acknowledged"]:
                model_ack = model_ack and _time(matching_tool[0]["at"]) <= _time(transition_rows["acknowledged"][0]["at"])
            model_ack = model_ack and sender is not None and code == nonces.get(sender)
            transition_details[delivery_id] = {"separate_ordered_states_and_active_delivery": bool(valid), "model_code_acknowledgement": bool(model_ack)}
        check("delivery_acceptance_receipt_ack_and_incorporation_are_separate", len(transition_details) == pairs
              and all(row["separate_ordered_states_and_active_delivery"] for row in transition_details.values()), transition_details)
        check("six_actual_model_code_acknowledgements", len(transition_details) == pairs
              and all(row["model_code_acknowledgement"] for row in transition_details.values()))

        final_checks = {}
        final_rows = [row for row in groups.get("agent_message", []) if row["data"].get("phase") in ("final", "final_answer")]
        final_by = {row["data"].get("agent"): row for row in final_rows}
        for agent in agents:
            try:
                output = _json_object(workers.get(agent, {}).get("final", ""))
                ledger_output = _json_object(final_by.get(agent, {}).get("data", {}).get("text", ""))
                used = output.get("used_event_ids", [])
                peer_ids = {required[a] for a in agents if a != agent}
                final_checks[agent] = {
                    "correct_worker": output.get("worker") == agent,
                    "correct_candidate": output.get("chosen_candidate") == fixture.get("expected_candidate", "Cedar"),
                    "all_private_codes": output.get("evidence_codes") == nonces,
                    "two_actual_peer_event_ids" if count == 3 else "all_actual_peer_event_ids": isinstance(used, list) and len(used) == count - 1 and set(used) == peer_ids,
                    "matches_runtime_final": output == ledger_output,
                    "after_acknowledgements": all(_time(row["at"]) <= _time(final_by[agent]["at"]) for row in
                        groups.get("acknowledged", []) if row["data"].get("recipient") == agent
                        and row["data"].get("event_id") in peer_ids),
                }
            except (ValueError, TypeError, KeyError):
                final_checks[agent] = {"valid_json_final": False}
        check("all_three_finals_prove_correct_incorporation", len(final_rows) == count and set(final_by) == agents
              and all(all(facts.values()) for facts in final_checks.values()), final_checks)

        mode = manifest.get("routing_mode", manifest.get("mode", "hybrid"))
        backend = manifest.get("index_backend", "hyperspace")
        check("recognized_index_backend", backend in {"hyperspace", "local_exact_poincare"})
        geometry = []
        for row in groups.get("routing_decision", []):
            d = row["data"]
            if backend != "hyperspace" or d.get("mandatory") or d.get("distance_source") != "hyperspace_index" or d.get("mode") not in ("hybrid", "hyperbolic"):
                continue
            if any(reason in d.get("reasons", []) for reason in ["sender_already_knows", "already_seen", "attention_budget_exhausted", "no_novel_information", "diversity_probe"]):
                continue
            value, radius = d.get("distance"), d.get("attention_radius")
            if not isinstance(value, (int, float)) or not isinstance(radius, (int, float)) or not math.isfinite(value) or not math.isfinite(radius):
                continue
            near = value <= radius
            semantic_gate = d.get("mode") == "hyperbolic" or d.get("path_related") or "lexical_match" in d.get("reasons", [])
            # If the non-geometric gate already rejects, distance did not decide.
            influenced = bool(semantic_gate) and d.get("deliver") is bool(near)
            if influenced:
                geometry.append({"event_id": d.get("event_id"), "recipient": d.get("recipient"), "distance": value,
                    "radius": radius, "deliver": d.get("deliver"), "mode": d.get("mode")})
        geometry_required = backend == "hyperspace" and mode in ("hybrid", "hyperbolic")
        if backend == "hyperspace":
            check("actual_index_distance_influenced_nonmandatory_routing", bool(geometry) if geometry_required else True, geometry)
        photon = _photon(directory, manifest, run_id)
        replay = _replay(directory)
        core_pass = all(checks.values())
        return {"run_id": run_id, "audited_at": datetime.now(timezone.utc).isoformat(), "model_calls": 0,
            "worker_count": count, "required_message_pairs": pairs, "provider": provider,
            "core_pass": core_pass, "passed": core_pass and photon.get("pass") is not False
                and photon.get("disabled_has_no_sends") is not False and replay.get("pass") is not False,
            "checks": checks, "failures": [name for name, valid in checks.items() if not valid], "details": details,
            "active_turn_overlap_seconds": round(overlap, 6), "mode": mode, "requested_model": requested_model,
            "http_requests": http,
            "geometry": {"required": geometry_required, "tested": bool(geometry), "status": "MEASURED" if geometry else "NOT_TESTED",
                "index_backend": backend, "decisive_index_routes": geometry,
                "scope": "Index distances affect routing at hierarchy-derived coordinates; learned semantic geometry and quality improvements are not established."
                    if backend == "hyperspace" else "Local exact Poincare routing is configured; Hyperspace index routing was not exercised."},
            "photon": photon, "replay": replay, "source_receipts": source_receipts,
            "limitations": ["Overlap includes model and tool waits; simultaneous provider inference is not verified.",
                "Ledger hashes detect alteration but are not external cryptographic attestation of provider execution.",
                "Steering acceptance and inbox delivery are distinct from final evidence-code incorporation."]}
    except Exception as exc:
        return {"core_pass": False, "passed": False, "model_calls": 0,
                "failures": ["invalid_or_missing_evidence"], "error": f"{type(exc).__name__}: {exc}",
                "source_receipts": source_receipts}
