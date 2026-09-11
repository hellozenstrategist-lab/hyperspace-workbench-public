"""Deterministic aggregation and batch-control tests; no model/server calls."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from astra_harness import benchmark
from astra_harness.hyperbolic_index import position
from astra_harness.schema import AGENTS, worker_ids


def inputs(planted=False, worker_count=3):
    agents = worker_ids(worker_count)
    run_id = "synthetic-run"
    root = ["root", "project", "document-organizer"]
    required = {a: "event-" + a for a in agents}
    nonces = {a: "unique-code-" + a for a in agents}
    fixture = {"run_id": run_id, "required_event_ids": required, "nonces": nonces, "expected_candidate": "Cedar", "private_prompts": {a: "Own code: " + nonces[a] for a in agents}}
    if planted:
        fixture["planted_assumption"] = {"claim": "Atlas satisfies all constraints", "candidate": "Atlas", "expected_candidate": "Cedar"}
        fixture["private_prompts"] = {a: text + ". Atlas satisfies all constraints" for a, text in fixture["private_prompts"].items()}
    manifest = {"routing_mode": "hybrid", "model": "example/default-model", "reasoning_effort": "low", "context_budget_per_worker": 3000, "index_backend": "hyperspace", "fixture": "synthetic-v1", "photon_enabled": False, "tool_input_rejection_limit": 3, "tool_validation_policy_version": "model_tool_validation_v1"}
    if worker_count != 3:
        manifest.update(worker_count=worker_count, context_budget_per_worker=benchmark.default_context_budget(worker_count))
    events, deliveries, ledger, workers = [], [], [], {}
    def row(kind, at, **data):
        ledger.append({"run_id": run_id, "kind": kind, "at": at, "data": json.dumps(data)})
    for index, agent in enumerate(agents):
        event = {"event_id": required[agent], "author_agent": agent, "claim": "Measured unique role " + agent + " " + nonces[agent], "scope_path": root}
        events.append(event)
        row("published", 10, event=event)
        row("worker_started", index, agent=agent)
        row("attention_region", index, agent=agent, position=position(root + [str(index)]), task_path=root + [str(index)])
        final = {"worker": agent, "chosen_candidate": "Cedar", "evidence_codes": nonces, "used_event_ids": [ident for peer, ident in required.items() if peer != agent]}
        workers[agent] = {"final": json.dumps(final)}
        row("agent_message", 20+index, agent=agent, phase="final_answer", text=json.dumps(final))
        for recipient in agents:
            if recipient != agent:
                ident = agent + "-to-" + recipient
                delivery = {"delivery_id": ident, "event_id": required[agent], "recipient": recipient, "state": "incorporated", "accepted_at": 12, "delivered_at": 13, "acknowledged_at": 14, "incorporated_at": 20, "token_cost": 50}
                deliveries.append(delivery)
                for stage, at in (("accepted", 12), ("delivered", 13), ("acknowledged", 14), ("incorporated", 20)):
                    row(stage, at, delivery_id=ident, event_id=required[agent], recipient=recipient)
    for ident, path in (("optional-relevant", root), ("optional-wrong", ["root", "project", "marketing"])):
        events.append({"event_id": ident, "scope_path": path, "claim": "optional synthetic example"})
        deliveries.append({"delivery_id": ident, "event_id": ident, "recipient": AGENTS[0], "state": "accepted", "accepted_at": 16, "token_cost": 50})
    checks = {name: True for name in ("ledger_hash_chain", "snapshot_events_match_hashed_publications", "all_three_finals_prove_correct_incorporation", "no_initial_peer_code_preknowledge", "private_prompts_match_runtime_inputs", "six_actual_model_code_acknowledgements")}
    audit = {"core_pass": True, "checks": checks, "details": {"all_three_finals_prove_correct_incorporation": {a: {"correct_candidate": True, "all_private_codes": True} for a in agents}}, "active_turn_overlap_seconds": 18, "failures": []}
    if worker_count != 3:
        audit["checks"]["all_finals_prove_correct_incorporation"] = audit["checks"].pop("all_three_finals_prove_correct_incorporation")
        audit["checks"]["all_actual_model_code_acknowledgements"] = audit["checks"].pop("six_actual_model_code_acknowledgements")
        audit["details"]["all_finals_prove_correct_incorporation"] = audit["details"].pop("all_three_finals_prove_correct_incorporation")
    return {"run": {"run_id": run_id, "manifest": manifest, "status": "completed"}, "events": events, "deliveries": deliveries, "workers": workers}, fixture, ledger, audit


def test_usage_uses_last_cumulative_snapshot_once():
    snapshot, fixture, ledger, audit = inputs()
    for agent, total, last in ((AGENTS[0], 10, 10), (AGENTS[0], 25, 5), (AGENTS[1], 30, 6), (AGENTS[2], 40, 7)):
        ledger.append({"run_id": "synthetic-run", "kind": "token_usage", "at": 22, "data": {"agent": agent, "usage": {"total": {"totalTokens": total}, "last": {"totalTokens": last}, "modelContextWindow": 100000}}})
    metrics = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    usage = metrics["runtime_usage"]
    assert usage["aggregate_final_totals"]["totalTokens"] == 95
    assert usage["aggregate_latest_requests"]["totalTokens"] == 18
    assert usage["per_agent"][AGENTS[0]]["update_count"] == 2
    assert usage["per_agent"][AGENTS[0]]["last"] == {"totalTokens": 5}


def test_missing_usage_is_not_zero_and_partial_not_full_total():
    snapshot, fixture, ledger, audit = inputs()
    assert benchmark.aggregate_run(snapshot, fixture, ledger, audit)["runtime_usage"]["status"] == "NOT_MEASURED"
    ledger.append({"run_id": "synthetic-run", "kind": "token_usage", "at": 22, "data": {"agent": AGENTS[0], "usage": {"total": {"totalTokens": 99}, "last": {"totalTokens": 12}}}})
    usage = benchmark.aggregate_run(snapshot, fixture, ledger, audit)["runtime_usage"]
    assert usage["status"] == "PARTIAL"
    assert usage["aggregate_final_totals"] is None
    assert usage["available_agent_totals"] == {"totalTokens": 99}


def test_five_worker_metrics_require_all_five_usage_totals_and_twenty_deliveries():
    snapshot, fixture, ledger, audit = inputs(planted=True, worker_count=5)
    for index, agent in enumerate(worker_ids(5)):
        ledger.append({"run_id": "synthetic-run", "kind": "token_usage", "at": 30, "data": {
            "agent": agent, "usage": {"total": {"totalTokens": 10 * (index + 1)}, "last": {"totalTokens": 2}}}})
    metrics = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    assert metrics["task_success"] and metrics["peer_only_evidence_code_incorporation"]
    assert metrics["worker_count"] == 5 and metrics["correct_final_workers"] == 5
    assert metrics["required_message_pairs"] == 20
    assert metrics["publication_to_incorporation_seconds"]["count"] == 20
    assert len(metrics["attention_diversity"]["pairwise_anchor_distances"]) == 10
    assert set(metrics["tool_validation"]["per_agent"]) == set(worker_ids(5))
    assert metrics["runtime_usage"]["aggregate_final_totals"]["totalTokens"] == 150
    assert metrics["planted_assumption_correction"]["status"] == "MEASURED_FINAL_BASED"
    ledger.pop()
    usage = benchmark.aggregate_run(snapshot, fixture, ledger, audit)["runtime_usage"]
    assert usage["status"] == "PARTIAL" and usage["aggregate_final_totals"] is None


def test_five_worker_controls_and_report_cannot_silently_use_three_worker_results():
    snapshot, _, _, _ = inputs(worker_count=5)
    manifest = snapshot["run"]["manifest"]
    benchmark._validate_controls(manifest, "hybrid", worker_count=5)
    with pytest.raises(ValueError, match="worker_count"):
        benchmark._validate_controls(manifest, "hybrid", worker_count=3)
    with pytest.raises(ValueError, match="provider"):
        benchmark._validate_controls(manifest, "hybrid", worker_count=5, provider="openrouter")
    rendered = benchmark.render_markdown({"status": "RUNNING", "controls": {
        "model": "test/model", "workers_per_group": 5, "context_allowance_per_worker": 6000}, "runs": {}})
    assert "5 concurrent workers" in rendered and "6000-unit" in rendered


def test_latency_context_categories_and_failure_metrics():
    snapshot, fixture, ledger, audit = inputs()
    metrics = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    assert metrics["publication_to_incorporation_seconds"]["median"] == 10
    assert metrics["publication_to_incorporation_seconds"]["count"] == 6
    assert metrics["accepted_context"]["unique_claim_recipient_pairs"] == 8
    assert metrics["accepted_context"]["categories"] == {"required_useful": 6, "same_branch_optional_unproven_usefulness": 1, "wrong_branch_irrelevant_to_fixture": 1}
    assert metrics["accepted_context"]["estimated_cost_total"] == 400
    assert metrics["failures"]["missing_required_acknowledged"] == 0
    assert len(metrics["attention_diversity"]["pairwise_anchor_distances"]) == 3
    assert metrics["attention_diversity"]["accepted_scope_diversity"][AGENTS[0]]["entropy_bits"] > 0
    snapshot["deliveries"].pop(0)
    snapshot["deliveries"][0]["acknowledged_at"] = None
    failed = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    assert failed["failures"]["missing_durable_delivery_pairs"] == 1
    assert failed["failures"]["missing_required_acknowledged"] == 2


def test_peer_codes_require_no_preknowledge_and_missing_assumption_not_measured():
    snapshot, fixture, ledger, audit = inputs()
    metrics = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    assert metrics["peer_only_evidence_code_incorporation"]
    assert metrics["planted_assumption_correction"]["status"] == "NOT_MEASURED"
    audit["checks"]["no_initial_peer_code_preknowledge"] = False
    assert not benchmark.aggregate_run(snapshot, fixture, ledger, audit)["peer_only_evidence_code_incorporation"]


def test_planted_correction_is_explicitly_final_based():
    snapshot, fixture, ledger, audit = inputs(planted=True)
    correction = benchmark.aggregate_run(snapshot, fixture, ledger, audit)["planted_assumption_correction"]
    assert correction["status"] == "MEASURED_FINAL_BASED"
    assert correction["all_workers_corrected_after_start_seconds"] == 22
    assert all(row["seconds_from_own_worker_start"] == 20 for row in correction["per_agent"].values())
    fixture["private_prompts"][AGENTS[0]] = "no planted claim"
    assert benchmark.aggregate_run(snapshot, fixture, ledger, audit)["planted_assumption_correction"]["status"] == "NOT_MEASURED"


def test_duplicate_proxy_does_not_call_matching_claims_duplicate_compute():
    snapshot, fixture, ledger, audit = inputs()
    for event in snapshot["events"][:2]:
        event["claim"] = "Same measured wording " + fixture["nonces"][event["author_agent"]]
    duplicates = benchmark.aggregate_run(snapshot, fixture, ledger, audit)["duplicate_work_proxies"]
    assert duplicates["repeated_required_claim_wording_after_code_removal"] == 1
    assert duplicates["repeated_state_transitions"] == 0
    assert "NOT_MEASURED" in duplicates["scope"]


def test_fixture_signature_ignores_only_run_specific_values():
    snapshot, fixture, _, _ = inputs()
    changed = copy.deepcopy(fixture)
    changed["nonces"][AGENTS[0]] = "a-new-random-code"
    changed["private_prompts"][AGENTS[0]] = "Own code: a-new-random-code"
    assert benchmark._fixture_signature(changed, snapshot["run"]["manifest"]) == benchmark._fixture_signature(fixture, snapshot["run"]["manifest"])
    changed["planted_assumption"] = {"claim": "new condition"}
    assert benchmark._fixture_signature(changed, snapshot["run"]["manifest"]) != benchmark._fixture_signature(fixture, snapshot["run"]["manifest"])


def test_batch_runs_sequentially_and_reuses_completed_modes(monkeypatch, tmp_path):
    calls = []
    class FakeStore:
        def __init__(self, directory): self.directory = directory
        def run(self, _): return {"status": "completed" if (self.directory / "done").exists() else "prepared"}
    class FakeCoordinator:
        active = 0
        def __init__(self, directory, **kwargs):
            assert kwargs["notifier"] is None and kwargs["context_budget"] == 3000
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.manifest = {"routing_mode": kwargs["mode"], "model": "example/default-model", "reasoning_effort": "low", "context_budget_per_worker": 3000, "index_backend": "hyperspace", "fixture": "synthetic-v1", "tool_input_rejection_limit": 3, "tool_validation_policy_version": "model_tool_validation_v1"}
            self.fixture = {"expected_candidate": "Cedar", "private_prompts": {"agent-a": "same prompt"}}
            self.store = FakeStore(self.directory)
            self.run_id = self.directory.name
        async def run(self):
            FakeCoordinator.active += 1
            assert FakeCoordinator.active == 1
            calls.append(self.run_id)
            await asyncio.sleep(0)
            FakeCoordinator.active -= 1
            (self.directory / "done").write_text("synthetic test marker")
            (self.directory / "snapshot.json").write_text(json.dumps({"run": {"manifest": self.manifest, "status": "completed"}}))
            (self.directory / "fixture.json").write_text(json.dumps(self.fixture))
        def close(self): pass
    monkeypatch.setattr(benchmark, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(benchmark, "summarize_run", lambda path: {"directory": str(path), "audit": {"core_pass": True}, "metrics": {"independent_audit_pass": True, "correct_final_workers": 3}})
    first = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051"))
    assert first["status"] == "COMPLETED" and calls == list(benchmark.MODES)
    second = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051"))
    assert second["status"] == "COMPLETED" and len(calls) == 5
    assert all(entry["execution"] == "reused_completed_run" for entry in second["runs"].values())
    text = (tmp_path / "benchmark.md").read_text()
    assert "does not mean identical" in text and "NOT_MEASURED" in text


def test_benchmark_requires_actual_endpoint_before_constructing_workers(tmp_path):
    with pytest.raises(ValueError, match="actual HyperspaceDB"):
        asyncio.run(benchmark.run_benchmark(tmp_path, None))
    with pytest.raises(ValueError, match="actual HyperspaceDB"):
        asyncio.run(benchmark.run_benchmark(tmp_path, "local"))


def test_default_model_rejects_alternate_controls():
    snapshot, _, _, _ = inputs()
    assert benchmark.MODEL == "example/default-model"
    benchmark._validate_controls(snapshot["run"]["manifest"], "hybrid")
    snapshot["run"]["manifest"]["model"] = "example/alternate-model"
    with pytest.raises(ValueError, match="example/default-model"):
        benchmark._validate_controls(snapshot["run"]["manifest"], "hybrid")


def test_report_names_model_from_comparison_manifest_control():
    text = benchmark.render_markdown({"status": "RUNNING", "controls": {"model": "example/default-model"}, "runs": {}})
    assert "example/default-model" in text and "GPT-6 Astra" not in text


def test_validation_rejections_and_corrected_calls_are_counted_separately():
    snapshot, fixture, ledger, audit = inputs()
    for kind, call_id, at in (("tool_call", "bad1", 1), ("tool_rejected", "bad1", 2), ("tool_rejected", "bad1", 3), ("tool_call", "bad2", 4), ("tool_rejected", "bad2", 5), ("tool_call", "good", 6), ("tool_completed", "good", 7)):
        ledger.append({"run_id": "synthetic-run", "kind": kind, "at": at, "data": {"agent": AGENTS[0], "tool": "acknowledge_findings", "call_id": call_id}})
    result = benchmark.aggregate_run(snapshot, fixture, ledger, audit)["tool_validation"]
    assert result["unique_tool_attempts"] == 3
    assert result["rejected_calls"] == 2 and result["rejection_rate"] == pytest.approx(2/3)
    assert result["recovered_rejected_calls"] == 2
    assert result["recovered_retry_episodes"] == 1 and result["unrecovered_retry_episodes"] == 0
    assert result["task_succeeded_after_rejections"]


def test_legacy_fatal_validation_trace_is_not_zero_rejection_rate():
    snapshot, fixture, ledger, audit = inputs()
    snapshot["run"]["manifest"].pop("tool_input_rejection_limit")
    snapshot["run"]["manifest"].pop("tool_validation_policy_version")
    ledger += [{"run_id": "synthetic-run", "kind": "tool_call", "at": 30, "data": {"agent": AGENTS[2], "tool": "acknowledge_findings", "call_id": "bad"}}, {"run_id": "synthetic-run", "kind": "error", "at": 31, "data": {"agent": AGENTS[2], "error": "Acknowledgement must include the evidence code actually read"}}]
    result = benchmark.aggregate_run(snapshot, fixture, ledger, audit)
    assert result["tool_validation"]["rejection_rate"] is None
    assert result["failure_trace"]["nearest_preceding_same_worker_tool"]["call_id"] == "bad"
    assert result["failure_trace"]["submitted_arguments"] == "NOT_RECORDED_IN_LEDGER"


def test_historical_retry_policy_cannot_be_reused_as_current_control():
    snapshot, _, _, _ = inputs()
    manifest = snapshot["run"]["manifest"]
    benchmark._validate_controls(manifest, "hybrid", require_policy=True)
    manifest.pop("tool_input_rejection_limit")
    with pytest.raises(ValueError, match="tool_validation_policy"):
        benchmark._validate_controls(manifest, "hybrid", require_policy=True)


def test_receipt_policy_is_not_invented_for_historical_manifests():
    snapshot, fixture, ledger, audit = inputs()
    assert benchmark.aggregate_run(snapshot, fixture, ledger, audit)["controls"]["receipt_policy_version"] == "NOT_RECORDED"
    snapshot["run"]["manifest"]["receipt_policy_version"] = "explicit_inbox_ack_v1"
    assert benchmark.aggregate_run(snapshot, fixture, ledger, audit)["controls"]["receipt_policy_version"] == "explicit_inbox_ack_v1"


def test_history_preserves_failed_and_successful_trials_without_recursive_payload(monkeypatch, tmp_path):
    for name in ("hybrid", "hybrid-attempt-2"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "snapshot.json").write_text("{}")
    first = {"directory": str(tmp_path / "hybrid"), "attempt_number": 1}
    second = {"directory": str(tmp_path / "hybrid-attempt-2"), "attempt_number": 2}
    previous = {"status": "COMPLETED_WITH_RETRIALS", "runs": {"hybrid": second}, "mode_attempts": {"hybrid": [first, second]}, "previous_attempts": [{"directory": "/preserved/older", "source_report_sha256": "a"*64, "recorded_status": "INCOMPLETE", "runs": {"broadcast": {"directory": "/preserved/older/broadcast", "audit": {"core_pass": False}, "raw_unused_payload": "UNUSED_PAYLOAD_MARKER"}}, "ancestor_attempt_reports": [{"directory": "/preserved/oldest", "source_report_sha256": "b"*64}]}]}
    (tmp_path / "benchmark.json").write_text(json.dumps(previous))
    monkeypatch.setattr(benchmark, "summarize_run", lambda path: {"directory": str(path), "audit": {"core_pass": Path(path).name.endswith("2")}, "metrics": {}})
    history = benchmark._previous_attempt(tmp_path)
    assert [row["audit"]["core_pass"] for row in history["mode_attempts"]["hybrid"]] == [False, True]
    assert history["runs"]["hybrid"]["attempt_number"] == 2
    assert len(history["ancestor_attempt_reports"]) == 2
    assert "UNUSED_PAYLOAD_MARKER" not in json.dumps(history)
    text = benchmark.render_markdown({"status": "RUNNING", "runs": {}, "previous_attempts": [history]})
    assert "hybrid, attempt 1: independent audit FAIL" in text
    assert "hybrid, attempt 2: independent audit PASS" in text


def test_previous_attempt_is_reaudited_and_does_not_change_current_pass_counts(monkeypatch, tmp_path):
    failed = tmp_path / "failed"
    (failed / "broadcast").mkdir(parents=True)
    (failed / "broadcast" / "snapshot.json").write_text("{}")
    (failed / "benchmark.json").write_text(json.dumps({"status": "INCOMPLETE", "runs": {"broadcast": {"directory": str(failed / "broadcast")}}}))
    monkeypatch.setattr(benchmark, "summarize_run", lambda path: {"audit": {"core_pass": False}, "metrics": {"task_success": False}})
    history = benchmark._previous_attempt(failed)
    assert history["recorded_status"] == "INCOMPLETE"
    assert not history["runs"]["broadcast"]["audit"]["core_pass"]
    assert history["included_in_current_mode_pass_counts"] is False
    assert len(history["source_report_sha256"]) == 64


@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_explicit_retrial_preserves_failure_reuses_four_modes_and_records_scope_change(monkeypatch, tmp_path, retry_succeeds):
    calls = []
    fixture = {"expected_candidate": "Cedar", "private_prompts": {"agent-a": "unchanged fixture"}}
    def manifest(mode, version="model_tool_validation_v1"):
        return {"routing_mode": mode, "model": "example/default-model", "reasoning_effort": "low", "context_budget_per_worker": 3000, "index_backend": "hyperspace", "fixture": "test-v1", "tool_input_rejection_limit": 3, "tool_validation_policy_version": version}
    for mode in benchmark.MODES:
        path = tmp_path / mode
        path.mkdir()
        (path / "snapshot.json").write_text(json.dumps({"run": {"manifest": manifest(mode), "status": "failed" if mode == "hybrid" else "completed"}}))
        (path / "fixture.json").write_text(json.dumps(fixture))
        (path / "runtime_state.json").write_text(json.dumps({"workers": {a: {"status": "interrupted" if mode == "hybrid" else "completed"} for a in AGENTS}}))
    original = {name: (tmp_path / "hybrid" / name).read_bytes() for name in ("snapshot.json", "fixture.json", "runtime_state.json")}
    def summary(path):
        data = json.loads((Path(path) / "snapshot.json").read_text())["run"]
        return {"directory": str(path), "audit": {"core_pass": data["status"] == "completed"}, "metrics": {"run_status": data["status"], "tool_validation": {"policy": benchmark._validation_policy(data["manifest"]), "rejected_calls": 0}}}
    class FakeCoordinator:
        def __init__(self, directory, **kwargs):
            self.directory = Path(directory)
            self.directory.mkdir(parents=True)
            self.manifest = manifest(kwargs["mode"], "model_tool_validation_v2")
            self.fixture = fixture
            (self.directory / "fixture.json").write_text(json.dumps(fixture))
        async def run(self):
            calls.append(self.directory.name)
            state = "completed" if retry_succeeds else "failed"
            (self.directory / "snapshot.json").write_text(json.dumps({"run": {"manifest": self.manifest, "status": state}}))
            (self.directory / "runtime_state.json").write_text(json.dumps({"workers": {a: {"status": "completed" if retry_succeeds else "interrupted"} for a in AGENTS}}))
            if not retry_succeeds:
                raise RuntimeError("synthetic failed retrial")
        def close(self): pass
    monkeypatch.setattr(benchmark, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(benchmark, "summarize_run", summary)
    # No implicit retry: preserve the failed original and launch no models.
    stopped = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051"))
    assert stopped["status"] == "INCOMPLETE" and calls == []
    report = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051", retry_failed=True))
    assert calls == ["hybrid-attempt-2"]
    assert report["attempt_summary"]["first_attempt_modes_passed"] == 4
    assert report["attempt_summary"]["total_attempts"] == 6
    assert report["mode_attempts"]["hybrid"][0]["outcome"] == "FAILED"
    assert report["attempt_summary"]["selected_result_validation_policies_equal"] is False
    assert report["attempt_summary"]["first_attempt_validation_policies_equal"] is True
    assert all((tmp_path / "hybrid" / name).read_bytes() == value for name, value in original.items())
    assert list((tmp_path / "report_history").glob("*.json"))
    if retry_succeeds:
        assert report["status"] == "COMPLETED_WITH_RETRIALS" and report["all_core_pass"]
        again = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051", retry_failed=True))
        assert len(calls) == 1 and again["attempt_summary"]["total_attempts"] == 6
    else:
        assert report["status"] == "INCOMPLETE"
        assert not (tmp_path / "hybrid-attempt-3").exists()
        third = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051", retry_failed=True))
        assert calls == ["hybrid-attempt-2", "hybrid-attempt-3"]
        exhausted = asyncio.run(benchmark.run_benchmark(tmp_path, "127.0.0.1:50051", retry_failed=True))
        assert len(calls) == 2 and "maximum" in exhausted["execution_error"]["message"]
