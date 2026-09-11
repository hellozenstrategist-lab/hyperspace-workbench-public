"""Mutation tests against exports written by the actual knowledge store."""
import json
from pathlib import Path
import tempfile
import time
import unittest
import uuid

from astra_harness.evidence_audit import audit_run, MODEL
from astra_harness.knowledge_store import KnowledgeStore
from astra_harness.schema import AGENTS, worker_ids, KnowledgeEvent, now


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / "run"
        self.directory.mkdir()
        self.store = KnowledgeStore(Path(self.tmp.name) / "data/state.sqlite")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def fixture(self, geometry=True, photon=False, bad_ack=False, model=MODEL, worker_count=3,
                provider="codex", http=True, returned_model=None):
        agents = worker_ids(worker_count)
        run_id = str(uuid.uuid4())
        codes = {agent: "private-" + str(uuid.uuid4()) for agent in agents}
        required = {agent: str(uuid.uuid4()) for agent in agents}
        prompts = {agent: "Privately measured evidence code: " + codes[agent] for agent in agents}
        manifest = {"requested_model": model, "runtime": {"auth": "chatgpt", "model": model, "model_fallback": False},
                    "mode": "hybrid", "photon_enabled": photon}
        if worker_count != 3:
            manifest["worker_count"] = worker_count
        if provider == "openrouter":
            manifest["provider"] = provider
            manifest["runtime"].update(provider=provider, auth="openrouter_api_key")
        self.store.create_run(run_id, manifest)
        for agent in agents:
            self.store.log(run_id, "worker_prompt", {"agent": agent, "prompt": prompts[agent]})
            self.store.log(run_id, "worker_started", {"agent": agent, "turn_id": "turn-" + agent})
            self.store.worker(run_id, agent, state="active", thread_id="thread-" + agent,
                              turn_id="turn-" + agent, started_at=now())
            if provider == "openrouter" and http:
                self.store.log(run_id, "request_started", {"agent": agent, "request_id": "request-" + agent,
                    "started_at": time.time(), "requested_model": model})
        for agent in agents:
            ref = self.store.add_evidence(json.dumps({"owner": agent, "code": codes[agent]}).encode())
            event = KnowledgeEvent(event_id=required[agent], run_id=run_id, author_agent=agent,
                type="evidence", claim="Measured independently: " + codes[agent],
                evidence_refs=[ref], dependencies=[a for a in agents if a != agent],
                verification_status="observed", confidence=1.0)
            self.store.put_event(event, [0.1, 0.2])
            self.store.route(event.event_id, [{"recipient": a, "deliver": True, "mandatory": True,
                "reasons": ["explicit_dependency"], "mode": "hybrid"} for a in agents if a != agent])
        if geometry:
            self.store.log(run_id, "routing_decision", {"event_id": required[AGENTS[0]], "recipient": AGENTS[1],
                "mandatory": False, "distance_source": "hyperspace_index", "mode": "hybrid", "distance": 0.5,
                "attention_radius": 1.5, "path_related": True, "deliver": True, "reasons": ["within_attention_radius", "shared_task_branch"]})
        for delivery in self.store.deliveries(run_id):
            sender = next(a for a in agents if required[a] == delivery["event_id"])
            self.store.transition(delivery["delivery_id"], "accepted", details={"source": "turn_steer"})
            self.store.transition(delivery["delivery_id"], "delivered", details={"source": "inbox_tool"})
            self.store.log(run_id, "tool_call", {"agent": delivery["recipient"], "tool": "acknowledge",
                "call_id": "ack-" + delivery["delivery_id"]})
            self.store.transition(delivery["delivery_id"], "acknowledged", details={"source": "model_tool",
                "evidence_code": "wrong" if bad_ack else codes[sender], "call_id": "ack-" + delivery["delivery_id"]})
            self.store.transition(delivery["delivery_id"], "incorporated")
        for agent in agents:
            if provider == "openrouter" and http:
                self.store.log(run_id, "request_completed", {"agent": agent, "request_id": "request-" + agent,
                    "completed_at": time.time(), "requested_model": model, "model": returned_model or model,
                    "provider": "test-provider", "response_id": "response-" + agent, "status": "completed", "http_status": 200})
            final = json.dumps({"worker": agent, "chosen_candidate": "Cedar", "evidence_codes": codes,
                "used_event_ids": [required[a] for a in agents if a != agent], "justification": "All measured constraints pass."})
            self.store.log(run_id, "agent_message", {"agent": agent, "phase": "final_answer", "text": final})
            self.store.log(run_id, "worker_completed", {"agent": agent, "turn_id": "turn-" + agent,
                "status": "completed", "error": None})
            self.store.worker(run_id, agent, state="completed", completed_at=now(), final=final)
        self.store.set_run_state(run_id, "completed")
        self.store.export(run_id, self.directory)
        (self.directory / "fixture.json").write_text(json.dumps({"nonces": codes, "required_event_ids": required,
            "expected_candidate": "Cedar", "private_prompts": prompts}))
        return run_id

    def test_valid_store_export_passes_without_trusting_saved_integrity(self):
        self.fixture()
        path = self.directory / "snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot["ledger_integrity"] = {"valid": False, "passed": False}
        path.write_text(json.dumps(snapshot))
        report = audit_run(self.directory)
        self.assertTrue(report["core_pass"], report)
        self.assertEqual(report["photon"]["status"], "NOT_TESTED")
        self.assertEqual(report["replay"]["status"], "NOT_TESTED")

    def test_tampered_ledger_is_detected(self):
        self.fixture()
        path = self.directory / "ledger.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["data"] = json.dumps({"state": "fabricated"})
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report = audit_run(self.directory)
        self.assertFalse(report["core_pass"])
        self.assertIn("ledger_hash_chain", report["failures"])

    def test_five_worker_openrouter_audit_distinguishes_http_from_task_overlap(self):
        self.fixture(worker_count=5, provider="openrouter", model="test/model")
        report = audit_run(self.directory)
        self.assertTrue(report["core_pass"], report)
        self.assertEqual(report["worker_count"], 5)
        self.assertEqual(report["required_message_pairs"], 20)
        self.assertNotIn("chatgpt_requested_model_manifest", report["checks"])
        self.assertEqual(report["http_requests"]["peak_in_flight_requests"], 5)
        self.assertGreater(report["http_requests"]["all_workers_requests_overlap_seconds"], 0)
        self.assertIn("GPU inference overlap is not established", report["http_requests"]["scope"])

    def test_openrouter_manifest_cannot_substitute_for_actual_request_evidence(self):
        self.fixture(worker_count=5, provider="openrouter", model="test/model", http=False)
        report = audit_run(self.directory)
        self.assertIn("openrouter_request_provenance", report["failures"])
        self.assertEqual(report["http_requests"]["status"], "NOT_MEASURED")

    def test_openrouter_returned_model_mismatch_is_detected(self):
        self.fixture(worker_count=5, provider="openrouter", model="test/model", returned_model="test/different-model")
        self.assertIn("openrouter_request_provenance", audit_run(self.directory)["failures"])

    def test_fifth_worker_acknowledgement_cannot_be_omitted(self):
        self.fixture(worker_count=5)
        path = self.directory / "snapshot.json"
        snapshot = json.loads(path.read_text())
        row = next(row for row in snapshot["deliveries"] if row["recipient"] == "agent-e")
        row["acknowledged_at"] = None
        path.write_text(json.dumps(snapshot))
        self.assertIn("delivery_acceptance_receipt_ack_and_incorporation_are_separate", audit_run(self.directory)["failures"])

    def test_six_worker_manifest_is_rejected_even_with_saved_pass_flag(self):
        self.fixture(worker_count=5)
        path = self.directory / "snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot["run"]["manifest"]["worker_count"] = 6
        path.write_text(json.dumps(snapshot))
        self.assertFalse(audit_run(self.directory)["core_pass"])

    def test_explicit_alternate_receipts_remain_auditable(self):
        self.fixture(model="example/alternate-model")
        report = audit_run(self.directory)
        self.assertTrue(report["core_pass"], report)
        self.assertEqual(report["requested_model"], "example/alternate-model")

    def test_requested_default_with_alternate_runtime_manifest_is_rejected(self):
        self.fixture()
        path = self.directory / "snapshot.json"
        snapshot = json.loads(path.read_text())
        snapshot["run"]["manifest"]["runtime"]["model"] = "example/alternate-model"
        path.write_text(json.dumps(snapshot))
        report = audit_run(self.directory)
        self.assertIn("chatgpt_requested_model_manifest", report["failures"])

    def test_changed_evidence_bytes_are_detected(self):
        self.fixture()
        file = next(self.store.evidence_dir.iterdir())
        file.write_bytes(b"changed evidence")
        report = audit_run(self.directory)
        self.assertFalse(report["core_pass"])
        self.assertIn("content_addressed_evidence_hashes", report["failures"])

    def test_incorrect_final_code_is_detected(self):
        self.fixture()
        path = self.directory / "snapshot.json"
        snapshot = json.loads(path.read_text())
        final = json.loads(snapshot["workers"]["agent-a"]["final"])
        final["evidence_codes"]["agent-b"] = "not-received"
        snapshot["workers"]["agent-a"]["final"] = json.dumps(final)
        path.write_text(json.dumps(snapshot))
        report = audit_run(self.directory)
        self.assertIn("all_three_finals_prove_correct_incorporation", report["failures"])

    def test_ack_requires_actual_model_code_not_client_submission(self):
        self.fixture(bad_ack=True)
        report = audit_run(self.directory)
        self.assertIn("six_actual_model_code_acknowledgements", report["failures"])

    def test_mandatory_delivery_alone_does_not_prove_geometry_used(self):
        self.fixture(geometry=False)
        report = audit_run(self.directory)
        self.assertIn("actual_index_distance_influenced_nonmandatory_routing", report["failures"])

    def test_photon_failure_is_separate_from_core_success(self):
        self.fixture(photon=True)
        report = audit_run(self.directory)
        self.assertTrue(report["core_pass"], report)
        self.assertFalse(report["passed"])
        self.assertEqual(report["photon"]["status"], "FAILED")

    def test_photon_three_unique_provider_receipts_pass(self):
        run_id = self.fixture(photon=True)
        receipts = [{"run_id": run_id, "kind": kind, "key": kind, "status": "delivered", "provider_id": "p-" + kind}
                    for kind in ("start", "verified_important", "completed")]
        (self.directory / "photon_receipts.json").write_text(json.dumps(receipts))
        report = audit_run(self.directory)
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["photon"]["pass"])

    def test_replay_added_delivery_is_detected(self):
        self.fixture()
        (self.directory / "replay.json").write_text(json.dumps({"passed": True,
            "before": {"turn_starts": 3, "deliveries": 6, "notifications": 0},
            "after": {"turn_starts": 3, "deliveries": 7, "notifications": 0}}))
        report = audit_run(self.directory)
        self.assertTrue(report["core_pass"])
        self.assertFalse(report["replay"]["pass"])
        self.assertFalse(report["passed"])

    def test_replay_unchanged_counters_pass(self):
        self.fixture()
        counts = {"turn_starts": 3, "deliveries": 6, "notifications": 0}
        (self.directory / "replay.json").write_text(json.dumps({"before": counts, "after": counts, "passed": False}))
        report = audit_run(self.directory)
        self.assertTrue(report["replay"]["pass"])


if __name__ == "__main__":
    unittest.main()
