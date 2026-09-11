"""Real mission/store/router/bus, deterministic fake worker runtime only."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_harness.mission import Mission, TOOLS, load_config, run_mission
from astra_harness.schema import AGENTS, worker_ids


class FakeRuntime:
    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents=AGENTS, model=None):
        self.agents = tuple(agents)
        self.on_event, self.on_tool = on_event, on_tool
        self.calls, self.tasks, self.started = [], [], set()
        self.barrier = asyncio.Event()
        self.instances.append(self)

    async def start(self):
        self.calls.append("start")
        return {"auth": "fake_no_model_calls", "model": "fake_only"}

    async def create_workers(self, tools, instructions, workspaces):
        self.tools, self.workspaces, self.instructions = tools, workspaces, instructions

    async def start_turns(self, prompts):
        self.calls.append("start_turns")
        self.tasks = [asyncio.create_task(self.worker(a, prompts[a])) for a in self.agents]
        await self.barrier.wait()

    async def worker(self, agent, prompt):
        await self.on_event("worker_started", agent, {"thread_id": "thread-" + agent, "turn_id": "turn-" + agent})
        self.started.add(agent)
        if len(self.started) == len(self.agents):
            self.barrier.set()
        await self.barrier.wait()
        await self.on_tool(agent, "publish_knowledge", {"key": "initial", "kind": "hypothesis",
            "claim": "Independent proposal from " + agent, "evidence_content": "Draft artifact from " + agent}, "publish")
        findings = (await self.on_tool(agent, "await_peer_findings", {}, "inbox"))["findings"]
        await self.on_tool(agent, "acknowledge_findings", {"receipts": [
            {"event_id": f["event_id"], "evidence_code": f["claim"].rsplit(" Receipt: ", 1)[1]} for f in findings]}, "ack")
        await self.on_tool(agent, "retrieve_evidence", {"event_id": findings[0]["event_id"]}, "artifact")
        quotes = [{"event_id": f["event_id"], "quote": f["claim"].split(" Receipt:")[0]} for f in findings]
        final = json.dumps({"worker": agent, "answer": "I considered: " + "; ".join(q["quote"] for q in quotes),
                            "used_event_ids": [f["event_id"] for f in findings], "event_quotes": quotes})
        await self.on_event("agent_message", agent, {"phase": "final_answer", "text": final})
        await self.on_event("worker_completed", agent, {"status": "completed"})
        return agent, final

    async def steer(self, agent, payload):
        self.calls.append("steer")
        return {"accepted": True}

    async def wait(self, timeout):
        return dict(await asyncio.wait_for(asyncio.gather(*self.tasks), timeout))

    async def close(self, cancel=False):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class MissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = self.root / "config.json"
        self.data = {"objective": "Compare supplied ideas", "check_final_quotes": True,
                     "workers": [{"id": a, "prompt": "Independent contribution " + a,
                                  "task_path": ["root", "ideas", a]} for a in AGENTS]}
        self.config.write_text(json.dumps(self.data))
        self.missions = []
        FakeRuntime.instances = []
        self.patch = patch("astra_harness.mission.Runtime", FakeRuntime)
        self.patch.start()

    async def asyncTearDown(self):
        for mission in self.missions:
            await mission.bus.close()
            await mission.runtime.close()
            mission.close()
        self.patch.stop()
        self.temp.cleanup()

    def make(self):
        mission = Mission(self.config, self.root / ("run-" + str(len(self.missions))))
        self.missions.append(mission)
        return mission

    async def publish(self, mission, agent="agent-a", **extra):
        args = {"key": "initial", "kind": "claim", "claim": "A tentative statement", **extra}
        return await mission.on_tool(agent, "publish_knowledge", args, "publish-" + args["key"])

    async def test_three_concurrent_workers_no_incorporation_claim(self):
        mission = self.make()
        report = await mission.run()
        self.assertTrue(report["protocol_passed"])
        self.assertFalse(report["semantic_correctness_verified"])
        self.assertFalse(report["incorporation_verified"])
        self.assertEqual(mission.runtime.started, set(AGENTS))
        self.assertEqual(set(mission.runtime.workspaces), set(AGENTS))
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(len(mission.store.deliveries(mission.run_id)), 6)
        self.assertTrue(all(r["acknowledged_at"] for r in mission.store.deliveries(mission.run_id)))
        self.assertFalse(any(r["incorporated_at"] for r in mission.store.deliveries(mission.run_id)))
        self.assertTrue(all(q["literal_match"] for c in report["checks"].values() for q in c["quote_checks"]))
        self.assertEqual(len([r for r in mission.store.ledger(mission.run_id) if r["kind"] == "declared_use"]), 3)
        self.assertEqual({t["name"] for t in mission.runtime.tools}, {t["name"] for t in TOOLS})
        self.assertFalse((mission.directory / "fixture.json").exists())

    async def test_completed_public_api_resume_makes_no_new_runtime_calls(self):
        directory = self.root / "resume"
        first = await run_mission(self.config, directory, None)
        second = await run_mission(self.config, directory, None)
        self.assertEqual(first, second)
        self.assertEqual(FakeRuntime.instances[-1].calls, [])

    async def test_five_workers_share_all_20_findings_and_replay_without_calls(self):
        identities = worker_ids(5)
        self.data["workers"] = [{"id": a, "prompt": "Independent contribution " + a,
                                  "task_path": ["root", "ideas", a]} for a in identities]
        self.config.write_text(json.dumps(self.data))
        mission = self.make()
        report = await mission.run()
        self.assertTrue(report["protocol_passed"])
        self.assertEqual(mission.context_budget, 12000)
        self.assertEqual(mission.runtime.started, set(identities))
        self.assertEqual(report["manifest"]["worker_count"], 5)
        self.assertEqual(report["concurrency"]["peak_active_workers"], 5)
        self.assertGreater(report["concurrency"]["all_workers_overlap_seconds"], 0)
        self.assertEqual(report["concurrency"]["http_requests"]["status"], "not_measured")
        rows = mission.store.deliveries(mission.run_id)
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row["acknowledged_at"] for row in rows))
        for agent in identities:
            self.assertEqual(len(report["checks"][agent]["event_ids"]), 4)
            self.assertEqual(len(mission.store.deliveries(mission.run_id, agent)), 4)
        calls = list(mission.runtime.calls)
        self.assertEqual(await mission.run(), report)
        self.assertEqual(mission.runtime.calls, calls)
        # Missing the fifth worker must fail the protocol, even with all others complete.
        finals = {a: json.dumps(report["checks"][a]["output"]) for a in identities[:-1]}
        self.assertFalse(mission.evaluate(finals)["protocol_passed"])

    async def test_config_worker_counts_hierarchy_labels_and_bounds(self):
        for update in ({"workers": self.data["workers"][:1]}, {"context_budget": True},
                       {"timeout_seconds": 10000}, {"shell": "not a supported capability"}):
            self.config.write_text(json.dumps({**self.data, **update}))
            with self.assertRaises(ValueError):
                load_config(self.config)
        for count in (2, 5):
            self.config.write_text(json.dumps({**self.data, "workers": [
                {"id": a, "prompt": "An independent task", "task_path": ["root", a]}
                for a in worker_ids(count)]}))
            self.assertEqual(len(load_config(self.config)["workers"]), count)
        self.config.write_text(json.dumps({**self.data, "workers": self.data["workers"] + self.data["workers"]}))
        with self.assertRaises(ValueError):
            load_config(self.config)
        self.data["workers"][0]["task_path"] = ["root", "/tmp/notes"]
        self.config.write_text(json.dumps(self.data))
        with self.assertRaises(ValueError):
            load_config(self.config)

    async def test_config_change_or_concurrent_owner_is_refused(self):
        mission = self.make()
        with self.assertRaises(BlockingIOError):
            Mission(self.config, mission.directory)
        mission.close()
        self.missions.remove(mission)
        self.data["objective"] = "Changed objective"
        self.config.write_text(json.dumps(self.data))
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            Mission(self.config, mission.directory)

    async def test_persisted_database_manifest_cannot_misreport_worker_count(self):
        mission = self.make()
        manifest = {**mission.manifest, "worker_count": 5}
        mission.store.update_manifest(mission.run_id, manifest)
        directory = mission.directory
        mission.close()
        self.missions.remove(mission)
        with self.assertRaisesRegex(ValueError, "different worker_count"):
            Mission(self.config, directory)

    async def test_completed_run_stays_completed_after_notification_error(self):
        mission = self.make()
        await mission.run()
        with patch.object(mission, "notify", side_effect=RuntimeError("notification unavailable")):
            with self.assertRaises(RuntimeError):
                await mission.run()
        self.assertEqual(mission.store.run(mission.run_id)["status"], "completed")

    async def test_no_worker_can_promote_verification_or_read_file(self):
        mission = self.make()
        for extra in ({"verification_status": "reproduced"}, {"evidence_path": "/tmp/file"}, {"confidence": 1}):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                await self.publish(mission, **extra)
        observed = await self.publish(mission, kind="evidence", evidence_content="Authored artifact, not proof of a claim")
        self.assertEqual(observed["verification_status"], "observed")
        self.assertEqual(mission.store.event(observed["event_id"]).confidence, 0)
        claim = await self.publish(mission, "agent-b", evidence_content="Some unverified numbers")
        self.assertEqual(claim["verification_status"], "unverified")

    async def test_invalid_publication_writes_no_orphan_artifact(self):
        mission = self.make()
        with self.assertRaisesRegex(ValueError, "Graph references"):
            await self.publish(mission, evidence_content="must not be persisted", parent_ids=["unknown"])
        with self.assertRaisesRegex(ValueError, "knowledge kind"):
            await self.publish(mission, evidence_content="must not be persisted", kind="self_verified")
        self.assertEqual(mission.store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0], 0)
        self.assertEqual(list(mission.store.evidence_dir.iterdir()), [])

    async def test_artifact_reads_require_delivery_and_preserve_content(self):
        mission = self.make()
        published = await self.publish(mission, evidence_content="Unicode evidence: café")
        await self.publish(mission, "agent-b")
        with self.assertRaisesRegex(ValueError, "own or delivered"):
            await mission.on_tool("agent-b", "retrieve_evidence", {"event_id": published["event_id"]}, "early")
        with self.assertRaisesRegex(ValueError, "not delivered"):
            await mission.on_tool("agent-b", "acknowledge_findings", {"receipts": [{"event_id": published["event_id"],
                "evidence_code": mission.store.event(published["event_id"]).claim.rsplit(" Receipt: ", 1)[1]}]}, "early-ack")
        row = mission.store.deliveries(mission.run_id, "agent-b")[0]
        mission.store.transition(row["delivery_id"], "delivered")
        result = await mission.on_tool("agent-b", "retrieve_evidence", {"event_id": published["event_id"]}, "read")
        self.assertEqual(result["artifacts"][0]["content"], "Unicode evidence: café")
        self.assertFalse(result["claims_verified"])

    async def test_stable_publications_reject_changed_payload_and_replay_after_cache_gap(self):
        mission = self.make()
        first = await self.publish(mission, evidence_content="draft")
        args = {"key": "initial", "kind": "claim", "claim": "A tentative statement", "evidence_content": "draft"}
        self.assertEqual(first, await mission.on_tool("agent-a", "publish_knowledge", args, "new-call"))
        with self.assertRaisesRegex(ValueError, "different content"):
            await mission.on_tool("agent-a", "publish_knowledge", {**args, "claim": "Changed statement"}, "change")
        self.assertEqual(len(mission.store.events(mission.run_id)), 1)

    async def test_graph_references_require_authorization_and_publication_limit(self):
        mission = self.make()
        first = await self.publish(mission)
        with self.assertRaisesRegex(ValueError, "Graph references"):
            await self.publish(mission, "agent-b", contradicts=[first["event_id"]])
        for n in range(7):
            await self.publish(mission, key="revision" + str(n), revision_of=first["event_id"])
        with self.assertRaises(BufferError):
            await self.publish(mission, key="overflow")

    async def test_artifact_budget_reserves_capacity_for_future_messages(self):
        mission = self.make()
        result = await self.publish(mission, evidence_content="x" * 500)
        await mission.on_tool("agent-a", "retrieve_evidence", {"event_id": result["event_id"]}, "read")
        reserved = mission.bus._reserved("agent-a")
        self.assertIn("artifact:" + result["event_id"], reserved)
        mission.bus.context_budget = sum(reserved.values()) + 1
        await self.publish(mission, "agent-b")
        mission.store.worker(mission.run_id, "agent-a", state="active")
        row = mission.store.deliveries(mission.run_id, "agent-a")[0]
        await mission.bus._deliver(row["delivery_id"])
        self.assertIn(row["delivery_id"], mission.bus.pressure_blocked)
        self.assertNotIn("steer", mission.runtime.calls)

    async def test_forged_final_citation_is_not_accepted_as_use(self):
        mission = self.make()
        report = mission.evaluate({a: json.dumps({"worker": a, "answer": "Unsupported answer",
            "used_event_ids": [mission.publication_id("agent-a", "initial")]}) for a in AGENTS})
        self.assertFalse(report["protocol_passed"])
        self.assertFalse(any(r["kind"] == "declared_use" for r in mission.store.ledger(mission.run_id)))

    async def test_completed_exchange_with_object_answer_preserves_failed_final(self):
        mission = self.make()
        original_wait = mission.runtime.wait
        captured = {}

        async def invalid_answer(timeout):
            finals = await original_wait(timeout)
            output = json.loads(finals["agent-c"])
            output["answer"] = {"proposal": "Structured model answer", "criteria": ["Preserve citations"]}
            finals["agent-c"] = json.dumps(output)
            captured.update(finals)
            return finals

        mission.runtime.wait = invalid_answer
        with self.assertRaisesRegex(RuntimeError, "protocol checks failed"):
            await mission.run()
        report = json.loads((mission.directory / "report.json").read_text())
        check = report["checks"]["agent-c"]
        self.assertFalse(report["protocol_passed"])
        self.assertTrue(report["checks"]["agent-a"]["protocol_passed"])
        self.assertTrue(report["checks"]["agent-b"]["protocol_passed"])
        self.assertEqual(check["error_field"], "answer")
        self.assertIn("must be a string", check["error"])
        self.assertEqual(check["output"], json.loads(captured["agent-c"]))
        self.assertEqual(check["raw_final"], captured["agent-c"])
        self.assertFalse(check["raw_final_truncated"])
        self.assertEqual(json.loads((mission.directory / "final_outputs.json").read_text()),
                         {"run_id": mission.run_id, "finals": captured})
        rows = mission.store.deliveries(mission.run_id)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["acknowledged_at"] for r in rows))
        self.assertEqual(mission.store.run(mission.run_id)["status"], "failed")
        self.assertIn("answer must be a single string", mission.runtime.instructions)
        self.assertIn("1..16000 characters", mission.runtime.instructions)

    async def test_final_diagnostics_identify_contract_fields_without_repair(self):
        mission = self.make()
        cases = [("not JSON", "final"), ("```", "final"), ("[]", "final"),
                 ('{"answer":NaN}', "final"),
                 (json.dumps({"worker": "agent-b", "answer": "x", "used_event_ids": []}), "worker"),
                 (json.dumps({"worker": "agent-a", "answer": {}, "used_event_ids": []}), "answer"),
                 (json.dumps({"worker": "agent-a", "answer": "", "used_event_ids": []}), "answer"),
                 (json.dumps({"worker": "agent-a", "answer": "x", "used_event_ids": "event"}), "used_event_ids"),
                 (json.dumps({"worker": "agent-a", "answer": "x", "used_event_ids": ["x", "x"]}), "used_event_ids")]
        for raw, field in cases:
            with self.subTest(field=field, raw=raw):
                report = mission.evaluate({"agent-a": raw})
                check = report["checks"]["agent-a"]
                self.assertFalse(check["protocol_passed"])
                self.assertEqual(check["error_field"], field)
                self.assertEqual(check["raw_final"], raw)
                json.dumps(report, allow_nan=False)
        self.assertFalse(any(r["kind"] == "declared_use" for r in mission.store.ledger(mission.run_id)))

    async def test_large_invalid_final_has_bounded_preview_and_exact_artifact(self):
        mission = self.make()
        raw = "invalid JSON " + "x" * 70000
        original_wait = mission.runtime.wait

        async def invalid_json(timeout):
            return {**await original_wait(timeout), "agent-c": raw}

        mission.runtime.wait = invalid_json
        with self.assertRaisesRegex(RuntimeError, "protocol checks failed"):
            await mission.run()
        check = json.loads((mission.directory / "report.json").read_text())["checks"]["agent-c"]
        self.assertEqual(check["error_field"], "final")
        self.assertTrue(check["raw_final_truncated"])
        self.assertEqual(len(check["raw_final"]), 64000)
        self.assertEqual(check["raw_final_characters"], len(raw))
        self.assertEqual(json.loads((mission.directory / "final_outputs.json").read_text())["finals"]["agent-c"], raw)


if __name__ == "__main__":
    unittest.main()
