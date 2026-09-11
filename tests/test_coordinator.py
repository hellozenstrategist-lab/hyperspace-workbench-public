"""Coordinator integration tests replace only Codex; storage/routing/tools are real."""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from astra_harness.coordinator import Coordinator, TOOLS
from astra_harness.schema import AGENTS, worker_ids
from astra_harness.evidence_audit import audit_run
from astra_harness import codex_runtime


class FakeRuntime:
    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents=None):
        self.agents = tuple(AGENTS if agents is None else agents)
        self.on_event, self.on_tool = on_event, on_tool
        self.calls = []
        self.tasks = []
        self.started = 0
        self.fatal = None
        self.worker_statuses = {a: "prepared" for a in self.agents}
        self.barrier = asyncio.Event()
        self.instances.append(self)

    async def start(self):
        self.calls.append("start")
        return {"auth": "chatgpt", "model": codex_runtime.MODEL, "requested_model": codex_runtime.MODEL,
                "codex_version": "0.153.4", "model_fallback": False}

    async def create_workers(self, tools, instructions, workspaces):
        self.calls.append("create_workers")
        self.tools, self.workspaces = tools, workspaces
        assert set(workspaces) == set(self.agents)
        return {a: "fake-thread-" + a for a in self.agents}

    async def start_turns(self, prompts):
        self.calls.append("start_turns")
        self.tasks = [asyncio.create_task(self.worker(agent, prompts[agent])) for agent in self.agents]
        await self.barrier.wait()

    async def call(self, agent, name, arguments):
        call_id = agent + ":" + name
        await self.on_event("tool_call", agent, {"tool": name, "call_id": call_id})
        result = await self.on_tool(agent, name, arguments, call_id)
        await self.on_event("tool_completed", agent, {"tool": name, "call_id": call_id})
        return result

    async def worker(self, agent, prompt):
        code = re.search(r"private evidence code: ([0-9a-f]+)", prompt).group(1)
        await self.on_event("worker_prompt", agent, {"prompt": prompt})
        await self.on_event("worker_started", agent, {"turn_id": "fake-turn-" + agent, "thread_id": "fake-thread-" + agent})
        self.worker_statuses[agent] = "active"
        self.started += 1
        if self.started == len(self.agents):
            self.barrier.set()
        await self.barrier.wait()
        measurement = prompt.split("Your private measurement: ", 1)[1].split(" Your private evidence code:", 1)[0]
        await self.call(agent, "publish_discovery", {"claim": measurement + " Evidence code: " + code,
                                                     "evidence_code": code})
        inbox = await self.call(agent, "await_peer_findings", {})
        findings = inbox["findings"]
        codes = {agent: code}
        receipts = []
        for finding in findings:
            peer_code = re.search(r"Evidence code: ([0-9a-f]+)", finding["claim"]).group(1)
            codes[finding["author_agent"]] = peer_code
            receipts.append({"event_id": finding["event_id"], "evidence_code": peer_code})
        await self.call(agent, "acknowledge_findings", {"receipts": receipts})
        # Give coordinator optional-probe lifecycle a chance to run before final.
        await asyncio.sleep(0.05)
        chosen = "Cedar"
        if len(self.agents) != 3:
            constraints = [measurement] + [finding["claim"] for finding in findings]
            passing = [{name for name, score in re.findall(r"([A-Z][a-z]+)=(\d+)", text) if int(score) >= 95} for text in constraints]
            survivors = set.intersection(*passing)
            assert len(survivors) == 1
            chosen = survivors.pop()
        final = json.dumps({"worker": agent, "chosen_candidate": chosen, "evidence_codes": codes,
                            "used_event_ids": [f["event_id"] for f in findings], "justification": "Combined private measurements."})
        await self.on_event("agent_message", agent, {"phase": "final_answer", "text": final})
        await self.on_event("worker_completed", agent, {"status": "completed", "turn_id": "fake-turn-" + agent, "error": None})
        self.worker_statuses[agent] = "completed"
        return agent, final

    async def steer(self, agent, event):
        self.calls.append("steer")
        await self.on_event("steer_accepted", agent, {"event": event})
        return {"accepted": True}

    async def wait(self, timeout=300):
        return dict(await asyncio.wait_for(asyncio.gather(*self.tasks), timeout))

    async def cancel(self):
        self.calls.append("cancel")
        return {}

    async def close(self, cancel=False):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name) / "run"
        self.coordinators = []
        FakeRuntime.instances = []
        self.patch = patch("astra_harness.coordinator.Runtime", FakeRuntime)
        self.patch.start()

    async def asyncTearDown(self):
        for coordinator in self.coordinators:
            await coordinator.runtime.close()
            coordinator.close()
        self.patch.stop()
        self.tmp.cleanup()

    def make(self, **kwargs):
        coordinator = Coordinator(self.directory, mode="broadcast", **kwargs)
        self.coordinators.append(coordinator)
        return coordinator

    async def publish(self, coordinator, agent, call_id="publish"):
        return await coordinator.on_tool(agent, "publish_discovery", {
            "claim": "Measured evidence: " + coordinator.fixture["nonces"][agent],
            "evidence_code": coordinator.fixture["nonces"][agent]}, call_id)

    async def test_one_coordinator_launches_exactly_three_workers_no_fourth_model(self):
        coordinator = self.make()
        result = await coordinator.run()
        self.assertTrue(result["core_result_passed"])
        self.assertEqual(len(FakeRuntime.instances), 1)
        self.assertEqual(set(coordinator.runtime.workspaces), set(AGENTS))
        self.assertEqual(coordinator.runtime.calls.count("start_turns"), 1)
        self.assertEqual(coordinator.runtime.started, 3)
        self.assertEqual({t["name"] for t in coordinator.runtime.tools}, {t["name"] for t in TOOLS})
        required = set(coordinator.fixture["required_event_ids"].values())
        deliveries = [r for r in coordinator.store.deliveries(coordinator.run_id) if r["event_id"] in required]
        self.assertEqual(len(deliveries), 6)
        self.assertTrue(all(r["incorporated_at"] for r in deliveries))

    async def test_completed_replay_adds_no_model_turns_or_deliveries(self):
        first = self.make()
        await first.run()
        run_id = first.run_id
        before_deliveries = len(first.store.deliveries(run_id))
        before_starts = len([r for r in first.store.ledger(run_id) if r["kind"] == "worker_started"])
        second = self.make()
        result = await second.run()
        self.assertTrue(result["core_result_passed"])
        self.assertEqual(second.runtime.calls, [])
        self.assertEqual(len(second.store.deliveries(run_id)), before_deliveries)
        self.assertEqual(len([r for r in second.store.ledger(run_id) if r["kind"] == "worker_started"]), before_starts)

    async def test_five_workers_overlap_and_incorporate_all_twenty_peer_deliveries(self):
        coordinator = self.make(worker_count=5)
        result = await coordinator.run()
        self.assertTrue(result["core_result_passed"])
        self.assertEqual(coordinator.runtime.started, 5)
        self.assertEqual(set(coordinator.runtime.workspaces), set(worker_ids(5)))
        self.assertEqual(result["concurrency"]["peak_active_workers"], 5)
        self.assertGreater(result["concurrency"]["all_workers_overlap_seconds"], 0)
        self.assertEqual(coordinator.context_budget, 6000)
        receipts = next(t for t in coordinator.runtime.tools if t["name"] == "acknowledge_findings")
        self.assertEqual(receipts["inputSchema"]["properties"]["receipts"]["maxItems"], 4)
        required = set(coordinator.fixture["required_event_ids"].values())
        deliveries = [r for r in coordinator.store.deliveries(coordinator.run_id) if r["event_id"] in required]
        self.assertEqual(len(deliveries), 20)
        self.assertTrue(all(r["acknowledged_at"] and r["incorporated_at"] for r in deliveries))
        for agent in coordinator.worker_ids:
            output = result["checks"][agent]["output"]
            self.assertEqual(set(output["evidence_codes"]), set(worker_ids(5)))
            self.assertEqual(len(output["used_event_ids"]), 4)
        audit = audit_run(self.directory)
        self.assertTrue(audit["core_pass"], audit)
        self.assertEqual(audit["required_message_pairs"], 20)
        self.assertEqual(audit["geometry"]["status"], "NOT_TESTED")
        self.assertFalse(audit["geometry"]["required"])
        snapshot_path = self.directory / "snapshot.json"
        snapshot = json.loads(snapshot_path.read_text())
        final = json.loads(snapshot["workers"]["agent-e"]["final"])
        final["evidence_codes"]["agent-d"] = "tampered-code"
        snapshot["workers"]["agent-e"]["final"] = json.dumps(final)
        snapshot_path.write_text(json.dumps(snapshot))
        self.assertIn("all_finals_prove_correct_incorporation", audit_run(self.directory)["failures"])

    async def test_each_five_worker_constraint_is_necessary(self):
        coordinator = self.make(worker_count=5)
        measurements = coordinator.fixture["measurements"]
        passing = [{name for name, score in re.findall(r"([A-Z][a-z]+)=(\d+)", text) if int(score) >= 95}
                   for text in measurements.values()]
        self.assertEqual(set.intersection(*passing), {"Cedar"})
        for index in range(5):
            self.assertEqual(len(set.intersection(*(p for i, p in enumerate(passing) if i != index))), 2)

    async def test_local_hybrid_run_passes_without_claiming_hyperspace_geometry(self):
        coordinator = Coordinator(self.directory, worker_count=5, mode="hybrid")
        self.coordinators.append(coordinator)
        await coordinator.run()
        audit = audit_run(self.directory)
        self.assertTrue(audit["core_pass"], audit)
        self.assertEqual(audit["geometry"]["index_backend"], "local_exact_poincare")
        self.assertEqual(audit["geometry"]["status"], "NOT_TESTED")
        self.assertFalse(audit["geometry"]["tested"])
        self.assertNotIn("actual_index_distance_influenced_nonmandatory_routing", audit["checks"])

    async def test_resume_rejects_worker_count_changes_before_any_turns(self):
        first = self.make(worker_count=5)
        await first.run()
        with self.assertRaisesRegex(ValueError, "worker count"):
            self.make(worker_count=3)
        self.assertEqual(len(FakeRuntime.instances[-1].calls), 0)

    async def test_extra_or_duplicate_peer_ids_do_not_pass_five_worker_final(self):
        coordinator = self.make(worker_count=5)
        result = await coordinator.run()
        finals = {a: json.dumps(check["output"]) for a, check in result["checks"].items()}
        output = json.loads(finals["agent-e"])
        output["used_event_ids"].append(output["used_event_ids"][0])
        finals["agent-e"] = json.dumps(output)
        self.assertFalse(coordinator.evaluate(finals)["core_result_passed"])

    async def test_five_openrouter_workers_complete_real_tool_protocol_with_one_key(self):
        from astra_harness.openrouter_runtime import OpenRouterRuntime
        from astra_harness.benchmark import summarize_run
        requests, observations = {}, {}
        barrier = asyncio.Event()

        async def fake_http(runtime, agent, record, payload):
            await runtime._request_started(agent, record, time.time())
            step = requests.get(agent, 0)
            requests[agent] = step + 1
            if step == 0:
                if len(requests) == 5:
                    barrier.set()
                await asyncio.wait_for(barrier.wait(), 5)
                prompt = next(message["content"] for message in payload["messages"]
                              if message["role"] == "user")
                code = re.search(r"private evidence code: ([0-9a-f]+)", prompt).group(1)
                measurement = prompt.split("Your private measurement: ", 1)[1].split(" Your private evidence code:", 1)[0]
                observations[agent] = {"code": code, "measurement": measurement}
                name, arguments = "publish_discovery", {"claim": measurement, "evidence_code": code}
            elif step == 1:
                name, arguments = "await_peer_findings", {}
            elif step == 2:
                returned = next(message for message in payload["messages"]
                                if message.get("tool_call_id") == agent + "-step-1")
                findings = json.loads(returned["content"])["findings"]
                observations[agent]["findings"] = findings
                name = "acknowledge_findings"
                arguments = {"receipts": [{"event_id": finding["event_id"], "evidence_code":
                    re.search(r"Evidence code: ([0-9a-f]+)", finding["claim"]).group(1)} for finding in findings]}
            else:
                self.assertEqual(step, 3)
                own = observations[agent]
                findings = own["findings"]
                codes = {agent: own["code"], **{finding["author_agent"]:
                    re.search(r"Evidence code: ([0-9a-f]+)", finding["claim"]).group(1) for finding in findings}}
                constraints = [own["measurement"]] + [finding["claim"] for finding in findings]
                passing = [{candidate for candidate, score in re.findall(r"([A-Z][a-z]+)=(\d+)", text)
                            if int(score) >= 95} for text in constraints]
                survivors = set.intersection(*passing)
                self.assertEqual(len(survivors), 1)
                final = {"worker": agent, "chosen_candidate": survivors.pop(), "evidence_codes": codes,
                         "used_event_ids": [finding["event_id"] for finding in findings]}
                message = {"role": "assistant", "content": json.dumps(final)}
                await asyncio.sleep(0.05)
            if step < 3:
                message = {"role": "assistant", "content": None, "tool_calls": [{"type": "function",
                    "id": agent + "-step-" + str(step), "function": {"name": name, "arguments": json.dumps(arguments)}}]}
            body = {"id": "generation-" + agent + "-" + str(step), "model": runtime.model,
                    "provider": "test-provider", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "choices": [{"finish_reason": "tool_calls" if step < 3 else "stop", "message": message}]}
            return {"http_status": 200, "completed_at": time.time(), "body": json.dumps(body).encode()}

        key = "synthetic-single-key-not-a-real-credential"
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": key}), patch.object(OpenRouterRuntime, "_http", fake_http):
            coordinator = self.make(worker_count=5, provider="openrouter", model="test/model")
            result = await coordinator.run()
        self.assertTrue(result["core_result_passed"])
        self.assertEqual(requests, {agent: 4 for agent in worker_ids(5)})
        self.assertEqual(result["concurrency"]["http_requests"]["peak_in_flight_requests"], 5)
        summary = summarize_run(self.directory)
        self.assertTrue(summary["audit"]["core_pass"], summary["audit"])
        self.assertEqual(summary["audit"]["required_message_pairs"], 20)
        self.assertEqual(summary["audit"]["http_requests"]["request_count"], 20)
        self.assertEqual(summary["metrics"]["runtime_usage"]["aggregate_final_totals"]["totalTokens"], 300)
        self.assertEqual(summary["metrics"]["tool_validation"]["policy"]["status"], "RECORDED")
        for path in self.directory.rglob("*"):
            if path.is_file():
                self.assertNotIn(key.encode(), path.read_bytes())

    async def test_tool_call_replay_does_not_duplicate_publication(self):
        coordinator = self.make()
        first = await self.publish(coordinator, "agent-a", "same-call")
        second = await self.publish(coordinator, "agent-a", "same-call")
        self.assertEqual(first, second)
        self.assertEqual(len(coordinator.store.events(coordinator.run_id)), 1)
        self.assertEqual(len(coordinator.store.deliveries(coordinator.run_id)), 2)

    async def test_changed_publication_under_stable_event_id_is_rejected(self):
        coordinator = self.make()
        await self.publish(coordinator, "agent-a", "first")
        with self.assertRaisesRegex(ValueError, "different content"):
            await coordinator.on_tool("agent-a", "publish_discovery", {
                "claim": "Changed measurement", "evidence_code": coordinator.fixture["nonces"]["agent-a"]}, "second")

    async def test_peer_reads_require_independent_publication(self):
        coordinator = self.make()
        with self.assertRaisesRegex(ValueError, "publication"):
                await coordinator.on_tool("agent-b", "await_peer_findings", {}, "read-too-soon")

    async def test_truncated_publication_code_is_recoverable_but_never_repaired(self):
        coordinator = self.make()
        exact = coordinator.fixture["nonces"]["agent-a"]
        with self.assertRaises(codex_runtime.ToolInputError):
            await coordinator.on_tool("agent-a", "publish_discovery", {"claim": "Measured latency", "evidence_code": exact[:-1]}, "bad-code")
        self.assertIsNone(coordinator.store.event(coordinator.fixture["required_event_ids"]["agent-a"]))
        self.assertEqual(coordinator.store.deliveries(coordinator.run_id), [])
        accepted = await coordinator.on_tool("agent-a", "publish_discovery", {"claim": "Measured latency", "evidence_code": exact}, "fixed-new-call")
        event = coordinator.store.event(accepted["published_event_id"])
        self.assertIn(exact, event.claim)
        requests = [json.loads(row["data"]) for row in coordinator.store.ledger(coordinator.run_id) if row["kind"] == "model_publication_request"]
        self.assertEqual([row["evidence_code"] for row in requests], [exact[:-1], exact])

    async def test_bad_publication_claim_and_schema_are_recoverable(self):
        coordinator = self.make()
        exact = coordinator.fixture["nonces"]["agent-a"]
        invalid = [{"claim": "", "evidence_code": exact}, {"claim": 123, "evidence_code": exact},
                   {"claim": "x" * 2001, "evidence_code": exact}, {"claim": "missing code"}, []]
        for number, args in enumerate(invalid):
            with self.subTest(args_type=type(args).__name__):
                with self.assertRaises(codex_runtime.ToolInputError):
                    await coordinator.on_tool("agent-a", "publish_discovery", args, f"invalid-{number}")
        self.assertEqual(coordinator.store.events(coordinator.run_id), [])

    async def test_first_pass_inbox_and_retrieval_errors_are_explicit_input_errors(self):
        coordinator = self.make()
        for tool_name in ("await_peer_findings", "retrieve_knowledge"):
            with self.assertRaises(codex_runtime.ToolInputError):
                await coordinator.on_tool("agent-a", tool_name, {}, "premature-" + tool_name)

    async def test_publication_collision_remains_fatal_invariant_error(self):
        coordinator = self.make()
        await self.publish(coordinator, "agent-a", "original")
        with self.assertRaises(ValueError) as error:
            await coordinator.on_tool("agent-a", "publish_discovery", {"claim": "Changed statement",
                "evidence_code": coordinator.fixture["nonces"]["agent-a"]}, "changed")
        self.assertNotIsInstance(error.exception, codex_runtime.ToolInputError)

    async def test_cannot_acknowledge_undelivered_or_invented_evidence(self):
        coordinator = self.make()
        await self.publish(coordinator, "agent-a")
        ident = coordinator.fixture["required_event_ids"]["agent-a"]
        with self.assertRaisesRegex(ValueError, "not delivered"):
            coordinator.bus.acknowledge("agent-b", [{"event_id": ident,
                "evidence_code": coordinator.fixture["nonces"]["agent-a"]}], "ack-early")
        row = coordinator.store.deliveries(coordinator.run_id, "agent-b")[0]
        coordinator.store.transition(row["delivery_id"], "delivered")
        with self.assertRaisesRegex(ValueError, "actually read"):
            coordinator.bus.acknowledge("agent-b", [{"event_id": ident, "evidence_code": "invented0000"}], "ack-fake")
        accepted = coordinator.bus.acknowledge("agent-b", [{"event_id": ident,
            "evidence_code": coordinator.fixture["nonces"]["agent-a"]}], "ack-real")
        self.assertEqual(accepted["acknowledged_event_ids"], [ident])

    async def test_correct_final_without_explicit_ack_is_not_incorporated(self):
        coordinator = self.make()
        for agent in AGENTS:
            await self.publish(coordinator, agent)
        finals = {agent: json.dumps({"worker": agent, "chosen_candidate": "Cedar",
            "evidence_codes": coordinator.fixture["nonces"], "used_event_ids": [
                value for peer, value in coordinator.fixture["required_event_ids"].items() if peer != agent]}) for agent in AGENTS}
        result = coordinator.evaluate(finals)
        self.assertFalse(result["core_result_passed"])
        self.assertFalse(any(row["incorporated_at"] for row in coordinator.store.deliveries(coordinator.run_id)))


class SourceLayoutAndPreservationTests(unittest.TestCase):
    def test_explicit_source_layout_dependencies_are_present(self):
        self.assertTrue(callable(codex_runtime.safe_env))
        self.assertTrue(callable(codex_runtime.codex_command))
        self.assertEqual(codex_runtime.codex_command()[:3], ["codex", "app-server", "--stdio"])
        self.assertEqual(codex_runtime.CODEX_VERSION, "0.153.4")
        self.assertEqual(len(codex_runtime.verify_protocol()), 64)


if __name__ == "__main__":
    unittest.main()
