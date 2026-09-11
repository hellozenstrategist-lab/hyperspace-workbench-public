"""Actual Mission/store/bus/geometry; deterministic worker runtime, no models."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_harness.codex_runtime import RecoveryRequired, ToolInputError
from astra_harness.conversation import Conversation, ConversationProtocolError, TOOLS
from astra_harness.schema import AGENTS, atomic_json


class FakeRoundRuntime:
    instances = []
    new_threads = 0
    new_rounds = 0
    invalid_final_agent = None
    fail_after_finals = False

    def __init__(self, state_path, artifacts_dir, on_event, on_tool):
        self.path, self.on_event, self.on_tool = Path(state_path), on_event, on_tool
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"workers": {}, "rounds": {}}
        self.calls, self.tasks = [], []
        self.round_id = self.state.get("current_round_id")
        self.instances.append(self)

    @property
    def threads(self):
        return {a: w["thread_id"] for a, w in self.state["workers"].items()}

    def save(self):
        atomic_json(self.path, self.state)

    async def start(self):
        self.calls.append("start")
        if self.round_id:
            for agent in AGENTS:
                w = self.state["workers"][agent]
                await self.on_event("recovered", agent, {"round_id": self.round_id, "turn_id": w.get("turn_id")})
        return {"model": "example/default-model", "auth": "fake_no_model_calls"}

    async def create_workers(self, tools, instructions, workspaces):
        self.calls.append("create_workers")
        self.tools, self.instructions = tools, instructions
        for agent in AGENTS:
            if agent not in self.state["workers"]:
                self.__class__.new_threads += 1
                self.state["workers"][agent] = {"thread_id": "stable-thread-" + agent, "status": "prepared"}
        self.save()

    async def start_round(self, round_id, prompts):
        self.calls.append(("start_round", round_id))
        if round_id in self.state["rounds"]:
            if self.state["rounds"][round_id]["prompts"] != prompts:
                raise RecoveryRequired("Changed prompts")
            return
        self.__class__.new_rounds += 1
        self.round_id = self.state["current_round_id"] = round_id
        self.state["rounds"][round_id] = {"status": "active", "prompts": prompts}
        self.started, self.barrier = set(), asyncio.Event()
        self.tasks = [asyncio.create_task(self.worker(a)) for a in AGENTS]
        await self.barrier.wait()

    async def worker(self, agent):
        worker = self.state["workers"][agent]
        worker.update(status="active", turn_id=self.round_id + ":" + agent, final=None)
        await self.on_event("worker_started", agent, {"round_id": self.round_id, "thread_id": worker["thread_id"], "turn_id": worker["turn_id"]})
        self.started.add(agent)
        if len(self.started) == 3:
            self.barrier.set()
        await self.barrier.wait()
        # Reuse tool call IDs deliberately across rounds to test host namespacing.
        await self.on_tool(agent, "publish_knowledge", {"key": "initial", "kind": "hypothesis",
            "claim": "Independent " + agent + " consideration for " + self.round_id,
            "evidence_content": "Draft notes for " + agent}, "publish")
        findings = (await self.on_tool(agent, "await_peer_findings", {}, "inbox"))["findings"]
        await self.on_tool(agent, "acknowledge_findings", {"receipts": [
            {"event_id": f["event_id"], "evidence_code": f["claim"].rsplit(" Receipt: ", 1)[1]} for f in findings]}, "ack")
        final = json.dumps({"worker": agent, "answer": "A concise answer informed by both peers.",
                            "used_event_ids": [f["event_id"] for f in findings]})
        if agent == self.invalid_final_agent:
            final = json.dumps({"worker": agent, "answer": "x" * 2501, "used_event_ids": []})
        await self.on_event("agent_message", agent, {"round_id": self.round_id, "phase": "final_answer", "text": final})
        worker.update(status="completed", final=final)
        await self.on_event("worker_completed", agent, {"round_id": self.round_id, "status": "completed", "turn_id": worker["turn_id"]})
        self.save()
        return agent, final

    async def wait_round(self, round_id, timeout=300):
        self.calls.append(("wait_round", round_id))
        record = self.state["rounds"][round_id]
        if record["status"] == "completed":
            return record["finals"]
        finals = dict(await asyncio.wait_for(asyncio.gather(*self.tasks), timeout))
        record.update(status="completed", finals=finals)
        self.save()
        if self.fail_after_finals:
            self.__class__.fail_after_finals = False
            raise RecoveryRequired("Simulated interruption after native finals, before host answer commit")
        return finals

    async def steer(self, agent, event):
        self.calls.append("steer")
        return {"accepted": True}

    async def close(self, cancel=False):
        self.calls.append(("close", cancel))
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class ConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.conversations = []
        FakeRoundRuntime.instances = []
        FakeRoundRuntime.new_threads = FakeRoundRuntime.new_rounds = 0
        FakeRoundRuntime.invalid_final_agent = None
        FakeRoundRuntime.fail_after_finals = False
        self.patch = patch("astra_harness.mission.Runtime", FakeRoundRuntime)
        self.patch.start()

    async def asyncTearDown(self):
        for convo in self.conversations:
            await convo.close()
        self.patch.stop()
        self.temp.cleanup()

    def make(self, directory="conversation", endpoint=None):
        convo = Conversation(self.root / directory, endpoint)
        self.conversations.append(convo)
        return convo

    async def test_two_rounds_use_same_three_threads_and_fresh_receipts(self):
        c = self.make()
        await c.start()
        self.assertEqual(await c.reply("msg-1", "Help me choose an approach"), "A concise answer informed by both peers.")
        first_ids = {c.publication_id(a, "initial") for a in AGENTS}
        threads = c.runtime.threads
        self.assertEqual(await c.reply("msg-2", "What alternative is worth considering?"), "A concise answer informed by both peers.")
        second_ids = {c.publication_id(a, "initial") for a in AGENTS}
        self.assertFalse(first_ids & second_ids)
        self.assertEqual(c.runtime.threads, threads)
        self.assertEqual(FakeRoundRuntime.new_threads, 3)
        self.assertEqual(FakeRoundRuntime.new_rounds, 2)
        self.assertEqual(len(c.store.deliveries(c.run_id)), 12)
        self.assertTrue(all(r["acknowledged_at"] and not r["incorporated_at"] for r in c.store.deliveries(c.run_id)))
        self.assertEqual(len([r for r in c.store.ledger(c.run_id) if r["kind"] == "declared_use"]), 6)
        self.assertEqual(c.runtime.calls.count("start"), 1)
        self.assertEqual(c.runtime.calls.count("create_workers"), 1)
        for row in c.store.conn.execute("SELECT report FROM conversation_rounds"):
            report = json.loads(row[0])
            self.assertTrue(report["protocol_passed"])
            self.assertGreater(report["three_worker_overlap_seconds"], 0)
            self.assertFalse(report["semantic_correctness_verified"])
            self.assertFalse(report["incorporation_verified"])

    async def test_message_replay_and_restart_use_saved_answer_without_turns(self):
        first = self.make()
        answer = await first.reply("msg", "Remember this discussion")
        calls = list(first.runtime.calls)
        self.assertEqual(await first.reply("msg", "Remember this discussion"), answer)
        self.assertEqual(first.runtime.calls, calls)
        await first.close()
        second = self.make()
        self.assertEqual(await second.reply("msg", "Remember this discussion"), answer)
        self.assertEqual(second.runtime.calls, [])
        await second.reply("next", "Continue")
        self.assertEqual(FakeRoundRuntime.new_threads, 3)
        self.assertEqual(FakeRoundRuntime.new_rounds, 2)

    async def test_message_collision_and_concurrent_owner_refused(self):
        c = self.make()
        await c.reply("id", "Original")
        with self.assertRaisesRegex(ValueError, "different text"):
            await c.reply("id", "Changed")
        with self.assertRaises(BlockingIOError):
            Conversation(c.directory)
        self.assertEqual(FakeRoundRuntime.new_rounds, 1)

    async def test_simultaneous_replays_coalesce_without_duplicate_rounds(self):
        c = self.make()
        a, b = await asyncio.gather(c.reply("same", "A message"), c.reply("same", "A message"))
        self.assertEqual(a, b)
        self.assertEqual(FakeRoundRuntime.new_rounds, 1)

    async def test_failed_final_is_saved_and_never_sent_or_silently_retried(self):
        FakeRoundRuntime.invalid_final_agent = "agent-c"
        c = self.make()
        with self.assertRaises(ConversationProtocolError):
            await c.reply("bad", "A message")
        row = c._round("bad")
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["answer"])
        report = json.loads((c.directory / "rounds" / row["round_id"] / "report.json").read_text())
        self.assertFalse(report["protocol_passed"])
        with self.assertRaises(ConversationProtocolError):
            await c.reply("bad", "A message")
        self.assertEqual(FakeRoundRuntime.new_rounds, 1)

    async def test_recover_completed_native_round_before_host_commit(self):
        FakeRoundRuntime.fail_after_finals = True
        c = self.make()
        with self.assertRaises(RecoveryRequired):
            await c.reply("recover", "Preserve this message")
        self.assertEqual(c._round("recover")["status"], "blocked")
        with self.assertRaisesRegex(RecoveryRequired, "earlier"):
            await c.reply("later", "Do not start yet")
        await c.close()
        restored = self.make()
        answer = await restored.reply("recover", "Preserve this message")
        self.assertEqual(answer, "A concise answer informed by both peers.")
        self.assertEqual(FakeRoundRuntime.new_rounds, 1)
        self.assertEqual(FakeRoundRuntime.new_threads, 3)

    async def test_old_initial_receipts_cannot_satisfy_new_round(self):
        c = self.make()
        await c.reply("one", "First")
        old = c.publication_id("agent-b", "initial")
        old_code = c.bus.expected_codes[old]
        await c.reply("two", "Second")
        for name,args in (("await_peer_findings", {"event_ids": [old]}),
                          ("acknowledge_findings", {"receipts": [{"event_id": old, "evidence_code": old_code}]})):
            with self.assertRaisesRegex(ToolInputError, "earlier-round"):
                await c.on_tool("agent-a", name, args, "old-" + name)

    async def test_publication_and_context_budgets_reset_per_round(self):
        c = self.make()
        for number in range(9):
            await c.reply(str(number), "Message " + str(number))
            self.assertEqual(c.publication_count("agent-a"), 1)
            self.assertEqual(len(c.bus._reserved("agent-a")), 2)
        self.assertEqual(FakeRoundRuntime.new_threads, 3)
        self.assertEqual(FakeRoundRuntime.new_rounds, 9)

    async def test_authorized_recall_uses_index_and_charges_current_round(self):
        c = self.make()
        await c.reply("first", "Alternative proposal")
        old_ids = {c.publication_id(a, "initial") for a in AGENTS}
        await c.reply("second", "Recall the alternative")
        class Index:
            def __init__(self): self.calls = []
            def search(self, xy, k, filter):
                self.calls.append((xy,k,filter))
                return [{"id": i, "distance": 0.2} for i in old_ids] + [{"id": "not-authorized", "distance": 0}]
            def close(self): pass
        c.index = Index()
        result = await c.on_tool("agent-a", "recall_knowledge", {"query": "alternative", "limit": 2}, "recall")
        self.assertEqual(len(result["findings"]), 2)
        self.assertTrue({r["event_id"] for r in result["findings"]} <= old_ids)
        self.assertEqual(result["backend"], "hyperspace")
        self.assertFalse(result["claims_verified"])
        self.assertEqual(c.index.calls[0][1], 64)
        self.assertEqual(len(c.bus._reserved("agent-a")), 3)
        await c.on_tool("agent-a", "recall_knowledge", {"query": "alternative", "limit": 2}, "recall")
        self.assertEqual(len(c.index.calls), 1)

    async def test_recall_before_current_initial_and_unbounded_tools_rejected(self):
        c = self.make()
        await c.reply("old", "Prior")
        c.round_id = "a14c6b5d-9559-48bc-945b-3c1ac5e4fba0"
        c.round_path = ["root", "phone", c.run_id, c.round_id]
        with self.assertRaisesRegex(ToolInputError, "current round"):
            await c.on_tool("agent-a", "recall_knowledge", {"query": "prior"}, "early")
        with self.assertRaises(ToolInputError):
            await c.on_tool("agent-a", "shell", {"command": "unused"}, "unsupported")
        self.assertEqual({t["name"] for t in TOOLS}, {"publish_knowledge", "await_peer_findings", "acknowledge_findings", "retrieve_evidence", "recall_knowledge"})

    async def test_stale_round_callbacks_do_not_replace_current_worker_state(self):
        c = self.make()
        await c.reply("first", "First")
        old = c.round_id
        await c.reply("second", "Second")
        before = c.store.workers(c.run_id)
        await c.on_event("worker_started", "agent-a", {"round_id": old, "thread_id": "stale", "turn_id": "stale"})
        self.assertEqual(before, c.store.workers(c.run_id))

    async def test_input_bounds_and_idempotent_shutdown(self):
        c = self.make()
        for ident,text in (("", "x"), ("x"*201, "x"), ("id", " "), ("id", "x"*12001)):
            with self.assertRaises(ValueError):
                await c.reply(ident, text)
        await c.shutdown()
        await c.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await c.reply("id", "A message")
        self.assertEqual(FakeRoundRuntime.new_rounds, 0)
