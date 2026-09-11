"""Attention wiring: real coordinator/SQLite/router, fake transport/index boundary."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from astra_harness.coordinator import Coordinator
from astra_harness.hyperbolic_index import distance, position
from astra_harness.schema import AGENTS, KnowledgeEvent, canonical


class RecordingRuntime:
    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents=None):
        self.calls, self.response, self.error, self.inspect_call = [], {"accepted": True}, None, None

    async def steer(self, agent, payload):
        self.calls.append((agent, deepcopy(payload)))
        if self.inspect_call:
            self.inspect_call(agent, payload)
        if self.error:
            raise self.error
        return self.response

    async def close(self, cancel=False):
        pass


class RecordingIndex:
    def __init__(self, endpoint, **kwargs):
        self.nodes, self.upserts, self.searches = {}, [], []

    def health(self):
        return {"test_double": True}

    def upsert(self, node_id, point, metadata):
        self.nodes[node_id] = {"position": list(point), "metadata": deepcopy(metadata)}
        self.upserts.append(node_id)

    def search(self, vector, k, filter=None):
        self.searches.append((list(vector), k, deepcopy(filter)))
        rows = [{"id": ident, "distance": distance(vector, row["position"]), "metadata": row["metadata"]}
                for ident, row in self.nodes.items() if all(row["metadata"].get(k) == v for k, v in (filter or {}).items())]
        return sorted(rows, key=lambda row: (row["distance"], row["id"]))[:k]

    def close(self):
        pass


class CoordinatorAttentionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / "run"
        self.coordinators = []
        self.runtime_patch = patch("astra_harness.coordinator.Runtime", RecordingRuntime)
        self.index_patch = patch("astra_harness.hyperspace_backend.HyperspaceBackend", RecordingIndex)
        self.runtime_patch.start()
        self.index_patch.start()

    async def asyncTearDown(self):
        for coordinator in self.coordinators:
            await coordinator.bus.close()
            await coordinator.runtime.close()
            coordinator.close()
        self.index_patch.stop()
        self.runtime_patch.stop()
        self.temp.cleanup()

    def make(self, **kwargs):
        coordinator = Coordinator(self.directory, mode="hyperbolic", endpoint="127.0.0.1:50051", **kwargs)
        self.coordinators.append(coordinator)
        return coordinator

    def novelty(self, coordinator):
        row = coordinator.store.conn.execute("SELECT value FROM meta WHERE key=?", ("novelty:" + coordinator.run_id,)).fetchone()
        return json.loads(row[0]) if row else None

    async def test_five_worker_index_routing_retrieves_every_anchor(self):
        coordinator = self.make(worker_count=5)
        await coordinator.initialize_index()
        await coordinator.on_tool("agent-e", "publish_discovery", {
            "claim": coordinator.fixture["measurements"]["agent-e"],
            "evidence_code": coordinator.fixture["nonces"]["agent-e"]}, "publish-agent-e")
        searches = [row for row in coordinator.index.searches if row[2].get("kind") == "anchor"]
        self.assertTrue(searches)
        self.assertTrue(all(row[1] == 5 for row in searches))
        deliveries = coordinator.store.deliveries(coordinator.run_id)
        self.assertEqual({row["recipient"] for row in deliveries}, set(coordinator.worker_ids) - {"agent-e"})

    async def test_initialize_index_restores_persisted_task_and_new_anchor(self):
        first = self.make()
        await first.initialize_index()
        path, text = ["root", "new-project", "evidence"], "A real reassigned evidence task"
        receipt = await first.assign_task("agent-b", path, text)
        self.assertEqual(receipt["state"], "no_active_turn")
        stored = json.loads(first.store.workers(first.run_id)["agent-b"]["task"])
        first.close()
        self.coordinators.remove(first)
        resumed = self.make()
        self.assertNotEqual(resumed.agents["agent-b"]["task_path"], path)
        await resumed.initialize_index()
        self.assertEqual(resumed.agents["agent-b"], stored)
        self.assertEqual(resumed.index.nodes["anchor:agent-b"]["position"], position(path))
        self.assertIsNotNone(resumed.store.event(resumed.task_id(path)))
        self.assertEqual(resumed.runtime.calls, [])

    async def test_assignment_commits_intent_and_anchor_before_active_steer_then_replays(self):
        coordinator = self.make()
        await coordinator.initialize_index()
        coordinator.agents["agent-b"]["budget_remaining"] = 73
        coordinator.store.worker(coordinator.run_id, "agent-b", state="active", thread_id="same-thread", turn_id="same-turn")
        path = ["root", "backlog", "review"]
        def inspect_call(agent, payload):
            stored = coordinator.store.workers(coordinator.run_id)[agent]
            self.assertEqual(json.loads(stored["task"])["task_path"], path)
            self.assertEqual(coordinator.index.nodes["anchor:" + agent]["position"], position(path))
            self.assertTrue(any(row["idem_key"] == "assignment_intent:" + payload["assignment_id"] for row in coordinator.store.ledger(coordinator.run_id)))
        coordinator.runtime.inspect_call = inspect_call
        receipt = await coordinator.assign_task("agent-b", path, "Review the supplied notes")
        self.assertEqual(receipt["state"], "accepted")
        self.assertFalse(receipt["worker_adoption_verified"])
        self.assertEqual(coordinator.agents["agent-b"]["budget_remaining"], 73)
        worker = coordinator.store.workers(coordinator.run_id)["agent-b"]
        self.assertEqual((worker["state"], worker["thread_id"], worker["turn_id"]), ("active", "same-thread", "same-turn"))
        self.assertEqual(await coordinator.assign_task("agent-b", path, "Review the supplied notes"), receipt)
        self.assertEqual(len(coordinator.runtime.calls), 1)

    async def test_assignment_refusal_and_unknown_outcome_never_claim_adoption_or_resend(self):
        coordinator = self.make()
        await coordinator.initialize_index()
        for agent, outcome, expected in (("agent-a", {"accepted": False, "reason": "no_active_turn"}, "refused"),
                                         ("agent-b", ConnectionError("transport outcome unknown"), "uncertain")):
            coordinator.store.worker(coordinator.run_id, agent, state="active")
            coordinator.runtime.response = outcome if isinstance(outcome, dict) else None
            coordinator.runtime.error = outcome if isinstance(outcome, Exception) else None
            path = ["root", "new", agent]
            receipt = await coordinator.assign_task(agent, path, "Assigned task")
            self.assertEqual(receipt["state"], expected)
            self.assertFalse(receipt["worker_adoption_verified"])
            calls = len(coordinator.runtime.calls)
            self.assertEqual(await coordinator.assign_task(agent, path, "Assigned task"), receipt)
            self.assertEqual(len(coordinator.runtime.calls), calls)

    async def test_assignment_intent_without_receipt_is_uncertain_on_recovery(self):
        coordinator = self.make()
        path, text = ["root", "interrupted", "task"], "Persisted request"
        assignment_id = str(uuid.uuid4())
        updated = {**coordinator.agents["agent-c"], "task_path": path, "task_text": text, "position": position(path)}
        coordinator.store.assign_worker_task(coordinator.run_id, "agent-c", updated, assignment_id, {"new_task_path": path})
        await coordinator.initialize_index()
        receipt = await coordinator.assign_task("agent-c", path, text, assignment_id=assignment_id)
        self.assertEqual(receipt["state"], "uncertain")
        self.assertFalse(receipt["worker_adoption_verified"])
        self.assertEqual(coordinator.runtime.calls, [])
        self.assertEqual(coordinator.index.nodes["anchor:agent-c"]["position"], position(path))

    async def test_rebalance_uses_real_backlog_and_preserves_first_task_and_budgets(self):
        coordinator = self.make()
        await coordinator.initialize_index()
        shared = ["root", "shared", "work"]
        for agent in AGENTS:
            await coordinator.assign_task(agent, shared, "Existing duplicate scope")
        before = deepcopy(coordinator.agents)
        self.assertEqual(await coordinator.rebalance_attention([]), [])
        self.assertEqual(coordinator.agents, before)
        backlog = [{"task_path": shared, "task_text": "Already claimed"},
                   {"task_path": ["root", "backlog-one", "work"], "task_text": "First real backlog task"},
                   {"task_path": ["root", "backlog-two", "work"], "task_text": "Second real backlog task"}]
        proposals = await coordinator.rebalance_attention(backlog)
        self.assertEqual([p["agent"] for p in proposals], ["agent-b", "agent-c"])
        self.assertEqual(coordinator.agents["agent-a"]["task_path"], shared)
        for proposal, task in zip(proposals, backlog[1:]):
            state = coordinator.agents[proposal["agent"]]
            self.assertEqual(state["task_path"], task["task_path"])
            self.assertEqual(state["task_text"], task["task_text"])
            self.assertEqual(state["budget_remaining"], before[proposal["agent"]]["budget_remaining"])
            self.assertEqual(coordinator.index.nodes["anchor:" + proposal["agent"]]["position"], state["position"])

    async def test_novelty_uses_peer_distances_persists_and_replays_without_new_routes(self):
        coordinator = self.make()
        await coordinator.initialize_index()
        await coordinator.assign_task("agent-a", ["root"], "Root investigator")
        for agent in ("agent-b", "agent-c"):
            await coordinator.assign_task(agent, ["root", "remote", "topic", agent], "Other branch")
        events = []
        for number in range(10):
            event = KnowledgeEvent(str(uuid.uuid4()), coordinator.run_id, "agent-a", "evidence", "Independent bounded observation " + str(number),
                scope_path=["root"], confidence=1 if number == 9 else 0, verification_status="observed" if number == 9 else "unverified")
            coordinator.store.put_event(event, position(event.scope_path))
            decisions = await coordinator.route_event(event)
            events.append(event)
        saved = self.novelty(coordinator)
        self.assertEqual(len(saved["events"]), 10)
        self.assertTrue(saved["events"][-1]["selected"])
        self.assertAlmostEqual(saved["events"][-1]["distance"], 1.95)
        self.assertEqual(sum("diversity_probe" in d["reasons"] for d in decisions), 1)
        deliveries, ledger, searches = coordinator.store.deliveries(coordinator.run_id), coordinator.store.ledger(coordinator.run_id), len(coordinator.index.searches)
        await coordinator.route_event(events[-1])
        self.assertEqual(self.novelty(coordinator), saved)
        self.assertEqual(coordinator.store.deliveries(coordinator.run_id), deliveries)
        self.assertEqual(coordinator.store.ledger(coordinator.run_id), ledger)
        self.assertEqual(len(coordinator.index.searches), searches)
        resumed = self.make()
        await resumed.initialize_index()
        self.assertEqual(self.novelty(resumed), saved)
        self.assertEqual(resumed.store.deliveries(resumed.run_id), deliveries)

    async def test_routing_failure_rolls_back_novelty_metadata_and_decisions_together(self):
        coordinator = self.make()
        await coordinator.initialize_index()
        event = KnowledgeEvent(str(uuid.uuid4()), coordinator.run_id, "agent-a", "claim", "Required peer event",
                               dependencies=["agent-b"], scope_path=["root"])
        coordinator.store.put_event(event, position(event.scope_path))
        original = coordinator.store.route
        def reject_capacity(event_id, decisions, **kwargs):
            return original(event_id, decisions, max_pending=0, **kwargs)
        with patch.object(coordinator.store, "route", reject_capacity):
            with self.assertRaises(BufferError):
                await coordinator.route_event(event)
        self.assertIsNone(self.novelty(coordinator))
        self.assertEqual(coordinator.store.deliveries(coordinator.run_id), [])
        self.assertFalse(any(row["kind"] == "routing_decision" for row in coordinator.store.ledger(coordinator.run_id)))
        await coordinator.route_event(event)
        self.assertEqual(len(self.novelty(coordinator)["events"]), 1)


if __name__ == "__main__":
    unittest.main()
