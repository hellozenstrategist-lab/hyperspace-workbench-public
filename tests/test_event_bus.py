"""Real SQLite inbox tests with a deterministic fake model transport."""
import asyncio
import json
import uuid
from unittest.mock import patch

import pytest

from astra_harness.event_bus import EventBus
from astra_harness.knowledge_store import KnowledgeStore
from astra_harness.schema import AGENTS, KnowledgeEvent


class FakeRuntime:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls = []
        self.inflight = 0
        self.max_inflight = 0

    async def steer(self, agent, event):
        self.calls.append((agent, event))
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.behavior:
                return await self.behavior(agent, event)
            await asyncio.sleep(0)
            return {"accepted": True}
        finally:
            self.inflight -= 1


def fixture(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    run_id = str(uuid.uuid4())
    store.create_run(run_id, {"test": "event_bus", "worker_count": 3})
    for agent in AGENTS:
        store.worker(run_id, agent, state="active")
    return store, run_id


def publish(store, run_id, recipients, *, code="abcdef012345", mandatory=True):
    event = KnowledgeEvent(str(uuid.uuid4()), run_id, "coordinator", "evidence",
                           "Measured fixture evidence code " + code, priority="important")
    store.put_event(event, [0.1, 0.1])
    store.route(event.event_id, [{"recipient": agent, "deliver": True, "mandatory": mandatory} for agent in recipients])
    return event


async def until(predicate, timeout=1):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("Test condition did not arrive")
        await asyncio.sleep(0.005)


def test_three_recipients_steer_concurrently_before_explicit_ack(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        gate = asyncio.Event()
        entered = set()

        async def behavior(agent, event):
            entered.add(agent)
            if len(entered) == 3:
                gate.set()
            await gate.wait()
            return {"accepted": True}

        runtime = FakeRuntime(behavior)
        event = publish(store, run_id, AGENTS)
        bus = EventBus(store, run_id, runtime)
        try:
            await bus.start()
            findings = await asyncio.wait_for(asyncio.gather(*(bus.inbox(a, [event.event_id]) for a in AGENTS)), 1)
            assert runtime.max_inflight == 3
            assert all(r["accepted_at"] and r["delivered_at"] and not r["acknowledged_at"] for r in store.deliveries(run_id))
            for agent, result in zip(AGENTS, findings):
                assert result["findings"][0]["event_id"] == event.event_id
                bus.acknowledge(agent, [{"event_id": event.event_id, "evidence_code": "abcdef012345"}], "call-" + agent)
            assert all(r["state"] == "acknowledged" for r in store.deliveries(run_id))
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_accepted_steer_is_not_implicit_delivery_or_ack(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        runtime = FakeRuntime()
        event = publish(store, run_id, ["agent-b"])
        bus = EventBus(store, run_id, runtime, expected_codes={event.event_id: "abcdef012345"})
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["accepted_at"])
            with pytest.raises(ValueError, match="not delivered"):
                bus.acknowledge("agent-b", [{"event_id": event.event_id, "evidence_code": "abcdef012345"}], "early")
            await bus.inbox("agent-b", [event.event_id])
            with pytest.raises(ValueError, match="evidence code"):
                bus.acknowledge("agent-b", [{"event_id": event.event_id, "evidence_code": "Measured fixture"}], "wrong")
            assert not store.deliveries(run_id)[0]["acknowledged_at"]
            bus.acknowledge("agent-b", [{"event_id": event.event_id, "evidence_code": "abcdef012345"}], "right")
            assert store.deliveries(run_id)[0]["acknowledged_at"]
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_inactive_turn_waits_and_cannot_pull(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        store.worker(run_id, "agent-b", state="prepared")
        event = publish(store, run_id, ["agent-b"])
        runtime = FakeRuntime()
        bus = EventBus(store, run_id, runtime)
        try:
            await bus.start()
            await asyncio.sleep(0.03)
            assert runtime.calls == []
            with pytest.raises(RuntimeError, match="inactive"):
                await bus.inbox("agent-b", [event.event_id])
            store.worker(run_id, "agent-b", state="active")
            bus.published()
            await until(lambda: store.deliveries(run_id)[0]["accepted_at"])
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_explicit_not_accepted_retries_then_succeeds(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"])
        attempts = []

        async def behavior(agent, event):
            attempts.append(agent)
            return {"accepted": False, "reason": "no_active_turn"} if len(attempts) == 1 else {"accepted": True}

        bus = EventBus(store, run_id, FakeRuntime(behavior))
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "retry_wait")
            row = store.deliveries(run_id)[0]
            assert row["attempts"] == 1 and row["accepted_at"] is None
            with store.transaction() as conn:
                conn.execute("UPDATE deliveries SET next_attempt=0 WHERE delivery_id=?", (row["delivery_id"],))
            bus.published()
            await until(lambda: store.deliveries(run_id)[0]["accepted_at"])
            assert len(attempts) == 2
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_rejected_steering_has_attempt_limit(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"])

        async def rejected(*_):
            return {"accepted": False, "reason": "not_ready"}

        runtime = FakeRuntime(rejected)
        bus = EventBus(store, run_id, runtime, max_attempts=1)
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "failed")
            assert len(runtime.calls) == 1
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_ambiguous_steer_is_not_resent_and_inbox_recovers(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        event = publish(store, run_id, ["agent-b"])

        async def disconnected(*_):
            raise ConnectionError("fixture transport loss")

        runtime = FakeRuntime(disconnected)
        bus = EventBus(store, run_id, runtime)
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "uncertain")
            await asyncio.sleep(0.06)
            assert len(runtime.calls) == 1
            await bus.inbox("agent-b", [event.event_id])
            delivery = next(json.loads(r["data"]) for r in store.ledger() if r["kind"] == "delivered")
            assert delivery["channel"] == "inbox_fallback_uncertain"
            assert store.deliveries(run_id)[0]["accepted_at"] is None
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_late_steer_error_cannot_regress_acknowledged_state(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        event = publish(store, run_id, ["agent-b"])

        async def delayed(*_):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                raise ConnectionError("late fixture outcome")

        runtime = FakeRuntime(delayed)
        bus = EventBus(store, run_id, runtime, steer_timeout=1, inbox_steer_wait=0.02)
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "attempting")
            await bus.inbox("agent-b", [event.event_id])
            bus.acknowledge("agent-b", [{"event_id": event.event_id, "evidence_code": "abcdef012345"}], "ack")
            await asyncio.sleep(0.03)
            assert store.deliveries(run_id)[0]["state"] == "acknowledged"
            assert len(runtime.calls) == 1
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_restart_quarantines_attempting_and_preserves_inbox(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        event = publish(store, run_id, ["agent-b"])
        ident = store.deliveries(run_id)[0]["delivery_id"]
        store.transition(ident, "attempting", attempt=True)
        store.close()
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        runtime = FakeRuntime()
        bus = EventBus(store, run_id, runtime)
        try:
            await bus.start()
            assert store.deliveries(run_id)[0]["state"] == "uncertain"
            await bus.inbox("agent-b", [event.event_id])
            assert runtime.calls == []
            assert store.deliveries(run_id)[0]["delivered_at"]
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_cumulative_inbox_budget_and_mandatory_backpressure(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        first = publish(store, run_id, ["agent-b"], code="111111111111")
        first_row = store.deliveries(run_id)[0]
        store.transition(first_row["delivery_id"], "delivered", details={"channel": "fixture_inbox"})
        second = publish(store, run_id, ["agent-b"], code="222222222222")
        rows = store.deliveries(run_id)
        runtime = FakeRuntime()
        bus = EventBus(store, run_id, runtime, context_budget=sum(r["token_cost"] for r in rows) - 1)
        try:
            await bus.start()
            with pytest.raises(BufferError, match="cumulative"):
                await bus.inbox("agent-b", [second.event_id])
            await until(lambda: any(r["kind"] == "backpressure" for r in store.ledger()))
            assert runtime.calls == []
            assert store.event(first.event_id) and store.event(second.event_id)
            assert next(r for r in store.deliveries(run_id) if r["event_id"] == second.event_id)["state"] == "queued"
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_optional_context_overflow_expires_without_call(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"], mandatory=False)
        runtime = FakeRuntime()
        bus = EventBus(store, run_id, runtime, context_budget=1)
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "expired")
            assert runtime.calls == []
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_declined_attempt_transition_never_sends_stale_snapshot(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"])
        ident = store.deliveries(run_id)[0]["delivery_id"]
        runtime = FakeRuntime()
        bus = EventBus(store, run_id, runtime)
        original = store.transition

        def race(delivery_id, state, **kwargs):
            if state == "attempting":
                original(delivery_id, "delivered", details={"channel": "concurrent_inbox_fixture"})
                return False
            return original(delivery_id, state, **kwargs)

        try:
            with patch.object(store, "transition", side_effect=race):
                await bus._deliver(ident)
            assert runtime.calls == []
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_ack_batch_validates_all_before_mutation(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        first = publish(store, run_id, ["agent-b"], code="111111111111")
        second = publish(store, run_id, ["agent-b"], code="222222222222")
        bus = EventBus(store, run_id, FakeRuntime())
        try:
            await bus.start()
            await bus.inbox("agent-b", [first.event_id, second.event_id])
            with pytest.raises(ValueError):
                bus.acknowledge("agent-b", [{"event_id": first.event_id, "evidence_code": "111111111111"},
                                            {"event_id": second.event_id, "evidence_code": "not-in-claim"}], "bad-batch")
            assert all(not r["acknowledged_at"] for r in store.deliveries(run_id))
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_same_recipient_reservation_prevents_concurrent_overcommit(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"], code="111111111111")
        publish(store, run_id, ["agent-b"], code="222222222222")
        budget = sum(r["token_cost"] for r in store.deliveries(run_id)) - 1
        release = asyncio.Event()

        async def delayed(*_):
            await release.wait()
            return {"accepted": True}

        runtime = FakeRuntime(delayed)
        bus = EventBus(store, run_id, runtime, context_budget=budget)
        try:
            await bus.start()
            await until(lambda: len(runtime.calls) == 1)
            await asyncio.sleep(0.02)
            assert len(runtime.calls) == 1
            release.set()
            await until(lambda: any(r["kind"] == "backpressure" for r in store.ledger()))
            assert len(runtime.calls) == 1 and runtime.max_inflight == 1
            assert len([r for r in store.deliveries(run_id) if r["accepted_at"]]) == 1
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())


def test_shutdown_cancels_pending_steer_and_records_uncertainty(tmp_path):
    async def scenario():
        store, run_id = fixture(tmp_path)
        publish(store, run_id, ["agent-b"])

        async def pending(*_):
            await asyncio.Event().wait()

        bus = EventBus(store, run_id, FakeRuntime(pending), steer_timeout=60)
        try:
            await bus.start()
            await until(lambda: store.deliveries(run_id)[0]["state"] == "attempting")
            await asyncio.wait_for(bus.close(), 0.2)
            assert store.deliveries(run_id)[0]["state"] == "uncertain"
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())
