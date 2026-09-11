import asyncio
import uuid

from astra_harness.event_bus import EventBus
from astra_harness.knowledge_store import KnowledgeStore
from astra_harness.schema import KnowledgeEvent


def test_steer_metadata_distinguishes_optional_context_from_required_inbox_receipts(tmp_path):
    async def scenario():
        store = KnowledgeStore(tmp_path / "knowledge.sqlite")
        run_id = str(uuid.uuid4())
        store.create_run(run_id, {"test": "receipt_policy"})
        store.worker(run_id, "agent-b", state="active")
        optional = KnowledgeEvent(str(uuid.uuid4()), run_id, "coordinator", "hypothesis", "Optional unverified branch example")
        required = KnowledgeEvent(str(uuid.uuid4()), run_id, "agent-a", "evidence", "Required code: abcdef012345")
        # Policy intentionally does not derive from mandatory routing: required
        # receipts can be optional routes, while broadcasts can have no receipt.
        for event, mandatory in ((optional, True), (required, False)):
            store.put_event(event, [0, 0])
            store.route(event.event_id, [{"recipient": "agent-b", "deliver": True, "mandatory": mandatory}])
        class RecordingRuntime:
            def __init__(self):
                self.calls = []
            async def steer(self, agent, payload):
                self.calls.append(payload)
                return {"accepted": True}
        runtime = RecordingRuntime()
        bus = EventBus(store, run_id, runtime, expected_codes={required.event_id: "abcdef012345"})
        try:
            for row in store.deliveries(run_id):
                await bus._deliver(row["delivery_id"])
            payloads = {p["knowledge"]["event_id"]: p for p in runtime.calls}
            assert payloads[optional.event_id]["requires_ack"] is False
            assert payloads[optional.event_id]["context_kind"] == "optional_context"
            assert "no acknowledgement" in payloads[optional.event_id]["handling"]
            assert payloads[required.event_id]["requires_ack"] is True
            assert payloads[required.event_id]["context_kind"] == "required_receipt"
            assert "await_peer_findings before acknowledging" in payloads[required.event_id]["handling"]
            assert all(p["receipt_policy_version"] == "explicit_inbox_ack_v1" for p in runtime.calls)
            assert all(r["accepted_at"] and not r["delivered_at"] and not r["acknowledged_at"] for r in store.deliveries(run_id))
            result = await bus.inbox("agent-b", [required.event_id])
            assert [f["event_id"] for f in result["findings"]] == [required.event_id]
            bus.acknowledge("agent-b", [{"event_id": required.event_id, "evidence_code": "abcdef012345"}], "ack")
            optional_row = next(r for r in store.deliveries(run_id) if r["event_id"] == optional.event_id)
            assert not optional_row["delivered_at"] and not optional_row["acknowledged_at"]
        finally:
            await bus.close()
            store.close()
    asyncio.run(scenario())
