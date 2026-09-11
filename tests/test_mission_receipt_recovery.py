"""Constructor recovery and manifest metadata, without model calls."""
import asyncio
import json
from unittest.mock import patch
import uuid

import pytest

from astra_harness.coordinator import Coordinator
from astra_harness.event_bus import RECEIPT_POLICY_VERSION
from astra_harness.mission import Mission
from astra_harness.schema import AGENTS, KnowledgeEvent


class NoModelRuntime:
    def __init__(self, *args):
        self.payloads = []

    async def steer(self, agent, payload):
        self.payloads.append(payload)
        return {"accepted": True}


def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"objective": "Review supplied notes", "workers": [
        {"id": agent, "prompt": "Review your supplied notes", "task_path": ["root", "notes", agent]} for agent in AGENTS]}))
    return path


def test_constructor_restores_exact_receipts_before_any_replayed_tool(tmp_path):
    async def scenario():
        path = config(tmp_path)
        with patch("astra_harness.mission.Runtime", NoModelRuntime), patch("astra_harness.coordinator.Runtime", NoModelRuntime):
            first = Mission(path, tmp_path / "mission")
            result = await first.on_tool("agent-a", "publish_knowledge", {"key": "initial", "kind": "claim",
                "claim": "Authored text with an internal Receipt: marker"}, "publish")
            expected = dict(first.bus.expected_codes)
            optional = KnowledgeEvent(str(uuid.uuid4()), first.run_id, "coordinator", "hypothesis", "Optional context without a receipt")
            first.store.put_event(optional, [0, 0])
            first.close()
            resumed = Mission(path, tmp_path / "mission")
            try:
                assert resumed.bus.expected_codes == expected
                assert optional.event_id not in resumed.bus.expected_codes
                assert resumed.manifest["receipt_policy_version"] == RECEIPT_POLICY_VERSION
                assert resumed.runtime.payloads == []
                resumed.store.worker(resumed.run_id, "agent-b", state="active")
                row = next(r for r in resumed.store.deliveries(resumed.run_id, "agent-b") if r["event_id"] == result["event_id"])
                await resumed.bus._deliver(row["delivery_id"])
                assert resumed.runtime.payloads[0]["requires_ack"] is True
                assert resumed.runtime.payloads[0]["context_kind"] == "required_receipt"
            finally:
                await resumed.bus.close()
                resumed.close()
            coordinator = Coordinator(tmp_path / "acceptance")
            try:
                assert coordinator.manifest["receipt_policy_version"] == RECEIPT_POLICY_VERSION
            finally:
                coordinator.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("claim", ["No explicit receipt suffix", "Claim Receipt: 0000000000000000", "Claim Receipt: ABCDEF0123456789"])
def test_constructor_refuses_invalid_persisted_receipts_and_releases_run_lock(tmp_path, claim):
    path = config(tmp_path)
    with patch("astra_harness.mission.Runtime", NoModelRuntime):
        first = Mission(path, tmp_path / "mission")
        first.store.put_event(KnowledgeEvent(str(uuid.uuid4()), first.run_id, "agent-a", "claim", claim), [0, 0])
        first.close()
        for _ in range(2):
            with pytest.raises(ValueError, match="receipt does not match"):
                Mission(path, tmp_path / "mission")
