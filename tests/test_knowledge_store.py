import concurrent.futures
import json
import uuid
import pytest
from astra_harness.knowledge_store import KnowledgeStore
from astra_harness.schema import KnowledgeEvent, canonical
from astra_harness.audit_ledger import verify


def fixture(tmp_path):
    store = KnowledgeStore(tmp_path / "graph.sqlite")
    run = str(uuid.uuid4())
    store.create_run(run, {"test": True})
    evidence = store.add_evidence(b'{"fact":42}')
    event = KnowledgeEvent(str(uuid.uuid4()), run, "agent-a", "evidence", "Measured42 code abcdef012345", evidence_refs=[evidence])
    return store, run, event


def test_durable_event_and_evidence_restart(tmp_path):
    store, run, event = fixture(tmp_path)
    assert store.put_event(event, [0.1, 0.2])
    assert not store.put_event(event, [0.1, 0.2])
    store.close()
    recovered = KnowledgeStore(tmp_path / "graph.sqlite")
    assert recovered.event(event.event_id) == event
    assert recovered.evidence(event.evidence_refs[0]) == b'{"fact":42}'
    assert verify(recovered.ledger())["valid"]
    recovered.close()


def test_evidence_required_before_publish(tmp_path):
    store, run, event = fixture(tmp_path)
    value = event.to_dict()
    value["evidence_refs"] = ["0" * 64]
    with pytest.raises(ValueError):
        store.put_event(KnowledgeEvent.from_dict(value), [0, 0])
    assert store.events(run) == []


def test_event_id_collision_rejected(tmp_path):
    store, run, event = fixture(tmp_path)
    store.put_event(event, [0, 0])
    value = event.to_dict()
    value["claim"] = "Changed meaning"
    with pytest.raises(ValueError, match="collision"):
        store.put_event(KnowledgeEvent.from_dict(value), [0, 0])


def test_delivery_ack_incorporation_distinct_and_idempotent(tmp_path):
    store, run, event = fixture(tmp_path)
    store.put_event(event, [0.1, 0.1])
    decision = [{"recipient": "agent-b", "deliver": True, "mandatory": True}]
    ident = store.route(event.event_id, decision)[0]
    assert store.route(event.event_id, decision) == []
    with pytest.raises(ValueError):
        store.transition(ident, "acknowledged")
    store.transition(ident, "accepted")
    assert store.deliveries(run)[0]["delivered_at"] is None
    store.transition(ident, "delivered")
    with pytest.raises(ValueError):
        store.transition(ident, "incorporated")
    store.transition(ident, "acknowledged")
    store.transition(ident, "incorporated")
    assert not store.transition(ident, "incorporated")
    assert len([r for r in store.ledger() if r["kind"] == "incorporated"]) == 1


def test_concurrent_publication_is_single_transaction(tmp_path):
    store, run, event = fixture(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: store.put_event(event, [0, 0]), range(3)))
    assert results.count(True) == 1
    assert len(store.events(run)) == 1
    assert verify(store.ledger())["valid"]


def test_backpressure_rolls_back_partial_fanout(tmp_path):
    store, run, event = fixture(tmp_path)
    store.put_event(event, [0, 0])
    with pytest.raises(BufferError):
        store.route(event.event_id, [{"recipient": "agent-b", "deliver": True}], max_pending=0)
    assert not store.deliveries(run)
    assert not any(row["kind"] == "routing_decision" for row in store.ledger())


def test_revision_and_coordinate_audit(tmp_path):
    store, run, event = fixture(tmp_path)
    store.put_event(event, [0, 0])
    value = event.to_dict()
    value.update(event_id=str(uuid.uuid4()), claim="Correction", revision_of=event.event_id)
    store.put_event(KnowledgeEvent.from_dict(value), [0.1, 0.1])
    store.update_coordinate(event.event_id, [0.2, 0.2], "reparent-v1")
    assert store.conn.execute("SELECT version FROM coordinates WHERE node_id=?", (event.event_id,)).fetchone()[0] == 2
    with pytest.raises(ValueError):
        store.update_coordinate(event.event_id, [1, 0], "invalid")


def test_tamper_detection(tmp_path):
    store, run, event = fixture(tmp_path)
    store.put_event(event, [0, 0])
    rows = store.ledger()
    rows[-1]["data"] = "{}"
    assert not verify(rows)["valid"]
