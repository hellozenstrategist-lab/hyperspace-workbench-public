import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from astra_harness.hyperbolic_index import LAYOUT_VERSION, coordinate_record, deserialize, distance, nearest, position, serialize, update_coordinate
from astra_harness.router import MODES, Router
from astra_harness.semantic import HashedTokenEncoder, cosine, embed, similarity

PATHS = st.lists(st.text(alphabet="abcdef0123456789", min_size=1, max_size=12), min_size=1, max_size=12)


@given(PATHS)
def test_layout_radius_encodes_depth_and_is_interior(path):
    point = position(path)
    assert sum(x*x for x in point) < 1
    assert distance([0, 0], point) == pytest.approx((len(path)-1)*.65, abs=1e-8)
    assert point == position(path, "changed semantic text", [[.1, .2]])


@given(PATHS, PATHS, PATHS)
@settings(max_examples=150)
def test_poincare_metric_properties(a, b, c):
    a, b, c = position(a), position(b), position(c)
    assert distance(a, a) == 0
    assert distance(a, b) >= 0
    assert distance(a, b) == pytest.approx(distance(b, a), abs=1e-10)
    assert distance(a, c) <= distance(a, b) + distance(b, c) + 1e-7


def test_reparent_changes_revision_without_text_warp():
    old = coordinate_record(["root", "organizer", "claim"])
    assert deserialize(serialize(old)) == old
    assert update_coordinate(old, old["path"], semantic_text="rewrite") == old
    moved = update_coordinate(old, ["root", "database", "claim"])
    assert moved["revision"] == 2 and moved["position"] != old["position"]
    assert moved["layout_version"] == LAYOUT_VERSION
    bad = dict(moved, layout_version="future-v99")
    with pytest.raises(ValueError, match="migration"):
        deserialize(bad)
    with pytest.raises(ValueError, match="does not match"):
        deserialize(dict(old, position=[.1, .2]))


def test_invalid_geometry_and_deterministic_neighbors():
    for point in ([1, 0], [float("nan"), 0], [0, 0, 0], [True, 0]):
        with pytest.raises(ValueError):
            distance(point, [0, 0])
    assert [row["id"] for row in nearest([0, 0], {"b": [.1, 0], "a": [-.1, 0]})] == ["a", "b"]
    assert nearest([0, 0], {}, 0) == []
    with pytest.raises(ValueError):
        position(["root"] * 18)


@given(st.text(max_size=100), st.text(max_size=100))
def test_lexical_representation_bounds(left, right):
    a, b = embed(left), embed(right)
    assert len(a) == 256
    assert -1 <= cosine(a, b) <= 1
    assert cosine(a, b) == pytest.approx(cosine(b, a))
    assert a == embed(left)


def test_lexical_representation_is_explicit_and_normalized():
    assert similarity("CLASSIFICATION precision", "classification precision") == pytest.approx(1)
    assert similarity("", "anything") == 0
    assert HashedTokenEncoder().describe()["learned"] is False
    assert HashedTokenEncoder().describe()["remote_calls"] is False


def fixture():
    return json.loads(Path(__file__).with_name("fixtures").joinpath("wrong_branch.json").read_text())


@pytest.mark.parametrize("mode", MODES)
def test_adversarial_wrong_branch_fixture(mode):
    case = fixture()
    rows = Router().route(case["event"], case["agents"], mode=mode)
    assert [r["recipient"] for r in rows if r["deliver"]] == case["expected_delivery"][mode]


@pytest.mark.parametrize("mode", MODES)
def test_mandatory_native_dependencies_override_distance_budget_and_novelty(mode):
    case = fixture()
    event = dict(case["event"], dependencies=["B", "C"], novelty=0, duplicate=True)
    for agent in case["agents"].values():
        agent.update(budget_remaining=0, seen_event_ids=[event["event_id"]])
    rows = Router().route(event, case["agents"], mode=mode)
    assert [r["recipient"] for r in rows if r["deliver"]] == ["B", "C"]
    assert all(r["mandatory"] and "explicit_dependency" in r["reasons"] for r in rows if r["deliver"])


@pytest.mark.parametrize("flags", [{"safety_constraint": True}, {"central": True}, {"verification_status": "disproved"}, {"priority": "critical", "verification_status": "observed"}, {"priority": "critical", "verification_status": "reproduced"}, {"type": "scope_change"}])
def test_native_critical_broadcast(flags):
    case = fixture()
    rows = Router().route(dict(case["event"], **flags), case["agents"], mode="flat")
    assert [r["recipient"] for r in rows if r["deliver"]] == ["B", "C"]
    assert all("critical_broadcast" in r["reasons"] for r in rows if r["deliver"])


def test_unverified_critical_does_not_claim_verification():
    case = fixture()
    rows = Router().route(dict(case["event"], priority="critical", verification_status="unverified"), case["agents"], mode="hyperbolic")
    assert not any("critical_broadcast" in r["reasons"] for r in rows)


def test_graph_owner_and_contradiction_links_native_schema():
    case = fixture()
    owner = {"old-claim": "C"}
    rows = Router().route(dict(case["event"], parent_ids=["old-claim"]), case["agents"], owners=owner, mode="graph")
    assert "parent_graph_owner" in rows[2]["reasons"] and rows[2]["deliver"]
    rows = Router().route(dict(case["event"], contradicts=["old-claim"]), case["agents"], owners=owner, mode="hyperbolic")
    assert "contradiction_owner" in rows[2]["reasons"] and rows[2]["mandatory"]


def test_actual_index_distances_drive_optional_attention():
    case = fixture()
    # Deliberately swap index distances to prove this contract is consumed.
    rows = Router().route(case["event"], case["agents"], mode="hyperbolic", neighbor_distances={"A": 0, "B": 9, "C": .1})
    assert [r["recipient"] for r in rows if r["deliver"]] == ["C"]
    assert all(r["distance_source"] == "hyperspace_index" for r in rows)
    assert rows[1]["attention_radius"] == 1.5
    with pytest.raises(ValueError, match="finite"):
        Router().route(case["event"], case["agents"], neighbor_distances={"B": float("nan")})


def test_novelty_budget_and_bounded_diversity():
    case = fixture()
    case["agents"]["B"]["seen_event_ids"] = [case["event"]["event_id"]]
    rows = Router().route(case["event"], case["agents"], mode="hybrid")
    assert not any(r["deliver"] for r in rows)
    rows = Router().route(dict(case["event"], diversity_probe=True), case["agents"], mode="hybrid")
    assert [r["recipient"] for r in rows if r["deliver"]] == ["C"]
    assert rows[2]["reasons"][-1] == "diversity_probe"
    case["agents"]["C"]["budget_remaining"] = 0
    assert not any(r["deliver"] for r in Router().route(dict(case["event"], diversity_probe=True), case["agents"], mode="hybrid"))


def test_shared_acceptance_parent_routes_all_peers_in_graph():
    parent = ["root", "project", "document-organizer"]
    agents = {ident: {"task_path": parent + [role], "task_text": role} for ident, role in zip("ABC", ("latency", "accuracy", "ram"))}
    rows = Router().route({"author_agent": "A", "scope_path": parent, "claim": "private finding"}, agents, mode="graph")
    assert [r["recipient"] for r in rows if r["deliver"]] == ["B", "C"]
