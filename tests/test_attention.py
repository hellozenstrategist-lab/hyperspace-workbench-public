from copy import deepcopy
import json
import math

import pytest

from astra_harness.attention import AttentionManager, ControlledNovelty
from astra_harness.hyperbolic_index import distance, position


def task(branch):
    return {"task_path": ["root", branch, "work"], "task_text": "Real task for " + branch}


def agent(branch, point=None):
    result = task(branch)
    result.update(position=point if point is not None else position(result["task_path"]), budget_remaining=50)
    return result


def test_overlap_assigns_real_unclaimed_backlog_preserves_first_and_inputs():
    agents = {"agent-c": agent("shared"), "agent-a": agent("shared"), "agent-b": agent("shared")}
    backlog = [task("shared"), task("backlog-one"), task("backlog-two")]
    before = deepcopy((agents, backlog))
    proposals = AttentionManager().reassign_overlaps(agents, backlog)
    assert [p["agent"] for p in proposals] == ["agent-b", "agent-c"]
    assert [p["new_task_path"] for p in proposals] == [backlog[1]["task_path"], backlog[2]["task_path"]]
    assert all(p["new_position"] == position(p["new_task_path"]) for p in proposals)
    assert (agents, backlog) == before
    assert proposals == AttentionManager().reassign_overlaps(dict(reversed(list(agents.items()))), backlog)


def test_no_backlog_does_not_invent_tasks_or_force_reassignment():
    agents = {"a": agent("same"), "b": agent("same")}
    assert AttentionManager().reassign_overlaps(agents, []) == []
    assert AttentionManager().reassign_overlaps(agents, [task("same")]) == []


def test_actual_poincare_metric_not_euclidean_or_path_equality():
    # Near the unit boundary, close Euclidean points can be far hyperbolically.
    agents = {"a": agent("same", [0.99, 0]), "b": agent("same", [0.991, 0])}
    assert distance(agents["a"]["position"], agents["b"]["position"]) > 0.05
    assert AttentionManager().reassign_overlaps(agents, [task("backlog")]) == []
    agents["b"] = agent("different-label", [0.99, 0])
    assert len(AttentionManager().reassign_overlaps(agents, [task("backlog")])) == 1


def test_candidate_must_clear_every_other_current_region():
    backlog = task("backlog")
    agents = {"a": agent("same"), "b": agent("same"), "c": agent("third", position(backlog["task_path"]))}
    assert AttentionManager().reassign_overlaps(agents, [backlog]) == []


def test_claimed_task_and_conflicting_backlog_are_rejected_or_skipped():
    agents = {"a": agent("same"), "b": agent("same"), "c": agent("claimed")}
    proposals = AttentionManager().reassign_overlaps(agents, [task("claimed"), task("free")])
    assert proposals[0]["new_task_path"] == task("free")["task_path"]
    with pytest.raises(ValueError, match="conflicting"):
        AttentionManager().reassign_overlaps(agents, [task("free"), {**task("free"), "task_text": "Different assignment"}])


@pytest.mark.parametrize("threshold", [-1, math.nan, math.inf, True])
def test_invalid_overlap_threshold(threshold):
    with pytest.raises(ValueError):
        AttentionManager().reassign_overlaps({}, [], threshold)


def test_probe_interval_lifetime_cap_and_duplicate_replay():
    scheduler = ControlledNovelty(max_probes=2)
    selected = [i for i in range(1, 41) if scheduler.select_probe(str(i), 2.0, True)]
    assert selected == [10, 20]
    before = scheduler.snapshot()
    assert scheduler.select_probe("10", 2.0, True)
    assert not scheduler.select_probe("9", 2.0, True)
    assert scheduler.snapshot() == before


def test_only_verified_distant_candidates_probe_without_burst_after_idle():
    scheduler = ControlledNovelty()
    for i in range(10):
        assert not scheduler.select_probe(str(i), 2.0, False)
    assert not scheduler.select_probe("near", 0.1, True)
    assert scheduler.select_probe("far", 3.0, True)
    assert not scheduler.select_probe("far2", 3.0, True)


def test_resume_json_state_identical_decisions_and_defensive_snapshot_copy():
    original = ControlledNovelty()
    for i in range(13):
        original.select_probe(str(i), 2.0, True)
    state = json.loads(json.dumps(original.snapshot(), sort_keys=True))
    resumed = ControlledNovelty(state)
    for i in range(13, 40):
        assert resumed.select_probe(str(i), 2.0, True) == original.select_probe(str(i), 2.0, True)
    assert resumed.snapshot() == original.snapshot()
    state["events"][0]["selected"] = True
    assert not resumed.snapshot()["events"][0]["selected"]


def test_finite_state_fails_closed_does_not_forget_old_ids():
    scheduler = ControlledNovelty(max_events=2)
    scheduler.select_probe("a", 2, True)
    scheduler.select_probe("b", 2, True)
    before = scheduler.snapshot()
    with pytest.raises(BufferError):
        scheduler.select_probe("c", 2, True)
    assert not scheduler.select_probe("a", 2, True)
    assert scheduler.snapshot() == before


def test_changed_inputs_interval_or_corrupt_history_refused():
    scheduler = ControlledNovelty()
    scheduler.select_probe("a", 2, True)
    with pytest.raises(ValueError, match="different inputs"):
        scheduler.select_probe("a", 3, True)
    with pytest.raises(ValueError, match="interval"):
        scheduler.select_probe("b", 2, True, interval=1)
    state = scheduler.snapshot()
    state["events"][0]["selected"] = True
    with pytest.raises(ValueError, match="quota"):
        ControlledNovelty(state)
    with pytest.raises(ValueError, match="configuration"):
        ControlledNovelty(scheduler.snapshot(), max_probes=5)


@pytest.mark.parametrize("distance,verified,interval", [(math.nan, True, 10), (-1, True, 10), (True, True, 10),
                                                      (2, "verified", 10), (2, True, False), (2, True, 0)])
def test_invalid_probe_observation_does_not_consume_capacity(distance, verified, interval):
    scheduler = ControlledNovelty()
    before = scheduler.snapshot()
    with pytest.raises(ValueError):
        scheduler.select_probe("a", distance, verified, interval)
    assert scheduler.snapshot() == before
