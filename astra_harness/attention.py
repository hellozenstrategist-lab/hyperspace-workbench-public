"""Deterministic attention proposals and bounded, caller-persisted novelty."""
from __future__ import annotations

from copy import deepcopy
import math

from .hyperbolic_index import distance as poincare_distance, normalize_path, position, validate_position


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError("Invalid " + label)
    return float(value)


def _task(task):
    if not isinstance(task, dict) or not isinstance(task.get("task_path"), list):
        raise ValueError("Tasks require hierarchy task_path and task_text")
    path = normalize_path(task["task_path"])
    if not path or any(len(label) > 200 for label in path):
        raise ValueError("Task hierarchy must contain bounded nonempty labels")
    if not isinstance(task.get("task_text"), str) or not 1 <= len(task["task_text"]) <= 12000:
        raise ValueError("Task text must contain 1..12000 characters")
    return tuple(path), task["task_text"]


class AttentionManager:
    def reassign_overlaps(self, agents, available_tasks, threshold=0.05):
        """Propose changes without modifying workers, budgets, tasks, or storage.

        Stable agent-ID order keeps the first overlapping worker. Backlog order
        is priority order. An assignment must clear every other current region;
        no real unclaimed task means no reassignment. This detects structural
        overlap, not duplicate hidden reasoning or semantic task equivalence.
        """
        threshold = _number(threshold, "overlap threshold")
        if (not isinstance(agents, dict) or len(agents) > 32
                or any(not isinstance(a, str) or not 1 <= len(a) <= 100 for a in agents)):
            raise ValueError("Agents require bounded string IDs")
        if not isinstance(available_tasks, list) or len(available_tasks) > 1000:
            raise ValueError("Backlog must be a bounded task list")
        paths, current, claimed = {}, {}, set()
        for agent, state in agents.items():
            paths[agent], _ = _task(state)
            current[agent] = validate_position(state.get("position", position(paths[agent])))
            claimed.add(paths[agent])
        backlog, texts = [], {}
        for task in available_tasks:
            path, text = _task(task)
            if path in texts and texts[path] != text:
                raise ValueError("One backlog task path has conflicting task text")
            if path not in texts:
                backlog.append((path, position(path)))
                texts[path] = text
        kept, proposals = [], []
        for agent in sorted(agents):
            overlaps = [(other, poincare_distance(current[agent], current[other])) for other in kept
                        if poincare_distance(current[agent], current[other]) <= threshold]
            if overlaps:
                for path, candidate in backlog:
                    if path in claimed or any(poincare_distance(candidate, point) <= threshold
                                              for other, point in current.items() if other != agent):
                        continue
                    other, overlap_distance = overlaps[0]
                    proposals.append({"agent": agent, "old_task_path": list(paths[agent]), "new_task_path": list(path),
                        "new_position": candidate[:], "reason": "Poincare overlap with " + other
                        + " (distance=" + format(overlap_distance, ".8g") + "); assigned existing unclaimed backlog task"})
                    current[agent] = candidate
                    claimed.add(path)
                    break
            kept.append(agent)
        return proposals


class ControlledNovelty:
    """At most one probe per interval unique observations and a global cap.

    snapshot() is JSON-compatible. The caller must durably commit its state and
    routing decision together, and deduplicate routing by event ID. A repeated
    ID replays the same bool, including True, without spending quota. State is
    deliberately finite: reaching max_events blocks new observations instead
    of silently forgetting IDs or resetting the cap. Verified is a trusted
    caller assertion, never a model's self-assigned verification label.
    """
    def __init__(self, state=None, *, max_probes=3, min_distance=1.5, max_events=10000):
        if type(max_probes) is not int or not 0 <= max_probes <= 1000:
            raise ValueError("Invalid global probe cap")
        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError("Invalid novelty observation cap")
        self.max_probes, self.max_events = max_probes, max_events
        self.min_distance = _number(min_distance, "minimum probe distance", positive=True)
        self.interval, self.events, self.decisions = None, [], {}
        self.since_probe, self.probes = 0, 0
        if state is not None:
            if not isinstance(state, dict) or set(state) != {"version", "max_probes", "max_events", "min_distance", "interval", "events"}:
                raise ValueError("Invalid novelty state")
            if (type(state["version"]) is not int or type(state["max_probes"]) is not int or type(state["max_events"]) is not int
                    or type(state["min_distance"]) not in (int, float)
                    or state["version"] != 1 or state["max_probes"] != max_probes or state["max_events"] != max_events
                    or state["min_distance"] != self.min_distance):
                raise ValueError("Persisted novelty configuration changed")
            if not isinstance(state["events"], list) or len(state["events"]) > max_events:
                raise ValueError("Invalid bounded novelty history")
            if state["events"] and state["interval"] is None:
                raise ValueError("Novelty history requires its original interval")
            if state["interval"] is not None and (type(state["interval"]) is not int or not 1 <= state["interval"] <= 10000):
                raise ValueError("Invalid persisted novelty interval")
            for record in state["events"]:
                if (not isinstance(record, dict) or set(record) != {"event_id", "distance", "verified", "selected"}
                        or type(record["selected"]) is not bool or not isinstance(record["event_id"], str)
                        or record["event_id"] in self.decisions):
                    raise ValueError("Invalid novelty history entry")
                selected = self.select_probe(record["event_id"], record["distance"], record["verified"], state["interval"])
                if selected != record["selected"]:
                    raise ValueError("Novelty history violates deterministic quota")
            self.interval = state["interval"]

    def select_probe(self, eventid, distance, verified, interval=10):
        if not isinstance(eventid, str) or not 1 <= len(eventid) <= 200:
            raise ValueError("Novelty event ID must be bounded text")
        distance = _number(distance, "probe distance")
        if type(verified) is not bool or type(interval) is not int or not 1 <= interval <= 10000:
            raise ValueError("Invalid verification flag or probe interval")
        if self.interval is not None and self.interval != interval:
            raise ValueError("Cannot change a persisted novelty interval")
        if eventid in self.decisions:
            previous = self.decisions[eventid]
            if previous["distance"] != distance or previous["verified"] != verified:
                raise ValueError("Novelty event ID already evaluated with different inputs")
            return previous["selected"]
        if len(self.events) >= self.max_events:
            raise BufferError("Novelty observation cap reached; explicit new scheduling epoch required")
        self.interval = interval
        self.since_probe += 1
        selected = bool(verified and distance >= self.min_distance and self.since_probe >= interval
                        and self.probes < self.max_probes)
        if selected:
            self.probes += 1
            self.since_probe = 0
        record = {"event_id": eventid, "distance": distance, "verified": verified, "selected": selected}
        self.events.append(record)
        self.decisions[eventid] = record
        return selected

    def snapshot(self):
        return deepcopy({"version": 1, "max_probes": self.max_probes, "max_events": self.max_events,
                         "min_distance": self.min_distance, "interval": self.interval, "events": self.events})
