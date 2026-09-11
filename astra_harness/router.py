"""Explainable attention routing; mandatory delivery precedes relevance scoring."""
from __future__ import annotations

import math
from collections.abc import Mapping

from .hyperbolic_index import distance, normalize_path, position
from .semantic import similarity

MODES = ("broadcast", "flat", "graph", "hyperbolic", "hybrid")
DEFAULT_ATTENTION_RADIUS = 1.5
DEFAULT_LEXICAL_THRESHOLD = 0.20


def _path_related(left, right):
    a, b = normalize_path(left), normalize_path(right)
    if not a or not b:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    # Ancestor/descendant, or two immediate sibling tasks with a shared parent.
    # A bare global root is not enough to relate otherwise unrelated branches.
    return common == min(len(a), len(b)) or (common >= 2 and common >= max(len(a), len(b)) - 1)


def _text(event):
    value = event.get("claim", event.get("text", event.get("content")))
    if value is not None:
        return str(value)
    payload = event.get("payload", {})
    if isinstance(payload, dict):
        return str(payload.get("finding", payload.get("text", payload.get("content", ""))))
    return str(payload)


def _bounded(value, default):
    try:
        number = float(value)
        return max(0.0, min(1.0, number)) if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


class Router:
    def __init__(self, attention_radius=DEFAULT_ATTENTION_RADIUS, lexical_threshold=DEFAULT_LEXICAL_THRESHOLD):
        if not math.isfinite(attention_radius) or attention_radius <= 0:
            raise ValueError("attention_radius must be finite and positive")
        if not 0 <= lexical_threshold <= 1:
            raise ValueError("lexical_threshold must be in [0,1]")
        self.attention_radius = float(attention_radius)
        self.lexical_threshold = float(lexical_threshold)

    def route(self, event: dict, agents: dict, owners: dict | None = None, dependencies: list | None = None, mode: str = "hybrid", neighbor_distances: dict | None = None) -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"unknown routing mode {mode!r}")
        owners, dependencies = owners or {}, dependencies or []
        sender = event.get("author_agent", event.get("from", event.get("sender", event.get("agent_id"))))
        event_id = event.get("id", event.get("event_id"))
        event_path = event.get("scope_path", event.get("task_path", event.get("path", [])))
        event_position = event.get("position") or position(event_path)
        text = _text(event)
        kind = event.get("kind", event.get("type", "finding"))
        broadcast_required = (
            kind in ("safety", "safety_alert", "scope_change", "critical_verification", "central", "disproved")
            or bool(event.get("scope_change")) or bool(event.get("safety_critical"))
            or bool(event.get("safety_constraint")) or bool(event.get("central"))
            or event.get("verification_status") == "disproved"
            or (event.get("priority") == "critical" and event.get("verification_status") in ("observed", "reproduced", "verified"))
            or (kind == "verification" and event.get("severity") == "critical")
        )
        node_ids = [event.get("node_id"), event.get("target_node_id")]
        node_ids.extend(event.get("related_node_ids", []))
        contradicts = event.get("contradicts", [])
        node_ids.extend([contradicts] if isinstance(contradicts, str) else contradicts)
        contradiction_owners = set()
        if kind == "contradiction" or event.get("contradicts"):
            for node_id in node_ids:
                value = owners.get(node_id)
                if isinstance(value, str):
                    contradiction_owners.add(value)
                elif isinstance(value, (list, tuple, set)):
                    contradiction_owners.update(value)
        parent_owners = set()
        for node_id in event.get("parent_ids", []):
            value = owners.get(node_id)
            if isinstance(value, str):
                parent_owners.add(value)
            elif isinstance(value, (list, tuple, set)):
                parent_owners.update(value)
        required = set(dependencies) | set(event.get("dependencies", []))
        novelty = 0.0 if event.get("duplicate") else _bounded(event.get("novelty", 1), 1.0)
        decisions = []
        for agent_id in sorted(agents):
            agent = agents[agent_id]
            agent_path = agent.get("task_path", agent.get("path", []))
            anchor = agent.get("position") or position(agent_path)
            supplied = (neighbor_distances or {}).get(agent_id)
            if isinstance(supplied, Mapping):
                supplied = supplied.get("distance")
            if supplied is not None:
                d = float(supplied)
                if not math.isfinite(d) or d < 0:
                    raise ValueError("actual index distances must be finite and nonnegative")
                distance_source = "hyperspace_index"
            else:
                d = distance(event_position, anchor)
                distance_source = "local_poincare"
            lexical = max(0.0, similarity(text, agent.get("task_text", agent.get("text", ""))))
            path_related = _path_related(event_path, agent_path)
            graph_related = path_related or agent_id in parent_owners
            radius = float(agent.get("attention_radius", self.attention_radius))
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("agent attention_radius must be finite and positive")
            nearby = d <= radius
            reasons = []
            if agent_id in required:
                reasons.append("explicit_dependency")
            if broadcast_required:
                reasons.append("critical_broadcast")
            if agent_id in contradiction_owners:
                reasons.append("contradiction_owner")
            mandatory = bool(reasons)
            if mandatory:
                deliver = True
            elif mode == "broadcast":
                deliver = True
                reasons.append("broadcast_baseline")
            elif mode == "flat":
                deliver = lexical >= self.lexical_threshold
                reasons.append("lexical_match" if deliver else "low_lexical_similarity")
            elif mode == "graph":
                deliver = graph_related
                reasons.append("parent_graph_owner" if agent_id in parent_owners else "shared_task_branch" if deliver else "unrelated_task_branch")
            elif mode == "hyperbolic":
                deliver = nearby
                reasons.append("within_attention_radius" if deliver else "outside_attention_radius")
            else:
                deliver = nearby and (graph_related or lexical >= self.lexical_threshold)
                reasons.append("within_attention_radius" if nearby else "outside_attention_radius")
                reasons.append("shared_task_branch" if path_related else "unrelated_task_branch")
                if lexical >= self.lexical_threshold:
                    reasons.append("lexical_match")
                if agent_id in parent_owners:
                    reasons.append("parent_graph_owner")
            seen_ids = agent.get("seen_event_ids", [])
            seen_hashes = agent.get("seen_content_hashes", [])
            seen = (event_id is not None and event_id in seen_ids) or (event.get("content_hash") is not None and event["content_hash"] in seen_hashes)
            if not mandatory and (seen or novelty == 0):
                deliver = False
                reasons.append("already_seen" if seen else "no_novel_information")
            if not mandatory and agent.get("budget_remaining", 1) <= 0:
                deliver = False
                reasons.append("attention_budget_exhausted")
            if agent_id == sender:
                deliver = False
                mandatory = False
                reasons.append("sender_already_knows")
            hierarchy_score = math.exp(-d / radius)
            score = 1.0 if mandatory else {"broadcast": 1.0, "flat": lexical, "graph": float(graph_related), "hyperbolic": hierarchy_score, "hybrid": 0.55 * hierarchy_score + 0.35 * lexical + 0.10 * novelty}[mode]
            decisions.append({"recipient": agent_id, "deliver": deliver, "mandatory": mandatory, "reasons": reasons, "score": score, "score_calibrated": False, "distance": d, "distance_source": distance_source, "attention_radius": radius, "lexical_similarity": lexical, "path_related": path_related, "graph_related": graph_related, "novelty": novelty, "mode": mode})
        # An explicit, bounded probe exposes one additional branch to competing
        # evidence. It never weakens mandatory delivery or silently broadcasts.
        if event.get("diversity_probe") and novelty >= 0.5:
            eligible = [row for row in decisions if not row["deliver"] and row["recipient"] != sender and not any(reason in row["reasons"] for reason in ("already_seen", "attention_budget_exhausted"))]
            if eligible:
                selected = min(eligible, key=lambda row: (row["distance"], row["recipient"]))
                selected["deliver"] = True
                selected["reasons"].append("diversity_probe")
        return decisions


def route(event, agents, owners=None, dependencies=None, mode="hybrid", neighbor_distances=None):
    return Router().route(event, agents, owners, dependencies, mode, neighbor_distances)
