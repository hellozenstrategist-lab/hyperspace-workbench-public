#!/usr/bin/env python3
"""Exercise the actual persistent index, including reparenting and restart."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_harness.hyperbolic_index import distance, position
from astra_harness.hyperspace_backend import HyperspaceBackend
from astra_harness.router import Router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    collection = "indexcheck_" + uuid.uuid4().hex[:16]
    mapping = output.with_suffix(".sqlite")
    original_env = dict(os.environ)
    paths = {"A": ["root", "project", "document-organizer", "latency"], "B": ["root", "project", "document-organizer", "accuracy"], "C": ["root", "project", "marketing"]}
    agents = {name: {"position": position(path), "task_path": path, "task_text": "sorting latency accuracy"} for name, path in paths.items()}
    result = {"collection": collection, "mapping": str(mapping), "positions": {name: info["position"] for name, info in agents.items()}, "real_server": True, "real_model_calls": 0}
    db = HyperspaceBackend(args.endpoint, state_path=mapping, collection=collection)
    result["before"] = db.health()
    for agent, anchor in agents.items():
        db.upsert("anchor:" + agent, anchor["position"], {"kind": "anchor", "agent": agent, "run_id": collection})
    query = agents["A"]["position"]
    rows = db.search(query, 3, {"kind": "anchor", "run_id": collection})
    assert len(rows) == 3, rows
    errors = [abs(row["distance"] - distance(query, agents[row["metadata"]["agent"]]["position"])) for row in rows]
    assert max(errors) < 1e-9, errors
    result["anchor_neighbors"] = rows
    result["maximum_distance_error"] = max(errors)
    d = {row["metadata"]["agent"]: row["distance"] for row in rows}
    event = {"author_agent": "coordinator", "claim": "sorting latency accuracy", "scope_path": paths["A"]}
    result["index_routing"] = Router().route(event, agents, mode="hyperbolic", neighbor_distances=d)
    assert {r["recipient"] for r in result["index_routing"] if r["deliver"]} == {"A", "B"}
    point = position(["root", "project", "document-organizer", "finding"])
    receipt = db.upsert("moving-finding", point, {"kind": "knowledge", "revision": 1})
    moved = position(["root", "project", "database", "finding"])
    updated = db.upsert("moving-finding", moved, {"kind": "knowledge", "revision": 2})
    assert receipt["vector_id"] == updated["vector_id"]
    fetched = db.get("moving-finding")
    assert fetched["metadata"]["revision"] == 2 and distance(fetched["position"], moved) < 1e-9, fetched
    result["updated_record"] = fetched
    result["updated_search"] = db.search(moved, 1, {"kind": "knowledge"})
    assert result["updated_search"][0]["id"] == "moving-finding" and result["updated_search"][0]["distance"] < 1e-9
    db.close()
    endpoint = args.endpoint
    if args.restart:
        service = Path(__file__).with_name("hyperspace_service.py")
        stop = subprocess.run([sys.executable, str(service), "stop"], capture_output=True, text=True, check=True)
        result["stopped"] = json.loads(stop.stdout)
        assert not result["stopped"]["running"]
        start = subprocess.run([sys.executable, str(service), "start"], capture_output=True, text=True, check=True)
        result["restarted"] = json.loads(start.stdout)
        endpoint = result["restarted"]["endpoint"]
    db = HyperspaceBackend(endpoint, state_path=mapping, collection=collection)
    fetched = db.get("moving-finding")
    assert fetched["metadata"]["revision"] == 2 and distance(fetched["position"], moved) < 1e-9, fetched
    assert len(db.search(query, 3, {"kind": "anchor", "run_id": collection})) == 3
    result["after"] = db.health()
    result["persistent_record"] = fetched
    result["reconcile"] = db.reconcile([{"id": "moving-finding", "position": moved, "metadata": {"kind": "knowledge", "revision": 2}}])
    assert db.get("moving-finding")["vector_id"] == updated["vector_id"]
    db.close()
    # Losing the separate local ID mapping must never overwrite an old point.
    wrong_map = output.with_suffix(".wrong-map.sqlite")
    other = HyperspaceBackend(endpoint, state_path=wrong_map, collection=collection)
    try:
        other.upsert("unrelated-new-node", [.1, .1], {})
    except RuntimeError as exc:
        assert "mapping disagrees" in str(exc)
        result["lost_mapping_collision_rejected"] = True
    else:
        raise AssertionError("lost mapping allowed an unrelated record overwrite")
    finally:
        other.close()
    assert dict(os.environ) == original_env, "backend changed global environment"
    result.update(passed=True, restart_tested=args.restart, global_environment_unchanged=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "endpoint": endpoint, "restart_tested": args.restart, "max_distance_error": max(errors)}))


if __name__ == "__main__":
    main()
