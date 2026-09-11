#!/usr/bin/env python3
"""Bounded, labelled wrong-branch routing comparison; no model calls."""
import argparse
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_harness.hyperbolic_index import position
from astra_harness.router import MODES, Router


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    case = json.loads(Path(__file__).resolve().parents[1].joinpath("tests/fixtures/wrong_branch.json").read_text())
    distances, index = None, None
    if args.endpoint:
        from astra_harness.hyperspace_backend import HyperspaceBackend
        collection = "branchcheck_" + uuid.uuid4().hex[:16]
        index = HyperspaceBackend(args.endpoint, state_path=output.with_suffix(".sqlite"), collection=collection)
        for agent, task in case["agents"].items():
            index.upsert("anchor:" + agent, position(task["task_path"]), {"kind": "anchor", "agent": agent})
        rows = index.search(position(case["event"]["scope_path"]), 3, {"kind": "anchor"})
        distances = {row["metadata"]["agent"]: row["distance"] for row in rows}
        assert set(distances) == set(case["agents"]), rows
    report = {"fixture": case["description"], "fixture_count": 1, "general_quality_claim": False, "real_model_calls": 0, "index_backend": "actual_hyperspace" if index else "exact_local_poincare", "neighbor_distances": distances, "modes": {}}
    relevant = set(case["relevant_recipients"])
    for mode in MODES:
        decisions = Router().route(case["event"], case["agents"], mode=mode, neighbor_distances=distances)
        delivered = {row["recipient"] for row in decisions if row["deliver"]}
        assert sorted(delivered) == case["expected_delivery"][mode]
        required = Router().route(dict(case["event"], dependencies=["B", "C"]), case["agents"], mode=mode, neighbor_distances=distances)
        mandatory = [row["recipient"] for row in required if row["deliver"]]
        assert mandatory == ["B", "C"]
        report["modes"][mode] = {"optional_delivered": sorted(delivered), "precision_on_constructed_fixture": len(delivered & relevant)/len(delivered) if delivered else 0, "recall_on_constructed_fixture": len(delivered & relevant)/len(relevant), "false_positives": sorted(delivered-relevant), "false_negatives": sorted(relevant-delivered), "mandatory_delivered_separate_check": mandatory, "decisions": decisions}
    if index:
        report["health"] = index.health()
        index.close()
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(output), "index_backend": report["index_backend"], "modes": list(report["modes"])}))


if __name__ == "__main__":
    main()
