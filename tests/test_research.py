"""Exercise the real research coordinator/runtime with simulated HTTP only."""
import asyncio
import copy
import json
from pathlib import Path
import time

import pytest

from astra_harness import cli
from astra_harness.codex_runtime import RecoveryRequired, ToolInputError
from astra_harness.openrouter_runtime import OpenRouterRuntime
from astra_harness.research import (
    Research, DEFAULT_MODELS, POLICY_VERSION, PLAN_SCHEMA, REVIEW_SCHEMA,
)

KEY = "test-private-research-api-key"
CRITERIA = [{"id": "C1", "requirement": "Compare the alternatives and support the tradeoffs", "basis": "analysis"}]


def configuration(**changes):
    return {"objective": "Compare two designs for an offline reading-notes application.",
            "models": dict(DEFAULT_MODELS), "workers": 3, "criteria": [],
            "policy_version": POLICY_VERSION, **changes}


class SimulatedProvider(OpenRouterRuntime):
    calls = []
    verdicts = ["revise", "accept"]
    active = 0
    peak = 0
    bad_accept = False
    never_submit = False
    fail = False
    skip_reads = False

    async def _http(self, agent, record, payload):
        cls = type(self)
        parent = self.state_path.parent
        if parent.name.startswith("attempt-"):
            phase = parent.parent.name
            number = int(parent.parent.parent.name)
        else:
            phase = parent.name
            number = int(parent.parent.name)
        await self._request_started(agent, record, time.time())
        cls.calls.append((number, phase, agent, copy.deepcopy(payload)))
        if cls.fail:
            raise OSError("provider failed with " + KEY)
        cls.active += 1
        cls.peak = max(cls.peak, cls.active)
        try:
            await asyncio.sleep(0.002)
        finally:
            cls.active -= 1
        prompt = json.loads(payload["messages"][1]["content"])
        context = prompt["context"]
        catalog = context["artifacts"]
        supplied = [a["id"] for a in catalog if a["origin"] == "supplied_evidence"]
        latest_work = [a["id"] for a in catalog if a["label"].startswith(f"round-{number}/work/")]
        latest_synthesis = [a["id"] for a in catalog if a["label"] == f"round-{number}/synthesize/agent-a"]
        sources = (supplied if phase == "work" else latest_work if phase == "synthesize" else latest_synthesis)[:8]
        required = list(dict.fromkeys(context["required_read_artifact_ids"] + sources))
        previous_calls = [c for m in payload["messages"] for c in m.get("tool_calls", [])]
        read_ids = {json.loads(c["function"]["arguments"])["artifact_id"] for c in previous_calls
                    if c["function"]["name"] == "read_artifact"}
        unread = [ref for ref in required if ref not in read_ids]
        results = [json.loads(m["content"]) for m in payload["messages"] if m["role"] == "tool"]
        submitted = any(r.get("accepted") for r in results)
        calls = []

        def call(name, arguments, suffix=""):
            return {"id": f"call-{len(previous_calls)}-{suffix}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}

        if unread and not cls.skip_reads:
            calls = [call("read_artifact", {"artifact_id": ref}, str(i)) for i, ref in enumerate(unread[:8])]
        elif not submitted and not cls.never_submit:
            criteria = context["criteria"] or copy.deepcopy(CRITERIA)
            if context["user_criteria"]:
                criteria = [{"id": f"C{i+1}", "requirement": value, "basis": "analysis"}
                            for i, value in enumerate(context["user_criteria"])]
            if phase == "plan":
                output = {"criteria": criteria, "tasks": [
                    {"worker": a, "task": "Compare offline storage tradeoffs and address the last QC objection."}
                    for a in prompt["assignment"]["workers"]]}
            elif phase == "work":
                output = {"summary": "The two options differ in operational complexity.",
                          "findings": [{"criterion_id": "C1", "claim": "Local files simplify export.", "evidence_ids": sources}],
                          "unresolved": []}
            elif phase == "synthesize":
                output = {"answer": "Use local files when portability outweighs structured queries.",
                          "evidence_ids": sources, "unresolved": []}
            else:
                verdict = cls.verdicts[min(number - 1, len(cls.verdicts) - 1)]
                output = {"verdict": verdict, "objective_met": verdict == "accept",
                          "summary": "Supported tradeoff comparison." if verdict == "accept" else "The portability tradeoff needs a clearer comparison.",
                          "checks": [{"criterion_id": c["id"],
                                      "status": "pass" if verdict == "accept" else "blocked" if verdict == "blocked" else "inconclusive",
                                      "reason": "Evidence supports the comparison." if verdict == "accept" else "Explain the portability tradeoff.",
                                      "evidence_ids": sources} for c in criteria],
                          "next_steps": [] if verdict == "accept" else ["Compare plain-text export with database export explicitly."]}
                if cls.bad_accept and not any(r.get("error") for r in results):
                    output.update(verdict="accept", objective_met=False)
            calls = [call("submit_result", output)]
        data = {"id": "simulated-response", "model": payload["model"], "provider": "simulated",
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "choices": [{"finish_reason": "tool_calls" if calls else "stop", "message":
                             {"role": "assistant", "content": None if calls else "Submitted.", **({"tool_calls": calls} if calls else {})}}]}
        return {"http_status": 200, "completed_at": time.time(), "body": json.dumps(data).encode()}


@pytest.fixture(autouse=True)
def isolated_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr("astra_harness.api_settings.openrouter_settings", lambda: {})
    monkeypatch.setattr("astra_harness.research.PHASE_RETRY_BASE_SECONDS", 0)
    SimulatedProvider.calls = []
    SimulatedProvider.verdicts = ["revise", "accept"]
    SimulatedProvider.active = SimulatedProvider.peak = 0
    SimulatedProvider.bad_accept = SimulatedProvider.never_submit = False
    SimulatedProvider.fail = SimulatedProvider.skip_reads = False


def execute(path, *, config=None, limits=None, evidence=(), resume=False):
    controller = Research(path, config=None if resume else config or configuration(),
                          limits=limits, evidence=evidence, runtime_class=SimulatedProvider)
    try:
        return asyncio.run(controller.run())
    finally:
        controller.close()


def test_reject_then_revise_and_accept_with_distinct_models_and_parallel_workers(tmp_path):
    result = execute(tmp_path)
    assert result["status"] == "completed", result
    assert [r["verdict"] for r in result["history"]] == ["revise", "accept"]
    assert result["accepted_by_qc"] is True
    assert SimulatedProvider.peak == 3
    for number, phase, agent, payload in SimulatedProvider.calls:
        assert payload["model"] == DEFAULT_MODELS["worker" if phase == "work" else "qc" if phase == "review" else "head"]
        assert {t["function"]["name"] for t in payload["tools"]} == {
            "read_artifact", "submit_result"}
        if number == 2:
            prompt = json.loads(payload["messages"][1]["content"])
            assert prompt["context"]["criteria"] == CRITERIA
            if phase in {"plan", "work", "review"}:
                assert prompt["assignment"]["previous_review"]["next_steps"]
    assert result["usage"]["requests_including_uncertain"] == len(SimulatedProvider.calls)
    assert result["usage"]["reported_total_tokens"] == len(SimulatedProvider.calls) * 30
    assert all(KEY not in path.read_text() for path in tmp_path.rglob("*.json"))


def test_invalid_qc_acceptance_is_rejected_and_corrected_in_turn(tmp_path):
    SimulatedProvider.bad_accept = True
    SimulatedProvider.verdicts = ["accept"]
    result = execute(tmp_path)
    assert result["status"] == "completed", result
    state = json.loads((tmp_path / "rounds/001/review/attempt-001/runtime_state.json").read_text())
    assert state["workers"]["agent-a"]["tool_input_rejections"] == 1


def test_model_must_read_cited_artifacts_before_submitting(tmp_path):
    SimulatedProvider.skip_reads = True
    result = execute(tmp_path)
    assert result["status"] == "paused"
    assert result["accepted_by_qc"] is False
    assert result["phase"] == "synthesize"
    states = [json.loads(path.read_text()) for path in
              sorted((tmp_path / "rounds/001/synthesize").glob("attempt-*/runtime_state.json"))]
    assert len(states) == 3
    assert all(state["workers"]["agent-a"]["tool_input_rejections"] == 4 for state in states)


def test_observations_cannot_be_accepted_using_model_output_only(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        controller.state["criteria"] = [{**CRITERIA[0], "basis": "observation"}]
        ref = controller.put_artifact("model_analysis", "model claim", {"claim": "I tested it"})
        review = {"verdict": "accept", "objective_met": True, "summary": "Fine", "next_steps": [],
                  "checks": [{"criterion_id": "C1", "status": "pass", "reason": "Claimed", "evidence_ids": [ref]}]}
        with pytest.raises(ToolInputError, match="Observed behavior"):
            controller._validate_result("review", review, REVIEW_SCHEMA, [ref])
    finally:
        controller.close()


@pytest.mark.parametrize("mutation", ["missing_check", "failed_check", "no_evidence", "next_steps", "objective_unmet", "unknown_evidence"])
def test_qc_gate_rejects_unsupported_completion(tmp_path, mutation):
    controller = Research(tmp_path, config=configuration())
    try:
        controller.state["criteria"] = copy.deepcopy(CRITERIA)
        ref = controller.put_artifact("model_analysis", "candidate", {"answer": "supported"})
        review = {"verdict": "accept", "objective_met": True, "summary": "Fine", "next_steps": [],
                  "checks": [{"criterion_id": "C1", "status": "pass", "reason": "Supported", "evidence_ids": [ref]}]}
        if mutation == "missing_check":
            review["checks"] = []
        elif mutation == "failed_check":
            review["checks"][0]["status"] = "fail"
        elif mutation == "no_evidence":
            review["checks"][0]["evidence_ids"] = []
        elif mutation == "next_steps":
            review["next_steps"] = ["Still need another check"]
        elif mutation == "objective_unmet":
            review["objective_met"] = False
        elif mutation == "unknown_evidence":
            review["checks"][0]["evidence_ids"] = ["a" * 64]
        with pytest.raises(ToolInputError):
            controller._validate_result("review", review, REVIEW_SCHEMA, [ref])
    finally:
        controller.close()


def test_round_budget_pauses_and_resume_continues_without_repeating_round(tmp_path):
    first = execute(tmp_path, limits={"max_rounds": 1})
    assert first["status"] == "paused", first
    count = len(SimulatedProvider.calls)
    second = execute(tmp_path, resume=True, limits={"max_rounds": 2})
    assert second["status"] == "completed", second
    assert all(number == 2 for number, _, _, _ in SimulatedProvider.calls[count:])


def test_mid_round_budget_pause_resumes_completed_first_plan(tmp_path):
    first = execute(tmp_path, limits={"max_requests": 3})
    assert first["status"] == "paused", first
    assert first["usage"]["requests_including_uncertain"] <= 3
    assert first["phase"] == "work"
    count = len(SimulatedProvider.calls)
    second = execute(tmp_path, resume=True, limits={"max_requests": 160})
    assert second["status"] == "completed", second
    assert not any(n == 1 and p == "plan" for n, p, _, _ in SimulatedProvider.calls[count:])


def test_completed_replay_makes_no_requests_and_needs_no_key(tmp_path, monkeypatch):
    first = execute(tmp_path)
    assert first["status"] == "completed"
    count = len(SimulatedProvider.calls)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    SimulatedProvider.fail = True
    second = execute(tmp_path, resume=True)
    assert second["answer"] == first["answer"]
    assert len(SimulatedProvider.calls) == count


def test_uncertain_provider_failure_uses_bounded_fresh_attempts(tmp_path):
    SimulatedProvider.fail = True
    result = execute(tmp_path)
    assert result["status"] == "paused"
    assert len(SimulatedProvider.calls) == 3
    assert KEY not in json.dumps(result)
    replay = execute(tmp_path, resume=True)
    assert replay["status"] == "paused"
    assert len(SimulatedProvider.calls) == 3


def test_plain_final_without_submission_does_not_complete_project(tmp_path):
    SimulatedProvider.never_submit = True
    result = execute(tmp_path)
    assert result["status"] == "paused"
    assert "node_no_submission" in result["reason"]
    assert not any(phase == "work" for _, phase, _, _ in SimulatedProvider.calls)


def test_qc_blocker_keeps_the_bounded_loop_autonomous(tmp_path):
    SimulatedProvider.verdicts = ["blocked"]
    result = execute(tmp_path)
    assert result["status"] == "paused"
    assert len(result["history"]) == 8
    assert all(row["verdict"] == "blocked" and row["next_steps"] for row in result["history"])


def test_supplied_evidence_is_snapshotted_and_hash_tampering_is_detected(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Observed export behavior from a dedicated test account.")
    root = tmp_path / "run"
    result = execute(root, evidence=[source])
    assert result["status"] == "completed", result
    source.write_text("Changed outside the run")
    replay = execute(root, resume=True)
    assert replay["status"] == "completed"
    state = json.loads((root / "research.json").read_text())
    ref = state["evidence_ids"][0]
    (root / "artifacts" / (ref + ".json")).write_text('{"origin":"tampered"}')
    with pytest.raises(ValueError, match="integrity"):
        execute(root, resume=True)


def test_fixed_criteria_cannot_be_weakened(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        controller.state["criteria"] = copy.deepcopy(CRITERIA)
        result = {"criteria": [{**CRITERIA[0], "requirement": "Just produce any answer"}],
                  "tasks": [{"worker": a, "task": "Do something"} for a in controller.workers]}
        with pytest.raises(ToolInputError, match="fixed"):
            controller._validate_result("plan", result, PLAN_SCHEMA, [])
    finally:
        controller.close()


def test_run_lock_prevents_duplicate_execution(tmp_path):
    first = Research(tmp_path, config=configuration())
    try:
        with pytest.raises(ValueError, match="already active"):
            Research(tmp_path)
    finally:
        first.close()


def test_single_deepseek_worker_supported_without_weakening_old_runtime_limits(tmp_path):
    SimulatedProvider.verdicts = ["accept"]
    result = execute(tmp_path, config=configuration(workers=1))
    assert result["status"] == "completed", result
    with pytest.raises(ValueError):
        OpenRouterRuntime(tmp_path / "unused", tmp_path, None, None,
                          model=DEFAULT_MODELS["head"], agents=("agent-a",))


def test_cli_start_prompt_file_and_dry_run_without_provider_calls(tmp_path, monkeypatch, capsys):
    source = tmp_path / "prompt.txt"
    source.write_text("Compare designs; keep `literal` and $(literal) text.")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    root = tmp_path / "run"
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["research", "--prompt-file", str(source), "--run-dir", str(root), "--dry-run"])
    assert exit_info.value.code == 0
    state = json.loads((root / "research.json").read_text())
    assert state["config"]["objective"] == source.read_text()
    assert state["config"]["models"] == DEFAULT_MODELS
    assert state["status"] == "ready"
    assert not list(root.rglob("runtime_state.json"))
    with pytest.raises(SystemExit) as status_exit:
        cli.main(["status", "--run-dir", str(root)])
    assert status_exit.value.code == 0
    assert '"ready"' in capsys.readouterr().out


def test_interactive_start_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda label: "Compare two storage designs")
    with pytest.raises(SystemExit) as exited:
        cli.main(["research", "--dry-run", "--run-dir", str(tmp_path)])
    assert exited.value.code == 0
    assert json.loads((tmp_path / "research.json").read_text())["config"]["objective"] == "Compare two storage designs"


@pytest.mark.parametrize("arguments", [
    ["--resume"], ["--resume", "changed prompt"], ["prompt", "--prompt-file", "other"],
    ["prompt", "--max-rounds", "0"], ["prompt", "--max-requests", "-1"],
])
def test_cli_invalid_inputs_never_launch_models(tmp_path, arguments):
    with pytest.raises(SystemExit) as exited:
        cli.main(["research", *arguments, "--run-dir", str(tmp_path), "--dry-run"])
    assert exited.value.code == 1
    assert SimulatedProvider.calls == []
