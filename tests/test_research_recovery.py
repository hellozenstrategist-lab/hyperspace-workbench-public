"""Recovery attempts and host QC invariants for the research loop.

The fake provider exercises the real OpenRouter runtime without network access.
Phase retries are separate transcripts; a fake timeout never asks the runtime to
replay its ambiguous request.
"""
import asyncio
import copy
import json
import time

import pytest

from astra_harness.codex_runtime import RuntimeFailure, ToolInputError
from astra_harness.openrouter_runtime import OpenRouterRuntime
from astra_harness import research as research_module
from astra_harness.research import (
    DEFAULT_MODELS,
    BudgetPause,
    POLICY_VERSION,
    PLAN_SCHEMA,
    REVIEW_SCHEMA,
    Research,
    runtime_diagnostics,
)


KEY = "test-private-recovery-api-key"
CRITERIA = [{
    "id": "C1",
    "requirement": "Support the recommendation with inspected evidence",
    "basis": "analysis",
}]


def configuration(**changes):
    return {
        "objective": "Compare two bounded designs and preserve unresolved gaps.",
        "models": dict(DEFAULT_MODELS),
        "workers": 1,
        "criteria": [],
        "policy_version": POLICY_VERSION,
        **changes,
    }


def _location(state_path):
    parent = state_path.parent
    if parent.name.startswith("attempt-"):
        return int(parent.parent.parent.name), parent.parent.name, parent.name
    return int(parent.parent.name), parent.name, "legacy"


def _tool_call(name, arguments, ordinal):
    return {
        "id": f"call-{ordinal}-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class RecoveryProvider(OpenRouterRuntime):
    """Scriptable provider for phase-attempt behavior."""

    calls = []
    failures = {}
    fired_failures = set()
    ack_timeouts = set()
    fired_ack_timeouts = set()
    incomplete_finals = set()
    fired_incomplete_finals = set()
    persistent_incomplete_finals = set()
    verdicts = ["accept"]

    async def _http(self, agent, record, payload):
        cls = type(self)
        number, phase, attempt = _location(self.state_path)
        await self._request_started(agent, record, time.time())
        call = {
            "round": number,
            "phase": phase,
            "attempt": attempt,
            "agent": agent,
            "request_id": record["request_id"],
            "payload": copy.deepcopy(payload),
        }
        cls.calls.append(call)

        incomplete_key = (phase, attempt)
        if (incomplete_key in cls.persistent_incomplete_finals
                or (incomplete_key in cls.incomplete_finals
                    and incomplete_key not in cls.fired_incomplete_finals)):
            cls.fired_incomplete_finals.add(incomplete_key)
            body = {
                "id": f"simulated-incomplete-{len(cls.calls)}",
                "model": payload["model"],
                "provider": "simulated",
                "usage": {"prompt_tokens": 20, "completion_tokens": 0, "total_tokens": 20},
                "choices": [{
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": ""},
                }],
            }
            return {
                "http_status": 200,
                "completed_at": time.time(),
                "body": json.dumps(body).encode(),
            }

        failure_key = (phase, attempt)
        failure = cls.failures.get(failure_key)
        if failure is not None and failure_key not in cls.fired_failures:
            cls.fired_failures.add(failure_key)
            if failure == "timeout":
                raise TimeoutError("simulated provider timeout containing " + KEY)
            if failure == "429":
                return {"http_status": 429, "retry_after": "0",
                        "completed_at": time.time()}
            raise AssertionError(f"unknown simulated failure: {failure}")

        tool_results = []
        for message in payload["messages"]:
            if message.get("role") != "tool":
                continue
            try:
                value = json.loads(message["content"])
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                tool_results.append(value)
        ack_key = (phase, attempt)
        if (phase in cls.ack_timeouts and any(row.get("accepted") for row in tool_results)
                and ack_key not in cls.fired_ack_timeouts):
            cls.fired_ack_timeouts.add(ack_key)
            raise TimeoutError("simulated timeout after durable submission containing " + KEY)

        prompt = json.loads(payload["messages"][1]["content"])
        context = prompt["context"]
        catalog = context["artifacts"]
        current_work = [
            item["id"] for item in catalog
            if item["label"].startswith(f"round-{number}/work/")
        ]
        current_synthesis = [
            item["id"] for item in catalog
            if item["label"] == f"round-{number}/synthesize/agent-a"
        ]
        cited = current_work if phase == "synthesize" else current_synthesis if phase == "review" else []
        required = list(dict.fromkeys(context["required_read_artifact_ids"] + cited))

        previous_calls = [
            tool_call
            for message in payload["messages"]
            for tool_call in message.get("tool_calls", [])
        ]
        read_ids = {
            json.loads(tool_call["function"]["arguments"])["artifact_id"]
            for tool_call in previous_calls
            if tool_call["function"]["name"] == "read_artifact"
        }
        unread = [artifact_id for artifact_id in required if artifact_id not in read_ids]
        submitted = any(row.get("accepted") for row in tool_results)

        calls = []
        if unread:
            calls = [
                _tool_call("read_artifact", {"artifact_id": artifact_id}, len(previous_calls) + index)
                for index, artifact_id in enumerate(unread[:8])
            ]
        elif not submitted:
            criteria = context["criteria"] or copy.deepcopy(CRITERIA)
            if phase == "plan":
                output = {
                    "criteria": criteria,
                    "tasks": [{
                        "worker": worker,
                        "task": "Compare the alternatives and address the latest QC objection.",
                    } for worker in prompt["assignment"]["workers"]],
                }
            elif phase == "work":
                output = {
                    "summary": "The bounded alternatives were compared.",
                    "findings": [{
                        "criterion_id": "C1",
                        "claim": "The simpler design has fewer moving parts.",
                        "evidence_ids": [],
                    }],
                    "unresolved": [],
                }
            elif phase == "synthesize":
                output = {
                    "answer": "Prefer the simpler design when both meet the requirement.",
                    "evidence_ids": cited,
                    "unresolved": [],
                }
            else:
                verdict = cls.verdicts[min(number - 1, len(cls.verdicts) - 1)]
                output = {
                    "verdict": verdict,
                    "objective_met": verdict == "accept",
                    "summary": "The evidence is sufficient." if verdict == "accept"
                               else "A concrete comparison remains.",
                    "checks": [{
                        "criterion_id": criterion["id"],
                        "status": "pass" if verdict == "accept"
                                  else "blocked" if verdict == "blocked"
                                  else "inconclusive",
                        "reason": "The current synthesis addresses the criterion."
                                  if verdict == "accept" else "Compare the alternatives directly.",
                        "evidence_ids": cited,
                    } for criterion in criteria],
                    "next_steps": [] if verdict == "accept"
                                  else ["Run a distinct comparison in the next round."],
                }
            calls = [_tool_call("submit_result", output, len(previous_calls))]

        body = {
            "id": f"simulated-{len(cls.calls)}",
            "model": payload["model"],
            "provider": "simulated",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "choices": [{
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": None if calls else "Submitted.",
                    **({"tool_calls": calls} if calls else {}),
                },
            }],
        }
        return {"http_status": 200, "completed_at": time.time(),
                "body": json.dumps(body).encode()}


@pytest.fixture(autouse=True)
def isolated_recovery_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setattr("astra_harness.api_settings.openrouter_settings", lambda: {})
    monkeypatch.setattr(research_module, "PHASE_RETRY_BASE_SECONDS", 0)
    RecoveryProvider.calls = []
    RecoveryProvider.failures = {}
    RecoveryProvider.fired_failures = set()
    RecoveryProvider.ack_timeouts = set()
    RecoveryProvider.fired_ack_timeouts = set()
    RecoveryProvider.incomplete_finals = set()
    RecoveryProvider.fired_incomplete_finals = set()
    RecoveryProvider.persistent_incomplete_finals = set()
    RecoveryProvider.verdicts = ["accept"]


def execute(path, *, limits=None):
    controller = Research(path, config=configuration(), limits=limits,
                          runtime_class=RecoveryProvider)
    try:
        return asyncio.run(controller.run())
    finally:
        controller.close()


def test_uncertain_synthesis_failure_uses_fresh_attempt_and_reaches_qc(tmp_path):
    RecoveryProvider.failures = {("synthesize", "attempt-001"): "timeout"}

    result = execute(tmp_path, limits={"max_rounds": 1})

    assert result["status"] == "completed", result
    assert result["accepted_by_qc"] is True
    attempts = sorted((tmp_path / "rounds/001/synthesize").glob("attempt-*"))
    assert [path.name for path in attempts] == ["attempt-001", "attempt-002"]
    first = json.loads((attempts[0] / "runtime_state.json").read_text())
    second = json.loads((attempts[1] / "runtime_state.json").read_text())
    assert first["workers"]["agent-a"]["requests"][-1]["status"] == "uncertain"
    assert second["workers"]["agent-a"]["status"] == "completed"
    assert first["workers"]["agent-a"]["requests"][-1]["request_id"] != \
           second["workers"]["agent-a"]["requests"][0]["request_id"]
    assert any(call["phase"] == "review" for call in RecoveryProvider.calls)
    assert result["usage"]["requests_including_uncertain"] == len(RecoveryProvider.calls)
    assert result["usage"]["unresolved_requests"] == 1
    assert result["usage"]["usage_complete"] is False


def test_research_429_retries_in_same_attempt_and_counts_physical_requests(tmp_path):
    RecoveryProvider.failures = {("synthesize", "attempt-001"): "429"}

    result = execute(tmp_path, limits={"max_rounds": 1})

    assert result["status"] == "completed", result
    assert result["accepted_by_qc"] is True
    attempts = sorted((tmp_path / "rounds/001/synthesize").glob("attempt-*"))
    assert [path.name for path in attempts] == ["attempt-001"]
    runtime = json.loads((attempts[0] / "runtime_state.json").read_text())
    worker = runtime["workers"]["agent-a"]
    assert runtime["limits"]["max_rate_limit_retries"] == 2
    assert [row["status"] for row in worker["requests"]] == [
        "rate_limited", "completed", "completed",
    ]
    assert [row["retry"] for row in worker["requests"]] == [0, 1, 0]
    assert len({row["request_id"] for row in worker["requests"]}) == 3
    assert result["usage"]["requests_including_uncertain"] == \
           len(RecoveryProvider.calls)
    assert result["usage"]["unresolved_requests"] == 0
    assert result["usage"]["usage_complete"] is True
    # Every runtime created after this change carries the same bounded retry
    # policy; old fixtures below remain pinned to zero.
    runtime_paths = list((tmp_path / "rounds").rglob("runtime_state.json"))
    assert runtime_paths
    assert all(json.loads(path.read_text())["limits"]["max_rate_limit_retries"] == 2
               for path in runtime_paths)


def test_truncated_empty_model_final_corrects_within_the_same_attempt(tmp_path):
    RecoveryProvider.incomplete_finals = {("plan", "attempt-001")}

    result = execute(tmp_path, limits={"max_rounds": 1})

    assert result["status"] == "completed", result
    assert result["accepted_by_qc"] is True
    attempts = sorted((tmp_path / "rounds/001/plan").glob("attempt-*"))
    assert [path.name for path in attempts] == ["attempt-001"]
    metadata = json.loads((attempts[0] / "attempt.json").read_text())
    inputs = json.loads((tmp_path / "rounds/001/plan/inputs.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["recovery_of"] is None
    assert metadata["input_hash"] == inputs["input_hash"]
    assert isinstance(metadata["prompt_hash"], str) and len(metadata["prompt_hash"]) == 64

    runtime = json.loads((attempts[0] / "runtime_state.json").read_text())
    worker = runtime["workers"]["agent-a"]
    assert len(worker["requests"]) == 2
    assert worker["requests"][0]["finish_reason"] == "length"
    correction = worker["incomplete_final_correction"]
    assert correction["count"] == 1
    assert correction["mode"] == "completion_closure"
    assert correction["status"] == "resolved"
    assert (attempts[0] / "submissions.json").exists()


@pytest.mark.parametrize("legacy_code", ["operational", "operational_failure"])
def test_legacy_incomplete_final_starts_fresh_attempt_without_rewriting_first(
        tmp_path, legacy_code):
    RecoveryProvider.persistent_incomplete_finals = {("plan", "attempt-001")}
    controller = Research(
        tmp_path,
        config=configuration(),
        limits={"max_rounds": 1, "max_phase_attempts": 1},
        runtime_class=RecoveryProvider,
    )
    controller.deadline = time.monotonic() + 30
    prompts = {
        "agent-a": {
            "role": "head",
            "workers": ["agent-a"],
            "task": "Create the fixed criteria and assign the worker.",
        },
    }
    try:
        with pytest.raises((BudgetPause, RuntimeFailure)):
            asyncio.run(controller._stage(
                "plan", DEFAULT_MODELS["head"], prompts, PLAN_SCHEMA, [],
            ))

        first = tmp_path / "rounds/001/plan/attempt-001"
        metadata_path = first / "attempt.json"
        legacy = json.loads(metadata_path.read_text())
        legacy.update(
            status="failed",
            failure_code=legacy_code,
            reason="OpenRouter worker did not return a complete final answer",
        )
        metadata_path.write_text(json.dumps(legacy, sort_keys=True))
        before = {
            path.relative_to(first): path.read_bytes()
            for path in first.rglob("*") if path.is_file()
        }
        assert controller._attempt_failure_code(first) == "node_no_submission"

        controller.limits["max_phase_attempts"] = 2
        outputs, artifacts = asyncio.run(controller._stage(
            "plan", DEFAULT_MODELS["head"], prompts, PLAN_SCHEMA, [],
        ))

        after = {
            path.relative_to(first): path.read_bytes()
            for path in first.rglob("*") if path.is_file()
        }
        assert set(outputs) == {"agent-a"}
        assert set(artifacts) == {"agent-a"}
        assert before == after
        second_metadata = json.loads(
            (tmp_path / "rounds/001/plan/attempt-002/attempt.json").read_text()
        )
        assert second_metadata["status"] == "completed"
        assert second_metadata["recovery_of"] == "attempt-001"
        assert json.loads(metadata_path.read_text())["failure_code"] == legacy_code
    finally:
        controller.close()


@pytest.mark.parametrize("legacy_code", ["operational", "operational_failure"])
def test_legacy_incomplete_final_mapping_requires_the_exact_controlled_reason(
        tmp_path, legacy_code):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt = tmp_path / "rounds/001/plan/attempt-001"
        attempt.mkdir(parents=True)
        (attempt / "attempt.json").write_text(json.dumps({
            "failure_code": legacy_code,
            "reason": "OpenRouter worker did not return a complete final answer (provider detail)",
        }))

        assert controller._attempt_failure_code(attempt) == legacy_code
    finally:
        controller.close()


def test_fresh_plan_attempt_migrates_4096_to_8192_without_changing_phase_hashes(
        tmp_path):
    RecoveryProvider.persistent_incomplete_finals = {("plan", "attempt-001")}
    controller = Research(
        tmp_path,
        config=configuration(),
        limits={"max_rounds": 1, "max_phase_attempts": 1},
        runtime_class=RecoveryProvider,
    )
    controller.deadline = time.monotonic() + 30
    prompts = {
        "agent-a": {
            "role": "head",
            "workers": ["agent-a"],
            "task": "Create the fixed criteria and assign the worker.",
        },
    }
    try:
        with pytest.raises(BudgetPause):
            asyncio.run(controller._stage(
                "plan", DEFAULT_MODELS["head"], prompts, PLAN_SCHEMA, [],
            ))
        first = tmp_path / "rounds/001/plan/attempt-001"
        first_meta = json.loads((first / "attempt.json").read_text())
        inputs = json.loads((tmp_path / "rounds/001/plan/inputs.json").read_text())
        assert first_meta["failure_code"] == "node_no_submission"
        assert first_meta["input_hash"] == inputs["input_hash"]

        # Represent an attempt written before the phase output allowance was
        # raised. Fresh-attempt recovery reads its classification and ledger but
        # never reopens or mutates that runtime.
        first_runtime_path = first / "runtime_state.json"
        first_runtime = json.loads(first_runtime_path.read_text())
        first_runtime["limits"]["max_tokens"] = 4096
        first_runtime_path.write_text(json.dumps(first_runtime, sort_keys=True))
        RecoveryProvider.calls = []

        controller.limits["max_phase_attempts"] = 2
        outputs, artifacts = asyncio.run(controller._stage(
            "plan", DEFAULT_MODELS["head"], prompts, PLAN_SCHEMA, [],
        ))
        second = tmp_path / "rounds/001/plan/attempt-002"
        second_meta = json.loads((second / "attempt.json").read_text())
        second_runtime = json.loads((second / "runtime_state.json").read_text())

        assert set(outputs) == {"agent-a"}
        assert set(artifacts) == {"agent-a"}
        assert second_meta["input_hash"] == first_meta["input_hash"]
        assert second_meta["prompt_hash"] == first_meta["prompt_hash"]
        assert second_meta["recovery_of"] == "attempt-001"
        assert second_runtime["limits"]["max_tokens"] == 8192
        assert RecoveryProvider.calls
        assert all(call["attempt"] == "attempt-002" for call in RecoveryProvider.calls)
        assert all(call["payload"]["max_tokens"] == 8192
                   for call in RecoveryProvider.calls)
    finally:
        controller.close()


def _saved_completion_reserve_tool_failure(*, tool="browser_navigate", arguments="{}",
                                           completion_reserve=True):
    return {
        "fatal": "Only registered bounded host tools are allowed",
        "fatal_kind": "operational",
        "failure_code": "operational_failure",
        "failure_agent": "agent-b",
        "limits": {"max_requests_per_worker": 16, "max_rate_limit_retries": 0},
        "completion_policy": {
            "completion_tool": "submit_result",
            "read_tool": "read_artifact",
            "reserve_requests": 4,
            "force_requests": 2,
        },
        "tool_specs": [{
            "type": "function",
            "function": {"name": name, "description": "bounded", "parameters": {}},
        } for name in ("read_artifact", "submit_result", "browser_navigate")],
        "workers": {
            "agent-b": {
                "status": "failed",
                "fatal_kind": "operational",
                "failure_code": "operational_failure",
                **({"completion_reserve": {"requests_remaining": 4}}
                   if completion_reserve else {}),
                "messages": [{
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "terminal-call",
                        "type": "function",
                        "function": {"name": tool, "arguments": arguments},
                    }],
                }],
                "requests": [{"status": "completed"} for _ in range(13)],
                "tools": {},
            },
        },
    }


def _saved_legacy_oversized_synthesis_attempt(path):
    attempt = path / "rounds/001/synthesize/attempt-001"
    attempt.mkdir(parents=True)
    required = [f"{index:064x}" for index in range(13)]
    read_ids = required[:3] + [f"{100 + index:064x}" for index in range(2)]
    calls = [
        _tool_call("read_artifact", {"artifact_id": artifact_id}, index)
        for index, artifact_id in enumerate(read_ids)
    ]
    tools = {
        call["id"]: {
            "digest": f"digest-{index}",
            "status": "completed",
            "tool": "read_artifact",
            "result": {"artifact_id": read_ids[index]},
        }
        for index, call in enumerate(calls)
    }
    state = {
        "fatal": "OpenRouter tool call count exceeded its bound",
        "fatal_kind": "operational",
        "failure_code": "operational_failure",
        "failure_agent": "agent-a",
        "manifest": {
            "tool_validation_policy_version": "openrouter_tool_validation_v3",
        },
        "limits": {
            "max_requests_per_worker": 16,
            "max_rate_limit_retries": 0,
        },
        "completion_policy": {
            "completion_tool": "submit_result",
            "read_tool": "read_artifact",
            "reserve_requests": 4,
            "force_requests": 2,
        },
        "tool_specs": [{
            "type": "function",
            "function": {"name": name, "description": "bounded", "parameters": {}},
        } for name in ("read_artifact", "submit_result")],
        "workers": {
            "agent-a": {
                "status": "failed",
                "fatal_kind": "operational",
                "failure_code": "operational_failure",
                "messages": [
                    {"role": "system", "content": "bounded"},
                    {"role": "user", "content": json.dumps({
                        "context": {"required_read_artifact_ids": required},
                    })},
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    *[{"role": "tool", "tool_call_id": call["id"],
                       "content": json.dumps({"accepted": True})}
                      for call in calls],
                ],
                "requests": [{
                    "status": "completed", "http_status": 200,
                    "response_id": f"response-{index}",
                } for index in range(2)],
                "tools": tools,
            },
        },
    }
    (attempt / "attempt.json").write_text(json.dumps({
        "failure_code": "operational",
        "reason": "OpenRouter tool call count exceeded its bound",
    }))
    (attempt / "runtime_state.json").write_text(json.dumps(state))
    (attempt / "reads.json").write_text(json.dumps({"agent-a": read_ids}))
    return attempt, state


def test_saved_legacy_synthesis_overflow_is_narrowly_retryable(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_legacy_oversized_synthesis_attempt(tmp_path)

        assert research_module._legacy_oversized_synthesis_read_batch_failure(
            state, attempt)
        assert controller._attempt_failure_code(attempt) == "tool_call_batch_exceeded"
        assert "tool_call_batch_exceeded" in research_module.RETRYABLE_ATTEMPT_FAILURES
    finally:
        controller.close()


@pytest.mark.parametrize("change", [
    "browser_tool", "unresolved_request", "effectful_record", "persisted_final",
    "submission", "small_unread_set",
])
def test_saved_legacy_synthesis_overflow_requires_complete_non_effectful_proof(
        tmp_path, change):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_legacy_oversized_synthesis_attempt(tmp_path)
        worker = state["workers"]["agent-a"]
        if change == "browser_tool":
            state["tool_specs"].append({
                "type": "function",
                "function": {"name": "browser_click", "description": "bounded",
                             "parameters": {}},
            })
        elif change == "unresolved_request":
            worker["requests"][-1]["status"] = "uncertain"
        elif change == "effectful_record":
            worker["tools"][next(iter(worker["tools"]))]["tool"] = "submit_result"
        elif change == "persisted_final":
            worker["messages"].append({
                "role": "assistant", "content": None,
                "tool_calls": [_tool_call("read_artifact", {
                    "artifact_id": f"{999:064x}",
                }, 99)],
            })
        elif change == "submission":
            (attempt / "submissions.json").write_text("{}")
        elif change == "small_unread_set":
            prompt = json.loads(worker["messages"][1]["content"])
            prompt["context"]["required_read_artifact_ids"] = \
                prompt["context"]["required_read_artifact_ids"][:8]
            worker["messages"][1]["content"] = json.dumps(prompt)
        (attempt / "runtime_state.json").write_text(json.dumps(state))

        assert not research_module._legacy_oversized_synthesis_read_batch_failure(
            state, attempt)
        assert controller._attempt_failure_code(attempt) == "operational"
    finally:
        controller.close()


def _saved_v4_capped_malformed_submission(path):
    attempt, state = _saved_legacy_oversized_synthesis_attempt(path)
    state["fatal"] = "Only registered bounded host tools are allowed"
    state["manifest"]["tool_validation_policy_version"] = \
        "openrouter_tool_validation_v4"
    state["limits"].update(max_tokens=4096, max_tool_calls_per_response=16)
    worker = state["workers"]["agent-a"]
    worker["requests"][-1]["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 4096,
        "total_tokens": 4196,
    }
    worker["messages"].append({
        "role": "assistant",
        "content": "Submitting the bounded synthesis.",
        "tool_calls": [{
            "id": "truncated-submit",
            "type": "function",
            "function": {
                "name": "submit_result",
                "arguments": '{"answer":"supported result","evidence_ids": ',
            },
        }],
    })
    (attempt / "attempt.json").write_text(json.dumps({
        "failure_code": "operational",
        "reason": "Only registered bounded host tools are allowed",
    }))
    (attempt / "runtime_state.json").write_text(json.dumps(state))
    return attempt, state


def test_saved_v4_capped_malformed_submission_is_narrowly_retryable(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_v4_capped_malformed_submission(tmp_path)

        assert research_module._legacy_unhandled_malformed_submission_failure(
            state, attempt)
        assert controller._attempt_failure_code(attempt) == "malformed_tool_arguments"
        assert "malformed_tool_arguments" in research_module.RETRYABLE_ATTEMPT_FAILURES
    finally:
        controller.close()


@pytest.mark.parametrize("change", [
    "below_cap", "valid_json", "unknown_tool", "non_string", "tool_record",
    "reserve_active", "unresolved_request",
])
def test_saved_v4_malformed_submission_mapping_preserves_ambiguous_failures(
        tmp_path, change):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_v4_capped_malformed_submission(tmp_path)
        worker = state["workers"]["agent-a"]
        terminal = worker["messages"][-1]["tool_calls"][0]
        if change == "below_cap":
            worker["requests"][-1]["usage"]["completion_tokens"] = 4095
        elif change == "valid_json":
            terminal["function"]["arguments"] = "{}"
        elif change == "unknown_tool":
            terminal["function"]["name"] = "unknown_tool"
        elif change == "non_string":
            terminal["function"]["arguments"] = {}
        elif change == "tool_record":
            worker["tools"][terminal["id"]] = {
                "digest": "unknown", "tool": "submit_result", "status": "started",
            }
        elif change == "reserve_active":
            worker["completion_reserve"] = {"requests_remaining": 4}
        elif change == "unresolved_request":
            worker["requests"][-1]["status"] = "uncertain"
        (attempt / "runtime_state.json").write_text(json.dumps(state))

        assert not research_module._legacy_unhandled_malformed_submission_failure(
            state, attempt)
        assert controller._attempt_failure_code(attempt) == "operational"
    finally:
        controller.close()


def _saved_legacy_resolved_provider_error(path, *, attempt_code="operational"):
    attempt = path / "rounds/001/plan/attempt-008"
    attempt.mkdir(parents=True)
    read_ids = [f"{500 + index:064x}" for index in range(10)]
    call_batches = [
        [_tool_call("read_artifact", {"artifact_id": artifact_id}, index)
         for index, artifact_id in enumerate(read_ids[:5])],
        [_tool_call("read_artifact", {"artifact_id": artifact_id}, index + 5)
         for index, artifact_id in enumerate(read_ids[5:])],
    ]
    messages = [
        {"role": "system", "content": "bounded"},
        {"role": "user", "content": "plan from captured artifacts"},
    ]
    tools = {}
    for calls in call_batches:
        messages.append({
            "role": "assistant", "content": "reading",
            "tool_calls": calls, "reasoning": "bounded",
            "reasoning_details": [{"type": "reasoning.text"}],
        })
        for call in calls:
            messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps({"artifact_id": "bounded"}),
            })
            tools[call["id"]] = {
                "digest": "bounded", "status": "completed",
                "tool": "read_artifact", "result": {"bounded": True},
            }
    state = {
        "fatal": "OpenRouter returned an invalid assistant message",
        "fatal_kind": "operational",
        "failure_code": "operational_failure",
        "failure_agent": "agent-a",
        "model": DEFAULT_MODELS["head"],
        "manifest": {
            "tool_validation_policy_version": "openrouter_tool_validation_v5",
            "completion_policy_version": "terminal_reserve_v3",
        },
        "limits": {
            "max_tokens": 8192,
            "max_requests_per_worker": 8,
            "request_timeout": 180,
            "max_rate_limit_retries": 0,
            "max_tool_calls_per_response": 16,
        },
        "completion_policy": {
            "completion_tool": "submit_result",
            "read_tool": "read_artifact",
            "reserve_requests": 4,
            "force_requests": 2,
            "max_incomplete_final_corrections": 7,
        },
        "tool_specs": [{
            "type": "function",
            "function": {"name": name, "description": "bounded", "parameters": {}},
        } for name in ("read_artifact", "submit_result")],
        "workers": {
            "agent-a": {
                "status": "failed",
                "fatal_kind": "operational",
                "failure_code": "operational_failure",
                "messages": messages,
                "requests": [{
                    "status": "completed", "http_status": 200,
                    "response_id": f"response-{index}",
                    "finish_reason": "tool_calls" if index < 2 else "error",
                    "usage": {
                        "prompt_tokens": 100 + index,
                        "completion_tokens": 10 + index,
                        "total_tokens": 110 + 2 * index,
                    },
                } for index in range(3)],
                "tools": tools,
            },
        },
    }
    (attempt / "attempt.json").write_text(json.dumps({
        "attempt": 8,
        "status": "failed",
        "failure_code": attempt_code,
        "reason": "OpenRouter returned an invalid assistant message",
    }, sort_keys=True))
    (attempt / "runtime_state.json").write_text(json.dumps(state, sort_keys=True))
    (attempt / "reads.json").write_text(json.dumps({"agent-a": read_ids}, sort_keys=True))
    return attempt, state


@pytest.mark.parametrize("legacy_code", ["operational", "operational_failure"])
def test_saved_resolved_provider_error_is_narrowly_retryable(tmp_path, legacy_code):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_legacy_resolved_provider_error(
            tmp_path, attempt_code=legacy_code)

        assert research_module._legacy_resolved_provider_completion_error(
            state, attempt)
        assert controller._attempt_failure_code(attempt) == \
               research_module.FAILURE_PROVIDER_COMPLETION_ERROR
        assert research_module.FAILURE_PROVIDER_COMPLETION_ERROR in \
               research_module.RETRYABLE_ATTEMPT_FAILURES
    finally:
        controller.close()


@pytest.mark.parametrize("change", [
    "unresolved_request", "transport_unsettled", "persisted_final",
    "submission", "started_tool", "unmatched_tool", "effectful_tool",
    "wrong_finish", "completion_accepted", "reads_mismatch", "wrong_fatal",
    "wrong_phase", "extra_tool", "modern_version", "completion_error_marker",
    "missing_response_id", "non_200", "malformed_function",
])
def test_saved_provider_error_mapping_preserves_ambiguous_or_effectful_failures(
        tmp_path, change):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt, state = _saved_legacy_resolved_provider_error(tmp_path)
        worker = state["workers"]["agent-a"]
        if change == "unresolved_request":
            worker["requests"][-1]["status"] = "uncertain"
        elif change == "transport_unsettled":
            worker["requests"][-1]["error_code"] = "transport_unsettled"
        elif change == "persisted_final":
            worker["messages"].append({"role": "assistant", "content": "unexpected"})
        elif change == "submission":
            (attempt / "submissions.json").write_text("{}")
        elif change == "started_tool":
            next(iter(worker["tools"].values()))["status"] = "started"
        elif change == "unmatched_tool":
            worker["tools"]["unmatched"] = {
                "tool": "read_artifact", "status": "completed",
            }
        elif change == "effectful_tool":
            next(iter(worker["tools"].values()))["tool"] = "submit_result"
        elif change == "wrong_finish":
            worker["requests"][-1]["finish_reason"] = "stop"
        elif change == "completion_accepted":
            worker["completion_accepted"] = {"call_id": "unknown"}
        elif change == "reads_mismatch":
            (attempt / "reads.json").write_text(json.dumps({"agent-a": []}))
        elif change == "wrong_fatal":
            state["fatal"] = "different failure"
        elif change == "wrong_phase":
            moved = attempt.parent.parent / "work" / attempt.name
            moved.parent.mkdir(parents=True)
            attempt.rename(moved)
            attempt = moved
        elif change == "extra_tool":
            state["tool_specs"].append({
                "type": "function",
                "function": {"name": "browser_inspect", "parameters": {}},
            })
        elif change == "modern_version":
            state["manifest"]["completion_policy_version"] = "terminal_reserve_v4"
        elif change == "completion_error_marker":
            worker["requests"][-1]["completion_error_code"] = \
                research_module.FAILURE_PROVIDER_COMPLETION_ERROR
        elif change == "missing_response_id":
            worker["requests"][-1].pop("response_id")
        elif change == "non_200":
            worker["requests"][-1]["http_status"] = 500
        elif change == "malformed_function":
            next(message for message in worker["messages"]
                 if message.get("role") == "assistant")["tool_calls"][0]["function"] = None
        (attempt / "runtime_state.json").write_text(json.dumps(state, sort_keys=True))

        assert not research_module._legacy_resolved_provider_completion_error(
            state, attempt)
        expected = ("transport_unsettled" if change == "transport_unsettled"
                    else "operational")
        assert controller._attempt_failure_code(attempt) == expected
    finally:
        controller.close()


def test_saved_provider_error_starts_attempt_nine_without_rewriting_attempt_eight(
        tmp_path, monkeypatch):
    controller = Research(
        tmp_path, config=configuration(),
        limits={"max_rounds": 1, "max_phase_attempts": 9},
        runtime_class=RecoveryProvider,
    )
    controller.deadline = time.monotonic() + 30
    monkeypatch.setattr(research_module, "PHASE_RETRY_BASE_SECONDS", 0)
    prompts = {
        "agent-a": {
            "role": "head",
            "workers": ["agent-a"],
            "task": "Create the fixed criteria and assign the worker.",
        },
    }
    try:
        first, _ = _saved_legacy_resolved_provider_error(tmp_path)
        before = {
            item.relative_to(first): item.read_bytes()
            for item in first.rglob("*") if item.is_file()
        }

        outputs, artifacts = asyncio.run(controller._stage(
            "plan", DEFAULT_MODELS["head"], prompts, PLAN_SCHEMA, [],
        ))

        after = {
            item.relative_to(first): item.read_bytes()
            for item in first.rglob("*") if item.is_file()
        }
        ninth = tmp_path / "rounds/001/plan/attempt-009"
        ninth_meta = json.loads((ninth / "attempt.json").read_text())
        assert set(outputs) == {"agent-a"}
        assert set(artifacts) == {"agent-a"}
        assert before == after
        assert ninth_meta["status"] == "completed"
        assert ninth_meta["recovery_of"] == "attempt-008"
        assert all(call["attempt"] == "attempt-009"
                   for call in RecoveryProvider.calls)
    finally:
        controller.close()


@pytest.mark.parametrize("legacy_code", ["operational", "operational_failure"])
def test_saved_completion_reserve_closed_tool_failure_is_narrowly_retryable(
        tmp_path, legacy_code):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt = tmp_path / "rounds/001/work/attempt-004"
        attempt.mkdir(parents=True)
        (attempt / "attempt.json").write_text(json.dumps({
            "failure_code": legacy_code,
            "reason": "Only registered bounded host tools are allowed",
        }))
        state = _saved_completion_reserve_tool_failure()
        (attempt / "runtime_state.json").write_text(json.dumps(state))

        assert research_module._runtime_failure_code(state) == \
               "completion_reserve_tool_disabled"
        assert controller._attempt_failure_code(attempt) == \
               "completion_reserve_tool_disabled"
        assert "completion_reserve_tool_disabled" in \
               research_module.RETRYABLE_ATTEMPT_FAILURES
    finally:
        controller.close()


@pytest.mark.parametrize("change", ["unregistered", "malformed", "outside_reserve"])
def test_saved_generic_tool_policy_failure_is_not_reclassified_without_transcript_proof(
        tmp_path, change):
    controller = Research(tmp_path, config=configuration())
    try:
        attempt = tmp_path / "rounds/001/work/attempt-004"
        attempt.mkdir(parents=True)
        (attempt / "attempt.json").write_text(json.dumps({
            "failure_code": "operational",
            "reason": "Only registered bounded host tools are allowed",
        }))
        if change == "unregistered":
            state = _saved_completion_reserve_tool_failure(tool="shell")
        elif change == "malformed":
            state = _saved_completion_reserve_tool_failure(arguments="{not-json")
        else:
            state = _saved_completion_reserve_tool_failure(completion_reserve=False)
        (attempt / "runtime_state.json").write_text(json.dumps(state))

        assert research_module._runtime_failure_code(state) == "operational"
        assert controller._attempt_failure_code(attempt) == "operational"
    finally:
        controller.close()


def test_saved_completion_reserve_mapping_does_not_override_unsettled_transport():
    state = _saved_completion_reserve_tool_failure()
    state["workers"]["agent-a"] = {
        "status": "failed",
        "failure_code": "transport_unsettled",
        "requests": [{"status": "uncertain", "error_code": "transport_unsettled"}],
    }

    assert research_module._runtime_failure_code(state) == "transport_unsettled"


def test_saved_completion_reserve_mapping_rejects_uncertain_tool_record():
    state = _saved_completion_reserve_tool_failure()
    state["workers"]["agent-b"]["tools"]["terminal-call"] = {
        "status": "started",
        "tool": "browser_navigate",
    }

    assert research_module._runtime_failure_code(state) == "operational"


@pytest.mark.parametrize("phase", ["synthesize", "review"])
def test_durable_submission_completes_without_an_ack_request(tmp_path, phase):
    RecoveryProvider.ack_timeouts = {phase}

    result = execute(tmp_path, limits={"max_rounds": 1})

    assert result["status"] == "completed", result
    assert result["accepted_by_qc"] is True
    phase_root = tmp_path / "rounds/001" / phase
    attempts = sorted(phase_root.glob("attempt-*"))
    assert [path.name for path in attempts] == ["attempt-001"]
    assert (phase_root / "result.json").exists()
    runtime = json.loads((attempts[0] / "runtime_state.json").read_text())
    worker = runtime["workers"]["agent-a"]
    assert worker["requests"][-1]["status"] == "completed"
    assert worker["completion_kind"] == "host_tool_submission"
    assert not RecoveryProvider.fired_ack_timeouts
    assert result["usage"]["unresolved_requests"] == 0
    assert result["usage"]["usage_complete"] is True


def test_phase_attempts_obey_exact_cumulative_request_budget(tmp_path):
    RecoveryProvider.failures = {("plan", "attempt-001"): "timeout"}

    result = execute(tmp_path, limits={"max_rounds": 1, "max_requests": 3})

    assert result["status"] == "paused", result
    assert result["accepted_by_qc"] is False
    assert result["usage"]["requests_including_uncertain"] == 2
    assert len(RecoveryProvider.calls) == 2
    attempts = sorted((tmp_path / "rounds/001/plan").glob("attempt-*"))
    assert [path.name for path in attempts] == ["attempt-001", "attempt-002"]
    assert not list((tmp_path / "rounds/001/work").glob("attempt-*"))


def _write_runtime(path, *, rows, total_tokens, incomplete=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": "example/model",
        "limits": {"request_timeout": 180},
        "workers": {"agent-a": {
            "status": "failed" if any(row["status"] != "completed" for row in rows) else "completed",
            "token_totals": {"totalTokens": total_tokens},
            "token_usage_incomplete": incomplete,
            "requests": rows,
        }},
    }))


def _completed(request_id, total_tokens):
    return {
        "request_id": request_id,
        "status": "completed",
        "usage": {
            "prompt_tokens": total_tokens - 10,
            "completion_tokens": 10,
            "total_tokens": total_tokens,
        },
    }


def test_nested_attempt_diagnostics_include_legacy_and_every_attempt(tmp_path):
    phase = tmp_path / "rounds/001/synthesize"
    _write_runtime(
        phase / "runtime_state.json",
        rows=[_completed("legacy-good", 15), {
            "request_id": "legacy-500", "status": "failed", "http_status": 500,
        }],
        total_tokens=15,
    )
    _write_runtime(
        phase / "attempt-001/runtime_state.json",
        rows=[_completed("attempt-one-good", 30), {
            "request_id": "attempt-one-timeout", "status": "uncertain",
            "error_type": "TimeoutError",
        }],
        total_tokens=30,
        incomplete=True,
    )
    _write_runtime(
        phase / "attempt-002/runtime_state.json",
        rows=[_completed("attempt-two-good", 60)],
        total_tokens=60,
    )

    result = runtime_diagnostics(tmp_path)

    assert result["usage"] == {
        "requests_including_uncertain": 5,
        "reported_total_tokens": None,
        "known_reported_total_tokens": 105,
        "unresolved_requests": 1,
        "usage_complete": False,
    }
    failures = {(row["request_id"], row["round"], row["phase"], row["attempt"])
                for row in result["request_failures"]}
    assert failures == {
        ("legacy-500", "001", "synthesize", "legacy"),
        ("attempt-one-timeout", "001", "synthesize", "attempt-001"),
    }


def test_runtime_failure_uses_failure_agent_before_cancelled_sibling():
    state = {
        "fatal_kind": "provider_timeout",
        "failure_code": "provider_timeout",
        "failure_agent": "agent-a",
        "workers": {
            "agent-a": {
                "status": "failed",
                "failure_code": "provider_timeout",
                "requests": [{
                    "request_id": "primary-timeout",
                    "status": "uncertain",
                    "error_type": "TimeoutError",
                    "error_code": "request_timeout",
                }],
            },
            "agent-b": {
                "status": "interrupted",
                "requests": [{
                    "request_id": "cancelled-sibling",
                    "status": "uncertain",
                    "error_type": "CancelledError",
                    "error_code": "request_cancelled",
                }],
            },
        },
    }

    assert research_module._runtime_failure_code(state) == "provider_timeout"


def test_runtime_failure_promotes_unsettled_cancelled_sibling():
    state = {
        "fatal_kind": "provider_timeout",
        "failure_code": "provider_timeout",
        "failure_agent": "agent-a",
        "workers": {
            "agent-a": {
                "status": "failed",
                "failure_code": "provider_timeout",
                "requests": [{
                    "request_id": "primary-timeout",
                    "status": "uncertain",
                    "error_type": "TimeoutError",
                    "error_code": "request_timeout",
                }],
            },
            "agent-b": {
                "status": "failed",
                "failure_code": "transport_unsettled",
                "requests": [{
                    "request_id": "stuck-sibling",
                    "status": "uncertain",
                    "error_type": "TransportUnsettled",
                    "error_code": "transport_unsettled",
                }],
            },
        },
    }

    assert research_module._runtime_failure_code(state) == "transport_unsettled"


def test_legacy_direct_runtime_is_attempt_one_for_recovery(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        phase = tmp_path / "rounds/001/synthesize"
        _write_runtime(
            phase / "runtime_state.json",
            rows=[{
                "request_id": "legacy-rate-limit",
                "status": "rate_limited",
                "http_status": 429,
            }],
            total_tokens=0,
        )

        attempts = controller._phase_attempts(phase)

        assert [(number, name, path) for number, name, path in attempts] == [
            (1, "legacy", phase),
        ]
        assert controller._attempt_failure_code(phase) == "http_429"
    finally:
        controller.close()


def test_actionable_qwen_blocked_verdict_continues_to_next_round(tmp_path):
    RecoveryProvider.verdicts = ["blocked", "accept"]

    result = execute(tmp_path, limits={"max_rounds": 2})

    assert result["status"] == "completed", result
    assert result["accepted_by_qc"] is True
    assert [row["verdict"] for row in result["history"]] == ["blocked", "accept"]
    assert any(call["round"] == 2 and call["phase"] == "plan"
               for call in RecoveryProvider.calls)


def test_qc_cannot_accept_selected_synthesis_with_unresolved_requirements(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        controller.state["criteria"] = copy.deepcopy(CRITERIA)
        synthesis_id = controller.put_artifact(
            "model_analysis",
            "round-1/synthesize/agent-a",
            {"answer": "Candidate", "evidence_ids": [], "unresolved": ["Inspect the final flow"]},
        )
        review = {
            "verdict": "accept",
            "objective_met": True,
            "summary": "Everything passes.",
            "checks": [{
                "criterion_id": "C1",
                "status": "pass",
                "reason": "The candidate addresses the criterion.",
                "evidence_ids": [synthesis_id],
            }],
            "next_steps": [],
        }

        with pytest.raises(ToolInputError, match="unresolved"):
            controller._validate_result("review", review, REVIEW_SCHEMA, [synthesis_id])
    finally:
        controller.close()


def test_accepted_by_qc_requires_latest_accept_history(tmp_path):
    controller = Research(tmp_path, config=configuration())
    try:
        controller.state["status"] = "completed"
        assert controller.report()["accepted_by_qc"] is False

        controller.state["history"] = [{"verdict": "accept"}, {"verdict": "revise"}]
        assert controller.report()["accepted_by_qc"] is False

        controller.state["history"][-1]["verdict"] = "accept"
        assert controller.report()["accepted_by_qc"] is True
    finally:
        controller.close()
