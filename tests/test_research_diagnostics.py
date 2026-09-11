"""Historical failure reporting requires no provider calls or run recovery."""
import json

import pytest

from astra_harness.cli import main
from astra_harness.research import runtime_diagnostics


def saved_run(root, request_status="uncertain", usage=None):
    phase = root / "rounds/001/synthesize"
    phase.mkdir(parents=True)
    (phase / "runtime_state.json").write_text(json.dumps({
        "model": "example/head", "limits": {"request_timeout": 60},
        "workers": {"agent-a": {
            "status": "failed", "token_totals": {"totalTokens": 150},
            # Historical runtimes omitted token_usage_incomplete on timeout.
            "requests": [
                {"request_id": "good", "status": "completed", "usage": usage if usage is not None else
                 {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
                {"request_id": "unanswered", "status": request_status, "error_type": "TimeoutError"},
            ]}}}))
    (root / "research.json").write_text(json.dumps({
        "status": "blocked", "phase": "synthesize", "config": {"models": {"head": "example/head"}},
        "history": []}))


@pytest.mark.parametrize("request_status", ["uncertain", "intent", "inflight"])
def test_historical_unanswered_request_invalidates_complete_usage(tmp_path, request_status):
    saved_run(tmp_path, request_status)
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    result = runtime_diagnostics(tmp_path)
    assert result["usage"] == {
        "requests_including_uncertain": 2, "reported_total_tokens": None,
        "known_reported_total_tokens": 150, "unresolved_requests": 1, "usage_complete": False}
    assert files == {p: p.read_bytes() for p in files}


def test_diagnostics_identify_phase_without_inventing_missing_timing(tmp_path):
    saved_run(tmp_path)
    failure = runtime_diagnostics(tmp_path)["request_failures"][0]
    assert failure["phase"] == "synthesize"
    assert failure["model"] == "example/head"
    assert failure["timeout_seconds"] == 60
    assert failure["elapsed_seconds"] is None


def test_status_recomputes_usage_without_restarting_or_rewriting_run(tmp_path, capsys):
    saved_run(tmp_path)
    files = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(SystemExit) as exited:
        main(["status", "--run-dir", str(tmp_path)])
    assert exited.value.code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert result["usage"]["usage_complete"] is False
    assert result["request_failures"][0]["error_type"] == "TimeoutError"
    assert files == {p: p.read_bytes() for p in files}


def test_missing_response_usage_does_not_look_complete(tmp_path):
    saved_run(tmp_path, request_status="failed", usage={"prompt_tokens": 100})
    result = runtime_diagnostics(tmp_path)
    assert result["usage"]["unresolved_requests"] == 0
    assert result["usage"]["usage_complete"] is False
    assert result["usage"]["known_reported_total_tokens"] == 150
