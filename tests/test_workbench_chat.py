"""Visual-draft assistant tests using synthetic context and an injected runner."""
import copy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from astra_harness.workbench import WorkbenchReader
from astra_harness.workbench_catalog import get_catalog


class FakeRunner:
    def __init__(self, result=None, *, authenticated=True, available=True, blocked=False, error=None):
        self.result = result if result is not None else {"answer": "The coordinator owns the worker lifecycle.", "proposal": None}
        self.authenticated = authenticated
        self.available = available
        self.error = error
        self.calls = []
        self.status_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        if not blocked:
            self.release.set()

    def status(self):
        self.status_calls += 1
        return {"authenticated": self.authenticated, "available": self.available,
                "model": "example/default-model", "cli_version": "0.154.0"}

    def __call__(self, context, cancel_event):
        self.calls.append(copy.deepcopy(context))
        self.started.set()
        while not self.release.wait(0.01):
            if cancel_event.is_set():
                self.cancelled.set()
                return {"answer": "Cancelled work must not become an accepted answer.", "proposal": None}
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.result)


@pytest.fixture
def chat_workspace(tmp_path):
    package = tmp_path / "astra_harness"
    package.mkdir()
    for component in get_catalog(tmp_path)["components"]:
        path = tmp_path / component["source"]
        path.write_text('"""Synthetic ' + component["id"] + ' source."""\n', encoding="utf-8")
    (package / "coordinator.py").write_text(
        'COORDINATOR_SENTINEL = "actual selected source"\nAPI_KEY = "SYNTHETIC_SECRET"\n', encoding="utf-8")
    (package / "router.py").write_text('ROUTER_SENTINEL = "actual routing source"\n', encoding="utf-8")
    (package / "research.py").write_text(
        'DEFAULT_MODELS = {"head": "example/head", "worker": "example/worker", "qc": "example/qc"}\n'
        'DEFAULT_LIMITS = {"max_rounds": 8}\n', encoding="utf-8")
    run = tmp_path / "runs" / "generic-comparison"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"model": "example/model", "provider": "codex",
        "worker_count": 2, "status": "completed", "run_id": str(uuid.uuid4())}), encoding="utf-8")
    return tmp_path


@pytest.fixture
def assistant_factory(chat_workspace):
    from astra_harness.workbench_chat import WorkbenchAssistant
    assistants = []

    def create(runner=None):
        actual_runner = runner or FakeRunner()
        assistant = WorkbenchAssistant(WorkbenchReader(chat_workspace), runner=actual_runner)
        assistants.append(assistant)
        return assistant, actual_runner

    yield create
    for assistant in reversed(assistants):
        assistant.close()


def payload(root, *, request_id=None, conversation_id=None, scene="mission", components=None):
    catalog = get_catalog(root)
    diagram = next(item for item in catalog["scenes"] if item["id"] == scene)
    return {
        "request_id": request_id or str(uuid.uuid4()), "conversation_id": conversation_id,
        "message": "Explain the selected component and suggest a clearer visual layout.",
        "selection": {"scene": scene, "component_ids": components or ["coordinator"],
                      "run_id": None, "event_id": None, "selected_text": ""},
        "workspace": {"version": 1, "name": "Hyperspace Workbench", "scene": scene,
                      "positions": {}, "notes": {},
                      "drafts": {scene: {"nodes": [], "edges": copy.deepcopy(diagram["edges"]), "parameters": {}}}},
    }


def completed(assistant, job_id, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = assistant.job(job_id)
        if job["status"] not in {"queued", "pending", "running"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Injected assistant job did not reach a terminal state")


def proposal(*operations):
    return {"title": "Clarify the assembly", "description": "A visual workspace draft only.",
            "operations": list(operations)}


def source_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in (root / "astra_harness").glob("*.py")}


def test_status_reports_chatgpt_subscription_without_generating(assistant_factory):
    assistant, runner = assistant_factory()
    status = assistant.status()
    assert status["authenticated"] is True
    assert status["available"] is True
    assert status["provider"] == "chatgpt"
    assert status["label"] == "ChatGPT Sub"
    assert status["cli_version"] == "0.154.0"
    assert isinstance(status["csrf"], str) and len(status["csrf"]) >= 24
    assert runner.status_calls >= 1
    assert runner.calls == []


@pytest.mark.parametrize("settings", [{"authenticated": False}, {"available": False}])
def test_unavailable_subscription_never_falls_back_to_another_provider(assistant_factory, chat_workspace, settings):
    assistant, runner = assistant_factory(FakeRunner(**settings))
    with pytest.raises(RuntimeError):
        assistant.submit(payload(chat_workspace))
    assert runner.calls == []


def test_selection_is_reconstructed_from_catalog_source_and_redacted(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    request["selection"]["run_id"] = WorkbenchReader(chat_workspace).list_runs()["runs"][0]["id"]
    request["selection"]["selected_text"] = "UNTRUSTED_HIGHLIGHT: describe the selected part"
    submitted = assistant.submit(request)
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert [part["id"] for part in submitted["context"]["components"]] == ["coordinator"]
    encoded = json.dumps(runner.calls[0])
    assert "COORDINATOR_SENTINEL" in encoded
    assert "SYNTHETIC_SECRET" not in encoded
    assert "ROUTER_SENTINEL" not in encoded
    assert "UNTRUSTED_HIGHLIGHT" in encoded
    assert "generic-comparison" in encoded


def test_selected_source_is_bounded(assistant_factory, chat_workspace):
    (chat_workspace / "astra_harness" / "coordinator.py").write_text(
        "SOURCE_START\n" + "ordinary source line\n" * 30_000 + "SHOULD_BE_OUTSIDE_CONTEXT\n", encoding="utf-8")
    assistant, runner = assistant_factory()
    submitted = assistant.submit(payload(chat_workspace))
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    encoded = json.dumps(runner.calls[0])
    assert "SOURCE_START" in encoded
    assert "SHOULD_BE_OUTSIDE_CONTEXT" not in encoded
    assert len(encoded.encode()) < 200_000


def test_chat_without_a_selected_component_does_not_invent_source_context(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    request["selection"]["component_ids"] = []
    submitted = assistant.submit(request)
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    assert submitted["context"]["components"] == []
    assert "COORDINATOR_SENTINEL" not in json.dumps(runner.calls[0])


@pytest.mark.parametrize("change", [
    {"message": ""}, {"message": "x" * 6001}, {"message": 12},
    {"request_id": "not-a-uuid"}, {"conversation_id": "not-a-uuid"},
    {"provider": "openrouter"}, {"model": "unrequested/model"}, {"api_key": "unused-synthetic-key"},
])
def test_invalid_message_and_provider_overrides_are_rejected_before_generation(assistant_factory, chat_workspace, change):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    request.update(change)
    with pytest.raises((ValueError, KeyError)):
        assistant.submit(request)
    assert runner.calls == []


@pytest.mark.parametrize("selection", [
    {"component_ids": ["missing-component"]},
    {"component_ids": ["research-qc"]},
    {"component_ids": ["coordinator"] * 9},
    {"selected_text": "x" * 4001},
    {"source": "INVENTED_CLIENT_SOURCE"},
    {"run_id": "../../private"},
    {"event_id": "invented-event-without-a-run"},
])
def test_selection_schema_unknown_parts_and_client_sources_fail_closed(assistant_factory, chat_workspace, selection):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    request["selection"].update(selection)
    with pytest.raises((ValueError, KeyError)):
        assistant.submit(request)
    assert runner.calls == []


def test_jobs_are_async_and_duplicate_request_ids_do_not_generate_twice(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory(FakeRunner(blocked=True))
    request = payload(chat_workspace)
    started = time.monotonic()
    submitted = assistant.submit(request)
    assert time.monotonic() - started < 1
    assert runner.started.wait(1)
    duplicate = assistant.submit(copy.deepcopy(request))
    assert duplicate["job_id"] == submitted["job_id"]
    changed = copy.deepcopy(request)
    changed["message"] = "A different message under the same request ID"
    with pytest.raises(ValueError):
        assistant.submit(changed)
    runner.release.set()
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert len(runner.calls) == 1
    assert assistant.submit(request)["job_id"] == submitted["job_id"]
    assert len(runner.calls) == 1


def test_only_one_model_job_runs_at_a_time(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory(FakeRunner(blocked=True))
    submitted = assistant.submit(payload(chat_workspace))
    assert runner.started.wait(1)
    with pytest.raises(RuntimeError):
        assistant.submit(payload(chat_workspace))
    runner.release.set()
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    assert len(runner.calls) == 1


def test_cancel_signals_runner_and_suppresses_late_success(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory(FakeRunner(blocked=True))
    submitted = assistant.submit(payload(chat_workspace))
    assert runner.started.wait(1)
    assistant.cancel(submitted["job_id"])
    assert runner.cancelled.wait(1)
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "cancelled"
    assert not result.get("proposal")
    assert "Cancelled work must not" not in result.get("answer", "")


def test_runner_failure_is_redacted_and_does_not_retry(assistant_factory, chat_workspace):
    secret = "sk-or-v1-" + "a" * 64
    assistant, runner = assistant_factory(FakeRunner(error=RuntimeError("Provider error " + secret)))
    submitted = assistant.submit(payload(chat_workspace))
    result = completed(assistant, submitted["job_id"])
    assert result["status"] in {"error", "failed"}
    assert secret not in json.dumps(result)
    assert len(runner.calls) == 1


def test_valid_visual_draft_never_modifies_harness_source(assistant_factory, chat_workspace):
    draft = proposal(
        {"type": "add_component", "component_id": "draft-review", "name": "Review checkpoint",
         "description": "An optional visual checkpoint", "category": "control", "x": 700, "y": 600},
        {"type": "connect", "source": "coordinator", "target": "draft-review", "label": "review", "kind": "control"},
        {"type": "move_component", "component_id": "coordinator", "x": 640, "y": 120},
        {"type": "set_note", "component_id": "coordinator", "text": "Keep orchestration deterministic."},
    )
    before = source_bytes(chat_workspace)
    assistant, _ = assistant_factory(FakeRunner(result={"answer": "Preview this arrangement before applying it.", "proposal": draft}))
    submitted = assistant.submit(payload(chat_workspace))
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert result["proposal"] == draft
    assert source_bytes(chat_workspace) == before


def test_existing_draft_component_can_be_selected_and_removed(assistant_factory, chat_workspace):
    request = payload(chat_workspace, components=["draft-review"])
    request["workspace"]["drafts"]["mission"]["nodes"].append({
        "id": "draft-review", "name": "Review checkpoint", "description": "A user-created visual draft.",
        "category": "control", "x": 700, "y": 600})
    draft = proposal({"type": "remove_component", "component_id": "draft-review"})
    assistant, runner = assistant_factory(FakeRunner(result={"answer": "Preview removing this visual draft.", "proposal": draft}))
    submitted = assistant.submit(request)
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert result["proposal"] == draft
    assert submitted["context"]["components"][0]["id"] == "draft-review"
    assert "COORDINATOR_SENTINEL" not in json.dumps(runner.calls[0])


def test_existing_negative_layout_coordinates_do_not_prevent_chat(assistant_factory, chat_workspace):
    request = payload(chat_workspace)
    request["workspace"]["positions"] = {"mission": {"coordinator": {"x": -120, "y": -80}}}
    assistant, _ = assistant_factory()
    submitted = assistant.submit(request)
    assert completed(assistant, submitted["job_id"])["status"] == "completed"


@pytest.mark.parametrize("operation", [
    {"type": "write_file", "path": "coordinator.py", "content": "changed"},
    {"type": "remove_component", "component_id": "coordinator"},
    {"type": "remove_component", "component_id": "draft-missing"},
    {"type": "add_component", "component_id": "coordinator", "name": "Replacement",
     "description": "Cannot shadow a canonical component", "category": "control", "x": 100, "y": 100},
    {"type": "move_component", "component_id": "missing", "x": 100, "y": 100},
    {"type": "move_component", "component_id": "research-qc", "x": 100, "y": 100},
    {"type": "move_component", "component_id": "coordinator", "x": -1, "y": 100},
    {"type": "move_component", "component_id": "coordinator", "x": 2401, "y": 100},
    {"type": "move_component", "component_id": "coordinator", "x": 100, "y": 1601},
    {"type": "move_component", "component_id": "coordinator", "x": float("nan"), "y": 100},
    {"type": "move_component", "component_id": "coordinator", "x": 100, "y": 100, "path": "coordinator.py"},
    {"type": "connect", "source": "coordinator", "target": "missing", "label": "missing", "kind": "data"},
    {"type": "connect", "source": "coordinator", "target": "router", "label": "invalid", "kind": "execute"},
    {"type": "set_note", "component_id": "coordinator", "text": "x" * 4001},
])
def test_unsupported_or_out_of_scope_proposals_fail_without_source_changes(assistant_factory, chat_workspace, operation):
    before = source_bytes(chat_workspace)
    assistant, runner = assistant_factory(FakeRunner(result={"answer": "A proposed edit", "proposal": proposal(operation)}))
    submitted = assistant.submit(payload(chat_workspace))
    result = completed(assistant, submitted["job_id"])
    assert result["status"] in {"error", "failed"}
    assert not result.get("proposal")
    assert len(runner.calls) == 1
    assert source_bytes(chat_workspace) == before


@pytest.mark.parametrize("result", ["{malformed", [], {"answer": 7, "proposal": None},
    {"answer": "", "proposal": None},
    {"answer": "Reply", "proposal": None, "execute": "unsupported"},
    {"answer": "Reply", "proposal": {"title": "Missing operations"}},
    {"answer": "Reply", "proposal": proposal(*[
        {"type": "set_note", "component_id": "coordinator", "text": "note"} for _ in range(13)])},
    {"answer": "Reply", "proposal": proposal(*[
        {"type": "set_note", "component_id": "coordinator", "text": "large note " * 300} for _ in range(3)])},
])
def test_malformed_model_results_and_oversized_proposals_fail(assistant_factory, chat_workspace, result):
    assistant, runner = assistant_factory(FakeRunner(result=result))
    submitted = assistant.submit(payload(chat_workspace))
    assert completed(assistant, submitted["job_id"])["status"] in {"error", "failed"}
    assert len(runner.calls) == 1


def test_close_cancels_owned_background_work(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory(FakeRunner(blocked=True))
    submitted = assistant.submit(payload(chat_workspace))
    assert runner.started.wait(1)
    assistant.close()
    assert runner.cancelled.wait(1)
    assert completed(assistant, submitted["job_id"])["status"] in {"cancelled", "uncertain"}


def test_history_bound_preserves_saved_conversations_and_refuses_new_ones(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory()
    conversations = []
    for index in range(20):
        request = payload(chat_workspace)
        request["message"] = "Explain this component, conversation " + str(index)
        submitted = assistant.submit(request)
        assert completed(assistant, submitted["job_id"])["status"] == "completed"
        conversations.append(submitted["conversation_id"])
    history = assistant.history()["conversations"]
    assert len(history) == 20
    assert conversations[-1] in {conversation["id"] for conversation in history}
    assert conversations[0] in {conversation["id"] for conversation in history}
    with pytest.raises(RuntimeError):
        assistant.submit(payload(chat_workspace))
    assert len(runner.calls) == 20
    continued = assistant.submit(payload(chat_workspace, conversation_id=conversations[-1]))
    assert completed(assistant, continued["job_id"])["status"] == "completed"
    assert all(message["role"] in {"user", "assistant"}
               for conversation in history for message in conversation["messages"])


def test_history_and_saved_chat_state_redact_credentials(assistant_factory, chat_workspace):
    secret = "sk-or-v1-" + "b" * 64
    assistant, runner = assistant_factory(FakeRunner(result={"answer": "Never echo " + secret, "proposal": None}))
    request = payload(chat_workspace)
    request["message"] = "Explain this component. API_KEY=\"" + secret + "\""
    submitted = assistant.submit(request)
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert secret not in json.dumps(result)
    assert secret not in json.dumps(runner.calls)
    assert secret not in json.dumps(assistant.history())
    for path in (chat_workspace / "data" / "workbench-chat").rglob("*.json"):
        assert secret not in path.read_text(encoding="utf-8")


def test_completed_request_reopens_without_authentication_or_generation(assistant_factory, chat_workspace):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    submitted = assistant.submit(request)
    original = completed(assistant, submitted["job_id"])
    assert original["status"] == "completed"
    assistant.close()
    reopened, offline = assistant_factory(FakeRunner(authenticated=False, available=False))
    replay = reopened.submit(request)
    assert replay["job_id"] == submitted["job_id"]
    assert reopened.job(replay["job_id"])["answer"] == original["answer"]
    assert offline.calls == []
    assert len(runner.calls) == 1


@pytest.mark.parametrize("removed_context", ["run", "source"])
def test_saved_request_replay_does_not_require_current_selected_context(assistant_factory, chat_workspace, removed_context):
    assistant, runner = assistant_factory()
    request = payload(chat_workspace)
    request["selection"]["run_id"] = WorkbenchReader(chat_workspace).list_runs()["runs"][0]["id"]
    submitted = assistant.submit(request)
    original = completed(assistant, submitted["job_id"])
    assert original["status"] == "completed"
    assistant.close()
    removed = (chat_workspace / "runs" / "generic-comparison" / "manifest.json" if removed_context == "run"
               else chat_workspace / "astra_harness" / "coordinator.py")
    removed.unlink()
    reopened, offline = assistant_factory(FakeRunner(authenticated=False, available=False))
    replay = reopened.submit(request)
    assert replay["job_id"] == submitted["job_id"]
    assert reopened.job(replay["job_id"])["answer"] == original["answer"]
    assert offline.calls == []
    assert len(runner.calls) == 1


def test_uncertain_restart_preserves_request_id_without_replaying_generation(assistant_factory, chat_workspace):
    assistant, _ = assistant_factory()
    request = payload(chat_workspace)
    submitted = assistant.submit(request)
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    assistant.close()
    path = chat_workspace / "data" / "workbench-chat" / "state.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["jobs"][submitted["job_id"]]["status"] = "running"
    saved["jobs"][submitted["job_id"]].pop("answer", None)
    saved["jobs"][submitted["job_id"]].pop("proposal", None)
    conversation = saved["conversations"][submitted["conversation_id"]]
    conversation["messages"] = [message for message in conversation["messages"] if message["role"] == "user"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    reopened, runner = assistant_factory()
    assert reopened.job(submitted["job_id"])["status"] == "uncertain"
    replay = reopened.submit(request)
    assert replay["job_id"] == submitted["job_id"]
    assert reopened.job(replay["job_id"])["status"] == "uncertain"
    assert runner.calls == []


def test_chat_history_is_saved_with_private_permissions(assistant_factory, chat_workspace):
    assistant, _ = assistant_factory()
    submitted = assistant.submit(payload(chat_workspace))
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    directory = chat_workspace / "data" / "workbench-chat"
    assert directory.stat().st_mode & 0o077 == 0
    assert (directory / "state.json").stat().st_mode & 0o077 == 0


def test_hidden_canonical_nodes_are_valid_visual_workspace_state(assistant_factory, chat_workspace):
    request = payload(chat_workspace)
    request["workspace"]["drafts"]["mission"]["hidden"] = ["event-bus"]
    assistant, _ = assistant_factory()
    submitted = assistant.submit(request)
    assert completed(assistant, submitted["job_id"])["status"] == "completed"


@pytest.mark.parametrize("hidden", [["missing-component"], ["research-qc"], ["draft-missing"]])
def test_hidden_workspace_ids_must_be_canonical_parts_of_the_scene(assistant_factory, chat_workspace, hidden):
    request = payload(chat_workspace)
    request["workspace"]["drafts"]["mission"]["hidden"] = hidden
    assistant, runner = assistant_factory()
    with pytest.raises(ValueError):
        assistant.submit(request)
    assert runner.calls == []


def test_native_command_disables_actions_and_drops_api_key_environment(monkeypatch, tmp_path):
    from astra_harness import workbench_chat as chat
    config_directory = tmp_path / "synthetic-codex-config"
    config_directory.mkdir()
    (config_directory / "config.toml").write_text(
        '[mcp_servers.example]\ncommand = "unused-command"\n', encoding="utf-8")
    environment = {"PATH": "/synthetic/bin", "CODEX_HOME": str(config_directory), "LANG": "en_US.UTF-8",
                   "OPENAI_API_KEY": "synthetic-openai-key", "OPENROUTER_API_KEY": "synthetic-openrouter-key",
                   "OPENAI_BASE_URL": "https://unrequested.example", "UNRELATED_SECRET": "synthetic-secret"}
    monkeypatch.setattr(chat, "os", SimpleNamespace(environ=environment))
    monkeypatch.setattr(chat.shutil, "which", lambda name: "/synthetic/codex")
    command = chat.native_command()
    assert command[:3] == ["/synthetic/codex", "app-server", "--stdio"]
    assert 'model_provider="openai"' in command
    assert 'forced_login_method="chatgpt"' in command
    assert 'sandbox_mode="read-only"' in command
    assert 'approval_policy="never"' in command
    assert "mcp_servers.example.enabled=false" in command
    disabled = {command[index + 1] for index, token in enumerate(command[:-1]) if token == "--disable"}
    assert {"apps", "plugins", "shell_tool", "multi_agent", "browser_use"} <= disabled
    assert chat.safe_env() == {"PATH": "/synthetic/bin", "CODEX_HOME": str(config_directory), "LANG": "en_US.UTF-8"}


@pytest.mark.parametrize("login_text,returncode,authenticated", [
    (b"Logged in using ChatGPT", 0, True),
    (b"Logged in using an API key", 0, False),
    (b"Not logged in", 1, False),
])
def test_native_status_checks_current_cli_and_subscription_without_starting_a_turn(monkeypatch, login_text, returncode, authenticated):
    from astra_harness import workbench_chat as chat
    calls = []

    def run(command, **options):
        calls.append((command, options))
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout=b"codex-cli 0.154.0\n", stderr=b"")
        assert command[-2:] == ["login", "status"]
        return SimpleNamespace(returncode=returncode, stdout=login_text, stderr=b"")

    def forbidden_process(*arguments, **options):
        raise AssertionError("Status must not start an app-server or generation")

    monkeypatch.setattr(chat.shutil, "which", lambda name: "/synthetic/codex")
    monkeypatch.setattr(chat.subprocess, "run", run)
    monkeypatch.setattr(chat.subprocess, "Popen", forbidden_process)
    result = chat.NativeCodexRunner().status()
    assert result["cli_version"] == "0.154.0"
    assert result["authenticated"] is authenticated
    assert result["available"] is authenticated
    assert len(calls) == 2
    assert 'forced_login_method="chatgpt"' in calls[1][0]


def native_fixture(monkeypatch, *, account="chatgpt", default_models=1, returned_model=None):
    from astra_harness import workbench_chat as chat
    processes = []

    class NativeProcess:
        def __init__(self, directory, cancel_event):
            self.directory = directory
            self.calls = []
            self.messages = []
            self.closed = False
            self.completed = False
            self.thread_id = None
            self.final = json.dumps({"answer": "A bounded explanation.", "proposal": None})
            processes.append(self)

        def request(self, method, parameters):
            self.calls.append((method, parameters))
            if method == "account/read":
                return {"account": {"type": account}}
            if method == "model/list":
                return {"data": [{"model": "example/model-" + str(index), "isDefault": True,
                                   "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}
                                  for index in range(default_models)], "nextCursor": None}
            if method == "thread/start":
                return {"thread": {"id": "synthetic-thread"}, "model": returned_model or "example/model-0",
                        "modelProvider": "openai"}
            if method == "turn/start":
                self.completed = True
            return {}

        def _send(self, message):
            self.messages.append(message)

        def close(self):
            self.closed = True

    monkeypatch.setattr(chat, "_NativeProcess", NativeProcess)
    return chat, processes


def test_native_generation_uses_exact_advertised_model_and_no_tools(monkeypatch):
    chat, processes = native_fixture(monkeypatch)
    result = chat.NativeCodexRunner()({"message": "Explain the selected component", "selected_text": "Untrusted quotation"}, threading.Event())
    assert result["model"] == "example/model-0"
    process = processes[0]
    calls = dict(process.calls)
    thread = calls["thread/start"]
    assert thread["modelProvider"] == "openai"
    assert thread["allowProviderModelFallback"] is False
    assert thread["ephemeral"] is True
    assert thread["sandbox"] == "read-only"
    assert thread["approvalPolicy"] == "never"
    assert thread["dynamicTools"] == []
    assert thread["environments"] == []
    assert thread["runtimeWorkspaceRoots"] == [process.directory]
    assert "untrusted data" in thread["baseInstructions"]
    assert calls["turn/start"]["model"] == thread["model"]
    assert calls["turn/start"]["outputSchema"]["additionalProperties"] is False
    assert process.closed is True
    assert not Path(process.directory).exists()


@pytest.mark.parametrize("configuration", [
    {"account": "apiKey"}, {"default_models": 0}, {"default_models": 2}, {"returned_model": "unrequested/model"},
])
def test_native_generation_refuses_auth_or_model_substitution_and_closes_process(monkeypatch, configuration):
    chat, processes = native_fixture(monkeypatch, **configuration)
    with pytest.raises(RuntimeError):
        chat.NativeCodexRunner()({"message": "Explain"}, threading.Event())
    assert processes[0].closed is True
    assert "turn/start" not in {method for method, _ in processes[0].calls}


@pytest.mark.parametrize("notification", [
    {"method": "model/rerouted", "params": {}},
    {"method": "item/started", "params": {"item": {"type": "commandExecution"}}},
    {"method": "item/completed", "params": {"item": {"type": "fileChange"}}},
    {"method": "item/tool/requestApproval", "id": 7, "params": {}},
])
def test_native_protocol_refuses_actions_and_reroutes(notification):
    from astra_harness.workbench_chat import _NativeProcess
    process = object.__new__(_NativeProcess)
    sent = []
    process._send = sent.append
    with pytest.raises(RuntimeError):
        process._notification(notification)
    if "id" in notification:
        assert sent[0]["id"] == notification["id"]
        assert sent[0]["error"]["code"] == -32601


@pytest.mark.parametrize("requires_kill", [False, True])
def test_native_process_cleanup_terminates_its_group_and_closes_streams(monkeypatch, requires_kill):
    from astra_harness import workbench_chat as chat
    closed = []
    signals = []

    class Process:
        def __init__(self):
            self.pid = 987654321
            self.exited = False
            self.waits = 0
            self.stdin = SimpleNamespace(close=lambda: closed.append("stdin"))
            self.stdout = SimpleNamespace(close=lambda: closed.append("stdout"))

        def poll(self):
            return 0 if self.exited else None

        def wait(self, timeout):
            self.waits += 1
            if requires_kill and self.waits == 1:
                raise chat.subprocess.TimeoutExpired("synthetic-native-process", timeout)
            self.exited = True
            return 0

    process = object.__new__(chat._NativeProcess)
    process.process = Process()
    process.selector = SimpleNamespace(close=lambda: closed.append("selector"))
    monkeypatch.setattr(chat.os, "killpg", lambda identifier, signal: signals.append((identifier, signal)))
    process.close()
    assert signals[0] == (987654321, chat.signal.SIGTERM)
    if requires_kill:
        assert signals[1] == (987654321, chat.signal.SIGKILL)
    assert set(closed) == {"selector", "stdin", "stdout"}
    assert process.process.exited is True


def test_timed_out_close_retains_history_ownership_until_runner_exits(assistant_factory, chat_workspace, monkeypatch):
    class SlowRunner(FakeRunner):
        def __call__(self, context, cancel_event):
            self.calls.append(copy.deepcopy(context))
            self.started.set()
            self.release.wait(2)
            return copy.deepcopy(self.result)

    assistant, runner = assistant_factory(SlowRunner(blocked=True))
    submitted = assistant.submit(payload(chat_workspace))
    assert runner.started.wait(1)
    original_join = assistant.thread.join
    monkeypatch.setattr(assistant.thread, "join", lambda timeout: None)
    assistant.close()
    competing, other_runner = assistant_factory()
    try:
        with pytest.raises(RuntimeError):
            competing.submit(payload(chat_workspace))
        assert other_runner.calls == []
    finally:
        runner.release.set()
        original_join(timeout=2)
    assert completed(assistant, submitted["job_id"])["status"] == "cancelled"
    new_job = competing.submit(payload(chat_workspace))
    assert completed(competing, new_job["job_id"])["status"] == "completed"
