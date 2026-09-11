"""Offline provider settings and visual-assistant boundary regressions."""
import copy
import http.client
import io
import json
import threading

import pytest

from astra_harness.workbench import WorkbenchReader
from test_workbench_chat import FakeRunner, chat_workspace, completed, payload, proposal


SECRET = "sk-or-v1-synthetic-settings-test-only"


class ProviderRunner(FakeRunner):
    def __init__(self, provider="chatgpt", **options):
        super().__init__(**options)
        self.provider = provider
        self.model_calls = 0

    def models(self):
        self.model_calls += 1
        return {"models": [{"id": "example/" + self.provider, "name": "Offline " + self.provider}],
                "default": "example/" + self.provider}


@pytest.fixture(autouse=True)
def synthetic_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def settings_factory(chat_workspace):
    from astra_harness.workbench_chat import WorkbenchAssistant
    assistants = []

    def create(native=None, router=None):
        native = native or ProviderRunner()
        router = router or ProviderRunner("openrouter")
        assistant = WorkbenchAssistant(WorkbenchReader(chat_workspace), runner=native, openrouter_runner=router)
        assistants.append(assistant)
        return assistant, native, router

    yield create
    for assistant in reversed(assistants):
        assistant.close()


def configure_router(assistant, **options):
    return assistant.update_settings({"provider": "openrouter", "openrouter": {
        "model": "example/openrouter", "max_tokens": 2048, "secret": SECRET, **options}})


def assert_no_secret(*values):
    assert SECRET not in json.dumps(values)


def test_settings_default_to_chatgpt_without_generation_or_saved_credentials(settings_factory, chat_workspace):
    assistant, native, router = settings_factory()
    settings = assistant.settings()
    assert settings["provider"] == "chatgpt"
    assert settings["chatgpt"]["model"] == ""
    assert settings["openrouter"]["model"] == ""
    assert settings["openrouter"]["max_tokens"] == 2048
    assert settings["openrouter"]["credential_present"] is False
    assert set(settings["privacy"]) == {"include_source", "include_run"}
    assert settings["csrf"] == assistant.status()["csrf"]
    assert native.calls == router.calls == []
    assert not (chat_workspace / "data/workbench-chat/credentials.json").exists()


@pytest.mark.parametrize("provider", ["chatgpt", "openrouter"])
def test_explicit_model_listing_does_not_generate_or_change_provider(settings_factory, provider):
    assistant, native, router = settings_factory()
    before = assistant.settings()
    listed = assistant.models(provider)
    assert listed["models"] == [{"id": "example/" + provider, "name": "Offline " + provider}]
    assert assistant.settings() == before
    assert native.calls == router.calls == []
    assert (native if provider == "chatgpt" else router).model_calls == 1


@pytest.mark.parametrize("provider", ["unknown", "https://example.invalid", "../chatgpt"])
def test_model_listing_refuses_unknown_providers(settings_factory, provider):
    assistant, native, router = settings_factory()
    with pytest.raises((ValueError, KeyError)):
        assistant.models(provider)
    assert native.calls == router.calls == []
    assert native.model_calls == router.model_calls == 0


def test_provider_settings_persist_but_session_secret_does_not(settings_factory, chat_workspace):
    assistant, native, router = settings_factory()
    result = configure_router(assistant)
    assert result["provider"] == "openrouter"
    assert result["openrouter"]["credential_present"] is True
    assert result["openrouter"]["credential_source"] == "session"
    assert_no_secret(result, assistant.status(), assistant.history())
    directory = chat_workspace / "data/workbench-chat"
    assert SECRET not in (directory / "settings.json").read_text()
    assert not (directory / "credentials.json").exists()
    assistant.close()
    restored, _, _ = settings_factory()
    settings = restored.settings()
    assert settings["provider"] == "openrouter"
    assert settings["openrouter"]["model"] == "example/openrouter"
    assert settings["openrouter"]["credential_present"] is False
    assert native.calls == router.calls == []


def test_explicit_secret_persistence_is_private_and_clear_removes_it(settings_factory, chat_workspace):
    assistant, _, _ = settings_factory()
    configure_router(assistant, remember_secret=True)
    path = chat_workspace / "data/workbench-chat/credentials.json"
    assert SECRET in path.read_text()
    assert path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0
    assert_no_secret(assistant.settings(), assistant.history(), assistant.status())
    assistant.close()
    restored, _, _ = settings_factory()
    settings = restored.settings()
    assert settings["openrouter"]["credential_present"] is True
    assert settings["openrouter"]["credential_source"] == "saved"
    cleared = restored.update_settings({"openrouter": {"clear_secret": True}})
    assert cleared["openrouter"]["credential_present"] is False
    assert not path.exists() or SECRET not in path.read_text()


def test_environment_credential_is_reported_without_echo_or_persistence(settings_factory, chat_workspace, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    assistant, native, router = settings_factory()
    settings = assistant.settings()
    assert settings["openrouter"]["credential_present"] is True
    assert settings["openrouter"]["credential_source"] == "environment"
    assert_no_secret(settings, assistant.status(), assistant.models("openrouter"))
    assert not (chat_workspace / "data/workbench-chat/credentials.json").exists()
    assert native.calls == router.calls == []


@pytest.mark.parametrize("change", [
    {"provider": "unknown"}, {"provider": None}, {"api_key": SECRET}, {"endpoint": "https://example.invalid"},
    {"chatgpt": {"model": 12}}, {"chatgpt": {"model": "bad\nmodel"}},
    {"openrouter": {"model": "openrouter/auto"}}, {"openrouter": {"model": "example/model:online"}},
    {"openrouter": {"base_url": "https://example.invalid"}}, {"openrouter": {"tools": []}},
    {"openrouter": {"max_tokens": 255}}, {"openrouter": {"max_tokens": 8193}},
    {"openrouter": {"max_tokens": True}}, {"openrouter": {"max_tokens": 2048.5}},
    {"openrouter": {"remember_secret": "yes"}}, {"openrouter": {"clear_secret": 1}},
    {"openrouter": {"secret": 123}}, {"privacy": {"include_source": "false"}},
    {"privacy": {"include_run": 0}}, {"privacy": {"include_credentials": True}},
])
def test_invalid_settings_are_atomic_and_do_not_generate(settings_factory, change):
    assistant, native, router = settings_factory()
    before = assistant.settings()
    with pytest.raises(ValueError):
        assistant.update_settings(change)
    assert assistant.settings() == before
    assert native.calls == router.calls == []


def test_privacy_settings_omit_source_and_recorded_run_from_context(settings_factory, chat_workspace):
    assistant, native, _ = settings_factory()
    assistant.update_settings({"privacy": {"include_source": False, "include_run": False}})
    request = payload(chat_workspace)
    request["selection"]["run_id"] = WorkbenchReader(chat_workspace).list_runs()["runs"][0]["id"]
    submitted = assistant.submit(request)
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    encoded = json.dumps(native.calls[0])
    assert "COORDINATOR_SENTINEL" not in encoded
    assert "generic-comparison" not in encoded
    assert "coordinator" in encoded


def test_openrouter_selection_uses_only_its_injected_runner(settings_factory, chat_workspace):
    assistant, native, router = settings_factory()
    configure_router(assistant)
    submitted = assistant.submit(payload(chat_workspace))
    result = completed(assistant, submitted["job_id"])
    assert result["status"] == "completed"
    assert result["provider"] == "openrouter"
    assert result["label"] == "OpenRouter"
    assert native.calls == []
    assert len(router.calls) == 1
    assert_no_secret(result, assistant.history(), router.calls)


def test_unavailable_openrouter_does_not_fall_back_to_chatgpt(settings_factory, chat_workspace):
    assistant, native, router = settings_factory(router=ProviderRunner("openrouter", available=False))
    configure_router(assistant)
    with pytest.raises(RuntimeError):
        assistant.submit(payload(chat_workspace))
    assert native.calls == router.calls == []


def test_settings_changes_are_blocked_during_active_reply(settings_factory, chat_workspace):
    assistant, native, router = settings_factory(native=ProviderRunner(blocked=True))
    submitted = assistant.submit(payload(chat_workspace))
    assert native.started.wait(1)
    before = assistant.settings()
    with pytest.raises(RuntimeError, match="(?i)(active|running|reply|busy)"):
        configure_router(assistant)
    assert assistant.settings() == before
    native.release.set()
    assert completed(assistant, submitted["job_id"])["status"] == "completed"
    assert router.calls == []


@pytest.mark.parametrize("change", [
    {"provider": "openrouter", "openrouter": {"model": "example/openrouter", "secret": SECRET}},
    {"chatgpt": {"model": "example/another-advertised-model"}},
])
def test_provider_or_model_switch_requires_new_conversation(settings_factory, chat_workspace, change):
    assistant, native, router = settings_factory()
    original = assistant.submit(payload(chat_workspace))
    assert completed(assistant, original["job_id"])["status"] == "completed"
    assistant.update_settings(change)
    with pytest.raises((ValueError, RuntimeError)):
        assistant.submit(payload(chat_workspace, conversation_id=original["conversation_id"]))
    fresh = assistant.submit(payload(chat_workspace))
    assert completed(assistant, fresh["job_id"])["status"] == "completed"
    runner = router if change.get("provider") == "openrouter" else native
    assert "The coordinator owns the worker lifecycle." not in json.dumps(runner.calls[-1])
    assert fresh["conversation_id"] != original["conversation_id"]


def test_completed_request_replay_keeps_original_provider_without_regeneration(settings_factory, chat_workspace):
    assistant, native, router = settings_factory()
    request = payload(chat_workspace)
    original = assistant.submit(request)
    assert completed(assistant, original["job_id"])["status"] == "completed"
    configure_router(assistant)
    replayed = assistant.submit(copy.deepcopy(request))
    assert replayed["job_id"] == original["job_id"]
    assert replayed["provider"] == "chatgpt"
    assert len(native.calls) == 1
    assert router.calls == []


def test_openrouter_replies_pass_the_same_visual_operation_validator(settings_factory, chat_workspace):
    result = {"answer": "This invalid change must not be accepted.", "proposal": proposal(
        {"type": "remove_component", "component_id": "coordinator"})}
    assistant, native, router = settings_factory(router=ProviderRunner("openrouter", result=result))
    configure_router(assistant)
    submitted = assistant.submit(payload(chat_workspace))
    final = completed(assistant, submitted["job_id"])
    assert final["status"] == "failed"
    assert final["proposal"] is None
    assert native.calls == []
    assert len(router.calls) == 1


def test_registered_credential_is_removed_from_context_errors_and_history(settings_factory, chat_workspace):
    assistant, _, router = settings_factory(router=ProviderRunner("openrouter", error=ValueError("Synthetic failure " + SECRET)))
    configure_router(assistant)
    request = payload(chat_workspace)
    request["selection"]["selected_text"] = "An accidental pasted value: " + SECRET
    submitted = assistant.submit(request)
    final = completed(assistant, submitted["job_id"])
    assert final["status"] == "failed"
    assert_no_secret(router.calls, final, assistant.history(), assistant.settings())
    assert SECRET not in (chat_workspace / "data/workbench-chat/state.json").read_text()


@pytest.mark.parametrize("filename", ["settings.json", "credentials.json"])
def test_malformed_saved_settings_fail_closed_without_provider_fallback(settings_factory, chat_workspace, filename):
    directory = chat_workspace / "data/workbench-chat"
    directory.mkdir(parents=True)
    (directory / filename).write_text("{invalid", encoding="utf-8")
    assistant, native, router = settings_factory()
    assert assistant.status()["available"] is False
    with pytest.raises(RuntimeError):
        assistant.submit(payload(chat_workspace))
    assert native.calls == router.calls == []


@pytest.fixture
def router_transport(monkeypatch, chat_workspace):
    from astra_harness import workbench_chat as chat
    from astra_harness.workbench_settings import WorkbenchSettings
    settings = WorkbenchSettings(WorkbenchReader(chat_workspace))
    settings.update({"provider": "openrouter", "openrouter": {"model": "example/openrouter", "secret": SECRET}})
    calls = []
    response = {"model": "example/openrouter", "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"answer": "A bounded offline explanation.", "proposal": None})}}]}

    def request(path, body=None, secret=None, cancel_event=None):
        calls.append({"path": path, "body": copy.deepcopy(body), "secret": secret})
        if body is None:
            return {"data": [
                {"id": "example/openrouter", "name": "Synthetic model", "supported_parameters": ["structured_outputs"]},
                {"id": "openrouter/auto", "name": "Refused automatic choice"},
                {"id": "example/web:online", "name": "Refused web choice"},
            ]}
        return copy.deepcopy(response)

    monkeypatch.setattr(chat, "openrouter_request", request)
    return chat.OpenRouterRunner(settings), calls, response


def test_openrouter_runner_lists_without_credentials_and_sends_exact_model_no_tools(router_transport):
    runner, calls, _ = router_transport
    assert runner.status()["available"] is True
    assert calls == []
    assert runner.models()["models"] == [{"id": "example/openrouter", "name": "Synthetic model"}]
    assert calls[0] == {"path": "/api/v1/models", "body": None, "secret": None}
    result = runner({"message": "Explain the visual architecture"}, threading.Event())
    assert result["model"] == "example/openrouter"
    sent = calls[1]
    assert sent["path"] == "/api/v1/chat/completions"
    assert sent["secret"] == SECRET
    assert sent["body"]["model"] == "example/openrouter"
    assert sent["body"]["provider"] == {"allow_fallbacks": False, "require_parameters": True}
    assert sent["body"]["tools"] == sent["body"]["plugins"] == []
    assert sent["body"]["stream"] is False
    assert sent["body"]["max_tokens"] == 2048
    assert sent["body"]["response_format"]["json_schema"]["strict"] is True
    assert SECRET not in json.dumps(sent["body"])


@pytest.mark.parametrize("change", ["model", "truncated", "tools", "function", "malformed", "array", "multiple"])
def test_openrouter_runner_refuses_substitution_tools_and_malformed_output_without_retry(router_transport, change):
    runner, calls, response = router_transport
    if change == "model":
        response["model"] = "example/different-model"
    elif change == "truncated":
        response["choices"][0]["finish_reason"] = "length"
    elif change == "tools":
        response["choices"][0]["message"]["tool_calls"] = [{"id": "synthetic-unused-tool"}]
    elif change == "function":
        response["choices"][0]["message"]["function_call"] = {"name": "synthetic-unused-function"}
    elif change == "malformed":
        response["choices"][0]["message"]["content"] = "{invalid"
    elif change == "array":
        response["choices"][0]["message"]["content"] = "[]"
    else:
        response["choices"].append(copy.deepcopy(response["choices"][0]))
    with pytest.raises(ValueError):
        runner({"message": "Explain"}, threading.Event())
    assert len(calls) == 1


@pytest.fixture
def fake_https(monkeypatch):
    from astra_harness import workbench_chat as chat
    connections = []
    fixture = {"status": 200, "body": b'{"data":[]}', "error": None}

    class Connection:
        def __init__(self, host, timeout):
            self.host = host
            self.timeout = timeout
            self.sock = None
            self.calls = []
            self.closed = False
            self.response_closed = False
            connections.append(self)

        def connect(self):
            self.calls.append("connect")

        def request(self, method, path, body, headers):
            self.calls.append({"method": method, "path": path, "body": body, "headers": headers})

        def getresponse(self):
            if fixture["error"]:
                raise fixture["error"]
            stream = io.BytesIO(fixture["body"])
            owner = self

            class Response:
                status = fixture["status"]

                def read(self, limit):
                    return stream.read(limit)

                def close(self):
                    owner.response_closed = True
                    stream.close()

            return Response()

        def close(self):
            self.closed = True

    monkeypatch.setattr(chat.http.client, "HTTPSConnection", Connection)
    return chat, connections, fixture


def test_openrouter_transport_uses_fixed_tls_host_and_ignores_endpoint_environment(fake_https, monkeypatch):
    chat, connections, _ = fake_https
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://unrequested.example")
    monkeypatch.setenv("HTTPS_PROXY", "https://unrequested.example")
    assert chat.openrouter_request("/api/v1/models") == {"data": []}
    connection = connections[0]
    assert connection.host == "openrouter.ai"
    assert connection.calls[1]["path"] == "/api/v1/models"
    assert "Authorization" not in connection.calls[1]["headers"]
    assert connection.closed is True
    assert connection.response_closed is True


@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_openrouter_transport_refuses_redirects_and_http_errors_without_retry(fake_https, status):
    chat, connections, fixture = fake_https
    fixture.update(status=status, body=("Do not echo " + SECRET).encode())
    with pytest.raises(RuntimeError) as error:
        chat.openrouter_request("/api/v1/chat/completions", body={"model": "example/model"}, secret=SECRET)
    assert SECRET not in str(error.value)
    assert len(connections) == 1
    assert len(connections[0].calls) == 2
    assert connections[0].closed is True


def test_openrouter_transport_connection_ambiguity_never_retries(fake_https):
    chat, connections, fixture = fake_https
    fixture["error"] = OSError("Synthetic lost response")
    with pytest.raises(chat.GenerationUncertain):
        chat.openrouter_request("/api/v1/chat/completions", body={"model": "example/model"}, secret=SECRET)
    assert len(connections) == 1
    assert connections[0].closed is True


def test_openrouter_transport_cancelled_before_sending_does_not_connect(fake_https):
    chat, connections, _ = fake_https
    cancellation = threading.Event()
    cancellation.set()
    with pytest.raises(chat.GenerationCancelled):
        chat.openrouter_request("/api/v1/chat/completions", body={}, secret=SECRET, cancel_event=cancellation)
    assert connections == []


def test_openrouter_transport_does_not_accept_arbitrary_urls(fake_https):
    chat, connections, _ = fake_https
    with pytest.raises(ValueError):
        chat.openrouter_request("https://unrequested.example/api/v1/models")
    assert connections == []


@pytest.fixture
def native_catalog(monkeypatch):
    from astra_harness import workbench_chat as chat
    processes = []
    catalog = [{"model": "example/default", "isDefault": True, "supportedReasoningEfforts": [{"reasoningEffort": "low"}]},
               {"model": "example/custom", "isDefault": False, "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]

    class Process:
        def __init__(self, directory, cancel_event):
            self.calls = []
            self.completed = False
            self.closed = False
            self.final = json.dumps({"answer": "Synthetic native answer.", "proposal": None})
            processes.append(self)

        def request(self, method, parameters):
            self.calls.append((method, parameters))
            if method == "account/read":
                return {"account": {"type": "chatgpt"}}
            if method == "model/list":
                return {"data": copy.deepcopy(catalog), "nextCursor": None}
            if method == "thread/start":
                return {"thread": {"id": "offline-thread"}, "model": parameters["model"], "modelProvider": "openai"}
            if method == "turn/start":
                self.completed = True
            return {}

        def _send(self, message):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(chat, "_NativeProcess", Process)
    return chat, processes, catalog


def test_native_model_listing_never_starts_a_generation(native_catalog):
    chat, processes, _ = native_catalog
    result = chat.NativeCodexRunner().models()
    assert [model["id"] for model in result["models"]] == ["example/default", "example/custom"]
    assert result["default"] == "example/default"
    assert {method for method, _ in processes[0].calls} == {"initialize", "account/read", "model/list"}
    assert processes[0].closed is True


def test_native_custom_model_is_exactly_advertised_pinned_and_never_falls_back(native_catalog):
    chat, processes, _ = native_catalog
    pinned = []
    result = chat.NativeCodexRunner("example/custom", on_model=pinned.append)({"message": "Explain"}, threading.Event())
    assert result["model"] == "example/custom"
    assert pinned == ["example/custom"]
    calls = dict(processes[0].calls)
    assert calls["thread/start"]["model"] == calls["turn/start"]["model"] == "example/custom"
    assert calls["thread/start"]["allowProviderModelFallback"] is False
    assert calls["thread/start"]["dynamicTools"] == []
    assert processes[0].closed is True


def test_native_unadvertised_custom_model_fails_without_default_fallback(native_catalog):
    chat, processes, _ = native_catalog
    with pytest.raises(RuntimeError):
        chat.NativeCodexRunner("example/unadvertised")({"message": "Explain"}, threading.Event())
    assert "thread/start" not in dict(processes[0].calls)
    assert processes[0].closed is True


@pytest.fixture
def settings_server(chat_workspace):
    from astra_harness.workbench import create_server
    native, router = ProviderRunner(), ProviderRunner("openrouter")
    server = create_server(chat_workspace, port=0, assistant_runner=native, openrouter_runner=router)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server, native, router
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def http_request(server, path, *, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        connection.close()


def http_headers(server):
    status, _, body = http_request(server, "/api/assistant/settings")
    assert status == 200
    return {"Origin": "http://127.0.0.1:" + str(server.server_port), "Content-Type": "application/json",
            "X-Workbench-CSRF": json.loads(body)["csrf"]}


def test_http_settings_roundtrip_and_model_listing_do_not_generate_or_echo_secret(settings_server):
    server, native, router = settings_server
    status, headers, body = http_request(server, "/api/assistant/settings", method="POST", headers=http_headers(server),
        body=json.dumps({"provider": "openrouter", "openrouter": {"model": "example/openrouter", "secret": SECRET}}))
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert json.loads(body)["openrouter"]["credential_present"] is True
    assert SECRET not in body
    for path in ("/api/assistant/settings", "/api/assistant/status", "/api/assistant/history",
                 "/api/assistant/models/chatgpt", "/api/assistant/models/openrouter"):
        status, _, body = http_request(server, path)
        assert status == 200
        assert SECRET not in body
    assert native.calls == router.calls == []


@pytest.mark.parametrize("missing", ["Origin", "X-Workbench-CSRF"])
def test_http_settings_requires_same_origin_and_csrf(settings_server, missing):
    server, native, router = settings_server
    headers = http_headers(server)
    headers.pop(missing)
    status, _, _ = http_request(server, "/api/assistant/settings", method="POST", headers=headers,
                               body=json.dumps({"provider": "openrouter"}))
    assert status == 403
    assert server.assistant.settings()["provider"] == "chatgpt"
    assert native.calls == router.calls == []


@pytest.mark.parametrize("change", [{"Origin": "https://untrusted.example"}, {"X-Workbench-CSRF": "invalid"},
                                     {"Content-Type": "text/plain"}])
def test_http_settings_rejects_untrusted_headers(settings_server, change):
    server, native, router = settings_server
    headers = {**http_headers(server), **change}
    status, _, _ = http_request(server, "/api/assistant/settings", method="POST", headers=headers,
                               body=json.dumps({"provider": "openrouter"}))
    assert status in {400, 403, 415}
    assert server.assistant.settings()["provider"] == "chatgpt"
    assert native.calls == router.calls == []


def test_http_active_settings_change_returns_conflict(settings_server, chat_workspace):
    server, native, router = settings_server
    native.release.clear()
    submitted = server.assistant.submit(payload(chat_workspace))
    assert native.started.wait(1)
    status, _, _ = http_request(server, "/api/assistant/settings", method="POST", headers=http_headers(server),
                               body=json.dumps({"provider": "openrouter"}))
    assert status == 409
    native.release.set()
    assert completed(server.assistant, submitted["job_id"])["status"] == "completed"
    assert router.calls == []
