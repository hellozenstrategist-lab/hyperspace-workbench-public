"""Read-only workbench behavior against synthetic runs and source files."""
from datetime import datetime
import hashlib
import http.client
import json
import sqlite3
import threading
import time
import uuid

import pytest

from astra_harness.knowledge_store import DDL
from astra_harness.workbench_catalog import get_catalog


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


@pytest.fixture
def workspace(tmp_path):
    package = tmp_path / "astra_harness"
    package.mkdir()
    (package / "research.py").write_text(
        'DEFAULT_MODELS = {"head": "example/head", "worker": "example/worker", "qc": "example/qc"}\n'
        'DEFAULT_LIMITS = {"max_rounds": 8, "max_requests": 160}\n'
        'raise AssertionError("Source must never execute")\n', encoding="utf-8")
    (package / "cli.py").write_text('"""Synthetic CLI source."""\n', encoding="utf-8")
    mission = tmp_path / "runs" / "mission-example"
    run_id = "d5855a41-bff0-42f6-9a73-19a68a3f3b82"
    manifest = {"run_id": run_id, "provider": "codex", "model": "example/model",
                "worker_count": 2, "created_at": "2025-01-01T10:00:00Z", "mode": "hybrid"}
    write_json(mission / "manifest.json", manifest)
    write_json(mission / "report.json", {"run_id": run_id, "status": "completed", "core_pass": True})
    database = mission / "data" / "knowledge.sqlite"
    database.parent.mkdir()
    connection = sqlite3.connect(database)
    connection.executescript(DDL)
    connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?)",
                       (run_id, json.dumps(manifest), "completed", manifest["created_at"]))
    connection.execute("INSERT INTO workers(run_id,agent,state,task) VALUES (?, ?, ?, ?)",
                       (run_id, "agent-a", "completed", "Compare the supplied designs"))
    event = {"event_id": "event-example", "run_id": run_id, "author_agent": "agent-a",
             "type": "claim", "claim": "The supplied design has two independent inputs",
             "scope_path": ["root", "design"], "verification_status": "observed", "evidence_refs": []}
    connection.execute("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
                       (event["event_id"], run_id, "agent-a", "claim", json.dumps(event), "e" * 64))
    connection.execute("INSERT INTO coordinates VALUES (?, ?, ?, ?, ?)",
                       (event["event_id"], "[0.2,0.3]", "hierarchy-v1", 1, manifest["created_at"]))
    connection.execute("INSERT INTO ledger(run_id,kind,at,data,prev_hash,hash) VALUES (?, ?, ?, ?, ?, ?)",
                       (run_id, "run_state", manifest["created_at"], '{"state":"completed"}', "", "f" * 64))
    connection.commit()
    connection.close()
    research = tmp_path / "research-runs" / "research-example"
    state = {"run_id": "research-example-id", "status": "paused", "round": 2, "phase": "review",
             "reason": "Round budget reached", "config": {"models": {"head": "example/head",
             "worker": "example/worker", "qc": "example/qc"}, "workers": 3},
             "criteria": [{"id": "C1", "requirement": "Compare supplied designs", "basis": "analysis"}],
             "history": [{"round": 1, "verdict": "revise", "summary": "Clarify the second design"}]}
    write_json(research / "research.json", state)
    write_json(research / "research-report.json", {
        **state, "accepted_by_qc": False,
        "usage": {"reported_total_tokens": None, "known_reported_total_tokens": 123,
                  "unresolved_requests": 1, "usage_complete": False},
    })
    artifact = {"origin": "supplied", "label": "Design brief", "content": "A generic comparison fixture"}
    encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    artifact_id = hashlib.sha256(encoded.encode()).hexdigest()
    write_json(research / "artifacts" / (artifact_id + ".json"), artifact)
    return {"root": tmp_path, "mission": mission, "research": research, "artifact_id": artifact_id}


def test_catalog_has_connected_scenes_and_explicit_research_boundary(workspace):
    catalog = get_catalog(workspace["root"])
    components = {component["id"]: component for component in catalog["components"]}
    assert len(components) == len(catalog["components"])
    assert catalog["defaults"] == {"research": {"head": "example/head", "worker": "example/worker",
                                                "qc": "example/qc"},
                                   "limits": {"max_rounds": 8, "max_requests": 160}}
    assert {scene["id"] for scene in catalog["scenes"]} == {"mission", "research"}
    for scene in catalog["scenes"]:
        identifiers = {node["id"] for node in scene["nodes"]}
        assert identifiers <= components.keys()
        assert all(edge["source"] in identifiers and edge["target"] in identifiers for edge in scene["edges"])
        assert all(0 <= node["x"] <= 1070 and 0 <= node["y"] <= 568 for node in scene["nodes"])
        assert identifiers == {endpoint for edge in scene["edges"] for endpoint in (edge["source"], edge["target"])}
    research = next(scene for scene in catalog["scenes"] if scene["id"] == "research")
    assert "hyperspace-index" not in {node["id"] for node in research["nodes"]}
    assert "knowledge-store" not in {node["id"] for node in research["nodes"]}
    assert all(component["inputs"] and component["outputs"] and component["invariants"]
               for component in components.values())


@pytest.mark.parametrize("source", [
    'DEFAULT_MODELS = dict(head="unexecuted")\n',
    'DEFAULT_MODELS = {"head": invalid_call()}\n',
    'DEFAULT_MODELS = {\n',
])
def test_catalog_rejects_executable_or_malformed_defaults(tmp_path, source):
    package = tmp_path / "astra_harness"
    package.mkdir()
    (package / "research.py").write_text(source, encoding="utf-8")
    assert get_catalog(tmp_path)["defaults"] == {"research": {}, "limits": {}}


def test_catalog_never_follows_source_symlink(tmp_path):
    package = tmp_path / "astra_harness"
    package.mkdir()
    outside = tmp_path / "private.py"
    outside.write_text('DEFAULT_MODELS = {"head": "private-value"}', encoding="utf-8")
    (package / "research.py").symlink_to(outside)
    assert get_catalog(tmp_path)["defaults"] == {"research": {}, "limits": {}}


def test_catalog_never_follows_package_symlink(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "research.py").write_text('DEFAULT_MODELS = {"head": "private-value"}', encoding="utf-8")
    (tmp_path / "astra_harness").symlink_to(actual, target_is_directory=True)
    assert get_catalog(tmp_path)["defaults"] == {"research": {}, "limits": {}}


@pytest.fixture
def reader(workspace):
    from astra_harness.workbench import WorkbenchReader
    return WorkbenchReader(workspace["root"])


@pytest.fixture
def server(workspace):
    from astra_harness.workbench import create_server
    instance = create_server(workspace["root"], port=0, host="127.0.0.1")
    thread = threading.Thread(target=lambda: instance.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=3)


def request(server, path, *, method="GET", headers=None, body=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request(method, path, headers=headers or {}, body=body)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read().decode("utf-8")
    finally:
        connection.close()


def test_source_reader_is_allowlisted_and_does_not_execute_code(reader):
    source = reader.source("cli")
    assert source["path"] == "astra_harness/cli.py"
    assert source["content"] == '"""Synthetic CLI source."""\n'
    assert source["truncated"] is False
    assert 'raise AssertionError' in reader.source("research-plan")["content"]


@pytest.mark.parametrize("identifier", ["../private", "/etc/passwd", "cli.py", "..%2fprivate"])
def test_source_reader_refuses_arbitrary_paths(reader, identifier):
    with pytest.raises((KeyError, ValueError)):
        reader.source(identifier)


def test_source_reader_refuses_allowlisted_symlink(reader, workspace):
    source = workspace["root"] / "astra_harness" / "cli.py"
    source.unlink()
    private = workspace["root"] / "private.txt"
    private.write_text("SYNTHETIC_PRIVATE_SOURCE", encoding="utf-8")
    source.symlink_to(private)
    with pytest.raises((KeyError, ValueError, FileNotFoundError)):
        reader.source("cli")


def test_source_response_is_bounded(reader, workspace):
    source = workspace["root"] / "astra_harness" / "cli.py"
    source.write_text("visible source\n" * 30_000, encoding="utf-8")
    result = reader.source("cli")
    assert result["truncated"] is True
    assert 0 < len(result["content"].encode()) <= 140_000


def run_named(reader, name):
    return next(run for run in reader.list_runs()["runs"] if run["name"] == name)


def test_overview_distinguishes_saved_modes_and_source_defaults(reader):
    overview = reader.overview()
    assert overview["counts"]["runs"] == 2
    assert overview["counts"]["mission_runs"] == 1
    assert overview["counts"]["research_runs"] == 1
    assert overview["defaults"]["research"]["head"] == "example/head"
    mission = run_named(reader, "mission-example")
    research = run_named(reader, "research-example")
    assert mission["kind"] == "mission"
    assert mission["status"] == "completed"
    assert mission["events"] == 1
    assert mission["workers"] == 2
    assert research["kind"] == "research"
    assert research["status"] == "paused"
    assert research["rounds"] == 2
    assert research["model"] == "example/head"


def test_saved_graph_and_phase_details_are_read_without_writes(reader, workspace):
    before = tree_bytes(workspace["root"])
    mission = reader.run_detail(run_named(reader, "mission-example")["id"])
    assert mission["nodes"][0]["id"] == "event-example"
    assert mission["nodes"][0]["position"] == [0.2, 0.3]
    assert mission["workers"][0]["state"] == "completed"
    assert mission["events"][0]["kind"] == "run_state"
    research = reader.run_detail(run_named(reader, "research-example")["id"])
    assert research["phases"][0]["name"] == "review"
    assert research["metrics"]["usage"]["reported_total_tokens"] is None
    assert research["metrics"]["usage"]["usage_complete"] is False
    assert research["nodes"] == []
    assert research["deliveries"] == []
    for artifact in research["artifacts"]:
        reader.artifact(research["id"], artifact["id"])
    reader.overview()
    assert tree_bytes(workspace["root"]) == before


def test_reader_includes_committed_wal_data_without_touching_original_database(reader, workspace):
    database = workspace["mission"] / "data" / "knowledge.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("INSERT INTO workers(run_id,agent,state) VALUES (?, ?, ?)",
                           ("d5855a41-bff0-42f6-9a73-19a68a3f3b82", "agent-b", "running"))
        connection.commit()
        before = tree_bytes(workspace["root"])
        detail = reader.run_detail(run_named(reader, "mission-example")["id"])
        assert {worker["id"] for worker in detail["workers"]} == {"agent-a", "agent-b"}
        assert tree_bytes(workspace["root"]) == before
    finally:
        connection.close()


def test_artifacts_are_scoped_to_their_saved_run(reader, workspace):
    research = reader.run_detail(run_named(reader, "research-example")["id"])
    artifact = next(item for item in research["artifacts"]
                    if item["name"] == "artifacts/" + workspace["artifact_id"] + ".json")
    result = reader.artifact(research["id"], artifact["id"])
    assert json.loads(result["content"])["content"] == "A generic comparison fixture"
    assert result["truncated"] is False
    mission = run_named(reader, "mission-example")
    with pytest.raises(KeyError):
        reader.artifact(mission["id"], artifact["id"])
    with pytest.raises(ValueError):
        reader.artifact(research["id"], "../../private.txt")


def test_credentials_are_redacted_from_configuration_artifacts_and_source(reader, workspace):
    secret_key = "sk-or-v1-" + "a" * 64
    bearer = "SYNTHETIC_BEARER_CREDENTIAL"
    private_value = "SYNTHETIC_PRIVATE_CONFIGURATION"
    state_path = workspace["research"] / "research.json"
    state = json.loads(state_path.read_text())
    state["config"]["api_key"] = private_value
    state["config"]["nested"] = {"Authorization": "Bearer " + bearer}
    write_json(state_path, state)
    write_json(workspace["research"] / "artifacts" / "credentials.json",
               {"origin": "supplied", "content": {"api_key": private_value,
                "headers": {"Authorization": "Bearer " + bearer}, "note": secret_key}})
    (workspace["root"] / "astra_harness" / "cli.py").write_text(
        'API_KEY = "' + secret_key + '"\n', encoding="utf-8")
    detail = reader.run_detail(run_named(reader, "research-example")["id"])
    artifact = next(item for item in detail["artifacts"] if item["name"] == "artifacts/credentials.json")
    responses = [reader.overview(), detail, reader.artifact(detail["id"], artifact["id"]), reader.source("cli")]
    encoded = json.dumps(responses)
    assert all(secret not in encoded for secret in (secret_key, bearer, private_value))
    assert "example/head" in encoded
    assert "[redacted]" in encoded


def test_malformed_metadata_and_database_produce_warnings_without_hiding_other_runs(reader, workspace):
    broken = workspace["root"] / "runs" / "malformed-example"
    broken.mkdir()
    (broken / "manifest.json").write_text("{invalid json", encoding="utf-8")
    (broken / "data").mkdir()
    (broken / "data" / "knowledge.sqlite").write_bytes(b"not a database")
    listing = reader.list_runs()
    assert {"mission-example", "research-example"} <= {run["name"] for run in listing["runs"]}
    assert listing["warnings"]
    malformed = run_named(reader, "malformed-example")
    assert reader.run_detail(malformed["id"])["warnings"]


def test_run_and_artifact_symlinks_are_not_discovered(reader, workspace):
    (workspace["root"] / "runs" / "linked-research").symlink_to(workspace["research"], target_is_directory=True)
    private = workspace["root"] / "private.json"
    write_json(private, {"secret": "SYNTHETIC_PRIVATE_ARTIFACT"})
    (workspace["research"] / "artifacts" / "linked.json").symlink_to(private)
    listing = reader.list_runs()
    assert len(listing["runs"]) == 2
    research = reader.run_detail(run_named(reader, "research-example")["id"])
    assert all(item["name"] != "artifacts/linked.json" for item in research["artifacts"])


def test_uncertain_requests_and_numbered_attempts_remain_visible(reader, workspace):
    runtime = {"model": "example/head", "workers": {"agent-a": {"status": "failed",
               "token_totals": {"totalTokens": 123}, "requests": [
                   {"request_id": "request-one", "status": "completed", "started_at": "2025-01-01T10:00:00Z"},
                   {"request_id": "request-two", "status": "uncertain", "started_at": "2025-01-01T10:00:01Z"},
               ]}}}
    write_json(workspace["research"] / "rounds" / "002" / "review" / "attempt-001" / "runtime_state.json", runtime)
    detail = reader.run_detail(run_named(reader, "research-example")["id"])
    assert detail["metrics"]["requests"] == 2
    assert {event["kind"] for event in detail["events"]} == {"request_completed", "request_uncertain"}
    assert detail["phases"][0]["name"] == "review"
    assert detail["phases"][0]["round"] == 2
    assert detail["phases"][0]["status"] == "failed"
    assert detail["metrics"]["usage"]["usage_complete"] is False


def test_research_roles_and_phases_preserve_workflow_and_chronological_order(reader, workspace):
    phase_records = [
        ("plan", "example/head", ["agent-a"], 100),
        ("work", "example/worker", ["agent-a", "agent-b", "agent-c"], 200),
        ("synthesize", "example/head", ["agent-a"], 300),
        ("review", "example/qc", ["agent-a"], 400),
    ]
    for phase, model, agents, started in phase_records:
        workers = {agent: {"status": "completed", "requests": [
            {"request_id": phase + ":" + agent, "status": "completed", "started_at": started + index}
        ]} for index, agent in enumerate(agents)}
        write_json(workspace["research"] / "rounds" / "001" / phase / "attempt-001" / "runtime_state.json",
                   {"model": model, "workers": workers})
    summary = run_named(reader, "research-example")
    detail = reader.run_detail(summary["id"])
    assert summary["workers"] == 5
    assert {worker["id"] for worker in detail["workers"]} == {
        "head", "worker/agent-a", "worker/agent-b", "worker/agent-c", "qc"}
    assert {worker["model"] for worker in detail["workers"] if worker["id"].startswith("worker/")} == {"example/worker"}
    assert [phase["name"] for phase in detail["phases"]] == ["plan", "work", "synthesize", "review"]
    assert [datetime.fromisoformat(event["at"].replace("Z", "+00:00")).timestamp()
            for event in detail["events"]] == [100, 200, 201, 202, 300, 400]


@pytest.mark.parametrize("coordinate", [None, "not json", "[1]", "[0.1,\"bad\"]", "[NaN,0]"])
def test_missing_or_invalid_saved_coordinates_are_not_fabricated_as_origin(reader, workspace, coordinate):
    connection = sqlite3.connect(workspace["mission"] / "data" / "knowledge.sqlite")
    try:
        if coordinate is None:
            connection.execute("DELETE FROM coordinates")
        else:
            connection.execute("UPDATE coordinates SET vector = ?", (coordinate,))
        connection.commit()
    finally:
        connection.close()
    detail = reader.run_detail(run_named(reader, "mission-example")["id"])
    assert detail["nodes"][0]["position"] is None


def test_http_allows_local_read_requests_and_same_origin(server):
    origin = "http://127.0.0.1:" + str(server.server_address[1])
    status, headers, body = request(server, "/api/runs", headers={"Origin": origin})
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert "runs" in json.loads(body)
    assert "Access-Control-Allow-Origin" not in headers


@pytest.mark.parametrize("headers", [
    {"Host": "untrusted.example"},
    {"Host": "127.0.0.1.untrusted.example"},
    {"Origin": "https://untrusted.example"},
    {"Origin": "null"},
])
def test_http_rejects_untrusted_authorities(server, headers):
    status, _, _ = request(server, "/api/runs", headers=headers)
    assert status == 403


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_http_rejects_mutating_methods_without_changing_runs(server, workspace, method):
    before = tree_bytes(workspace["root"])
    status, _, _ = request(server, "/api/runs", method=method)
    assert status == 405
    assert tree_bytes(workspace["root"]) == before


@pytest.mark.parametrize("path", [
    "/api/components/..%2fprivate/source",
    "/api/components/cli.py/source",
    "/api/runs/..%2fprivate",
    "/api/runs/unknown/artifacts/..%2fprivate",
    "/%2e%2e/private.txt",
])
def test_http_refuses_paths_outside_allowlisted_resources(server, workspace, path):
    (workspace["root"] / "private.txt").write_text("SYNTHETIC_PRIVATE_FILE", encoding="utf-8")
    status, _, body = request(server, path)
    assert status in {400, 404}
    assert "SYNTHETIC_PRIVATE_FILE" not in body


@pytest.fixture
def assistant_server(workspace):
    from astra_harness.workbench import create_server

    class Runner:
        def __init__(self):
            self.calls = []

        def status(self):
            return {"authenticated": True, "available": True, "model": "example/default-model", "cli_version": "0.154.0"}

        def __call__(self, context, cancel_event):
            self.calls.append(context)
            return {"answer": "This is a visual workspace explanation.", "proposal": None}

    runner = Runner()
    instance = create_server(workspace["root"], port=0, host="127.0.0.1", assistant_runner=runner)
    thread = threading.Thread(target=lambda: instance.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield instance, runner
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=3)


def assistant_http_payload(workspace):
    scene = next(item for item in get_catalog(workspace["root"])["scenes"] if item["id"] == "mission")
    return {"request_id": str(uuid.uuid4()), "conversation_id": None, "message": "Explain the command line component.",
            "selection": {"scene": "mission", "component_ids": ["cli"], "run_id": None,
                          "event_id": None, "selected_text": ""},
            "workspace": {"version": 1, "name": "Hyperspace Workbench", "scene": "mission", "positions": {},
                          "notes": {}, "drafts": {"mission": {"nodes": [], "edges": scene["edges"], "parameters": {}}}}}


def assistant_http_headers(server):
    status, _, body = request(server, "/api/assistant/status")
    assert status == 200
    return {"Origin": "http://127.0.0.1:" + str(server.server_address[1]),
            "Content-Type": "application/json", "X-Workbench-CSRF": json.loads(body)["csrf"]}


def test_assistant_status_does_not_generate_and_json_post_returns_async_job(assistant_server, workspace):
    server, runner = assistant_server
    headers = assistant_http_headers(server)
    assert runner.calls == []
    status, _, body = request(server, "/api/assistant/messages", method="POST", headers=headers,
                              body=json.dumps(assistant_http_payload(workspace)))
    assert status in {200, 202}
    submitted = json.loads(body)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        status, _, body = request(server, "/api/assistant/jobs/" + submitted["job_id"])
        assert status == 200
        if json.loads(body)["status"] == "completed":
            break
        time.sleep(0.01)
    assert json.loads(body)["status"] == "completed"
    assert len(runner.calls) == 1


@pytest.mark.parametrize("change", ["missing_origin", "wrong_origin", "missing_csrf", "wrong_csrf"])
@pytest.mark.parametrize("endpoint", ["/api/assistant/messages", "/api/assistant/jobs/00000000-0000-4000-8000-000000000001/cancel"])
def test_assistant_post_requires_same_origin_and_csrf_without_generation(assistant_server, workspace, change, endpoint):
    server, runner = assistant_server
    headers = assistant_http_headers(server)
    if change == "missing_origin":
        headers.pop("Origin")
    elif change == "wrong_origin":
        headers["Origin"] = "https://untrusted.example"
    elif change == "missing_csrf":
        headers.pop("X-Workbench-CSRF")
    else:
        headers["X-Workbench-CSRF"] = "not-the-issued-token"
    status, _, _ = request(server, endpoint, method="POST", headers=headers,
                           body=json.dumps(assistant_http_payload(workspace)))
    assert status == 403
    assert runner.calls == []


@pytest.mark.parametrize("content_type,body", [("text/plain", "{}"), ("application/json", "{malformed"),
                                               ("application/json", "[]")])
def test_assistant_post_requires_valid_json_object(assistant_server, content_type, body):
    server, runner = assistant_server
    headers = assistant_http_headers(server)
    headers["Content-Type"] = content_type
    status, _, _ = request(server, "/api/assistant/messages", method="POST", headers=headers, body=body)
    assert status in {400, 415}
    assert runner.calls == []


@pytest.mark.parametrize("path", ["/api/runs", "/api/overview", "/api/components/cli/source"])
def test_assistant_permissions_do_not_enable_writes_to_inspection_routes(assistant_server, workspace, path):
    server, runner = assistant_server
    headers = assistant_http_headers(server)
    before = tree_bytes(workspace["root"])
    status, _, _ = request(server, path, method="POST", headers=headers, body="{}")
    assert status == 405
    assert runner.calls == []
    assert tree_bytes(workspace["root"]) == before
