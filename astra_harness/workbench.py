"""Local, read-only views of harness architecture and saved run evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
import tempfile
import time
from urllib.parse import unquote, urlsplit

from .workbench_catalog import get_catalog


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 128 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_DATABASE_BYTES = 64 * 1024 * 1024
MAX_ROWS = 500
MAX_RUNS = 500
MAX_FILES = 4000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
PHASE_ORDER = {"plan": 0, "work": 1, "falsify": 2, "synthesize": 3, "review": 4}
STATIC_FILES = {"/": "index.html", "/index.html": "index.html",
                "/app.js": "app.js", "/assistant.js": "assistant.js", "/styles.css": "styles.css", "/menu.css": "menu.css",
                "/settings.js": "settings.js", "/settings.css": "settings.css",
                "/texture-reference.png": "texture-reference.png", "/menu-burst.png": "menu-burst.png"}
ARTIFACT_NAMES = {"manifest.json", "mission.json", "research.json", "research-report.json",
                  "report.json", "audit.json", "concurrency.json", "snapshot.json",
                  "final_outputs.json", "result.json", "attempt.json", "submissions.json",
                  "reads.json", "inputs.json", "ledger.jsonl", "evidence_manifest.json"}
SECRET_FIELD = re.compile(
    r"(?i)^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|token|"
    r"authorization|authentication|auth|cookie|cookies|set[_-]?cookie|password|passwd|"
    r"secret|credentials?|client[_-]?secret|private[_-]?key|session[_-]?(?:id|key|token)|"
    r".*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret))$")
SECRET_ASSIGNMENT = re.compile(
    r'''(?im)(["']?\b(?:[\w-]*(?:api[ _-]?key|access[_-]?token|refresh[_-]?token|password|secret)|authorization|authentication|auth|cookies?|set-cookie|credentials?|token)["']?\s*[:=]\s*)(?:"{3}[\s\S]*?(?:"{3}|\Z)|'{3}[\s\S]*?(?:'{3}|\Z)|"(?:\\.|[^"\r\n])*"|'(?:\\.|[^'\r\n])*'|[^\s,;}\r\n]+)''')
SECRET_HEADER = re.compile(r"(?im)(\b(?:authorization|proxy-authorization|cookie|set-cookie)\s*:\s*)[^\r\n]+")
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{12,}|github_pat_[A-Za-z0-9_]{12,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|\Z)"),
)


def _text(value, limit=2000):
    return str(value)[:limit] if value is not None else ""


def _number(value):
    return value if type(value) is int and value >= 0 else 0


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _timestamp(value):
    try:
        if type(value) in {int, float} and math.isfinite(value):
            return datetime.fromtimestamp(value, timezone.utc).isoformat()
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc).isoformat() if parsed.tzinfo else ""
    except (OverflowError, OSError, ValueError):
        pass
    return ""


def _redact_text(value):
    value = SECRET_HEADER.sub(lambda match: match.group(1) + "[redacted]", value)
    value = SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + '"[redacted]"', value)
    for pattern in SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    value = re.sub(r"(?i)([?&](?:api_key|access_token|token|key|secret)=)[^&#\s]+", r"\1[redacted]", value)
    return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", value)


def redact(value, depth=0):
    if depth > 16:
        return "[depth limit]"
    if isinstance(value, dict):
        return {_redact_text(_text(key, 200)): "[redacted]" if SECRET_FIELD.match(str(key))
                else redact(item, depth + 1) for key, item in list(value.items())[:500]}
    if isinstance(value, list):
        return [redact(item, depth + 1) for item in value[:500]]
    if isinstance(value, str):
        return _redact_text(value[:MAX_TEXT_BYTES])
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class WorkbenchReader:
    """Inspect only configured run roots and catalogued source files."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError("Workbench root must be an existing directory")

    def _safe(self, path):
        path = Path(path)
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            raise ValueError("Path is outside the workbench root") from None
        if any(part in {".", ".."} for part in relative.parts):
            raise ValueError("Invalid workbench path")
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symbolic links are not available in the workbench")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Path is outside the workbench root")
        return path

    def _read(self, path, limit=MAX_METADATA_BYTES):
        path = self._safe(path)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Only regular files are available")
            content = stream.read(limit + 1)
        return content[:limit], len(content) > limit

    def _json(self, path, warnings):
        try:
            content, truncated = self._read(path)
            if truncated:
                raise ValueError("Metadata exceeds inspection limit")
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("Metadata must be a JSON object")
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, RecursionError):
            warnings.append("Unable to read complete metadata: " + str(path.relative_to(self.root)))
            return {}

    def _walk(self, base, depth=4):
        try:
            self._safe(base)
        except ValueError:
            return
        visited = 0
        for directory, children, files in os.walk(base, followlinks=False):
            relative = Path(directory).relative_to(base)
            children[:] = sorted(child for child in children if not child.startswith(".")
                                 and not (Path(directory) / child).is_symlink()) if len(relative.parts) < depth else []
            for name in sorted(files):
                visited += 1
                if visited > MAX_FILES:
                    return
                path = Path(directory) / name
                if not name.startswith(".") and not path.is_symlink():
                    yield path

    def _run_paths(self):
        found = {}
        for base in (self.root / "runs", self.root / "research-runs", self.root / "data" / "research"):
            for path in self._walk(base):
                if path.name in {"research.json", "research-report.json", "mission.json", "manifest.json"}:
                    directory = path.parent
                    relative = directory.relative_to(self.root)
                    if "rounds" in relative.parts or "runtime" in relative.parts:
                        continue
                    kind = "research" if path.name.startswith("research") else "mission"
                    if found.get(directory) != "research":
                        found[directory] = kind
                    if len(found) >= MAX_RUNS:
                        break
        return sorted(found.items(), key=lambda item: str(item[0]))[:MAX_RUNS]

    def _id(self, path):
        return hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:24]

    def _run(self, ident):
        if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{24}", ident):
            raise ValueError("Invalid run ID")
        for path, kind in self._run_paths():
            if self._id(path) == ident:
                return path, kind
        raise KeyError("Run not found")

    @contextmanager
    def _database(self, path):
        database = self._safe(path / "data" / "knowledge.sqlite")
        wal = database.with_name(database.name + "-wal")

        def fingerprint(original):
            try:
                metadata = self._safe(original).stat()
                return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns
            except FileNotFoundError:
                return None

        before = [fingerprint(original) for original in (database, wal)]
        with tempfile.TemporaryDirectory(prefix="hyperspace-workbench-") as temporary:
            snapshot = Path(temporary) / "knowledge.sqlite"
            for original, destination in ((database, snapshot),
                    (wal, snapshot.with_name(snapshot.name + "-wal"))):
                try:
                    content, truncated = self._read(original, MAX_DATABASE_BYTES)
                except FileNotFoundError:
                    if original == database:
                        raise
                    continue
                if truncated:
                    raise ValueError("Database snapshot exceeds inspection limit")
                destination.write_bytes(content)
            if before != [fingerprint(original) for original in (database, wal)]:
                raise ValueError("Database changed while taking its read-only snapshot")
            connection = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True, timeout=0.2)
            try:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                deadline = time.monotonic() + 2
                connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
                yield connection
            finally:
                connection.close()

    def _sql(self, connection, statement, warnings):
        try:
            return [dict(row) for row in connection.execute(statement)]
        except sqlite3.Error:
            warnings.append("Some saved database fields are unavailable")
            return []

    def _metadata(self, path, kind, warnings):
        if kind == "research":
            state = self._json(path / "research.json", warnings)
            report = self._json(path / "research-report.json", warnings)
            config = _mapping(state.get("config"))
            models = _mapping(config.get("models")) or _mapping(report.get("models"))
            usage = _mapping(report.get("usage"))
            workers = _number(config.get("workers")) or _number(report.get("workers"))
            summary = {"status": state.get("status", report.get("status", "unknown")),
                       "provider": "openrouter", "model": models.get("head", "unknown"),
                       "workers": workers + 2 if workers else 0, "rounds": _number(state.get("round", report.get("round"))),
                       "requests": _number(usage.get("requests_including_uncertain")),
                       "tokens": _number(usage.get("reported_total_tokens")) or _number(usage.get("known_reported_total_tokens"))}
        else:
            state = self._json(path / "manifest.json", warnings)
            report = self._json(path / "report.json", warnings)
            mission = self._json(path / "mission.json", warnings)
            config = _mapping(mission.get("config")) or mission or state
            summary = {"status": report.get("status", state.get("status", "unknown")),
                       "provider": state.get("provider", mission.get("provider", "codex")),
                       "model": state.get("model", mission.get("model", "unknown")),
                       "workers": _number(state.get("worker_count")) or len(_list(config.get("workers"))),
                       "rounds": 0, "requests": 0, "tokens": 0}
            if report.get("core_result_passed") is True or report.get("protocol_passed") is True:
                summary["status"] = "completed"
        summary.update(id=self._id(path), name=path.name, kind=kind,
                       relative_path=str(path.relative_to(self.root)), events=0, deliveries=0)
        modified = 0
        for name in ("research.json", "research-report.json", "manifest.json", "report.json", "mission.json",
                     "runtime_state.json", "data/knowledge.sqlite", "data/knowledge.sqlite-wal"):
            try:
                modified = max(modified, self._safe(path / name).stat().st_mtime)
            except (OSError, ValueError):
                pass
        summary["updated_at"] = datetime.fromtimestamp(modified, timezone.utc).isoformat()
        return summary, config, state, report

    def _runtime_data(self, path, kind, warnings):
        runtime_paths = [path / "runtime_state.json"] if kind == "mission" else [
            entry for entry in self._walk(path / "rounds", depth=5) if entry.name == "runtime_state.json"]
        requests, tokens, workers, phases, events = 0, 0, {}, {}, []
        worker_order = {}
        for runtime_path in runtime_paths[:MAX_ROWS]:
            state = self._json(runtime_path, warnings)
            model = state.get("model", "unknown")
            relative = runtime_path.relative_to(path).parts
            phase = relative[2] if len(relative) >= 4 else "run"
            round_number = int(relative[1]) if len(relative) >= 4 and relative[1].isdigit() else 0
            attempt = relative[3] if len(relative) >= 5 else ""
            phase_order = (round_number, PHASE_ORDER.get(phase, 99), attempt)
            statuses = []
            for agent, raw in _mapping(state.get("workers")).items():
                worker = _mapping(raw)
                statuses.append(worker.get("status", "unknown"))
                worker_id = "head" if phase in {"plan", "synthesize"} else "qc" if phase == "review" else "worker/" + agent if kind == "research" else agent
                if worker_id not in worker_order or phase_order >= worker_order[worker_id]:
                    workers[worker_id] = {"id": worker_id, "state": worker.get("status", "unknown"), "model": model}
                    worker_order[worker_id] = phase_order
                records = _list(worker.get("requests"))
                requests += len(records)
                totals = _mapping(worker.get("token_totals"))
                tokens += _number(totals.get("totalTokens", totals.get("total_tokens")))
                for index, record in enumerate(records):
                    record = _mapping(record)
                    events.append({"id": str(runtime_path.relative_to(path)) + ":" + agent + ":" + str(index),
                        "kind": "request_" + _text(record.get("status", "unknown"), 80),
                        "at": _timestamp(record.get("completed_at", record.get("failed_at", record.get("started_at", record.get("intent_at", record.get("created_at", "")))))),
                        "agent": worker_id, "summary": phase + " · " + _text(model, 100),
                        "data": {**record, "phase": phase, "round": round_number, "worker": agent}})
            if kind == "research" and statuses:
                phase_status = "completed" if all(status == "completed" for status in statuses) else "failed" if "failed" in statuses else "running" if any(status in {"active", "start_intent"} for status in statuses) else statuses[-1]
                phases[(round_number, phase)] = {"name": phase, "status": phase_status, "round": round_number, "model": model}
        events.sort(key=lambda event: (event["at"], event["id"]))
        ordered_phases = sorted(phases.values(), key=lambda phase: (phase["round"], PHASE_ORDER.get(phase["name"], 99), phase["name"]))
        return {"requests": requests, "tokens": tokens, "workers": list(workers.values()),
                "phases": ordered_phases, "events": events[-MAX_ROWS:], "event_count": len(events)}

    def _inspect(self, path, kind, detail=False):
        warnings = []
        summary, config, state, report = self._metadata(path, kind, warnings)
        runtime = self._runtime_data(path, kind, warnings)
        summary["requests"] = max(summary["requests"], runtime["requests"])
        summary["tokens"] = max(summary["tokens"], runtime["tokens"])
        workers, events, deliveries, nodes, edges = runtime["workers"], runtime["events"], [], [], []
        ledger_events = 0
        if kind == "mission":
            try:
                with self._database(path) as connection:
                    rows = self._sql(connection, "SELECT status FROM runs LIMIT 1", warnings)
                    if rows:
                        summary["status"] = rows[0]["status"]
                    for table, key in (("nodes", "events"), ("deliveries", "deliveries")):
                        rows = self._sql(connection, "SELECT COUNT(*) AS total FROM " + table, warnings)
                        if rows:
                            summary[key] = rows[0]["total"]
                    rows = self._sql(connection, "SELECT COUNT(*) AS total FROM ledger", warnings)
                    if rows:
                        ledger_events = rows[0]["total"]
                    rows = self._sql(connection, "SELECT data FROM ledger WHERE kind='token_usage' ORDER BY seq DESC LIMIT 500", warnings)
                    usage_by_agent = {}
                    for row in rows:
                        data = self._decode(row["data"])
                        agent = _text(data.get("agent"), 80)
                        total = _mapping(_mapping(data.get("usage")).get("total"))
                        if agent and agent not in usage_by_agent and "totalTokens" in total:
                            usage_by_agent[agent] = _number(total["totalTokens"])
                    summary["tokens"] = max(summary["tokens"], sum(usage_by_agent.values()))
                    rows = self._sql(connection, "SELECT agent,state FROM workers ORDER BY agent LIMIT 500", warnings)
                    if rows:
                        workers = [{"id": row["agent"], "state": row["state"], "model": summary["model"]} for row in rows]
                        summary["workers"] = max(summary["workers"], len(rows))
                    if detail:
                        rows = self._sql(connection, "SELECT seq,kind,at,data FROM ledger ORDER BY seq DESC LIMIT 500", warnings)
                        events = []
                        for row in reversed(rows):
                            data = self._decode(row["data"])
                            events.append({"id": str(row["seq"]), "kind": row["kind"], "at": row["at"],
                                "agent": data.get("agent", data.get("author", "coordinator")),
                                "summary": _text(data.get("reason", data.get("event_id", row["kind"]))), "data": data})
                        rows = self._sql(connection, "SELECT delivery_id,event_id,recipient,state,priority FROM deliveries ORDER BY rowid DESC LIMIT 500", warnings)
                        deliveries = [{"id": row["delivery_id"], "event_id": row["event_id"], "sender": "", "recipient": row["recipient"],
                                       "state": row["state"], "priority": row["priority"]} for row in rows]
                        rows = self._sql(connection, "SELECT nodes.event_id,nodes.author,nodes.type,nodes.body,coordinates.vector FROM nodes LEFT JOIN coordinates ON coordinates.node_id=nodes.event_id ORDER BY nodes.rowid DESC LIMIT 500", warnings)
                        for row in rows:
                            body = self._decode(row["body"])
                            coordinate = self._decode(row["vector"], [])
                            valid = (isinstance(coordinate, list) and len(coordinate) == 2
                                     and all(type(value) in {int, float} and math.isfinite(value) for value in coordinate)
                                     and sum(value * value for value in coordinate) < 1)
                            if not valid:
                                warnings.append("Some knowledge records have no valid saved Poincare coordinate")
                            nodes.append({"id": row["event_id"], "author": row["author"], "type": row["type"],
                                "claim": _text(body.get("claim")), "position": coordinate if valid else None,
                                "scope_path": _list(body.get("scope_path"))[:32]})
                        authors = {node["id"]: node["author"] for node in nodes}
                        for delivery in deliveries:
                            delivery["sender"] = authors.get(delivery["event_id"], "unknown")
                        edges = self._sql(connection, "SELECT source,target,relation FROM edges ORDER BY rowid DESC LIMIT 500", warnings)
            except FileNotFoundError:
                warnings.append("No saved knowledge database is available")
            except (OSError, ValueError, sqlite3.Error):
                warnings.append("The saved knowledge database is temporarily unreadable")
        else:
            summary["events"] = runtime["event_count"]
        if not detail:
            return redact(summary), warnings
        phases = runtime["phases"]
        if kind == "research" and not phases:
            phase = _text(state.get("phase", report.get("phase", "plan")), 80)
            phases = [{"name": phase, "status": summary["status"], "round": summary["rounds"], "model": summary["model"]}]
        if kind == "research":
            models = _mapping(config.get("models"))
            configured = [{"id": "head", "state": "unrecorded", "model": models.get("head", "unknown")}]
            configured += [{"id": "worker/agent-" + letter, "state": "unrecorded", "model": models.get("worker", "unknown")}
                           for letter in "abc"[:min(_number(config.get("workers")), 3)]]
            configured.append({"id": "qc", "state": "unrecorded", "model": models.get("qc", "unknown")})
            observed = {worker["id"]: worker for worker in workers}
            workers = [observed.pop(worker["id"], worker) for worker in configured] + list(observed.values())
        elif not workers:
            workers = [{"id": "agent-" + letter, "state": "unrecorded", "model": summary["model"]}
                       for letter in "abcde"[:min(summary["workers"], 5)]]
        result = {**summary, "config": config, "metrics": {"requests": summary["requests"], "tokens": summary["tokens"],
                    "events": summary["events"], "deliveries": summary["deliveries"],
                    "knowledge_records": summary["events"] if kind == "mission" else 0,
                    "ledger_events": ledger_events, "recorded_request_events": runtime["event_count"],
                    "analysis_workers": _number(config.get("workers")) if kind == "research" else summary["workers"],
                    "usage": _mapping(report.get("usage")), "concurrency": _mapping(report.get("concurrency")),
                    "semantic_correctness_verified": report.get("semantic_correctness_verified", False)},
                  "workers": workers, "events": events[-MAX_ROWS:], "deliveries": deliveries,
                  "nodes": nodes, "edges": edges, "phases": phases,
                  "artifacts": self._artifacts(path), "warnings": sorted(set(warnings))}
        return redact(result)

    def _decode(self, value, default=None):
        try:
            result = json.loads(value) if isinstance(value, str) and len(value) <= MAX_METADATA_BYTES else None
            return result if isinstance(result, type(default) if default is not None else dict) else default if default is not None else {}
        except (ValueError, RecursionError):
            return default if default is not None else {}

    def _artifacts(self, path):
        artifacts = []
        for candidate in self._walk(path, depth=5):
            relative = candidate.relative_to(path)
            content_addressed = (relative.parts[0] == "artifacts" and candidate.suffix in {".json", ".txt", ".md"})
            evidence = (relative.parts[:2] == ("data", "evidence") and re.fullmatch(r"[a-f0-9]{64}", candidate.name))
            known = candidate.name in ARTIFACT_NAMES and (len(relative.parts) == 1 or relative.parts[0] == "rounds")
            if not (known or content_addressed or evidence):
                continue
            try:
                metadata = self._safe(candidate).stat()
            except (OSError, ValueError):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                continue
            artifacts.append({"id": self._id(candidate), "name": str(relative), "size": metadata.st_size,
                              "kind": "evidence" if evidence else "json" if candidate.suffix == ".json" else "text"})
        return artifacts[:MAX_ROWS]

    def list_runs(self):
        runs, warnings = [], []
        paths = self._run_paths()
        if len(paths) >= MAX_RUNS:
            warnings.append("Run discovery reached its 500-run limit; counts describe the visible runs")
        for path, kind in paths:
            try:
                summary, notes = self._inspect(path, kind)
                runs.append(summary)
                warnings.extend(notes)
            except (OSError, ValueError, TypeError, OverflowError):
                warnings.append("Unable to inspect saved run: " + str(path.relative_to(self.root)))
        runs.sort(key=lambda row: (row["updated_at"], row["relative_path"]), reverse=True)
        return redact({"runs": runs, "warnings": sorted(set(warnings))[:MAX_ROWS]})

    def overview(self):
        listing = self.list_runs()
        catalog = get_catalog(self.root)
        return redact({"root": str(self.root), "generated_at": datetime.now(timezone.utc).isoformat(),
            "defaults": catalog.get("defaults", {}),
            "counts": {"runs": len(listing["runs"]), "mission_runs": sum(row["kind"] == "mission" for row in listing["runs"]),
                       "research_runs": sum(row["kind"] == "research" for row in listing["runs"]),
                       "components": len(catalog.get("components", []))},
            "catalog": catalog, "runs": listing["runs"], "diagnostics": listing["warnings"]})

    def run_detail(self, ident):
        path, kind = self._run(ident)
        return self._inspect(path, kind, detail=True)

    def artifact(self, ident, artifact_id):
        path, _ = self._run(ident)
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[a-f0-9]{24}", artifact_id):
            raise ValueError("Invalid artifact ID")
        artifact = next((item for item in self._artifacts(path) if item["id"] == artifact_id), None)
        if artifact is None:
            raise KeyError("Artifact not found")
        content, truncated = self._read(path / artifact["name"], MAX_TEXT_BYTES)
        text = content.decode("utf-8", errors="replace")
        if not truncated:
            try:
                text = json.dumps(redact(json.loads(text)), ensure_ascii=False, indent=2)
            except (ValueError, RecursionError):
                text = _redact_text(text)
        else:
            text = _redact_text(text)
        encoded = text.encode("utf-8")
        return {"id": artifact_id, "name": _redact_text(artifact["name"]), "content": encoded[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore"),
                "type": artifact["kind"], "truncated": truncated or len(encoded) > MAX_TEXT_BYTES}

    def source(self, component_id):
        if not isinstance(component_id, str) or not re.fullmatch(r"[a-z0-9-]{1,80}", component_id):
            raise ValueError("Invalid component ID")
        component = next((item for item in get_catalog(self.root).get("components", []) if item["id"] == component_id), None)
        if component is None:
            raise KeyError("Component not found")
        content, truncated = self._read(self.root / component["source"], MAX_TEXT_BYTES)
        text = _redact_text(content.decode("utf-8", errors="replace"))
        encoded = text.encode("utf-8")
        return {"path": component["source"], "content": encoded[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore"),
                "truncated": truncated or len(encoded) > MAX_TEXT_BYTES}


def create_server(root, port=8765, host="127.0.0.1", assistant_runner=None, openrouter_runner=None):
    from .workbench_chat import WorkbenchAssistant, AssistantBusy, AssistantUnavailable

    if host != "127.0.0.1":
        raise ValueError("Workbench binds only to 127.0.0.1")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    reader = WorkbenchReader(root)
    assistant = WorkbenchAssistant(reader, runner=assistant_runner, openrouter_runner=openrouter_runner)

    class Handler(BaseHTTPRequestHandler):
        server_version = "HyperspaceWorkbench"
        sys_version = ""

        def log_message(self, format, *args):
            return

        def _send(self, status, content, content_type="application/json; charset=utf-8"):
            if len(content) > MAX_RESPONSE_BYTES:
                status, content = 413, b'{"error":"Response exceeds inspection limit"}'
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(content)

        def _error(self, status, message):
            self._send(status, json.dumps({"error": message}).encode())

        def _allowed(self):
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1:
                return False
            authority = hosts[0]
            allowed = {"127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port)}
            if self.server.server_port == 80:
                allowed.update({"127.0.0.1", "localhost"})
            if authority not in allowed:
                return False
            origins = self.headers.get_all("Origin", [])
            if origins and (len(origins) != 1 or origins[0] != "http://" + authority):
                return False
            return self.headers.get("Sec-Fetch-Site") not in {"cross-site", "same-site"}

        def do_GET(self):
            if not self._allowed():
                self._error(403, "Only same-origin loopback requests are allowed")
                return
            if len(self.path) > 2048:
                self._error(414, "Request path is too long")
                return
            try:
                parsed = urlsplit(self.path)
            except ValueError:
                self._error(400, "Invalid workbench path")
                return
            path = unquote(parsed.path)
            if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in path or "\x00" in path or any(part in {".", ".."} for part in path.split("/")):
                self._error(400, "Invalid workbench path")
                return
            try:
                if path in STATIC_FILES:
                    filename = STATIC_FILES[path]
                    content, truncated = reader._read(reader.root / "astra_harness" / "workbench_static" / filename, 4 * 1024 * 1024)
                    if truncated:
                        raise ValueError("Static file exceeds limit")
                    content_type = "image/png" if filename.endswith(".png") else "text/html; charset=utf-8" if filename.endswith(".html") else "text/css; charset=utf-8" if filename.endswith(".css") else "text/javascript; charset=utf-8"
                    self._send(200, content, content_type)
                    return
                parts = path.strip("/").split("/")
                if path == "/api/overview":
                    result = reader.overview()
                elif path == "/api/assistant/status":
                    result = assistant.status()
                elif path == "/api/assistant/settings":
                    result = assistant.settings()
                elif len(parts) == 4 and parts[:3] == ["api", "assistant", "models"]:
                    result = assistant.models(parts[3])
                elif path == "/api/assistant/history":
                    result = assistant.history()
                elif len(parts) == 4 and parts[:3] == ["api", "assistant", "jobs"]:
                    result = assistant.job(parts[3])
                elif path == "/api/runs":
                    result = reader.list_runs()
                elif len(parts) == 3 and parts[:2] == ["api", "runs"]:
                    result = reader.run_detail(parts[2])
                elif len(parts) == 5 and parts[:2] == ["api", "runs"] and parts[3] == "artifacts":
                    result = reader.artifact(parts[2], parts[4])
                elif len(parts) == 4 and parts[:2] == ["api", "components"] and parts[3] == "source":
                    result = reader.source(parts[2])
                else:
                    raise KeyError("Not found")
                self._send(200, json.dumps(redact(result), ensure_ascii=False, allow_nan=False).encode("utf-8"))
            except (KeyError, FileNotFoundError):
                self._error(404, "Saved item not found")
            except ValueError:
                self._error(400, "Invalid or unavailable saved item")
            except AssistantBusy as exc:
                self._error(409, str(exc))
            except AssistantUnavailable as exc:
                self._error(503, str(exc))
            except RuntimeError:
                self._error(503, "Assistant provider is temporarily unavailable")
            except (OSError, sqlite3.Error, TypeError, RecursionError):
                self._error(503, "Saved data is temporarily unavailable")

        def do_POST(self):
            if not self.path.startswith("/api/assistant/"):
                self._error(405, "Saved run and source inspection is read-only")
                return
            if not self._allowed() or len(self.headers.get_all("Origin", [])) != 1:
                self._error(403, "Assistant requests require an exact same-origin Origin header")
                return
            csrf = self.headers.get_all("X-Workbench-CSRF", [])
            if len(csrf) != 1 or not secrets.compare_digest(csrf[0], assistant.csrf):
                self._error(403, "A current Workbench request nonce is required")
                return
            lengths = self.headers.get_all("Content-Length", [])
            content_types = self.headers.get_all("Content-Type", [])
            if (self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1
                    or not lengths[0].isdigit() or len(content_types) != 1
                    or content_types[0].split(";", 1)[0].strip().lower() != "application/json"):
                self._error(400, "A bounded JSON body with Content-Length is required")
                return
            length = int(lengths[0])
            if length < 1 or length > 128 * 1024:
                self._error(413, "Assistant request exceeds its body limit")
                return
            if len(self.path) > 2048 or "?" in self.path or "#" in self.path or "%" in self.path or "\\" in self.path:
                self._error(400, "Invalid assistant endpoint")
                return
            try:
                self.connection.settimeout(5)
                content = self.rfile.read(length)
                if len(content) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(content)
                parts = self.path.strip("/").split("/")
                if self.path == "/api/assistant/messages":
                    result = assistant.submit(body)
                    status = 202
                elif self.path == "/api/assistant/settings":
                    result = assistant.update_settings(body)
                    status = 200
                elif len(parts) == 5 and parts[:3] == ["api", "assistant", "jobs"] and parts[4] == "cancel":
                    if body != {}:
                        raise ValueError("Cancel expects an empty JSON object")
                    result = assistant.cancel(parts[3])
                    status = 200
                else:
                    raise KeyError("Assistant endpoint not found")
                self._send(status, json.dumps(redact(result), ensure_ascii=False, allow_nan=False).encode())
            except (ValueError, TypeError, RecursionError):
                self._error(400, "Invalid assistant request or visual draft")
            except KeyError:
                self._error(404, "Saved assistant item not found")
            except AssistantBusy as exc:
                self._error(409, str(exc))
            except AssistantUnavailable as exc:
                self._error(503, str(exc))
            except (OSError, RuntimeError):
                self._error(503, "Assistant is temporarily unavailable")

        def _unsupported(self):
            self._error(405, "This HTTP method is unavailable")

        do_PUT = _unsupported
        do_PATCH = _unsupported
        do_DELETE = _unsupported
        do_OPTIONS = _unsupported
        do_HEAD = _unsupported

    class Server(ThreadingHTTPServer):
        def server_close(self):
            assistant.close()
            super().server_close()

    server = Server((host, port), Handler)
    server.assistant = assistant
    server.daemon_threads = True
    return server


def serve(root=ROOT, port=8765):
    server = create_server(root, port)
    print(f"Hyperspace Workbench: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(port=args.port)


if __name__ == "__main__":
    main()
