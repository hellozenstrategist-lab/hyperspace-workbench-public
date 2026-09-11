"""Canonical SQLite graph, durable addressed inboxes and content-addressed evidence."""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import threading
import uuid
from .schema import KnowledgeEvent, canonical, digest, now, PRIORITIES
from . import audit_ledger

DDL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
INSERT OR IGNORE INTO meta VALUES('schema_version','1');
CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY,manifest TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence(hash TEXT PRIMARY KEY,mime TEXT NOT NULL,size INTEGER NOT NULL,path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes(event_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),author TEXT NOT NULL,type TEXT NOT NULL,body TEXT NOT NULL,hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS edges(source TEXT NOT NULL REFERENCES nodes(event_id),target TEXT NOT NULL,relation TEXT NOT NULL,PRIMARY KEY(source,target,relation));
CREATE TABLE IF NOT EXISTS node_evidence(node_id TEXT NOT NULL REFERENCES nodes(event_id),evidence_hash TEXT NOT NULL REFERENCES evidence(hash),PRIMARY KEY(node_id,evidence_hash));
CREATE TABLE IF NOT EXISTS coordinates(node_id TEXT PRIMARY KEY,vector TEXT NOT NULL,method TEXT NOT NULL,version INTEGER NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS workers(run_id TEXT NOT NULL REFERENCES runs(run_id),agent TEXT NOT NULL,state TEXT NOT NULL,thread_id TEXT,turn_id TEXT,started_at TEXT,completed_at TEXT,final TEXT,task TEXT,PRIMARY KEY(run_id,agent));
CREATE TABLE IF NOT EXISTS deliveries(delivery_id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(run_id),event_id TEXT NOT NULL REFERENCES nodes(event_id),recipient TEXT NOT NULL,priority INTEGER NOT NULL,state TEXT NOT NULL DEFAULT 'queued',attempts INTEGER NOT NULL DEFAULT 0,next_attempt REAL NOT NULL DEFAULT 0,created_at TEXT NOT NULL,expires_at REAL NOT NULL,accepted_at TEXT,delivered_at TEXT,acknowledged_at TEXT,incorporated_at TEXT,token_cost INTEGER NOT NULL,decision TEXT NOT NULL,UNIQUE(event_id,recipient));
CREATE INDEX IF NOT EXISTS pending_delivery ON deliveries(run_id,recipient,state,priority,next_attempt);
CREATE TABLE IF NOT EXISTS tool_results(run_id TEXT NOT NULL,agent TEXT NOT NULL,call_id TEXT NOT NULL,result TEXT NOT NULL,PRIMARY KEY(run_id,agent,call_id));
CREATE TABLE IF NOT EXISTS ledger(seq INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT NOT NULL,kind TEXT NOT NULL,at TEXT NOT NULL,data TEXT NOT NULL,prev_hash TEXT NOT NULL,hash TEXT NOT NULL UNIQUE,idem_key TEXT,UNIQUE(run_id,idem_key));
"""


class KnowledgeStore:
    def __init__(self, path, evidence_dir=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.evidence_dir = Path(evidence_dir or self.path.parent / "evidence")
        self.evidence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(DDL)
        os.chmod(self.path, 0o600)
        if self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] != "1":
            raise ValueError("Unsupported persistent schema; explicit migration required")

    @contextmanager
    def transaction(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def log(self, run_id, kind, data, key=None):
        with self.transaction() as conn:
            return audit_ledger.append(conn, run_id, kind, data, key=key)

    def create_run(self, run_id, manifest):
        uuid.UUID(run_id)
        with self.transaction() as conn:
            existing = conn.execute("SELECT manifest FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if existing:
                return json.loads(existing[0])
            conn.execute("INSERT INTO runs VALUES(?,?,?,?)", (run_id, canonical(manifest), "prepared", now()))
            audit_ledger.append(conn, run_id, "run_created", manifest, key="run_created")
        return manifest

    def update_manifest(self, run_id, manifest):
        with self.transaction() as conn:
            conn.execute("UPDATE runs SET manifest=? WHERE run_id=?", (canonical(manifest), run_id))
            audit_ledger.append(conn, run_id, "manifest_updated", manifest)

    def set_run_state(self, run_id, state):
        with self.transaction() as conn:
            conn.execute("UPDATE runs SET status=? WHERE run_id=?", (state, run_id))
            audit_ledger.append(conn, run_id, "run_state", {"state": state})

    def run(self, run_id):
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError("Unknown run ID")
        return {**dict(row), "manifest": json.loads(row["manifest"])}

    def add_evidence(self, content: bytes, mime="application/json"):
        if not isinstance(content, bytes) or len(content) > 10_000_000:
            raise ValueError("Evidence requires bounded bytes")
        checksum = digest(content)
        path = self.evidence_dir / checksum
        # Files precede references. An interrupted write can leave an orphan,
        # never a committed reference to partially written content.
        if not path.exists():
            temporary = self.evidence_dir / ("." + checksum + "." + uuid.uuid4().hex)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(self.evidence_dir, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        if digest(path.read_bytes()) != checksum:
            raise ValueError("Evidence content checksum mismatch")
        with self.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)", (checksum, mime, len(content), str(path)))
        return checksum

    def evidence(self, checksum):
        row = self.conn.execute("SELECT * FROM evidence WHERE hash=?", (checksum,)).fetchone()
        if not row:
            raise KeyError("Unknown evidence reference")
        content = Path(row["path"]).read_bytes()
        if digest(content) != checksum:
            raise ValueError("Evidence integrity failure")
        return content

    def put_event(self, event: KnowledgeEvent, position, coordinate_method="hierarchy-v1"):
        import math
        if len(position) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in position) or sum(x*x for x in position) >= 1:
            raise ValueError("Coordinate must be finite 2D strictly inside the unit ball")
        body = event.to_dict()
        text = canonical(body)
        with self.transaction() as conn:
            existing = conn.execute("SELECT hash FROM nodes WHERE event_id=?", (event.event_id,)).fetchone()
            if existing:
                if existing[0] != digest(text.encode()):
                    raise ValueError("Event ID collision with different content")
                return False
            for reference in event.evidence_refs:
                if not conn.execute("SELECT 1 FROM evidence WHERE hash=?", (reference,)).fetchone():
                    raise ValueError("Evidence must be durably stored before publication")
            conn.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?)", (event.event_id, event.run_id, event.author_agent, event.type, text, digest(text.encode())))
            for target in event.parent_ids:
                conn.execute("INSERT INTO edges VALUES(?,?,?)", (event.event_id, target, "parent"))
            for target in event.contradicts:
                conn.execute("INSERT INTO edges VALUES(?,?,?)", (event.event_id, target, "contradicts"))
            if event.revision_of:
                if not conn.execute("SELECT 1 FROM nodes WHERE event_id=?", (event.revision_of,)).fetchone():
                    raise ValueError("A revision must reference an existing node")
                conn.execute("INSERT INTO edges VALUES(?,?,?)", (event.event_id, event.revision_of, "revises"))
            for reference in event.evidence_refs:
                conn.execute("INSERT INTO node_evidence VALUES(?,?)", (event.event_id, reference))
            conn.execute("INSERT INTO coordinates VALUES(?,?,?,?,?)", (event.event_id, canonical(position), coordinate_method, 1, now()))
            audit_ledger.append(conn, event.run_id, "published", {"event": body, "position": position, "coordinate_method": coordinate_method}, key="publish:" + event.event_id)
        return True

    def event(self, event_id):
        row = self.conn.execute("SELECT body FROM nodes WHERE event_id=?", (event_id,)).fetchone()
        return KnowledgeEvent.from_dict(json.loads(row[0])) if row else None

    def events(self, run_id):
        return [KnowledgeEvent.from_dict(json.loads(row[0])) for row in self.conn.execute("SELECT body FROM nodes WHERE run_id=? ORDER BY rowid", (run_id,))]

    def update_coordinate(self, node_id, position, method):
        import math
        if len(position) != 2 or any(not math.isfinite(x) for x in position) or sum(x*x for x in position) >= 1:
            raise ValueError("Coordinate must remain strictly within Poincare ball")
        with self.transaction() as conn:
            old = conn.execute("SELECT * FROM coordinates WHERE node_id=?", (node_id,)).fetchone()
            if not old:
                raise KeyError(node_id)
            conn.execute("UPDATE coordinates SET vector=?,method=?,version=version+1,updated_at=? WHERE node_id=?", (canonical(position), method, now(), node_id))
            event = conn.execute("SELECT run_id FROM nodes WHERE event_id=?", (node_id,)).fetchone()
            audit_ledger.append(conn, event[0], "coordinate_updated", {"node_id": node_id, "old": json.loads(old["vector"]), "new": position, "version": old["version"] + 1, "method": method})

    def route(self, event_id, decisions, *, ttl=300, max_pending=256, meta_updates=None):
        import time
        event = self.event(event_id)
        if not event:
            raise ValueError("Persist event before routing")
        cost = max(1, (len(canonical(event.compact()).encode()) + 2) // 3)
        created = []
        with self.transaction() as conn:
            for decision in decisions:
                recipient = decision["recipient"]
                audit_ledger.append(conn, event.run_id, "routing_decision", {"event_id": event_id, **decision}, key=f"route:{event_id}:{recipient}")
                if not decision["deliver"] or recipient == event.author_agent:
                    continue
                ident = str(uuid.uuid5(uuid.UUID(event_id), recipient))
                if conn.execute("SELECT 1 FROM deliveries WHERE delivery_id=?", (ident,)).fetchone():
                    continue
                pending = conn.execute("SELECT COUNT(*) FROM deliveries WHERE run_id=? AND recipient=? AND acknowledged_at IS NULL AND state NOT IN ('expired','failed')", (event.run_id, recipient)).fetchone()[0]
                if pending >= max_pending:
                    # Never silently discard mandatory messages: publication
                    # remains durable; reroute can be retried once pressure clears.
                    raise BufferError("Durable inbox capacity reached; publisher must retry routing")
                conn.execute("INSERT INTO deliveries(delivery_id,run_id,event_id,recipient,priority,state,created_at,expires_at,token_cost,decision) VALUES(?,?,?,?,?,'queued',?,?,?,?)",
                             (ident, event.run_id, event_id, recipient, PRIORITIES[event.priority], now(), time.time() + ttl, cost, canonical(decision)))
                audit_ledger.append(conn, event.run_id, "message_published", {"delivery_id": ident, "event_id": event_id, "sender": event.author_agent, "recipient": recipient}, key="message:" + ident)
                created.append(ident)
            for key, value in (meta_updates or {}).items():
                conn.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))
        return created

    def deliveries(self, run_id, recipient=None):
        if recipient:
            rows = self.conn.execute("SELECT * FROM deliveries WHERE run_id=? AND recipient=? ORDER BY priority,created_at,delivery_id", (run_id, recipient))
        else:
            rows = self.conn.execute("SELECT * FROM deliveries WHERE run_id=? ORDER BY priority,created_at,delivery_id", (run_id,))
        return [dict(row) for row in rows]

    def transition(self, delivery_id, state, *, details=None, attempt=False, next_attempt=0):
        columns = {"accepted": "accepted_at", "delivered": "delivered_at", "acknowledged": "acknowledged_at", "incorporated": "incorporated_at"}
        if state not in {*columns, "attempting", "retry_wait", "expired", "failed", "uncertain"}:
            raise ValueError("Invalid delivery state")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM deliveries WHERE delivery_id=?", (delivery_id,)).fetchone()
            if not row:
                raise KeyError(delivery_id)
            if state in columns and row[columns[state]]:
                return False
            if state == "acknowledged" and not row["delivered_at"]:
                raise ValueError("Cannot acknowledge undelivered knowledge")
            if state == "incorporated" and not row["acknowledged_at"]:
                raise ValueError("Cannot incorporate unacknowledged knowledge")
            if row["delivered_at"] and state not in {"acknowledged", "incorporated"}:
                return False
            if state == "attempting" and row["state"] not in {"queued", "retry_wait"}:
                return False
            if row["state"] in {"expired", "failed"} and state not in {"failed", "expired"}:
                raise ValueError("Terminal delivery cannot transition")
            assignments = "state=?,attempts=attempts+?,next_attempt=?"
            values = [state, int(attempt), next_attempt]
            if state in columns:
                assignments += "," + columns[state] + "=?"
                values.append(now())
            values.append(delivery_id)
            conn.execute("UPDATE deliveries SET " + assignments + " WHERE delivery_id=?", values)
            payload = {"delivery_id": delivery_id, "event_id": row["event_id"], "recipient": row["recipient"], **(details or {})}
            audit_ledger.append(conn, row["run_id"], state, payload, key=(state + ":" + delivery_id) if state in columns else None)
            return True

    def worker(self, run_id, agent, *, state, **fields):
        allowed = {"thread_id", "turn_id", "started_at", "completed_at", "final", "task"}
        if set(fields) - allowed:
            raise ValueError("Unknown worker field")
        with self.transaction() as conn:
            conn.execute("INSERT INTO workers(run_id,agent,state) VALUES(?,?,?) ON CONFLICT(run_id,agent) DO UPDATE SET state=excluded.state", (run_id, agent, state))
            for name, value in fields.items():
                conn.execute(f"UPDATE workers SET {name}=? WHERE run_id=? AND agent=?", (value, run_id, agent))

    def workers(self, run_id):
        return {row["agent"]: dict(row) for row in self.conn.execute("SELECT * FROM workers WHERE run_id=?", (run_id,))}

    def assign_worker_task(self, run_id, agent, task, assignment_id, details):
        """Commit the host's requested region and assignment intent together."""
        with self.transaction() as conn:
            key = "assignment_intent:" + assignment_id
            if conn.execute("SELECT 1 FROM ledger WHERE run_id=? AND idem_key=?", (run_id, key)).fetchone():
                return False
            conn.execute("INSERT INTO workers(run_id,agent,state,task) VALUES(?,?,'prepared',?) "
                         "ON CONFLICT(run_id,agent) DO UPDATE SET task=excluded.task", (run_id, agent, canonical(task)))
            audit_ledger.append(conn, run_id, "task_reassigned", {"assignment_id": assignment_id, "agent": agent,
                **details, "worker_adoption_verified": False, "scope": "host requested attention region"}, key=key)
            return True

    def cached_tool(self, run_id, agent, call_id):
        row = self.conn.execute("SELECT result FROM tool_results WHERE run_id=? AND agent=? AND call_id=?", (run_id, agent, call_id)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_tool(self, run_id, agent, call_id, result):
        with self.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO tool_results VALUES(?,?,?,?)", (run_id, agent, call_id, canonical(result)))

    def ledger(self, run_id=None):
        query, args = ("SELECT * FROM ledger WHERE run_id=? ORDER BY seq", (run_id,)) if run_id else ("SELECT * FROM ledger ORDER BY seq", ())
        return [dict(row) for row in self.conn.execute(query, args)]

    def export(self, run_id, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "ledger.jsonl").open("w") as stream:
            # Export the entire chain so the first prev_hash is independently checkable.
            for row in self.ledger():
                stream.write(canonical(row) + "\n")
        snapshot = {"run": self.run(run_id), "workers": self.workers(run_id), "events": [e.to_dict() for e in self.events(run_id)], "deliveries": self.deliveries(run_id), "ledger_integrity": audit_ledger.verify(self.ledger())}
        (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
        return snapshot

    def close(self):
        self.conn.execute("PRAGMA wal_checkpoint(FULL)")
        self.conn.close()
