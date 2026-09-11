"""Local Photon-shaped server with durable idempotency, for acceptance tests."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


class MockPhotonServer:
    def __init__(self, db_path: str | Path, port: int = 0) -> None:
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    idempotency_key TEXT PRIMARY KEY, message_id TEXT NOT NULL,
                    text TEXT NOT NULL, accepted_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, idempotency_key TEXT,
                    at REAL NOT NULL, mode TEXT NOT NULL);
            """)
        self.failures: list[str] = []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                if self.path == "/health":
                    return self.reply(200, {"ok": True, "mock": True})
                self.reply(404, {"error": "not found"})

            def do_POST(self):
                if self.path != "/send":
                    return self.reply(404, {"error": "not found"})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size <= 0 or size > 65536:
                        return self.reply(413, {"error": "body limit"})
                    payload = json.loads(self.rfile.read(size))
                    if not isinstance(payload.get("space_id"), str) or not isinstance(payload.get("text"), str):
                        return self.reply(400, {"error": "invalid body"})
                except (ValueError, AttributeError):
                    return self.reply(400, {"error": "invalid body"})
                key = self.headers.get("Idempotency-Key") or "unkeyed-" + uuid.uuid4().hex
                with owner.lock:
                    mode = owner.failures.pop(0) if owner.failures else "ok"
                with owner._db() as db:
                    db.execute("INSERT INTO attempts(idempotency_key,at,mode) VALUES(?,?,?)", (key, time.time(), mode))
                if mode == "unavailable":
                    return self.reply(503, {"error": "not connected to Photon"})
                if mode == "reject":
                    return self.reply(400, {"error": "fixture rejection"})
                with owner._db() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("INSERT OR IGNORE INTO deliveries VALUES(?,?,?,?)", (key, uuid.uuid4().hex, payload["text"], time.time()))
                    row = db.execute("SELECT message_id FROM deliveries WHERE idempotency_key=?", (key,)).fetchone()
                if mode == "disconnect_after_accept":
                    self.close_connection = True
                    self.connection.close()
                    return
                if mode == "ambiguous_after_accept":
                    return self.reply(502, {"error": "ambiguous fixture"})
                return self.reply(200, {"ok": True, "message_id": row[0], "mock": True})

            def reply(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def start(self):
        self.thread.start()
        return self

    def close(self):
        if self.thread.is_alive():
            self.httpd.shutdown()
            self.thread.join(timeout=3)
        self.httpd.server_close()

    def deliveries(self):
        with self._db() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM deliveries ORDER BY accepted_at")]

    def attempts(self):
        with self._db() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY id")]

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    server = MockPhotonServer(args.db, args.port)
    print(json.dumps({"mock": True, "base_url": server.base_url}), flush=True)
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.httpd.server_close()


if __name__ == "__main__":
    main()
