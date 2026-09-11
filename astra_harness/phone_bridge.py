"""Durable operator-only Photon inbox -> persistent GLM harness chat.

The loopback gateway owns provider authentication and sender authorization.
This process sees authorized text and stable event IDs only, never phone numbers
or provider credentials. Outbound uncertainty is quarantined, never replayed.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import sqlite3
import time
from urllib import parse, request
import uuid

from .photon_notifier import PhotonNotifier, NotificationPolicy, _NoRedirect
from .schema import atomic_json, canonical, now


class IntakeError(ValueError):
    pass


class PhoneStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / "phone.sqlite"
        path.touch(mode=0o600, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA busy_timeout=30000;
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                event_id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, body_hash TEXT NOT NULL,
                text TEXT NOT NULL, attachment INTEGER NOT NULL, received_at TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'queued', session_id TEXT, reply TEXT,
                failure_kind TEXT, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit(
                seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, kind TEXT NOT NULL,
                event_id TEXT, data TEXT NOT NULL);
        """)
        if self.get("session_id") is None:
            self.set("session_id", str(uuid.uuid4()))

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))

    def log(self, kind, event_id=None, **data):
        self.db.execute("INSERT INTO audit(at,kind,event_id,data) VALUES(?,?,?,?)", (now(), kind, event_id, canonical(data)))

    def ingest(self, batch):
        """Atomically persist every authorized event before advancing its cursor."""
        if not isinstance(batch, dict) or not isinstance(batch.get("messages"), list) or len(batch["messages"]) > 100:
            raise IntakeError("Invalid bounded gateway inbox response")
        cursor = self.get("cursor", 0)
        journal_id = batch.get("journal_id")
        if not isinstance(journal_id, str) or not 1 <= len(journal_id) <= 100:
            raise IntakeError("Gateway journal identity is required")
        previous_journal = self.get("journal_id")
        if previous_journal is not None and previous_journal != journal_id:
            raise IntakeError("Gateway journal changed; explicit intake migration is required")
        next_cursor = batch.get("next_cursor")
        if type(next_cursor) is not int or next_cursor < cursor:
            raise IntakeError("Gateway cursor regressed")
        latest = batch.get("latest_seq", next_cursor)
        if type(latest) is not int or latest < next_cursor:
            raise IntakeError("Invalid gateway sequence boundary")
        normalized = []
        prior_seq = cursor
        for message in batch["messages"]:
            if not isinstance(message, dict):
                raise IntakeError("Malformed gateway event")
            ident, seq, text = message.get("event_id"), message.get("seq"), message.get("text", "")
            if (not isinstance(ident, str) or not 1 <= len(ident) <= 200 or type(seq) is not int
                    or not prior_seq < seq <= next_cursor or not isinstance(text, str) or len(text) > 12000):
                raise IntakeError("Malformed bounded gateway event")
            attachment = bool(message.get("attachment"))
            normalized.append((ident, seq, hashlib.sha256(canonical([text, attachment]).encode()).hexdigest(), text, int(attachment)))
            prior_seq = seq
        if normalized and prior_seq != next_cursor:
            raise IntakeError("Gateway cursor skipped an unpersisted event")
        if not normalized and next_cursor != cursor:
            raise IntakeError("Gateway cursor advanced without events")
        queued = self.db.execute("SELECT count(*) FROM messages WHERE state IN ('queued','processing')").fetchone()[0]
        if queued + len(normalized) > 128:
            raise BufferError("Phone inbox capacity reached; gateway retains unread events")
        inserted = []
        with self.transaction():
            for ident, seq, checksum, text, attachment in normalized:
                previous = self.db.execute("SELECT * FROM messages WHERE event_id=?", (ident,)).fetchone()
                if previous:
                    if previous["body_hash"] != checksum or previous["seq"] != seq:
                        raise IntakeError("Gateway event identity changed")
                    continue
                self.db.execute("INSERT INTO messages(event_id,seq,body_hash,text,attachment,received_at,updated_at) VALUES(?,?,?,?,?,?,?)", (ident, seq, checksum, text, attachment, now(), now()))
                self.log("received", ident, gateway_seq=seq)
                inserted.append(ident)
            self.set("cursor", next_cursor)
            self.set("journal_id", journal_id)
        return inserted

    def message(self, event_id):
        row = self.db.execute("SELECT * FROM messages WHERE event_id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def next_message(self):
        row = self.db.execute("SELECT * FROM messages WHERE state IN ('processing','queued') ORDER BY seq LIMIT 1").fetchone()
        return dict(row) if row else None

    def start_message(self, event_id):
        with self.transaction():
            row = self.message(event_id)
            session_id = row["session_id"] or self.get("session_id")
            if row["state"] == "queued":
                self.db.execute("UPDATE messages SET state='processing',session_id=?,updated_at=? WHERE event_id=?", (session_id, now(), event_id))
                self.log("processing", event_id, session_id=session_id)
            return self.message(event_id)

    def finish(self, event_id, text, failure_kind=None):
        if not isinstance(text, str) or not 1 <= len(text) <= 4000:
            raise ValueError("Invalid bounded phone reply")
        with self.transaction():
            row = self.message(event_id)
            if row["state"] not in {"queued", "processing"}:
                return False
            self.db.execute("UPDATE messages SET state='reply_ready',reply=?,failure_kind=?,session_id=COALESCE(session_id,?),updated_at=? WHERE event_id=?", (text, failure_kind, self.get("session_id"), now(), event_id))
            self.log("reply_ready", event_id, failure_kind=failure_kind)
            return True

    def reset(self, control_id):
        with self.transaction():
            control_seq = self.message(control_id)["seq"]
            rows = self.db.execute("SELECT event_id FROM messages WHERE state IN ('queued','processing') AND seq<?", (control_seq,)).fetchall()
            for row in rows:
                self.db.execute("UPDATE messages SET state='cancelled',updated_at=? WHERE event_id=?", (now(), row[0]))
                self.log("cancelled", row[0], source="operator_control")
            self.set("session_id", str(uuid.uuid4()))
            self.log("conversation_reset", control_id)

    def status(self):
        counts = {r[0]: r[1] for r in self.db.execute("SELECT state,count(*) FROM messages GROUP BY state")}
        return {"session_id": self.get("session_id"), "cursor": self.get("cursor", 0), "messages": counts}

    def close(self):
        self.db.execute("PRAGMA wal_checkpoint(FULL)")
        self.db.close()


class ChatOutbox(PhotonNotifier):
    """Reactive chat has its own outbox/rate policy; notification rules stay intact."""
    def _eligibility(self, kind, severity):
        return None if kind == "chat_reply" and severity == "info" else "ineligible_chat_kind"


def chat_outbox(directory, base_url, state_path):
    return ChatOutbox(Path(directory) / "replies.sqlite", base_url=base_url, state_path=state_path,
        provider_idempotency=False, policy=NotificationPolicy(quiet_start_hour=None, quiet_end_hour=None,
            max_per_hour=120, min_interval_seconds=1, max_attempts=4))


class Gateway:
    def __init__(self, base_url):
        parts = parse.urlsplit(base_url)
        try:
            valid = ipaddress.ip_address(parts.hostname or "").is_loopback
        except ValueError:
            valid = False
        if not valid or parts.scheme != "http" or parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("Gateway must be an HTTP loopback origin")
        self.base = base_url.rstrip("/")
        self.opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())

    def get(self, path):
        with self.opener.open(self.base + path, timeout=5) as response:
            raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise IntakeError("Gateway response exceeds bounded inbox size")
            return json.loads(raw)


class PhoneBridge:
    def __init__(self, directory, *, endpoint, base_url="http://127.0.0.1:8790",
                 state_path="~/.config/hyperspace-harness/dispatch_state.json",
                 line_lock="~/.config/hyperspace-harness/dispatch.lock", conversation_factory=None):
        self.directory = Path(directory).resolve()
        self.store = PhoneStore(self.directory)
        self.endpoint, self.base_url, self.state_path = endpoint, base_url, state_path
        self.gateway = Gateway(base_url)
        self.conversation_factory = conversation_factory
        self.conversation = None
        self.conversation_session = None
        self.active_task = None
        self.active_event_id = None
        self.wake = asyncio.Event()
        self.stop = asyncio.Event()
        self.line_lock_path = Path(line_lock) if line_lock else None
        self.locks = []
        self.inbound_healthy = False
        self.last_error = None
        self.resetting = False
        self.control_lock = asyncio.Lock()

    def acquire(self):
        try:
            handle = (self.directory / "bridge.lock").open("a")
            self.locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.line_lock_path:
                # Existing Grimoire dispatch uses the same advisory lock. No
                # journal/source/config file is changed to claim this line.
                handle = self.line_lock_path.open("r")
                self.locks.append(handle)
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            for handle in self.locks:
                handle.close()
            self.locks.clear()
            raise RuntimeError("Another process owns the Photon conversation line") from None

    async def _conversation(self, session_id):
        if self.conversation and self.conversation_session != session_id:
            await self.conversation.close()
            self.conversation = None
        if not self.conversation:
            if self.conversation_factory is None:
                from .phone_chat import PhoneChat
                factory = PhoneChat
            else:
                factory = self.conversation_factory
            self.conversation = factory(self.directory / "sessions" / session_id, endpoint=self.endpoint)
            self.conversation_session = session_id
            await self.conversation.start()
        return self.conversation

    async def control(self, event_id):
        row = self.store.message(event_id)
        if row["text"].strip().lower() not in {"/status", "/help", "/new", "/stop"}:
            return False
        async with self.control_lock:
            if self.store.message(event_id)["state"] not in {"queued", "processing"}:
                return True
            self.resetting = True
            try:
                return await self._control(event_id)
            finally:
                self.resetting = False
                self.wake.set()

    async def _control(self, event_id):
        row = self.store.message(event_id)
        command = row["text"].strip().lower()
        if command == "/status":
            queued = self.store.status()["messages"].get("queued", 0) - 1
            active = bool(self.active_task and not self.active_task.done())
            reply = ("Hyperspace is working on your message." if active else "Hyperspace is ready with your saved API model.")
            reply += f" {max(0, queued)} message(s) waiting."
        elif command == "/help":
            reply = "Text normally to talk to your Hyperspace harness using GLM Flash. /status shows activity. /new starts fresh. /stop cancels work and starts fresh on your next text. Text only for now. Actions needing terminal approval must be completed in the terminal."
        elif command in {"/new", "/stop"}:
            if self.active_task and not self.active_task.done():
                self.active_task.cancel()
                await asyncio.gather(self.active_task, return_exceptions=True)
            if self.conversation:
                if hasattr(self.conversation, "cancel"):
                    await self.conversation.cancel()
                elif hasattr(self.conversation, "runtime"):
                    await self.conversation.runtime.cancel()
                await self.conversation.close()
                self.conversation = None
                self.conversation_session = None
            self.store.reset(event_id)
            reply = "Fresh conversation ready. What would you like to talk about?" if command == "/new" else "Stopped. Your next text starts a fresh Hyperspace conversation."
        else:
            return False
        self.store.finish(event_id, reply)
        self.wake.set()
        return True

    async def intake_once(self):
        health = await asyncio.to_thread(self.gateway.get, "/health")
        batch = await asyncio.to_thread(self.gateway.get, "/inbox?after=" + str(self.store.get("cursor", 0)) + "&limit=50")
        ids = self.store.ingest(batch)
        self.inbound_healthy = health.get("connected") is True and not health.get("blocked")
        self.last_error = None if self.inbound_healthy else "GatewayDisconnected"
        for ident in ids:
            await self.control(ident)
        if ids:
            self.wake.set()
        return ids

    async def _intake_loop(self):
        while not self.stop.is_set():
            try:
                await self.intake_once()
            except Exception as exc:
                self.inbound_healthy = False
                error = type(exc).__name__
                if error != self.last_error:
                    self.store.log("gateway_unavailable", error_type=error)
                self.last_error = error
            await asyncio.sleep(1)

    async def process_message(self, event_id):
        row = self.store.start_message(event_id)
        if not row["text"].strip():
            self.store.finish(event_id, "I can read text here. Please type your message; attachments aren't supported yet.")
            return
        try:
            conversation = await self._conversation(row["session_id"])
            answer = await conversation.reply(event_id, row["text"])
            self.store.finish(event_id, answer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.store.log("conversation_failed", event_id, error_type=type(exc).__name__)
            self.store.finish(event_id, "Hyperspace stopped before I could finish that reply. Your message is saved. Text /new to start a fresh conversation, or /status to check the line.", failure_kind=type(exc).__name__)

    async def _worker_loop(self):
        while not self.stop.is_set():
            if self.resetting:
                await asyncio.sleep(0.02)
                continue
            row = self.store.next_message()
            if row:
                # Re-evaluate persisted controls after a process restart too.
                if await self.control(row["event_id"]):
                    continue
                self.active_event_id = row["event_id"]
                self.active_task = asyncio.create_task(self.process_message(row["event_id"]))
                try:
                    await self.active_task
                except asyncio.CancelledError:
                    if self.stop.is_set():
                        raise
                finally:
                    self.active_task = None
                    self.active_event_id = None
            else:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                except TimeoutError:
                    pass

    def _flush(self):
        client = chat_outbox(self.directory, self.base_url, self.state_path)
        try:
            client.flush()
            return client.pending()
        finally:
            client.close()

    async def send_once(self):
        # Intent is durably in messages before entering the independent outbox.
        client = chat_outbox(self.directory, self.base_url, self.state_path)
        try:
            for row in self.store.db.execute("SELECT * FROM messages WHERE state IN ('reply_ready','reply_queued') ORDER BY seq").fetchall():
                client.enqueue(row["session_id"] or self.store.get("session_id"), "chat_reply", row["reply"], "", key="chat-reply:" + row["event_id"])
                self.store.db.execute("UPDATE messages SET state='reply_queued' WHERE event_id=?", (row["event_id"],))
        finally:
            client.close()
        states = await asyncio.to_thread(self._flush)
        with self.store.transaction():
            for receipt in states:
                ident = receipt["key"].removeprefix("chat-reply:")
                row = self.store.message(ident)
                if not row or row["state"] != "reply_queued":
                    continue
                state = receipt["state"]
                if state in {"delivered", "quarantined", "permanent_failure", "suppressed"}:
                    self.store.db.execute("UPDATE messages SET state=?,updated_at=? WHERE event_id=?", (state, now(), ident))
                    self.store.log("reply_" + state, ident)

    async def _send_loop(self):
        while not self.stop.is_set():
            try:
                await self.send_once()
            except Exception as exc:
                self.store.log("outbox_error", error_type=type(exc).__name__)
            self.write_status()
            await asyncio.sleep(1)

    def write_status(self):
        from .api_settings import api_status
        value = {**self.store.status(), "updated_at": now(), "inbound_gateway_healthy": self.inbound_healthy,
                 "active": self.active_event_id is not None, "model": api_status()["model"], "workers": 1,
                 "last_error_type": self.last_error}
        atomic_json(self.directory / "status.json", value)

    async def run(self):
        self.acquire()
        tasks = [asyncio.create_task(fn()) for fn in (self._intake_loop, self._worker_loop, self._send_loop)]
        try:
            await self.stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.conversation:
                await self.conversation.close()
            self.write_status()
            for handle in self.locks:
                handle.close()
            self.store.close()


async def serve(directory, endpoint, *, base_url="http://127.0.0.1:8790", state_path="~/.config/hyperspace-harness/dispatch_state.json", line_lock="~/.config/hyperspace-harness/dispatch.lock"):
    bridge = PhoneBridge(directory, endpoint=endpoint, base_url=base_url, state_path=state_path, line_lock=line_lock)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, bridge.stop.set)
    await bridge.run()
