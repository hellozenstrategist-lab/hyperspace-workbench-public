"""Durable, conservative notification outbox for the existing Photon sidecar.

No model calls. No credential files. Unknown delivery outcomes are quarantined
unless a transport explicitly guarantees provider-side idempotency.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import ipaddress
import json
from pathlib import Path
import random
import socket
import sqlite3
import time
from typing import Callable
from urllib import error, parse, request
import uuid
from zoneinfo import ZoneInfo

KINDS = frozenset({"start", "completed", "permanent_fail", "block",
                   "verified_important", "central_disproved", "digest"})
SEVERITIES = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}


@dataclass(frozen=True)
class NotificationPolicy:
    min_severity: str = "info"
    digest_enabled: bool = False
    timezone: str = "America/Phoenix"
    quiet_start_hour: int | None = 22
    quiet_end_hour: int | None = 7
    quiet_bypass_severity: str = "error"
    max_per_hour: int = 6
    min_interval_seconds: float = 30.0
    max_attempts: int = 4
    backoff_base_seconds: float = 5.0
    backoff_cap_seconds: float = 300.0
    jitter_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.min_severity not in SEVERITIES or self.quiet_bypass_severity not in SEVERITIES:
            raise ValueError("Unknown severity threshold")
        ZoneInfo(self.timezone)
        hours = (self.quiet_start_hour, self.quiet_end_hour)
        if (hours[0] is None) != (hours[1] is None):
            raise ValueError("Both quiet-hour endpoints must be set or disabled")
        if any(h is not None and (isinstance(h, bool) or not isinstance(h, int) or not 0 <= h <= 23) for h in hours):
            raise ValueError("Quiet hours must be integer hours 0..23")
        if self.max_per_hour < 1 or self.max_attempts < 1 or self.min_interval_seconds < 0:
            raise ValueError("Invalid rate or attempt limit")
        if self.backoff_base_seconds <= 0 or self.backoff_cap_seconds < self.backoff_base_seconds:
            raise ValueError("Invalid backoff bounds")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("Invalid jitter fraction")


class PhotonNotifier:
    def __init__(self, db_path: str | Path, *, base_url: str = "http://127.0.0.1:8790",
                 state_path: str | Path = "~/.config/hyperspace-harness/dispatch_state.json",
                 policy: NotificationPolicy | None = None, provider_idempotency: bool = False,
                 timeout_seconds: float = 10.0, clock: Callable[[], float] = time.time,
                 random_value: Callable[[], float] = random.random) -> None:
        parts = parse.urlsplit(base_url)
        try:
            loopback = ipaddress.ip_address(parts.hostname or "").is_loopback
        except ValueError:
            loopback = False
        if (parts.scheme != "http" or not loopback or parts.username or parts.password
                or parts.query or parts.fragment or parts.path not in ("", "/")):
            raise ValueError("Photon sidecar URL must be an HTTP loopback origin")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.state_path = Path(state_path)
        self.policy = policy or NotificationPolicy()
        self.provider_idempotency = provider_idempotency
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.random_value = random_value
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS notifications (
                key TEXT PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
                text TEXT NOT NULL, artifact TEXT NOT NULL, severity TEXT NOT NULL,
                state TEXT NOT NULL, created_at REAL NOT NULL, next_attempt_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, last_attempt_at REAL,
                lease_until REAL, lease_token TEXT, updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS notifications_due ON notifications(state,next_attempt_at);
            CREATE TABLE IF NOT EXISTS notification_receipts (
                receipt_id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,
                run_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
                at REAL NOT NULL, attempt INTEGER NOT NULL, code TEXT NOT NULL,
                provider_id TEXT
            );
        """)

    def close(self) -> None:
        self.db.close()

    def _receipt(self, row: dict | sqlite3.Row, status: str, code: str,
                 provider_id: str | None = None) -> dict:
        at = self.clock()
        cur = self.db.execute(
            "INSERT INTO notification_receipts(key,run_id,kind,status,at,attempt,code,provider_id) VALUES(?,?,?,?,?,?,?,?)",
            (row["key"], row["run_id"], row["kind"], status, at, row["attempts"], code, provider_id))
        return {"receipt_id": cur.lastrowid, "key": row["key"], "run_id": row["run_id"],
                "kind": row["kind"], "status": status, "at": at,
                "attempt": row["attempts"], "code": code, "provider_id": provider_id}

    def enqueue(self, run_id: str, kind: str, text: str, artifact: str,
                severity: str = "info", key: str | None = None) -> dict:
        if not run_id or not isinstance(text, str) or not text.strip() or not isinstance(artifact, str):
            raise ValueError("run_id, notification text and string artifact are required")
        if severity not in SEVERITIES:
            raise ValueError("Unknown severity")
        key = key or hashlib.sha256(json.dumps([run_id, kind], separators=(",", ":")).encode()).hexdigest()
        now = self.clock()
        code = self._eligibility(kind, severity)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute("SELECT * FROM notifications WHERE key=?", (key,)).fetchone()
            if existing is not None:
                if existing["run_id"] != run_id or existing["kind"] != kind:
                    raise ValueError("Notification key belongs to another event")
                self.db.execute("COMMIT")
                return {"key": key, "state": existing["state"], "deduplicated": True}
            state = "suppressed" if code else "queued"
            self.db.execute("INSERT INTO notifications(key,run_id,kind,text,artifact,severity,state,created_at,next_attempt_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (key, run_id, kind, text, artifact, severity, state, now, now, now))
            row = self.db.execute("SELECT * FROM notifications WHERE key=?", (key,)).fetchone()
            self._receipt(row, state, code or "enqueued")
            self.db.execute("COMMIT")
            return {"key": key, "state": state, "deduplicated": False}
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def _eligibility(self, kind: str, severity: str) -> str | None:
        if kind not in KINDS:
            return "ineligible_kind"
        if kind == "digest" and not self.policy.digest_enabled:
            return "digest_disabled"
        if SEVERITIES[severity] < SEVERITIES[self.policy.min_severity]:
            return "below_severity_threshold"
        return None

    def receipts(self, run_id: str | None = None) -> list[dict]:
        sql = "SELECT * FROM notification_receipts"
        params = ()
        if run_id is not None:
            sql += " WHERE run_id=?"
            params = (run_id,)
        return [dict(r) for r in self.db.execute(sql + " ORDER BY receipt_id", params)]

    def pending(self) -> list[dict]:
        """Audit metadata only; excludes recipient, notification text and artifact."""
        return [dict(r) for r in self.db.execute(
            "SELECT key,run_id,kind,severity,state,attempts,next_attempt_at FROM notifications ORDER BY created_at")]

    def _quiet_until(self, severity: str, now: float) -> float | None:
        p = self.policy
        if p.quiet_start_hour is None or SEVERITIES[severity] >= SEVERITIES[p.quiet_bypass_severity]:
            return None
        local = datetime.fromtimestamp(now, ZoneInfo(p.timezone))
        start, end = p.quiet_start_hour, p.quiet_end_hour
        quiet = (local.hour >= start or local.hour < end) if start > end else start <= local.hour < end
        if not quiet or start == end:
            return None
        boundary = local.replace(hour=end, minute=0, second=0, microsecond=0)
        if boundary <= local:
            boundary += timedelta(days=1)
        return boundary.timestamp()

    def _backoff(self, attempt: int) -> float:
        p = self.policy
        base = min(p.backoff_cap_seconds, p.backoff_base_seconds * (2 ** min(attempt - 1, 30)))
        return min(p.backoff_cap_seconds, max(0.001, base * (1 + p.jitter_fraction * (2 * self.random_value() - 1))))

    def _deliver(self, row: sqlite3.Row) -> tuple[str, str, str | None]:
        # Read just the home_space value into the request. Never expose it to
        # logging, receipts, exceptions, command lines, or the model context.
        try:
            state = json.loads(self.state_path.read_text())
            space = state.get("home_space")
            if not isinstance(space, str) or not space:
                return "permanent_failure", "recipient_unconfigured", None
        except (OSError, ValueError, AttributeError):
            return "permanent_failure", "recipient_unconfigured", None
        text = row["text"]
        if row["artifact"]:
            text += "\nArtifact: " + row["artifact"]
        headers = {"Content-Type": "application/json"}
        if self.provider_idempotency:
            headers["Idempotency-Key"] = row["key"]
        req = request.Request(self.base_url + "/send", data=json.dumps({"space_id": space, "text": text}).encode(), headers=headers, method="POST")
        # Never inherit HTTP_PROXY for a loopback notification destination.
        opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(req, timeout=self.timeout_seconds) as response:
                body = response.read(65537)
                if len(body) > 65536:
                    return self._unknown("oversized_response")
                try:
                    result = json.loads(body)
                except ValueError:
                    return self._unknown("invalid_success_response")
                if response.status == 200 and isinstance(result, dict) and result.get("ok") is True:
                    # Real sidecar returns no provider delivery receipt. Only
                    # trust synthetic provider IDs from our idempotent mock.
                    provider_id = str(result.get("message_id")) if self.provider_idempotency and result.get("message_id") else None
                    return "delivered", "sidecar_accepted", provider_id
                return self._unknown("unrecognized_response")
        except error.HTTPError as exc:
            # The inspected sidecar rejects these before calling space.send.
            if exc.code == 503:
                return "retry_wait", "sidecar_disconnected", None
            if exc.code in (400, 401, 403, 404, 405, 413, 422):
                return "permanent_failure", f"http_{exc.code}_rejected", None
            return self._unknown(f"http_{exc.code}_ambiguous")
        except error.URLError as exc:
            if isinstance(exc.reason, (ConnectionRefusedError, socket.gaierror)):
                return "retry_wait", "connection_not_established", None
            return self._unknown("transport_outcome_unknown")
        except (TimeoutError, OSError):
            return self._unknown("transport_outcome_unknown")

    def _unknown(self, code: str) -> tuple[str, str, None]:
        return ("retry_wait" if self.provider_idempotency else "quarantined", code, None)

    def flush(self, limit: int = 5) -> list[dict]:
        output = []
        for _ in range(max(0, limit)):
            now = self.clock()
            token = uuid.uuid4().hex
            self.db.execute("BEGIN IMMEDIATE")
            try:
                abandoned = self.db.execute("SELECT * FROM notifications WHERE state='sending' AND lease_until<=?", (now,)).fetchall()
                for stale in abandoned:
                    state = "retry_wait" if self.provider_idempotency else "quarantined"
                    self.db.execute("UPDATE notifications SET state=?,next_attempt_at=?,updated_at=? WHERE key=?", (state, now, now, stale["key"]))
                    output.append(self._receipt(stale, state, "expired_send_lease_outcome_unknown"))
                row = self.db.execute("SELECT * FROM notifications WHERE state IN ('queued','retry_wait') AND next_attempt_at<=? ORDER BY created_at,key LIMIT 1", (now,)).fetchone()
                if row is None:
                    self.db.execute("COMMIT")
                    break
                policy_code = self._eligibility(row["kind"], row["severity"])
                if policy_code:
                    self.db.execute("UPDATE notifications SET state='suppressed',updated_at=? WHERE key=?", (now, row["key"]))
                    output.append(self._receipt(row, "suppressed", policy_code))
                    self.db.execute("COMMIT")
                    continue
                if row["attempts"] >= self.policy.max_attempts:
                    self.db.execute("UPDATE notifications SET state='permanent_failure',updated_at=? WHERE key=?", (now, row["key"]))
                    output.append(self._receipt(row, "permanent_failure", "attempts_exhausted"))
                    self.db.execute("COMMIT")
                    continue
                due = self._quiet_until(row["severity"], now)
                reason = "quiet_hours" if due else None
                sends = [r[0] for r in self.db.execute("SELECT at FROM notification_receipts WHERE status='sending' AND at>? ORDER BY at", (now - 3600,))]
                if len(sends) >= self.policy.max_per_hour:
                    rate_due = sends[-self.policy.max_per_hour] + 3600.001
                    due, reason = max(due or now, rate_due), "hourly_rate_limit"
                if sends and sends[-1] + self.policy.min_interval_seconds > now:
                    due, reason = max(due or now, sends[-1] + self.policy.min_interval_seconds), "minimum_interval"
                if due:
                    self.db.execute("UPDATE notifications SET next_attempt_at=?,updated_at=? WHERE key=?", (due, now, row["key"]))
                    output.append(self._receipt(row, "deferred", reason))
                    self.db.execute("COMMIT")
                    continue
                self.db.execute("UPDATE notifications SET state='sending',attempts=attempts+1,last_attempt_at=?,lease_until=?,lease_token=?,updated_at=? WHERE key=?",
                                (now, now + self.timeout_seconds * 2 + 60, token, now, row["key"]))
                row = self.db.execute("SELECT * FROM notifications WHERE key=?", (row["key"],)).fetchone()
                self._receipt(row, "sending", "attempt_started")
                self.db.execute("COMMIT")
            except Exception:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise
            try:
                state, code, provider_id = self._deliver(row)
            except Exception:
                state, code, provider_id = self._unknown("unexpected_transport_outcome_unknown")
            next_attempt = self.clock() + self._backoff(row["attempts"]) if state == "retry_wait" else self.clock()
            self.db.execute("BEGIN IMMEDIATE")
            try:
                changed = self.db.execute("UPDATE notifications SET state=?,next_attempt_at=?,lease_until=NULL,lease_token=NULL,updated_at=? WHERE key=? AND state='sending' AND lease_token=?",
                                          (state, next_attempt, self.clock(), row["key"], token)).rowcount
                if changed:
                    output.append(self._receipt(row, state, code, provider_id))
                self.db.execute("COMMIT")
            except Exception:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise
        return output


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
