"""Audit worker lifetimes and client HTTP overlap without inferring GPU activity."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math


def _timestamp(value):
    """Return UTC microseconds, rejecting ambiguous or nonfinite timestamps."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            return round(value * 1_000_000)
        raise ValueError("Nonfinite timestamp")
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timestamp requires a timezone")
    delta = stamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _overlap(intervals, workers, *, distinct_workers):
    """Sweep half-open intervals; touching endpoints never count as overlap."""
    changes = defaultdict(Counter)
    for agent, start, end in intervals:
        if end > start:
            changes[start][agent] += 1
            changes[end][agent] -= 1
    active, peak, overlap, previous = Counter(), 0, 0, None
    for stamp in sorted(changes):
        if previous is not None and workers and all(active[a] > 0 for a in workers):
            overlap += stamp - previous
        active.update(changes[stamp])
        peak = max(peak, sum(value > 0 for value in active.values()) if distinct_workers else sum(active.values()))
        previous = stamp
    return peak, overlap / 1_000_000


class _Intervals:
    def __init__(self, requests=False):
        self.requests = requests
        self.pending, self.seen, self.closed = {}, set(), []
        self.count = self.invalid_events = self.invalid_timing = 0

    def add(self, kind, agent, data, at):
        identifier = data.get("request_id" if self.requests else "turn_id")
        if self.requests and (not isinstance(identifier, str) or not identifier):
            self.invalid_events += 1
            return
        if identifier is not None and not isinstance(identifier, str):
            self.invalid_events += 1
            return
        key = (agent, identifier)
        start = kind.endswith("_started")
        # Legacy worker completions may omit the turn ID recorded at start.
        if not start and not self.requests and identifier is None and key not in self.pending:
            candidates = [candidate for candidate in self.pending if candidate[0] == agent]
            if len(candidates) == 1:
                key = candidates[0]
        field = "started_at" if start else "completed_at"
        try:
            stamp = _timestamp(data[field] if self.requests and field in data else at)
        except (ValueError, TypeError, OverflowError):
            stamp = None
            self.invalid_timing += 1
        if start:
            if key in self.pending or (self.requests and key in self.seen):
                self.invalid_events += 1
                return
            self.count += 1
            self.seen.add(key)
            self.pending[key] = stamp
            return
        if key not in self.pending:
            self.invalid_events += 1
            return
        if self.requests and data.get("status") in {"uncertain", "cancelled"} and "completed_at" not in data:
            return
        began = self.pending.pop(key)
        if began is None or stamp is None:
            return
        if stamp < began:
            self.invalid_timing += 1
            return
        self.closed.append((agent, began, stamp))


def concurrency_report(ledger, worker_ids):
    """Report closed intervals; unfinished requests remain explicitly incomplete.

    HTTP payload timestamps describe actual client calls when available; ledger
    timestamps are the fallback. Missing request instrumentation yields null HTTP
    metrics. Neither worker nor HTTP overlap measures provider GPU generation.
    """
    workers = set(worker_ids)
    if any(not isinstance(agent, str) or not agent for agent in workers):
        raise ValueError("Worker IDs must be nonempty strings")
    worker_intervals, requests = _Intervals(), _Intervals(requests=True)
    completed, failed, request_errors, rate_limits = set(), set(), set(), set()
    request_instrumented = False
    invalid_rows = error_events = 0
    for index, row in enumerate(ledger):
        try:
            kind, at = row["kind"], row["at"]
            data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            if not isinstance(data, dict):
                raise ValueError("Ledger data must be an object")
        except (KeyError, TypeError, ValueError):
            invalid_rows += 1
            continue
        agent = data.get("agent")
        if agent not in workers:
            if kind == "error" and agent is None:
                error_events += 1
            continue
        if kind == "error":
            error_events += 1
        if kind in {"worker_started", "worker_completed"}:
            worker_intervals.add(kind, agent, data, at)
            if kind == "worker_completed":
                if data.get("status", "completed") == "completed":
                    completed.add(agent)
                else:
                    failed.add(agent)
        if kind in {"request_started", "request_completed", "request_uncertain", "request_rate_limited", "request_failed"}:
            request_instrumented = True
            request_id = data.get("request_id")
            key = (agent, request_id) if isinstance(request_id, str) and request_id else (agent, index)
            if kind in {"request_started", "request_completed"}:
                requests.add(kind, agent, data, at)
            if kind in {"request_uncertain", "request_failed"} or data.get("status") in {"failed", "uncertain", "cancelled", "rate_limited"}:
                request_errors.add(key)
            if kind == "request_rate_limited" or data.get("status") == "rate_limited" or data.get("http_status") == 429:
                rate_limits.add(key)
                request_errors.add(key)
    peak_workers, worker_overlap = _overlap(worker_intervals.closed, workers, distinct_workers=True)
    peak_requests, request_overlap = _overlap(requests.closed, workers, distinct_workers=False)
    http = {
        "status": "measured" if request_instrumented else "not_measured",
        "request_count": requests.count,
        "peak_in_flight_requests": peak_requests,
        "all_workers_requests_overlap_seconds": request_overlap,
        "incomplete_requests": len(requests.pending),
        "rate_limit_events": len(rate_limits),
        "error_requests": len(request_errors),
        "invalid_timing_events": requests.invalid_timing,
        "invalid_lifecycle_events": requests.invalid_events,
    }
    if not request_instrumented:
        http.update({key: None for key in http if key != "status"})
    return {
        "worker_count": len(workers),
        "peak_active_workers": peak_workers,
        "all_workers_overlap_seconds": worker_overlap,
        "workers_completed": len(completed),
        "workers_failed": len(failed),
        "incomplete_worker_intervals": len(worker_intervals.pending),
        "invalid_worker_timing_events": worker_intervals.invalid_timing,
        "invalid_worker_lifecycle_events": worker_intervals.invalid_events,
        "invalid_ledger_rows": invalid_rows,
        "error_events": error_events,
        "http_requests": http,
        "scope": "Closed worker lifetimes include tool and scheduling waits. HTTP overlap measures client requests, including network and provider waits; provider GPU generation concurrency is not measured.",
    }
