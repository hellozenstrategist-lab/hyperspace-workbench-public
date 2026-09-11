"""Provider-neutral outbox draining and explicitly requested count-only digests.

No model, credentials, recipient resolution, transport, or scheduler lives here.
The supplied notifier owns durable enqueue, delivery policy, retries, and receipts.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import time

MIN_DIGEST_INTERVAL = 600
DIGEST_VERSION = "run_counts_v1"


def _interval(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("digest_interval must be a whole number of seconds, at least 600")
    if value < MIN_DIGEST_INTERVAL or int(value) != value:
        raise ValueError("digest_interval must be a whole number of seconds, at least 600")
    return int(value)


def _event_value(event, name):
    return event.get(name) if isinstance(event, dict) else getattr(event, name, None)


def summarize_run(store, run_id):
    """Aggregate operational counts; never expose claims, prompts, or evidence codes."""
    run = store.run(run_id)
    workers = store.workers(run_id)
    events = store.events(run_id)
    deliveries = store.deliveries(run_id)
    ledger = store.ledger(run_id)
    return {
        "run_status": run.get("status", "unknown"),
        "workers": {"total": len(workers), "by_state": dict(sorted(Counter(
            worker.get("state", "unknown") for worker in workers.values()).items()))},
        "publications": {"total": len(events), "by_workers": sum(
            _event_value(event, "author_agent") in workers for event in events)},
        "deliveries": {"total": len(deliveries), "by_state": dict(sorted(Counter(
            row.get("state", "unknown") for row in deliveries).items())),
            **{name: sum(bool(row.get(name + "_at")) for row in deliveries)
               for name in ("accepted", "delivered", "acknowledged", "incorporated")}},
        "failures": {
            "ledger_events": sum(row.get("kind") in ("error", "failure", "runtime_error", "permanent_fail") for row in ledger),
            "workers": sum(worker.get("state") == "failed" for worker in workers.values()),
            "deliveries": sum(row.get("state") == "failed" for row in deliveries),
        },
        "tool_rejections": sum(row.get("kind") == "tool_rejected" for row in ledger),
    }


def _digest_text(summary, bucket_end):
    worker_states = ", ".join(f"{state}: {count}" for state, count in summary["workers"]["by_state"].items()) or "none"
    deliveries = summary["deliveries"]
    failures = summary["failures"]
    label = datetime.fromtimestamp(bucket_end, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f"Run digest ({label} bucket): {summary['run_status']}. "
        f"Workers {summary['workers']['total']} ({worker_states}). "
        f"Published knowledge {summary['publications']['total']} "
        f"({summary['publications']['by_workers']} by workers). "
        f"Deliveries {deliveries['delivered']}/{deliveries['total']}; "
        f"acknowledged {deliveries['acknowledged']}; incorporated {deliveries['incorporated']}. "
        f"Failure events {failures['ledger_events']}; failed workers {failures['workers']}; "
        f"failed deliveries {failures['deliveries']}; tool rejections {summary['tool_rejections']}.")


def pump_once(store, run_id, notifier, artifact, *, digest_interval=None, now=None):
    """Queue at most one requested current-bucket digest, then flush the outbox.

    `digest_interval=None` (default) drains only existing notifications. An
    explicit interval must be integral seconds >=600. The notifier still applies
    its own digest eligibility, quiet hours, rate limits, and uncertain-send
    policy; this function never enables/bypasses them. `now` affects bucket
    selection only, not the notifier's clock. Invoke periodically from a separate
    CLI/service after the model workers exit. Missed buckets are not backfilled.

    The notifier's flush may drain due notifications for other runs sharing its
    outbox. Return receipts retain their actual run IDs. Canonical run knowledge
    is read only; durable enqueue and receipts belong to the supplied notifier.
    """
    interval = _interval(digest_interval)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(artifact, str):
        raise ValueError("artifact must be a string reference")
    current = time.time() if now is None else now
    if isinstance(current, bool) or not isinstance(current, (int, float)) or not math.isfinite(current) or current < 0:
        raise ValueError("now must be a finite nonnegative Unix timestamp")
    # Fail a mistaken run reference before enqueueing or flushing a shared outbox.
    store.run(run_id)
    digest = {"requested": interval is not None, "queued": False}
    if interval is not None:
        bucket = int(current) // interval
        key_basis = [DIGEST_VERSION, run_id, interval, bucket]
        key = hashlib.sha256(json.dumps(key_basis, separators=(",", ":")).encode()).hexdigest()
        summary = summarize_run(store, run_id)
        record = notifier.enqueue(run_id, "digest", _digest_text(summary, (bucket + 1) * interval),
                                  artifact, severity="info", key=key)
        digest = {"requested": True, "interval_seconds": interval, "bucket": bucket, "key": key,
            "queued": record.get("state") in ("queued", "retry_wait", "sending"),
            "enqueue": record, "summary": summary, "policy": DIGEST_VERSION}
    receipts = notifier.flush()
    return {"run_id": run_id, "digest": digest, "receipts": receipts, "receipt_count": len(receipts)}
