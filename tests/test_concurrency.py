from datetime import datetime, timedelta, timezone
import json

import pytest

from astra_harness.concurrency import concurrency_report


WORKERS = tuple("agent-" + str(index) for index in range(5))
BASE = datetime(2026, 9, 8, tzinfo=timezone.utc)


def row(kind, seconds, agent=WORKERS[0], **data):
    return {"kind": kind, "at": (BASE + timedelta(seconds=seconds)).isoformat(),
            "data": json.dumps({"agent": agent, **data})}


def lifetimes():
    return [row("worker_started", 0, agent) for agent in WORKERS] + [
        row("worker_completed", 20, agent, status="completed") for agent in WORKERS]


def test_five_worker_lifetimes_do_not_prove_five_http_calls():
    ledger = lifetimes()
    for index, agent in enumerate(WORKERS):
        ledger.extend([row("request_started", index, agent, request_id=str(index)),
                       row("request_completed", index + 1, agent, request_id=str(index), status="completed")])
    report = concurrency_report(ledger, WORKERS)
    assert report["worker_count"] == report["workers_completed"] == report["peak_active_workers"] == 5
    assert report["all_workers_overlap_seconds"] == 20
    assert report["http_requests"]["request_count"] == 5
    assert report["http_requests"]["peak_in_flight_requests"] == 1
    assert report["http_requests"]["all_workers_requests_overlap_seconds"] == 0


def test_five_overlapping_calls_use_actual_client_timestamps():
    ledger = lifetimes()
    for index, agent in enumerate(WORKERS):
        # Callback receipt times are serialized; payloads record true client overlap.
        ledger.extend([row("request_started", index * 2, agent, request_id=str(index), started_at=100 + index / 10),
                       row("request_completed", index * 2 + 1, agent, request_id=str(index), completed_at=102 + index / 10, status="completed")])
    http = concurrency_report(ledger, WORKERS)["http_requests"]
    assert http["status"] == "measured"
    assert http["peak_in_flight_requests"] == 5
    assert http["all_workers_requests_overlap_seconds"] == pytest.approx(1.6)
    assert http["incomplete_requests"] == http["error_requests"] == 0


def test_old_worker_events_leave_http_metrics_unmeasured():
    report = concurrency_report(lifetimes(), WORKERS)
    assert report["all_workers_overlap_seconds"] == 20
    assert report["http_requests"]["status"] == "not_measured"
    assert all(value is None for key, value in report["http_requests"].items() if key != "status")


def test_incomplete_uncertain_and_invalid_timing_do_not_invent_overlap():
    ledger = [row("worker_started", 0), row("worker_completed", 10, status="completed"),
              row("request_started", 1, request_id="incomplete"),
              row("request_uncertain", 5, request_id="incomplete", status="uncertain"),
              row("request_started", 8, request_id="reversed"),
              row("request_completed", 4, request_id="reversed", status="completed"),
              row("request_started", 2, request_id="invalid", started_at=float("nan")),
              row("request_completed", 3, request_id="invalid", status="completed"),
              row("error", 9, message="request interrupted")]
    report = concurrency_report(ledger, WORKERS[:1])
    http = report["http_requests"]
    assert http["request_count"] == 3
    assert http["incomplete_requests"] == http["error_requests"] == 1
    assert http["invalid_timing_events"] == 2
    assert http["peak_in_flight_requests"] == http["all_workers_requests_overlap_seconds"] == 0
    assert report["error_events"] == 1


def test_rate_limit_retry_counts_attempts_and_deduplicates_notifications():
    ledger = [row("request_started", 0, request_id="retry-1"),
              row("request_completed", 1, request_id="retry-1", status="rate_limited", http_status=429),
              row("request_rate_limited", 1, request_id="retry-1", http_status=429),
              row("request_started", 2, request_id="retry-2"),
              row("request_completed", 3, request_id="retry-2", status="failed"),
              row("request_failed", 3, request_id="retry-2")]
    http = concurrency_report(ledger, WORKERS[:1])["http_requests"]
    assert http["request_count"] == 2
    assert http["rate_limit_events"] == 1
    assert http["error_requests"] == 2
    assert http["incomplete_requests"] == 0
    assert http["all_workers_requests_overlap_seconds"] == 2


def test_touching_worker_lifetimes_and_zero_duration_have_no_overlap():
    ledger = [row("worker_started", 0, "a"), row("worker_completed", 1, "a"),
              row("worker_started", 1, "b"), row("worker_completed", 2, "b"),
              row("worker_started", 1, "c"), row("worker_completed", 1, "c")]
    report = concurrency_report(ledger, ["a", "b", "c"])
    assert report["peak_active_workers"] == 1
    assert report["all_workers_overlap_seconds"] == 0


def test_distinct_worker_coverage_and_repeated_turns_not_request_count():
    ledger = [row("worker_started", 0, "a", turn_id="a1"),
              row("worker_completed", 2, "a", turn_id="a1", status="completed"),
              row("worker_started", 4, "a", turn_id="a2"),
              row("worker_completed", 6, "a", turn_id="a2", status="completed"),
              row("worker_started", 1, "b"), row("worker_completed", 5, "b", status="failed"),
              row("request_started", 0, "a", request_id="one"),
              row("request_started", 0, "a", request_id="two"),
              row("request_completed", 6, "a", request_id="one"),
              row("request_completed", 6, "a", request_id="two")]
    report = concurrency_report(ledger, ["a", "b", "a"])
    assert report["workers_completed"] == report["workers_failed"] == 1
    assert report["peak_active_workers"] == 2
    assert report["all_workers_overlap_seconds"] == 2
    assert report["http_requests"]["peak_in_flight_requests"] == 2
    assert report["http_requests"]["all_workers_requests_overlap_seconds"] == 0


def test_malformed_rows_missing_events_and_scope_are_reported():
    ledger = [row("worker_started", 1, turn_id="first"),
              row("worker_completed", 0, status="completed"),
              row("worker_started", 2, "second"),
              row("request_started", 1, request_id="one"),
              row("request_started", 1, request_id="one"),
              row("request_completed", 2, request_id="orphan"),
              row("request_started", 1, "unrequested", request_id="other"),
              {"kind": "request_started", "at": "invalid", "data": "{"}]
    report = concurrency_report(ledger, [WORKERS[0], "second"])
    assert report["invalid_ledger_rows"] == 1
    assert report["invalid_worker_timing_events"] == 1
    assert report["incomplete_worker_intervals"] == 1
    assert report["http_requests"]["request_count"] == 1
    assert report["http_requests"]["incomplete_requests"] == 1
    assert report["http_requests"]["invalid_lifecycle_events"] == 2


def test_timezone_naive_timestamps_are_invalid_and_empty_selection_has_no_overlap():
    ledger = [row("worker_started", 0), row("worker_completed", 1)]
    ledger[0]["at"] = "2026-09-08T00:00:00"
    report = concurrency_report(ledger, WORKERS[:1])
    assert report["invalid_worker_timing_events"] == 1
    assert report["peak_active_workers"] == 0
    assert concurrency_report([], [])["all_workers_overlap_seconds"] == 0
