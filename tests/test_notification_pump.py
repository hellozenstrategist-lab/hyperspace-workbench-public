"""Pure local fakes: no Photon connection, network, secrets, or model calls."""
import copy
from types import SimpleNamespace
import unittest

from astra_harness.notification_pump import pump_once


class Store:
    def __init__(self):
        self.status = "running"
        self.worker_rows = {"agent-a": {"state": "active"}, "agent-b": {"state": "active"},
                            "agent-c": {"state": "completed"}}
        self.event_rows = [SimpleNamespace(author_agent=a, claim="PRIVATE_CODE_DO_NOT_SUMMARIZE")
                           for a in ("agent-a", "agent-b", "agent-c", "coordinator")]
        self.delivery_rows = [{"state": "incorporated", "accepted_at": "t", "delivered_at": "t",
                               "acknowledged_at": "t", "incorporated_at": "t"} for _ in range(3)] + [
            {"state": "acknowledged", "accepted_at": "t", "delivered_at": "t", "acknowledged_at": "t"},
            {"state": "delivered", "accepted_at": "t", "delivered_at": "t"}, {"state": "failed"}]

    def run(self, run_id):
        if run_id != "run-1":
            raise KeyError(run_id)
        return {"run_id": run_id, "status": self.status}

    def workers(self, run_id):
        return self.worker_rows

    def events(self, run_id):
        return self.event_rows

    def deliveries(self, run_id):
        return self.delivery_rows

    def ledger(self, run_id):
        return [{"kind": "error"}, {"kind": "failure"}, {"kind": "tool_rejected"}]


class FakeNotifier:
    def __init__(self, backing=None, clock=0, quiet_until=0, digest_enabled=True):
        self.rows = backing if backing is not None else {}
        self.clock, self.quiet_until, self.digest_enabled = clock, quiet_until, digest_enabled
        self.flush_calls = 0

    def enqueue(self, run_id, kind, text, artifact, severity="info", key=None):
        if key in self.rows:
            return {"key": key, "state": self.rows[key]["state"], "deduplicated": True}
        state = "suppressed" if kind == "digest" and not self.digest_enabled else "queued"
        self.rows[key] = {"key": key, "run_id": run_id, "kind": kind, "text": text,
                          "artifact": artifact, "state": state}
        return {"key": key, "state": state, "deduplicated": False}

    def flush(self):
        self.flush_calls += 1
        receipts = []
        if self.clock < self.quiet_until:
            return receipts
        for row in self.rows.values():
            if row["state"] == "queued":
                row["state"] = "delivered"
                receipts.append({"key": row["key"], "run_id": row["run_id"], "status": "delivered"})
        return receipts


class NotificationPumpTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def test_default_drains_existing_backlog_without_creating_digest(self):
        notifier = FakeNotifier()
        notifier.enqueue("run-1", "completed", "Completed", "report.json", key="done")
        result = pump_once(self.store, "run-1", notifier, "report.json", now=1200)
        self.assertFalse(result["digest"]["requested"])
        self.assertEqual(result["receipt_count"], 1)
        self.assertEqual(len(notifier.rows), 1)

    def test_quiet_hours_backlog_drains_after_worker_run_exits(self):
        notifier = FakeNotifier(clock=100, quiet_until=700)
        notifier.enqueue("run-1", "completed", "Completed", "report.json", key="done")
        self.store.status = "completed"
        early = pump_once(self.store, "run-1", notifier, "report.json", now=100)
        self.assertEqual(early["receipt_count"], 0)
        notifier.clock = 701
        late = pump_once(self.store, "run-1", notifier, "report.json", now=701)
        self.assertEqual(late["receipt_count"], 1)

    def test_requested_digest_has_aggregate_counts_without_private_content(self):
        notifier = FakeNotifier()
        result = pump_once(self.store, "run-1", notifier, "report.json", digest_interval=600, now=1250)
        summary = result["digest"]["summary"]
        self.assertEqual(summary["workers"], {"total": 3, "by_state": {"active": 2, "completed": 1}})
        self.assertEqual(summary["publications"], {"total": 4, "by_workers": 3})
        self.assertEqual(summary["deliveries"]["delivered"], 5)
        self.assertEqual(summary["deliveries"]["acknowledged"], 4)
        self.assertEqual(summary["deliveries"]["incorporated"], 3)
        self.assertEqual(summary["failures"]["ledger_events"], 2)
        self.assertEqual(summary["failures"]["deliveries"], 1)
        text = next(iter(notifier.rows.values()))["text"]
        self.assertNotIn("PRIVATE_CODE", text)
        self.assertIn("acknowledged 4", text)

    def test_restart_same_bucket_has_one_durable_notification_key(self):
        backing = {}
        first = FakeNotifier(backing)
        original = pump_once(self.store, "run-1", first, "report.json", digest_interval=600, now=1250)
        original_rows = copy.deepcopy(backing)
        self.store.worker_rows["agent-a"]["state"] = "completed"
        restarted = FakeNotifier(backing)
        again = pump_once(self.store, "run-1", restarted, "report.json", digest_interval=600, now=1799)
        self.assertEqual(original["digest"]["key"], again["digest"]["key"])
        self.assertTrue(again["digest"]["enqueue"]["deduplicated"])
        self.assertEqual(again["receipt_count"], 0)
        self.assertEqual(backing, original_rows)

    def test_new_bucket_enqueues_one_digest_without_backfilling_missed_buckets(self):
        notifier = FakeNotifier()
        first = pump_once(self.store, "run-1", notifier, "report.json", digest_interval=600, now=1250)
        later = pump_once(self.store, "run-1", notifier, "report.json", digest_interval=600, now=7200)
        self.assertNotEqual(first["digest"]["key"], later["digest"]["key"])
        self.assertEqual(len(notifier.rows), 2)

    def test_digest_does_not_bypass_notifier_disabled_policy(self):
        notifier = FakeNotifier(digest_enabled=False)
        result = pump_once(self.store, "run-1", notifier, "report.json", digest_interval=600, now=1200)
        self.assertEqual(result["digest"]["enqueue"]["state"], "suppressed")
        self.assertEqual(result["receipt_count"], 0)

    def test_too_short_or_invalid_interval_rejected_before_any_notification_action(self):
        for value in (0, 599, 600.5, True, float("nan"), float("inf"), "600"):
            with self.subTest(interval=value):
                notifier = FakeNotifier()
                with self.assertRaises(ValueError):
                    pump_once(self.store, "run-1", notifier, "report.json", digest_interval=value, now=1200)
                self.assertEqual(notifier.flush_calls, 0)
                self.assertEqual(notifier.rows, {})

    def test_unknown_run_does_not_flush_shared_outbox(self):
        notifier = FakeNotifier()
        with self.assertRaises(KeyError):
            pump_once(self.store, "wrong-run", notifier, "report.json")
        self.assertEqual(notifier.flush_calls, 0)

    def test_bucket_clock_does_not_override_notifier_delivery_clock(self):
        notifier = FakeNotifier(clock=100, quiet_until=1000)
        result = pump_once(self.store, "run-1", notifier, "report.json", digest_interval=600, now=7200)
        self.assertEqual(result["digest"]["bucket"], 12)
        self.assertEqual(result["receipt_count"], 0)


if __name__ == "__main__":
    unittest.main()
