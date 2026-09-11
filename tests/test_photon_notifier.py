from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from astra_harness.photon_notifier import NotificationPolicy, PhotonNotifier
from astra_harness.mock_photon import MockPhotonServer


class Clock:
    def __init__(self, hour=12):
        self.now = datetime(2026, 9, 7, hour, tzinfo=ZoneInfo("America/Phoenix")).timestamp()

    def __call__(self):
        return self.now


class PhotonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = self.root / "state.json"
        self.state.write_text(json.dumps({"home_space": "PRIVATE-RECIPIENT-FIXTURE", "unrelated": "not used"}))
        self.clock = Clock()
        self.server = MockPhotonServer(self.root / "mock.db").start()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            client.close()
        self.server.close()
        self.tmp.cleanup()

    def client(self, *, policy=None, provider_idempotency=False):
        client = PhotonNotifier(self.root / "outbox.db", base_url=self.server.base_url,
                                state_path=self.state, clock=self.clock,
                                random_value=lambda: 0.5,
                                policy=policy or NotificationPolicy(min_interval_seconds=0),
                                provider_idempotency=provider_idempotency)
        self.clients.append(client)
        return client

    def test_dedup_survives_reopen_and_receipts_hide_recipient(self):
        client = self.client()
        one = client.enqueue("run", "start", "Starting", "artifact.json")
        self.assertEqual(client.flush()[0]["status"], "delivered")
        client.close()
        self.clients.remove(client)
        reopened = self.client()
        two = reopened.enqueue("run", "start", "Different prose", "other.json")
        self.assertEqual(one["key"], two["key"])
        self.assertTrue(two["deduplicated"])
        self.assertEqual(reopened.flush(), [])
        self.assertEqual(len(self.server.deliveries()), 1)
        self.assertNotIn("PRIVATE-RECIPIENT-FIXTURE", json.dumps(reopened.receipts()))
        self.assertNotIn("PRIVATE-RECIPIENT-FIXTURE", json.dumps(self.server.deliveries()))

    def test_policy_suppresses_noise_digest_and_low_severity(self):
        client = self.client(policy=NotificationPolicy(min_severity="warning"))
        for kind, severity in (("routine_thought", "warning"), ("digest", "warning"), ("start", "info")):
            self.assertEqual(client.enqueue("run", kind, "fixture", "", severity)["state"], "suppressed")
        self.assertEqual(client.flush(), [])
        self.assertEqual(self.server.attempts(), [])

    def test_quiet_hours_and_rate_limit(self):
        self.clock.now = Clock(23).now
        client = self.client(policy=NotificationPolicy(max_per_hour=1, min_interval_seconds=0))
        client.enqueue("run", "start", "quiet start", "")
        self.assertEqual(client.flush()[0]["code"], "quiet_hours")
        client.enqueue("run", "block", "urgent block", "", severity="error")
        self.assertEqual(client.flush()[0]["status"], "delivered")
        client.enqueue("run", "permanent_fail", "urgent failure", "", severity="error")
        self.assertEqual(client.flush()[0]["code"], "hourly_rate_limit")
        self.assertEqual(len(self.server.deliveries()), 1)
        self.clock.now += 3601
        self.assertEqual(client.flush()[0]["status"], "delivered")

    def test_reopened_policy_applies_before_sending_queued_events(self):
        original = self.client(policy=NotificationPolicy(digest_enabled=True))
        original.enqueue("run", "digest", "optional digest", "")
        reopened = self.client(policy=NotificationPolicy(digest_enabled=False))
        self.assertEqual(reopened.flush()[0]["code"], "digest_disabled")
        self.assertEqual(self.server.attempts(), [])

    def test_known_not_sent_retries_bounded_and_durable(self):
        self.server.failures = ["unavailable"] * 4
        client = self.client(policy=NotificationPolicy(min_interval_seconds=0, max_attempts=2))
        client.enqueue("run", "completed", "done", "")
        self.assertEqual(client.flush()[0]["status"], "retry_wait")
        self.assertEqual(client.flush(), [])
        self.clock.now += 5
        self.assertEqual(client.flush()[0]["status"], "retry_wait")
        self.clock.now += 10
        self.assertEqual(client.flush()[0]["code"], "attempts_exhausted")
        self.assertEqual(len(self.server.attempts()), 2)
        self.assertEqual(self.server.deliveries(), [])

    def test_unknown_after_accept_quarantined_without_retry(self):
        self.server.failures = ["disconnect_after_accept"]
        client = self.client()
        client.enqueue("run", "completed", "done", "")
        self.assertEqual(client.flush()[0]["status"], "quarantined")
        self.assertEqual(len(self.server.deliveries()), 1)
        self.clock.now += 10000
        self.assertEqual(self.client().flush(), [])
        self.assertEqual(len(self.server.attempts()), 1)

    def test_mock_provider_idempotency_handles_lost_ack_and_restart(self):
        self.server.failures = ["ambiguous_after_accept"]
        client = self.client(provider_idempotency=True)
        client.enqueue("run", "completed", "done", "")
        self.assertEqual(client.flush()[0]["status"], "retry_wait")
        self.server.close()
        self.server = MockPhotonServer(self.root / "mock.db").start()
        client.base_url = self.server.base_url
        self.clock.now += 5
        self.assertEqual(client.flush()[0]["status"], "delivered")
        self.assertEqual(len(self.server.deliveries()), 1)
        self.assertEqual(len(self.server.attempts()), 2)

    def test_expired_send_lease_is_quarantined_after_crash(self):
        client = self.client()
        row = client.enqueue("run", "start", "starting", "")
        client.db.execute("UPDATE notifications SET state='sending',attempts=1,lease_until=? WHERE key=?", (self.clock.now - 1, row["key"]))
        self.assertEqual(self.client().flush()[0]["status"], "quarantined")
        self.assertEqual(self.server.attempts(), [])

    def test_two_flushers_do_not_duplicate(self):
        client = self.client()
        client.enqueue("run", "start", "starting", "")

        def flush_in_thread(_):
            other = PhotonNotifier(self.root / "outbox.db", base_url=self.server.base_url,
                                   state_path=self.state, clock=self.clock,
                                   policy=NotificationPolicy(min_interval_seconds=0))
            try:
                return other.flush()
            finally:
                other.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(flush_in_thread, range(2)))
        self.assertEqual(len(self.server.attempts()), 1)

    def test_missing_recipient_permanent_failure_and_no_send(self):
        self.state.write_text('{}')
        client = self.client()
        client.enqueue("run", "start", "starting", "")
        self.assertEqual(client.flush()[0]["code"], "recipient_unconfigured")
        self.assertEqual(self.server.attempts(), [])

    def test_reject_nonloopback_and_configuration_errors(self):
        with self.assertRaises(ValueError):
            PhotonNotifier(self.root / "other.db", base_url="https://example.com")
        with self.assertRaises(ValueError):
            NotificationPolicy(quiet_start_hour=None, quiet_end_hour=7)


if __name__ == "__main__":
    unittest.main()
