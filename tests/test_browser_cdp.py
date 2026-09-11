"""The browser boundary is tested with local endpoint names and fake CDP only."""
import asyncio
import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from astra_harness import browser_cdp


LOCAL_ENDPOINTS = {
    "attacker_browser": "http://127.0.0.1:9230",
    "victim_browser": "http://127.0.0.1:9232",
}


class FakeTransport:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.calls = []
        self.handlers = {}
        self.closed = False
        self.fail_method = None
        self.cancel_method = None
        self.response_bodies = {}
        self.url = "https://app.example/dashboard"
        self.click_text_target = "View promo codes"
        self.click_text_match_count = 1
        self.fetch_result = {
            "url": "https://app.example/v1/users/me", "status": 200, "status_text": "OK",
            "redirected": False, "response_type": "basic",
            "headers": {"content-type": "application/json", "set-cookie": "secret=hidden"},
            "body": '{"email":"owner@example.com","token":"eyJabcdef.ghijkl.mn012345"}',
            "total_body_characters": 70, "truncated_in_page": False,
            "auth_mode": "auto_bearer", "bearer_applied": True,
        }
        # Model-authored script stays opt-in per test so the default fake keeps
        # rejecting any expression the adapter was not built to send.
        self.script_eval_allowed = False
        self.script_eval_result = {"result": {"type": "string", "value": "ok"}}
        self.script_expressions = []
        self.cookies = []
        self.local_storage_entries = []
        self.session_storage_entries = []
        self.indexeddb_names = []
        self.cache_names = []
        self.storage_usage = 0
        self.storage_usage_breakdown = []
        self.node = {"nodeId": 9, "nodeType": 1, "nodeName": "INPUT",
                     "attributes": ["type", "text", "name", "query"]}

    def add_event_handler(self, method, handler):
        self.handlers.setdefault(method, []).append(handler)

    async def emit(self, method, params):
        for handler in self.handlers.get(method, []):
            result = handler(copy.deepcopy(params))
            if inspect.isawaitable(result):
                await result

    async def command(self, method, params=None, *, session_id=None):
        params = copy.deepcopy(params or {})
        self.calls.append((method, params, session_id))
        if method == self.cancel_method:
            raise asyncio.CancelledError
        if method == self.fail_method:
            raise browser_cdp.BrowserCDPError("injected uncertain transport failure")
        if method == "Target.getTargets":
            return {"targetInfos": [{"targetId": "page-1", "type": "page", "url": self.url}]}
        if method == "Target.createTarget":
            self.url = "about:blank"
            return {"targetId": "new-page"}
        if method == "Target.attachToTarget":
            return {"sessionId": "session-" + self.endpoint.rsplit(":", 1)[-1]}
        if method == "Runtime.evaluate":
            if params["expression"] == browser_cdp._PAGE_STATE_EXPRESSION:
                value = {"url": self.url, "readyState": "complete"}
            elif params["expression"] == browser_cdp._INSPECT_EXPRESSION:
                value = {
                    "url": self.url + "?token=query-secret&view=summary",
                    "title": "Research user@example.com",
                    "headings": [{"level": "h1", "text": "Dashboard"}],
                    "links": [], "buttons": [], "controls": [],
                    "visible_text": "Bearer abcdefghijklmnopqrstuvwxyz",
                }
            elif params["expression"] == browser_cdp._CLICK_TEXT_RESOLVER_EXPRESSION:
                return {"result": {"type": "function", "objectId": "click-text-resolver"}}
            elif params["expression"] == browser_cdp._FETCH_FUNCTION_EXPRESSION:
                return {"result": {"type": "function", "objectId": "fetch-function"}}
            else:
                if not self.script_eval_allowed:
                    raise AssertionError("adapter sent arbitrary JavaScript")
                self.script_expressions.append(params["expression"])
                return copy.deepcopy(self.script_eval_result)
            return {"result": {"value": value}}
        if method == "Runtime.callFunctionOn":
            if params.get("objectId") == "fetch-function":
                if (params.get("functionDeclaration") != browser_cdp._FETCH_CALL_FUNCTION
                        or params.get("awaitPromise") is not True):
                    raise AssertionError("adapter sent an unbounded fetch call")
                return {"result": {"type": "object", "value": copy.deepcopy(self.fetch_result)}}
            if (params.get("objectId") != "click-text-resolver"
                    or params.get("functionDeclaration") != browser_cdp._CLICK_TEXT_CALL_FUNCTION
                    or params.get("arguments") != [{"value": self.click_text_target}]):
                raise AssertionError("adapter sent an unbounded visible-text resolver call")
            if self.click_text_match_count == 1:
                return {"result": {"type": "object", "subtype": "node",
                                   "objectId": "click-text-node"}}
            return {"result": {"type": "number",
                               "value": 0 if self.click_text_match_count == 0 else -1}}
        if method == "Page.navigate":
            self.url = params["url"]
            return {"frameId": "main"}
        if method == "Network.getResponseBody":
            return copy.deepcopy(self.response_bodies.get(
                params["requestId"], {"body": "", "base64Encoded": False}))
        if method == "Network.getCookies":
            return {"cookies": copy.deepcopy(self.cookies)}
        if method == "DOMStorage.getDOMStorageItems":
            entries = (self.local_storage_entries if params["storageId"]["isLocalStorage"]
                       else self.session_storage_entries)
            return {"entries": copy.deepcopy(entries)}
        if method == "IndexedDB.requestDatabaseNames":
            return {"databaseNames": copy.deepcopy(self.indexeddb_names)}
        if method == "CacheStorage.requestCacheNames":
            return {"caches": copy.deepcopy(self.cache_names)}
        if method == "Storage.getUsageAndQuota":
            return {"usage": self.storage_usage, "quota": 1_000_000,
                    "usageBreakdown": copy.deepcopy(self.storage_usage_breakdown)}
        if method == "DOM.getDocument":
            return {"root": {"nodeId": 1}}
        if method == "DOM.requestNode":
            return {"nodeId": self.node["nodeId"]}
        if method == "DOM.querySelector":
            return {"nodeId": self.node["nodeId"]}
        if method == "DOM.describeNode":
            return {"node": copy.deepcopy(self.node)}
        if method == "DOM.getBoxModel":
            return {"model": {"content": [10, 20, 30, 20, 30, 40, 10, 40]}}
        return {}

    async def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self):
        self.transports = {}

    async def __call__(self, endpoint, timeout):
        transport = FakeTransport(endpoint)
        self.transports[endpoint] = transport
        return transport


class BrowserCDPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = Path(self.temp.name)
        self.adapters = []

    async def asyncTearDown(self):
        for adapter in self.adapters:
            await adapter.close()
        self.temp.cleanup()

    def policy(self, **overrides):
        values = {
            "allowed_origins": frozenset({"https://app.example"}),
            "interaction_enabled": True,
            "request_replay_enabled": True,
            "max_actions_per_session": 8,
            "max_replays_per_session": 2,
        }
        values.update(overrides)
        return browser_cdp.BrowserPolicy(**values)

    def adapter(self, *, factory=None, policy=None, artifacts=None, endpoints=None):
        factory = factory or FakeFactory()
        adapter = browser_cdp.BrowserCDPAdapter(
            endpoints or LOCAL_ENDPOINTS,
            artifacts or self.artifacts,
            policy or self.policy(),
            transport_factory=factory,
        )
        self.adapters.append(adapter)
        return adapter, factory

    async def observe_response(self, transport, request_id, resource_type, *,
                               url="https://app.example/app.js", method="GET", finished=True):
        await transport.emit("Network.requestWillBeSent", {
            "requestId": request_id, "type": resource_type,
            "request": {"url": url, "method": method}})
        await transport.emit("Network.responseReceived", {
            "requestId": request_id, "type": resource_type,
            "response": {"url": url, "status": 200, "mimeType": "text/javascript"}})
        if finished:
            await transport.emit("Network.loadingFinished", {
                "requestId": request_id, "encodedDataLength": 20})

    async def test_preflight_attaches_two_isolated_host_local_sessions(self):
        adapter, factory = self.adapter()
        status = await adapter.preflight()
        self.assertEqual({item["endpoint"] for item in status["sessions"]}, set(LOCAL_ENDPOINTS.values()))
        self.assertEqual(len(factory.transports), 2)
        self.assertEqual(len({id(item) for item in factory.transports.values()}), 2)
        for transport in factory.transports.values():
            self.assertTrue(any(method == "Target.attachToTarget" for method, _, _ in transport.calls))

    def test_endpoint_and_origin_policy_rejects_ambiguous_scope(self):
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "Two or three"):
            browser_cdp.BrowserCDPAdapter(
                {"one": "http://127.0.0.1:9230"}, self.artifacts, self.policy())
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "unknown browser session"):
            browser_cdp.BrowserCDPAdapter(
                LOCAL_ENDPOINTS, self.artifacts,
                self.policy(script_eval_sessions=frozenset({"absent_session"})))
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "unique"):
            browser_cdp.BrowserCDPAdapter(
                {"one": "http://127.0.0.1:9230", "two": "http://127.0.0.1:9230"},
                self.artifacts, self.policy())
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "Public"):
            browser_cdp.BrowserCDPAdapter(
                {"one": "http://8.8.8.8:9230", "two": "http://127.0.0.1:9232"},
                self.artifacts, self.policy())
        with self.assertRaises(browser_cdp.BrowserPolicyError):
            browser_cdp.BrowserPolicy(frozenset({"https://app.example/a/path"}))
        with self.assertRaises(browser_cdp.BrowserPolicyError):
            browser_cdp.BrowserPolicy("https://app.example")

    async def test_disallowed_navigation_never_connects_or_creates_a_journal(self):
        adapter, factory = self.adapter()
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "allowlist"):
            await adapter.navigate("attacker_browser", "https://example.com/", call_id="nav-out")
        self.assertEqual(factory.transports, {})
        self.assertFalse((self.artifacts / "browser-actions.json").exists())

    async def test_inspection_is_redacted_and_content_addressed(self):
        adapter, _ = self.adapter()
        result = await adapter.inspect("attacker_browser", call_id="inspect-1")
        path = adapter.observation_dir / f"{result['artifact_id']}.json"
        observation = adapter.get_observation(result["artifact_id"])
        self.assertEqual(result["artifact_id"], hashlib.sha256(
            browser_cdp.canonical(observation).encode()).hexdigest())
        self.assertEqual(result["artifact_id"], result["evidence_id"])
        serialized = path.read_text()
        self.assertNotIn("query-secret", serialized)
        self.assertNotIn("user@example.com", serialized)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", serialized)
        self.assertIn("REDACTED", serialized)

    async def test_completed_call_is_idempotent_without_second_browser_command(self):
        adapter, factory = self.adapter()
        first = await adapter.inspect("attacker_browser", call_id="same-call")
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        count = len([call for call in transport.calls if call[0] == "Runtime.evaluate"])
        second = await adapter.inspect("attacker_browser", call_id="same-call")
        self.assertEqual(first, second)
        self.assertEqual(count, len([call for call in transport.calls if call[0] == "Runtime.evaluate"]))
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "different"):
            await adapter.navigate("attacker_browser", "https://app.example/other", call_id="same-call")

    async def test_uncertain_side_effect_is_durable_and_never_retried(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        transport.fail_method = "Page.navigate"
        with self.assertRaisesRegex(browser_cdp.BrowserRecoveryRequired, "may have run"):
            await adapter.navigate("attacker_browser", "https://app.example/projects", call_id="uncertain-nav")
        with self.assertRaisesRegex(browser_cdp.BrowserRecoveryRequired, "not retried"):
            await adapter.navigate("attacker_browser", "https://app.example/projects", call_id="uncertain-nav")
        self.assertEqual(1, len([call for call in transport.calls if call[0] == "Page.navigate"]))
        journal = json.loads((self.artifacts / "browser-actions.json").read_text())
        self.assertEqual(journal["records"]["uncertain-nav"]["status"], "uncertain")

    async def test_action_budget_survives_adapter_restart(self):
        policy = self.policy(max_actions_per_session=1)
        first, _ = self.adapter(policy=policy)
        await first.inspect("attacker_browser", call_id="budget-1")
        await first.close()
        second, _ = self.adapter(policy=policy)
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "budget"):
            await second.inspect("attacker_browser", call_id="budget-2")
        self.assertNotIn("budget-2", json.loads((self.artifacts / "browser-actions.json").read_text())["records"])

    async def test_two_session_budgets_are_independent(self):
        adapter, _ = self.adapter(policy=self.policy(max_actions_per_session=1))
        await adapter.inspect("attacker_browser", call_id="one-a")
        await adapter.inspect("victim_browser", call_id="one-v")
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "budget"):
            await adapter.inspect("attacker_browser", call_id="two-a")

    async def test_budget_status_reports_charged_actions_and_replay_capacity(self):
        adapter, _ = self.adapter(policy=self.policy(
            max_actions_per_session=3, max_replays_per_session=2))
        self.assertEqual(adapter.budget_status("attacker_browser"), {
            "actions_used": 0,
            "actions_remaining": 3,
            "replays_used": 0,
            "replays_remaining": 2,
            "currently_eligible_replay_requests": 0,
            "replay_consumes_action": True,
            "navigate_and_click_return_snapshots": True,
        })
        await adapter.inspect("attacker_browser", call_id="budget-status-inspect")
        status = adapter.budget_status("attacker_browser")
        self.assertEqual(status["actions_used"], 1)
        self.assertEqual(status["actions_remaining"], 2)
        self.assertEqual(status["replays_used"], 0)
        self.assertEqual(status["replays_remaining"], 2)

    async def test_type_uses_input_cdp_and_never_persists_plaintext(self):
        adapter, factory = self.adapter()
        secret = "plain-value-that-must-not-persist"
        result = await adapter.type_text("attacker_browser", "input[name=query]", secret, call_id="type-1")
        observation = adapter.get_observation(result["artifact_id"])
        self.assertEqual(observation["data"]["interaction"]["inserted_characters"], len(secret))
        all_durable = "\n".join(path.read_text() for path in self.artifacts.rglob("*.json"))
        self.assertNotIn(secret, all_durable)
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        inserted = [params for method, params, _ in transport.calls if method == "Input.insertText"]
        self.assertEqual(inserted, [{"text": secret}])
        self.assertFalse(any(method == "Runtime.evaluate" for method, _, _ in transport.calls))

    async def test_bounded_fetch_uses_session_auth_without_exposing_token(self):
        adapter, factory = self.adapter()
        observed = await adapter.fetch(
            "attacker_browser", "https://app.example/v1/users/me", "GET",
            {"Accept": "application/json"}, "", "auto_bearer", call_id="fetch-me")
        artifact = adapter.get_observation(observed["artifact_id"])
        response = artifact["data"]["fetch"]["response"]
        self.assertEqual(response["status"], 200)
        self.assertTrue(response["bearer_applied"])
        self.assertEqual(artifact["data"]["fetch"]["request"]["header_names"], ["Accept"])
        serialized = json.dumps(artifact)
        self.assertNotIn("owner@example.com", serialized)
        self.assertNotIn("eyJabcdef", serialized)
        self.assertNotIn("secret=hidden", serialized)
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        call = next(params for method, params, _ in transport.calls
                    if method == "Runtime.callFunctionOn" and params.get("objectId") == "fetch-function")
        self.assertEqual(call["arguments"][:5], [
            {"value": "https://app.example/v1/users/me"}, {"value": "GET"},
            {"value": {"Accept": "application/json"}}, {"value": ""},
            {"value": "auto_bearer"},
        ])

    async def test_fetch_rejects_cross_origin_body_and_sensitive_headers_before_cdp(self):
        adapter, factory = self.adapter()
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "outside"):
            await adapter.fetch("attacker_browser", "https://api.app.example/v1/me", "GET", {}, "",
                                "cookies", call_id="fetch-cross-origin")
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "empty body"):
            await adapter.fetch("attacker_browser", "https://app.example/v1/me", "GET", {}, "x",
                                "cookies", call_id="fetch-get-body")
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "forbidden"):
            await adapter.fetch("attacker_browser", "https://app.example/v1/me", "GET",
                                {"Authorization": "Bearer model-secret"}, "", "cookies",
                                call_id="fetch-auth-header")
        self.assertEqual(factory.transports, {})

    async def test_password_typing_is_blocked_before_text_reaches_chrome(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        transport.node["attributes"] = ["type", "password", "name", "password"]
        with self.assertRaises(browser_cdp.BrowserRecoveryRequired):
            await adapter.type_text("attacker_browser", "input", "never-send-me", call_id="password")
        self.assertFalse(any(method == "Input.insertText" for method, _, _ in transport.calls))
        self.assertNotIn("never-send-me", (self.artifacts / "browser-actions.json").read_text())

    async def test_click_uses_dom_and_mouse_only_plus_fixed_snapshot(self):
        adapter, factory = self.adapter()
        result = await adapter.click("victim_browser", "button.submit", call_id="click-1")
        self.assertEqual(result["action"], "click")
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        methods = [method for method, _, _ in transport.calls]
        self.assertIn("DOM.querySelector", methods)
        self.assertEqual(methods.count("Input.dispatchMouseEvent"), 2)
        expressions = [params["expression"] for method, params, _ in transport.calls
                       if method == "Runtime.evaluate"]
        self.assertEqual(expressions, [browser_cdp._INSPECT_EXPRESSION])

    async def test_click_text_uses_fixed_resolver_exact_node_and_hides_target_in_journal(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        transport.node = {"nodeId": 12, "nodeType": 1, "nodeName": "BUTTON",
                          "attributes": ["type", "button"]}
        raw_target = "  View \n promo   codes  "
        result = await adapter.click_text(
            "victim_browser", raw_target, call_id="click-text-1")
        self.assertEqual(result["action"], "click_text")
        observation = adapter.get_observation(result["artifact_id"])
        interaction = observation["data"]["interaction"]
        self.assertEqual(interaction["kind"], "click_text")
        self.assertEqual(interaction["target_text_sha256"], "[REDACTED_OPAQUE]")
        self.assertEqual(interaction["target_text_length"], len(transport.click_text_target))
        methods = [method for method, _, _ in transport.calls]
        self.assertEqual(methods.count("Runtime.callFunctionOn"), 1)
        self.assertEqual(methods.count("DOM.requestNode"), 1)
        self.assertEqual(methods.count("Input.dispatchMouseEvent"), 2)
        expressions = [params["expression"] for method, params, _ in transport.calls
                       if method == "Runtime.evaluate"]
        self.assertEqual(expressions, [browser_cdp._CLICK_TEXT_RESOLVER_EXPRESSION,
                                       browser_cdp._INSPECT_EXPRESSION])
        journal = (self.artifacts / "browser-actions.json").read_text()
        self.assertNotIn(transport.click_text_target, journal)
        self.assertNotIn(raw_target, journal)
        record = json.loads(journal)["records"]["click-text-1"]
        self.assertRegex(record["intent_digest"], r"^[0-9a-f]{64}$")

    async def test_click_text_zero_and_ambiguous_matches_are_safe_durable_rejections(self):
        adapter, factory = self.adapter(policy=self.policy(max_actions_per_session=1))
        await adapter.start()
        for session, count, call_id in (
                ("attacker_browser", 0, "click-text-zero"),
                ("victim_browser", 2, "click-text-ambiguous")):
            transport = factory.transports[LOCAL_ENDPOINTS[session]]
            transport.click_text_match_count = count
            before = len(transport.calls)
            with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "exactly one"):
                await adapter.click_text(session, transport.click_text_target, call_id=call_id)
            self.assertFalse(any(method == "Input.dispatchMouseEvent"
                                 for method, _, _ in transport.calls))
            first_count = len(transport.calls)
            transport.click_text_match_count = 1
            with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "exactly one"):
                await adapter.click_text(session, transport.click_text_target, call_id=call_id)
            self.assertEqual(first_count, len(transport.calls))
            self.assertGreater(first_count, before)
            with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "budget"):
                await adapter.inspect(session, call_id=call_id + "-budget")
        records = json.loads((self.artifacts / "browser-actions.json").read_text())["records"]
        for call_id in ("click-text-zero", "click-text-ambiguous"):
            self.assertEqual(records[call_id]["status"], "rejected")
            self.assertEqual(records[call_id]["error_kind"], "target_not_unique")
            self.assertNotIn("error_type", records[call_id])
        durable = (self.artifacts / "browser-actions.json").read_text()
        self.assertNotIn("View promo codes", durable)
        self.assertEqual([], list(adapter.observation_dir.glob("*.json")))

    async def test_click_text_rejects_unbounded_or_control_targets_before_connecting(self):
        adapter, factory = self.adapter()
        for index, target in enumerate(("", "   \n ", "x" * 301, "safe\x00unsafe")):
            with self.assertRaises(browser_cdp.BrowserPolicyError):
                await adapter.click_text(
                    "attacker_browser", target, call_id=f"invalid-click-text-{index}")
        self.assertEqual(factory.transports, {})
        self.assertFalse((self.artifacts / "browser-actions.json").exists())

    async def test_replay_accepts_only_observed_same_origin_read_xhr_and_has_own_budget(self):
        adapter, factory = self.adapter(policy=self.policy(max_replays_per_session=1))
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        await transport.emit("Network.requestWillBeSent", {
            "requestId": "ok", "type": "XHR",
            "request": {"url": "https://app.example/api/projects?token=hidden", "method": "GET"}})
        await transport.emit("Network.requestWillBeSent", {
            "requestId": "post", "type": "XHR",
            "request": {"url": "https://app.example/api/projects", "method": "POST"}})
        await adapter.replay_request("attacker_browser", "ok", call_id="replay-1")
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "Only observed"):
            await adapter.replay_request("attacker_browser", "post", call_id="replay-post")
        await transport.emit("Network.requestWillBeSent", {
            "requestId": "second", "type": "XHR",
            "request": {"url": "https://app.example/api/other", "method": "HEAD"}})
        with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, "replay budget"):
            await adapter.replay_request("attacker_browser", "second", call_id="replay-2")
        replayed = [params for method, params, _ in transport.calls if method == "Network.replayXHR"]
        self.assertEqual(replayed, [{"requestId": "ok"}])
        inspected = await adapter.inspect("attacker_browser", call_id="after-replay")
        serialized = json.dumps(adapter.get_observation(inspected["artifact_id"]))
        self.assertNotIn("hidden", serialized)

    async def test_read_response_accepts_each_bounded_textual_resource_type(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        for index, resource_type in enumerate(("Document", "Script", "XHR", "Fetch"), 1):
            request_id = f"body-{index}"
            await self.observe_response(transport, request_id, resource_type)
            transport.response_bodies[request_id] = {
                "body": (f"const owner = 'user@example.com'; // Bearer abcdefghijklmnop {resource_type}; "
                         'const access_token="tiny-secret";'),
                "base64Encoded": False,
            }
            result = await adapter.read_response(
                "attacker_browser", request_id, call_id=f"read-{resource_type.lower()}")
            observation = adapter.get_observation(result["artifact_id"])
            response = observation["data"]["response"]
            self.assertEqual(response["request"]["resource_type"], resource_type)
            self.assertEqual(response["total_decoded_bytes"],
                             len(transport.response_bodies[request_id]["body"].encode()))
            self.assertFalse(response["truncated"])
            serialized = json.dumps(observation)
            self.assertNotIn("user@example.com", serialized)
            self.assertNotIn("abcdefghijklmnop", serialized)
            self.assertNotIn("tiny-secret", serialized)

    async def test_read_response_decodes_base64_before_enforcing_byte_cap(self):
        adapter, factory = self.adapter(policy=self.policy(max_response_body_bytes=9))
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        await self.observe_response(transport, "encoded", "XHR", url="https://app.example/api/state")
        transport.response_bodies["encoded"] = {
            "body": base64.b64encode("π-value".encode()).decode(), "base64Encoded": True}
        result = await adapter.read_response("victim_browser", "encoded", call_id="read-base64")
        response = adapter.get_observation(result["artifact_id"])["data"]["response"]
        self.assertEqual(response["body"], "π-value")
        self.assertEqual(response["total_decoded_bytes"], 8)
        self.assertEqual(response["returned_body_bytes"], 8)
        self.assertFalse(response["truncated"])
        self.assertTrue(response["was_base64_encoded"])

    async def test_read_response_rejects_missing_third_party_nontext_and_incomplete_ids(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        await self.observe_response(
            transport, "third-party", "Script", url="https://example.com/tracker.js")
        await self.observe_response(transport, "image", "Image", url="https://app.example/logo.png")
        await self.observe_response(transport, "pending", "Fetch", finished=False)
        for request_id, reason in (("missing", "Only observed"), ("third-party", "Only observed"),
                                   ("image", "Only observed"), ("pending", "completed")):
            with self.assertRaisesRegex(browser_cdp.BrowserPolicyError, reason):
                await adapter.read_response(
                    "attacker_browser", request_id, call_id=f"reject-{request_id}")
        self.assertFalse(any(method == "Network.getResponseBody" for method, _, _ in transport.calls))
        journal_path = self.artifacts / "browser-actions.json"
        self.assertFalse(journal_path.exists())

    async def test_read_response_returns_a_bounded_prefix_for_large_bundles(self):
        adapter, factory = self.adapter(policy=self.policy(max_response_body_bytes=8))
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        await self.observe_response(transport, "too-large", "Script")
        transport.response_bodies["too-large"] = {
            "body": "0123456789-large-javascript-bundle", "base64Encoded": False}
        result = await adapter.read_response("attacker_browser", "too-large", call_id="oversize")
        response = adapter.get_observation(result["artifact_id"])["data"]["response"]
        self.assertEqual(response["body"], "01234567")
        self.assertEqual(response["returned_body_bytes"], 8)
        self.assertEqual(response["total_decoded_bytes"], 34)
        self.assertTrue(response["truncated"])
        self.assertEqual(1, len([call for call in transport.calls if call[0] == "Network.getResponseBody"]))
        journal = json.loads((self.artifacts / "browser-actions.json").read_text())
        self.assertEqual(journal["records"]["oversize"]["status"], "completed")

    async def test_read_response_rejects_invalid_base64_without_artifact(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        await self.observe_response(transport, "bad-base64", "Document")
        transport.response_bodies["bad-base64"] = {"body": "%%%not-base64%%%", "base64Encoded": True}
        with self.assertRaisesRegex(browser_cdp.BrowserReadUnavailable, "could not be read safely"):
            await adapter.read_response("victim_browser", "bad-base64", call_id="invalid-base64")
        self.assertEqual([], list(adapter.observation_dir.glob("*.json")))
        journal = json.loads((self.artifacts / "browser-actions.json").read_text())
        self.assertEqual(journal["records"]["invalid-base64"]["status"], "rejected")
        self.assertEqual(journal["records"]["invalid-base64"]["reason"],
                         browser_cdp._RESPONSE_READ_REJECTION)

    async def test_read_response_transport_failure_is_a_cached_safe_rejection(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        await self.observe_response(transport, "transport-error", "Fetch")
        transport.fail_method = "Network.getResponseBody"
        for _ in range(2):
            with self.assertRaisesRegex(browser_cdp.BrowserReadUnavailable, "could not be read safely"):
                await adapter.read_response(
                    "attacker_browser", "transport-error", call_id="safe-transport-rejection")
        self.assertEqual(1, len([call for call in transport.calls if call[0] == "Network.getResponseBody"]))
        record = json.loads((self.artifacts / "browser-actions.json").read_text())[
            "records"]["safe-transport-rejection"]
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["error_kind"], "read_unavailable")
        self.assertNotIn("error_type", record)

    async def test_read_response_cancellation_propagates_after_durable_safe_rejection(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        await self.observe_response(transport, "cancelled-body", "Document")
        transport.cancel_method = "Network.getResponseBody"
        with self.assertRaises(asyncio.CancelledError):
            await adapter.read_response(
                "victim_browser", "cancelled-body", call_id="cancelled-response-read")
        record = json.loads((self.artifacts / "browser-actions.json").read_text())[
            "records"]["cancelled-response-read"]
        self.assertEqual(record["status"], "rejected")
        self.assertEqual(record["reason"], browser_cdp._RESPONSE_READ_REJECTION)
        with self.assertRaisesRegex(browser_cdp.BrowserReadUnavailable, "could not be read safely"):
            await adapter.read_response(
                "victim_browser", "cancelled-body", call_id="cancelled-response-read")
        self.assertEqual(1, len([call for call in transport.calls if call[0] == "Network.getResponseBody"]))

    async def test_legacy_uncertain_response_reads_normalize_to_safe_rejections(self):
        journal = {
            "schema_version": 1,
            "records": {
                "legacy-started": {"session": "attacker_browser", "action": "read_response",
                                   "intent_digest": "a" * 64, "status": "started"},
                "legacy-uncertain": {"session": "victim_browser", "action": "read_response",
                                     "intent_digest": "b" * 64, "status": "uncertain",
                                     "error_type": "BrowserCDPError"},
                "real-side-effect": {"session": "attacker_browser", "action": "navigate",
                                     "intent_digest": "c" * 64, "status": "uncertain"},
            },
        }
        browser_cdp.atomic_json(self.artifacts / "browser-actions.json", journal)
        adapter, _ = self.adapter()
        normalized = json.loads((self.artifacts / "browser-actions.json").read_text())["records"]
        for call_id in ("legacy-started", "legacy-uncertain"):
            self.assertEqual(normalized[call_id]["status"], "rejected")
            self.assertEqual(normalized[call_id]["reason"], browser_cdp._RESPONSE_READ_REJECTION)
            self.assertEqual(normalized[call_id]["error_kind"], "read_unavailable")
            self.assertNotIn("error_type", normalized[call_id])
        self.assertEqual(normalized["real-side-effect"]["status"], "uncertain")

    async def test_observation_reader_rejects_tampering(self):
        adapter, _ = self.adapter()
        result = await adapter.inspect("attacker_browser", call_id="integrity")
        path = adapter.observation_dir / f"{result['artifact_id']}.json"
        value = json.loads(path.read_text())
        value["data"] = {"tampered": True}
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(browser_cdp.BrowserCDPError, "integrity"):
            adapter.get_observation(result["artifact_id"])

    async def test_fetch_guard_fails_disallowed_document_before_it_loads(self):
        adapter, factory = self.adapter()
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        await transport.emit("Fetch.requestPaused", {
            "requestId": "fetch-1", "request": {"url": "https://example.com/escape"}})
        await transport.emit("Fetch.requestPaused", {
            "requestId": "fetch-2", "request": {"url": "https://app.example/ok"}})
        controls = [(method, params) for method, params, _ in transport.calls if method.startswith("Fetch.")]
        self.assertIn(("Fetch.failRequest", {"requestId": "fetch-1", "errorReason": "BlockedByClient"}), controls)
        self.assertIn(("Fetch.continueRequest", {"requestId": "fetch-2"}), controls)

    def test_tool_specs_only_advertise_enabled_bounded_capabilities(self):
        inspect_only, _ = self.adapter(policy=self.policy(
            interaction_enabled=False, request_replay_enabled=False))
        self.assertEqual([item["function"]["name"] for item in inspect_only.tool_specs()],
                         ["browser_inspect", "browser_read_response"])
        full, _ = self.adapter(artifacts=self.artifacts / "other")
        names = [item["function"]["name"] for item in full.tool_specs()]
        self.assertEqual(names, ["browser_inspect", "browser_read_response", "browser_navigate", "browser_click",
                                 "browser_click_text", "browser_type", "browser_fetch", "browser_replay_request"])
        for item in full.tool_specs():
            parameters = item["function"]["parameters"]
            self.assertFalse(parameters["additionalProperties"])
            self.assertEqual(parameters["properties"]["session"]["enum"], list(LOCAL_ENDPOINTS))
            self.assertNotIn("call_id", parameters["properties"])
        runtime_tools = full.runtime_tools()
        self.assertEqual([tool["name"] for tool in runtime_tools], names)
        self.assertTrue(all("inputSchema" in tool for tool in runtime_tools))

    async def test_script_eval_is_rejected_on_sessions_policy_did_not_designate(self):
        adapter, factory = self.adapter(
            policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
        await adapter.start()
        with self.assertRaises(browser_cdp.BrowserPolicyError):
            await adapter.dispatch_tool(
                "browser_evaluate",
                {"session": "victim_browser", "code": "document.cookie"},
                call_id="eval-denied")
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        self.assertEqual(transport.script_expressions, [])

    async def test_script_eval_fails_closed_when_the_designated_session_holds_state(self):
        for cookie_state in ([{"name": "session", "value": "abc"}], None):
            with self.subTest(cookies=cookie_state):
                adapter, factory = self.adapter(
                    artifacts=self.artifacts / f"state-{cookie_state is None}",
                    policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
                await adapter.start()
                transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
                transport.script_eval_allowed = True
                transport.cookies = cookie_state
                with self.assertRaises(browser_cdp.BrowserPolicyError) as caught:
                    await adapter.dispatch_tool(
                        "browser_evaluate",
                        {"session": "attacker_browser", "code": "1+1"},
                        call_id=f"eval-stateful-{cookie_state is None}")
                self.assertEqual(str(caught.exception), browser_cdp._STATEFUL_SESSION_REJECTION)
                self.assertEqual(transport.script_expressions, [])

    async def test_script_eval_fails_closed_for_every_browser_storage_tier(self):
        states = [
            ("local", "local_storage_entries", [["token", "secret"]]),
            ("session", "session_storage_entries", [["token", "secret"]]),
            ("indexeddb", "indexeddb_names", ["auth"]),
            ("cache", "cache_names", [{"cacheId": "auth-cache"}]),
            ("persistent", "storage_usage_breakdown",
             [{"storageType": "service_workers", "usage": 1}]),
        ]
        for label, attribute, value in states:
            with self.subTest(storage=label):
                adapter, factory = self.adapter(
                    artifacts=self.artifacts / f"storage-{label}",
                    policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
                await adapter.start()
                transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
                transport.script_eval_allowed = True
                setattr(transport, attribute, value)
                if label == "persistent":
                    transport.storage_usage = 1
                with self.assertRaises(browser_cdp.BrowserPolicyError) as caught:
                    await adapter.dispatch_tool(
                        "browser_evaluate", {"session": "attacker_browser", "code": "1+1"},
                        call_id=f"eval-{label}")
                self.assertEqual(str(caught.exception), browser_cdp._STATEFUL_SESSION_REJECTION)
                self.assertEqual(transport.script_expressions, [])

    async def test_script_eval_state_gate_runs_inside_the_session_lock(self):
        adapter, factory = self.adapter(
            policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
        await adapter.start()
        session = adapter.sessions["attacker_browser"]
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        transport.script_eval_allowed = True
        async with session.lock:
            task = asyncio.create_task(adapter.dispatch_tool(
                "browser_evaluate", {"session": "attacker_browser", "code": "1+1"},
                call_id="eval-lock-order"))
            await asyncio.sleep(0)
            self.assertFalse(any(method == "Network.getCookies" for method, _, _ in transport.calls))
        await task
        self.assertEqual(transport.script_expressions, ["1+1"])

    async def test_script_eval_runs_redacted_only_on_a_stateless_designated_session(self):
        adapter, factory = self.adapter(
            policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
        specs = {item["function"]["name"]: item["function"]["parameters"]
                 for item in adapter.tool_specs()}
        self.assertEqual(specs["browser_evaluate"]["properties"]["session"]["enum"],
                         ["attacker_browser"])
        self.assertEqual(specs["browser_inspect"]["properties"]["session"]["enum"],
                         list(LOCAL_ENDPOINTS))
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        transport.script_eval_allowed = True
        transport.script_eval_result = {
            "result": {"type": "string", "value": "Bearer abcdefghijklmnopqrstuvwxyz"}}
        result = await adapter.dispatch_tool(
            "browser_evaluate",
            {"session": "attacker_browser", "code": "document.title"},
            call_id="eval-allowed")
        self.assertEqual(result["action"], "evaluate")
        self.assertEqual(transport.script_expressions, ["document.title"])
        observation = adapter.get_observation(result["artifact_id"])
        self.assertIn("[REDACTED_CREDENTIAL]",
                      observation["data"]["evaluation"]["value"])
        self.assertEqual(adapter.sessions["attacker_browser"].actions_started, 1)

    async def test_script_eval_is_refused_once_the_session_gains_state(self):
        adapter, factory = self.adapter(
            policy=self.policy(script_eval_sessions=frozenset({"attacker_browser"})))
        await adapter.start()
        transport = factory.transports[LOCAL_ENDPOINTS["attacker_browser"]]
        transport.script_eval_allowed = True
        await adapter.dispatch_tool(
            "browser_evaluate", {"session": "attacker_browser", "code": "1+1"},
            call_id="eval-before-login")
        transport.cookies = [{"name": "session", "value": "abc"}]
        with self.assertRaises(browser_cdp.BrowserPolicyError):
            await adapter.dispatch_tool(
                "browser_evaluate", {"session": "attacker_browser", "code": "2+2"},
                call_id="eval-after-login")
        self.assertEqual(transport.script_expressions, ["1+1"])

    async def test_script_eval_is_not_advertised_without_a_designated_session(self):
        adapter, _ = self.adapter()
        names = [item["function"]["name"] for item in adapter.tool_specs()]
        self.assertNotIn("browser_evaluate", names)

    async def test_runtime_dispatch_uses_host_call_id(self):
        adapter, factory = self.adapter()
        result = await adapter.dispatch_tool(
            "browser_inspect", {"session": "attacker_browser"}, call_id="host-call-1")
        self.assertEqual(result["action"], "inspect")
        transport = factory.transports[LOCAL_ENDPOINTS["victim_browser"]]
        transport.node = {"nodeId": 13, "nodeType": 1, "nodeName": "A",
                          "attributes": ["href", "/promo"]}
        clicked = await adapter.dispatch_tool(
            "browser_click_text",
            {"session": "victim_browser", "text": transport.click_text_target},
            call_id="host-call-text")
        self.assertEqual(clicked["action"], "click_text")
        journal = json.loads((self.artifacts / "browser-actions.json").read_text())
        self.assertIn("host-call-1", journal["records"])
        self.assertIn("host-call-text", journal["records"])

    def test_bridge_websocket_rewrite_keeps_devtools_path_and_configured_endpoint(self):
        rewritten = browser_cdp._rewrite_websocket_url(
            "ws://127.0.0.1:9230/devtools/browser/browser-id?x=1",
            "http://172.17.0.1:9231")
        self.assertEqual(rewritten, "ws://172.17.0.1:9231/devtools/browser/browser-id?x=1")


if __name__ == "__main__":
    unittest.main()
