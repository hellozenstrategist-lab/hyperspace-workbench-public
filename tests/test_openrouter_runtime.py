"""OpenRouter tests use fake HTTPS/transport; never contact a paid provider."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from astra_harness.codex_runtime import RecoveryRequired, RuntimeFailure, ToolInputError
from astra_harness.openrouter_runtime import (
    FAILURE_PROVIDER_COMPLETION_ERROR,
    FAILURE_TOOL_CALL_BATCH_EXCEEDED,
    OpenRouterRuntime,
)
from astra_harness.schema import worker_ids

MODEL = "example/research-model"
KEY = "sk-test-private-shared-key-never-log"
AGENTS = worker_ids(5)
TOOLS = [{"name": "observe", "description": "Record bounded research observations",
          "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                          "required": ["text"], "additionalProperties": False}}]


def completion(content="done", calls=None, finish=None):
    message = {"role": "assistant", "content": None if calls else content}
    if calls:
        message["tool_calls"] = calls
        message["reasoning_details"] = [{"type": "reasoning.text", "text": "test reasoning"}]
    return {"id": "gen-test", "model": MODEL, "provider": "fake-provider",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [{"finish_reason": finish or ("tool_calls" if calls else "stop"), "message": message}]}


def tool_call(name="observe", call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": '{"text":"test evidence"}'}}


class OpenRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.instances, self.events, self.requests = [], [], []
        self.env = patch.dict("os.environ", {"OPENROUTER_API_KEY": KEY}, clear=True)
        self.env.start()
        self.settings = patch.dict("sys.modules", {"astra_harness.api_settings":
                                  SimpleNamespace(openrouter_settings=lambda: {})})
        self.settings.start()

    async def asyncTearDown(self):
        for instance in self.instances:
            await instance.close()
        self.env.stop()
        self.settings.stop()
        self.tmp.cleanup()

    async def event(self, kind, agent, data):
        self.events.append((kind, agent, copy.deepcopy(data)))

    def make(self, **kwargs):
        instance = OpenRouterRuntime(self.path / "state.json", self.path / "runtime",
            self.event, kwargs.pop("on_tool", AsyncMock(return_value={"recorded": True})),
            model=kwargs.pop("model", MODEL), agents=kwargs.pop("agents", AGENTS), **kwargs)
        self.instances.append(instance)
        return instance

    async def prepare(self, instance):
        await instance.start()
        await instance.create_workers(TOOLS, "Use only registered research tools",
                                      {a: self.path / a for a in instance.agents})

    def transport(self, instance, responder=None):
        async def fake(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            self.requests.append((agent, copy.deepcopy(payload)))
            data = await responder(agent, payload) if responder else completion(agent)
            return {"http_status": 200, "completed_at": time.time(), "body": json.dumps(data).encode()}
        instance._http = fake

    async def launch(self, instance):
        await instance.start_turns({a: "Task for " + a for a in instance.agents})
        return await instance.wait(timeout=5)

    async def test_tool_requests_do_not_require_parallel_call_support(self):
        instance = self.make(agents=worker_ids(2))
        await self.prepare(instance)

        async def responder(agent, payload):
            self.assertNotIn("parallel_tool_calls", payload)
            self.assertTrue(payload["provider"]["require_parameters"])
            self.assertTrue(payload["provider"]["allow_fallbacks"])
            self.assertEqual(payload["model"], MODEL)
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertTrue(payload["tools"])
            if not any(m["role"] == "tool" for m in payload["messages"]):
                return completion(calls=[tool_call(call_id="first"), tool_call(call_id="second")])
            self.assertEqual([m["tool_call_id"] for m in payload["messages"]
                              if m["role"] == "tool"], ["first", "second"])
            return completion()

        self.transport(instance, responder)
        await self.launch(instance)
        self.assertTrue(all(w["status"] == "completed" for w in instance.workers.values()))

    async def test_per_worker_tool_projection_reaches_only_its_worker_payload(self):
        agents = worker_ids(2)
        inspect_tool = copy.deepcopy(TOOLS[0])
        inspect_tool["name"] = "browser_inspect"
        eval_tool = copy.deepcopy(TOOLS[0])
        eval_tool["name"] = "browser_evaluate"
        instance = self.make(agents=agents)
        await instance.start()
        await instance.create_workers(
            {agents[0]: [inspect_tool], agents[1]: [inspect_tool, eval_tool]},
            "Use only your registered tools",
            {agent: self.path / agent for agent in agents},
        )
        self.transport(instance)
        await self.launch(instance)
        advertised = {
            agent: [tool["function"]["name"] for tool in payload["tools"]]
            for agent, payload in self.requests
        }
        self.assertEqual(advertised[agents[0]], ["browser_inspect"])
        self.assertEqual(advertised[agents[1]], ["browser_inspect", "browser_evaluate"])

    async def test_configured_sixteen_call_batch_executes_after_full_validation(self):
        tools = [
            {"name": "read_artifact", "description": "Read bounded evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit bounded output",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=3, max_tool_calls_per_response=16,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, ordinal):
            return {"id": f"call-{ordinal}", "type": "function",
                    "function": {"name": name, "arguments": "{}"}}

        async def respond(agent, payload):
            if not any(message.get("role") == "tool" for message in payload["messages"]):
                return completion(calls=[call("read_artifact", index) for index in range(10)])
            return completion(calls=[call("submit_result", "submit")])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        self.assertEqual(callback.await_count, 11)
        self.assertEqual(
            [invocation.args[1] for invocation in callback.await_args_list],
            ["read_artifact"] * 10 + ["submit_result"])
        self.assertEqual(instance.state["limits"]["max_tool_calls_per_response"], 16)
        self.assertEqual(
            instance.state["manifest"]["tool_call_batch_policy"]["configured_max"], 16)

    async def test_default_eight_call_batch_bound_is_typed_and_executes_nothing(self):
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1)
        await self.prepare(instance)

        async def respond(agent, payload):
            return completion(calls=[
                tool_call(call_id=f"call-{index}") for index in range(9)
            ])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "tool call count exceeded"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["failure_code"], FAILURE_TOOL_CALL_BATCH_EXCEEDED)
        self.assertEqual(worker["rejected_tool_batch"], {
            "call_count": 9,
            "allowed_count": 8,
            "reason": FAILURE_TOOL_CALL_BATCH_EXCEEDED,
        })
        self.assertEqual(worker["tools"], {})

    async def test_larger_batch_still_validates_every_call_before_callback(self):
        tools = [
            {"name": "read_artifact", "description": "Read bounded evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit bounded output",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1, max_tool_calls_per_response=16,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        calls = [{"id": f"call-{index}", "type": "function",
                  "function": {"name": "read_artifact", "arguments": "{}"}}
                 for index in range(10)]
        calls[-1]["function"]["arguments"] = "{not-json"

        async def respond(agent, payload):
            return completion(calls=calls)

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "registered bounded host tools"):
            await self.launch(instance)
        callback.assert_not_awaited()
        self.assertEqual(instance.workers["agent-a"]["tools"], {})

    async def test_larger_batch_configuration_rejects_browser_capabilities(self):
        instance = self.make(
            agents=("agent-a",), single_node=True,
            max_tool_calls_per_response=16,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result", "browser_click")
        ]
        with self.assertRaisesRegex(ValueError, "limited to the configured read"):
            await instance.create_workers(
                tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})
        self.assertEqual(instance.workers, {})

    async def test_malformed_registered_tool_json_gets_bounded_correction_turn(self):
        tools = [
            {"name": "read_artifact", "description": "Read bounded evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit bounded output",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=2,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            if not any(message.get("role") == "tool" for message in payload["messages"]):
                return completion(calls=[{
                    "id": "malformed-submit", "type": "function",
                    "function": {"name": "submit_result",
                                 "arguments": '{"answer":"truncated","evidence_ids": '},
                }])
            rejected = [
                json.loads(message["content"])
                for message in payload["messages"]
                if message.get("tool_call_id") == "malformed-submit"
            ]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["failure_code"], "malformed_tool_arguments")
            self.assertTrue(rejected[0]["retryable"])
            return completion(calls=[{
                "id": "corrected-submit", "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        callback.assert_awaited_once()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["tool_input_rejections"], 1)
        self.assertEqual(worker["tools"]["malformed-submit"]["status"], "rejected")
        self.assertEqual(worker["tools"]["malformed-submit"]["digest_mode"], "raw")

    async def test_non_string_tool_arguments_remain_a_fatal_envelope_error(self):
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1)
        await self.prepare(instance)
        invalid = tool_call(call_id="invalid-envelope")
        invalid["function"]["arguments"] = {"text": "not a JSON string"}

        async def respond(agent, payload):
            return completion(calls=[invalid])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "registered bounded host tools"):
            await self.launch(instance)
        callback.assert_not_awaited()
        self.assertEqual(instance.workers["agent-a"]["tools"], {})

    async def test_five_real_transport_threads_overlap_with_one_key(self):
        """Exercise the actual executor/timing code with an in-memory HTTP socket."""
        barrier = threading.Barrier(5, timeout=3)
        gate = threading.Lock()
        active, peak, authorizations = 0, 0, []
        path = self.path / "state.json"

        class Connection:
            sock = None
            status = 200

            def __init__(self, host, timeout):
                self.assert_host = host

            def request(self, method, target, body, headers):
                nonlocal active, peak
                durable = json.loads(path.read_text())
                assert len(durable["workers"]) == 5
                assert all(w["start_intent"] for w in durable["workers"].values())
                assert method == "POST" and target == "/api/v1/chat/completions"
                self.prompt = json.loads(body)["messages"][1]["content"]
                with gate:
                    authorizations.append(headers["Authorization"])
                    active += 1
                    peak = max(peak, active)
                barrier.wait()

            def getresponse(self):
                return self

            def getheader(self, name):
                return None

            def read(self, maximum):
                nonlocal active
                with gate:
                    active -= 1
                return json.dumps(completion(self.prompt)).encode()

            def close(self):
                pass

        instance = self.make()
        await self.prepare(instance)
        with patch("astra_harness.openrouter_runtime.http.client.HTTPSConnection", Connection):
            finals = await self.launch(instance)
        self.assertEqual(set(finals), set(AGENTS))
        self.assertEqual(peak, 5)
        self.assertEqual(authorizations, ["Bearer " + KEY] * 5)
        starts = [data for kind, _, data in self.events if kind == "request_started"]
        finishes = [data for kind, _, data in self.events if kind == "request_completed"]
        self.assertEqual(len(starts), 5)
        self.assertEqual(len(finishes), 5)
        self.assertLess(max(d["started_at"] for d in starts), min(d["completed_at"] for d in finishes))
        self.assertTrue(all(d["usage"]["total_tokens"] == 15 and d["provider"] == "fake-provider" for d in finishes))
        self.assertNotIn(KEY, json.dumps(self.events))
        for artifact in self.path.rglob("*.json"):
            self.assertNotIn(KEY, artifact.read_text())

    async def test_real_transport_salvages_complete_response_during_timeout_drain(self):
        class Connection:
            sock = None
            status = 200

            def __init__(self, host, timeout):
                pass

            def request(self, method, target, body, headers):
                pass

            def getresponse(self):
                return self

            def getheader(self, name):
                return None

            def read(self, maximum):
                time.sleep(1.05)
                return json.dumps(completion("late but complete")).encode()

            def close(self):
                pass

        instance = self.make(agents=("agent-a",), single_node=True, request_timeout=1)
        await self.prepare(instance)
        with (patch("astra_harness.openrouter_runtime.http.client.HTTPSConnection", Connection),
              patch("astra_harness.openrouter_runtime.TRANSPORT_DRAIN_SECONDS", 0.25)):
            finals = await self.launch(instance)
        self.assertEqual(finals, {"agent-a": "late but complete"})
        self.assertEqual(len(instance.workers["agent-a"]["requests"]), 1)
        self.assertEqual(instance.workers["agent-a"]["requests"][0]["status"], "completed")
        self.assertIsNone(instance.state["fatal"])

    async def test_unsettled_transport_is_not_misclassified_as_retryable_timeout(self):
        release = threading.Event()
        thread_exited = threading.Event()

        class Connection:
            sock = None
            status = 200

            def __init__(self, host, timeout):
                pass

            def request(self, method, target, body, headers):
                pass

            def getresponse(self):
                return self

            def getheader(self, name):
                return None

            def read(self, maximum):
                release.wait(5)
                thread_exited.set()
                return json.dumps(completion()).encode()

            def close(self):
                pass

        instance = self.make(agents=("agent-a",), single_node=True, request_timeout=1)
        await self.prepare(instance)
        try:
            with (patch("astra_harness.openrouter_runtime.http.client.HTTPSConnection", Connection),
                  patch("astra_harness.openrouter_runtime.TRANSPORT_DRAIN_SECONDS", 0.05)):
                with self.assertRaisesRegex(RuntimeFailure, "transport did not settle"):
                    await self.launch(instance)
            worker = instance.workers["agent-a"]
            self.assertEqual(len(worker["requests"]), 1)
            self.assertEqual(worker["requests"][0]["error_code"], "transport_unsettled")
            self.assertEqual(instance.state["fatal_kind"], "transport_unsettled")
            self.assertEqual(instance.state["failure_code"], "transport_unsettled")
        finally:
            release.set()
            await asyncio.to_thread(thread_exited.wait, 2)

    async def test_cancellation_waits_for_closed_transport_then_remains_cancelled(self):
        read_started = threading.Event()
        transport_closed = threading.Event()
        thread_exited = threading.Event()

        class Connection:
            sock = None
            status = 200

            def __init__(self, host, timeout):
                pass

            def request(self, method, target, body, headers):
                pass

            def getresponse(self):
                return self

            def getheader(self, name):
                return None

            def read(self, maximum):
                read_started.set()
                transport_closed.wait(2)
                thread_exited.set()
                return json.dumps(completion()).encode()

            def close(self):
                transport_closed.set()

        instance = self.make(agents=("agent-a",), single_node=True, request_timeout=5)
        await self.prepare(instance)
        with (patch("astra_harness.openrouter_runtime.http.client.HTTPSConnection", Connection),
              patch("astra_harness.openrouter_runtime.TRANSPORT_DRAIN_SECONDS", 0.5)):
            await instance.start_turns({"agent-a": "Task for agent-a"})
            self.assertTrue(await asyncio.to_thread(read_started.wait, 2))
            await instance.cancel()

        self.assertTrue(thread_exited.is_set())
        self.assertIsNone(instance.state["fatal"])
        self.assertEqual(instance.workers["agent-a"]["status"], "interrupted")
        request = instance.workers["agent-a"]["requests"][0]
        self.assertEqual(request["status"], "uncertain")
        self.assertEqual(request["error_code"], "request_cancelled")

    async def test_cancellation_marks_transport_unsettled_when_executor_does_not_exit(self):
        read_started = threading.Event()
        release = threading.Event()
        thread_exited = threading.Event()

        class Connection:
            sock = None
            status = 200

            def __init__(self, host, timeout):
                pass

            def request(self, method, target, body, headers):
                pass

            def getresponse(self):
                return self

            def getheader(self, name):
                return None

            def read(self, maximum):
                read_started.set()
                release.wait(5)
                thread_exited.set()
                return json.dumps(completion()).encode()

            def close(self):
                pass

        instance = self.make(agents=("agent-a",), single_node=True, request_timeout=5)
        await self.prepare(instance)
        try:
            with (patch("astra_harness.openrouter_runtime.http.client.HTTPSConnection", Connection),
                  patch("astra_harness.openrouter_runtime.TRANSPORT_DRAIN_SECONDS", 0.05)):
                await instance.start_turns({"agent-a": "Task for agent-a"})
                self.assertTrue(await asyncio.to_thread(read_started.wait, 2))
                await instance.cancel()

            self.assertEqual(instance.state["fatal_kind"], "transport_unsettled")
            self.assertEqual(instance.state["failure_code"], "transport_unsettled")
            worker = instance.workers["agent-a"]
            self.assertEqual(worker["status"], "failed")
            self.assertEqual(worker["failure_code"], "transport_unsettled")
            request = worker["requests"][0]
            self.assertEqual(request["status"], "uncertain")
            self.assertEqual(request["error_code"], "transport_unsettled")
        finally:
            release.set()
            await asyncio.to_thread(thread_exited.wait, 2)

    async def test_tool_loop_preserves_independent_conversations_and_reasoning(self):
        callback = AsyncMock(side_effect=lambda agent, name, args, call_id: {"agent": agent, "recorded": args["text"]})
        instance = self.make(on_tool=callback)
        await self.prepare(instance)

        async def respond(agent, payload):
            results = [m for m in payload["messages"] if m["role"] == "tool"]
            if not results:
                return completion(calls=[tool_call()])
            self.assertEqual(json.loads(results[0]["content"])["agent"], agent)
            self.assertTrue(any(m.get("reasoning_details") for m in payload["messages"]))
            return completion("Final " + agent)

        self.transport(instance, respond)
        self.assertEqual(await self.launch(instance), {a: "Final " + a for a in AGENTS})
        self.assertEqual(callback.await_count, 5)
        self.assertEqual(len(self.requests), 10)
        self.assertTrue(all(p["tools"] == instance.state["tool_specs"] for _, p in self.requests))
        self.assertTrue(all(w["token_totals"]["totalTokens"] == 30 for w in instance.workers.values()))

    async def test_completed_replay_requires_no_key_or_transport(self):
        first = self.make()
        await self.prepare(first)
        self.transport(first)
        finals = await self.launch(first)
        await first.close()
        with patch.dict("os.environ", {}, clear=True):
            replay = self.make()
            replay._http = AsyncMock(side_effect=AssertionError("Replay made a paid call"))
            await self.prepare(replay)
            self.assertEqual(await self.launch(replay), finals)
            replay._http.assert_not_awaited()

    async def test_missing_usage_keeps_later_cumulative_total_unknown(self):
        instance = self.make()
        await self.prepare(instance)
        async def respond(agent, payload):
            if not any(m["role"] == "tool" for m in payload["messages"]):
                data = completion(calls=[tool_call()])
                data.pop("usage")
                return data
            return completion()
        self.transport(instance, respond)
        await self.launch(instance)
        usages = [d["usage"] for k, _, d in self.events if k == "token_usage"]
        self.assertEqual(len(usages), 10)
        self.assertTrue(all(d["total"] is None for d in usages))
        self.assertEqual(sum(d["last"] is not None for d in usages), 5)
        self.assertFalse(any("cachedInputTokens" in (d["last"] or {}) for d in usages))

    async def test_partial_usage_does_not_invent_zero_counts(self):
        instance = self.make()
        await self.prepare(instance)
        async def respond(agent, payload):
            data = completion()
            data["usage"] = {"prompt_tokens": 10}
            return data
        self.transport(instance, respond)
        await self.launch(instance)
        self.assertTrue(all(d["usage"] == {"total": None, "last": None}
                            for k, _, d in self.events if k == "token_usage"))

    async def test_request_failure_never_retries_or_logs_provider_exception(self):
        instance = self.make()
        await self.prepare(instance)
        async def failing(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            raise OSError("Provider echoed Authorization: " + KEY)
        instance._http = AsyncMock(side_effect=failing)
        with self.assertRaisesRegex(RuntimeFailure, "uncertain"):
            await self.launch(instance)
        self.assertLessEqual(instance._http.await_count, 5)
        self.assertTrue(all(len(w["requests"]) <= 1 for w in instance.workers.values()))
        self.assertNotIn(KEY, instance.state_path.read_text())
        self.assertNotIn(KEY, json.dumps(self.events))
        await instance.close()
        replay = self.make()
        with self.assertRaisesRegex(RecoveryRequired, "Automatic paid request replay"):
            await replay.start()

    async def test_only_explicit_429_is_retried_with_bounded_attempts(self):
        instance = self.make()
        await self.prepare(instance)
        async def limited(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            if len(instance.workers[agent]["requests"]) == 1:
                return {"http_status": 429, "retry_after": "0", "completed_at": time.time()}
            return {"http_status": 200, "completed_at": time.time(), "body": json.dumps(completion()).encode()}
        instance._http = limited
        self.assertEqual(set(await self.launch(instance)), set(AGENTS))
        self.assertTrue(all(len(w["requests"]) == 2 for w in instance.workers.values()))
        self.assertEqual(sum(k == "request_rate_limited" for k, _, _ in self.events), 5)

    async def test_rate_limit_retries_cannot_exceed_physical_request_row_budget(self):
        instance = self.make(agents=("agent-a",), single_node=True,
                             max_requests_per_worker=2, max_rate_limit_retries=3)
        await self.prepare(instance)

        async def limited(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            return {"http_status": 429, "retry_after": "0", "completed_at": time.time()}

        instance._http = limited
        with self.assertRaisesRegex(RuntimeFailure, "request/tool-round budget"):
            await self.launch(instance)
        worker = instance.workers["agent-a"]
        self.assertEqual(len(worker["requests"]), 2)
        self.assertEqual(instance.state["fatal_kind"], "request_round_exhausted")
        self.assertEqual(instance.state["failure_code"], "request_round_exhausted")
        self.assertTrue(all(r["error_code"] == "transient_http" for r in worker["requests"]))

    async def test_exhausted_rate_limit_has_leaf_http_failure_code(self):
        instance = self.make(agents=("agent-a",), single_node=True,
                             max_rate_limit_retries=0)
        await self.prepare(instance)

        async def limited(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            return {"http_status": 429, "retry_after": "0", "completed_at": time.time()}

        instance._http = limited
        with self.assertRaisesRegex(RuntimeFailure, "HTTP 429"):
            await self.launch(instance)
        self.assertEqual(instance.state["fatal_kind"], "http_429")
        self.assertEqual(instance.state["failure_code"], "http_429")
        self.assertEqual(instance.workers["agent-a"]["requests"][0]["error_code"], "transient_http")

    async def test_timeout_after_success_marks_usage_unknown_and_identifies_limit(self):
        instance = self.make(agents=("agent-a",), single_node=True)
        await self.prepare(instance)

        async def response_then_timeout(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            if len(instance.workers[agent]["requests"]) == 1:
                return {"http_status": 200, "completed_at": time.time(),
                        "body": json.dumps(completion(calls=[tool_call()])).encode()}
            raise TimeoutError("Provider diagnostic containing " + KEY)

        instance._http = AsyncMock(side_effect=response_then_timeout)
        with self.assertRaisesRegex(RuntimeFailure, "timed out.*60 seconds"):
            await self.launch(instance)
        worker = instance.workers["agent-a"]
        self.assertEqual(instance._http.await_count, 2)
        self.assertEqual(worker["token_totals"]["totalTokens"], 15)
        self.assertTrue(worker["token_usage_incomplete"])
        failure = worker["requests"][-1]
        self.assertEqual(failure["error_code"], "request_timeout")
        self.assertEqual(failure["timeout_seconds"], 60)
        self.assertGreaterEqual(failure["elapsed_seconds"], 0)
        self.assertEqual(failure["status"], "uncertain")
        self.assertEqual(instance.state["fatal_kind"], "provider_timeout")
        self.assertEqual(instance.state["failure_code"], "provider_timeout")
        self.assertEqual(worker["fatal_kind"], "provider_timeout")
        self.assertEqual(worker["failure_code"], "provider_timeout")
        self.assertNotIn(KEY, instance.state_path.read_text())
        self.assertNotIn(KEY, json.dumps(self.events))

    async def test_http_500_and_error_body_are_not_retried_or_saved(self):
        instance = self.make(agents=("agent-a",), single_node=True)
        await self.prepare(instance)
        async def failed(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            return {"http_status": 500, "completed_at": time.time(), "body": KEY.encode()}
        instance._http = failed
        with self.assertRaisesRegex(RuntimeFailure, "HTTP 500"):
            await self.launch(instance)
        self.assertTrue(all(len(w["requests"]) <= 1 for w in instance.workers.values()))
        self.assertEqual(instance.state["fatal_kind"], "http_500")
        self.assertEqual(instance.state["failure_code"], "http_500")
        self.assertEqual(instance.workers["agent-a"]["requests"][0]["error_code"], "transient_http")
        self.assertNotIn(KEY, instance.state_path.read_text())

    async def test_permanent_http_failure_has_stable_machine_readable_codes(self):
        instance = self.make(agents=("agent-a",), single_node=True)
        await self.prepare(instance)

        async def unauthorized(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            return {"http_status": 401, "completed_at": time.time(), "body": KEY.encode()}

        instance._http = unauthorized
        with self.assertRaisesRegex(RuntimeFailure, "HTTP 401"):
            await self.launch(instance)
        self.assertEqual(instance.state["fatal_kind"], "http_401")
        self.assertEqual(instance.state["failure_code"], "http_401")
        self.assertEqual(instance.workers["agent-a"]["requests"][0]["error_code"], "permanent_http")
        self.assertNotIn(KEY, instance.state_path.read_text())

    async def test_changed_worker_count_and_model_are_rejected_without_writes(self):
        instance = self.make()
        await instance.start()
        await instance.close()
        original = instance.state_path.read_bytes()
        for kwargs in ({"agents": worker_ids(3)}, {"model": "example/other-model"}):
            with self.assertRaisesRegex(RuntimeFailure, "state was not modified"):
                self.make(**kwargs)
            self.assertEqual(original, instance.state_path.read_bytes())

    async def test_completion_policy_is_manifested_and_part_of_persisted_identity(self):
        kwargs = {
            "agents": ("agent-a",),
            "single_node": True,
            "max_requests_per_worker": 6,
            "completion_tool": "submit_result",
            "completion_read_tool": "read_artifact",
            "completion_reserve": 4,
        }
        instance = self.make(**kwargs)
        manifest = await instance.start()
        await instance.close()
        original = instance.state_path.read_bytes()

        self.assertEqual(manifest["completion_policy"], {
            "completion_tool": "submit_result",
            "read_tool": "read_artifact",
            "reserve_requests": 4,
            "force_requests": 2,
            "max_incomplete_final_corrections": 5,
        })
        replay = self.make(**kwargs)
        self.assertEqual(replay.completion_policy, manifest["completion_policy"])
        await replay.close()
        self.assertEqual(original, instance.state_path.read_bytes())

        with self.assertRaisesRegex(RuntimeFailure, "completion policy differ"):
            self.make(**{**kwargs, "completion_reserve": 3})
        self.assertEqual(original, instance.state_path.read_bytes())

    async def test_key_and_explicit_model_required_before_any_network(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeFailure, "explicit OpenRouter"):
                self.make(model=None)
            instance = self.make()
            with self.assertRaisesRegex(RuntimeFailure, "OPENROUTER_API_KEY"):
                await instance.start()
        self.assertFalse(instance.state_path.exists())

    async def test_research_request_timeout_supports_bounded_long_generations(self):
        instance = self.make(request_timeout=600)
        self.assertEqual(instance.limits["request_timeout"], 600)
        with self.assertRaisesRegex(ValueError, "request_timeout"):
            self.make(request_timeout=601)

    async def test_single_node_accepts_any_one_canonical_agent(self):
        instance = self.make(agents=("agent-c",), single_node=True)
        self.assertEqual(instance.agents, ("agent-c",))
        for agents in ((), ("agent-a", "agent-b"), ("agent-z",)):
            with self.subTest(agents=agents):
                with self.assertRaisesRegex(ValueError, "one canonical agent ID"):
                    self.make(agents=agents, single_node=True)

    async def test_privately_supplied_key_is_supported_without_persistence(self):
        with patch.dict("os.environ", {}, clear=True):
            instance = self.make(api_key=KEY)
            await self.prepare(instance)
            self.transport(instance)
            await self.launch(instance)
        self.assertNotIn(KEY, instance.state_path.read_text())

    async def test_steering_is_queued_until_next_request_and_deduplicated(self):
        instance = self.make()
        await self.prepare(instance)
        first_request = asyncio.Event()
        release = asyncio.Event()
        async def respond(agent, payload):
            if not any(m["role"] == "tool" for m in payload["messages"]):
                if agent == "agent-a":
                    first_request.set()
                    await release.wait()
                return completion(calls=[tool_call()])
            if agent == "agent-a":
                self.assertEqual(sum("shared observation" in (m.get("content") or "") for m in payload["messages"]), 1)
            return completion()
        self.transport(instance, respond)
        await instance.start_turns({a: "Task for " + a for a in AGENTS})
        await first_request.wait()
        event = {"claim": "shared observation"}
        receipt = await instance.steer("agent-a", event)
        await instance.steer("agent-a", event)
        self.assertTrue(receipt["accepted"] and receipt["queued"])
        self.assertFalse(receipt["delivered"])
        self.assertEqual(len(instance.workers["agent-a"]["steering"]), 1)
        self.assertFalse(any(k == "steer_accepted" for k, _, _ in self.events))
        release.set()
        await instance.wait(timeout=5)
        self.assertTrue(instance.workers["agent-a"]["steering"][0]["included_at"])

    async def test_unknown_tools_and_exhausted_round_budget_stop(self):
        instance = self.make(agents=("agent-a",), single_node=True, max_requests_per_worker=1)
        await self.prepare(instance)
        async def respond(agent, payload):
            return completion(calls=[tool_call()])
        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "request/tool-round budget"):
            await self.launch(instance)
        self.assertTrue(all(len(w["requests"]) <= 1 for w in instance.workers.values()))
        self.assertEqual(instance.state["fatal_kind"], "request_round_exhausted")
        self.assertEqual(instance.state["failure_code"], "request_round_exhausted")

    async def test_completion_reserve_closes_research_tools_then_forces_submission(self):
        tools = [
            {"name": "browser_inspect", "description": "Inspect bounded browser state",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        calls = []

        async def on_tool(agent, name, arguments, call_id):
            calls.append(name)
            return {"accepted": True}

        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=6,
            on_tool=on_tool, completion_tool="submit_result",
            completion_read_tool="read_artifact", completion_reserve=4)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, ordinal):
            return {"id": f"call-{ordinal}", "type": "function",
                    "function": {"name": name, "arguments": "{}"}}

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            names = {tool["function"]["name"] for tool in payload.get("tools", [])}
            if ordinal <= 2:
                self.assertEqual(names, {"browser_inspect", "read_artifact", "submit_result"})
                self.assertEqual(payload["tool_choice"], "auto")
                return completion(calls=[call("browser_inspect", ordinal)])
            if ordinal <= 4:
                self.assertEqual(names, {"read_artifact", "submit_result"})
                self.assertEqual(payload["tool_choice"], "auto")
                self.assertNotIn("browser_inspect", names)
                return completion(calls=[call("read_artifact", ordinal)])
            if ordinal == 5:
                self.assertEqual(names, {"submit_result"})
                self.assertEqual(payload["tool_choice"], {
                    "type": "function", "function": {"name": "submit_result"}})
                return completion(calls=[call("submit_result", ordinal)])
            self.fail("A durably accepted submission must not start an acknowledgement request")

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        self.assertEqual(calls,
                         ["browser_inspect", "browser_inspect", "read_artifact",
                          "read_artifact", "submit_result"])
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["completion_reserve"]["requests_remaining"], 4)
        self.assertEqual(worker["completion_accepted"]["tool"], "submit_result")
        self.assertEqual(worker["completion_kind"], "host_tool_submission")
        self.assertEqual(len(worker["requests"]), 5)
        notices = [message["content"] for message in worker["messages"]
                   if message["role"] == "user" and "budget notice" in message["content"]]
        self.assertEqual(len(notices), 1)
        self.assertIn("New evidence-gathering actions are closed", notices[0])
        self.assertEqual(instance.state["manifest"]["completion_policy_version"],
                         "terminal_reserve_v4")

    async def test_completion_reserve_replies_to_closed_registered_tool_without_executing_it(self):
        tools = [
            {"name": "browser_inspect", "description": "Inspect bounded browser state",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=2,
            on_tool=callback, completion_tool="submit_result",
            completion_read_tool="read_artifact", completion_reserve=2)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, ordinal, arguments="{}"):
            return {"id": f"call-{ordinal}", "type": "function",
                    "function": {"name": name, "arguments": arguments}}

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal == 1:
                self.assertEqual(
                    {tool["function"]["name"] for tool in payload["tools"]},
                    {"read_artifact", "submit_result"})
                return completion(calls=[call("browser_inspect", ordinal)])
            tool_result = json.loads([
                message["content"] for message in payload["messages"]
                if message.get("tool_call_id") == "call-1"
            ][0])
            self.assertEqual(tool_result["error_type"],
                             "completion_reserve_tool_disabled")
            self.assertEqual(tool_result["required_tool"], "submit_result")
            self.assertIn("Call submit_result now", tool_result["next_action"])
            self.assertEqual(
                {tool["function"]["name"] for tool in payload["tools"]},
                {"submit_result"})
            return completion(calls=[call("submit_result", ordinal)])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.args[1], "submit_result")
        disabled = instance.workers["agent-a"]["tools"]["call-1"]
        self.assertEqual(disabled["status"], "disabled")
        self.assertEqual(disabled["tool"], "browser_inspect")
        self.assertEqual(sum(kind == "tool_disabled" for kind, _, _ in self.events), 1)

    async def test_disabled_reserve_tool_does_not_suppress_later_submit_in_same_batch(self):
        tools = [
            {"name": "browser_click", "description": "Perform a bounded browser click",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=1,
            on_tool=callback, completion_tool="submit_result",
            completion_read_tool="read_artifact", completion_reserve=1)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, call_id):
            return {"id": call_id, "type": "function",
                    "function": {"name": name, "arguments": "{}"}}

        async def respond(agent, payload):
            self.assertEqual(
                {tool["function"]["name"] for tool in payload["tools"]},
                {"submit_result"})
            self.assertEqual(payload["tool_choice"], {
                "type": "function", "function": {"name": "submit_result"}})
            return completion(calls=[
                call("browser_click", "disabled-first"),
                call("submit_result", "submit-second"),
            ])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})

        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.args[1], "submit_result")
        worker = instance.workers["agent-a"]
        self.assertEqual(worker.get("tool_input_rejections", 0), 0)
        self.assertEqual(worker["status"], "completed")
        self.assertEqual(worker["completion_kind"], "host_tool_submission")
        tool_messages = [message for message in worker["messages"]
                         if message.get("tool_call_id") in {
                             "disabled-first", "submit-second"}]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["disabled-first", "submit-second"])
        self.assertEqual(
            json.loads(tool_messages[0]["content"])["error_type"],
            "completion_reserve_tool_disabled")
        self.assertTrue(json.loads(tool_messages[1]["content"])["accepted"])

    async def test_accepted_submission_suppresses_later_mutation_and_duplicate_submit(self):
        tools = [
            {"name": "browser_click", "description": "Perform a bounded browser click",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=3,
            on_tool=callback, completion_tool="submit_result",
            completion_read_tool="read_artifact", completion_reserve=2)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, call_id):
            return {"id": call_id, "type": "function",
                    "function": {"name": name, "arguments": "{}"}}

        async def respond(agent, payload):
            return completion(calls=[
                call("submit_result", "submit-first"),
                call("browser_click", "mutation-after-submit"),
                call("submit_result", "submit-duplicate"),
            ])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.args[1], "submit_result")
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["tools"]["mutation-after-submit"]["status"],
                         "skipped_after_completion")
        self.assertEqual(worker["tools"]["submit-duplicate"]["status"],
                         "skipped_after_completion")
        results = [json.loads(message["content"]) for message in worker["messages"]
                   if message.get("tool_call_id") in {
                       "mutation-after-submit", "submit-duplicate"}]
        self.assertEqual(
            [result["error_type"] for result in results],
            ["tool_call_skipped_after_completion"] * 2)

    async def test_completion_reserve_still_fails_truly_unregistered_tool(self):
        callback = AsyncMock()
        tools = [
            *TOOLS,
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=2,
            on_tool=callback, completion_tool="observe",
            completion_read_tool="read_artifact",
            completion_reserve=2)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            return completion(calls=[tool_call(name="shell")])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "registered bounded host tools"):
            await self.launch(instance)
        callback.assert_not_awaited()

    async def test_completion_reserve_still_fails_malformed_closed_tool_call(self):
        tools = [
            *TOOLS,
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        callback = AsyncMock()
        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=2,
            on_tool=callback, completion_tool="submit_result",
            completion_read_tool="read_artifact",
            completion_reserve=2)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            malformed = tool_call(name="observe")
            valid = {"id": "valid-submit", "type": "function",
                     "function": {"name": "submit_result", "arguments": "{}"}}
            malformed["function"]["arguments"] = "{not-json"
            return completion(calls=[valid, malformed])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "registered bounded host tools"):
            await self.launch(instance)
        callback.assert_not_awaited()

    async def test_completion_reserve_uses_final_forced_turn_to_correct_rejection(self):
        tools = [
            {"name": "browser_inspect", "description": "Inspect bounded browser state",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "read_artifact", "description": "Read captured evidence",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
            {"name": "submit_result", "description": "Submit the structured result",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}},
        ]
        submitted = 0
        calls = []

        async def on_tool(agent, name, arguments, call_id):
            nonlocal submitted
            calls.append(name)
            if name == "submit_result":
                submitted += 1
                if submitted == 1:
                    raise ToolInputError("structured result needs correction")
                return {"accepted": True}
            return {"accepted": True}

        instance = self.make(
            agents=("agent-a",), single_node=True, max_requests_per_worker=5,
            on_tool=on_tool, completion_tool="submit_result",
            completion_read_tool="read_artifact", completion_reserve=4)
        await instance.start()
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        def call(name, ordinal):
            return {"id": f"call-{ordinal}", "type": "function",
                    "function": {"name": name, "arguments": "{}"}}

        forced_payloads = []

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal == 1:
                return completion(calls=[call("browser_inspect", ordinal)])
            if ordinal in {2, 3}:
                self.assertEqual(
                    {tool["function"]["name"] for tool in payload["tools"]},
                    {"read_artifact", "submit_result"})
                return completion(calls=[call("read_artifact", ordinal)])
            forced_payloads.append(copy.deepcopy(payload))
            return completion(calls=[call("submit_result", ordinal)])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        self.assertEqual(calls,
                         ["browser_inspect", "read_artifact", "read_artifact",
                          "submit_result", "submit_result"])
        self.assertEqual(submitted, 2)
        self.assertEqual(instance.workers["agent-a"]["tool_input_rejections"], 1)
        self.assertEqual(len(forced_payloads), 2)
        self.assertTrue(all(payload["tool_choice"] == {
            "type": "function", "function": {"name": "submit_result"}}
            for payload in forced_payloads))
        self.assertEqual(instance.workers["agent-a"]["completion_kind"],
                         "host_tool_submission")

    async def test_tool_input_exhaustion_has_stable_machine_readable_codes(self):
        callback = AsyncMock(side_effect=ToolInputError("invalid observation"))
        instance = self.make(agents=("agent-a",), single_node=True, on_tool=callback,
                             max_requests_per_worker=8)
        await self.prepare(instance)

        async def reject_unique_call(agent, payload):
            request = len(instance.workers[agent]["requests"])
            return completion(calls=[tool_call(call_id=f"call-{request}-{index}")
                                     for index in range(4)])

        self.transport(instance, reject_unique_call)
        with self.assertRaisesRegex(RuntimeFailure, "tool input rejection budget exhausted"):
            await self.launch(instance)
        self.assertEqual(callback.await_count, 4)
        self.assertEqual(instance.workers["agent-a"]["tool_input_rejections"], 4)
        statuses = [record["status"] for record in instance.workers["agent-a"]["tools"].values()]
        self.assertEqual(statuses.count("rejected"), 4)
        self.assertEqual(statuses.count("skipped"), 12)
        self.assertEqual(instance.state["fatal_kind"], "tool_input_exhausted")
        self.assertEqual(instance.state["failure_code"], "tool_input_exhausted")

    async def test_one_multi_call_response_consumes_one_rejection_and_answers_every_call(self):
        callback = AsyncMock(side_effect=ToolInputError("invalid observation"))
        instance = self.make(agents=("agent-a",), single_node=True, on_tool=callback)
        await self.prepare(instance)

        async def reject_batch_then_finish(agent, payload):
            if not any(message["role"] == "tool" for message in payload["messages"]):
                return completion(calls=[tool_call(call_id=f"batch-{index}") for index in range(4)])
            return completion("corrected")

        self.transport(instance, reject_batch_then_finish)
        self.assertEqual(await self.launch(instance), {"agent-a": "corrected"})
        worker = instance.workers["agent-a"]
        self.assertEqual(callback.await_count, 1)
        self.assertEqual(worker["tool_input_rejections"], 1)
        self.assertEqual([worker["tools"][f"batch-{index}"]["status"] for index in range(4)],
                         ["rejected", "skipped", "skipped", "skipped"])

        tool_messages = [message for message in self.requests[1][1]["messages"]
                         if message["role"] == "tool"]
        self.assertEqual([message["tool_call_id"] for message in tool_messages],
                         [f"batch-{index}" for index in range(4)])
        results = [json.loads(message["content"]) for message in tool_messages]
        self.assertEqual(results[0]["error_type"], "tool_input_rejected")
        self.assertTrue(all(result["error_type"] == "tool_call_skipped_after_rejection"
                            for result in results[1:]))
        self.assertTrue(all(result["rejection_count"] == 1 for result in results))
        self.assertEqual(sum(kind == "tool_rejected" for kind, _, _ in self.events), 1)
        self.assertEqual(sum(kind == "tool_skipped" for kind, _, _ in self.events), 3)

        durable = json.loads(instance.state_path.read_text())
        self.assertEqual(durable["workers"]["agent-a"]["tool_input_rejections"], 1)
        self.assertEqual(durable["workers"]["agent-a"]["tools"]["batch-3"]["status"], "skipped")

    async def test_unregistered_tool_never_reaches_host_callback(self):
        callback = AsyncMock()
        instance = self.make(on_tool=callback)
        await self.prepare(instance)
        async def respond(agent, payload):
            return completion(calls=[
                tool_call(call_id="valid-before-unknown"),
                tool_call(name="shell", call_id="unknown"),
            ])
        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "registered bounded host tools"):
            await self.launch(instance)
        callback.assert_not_awaited()

    async def test_incomplete_or_empty_final_is_retryable_node_no_submission(self):
        for suffix, content, finish in (
                ("truncated", "partial", "length"),
                ("empty", "", "stop"),
                ("missing", None, "stop"),
                ("whitespace", "   ", "stop")):
            with self.subTest(suffix=suffix):
                instance = OpenRouterRuntime(
                    self.path / f"state-{suffix}.json",
                    self.path / f"runtime-{suffix}",
                    self.event,
                    AsyncMock(return_value={"recorded": True}),
                    model=MODEL,
                    agents=("agent-a",),
                    single_node=True,
                )
                self.instances.append(instance)
                await self.prepare(instance)

                async def respond(agent, payload, *, content=content, finish=finish):
                    return completion(content=content, finish=finish)

                self.transport(instance, respond)
                with self.assertRaisesRegex(RuntimeFailure, "complete final"):
                    await self.launch(instance)
                self.assertEqual(instance.state["fatal_kind"], "node_no_submission")
                self.assertEqual(instance.state["failure_code"], "node_no_submission")
                worker = instance.workers["agent-a"]
                self.assertEqual(worker["fatal_kind"], "node_no_submission")
                self.assertEqual(worker["failure_code"], "node_no_submission")

    async def test_completion_policy_corrects_every_no_tool_final_in_same_ledger(self):
        variants = (
            ("truncated", "partial answer", "length"),
            ("empty", "", "stop"),
            ("missing", None, "stop"),
            ("whitespace", "   ", "stop"),
            ("printed", '{"answer":"printed instead of submitted"}', "stop"),
        )
        for suffix, content, finish in variants:
            with self.subTest(suffix=suffix):
                callback = AsyncMock(return_value={"accepted": True})
                instance = OpenRouterRuntime(
                    self.path / f"correction-{suffix}.json",
                    self.path / f"correction-runtime-{suffix}",
                    self.event, callback, model=MODEL,
                    agents=("agent-a",), single_node=True,
                    max_requests_per_worker=3,
                    completion_tool="submit_result",
                    completion_read_tool="read_artifact",
                    completion_reserve=1,
                )
                self.instances.append(instance)
                await instance.start()
                tools = [
                    {"name": name, "description": "bounded",
                     "inputSchema": {"type": "object", "properties": {},
                                     "required": [], "additionalProperties": False}}
                    for name in ("read_artifact", "submit_result")
                ]
                await instance.create_workers(
                    tools, "Use only registered tools",
                    {"agent-a": self.path / f"workspace-{suffix}"})

                async def respond(agent, payload, *, content=content, finish=finish):
                    if len(instance.workers[agent]["requests"]) == 1:
                        return completion(content=content, finish=finish)
                    self.assertEqual(
                        {tool["function"]["name"] for tool in payload["tools"]},
                        {"read_artifact", "submit_result"})
                    self.assertEqual(payload["tool_choice"], "auto")
                    correction_messages = [
                        message["content"] for message in payload["messages"]
                        if message.get("role") == "user"
                        and "Trusted coordinator correction" in message.get("content", "")
                    ]
                    self.assertEqual(len(correction_messages), 1)
                    return completion(calls=[{
                        "id": f"submit-{suffix}", "type": "function",
                        "function": {"name": "submit_result", "arguments": "{}"},
                    }])

                correction_events_before = sum(
                    kind == "incomplete_final_correction"
                    for kind, _, _ in self.events)
                self.transport(instance, respond)
                self.assertEqual(
                    await self.launch(instance),
                    {"agent-a": "Structured result accepted by the host."})
                callback.assert_awaited_once()
                worker = instance.workers["agent-a"]
                self.assertEqual(len(worker["requests"]), 2)
                self.assertEqual(worker["incomplete_final_correction"]["count"], 1)
                self.assertEqual(worker["incomplete_final_correction"]["status"], "resolved")
                self.assertFalse(worker["incomplete_final_correction"]["pending"])
                self.assertEqual(
                    worker["requests"][0]["finish_reason"],
                    finish if finish in {
                        "stop", "tool_calls", "length", "content_filter", "error"
                    } else "other")
                self.assertEqual(
                    sum(kind == "incomplete_final_correction"
                        for kind, _, _ in self.events) - correction_events_before,
                    1)

    async def test_completion_policy_caps_no_tool_corrections_durably(self):
        callback = AsyncMock()
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=5,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal > 1:
                if ordinal == 5:
                    self.assertEqual(payload["tool_choice"], {
                        "type": "function", "function": {"name": "submit_result"}})
                    self.assertEqual(
                        {tool["function"]["name"] for tool in payload["tools"]},
                        {"submit_result"})
                else:
                    self.assertEqual(payload["tool_choice"], "auto")
                    self.assertEqual(
                        {tool["function"]["name"] for tool in payload["tools"]},
                        {"read_artifact", "submit_result"})
            return completion(content="printed result", finish="stop")

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "complete final"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(len(worker["requests"]), 5)
        self.assertEqual(worker["failure_code"], "node_no_submission")
        self.assertEqual(worker["incomplete_final_correction"]["count"], 4)
        self.assertEqual(worker["incomplete_final_correction"]["status"], "exhausted")
        self.assertFalse(worker["incomplete_final_correction"]["pending"])
        self.assertEqual(len(worker["incomplete_final_correction"]["events"]), 4)
        self.assertEqual(worker["incomplete_final_correction"]["events"][-1]["ordinal"], 4)
        self.assertEqual(self.requests[-1][1]["tool_choice"], {
            "type": "function", "function": {"name": "submit_result"}})

    async def test_no_request_budget_retains_node_no_submission(self):
        callback = AsyncMock()
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})
        async def respond(agent, payload):
            return completion(content="", finish="stop")
        self.transport(instance, respond)

        with self.assertRaisesRegex(RuntimeFailure, "complete final"):
            await self.launch(instance)
        callback.assert_not_awaited()
        correction = instance.workers["agent-a"]["incomplete_final_correction"]
        self.assertEqual(correction["count"], 0)
        self.assertEqual(correction["status"], "exhausted")

    async def test_forced_correction_integrates_with_malformed_json_rejection(self):
        callback = AsyncMock(return_value={"accepted": True})
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=4,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal == 1:
                return completion(content=None, finish="stop")
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(
                {tool["function"]["name"] for tool in payload["tools"]},
                {"read_artifact", "submit_result"})
            if ordinal == 2:
                return completion(calls=[{
                    "id": "malformed-after-empty", "type": "function",
                    "function": {"name": "submit_result", "arguments": '{"answer":'},
                }])
            return completion(calls=[{
                "id": "corrected-after-empty", "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        callback.assert_awaited_once()
        worker = instance.workers["agent-a"]
        self.assertEqual(len(worker["requests"]), 3)
        self.assertEqual(worker["tool_input_rejections"], 1)
        self.assertEqual(worker["tools"]["malformed-after-empty"]["status"], "rejected")
        self.assertEqual(worker["incomplete_final_correction"]["status"], "resolved")

    async def test_completion_policy_final_rejections_end_as_node_no_submission(self):
        for suffix, malformed, callback in (
                ("malformed", True, AsyncMock()),
                ("schema", False, AsyncMock(side_effect=ToolInputError("invalid result")))):
            with self.subTest(suffix=suffix):
                instance = OpenRouterRuntime(
                    self.path / f"final-rejection-{suffix}.json",
                    self.path / f"final-rejection-runtime-{suffix}",
                    self.event, callback, model=MODEL,
                    agents=("agent-a",), single_node=True,
                    max_requests_per_worker=1,
                    completion_tool="submit_result",
                    completion_read_tool="read_artifact", completion_reserve=1)
                self.instances.append(instance)
                await instance.start()
                tools = [
                    {"name": name, "description": "bounded",
                     "inputSchema": {"type": "object", "properties": {},
                                     "required": [], "additionalProperties": False}}
                    for name in ("read_artifact", "submit_result")
                ]
                await instance.create_workers(
                    tools, "Use only registered tools",
                    {"agent-a": self.path / f"final-rejection-workspace-{suffix}"})

                async def respond(agent, payload, *, malformed=malformed):
                    return completion(calls=[{
                        "id": f"final-{suffix}", "type": "function",
                        "function": {
                            "name": "submit_result",
                            "arguments": '{"answer":' if malformed else "{}",
                        },
                    }])

                self.transport(instance, respond)
                with self.assertRaisesRegex(RuntimeFailure, "complete final"):
                    await self.launch(instance)
                worker = instance.workers["agent-a"]
                self.assertEqual(worker["failure_code"], "node_no_submission")
                self.assertEqual(worker["tools"][f"final-{suffix}"]["status"], "rejected")
                if malformed:
                    callback.assert_not_awaited()
                else:
                    callback.assert_awaited_once()

    async def test_completion_policy_final_disabled_read_ends_as_node_no_submission(self):
        callback = AsyncMock()
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            self.assertEqual(payload["tool_choice"], {
                "type": "function", "function": {"name": "submit_result"}})
            return completion(calls=[{
                "id": "disabled-final-read", "type": "function",
                "function": {"name": "read_artifact", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "complete final"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["failure_code"], "node_no_submission")
        self.assertEqual(worker["tools"]["disabled-final-read"]["status"], "disabled")

    async def test_completion_policy_preserves_stronger_rejection_exhaustion(self):
        callback = AsyncMock()
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=5,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            return completion(calls=[{
                "id": f"malformed-{ordinal}", "type": "function",
                "function": {"name": "submit_result", "arguments": '{"answer":'},
            }])

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "rejection budget exhausted"):
            await self.launch(instance)
        callback.assert_not_awaited()
        self.assertEqual(instance.workers["agent-a"]["failure_code"],
                         "tool_input_exhausted")
        self.assertEqual(len(instance.workers["agent-a"]["requests"]), 4)

    async def test_early_correction_disables_browser_call_without_callback(self):
        calls = []

        async def on_tool(agent, name, arguments, call_id):
            calls.append(name)
            return {"accepted": True}

        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=on_tool,
            max_requests_per_worker=4,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("browser_click", "read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal == 1:
                return completion(content="printed result", finish="stop")
            self.assertNotIn(
                "browser_click",
                {tool["function"]["name"] for tool in payload["tools"]})
            if ordinal == 2:
                return completion(calls=[{
                    "id": "closed-browser-click", "type": "function",
                    "function": {"name": "browser_click", "arguments": "{}"},
                }])
            return completion(calls=[{
                "id": "submit-after-closed-browser", "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        self.assertEqual(calls, ["submit_result"])
        self.assertEqual(
            instance.workers["agent-a"]["tools"]["closed-browser-click"]["status"],
            "disabled")

    async def test_clean_scheduled_correction_reopen_fails_closed(self):
        state_path = self.path / "clean-pending-correction.json"
        runtime_path = self.path / "clean-pending-correction-runtime"
        callback = AsyncMock()
        kwargs = dict(
            model=MODEL, agents=("agent-a",), single_node=True,
            max_requests_per_worker=4,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        instance = OpenRouterRuntime(
            state_path, runtime_path, self.event, callback, **kwargs)
        self.instances.append(instance)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        workspace = self.path / "clean-pending-workspace"
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": workspace})
        original_schedule = instance._schedule_incomplete_final_correction

        async def stop_after_schedule(agent, reason):
            scheduled = await original_schedule(agent, reason)
            self.assertTrue(scheduled)
            raise asyncio.CancelledError

        instance._schedule_incomplete_final_correction = stop_after_schedule

        async def respond(agent, payload):
            return completion(content="printed result", finish="stop")

        self.transport(instance, respond)
        await instance.start_turns({"agent-a": "Task for agent-a"})
        task = instance._tasks["agent-a"]
        with self.assertRaises(asyncio.CancelledError):
            await task
        durable = json.loads(state_path.read_text())
        worker = durable["workers"]["agent-a"]
        self.assertEqual(len(worker["requests"]), 1)
        self.assertEqual(worker["requests"][0]["status"], "completed")
        self.assertTrue(worker["incomplete_final_correction"]["pending"])
        self.assertEqual(worker["status"], "interrupted")
        self.assertFalse(any(row["status"] in {"intent", "inflight", "uncertain"}
                             for row in worker["requests"]))

        await instance.close()
        replay = OpenRouterRuntime(
            state_path, runtime_path, self.event, callback, **kwargs)
        self.instances.append(replay)
        with self.assertRaisesRegex(RecoveryRequired, "unresolved or failed request"):
            await replay.start()
        callback.assert_not_awaited()

    async def test_early_no_tool_recovery_retains_required_read_before_submission(self):
        calls = []

        async def on_tool(agent, name, arguments, call_id):
            calls.append(name)
            return {"accepted": True, "artifact": "bounded"}

        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=on_tool,
            max_requests_per_worker=6,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=2)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("browser_inspect", "read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Read required evidence before submission",
            {"agent-a": self.path / "agent-a"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            if ordinal == 1:
                self.assertEqual(
                    {tool["function"]["name"] for tool in payload["tools"]},
                    {"browser_inspect", "read_artifact", "submit_result"})
                return completion(content="I printed a result instead of submitting it")
            self.assertEqual(
                {tool["function"]["name"] for tool in payload["tools"]},
                {"read_artifact", "submit_result"})
            self.assertEqual(payload["tool_choice"], "auto")
            correction_notices = [
                message for message in payload["messages"]
                if message.get("role") == "user"
                and "Trusted coordinator correction" in message.get("content", "")
            ]
            self.assertEqual(len(correction_notices), 1)
            if ordinal == 2:
                return completion(calls=[{
                    "id": "required-read", "type": "function",
                    "function": {"name": "read_artifact", "arguments": "{}"},
                }])
            tool_receipts = [
                message for message in payload["messages"]
                if message.get("tool_call_id") == "required-read"
            ]
            self.assertEqual(len(tool_receipts), 1)
            return completion(calls=[{
                "id": "submit-after-read", "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        self.assertEqual(calls, ["read_artifact", "submit_result"])
        worker = instance.workers["agent-a"]
        self.assertNotIn("completion_reserve", worker)
        self.assertEqual(worker["incomplete_final_correction"]["mode"],
                         "completion_closure")
        self.assertEqual(worker["incomplete_final_correction"]["status"], "resolved")

    async def test_pending_no_tool_correction_is_durable_and_restart_fails_closed(self):
        state_path = self.path / "pending-correction.json"
        runtime_path = self.path / "pending-correction-runtime"
        second_request_started = asyncio.Event()
        never = asyncio.Event()
        callback = AsyncMock()
        kwargs = dict(
            model=MODEL, agents=("agent-a",), single_node=True,
            max_requests_per_worker=4,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        instance = OpenRouterRuntime(
            state_path, runtime_path, self.event, callback, **kwargs)
        self.instances.append(instance)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        workspace = self.path / "pending-workspace"
        await instance.create_workers(
            tools, "Use only registered tools", {"agent-a": workspace})

        async def respond(agent, payload):
            if len(instance.workers[agent]["requests"]) == 1:
                return completion(content=None, finish="length")
            second_request_started.set()
            await never.wait()

        self.transport(instance, respond)
        await instance.start_turns({"agent-a": "Task for agent-a"})
        await asyncio.wait_for(second_request_started.wait(), 2)
        durable = json.loads(state_path.read_text())
        correction = durable["workers"]["agent-a"]["incomplete_final_correction"]
        self.assertEqual(correction["count"], 1)
        self.assertTrue(correction["pending"])
        self.assertEqual(correction["mode"], "completion_closure")
        self.assertEqual(correction["events"][0]["reason"], "non_stop_finish")
        correction_messages = [
            message for message in durable["workers"]["agent-a"]["messages"]
            if message.get("role") == "user"
            and "Trusted coordinator correction" in message.get("content", "")
        ]
        self.assertEqual(len(correction_messages), 1)

        await instance.cancel()
        await instance.close()
        replay = OpenRouterRuntime(
            state_path, runtime_path, self.event, callback, **kwargs)
        self.instances.append(replay)
        with self.assertRaisesRegex(RecoveryRequired, "unresolved or failed request"):
            await replay.start()
        callback.assert_not_awaited()

    async def test_invalid_or_fail_closed_no_tool_message_is_not_corrected(self):
        for suffix, mutate in (
                ("content", lambda data: data["choices"][0]["message"].update(content=7)),
                ("calls", lambda data: data["choices"][0]["message"].update(tool_calls={})),
                ("finish", lambda data: data["choices"][0].update(finish_reason=None)),
                ("empty-tool-finish", lambda data: data["choices"][0].update(
                    finish_reason="tool_calls")),
                ("content-filter", lambda data: data["choices"][0].update(
                    finish_reason="content_filter")),
                ("vendor-finish", lambda data: data["choices"][0].update(
                    finish_reason="vendor_specific_finish"))):
            with self.subTest(suffix=suffix):
                callback = AsyncMock()
                instance = OpenRouterRuntime(
                    self.path / f"invalid-{suffix}.json",
                    self.path / f"invalid-runtime-{suffix}", self.event, callback,
                    model=MODEL, agents=("agent-a",), single_node=True,
                    max_requests_per_worker=2,
                    completion_tool="submit_result",
                    completion_read_tool="read_artifact", completion_reserve=1)
                self.instances.append(instance)
                await instance.start()
                tools = [
                    {"name": name, "description": "bounded",
                     "inputSchema": {"type": "object", "properties": {},
                                     "required": [], "additionalProperties": False}}
                    for name in ("read_artifact", "submit_result")
                ]
                await instance.create_workers(
                    tools, "Use only registered tools",
                    {"agent-a": self.path / f"invalid-workspace-{suffix}"})

                async def respond(agent, payload, *, mutate=mutate):
                    data = completion(content="complete", finish="stop")
                    mutate(data)
                    return data

                self.transport(instance, respond)
                with self.assertRaises(RuntimeFailure):
                    await self.launch(instance)
                callback.assert_not_awaited()
                worker = instance.workers["agent-a"]
                self.assertEqual(worker["failure_code"], "operational_failure")
                self.assertNotIn("incomplete_final_correction", worker)

    async def test_resolved_provider_error_uses_sanitized_same_ledger_correction(self):
        callback = AsyncMock(return_value={"accepted": True})
        state_path = self.path / "provider-error-correction.json"
        instance = OpenRouterRuntime(
            state_path, self.path / "provider-error-correction-runtime",
            self.event, callback, model=MODEL, agents=("agent-a",),
            single_node=True, max_requests_per_worker=3,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        self.instances.append(instance)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools",
            {"agent-a": self.path / "provider-error-correction-workspace"})
        diagnostic = "provider-private-diagnostic-must-not-persist"

        async def respond(agent, payload):
            if len(instance.workers[agent]["requests"]) == 1:
                data = completion(content="discarded", finish="error")
                # Every provider-controlled choice field besides finish_reason
                # is quarantined, including an alleged tool call and malformed
                # assistant shape.
                data["choices"][0].update(
                    error={"message": diagnostic},
                    message={
                        "role": "provider-error",
                        "content": {"diagnostic": diagnostic},
                        "tool_calls": [
                            7,
                            {"id": "malformed-call", "type": "function",
                             "function": None},
                            tool_call(name="submit_result",
                                      call_id="quarantined-submit"),
                        ],
                    },
                )
                return data
            callback.assert_not_awaited()
            self.assertEqual(
                {tool["function"]["name"] for tool in payload["tools"]},
                {"read_artifact", "submit_result"})
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertNotIn(diagnostic, json.dumps(payload))
            return completion(calls=[{
                "id": "corrected-submit", "type": "function",
                "function": {"name": "submit_result", "arguments": "{}"},
            }])

        self.transport(instance, respond)
        self.assertEqual(
            await self.launch(instance),
            {"agent-a": "Structured result accepted by the host."})
        callback.assert_awaited_once()
        worker = instance.workers["agent-a"]
        first = worker["requests"][0]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["finish_reason"], "error")
        self.assertEqual(
            first["completion_error_code"], FAILURE_PROVIDER_COMPLETION_ERROR)
        correction = worker["incomplete_final_correction"]
        self.assertEqual(correction["count"], 1)
        self.assertEqual(
            correction["events"][0]["reason"], FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertEqual(correction["status"], "resolved")
        self.assertNotIn("quarantined-submit", worker["tools"])
        self.assertNotIn(diagnostic, state_path.read_text())
        self.assertNotIn(diagnostic, json.dumps(self.events))
        self.assertFalse(any(
            message.get("role") == "provider-error"
            for message in worker["messages"] if isinstance(message, dict)))
        assistant_messages = [
            message for message in worker["messages"]
            if isinstance(message, dict) and message.get("role") == "assistant"
        ]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(
            assistant_messages[0]["tool_calls"][0]["id"], "corrected-submit")

    async def test_provider_error_without_correction_budget_keeps_stronger_failure(self):
        callback = AsyncMock()
        state_path = self.path / "provider-error-no-budget.json"
        instance = OpenRouterRuntime(
            state_path, self.path / "provider-error-no-budget-runtime",
            self.event, callback, model=MODEL, agents=("agent-a",),
            single_node=True, max_requests_per_worker=1,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        self.instances.append(instance)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools",
            {"agent-a": self.path / "provider-error-no-budget-workspace"})

        async def respond(agent, payload):
            data = completion(content="discarded", finish="error")
            data["choices"][0]["error"] = {"message": "discarded provider detail"}
            return data

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "provider generation error"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["failure_code"], FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertEqual(
            worker["requests"][0]["completion_error_code"],
            FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertEqual(worker["incomplete_final_correction"]["status"], "exhausted")
        self.assertNotIn("discarded provider detail", state_path.read_text())

    async def test_generic_runtime_provider_error_has_stable_failure_code(self):
        callback = AsyncMock()
        state_path = self.path / "generic-provider-error.json"
        instance = self.make(
            agents=("agent-a",), single_node=True, on_tool=callback,
            max_requests_per_worker=1)
        await self.prepare(instance)

        async def respond(agent, payload):
            data = completion(content="discarded", finish="error")
            data["choices"][0].update(
                error={"message": "generic discarded detail"},
                message={"invalid": "and quarantined"},
            )
            return data

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "provider generation error"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["failure_code"], FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertEqual(
            worker["requests"][0]["completion_error_code"],
            FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertNotIn("incomplete_final_correction", worker)
        self.assertNotIn("generic discarded detail", instance.state_path.read_text())

    async def test_repeated_provider_errors_keep_priority_at_final_budget(self):
        callback = AsyncMock()
        state_path = self.path / "repeated-provider-error.json"
        instance = OpenRouterRuntime(
            state_path, self.path / "repeated-provider-error-runtime",
            self.event, callback, model=MODEL, agents=("agent-a",),
            single_node=True, max_requests_per_worker=3,
            completion_tool="submit_result", completion_read_tool="read_artifact",
            completion_reserve=1)
        self.instances.append(instance)
        await instance.start()
        tools = [
            {"name": name, "description": "bounded",
             "inputSchema": {"type": "object", "properties": {},
                             "required": [], "additionalProperties": False}}
            for name in ("read_artifact", "submit_result")
        ]
        await instance.create_workers(
            tools, "Use only registered tools",
            {"agent-a": self.path / "repeated-provider-error-workspace"})

        async def respond(agent, payload):
            ordinal = len(instance.workers[agent]["requests"])
            callback.assert_not_awaited()
            data = completion(content="discarded", finish="error")
            data["choices"][0].update(
                error={"message": f"discarded-{ordinal}"},
                message={"tool_calls": [7, None, {"function": "invalid"}]},
            )
            return data

        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "provider generation error"):
            await self.launch(instance)
        callback.assert_not_awaited()
        worker = instance.workers["agent-a"]
        self.assertEqual(worker["failure_code"], FAILURE_PROVIDER_COMPLETION_ERROR)
        self.assertEqual(len(worker["requests"]), 3)
        self.assertTrue(all(
            request.get("completion_error_code") == FAILURE_PROVIDER_COMPLETION_ERROR
            for request in worker["requests"]))
        self.assertEqual(worker["incomplete_final_correction"]["count"], 2)
        self.assertEqual(worker["incomplete_final_correction"]["status"], "exhausted")
        self.assertFalse(any(
            message.get("role") == "assistant"
            for message in worker["messages"] if isinstance(message, dict)))
        self.assertNotIn("discarded-", state_path.read_text())

    async def test_model_substitution_is_not_accepted(self):
        instance = self.make()
        await self.prepare(instance)
        async def respond(agent, payload):
            data = completion()
            data["model"] = "example/unrequested-model"
            return data
        self.transport(instance, respond)
        with self.assertRaisesRegex(RuntimeFailure, "different model"):
            await self.launch(instance)

    async def test_cancelled_request_is_uncertain_and_cannot_replay(self):
        instance = self.make()
        await self.prepare(instance)
        async def blocked(agent, record, payload):
            await instance._request_started(agent, record, time.time())
            await asyncio.Event().wait()
        instance._http = blocked
        await instance.start_turns({a: "Task for " + a for a in AGENTS})
        with self.assertRaises(TimeoutError):
            await instance.wait(timeout=0.05)
        self.assertTrue(all(r["status"] == "uncertain" and "completed_at" not in r
                            for w in instance.workers.values() for r in w["requests"]))
        self.assertTrue(all(r["error_code"] == "request_cancelled"
                            for w in instance.workers.values() for r in w["requests"]))
        self.assertIsNone(instance.state["fatal"])
        await instance.close()
        replay = self.make()
        with self.assertRaises(RecoveryRequired):
            await replay.start()


if __name__ == "__main__":
    unittest.main()
