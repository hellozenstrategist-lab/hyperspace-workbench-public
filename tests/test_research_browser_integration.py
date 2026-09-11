"""Focused Research integration tests for the bounded browser adapter."""
import asyncio
import copy
import json
from pathlib import Path
import time

import pytest

from astra_harness import cli
from astra_harness import research as research_module
from astra_harness.browser_cdp import (
    BrowserCDPError,
    BrowserPolicy,
    BrowserPolicyError,
    BrowserReadUnavailable,
    BrowserRecoveryRequired,
)
from astra_harness.codex_runtime import RuntimeFailure, ToolInputError
from astra_harness.research import (
    DEFAULT_MODELS,
    POLICY_VERSION,
    WORK_SCHEMA,
    Research,
    digest,
)


ENDPOINTS = {
    "account_a": "http://172.17.0.1:9231",
    "account_b": "http://172.17.0.1:9233",
}
SESSIONS = {"agent-a": "account_a", "agent-b": "account_b"}
CRITERIA = [{
    "id": "C1",
    "requirement": "Inspect both dedicated test sessions and preserve the observations",
    "basis": "observation",
}]


def browser_settings(**overrides):
    values = {
        "endpoints": dict(ENDPOINTS),
        "worker_sessions": dict(SESSIONS),
        "allowed_origins": ["https://app.example"],
        "interaction_enabled": False,
        "request_replay_enabled": False,
        "max_actions_per_session": 12,
        "max_replays_per_session": 0,
        "script_eval_sessions": [],
    }
    values.update(overrides)
    return values


def configuration(*, workers=2, browser=None):
    return {
        "objective": "Inspect the two authorized sample web application test accounts.",
        "models": dict(DEFAULT_MODELS),
        "workers": workers,
        "criteria": [],
        "policy_version": POLICY_VERSION,
        "browser": browser_settings() if browser is None else browser,
    }


class FakeBrowserAdapter:
    def __init__(self, artifact_dir, *, events=None, fail_preflight=False):
        self.artifact_dir = Path(artifact_dir)
        self.observation_dir = self.artifact_dir / "observations"
        self.events = events if events is not None else []
        self.fail_preflight = fail_preflight
        self.preflight_count = 0
        self.close_count = 0
        self.calls = []
        self.sequence = 0

    def budget_status(self, session):
        used = sum(call["session"] == session for call in self.calls)
        return {
            "actions_used": used,
            "actions_remaining": 12 - used,
            "replays_used": 0,
            "replays_remaining": 0,
            "currently_eligible_replay_requests": 0,
            "replay_consumes_action": True,
            "navigate_and_click_return_snapshots": True,
        }

    async def preflight(self):
        self.preflight_count += 1
        self.events.append("browser_preflight")
        if self.fail_preflight:
            raise BrowserCDPError("simulated browser preflight failure")
        return {
            "schema_version": 1,
            "sessions": [
                {"session": name, "endpoint": endpoint, "connected": True}
                for name, endpoint in ENDPOINTS.items()
            ],
        }

    def runtime_tools(self):
        return [{
            "name": "browser_inspect",
            "description": "Capture a bounded fake browser observation.",
            "inputSchema": {
                "type": "object",
                "properties": {"session": {"type": "string", "enum": list(ENDPOINTS)}},
                "required": ["session"],
                "additionalProperties": False,
            },
        }]

    async def dispatch_tool(self, name, arguments, *, call_id):
        self.sequence += 1
        session = arguments["session"]
        self.calls.append({
            "name": name,
            "session": session,
            "call_id": call_id,
        })
        observation = {
            "schema_version": 1,
            "origin": "browser_observation",
            "session": session,
            "action": "inspect",
            "sequence": self.sequence,
            "data": {
                "url": "https://app.example/test-account",
                "title": f"Authorized {session}",
            },
        }
        artifact_id = digest(observation)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        path = self.observation_dir / f"{artifact_id}.json"
        path.write_text(json.dumps(observation))
        return {
            "artifact_id": artifact_id,
            "evidence_id": artifact_id,
            "path": str(path),
            "session": session,
            "action": "inspect",
            "sequence": self.sequence,
        }

    def get_observation(self, artifact_id):
        path = self.observation_dir / f"{artifact_id}.json"
        observation = json.loads(path.read_text())
        if digest(observation) != artifact_id:
            raise BrowserCDPError("fake observation integrity failure")
        return observation

    async def close(self):
        self.close_count += 1
        self.events.append("browser_close")


class FakeBrowserFactory:
    def __init__(self, *, events=None, fail_preflight=False):
        self.events = events if events is not None else []
        self.fail_preflight = fail_preflight
        self.calls = []
        self.adapter = None

    def __call__(self, endpoints, artifact_dir, policy):
        self.calls.append({
            "endpoints": copy.deepcopy(endpoints),
            "artifact_dir": Path(artifact_dir),
            "policy": policy,
        })
        self.events.append("browser_factory")
        self.adapter = FakeBrowserAdapter(
            artifact_dir,
            events=self.events,
            fail_preflight=self.fail_preflight,
        )
        return self.adapter


class BrowserDrivingRuntime:
    """Drive browser/read/submit calls through Research's real host callback."""

    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.state_path = Path(state_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.on_event = on_event
        self.on_tool = on_tool
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.registered_tools = []
        self.prompts = {}
        self.cross_session_errors = {}
        self.read_results = {}
        self.runtime_options = copy.deepcopy(kwargs)
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        return {}

    async def create_workers(self, tools, instructions, workspaces):
        self.registered_tools = copy.deepcopy(tools)
        self.workers.update({agent: {"status": "prepared"} for agent in self.agents})
        return {agent: f"thread-{agent}" for agent in self.agents}

    async def start_turns(self, prompts):
        self.prompts = copy.deepcopy(prompts)
        return {agent: f"turn-{agent}" for agent in self.agents}

    async def wait(self, timeout):
        for agent in self.agents:
            prompt = json.loads(self.prompts[agent])
            assignments = prompt["context"]["browser"]["worker_sessions"]
            assigned = assignments.get(agent)
            other = ("account_b" if assigned == "account_a" else "account_a")
            try:
                await self.on_tool(
                    agent,
                    "browser_inspect",
                    {"session": other},
                    f"wrong-session-{agent}",
                )
            except ToolInputError as exc:
                self.cross_session_errors[agent] = str(exc)
            else:
                raise AssertionError("Research admitted a worker into another browser session")

            if assigned is None:
                assert prompt["context"]["browser"]["tools_available"] is False
                assert prompt["context"]["browser"]["capabilities"] == []
                await self.on_tool(
                    agent,
                    "submit_result",
                    {
                        "summary": "Performed independent analysis without owning a browser session.",
                        "findings": [],
                        "unresolved": [],
                    },
                    f"submit-{agent}",
                )
                self.workers[agent]["status"] = "completed"
                continue

            observed = await self.on_tool(
                agent,
                "browser_inspect",
                {"session": assigned},
                f"inspect-{agent}",
            )
            artifact_id = observed["artifact_id"]
            self.read_results[agent] = await self.on_tool(
                agent,
                "read_artifact",
                {"artifact_id": artifact_id},
                f"read-{agent}",
            )
            await self.on_tool(
                agent,
                "submit_result",
                {
                    "summary": f"Inspected {assigned}.",
                    "findings": [{
                        "criterion_id": "C1",
                        "claim": f"Captured a bounded observation from {assigned}.",
                        "evidence_ids": [artifact_id],
                    }],
                    "unresolved": [],
                },
                f"submit-{agent}",
            )
            self.workers[agent]["status"] = "completed"
        return {agent: "Submitted." for agent in self.agents}

    async def close(self):
        self.closed = True


class StopAfterPreflightRuntime:
    events = []
    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        type(self).events.append("runtime_start")
        self.state.update(
            fatal="deliberate permanent runtime stop",
            fatal_kind="permanent_provider",
        )
        raise RuntimeFailure("stop after lifecycle ordering check")

    async def close(self):
        self.closed = True
        type(self).events.append("runtime_close")


class WorkflowRuntime:
    """Complete each real phase while recording its worker and tool boundary."""

    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.phase = Path(state_path).parent.parent.name
        self.on_tool = on_tool
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.registered_tools = []
        self.prompts = {}
        self.read_artifacts = {}
        self.runtime_options = copy.deepcopy(kwargs)
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        return {}

    async def create_workers(self, tools, instructions, workspaces):
        self.registered_tools = copy.deepcopy(tools)
        self.workers.update({agent: {"status": "prepared"} for agent in self.agents})
        return {agent: f"thread-{agent}" for agent in self.agents}

    async def start_turns(self, prompts):
        self.prompts = copy.deepcopy(prompts)
        return {agent: f"turn-{agent}" for agent in self.agents}

    async def _read_required(self, agent, context):
        values = {}
        for artifact_id in context["required_read_artifact_ids"]:
            values[artifact_id] = await self.on_tool(
                agent,
                "read_artifact",
                {"artifact_id": artifact_id},
                f"read-{self.phase}-{agent}-{artifact_id[:8]}",
            )
        self.read_artifacts[agent] = values
        return values

    async def wait(self, timeout):
        for agent in self.agents:
            prompt = json.loads(self.prompts[agent])
            context = prompt["context"]
            required = await self._read_required(agent, context)

            if self.phase == "plan":
                output = {
                    "criteria": copy.deepcopy(CRITERIA),
                    "tasks": [{
                        "worker": worker,
                        "task": f"Investigate the assigned role for {worker}.",
                    } for worker in prompt["assignment"]["workers"]],
                }
            elif self.phase == "work":
                browser = context["browser"]
                if browser["tools_available"]:
                    observed = await self.on_tool(
                        agent,
                        "browser_inspect",
                        {"session": browser["assigned_session"]},
                        f"inspect-{agent}",
                    )
                    artifact_id = observed["artifact_id"]
                    self.read_artifacts[agent][artifact_id] = await self.on_tool(
                        agent,
                        "read_artifact",
                        {"artifact_id": artifact_id},
                        f"read-observation-{agent}",
                    )
                    findings = [{
                        "criterion_id": "C1",
                        "claim": f"Captured the bounded {browser['assigned_session']} observation.",
                        "evidence_ids": [artifact_id],
                    }]
                else:
                    findings = []
                output = {
                    "summary": f"Completed the work assignment for {agent}.",
                    "findings": findings,
                    "unresolved": [],
                }
            elif self.phase == "falsify":
                output = {
                    "summary": "Independently challenged both browser-worker findings.",
                    "findings": [{
                        "criterion_id": "C1",
                        "claim": "Compared the two worker claims against their captured observations.",
                        "evidence_ids": list(required),
                    }],
                    "unresolved": [],
                }
            elif self.phase == "synthesize":
                output = {
                    "answer": "The bounded observations and independent falsification support the result.",
                    "evidence_ids": list(required),
                    "unresolved": [],
                }
            elif self.phase == "review":
                browser_ids = [
                    row["id"] for row in context["artifacts"]
                    if row["origin"] == "browser_observation"
                ]
                output = {
                    "verdict": "accept",
                    "objective_met": True,
                    "summary": "Both isolated observations survived independent falsification.",
                    "checks": [{
                        "criterion_id": "C1",
                        "status": "pass",
                        "reason": "The browser observations are present and were independently checked.",
                        "evidence_ids": browser_ids,
                    }],
                    "next_steps": [],
                }
            else:
                raise AssertionError(f"unexpected phase {self.phase}")

            await self.on_tool(
                agent,
                "submit_result",
                output,
                f"submit-{self.phase}-{agent}",
            )
            self.workers[agent]["status"] = "completed"
        return {agent: "Submitted." for agent in self.agents}

    async def close(self):
        self.closed = True


class IntentDedupBrowserAdapter:
    """Model the adapter's durable-call collision and deduplication contract."""

    def __init__(self, artifact_dir):
        self.observation_dir = Path(artifact_dir) / "observations"
        self.calls = []
        self.executions = []
        self._intents = {}
        self._results = {}

    def runtime_tools(self):
        return [{
            "name": "browser_navigate",
            "description": "Navigate the assigned fake session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session": {"type": "string", "enum": list(ENDPOINTS)},
                    "url": {"type": "string"},
                },
                "required": ["session", "url"],
                "additionalProperties": False,
            },
        }]

    async def dispatch_tool(self, name, arguments, *, call_id):
        intent = digest({"tool": name, "arguments": arguments})
        self.calls.append({
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "call_id": call_id,
            "intent": intent,
        })
        if call_id in self._intents:
            if self._intents[call_id] != intent:
                raise BrowserPolicyError("durable browser call ID was reused for a different intent")
            return copy.deepcopy(self._results[call_id])

        self._intents[call_id] = intent
        self.executions.append(call_id)
        sequence = len(self.executions)
        observation = {
            "schema_version": 1,
            "origin": "browser_observation",
            "session": arguments["session"],
            "action": "navigate",
            "sequence": sequence,
            "data": {"url": arguments["url"], "title": f"Navigation {sequence}"},
        }
        artifact_id = digest(observation)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        (self.observation_dir / f"{artifact_id}.json").write_text(json.dumps(observation))
        result = {
            "artifact_id": artifact_id,
            "evidence_id": artifact_id,
            "session": arguments["session"],
            "action": "navigate",
            "sequence": sequence,
        }
        self._results[call_id] = result
        return copy.deepcopy(result)

    def get_observation(self, artifact_id):
        observation = json.loads(
            (self.observation_dir / f"{artifact_id}.json").read_text()
        )
        if digest(observation) != artifact_id:
            raise BrowserCDPError("fake observation integrity failure")
        return observation


class IntentRecoveryRuntime:
    """End the first work attempt before submission, then recover with an intent."""

    urls = []
    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.attempt = int(Path(state_path).parent.name.removeprefix("attempt-"))
        self.on_tool = on_tool
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        return {}

    async def create_workers(self, tools, instructions, workspaces):
        self.workers.update({agent: {"status": "prepared"} for agent in self.agents})
        return {agent: f"thread-{agent}" for agent in self.agents}

    async def start_turns(self, prompts):
        return {agent: f"turn-{agent}" for agent in self.agents}

    async def wait(self, timeout):
        assert self.agents == ("agent-a",)
        url = type(self).urls[self.attempt - 1]
        observed = await self.on_tool(
            "agent-a",
            "browser_navigate",
            {"session": "account_a", "url": url},
            f"provider-call-attempt-{self.attempt}",
        )
        artifact_id = observed["artifact_id"]
        await self.on_tool(
            "agent-a",
            "read_artifact",
            {"artifact_id": artifact_id},
            f"read-attempt-{self.attempt}",
        )
        if self.attempt > 1:
            await self.on_tool(
                "agent-a",
                "submit_result",
                {
                    "summary": "Recovered and completed the bounded browser action.",
                    "findings": [{
                        "criterion_id": "C1",
                        "claim": "Captured the intended recovered browser observation.",
                        "evidence_ids": [artifact_id],
                    }],
                    "unresolved": [],
                },
                f"submit-attempt-{self.attempt}",
            )
        self.workers["agent-a"]["status"] = "completed"
        return {"agent-a": "Submitted." if self.attempt > 1 else "Ended early."}

    async def close(self):
        self.closed = True


class ReadRefreshBrowserAdapter(IntentDedupBrowserAdapter):
    """Execute a fresh observation whenever Research supplies a fresh call ID."""

    def runtime_tools(self):
        return [
            {
                "name": "browser_inspect",
                "description": "Capture the current bounded browser state.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session": {"type": "string", "enum": list(ENDPOINTS)},
                    },
                    "required": ["session"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "browser_read_response",
                "description": "Read a bounded response observed in the assigned session.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session": {"type": "string", "enum": list(ENDPOINTS)},
                        "request_id": {"type": "string"},
                    },
                    "required": ["session", "request_id"],
                    "additionalProperties": False,
                },
            },
        ]

    async def dispatch_tool(self, name, arguments, *, call_id):
        intent = digest({"tool": name, "arguments": arguments})
        self.calls.append({
            "name": name,
            "arguments": copy.deepcopy(arguments),
            "call_id": call_id,
            "intent": intent,
        })
        if call_id in self._intents:
            if self._intents[call_id] != intent:
                raise BrowserPolicyError("durable browser call ID was reused for a different intent")
            return copy.deepcopy(self._results[call_id])

        self._intents[call_id] = intent
        self.executions.append(call_id)
        sequence = len(self.executions)
        action = name.removeprefix("browser_")
        observation = {
            "schema_version": 1,
            "origin": "browser_observation",
            "session": arguments["session"],
            "action": action,
            "sequence": sequence,
            "data": {
                "sample": sequence,
                **({"request_id": arguments["request_id"]}
                   if "request_id" in arguments else {}),
            },
        }
        artifact_id = digest(observation)
        self.observation_dir.mkdir(parents=True, exist_ok=True)
        (self.observation_dir / f"{artifact_id}.json").write_text(json.dumps(observation))
        result = {
            "artifact_id": artifact_id,
            "evidence_id": artifact_id,
            "session": arguments["session"],
            "action": action,
            "sequence": sequence,
        }
        self._results[call_id] = result
        return copy.deepcopy(result)


class ReadIntentRecoveryRuntime:
    """Repeat one identical read-only intent in a fresh phase attempt."""

    tool_name = None
    arguments = None
    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.attempt = int(Path(state_path).parent.name.removeprefix("attempt-"))
        self.on_tool = on_tool
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        return {}

    async def create_workers(self, tools, instructions, workspaces):
        self.workers.update({agent: {"status": "prepared"} for agent in self.agents})
        return {agent: f"thread-{agent}" for agent in self.agents}

    async def start_turns(self, prompts):
        return {agent: f"turn-{agent}" for agent in self.agents}

    async def wait(self, timeout):
        assert self.agents == ("agent-a",)
        observed = await self.on_tool(
            "agent-a",
            type(self).tool_name,
            copy.deepcopy(type(self).arguments),
            f"provider-read-attempt-{self.attempt}",
        )
        artifact_id = observed["artifact_id"]
        await self.on_tool(
            "agent-a",
            "read_artifact",
            {"artifact_id": artifact_id},
            f"read-observation-attempt-{self.attempt}",
        )
        if self.attempt > 1:
            await self.on_tool(
                "agent-a",
                "submit_result",
                {
                    "summary": "Recovered using a fresh current-state observation.",
                    "findings": [{
                        "criterion_id": "C1",
                        "claim": "The read-only intent was sampled again in the fresh attempt.",
                        "evidence_ids": [artifact_id],
                    }],
                    "unresolved": [],
                },
                "submit-fresh-observation",
            )
        self.workers["agent-a"]["status"] = "completed"
        return {"agent-a": "Submitted." if self.attempt > 1 else "Ended early."}

    async def close(self):
        self.closed = True


class UncertainActionBrowserAdapter(FakeBrowserAdapter):
    """Return an observation for inspect but an unknown outcome for click."""

    async def dispatch_tool(self, name, arguments, *, call_id):
        if name == "browser_click":
            self.calls.append({
                "name": name,
                "session": arguments["session"],
                "call_id": call_id,
            })
            raise BrowserRecoveryRequired(
                "adapter-private-detail: the browser action may have run"
            )
        return await super().dispatch_tool(name, arguments, call_id=call_id)

    def runtime_tools(self):
        return super().runtime_tools() + [{
            "name": "browser_click",
            "description": "Click a bounded element in the assigned fake session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session": {"type": "string", "enum": list(ENDPOINTS)},
                    "selector": {"type": "string"},
                },
                "required": ["session", "selector"],
                "additionalProperties": False,
            },
        }]


class UncertainActionRuntime:
    """Recover from an uncertain action by inspecting rather than replaying it."""

    instances = []

    def __init__(self, state_path, artifacts_dir, on_event, on_tool, *, agents, **kwargs):
        self.on_tool = on_tool
        self.agents = tuple(agents)
        self.workers = {}
        self.state = {"fatal": None, "workers": self.workers}
        self.registered_tools = []
        self.uncertain_result = None
        self.inspection_result = None
        self.closed = False
        type(self).instances.append(self)

    async def start(self):
        return {}

    async def create_workers(self, tools, instructions, workspaces):
        self.registered_tools = copy.deepcopy(tools)
        self.workers.update({agent: {"status": "prepared"} for agent in self.agents})
        return {agent: f"thread-{agent}" for agent in self.agents}

    async def start_turns(self, prompts):
        return {agent: f"turn-{agent}" for agent in self.agents}

    async def wait(self, timeout):
        assert self.agents == ("agent-a",)
        self.uncertain_result = await self.on_tool(
            "agent-a",
            "browser_click",
            {"session": "account_a", "selector": "button[data-test='save']"},
            "click-with-unknown-outcome",
        )
        self.inspection_result = await self.on_tool(
            "agent-a",
            "browser_inspect",
            {"session": "account_a"},
            "inspect-after-unknown-outcome",
        )
        artifact_id = self.inspection_result["artifact_id"]
        await self.on_tool(
            "agent-a",
            "read_artifact",
            {"artifact_id": artifact_id},
            "read-current-state",
        )
        await self.on_tool(
            "agent-a",
            "submit_result",
            {
                "summary": "Inspected current state without replaying the uncertain click.",
                "findings": [{
                    "criterion_id": "C1",
                    "claim": "Captured current browser state after the uncertain action.",
                    "evidence_ids": [artifact_id],
                }],
                "unresolved": ["The click outcome itself remains uncertain."],
            },
            "submit-after-inspection",
        )
        self.workers["agent-a"]["status"] = "completed"
        return {"agent-a": "Submitted."}

    async def close(self):
        self.closed = True


class UnavailableReadBrowserAdapter(FakeBrowserAdapter):
    """Model Chrome expiring a valid response body before it can be read."""

    async def dispatch_tool(self, name, arguments, *, call_id):
        if name == "browser_read_response":
            self.calls.append({
                "name": name,
                "session": arguments["session"],
                "call_id": call_id,
            })
            raise BrowserReadUnavailable(
                "adapter-private-detail: response body was evicted"
            )
        return await super().dispatch_tool(name, arguments, call_id=call_id)

    def runtime_tools(self):
        return super().runtime_tools() + [{
            "name": "browser_read_response",
            "description": "Read a completed bounded response.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session": {"type": "string", "enum": list(ENDPOINTS)},
                    "request_id": {"type": "string"},
                },
                "required": ["session", "request_id"],
                "additionalProperties": False,
            },
        }]


class UnavailableReadRuntime(UncertainActionRuntime):
    """Continue normally after a valid Chrome response becomes unavailable."""

    instances = []

    async def wait(self, timeout):
        assert self.agents == ("agent-a",)
        self.unavailable_result = await self.on_tool(
            "agent-a",
            "browser_read_response",
            {"session": "account_a", "request_id": "finished-request-1"},
            "read-expired-response",
        )
        self.inspection_result = await self.on_tool(
            "agent-a",
            "browser_inspect",
            {"session": "account_a"},
            "inspect-after-expired-response",
        )
        artifact_id = self.inspection_result["artifact_id"]
        await self.on_tool(
            "agent-a",
            "read_artifact",
            {"artifact_id": artifact_id},
            "read-current-state-after-expired-response",
        )
        await self.on_tool(
            "agent-a",
            "submit_result",
            {
                "summary": "Continued after the expired response body.",
                "findings": [{
                    "criterion_id": "C1",
                    "claim": "Captured a fresh browser observation after the unavailable read.",
                    "evidence_ids": [artifact_id],
                }],
                "unresolved": ["The expired response body was unavailable."],
            },
            "submit-after-expired-response",
        )
        self.workers["agent-a"]["status"] = "completed"
        return {"agent-a": "Submitted."}


@pytest.fixture(autouse=True)
def isolated_fakes(monkeypatch):
    BrowserDrivingRuntime.instances = []
    WorkflowRuntime.instances = []
    IntentRecoveryRuntime.instances = []
    IntentRecoveryRuntime.urls = []
    ReadIntentRecoveryRuntime.instances = []
    ReadIntentRecoveryRuntime.tool_name = None
    ReadIntentRecoveryRuntime.arguments = None
    UncertainActionRuntime.instances = []
    UnavailableReadRuntime.instances = []
    StopAfterPreflightRuntime.events = []
    StopAfterPreflightRuntime.instances = []
    monkeypatch.setattr("astra_harness.api_settings.openrouter_settings", lambda: {})
    monkeypatch.setattr(research_module, "PHASE_RETRY_BASE_SECONDS", 0)


def test_valid_browser_config_is_saved_without_connecting(tmp_path):
    factory = FakeBrowserFactory()
    controller = Research(tmp_path, config=configuration(), browser_factory=factory)
    try:
        report = controller.report()
        assert factory.calls == []
        assert report["browser"] == {
            "enabled": True,
            "sessions": ["account_a", "account_b"],
            "worker_sessions": SESSIONS,
            "allowed_origins": ["https://app.example"],
            "interaction_enabled": False,
            "request_replay_enabled": False,
            "max_actions_per_session": 12,
            "max_replays_per_session": 0,
            "script_eval_sessions": [],
            "observations": 0,
            "preflight": None,
        }
    finally:
        controller.close()


def test_cli_dry_run_builds_exact_read_only_browser_config(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as exited:
        cli.main([
            "research",
            "Inspect the authorized test accounts",
            "--workers", "2",
            "--browser-cdp", ENDPOINTS["account_a"],
            "--browser-cdp", ENDPOINTS["account_b"],
            "--browser-origin", "https://app.example",
            "--browser-read-only",
            "--max-browser-actions", "7",
            "--max-browser-replays", "0",
            "--run-dir", str(run_dir),
            "--dry-run",
        ])

    assert exited.value.code == 0
    state = json.loads((run_dir / "research.json").read_text())
    assert state["config"]["browser"] == {
        "endpoints": ENDPOINTS,
        "worker_sessions": SESSIONS,
        "allowed_origins": ["https://app.example"],
        "interaction_enabled": False,
        "request_replay_enabled": False,
        "max_actions_per_session": 7,
        "max_replays_per_session": 0,
        "script_eval_sessions": [],
    }
    assert state["status"] == "ready"
    assert not list(run_dir.rglob("runtime_state.json"))
    assert not (run_dir / "browser").exists()


def test_cli_resume_rejects_browser_policy_changes_without_mutating_run(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as created:
        cli.main([
            "research", "Inspect the authorized test accounts",
            "--workers", "2",
            "--browser-cdp", ENDPOINTS["account_a"],
            "--browser-cdp", ENDPOINTS["account_b"],
            "--browser-origin", "https://app.example",
            "--browser-read-only",
            "--run-dir", str(run_dir),
            "--dry-run",
        ])
    assert created.value.code == 0
    before = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}

    with pytest.raises(SystemExit) as resumed:
        cli.main([
            "research", "--resume", "--run-dir", str(run_dir), "--dry-run",
            "--browser-origin", "https://app.example",
        ])

    assert resumed.value.code == 1
    assert before == {path: path.read_bytes() for path in before}


@pytest.mark.parametrize("browser, workers", [
    (browser_settings(endpoints={
        "account_a": ENDPOINTS["account_a"],
        "account_b": ENDPOINTS["account_a"],
    }), 2),
    (browser_settings(worker_sessions={"agent-a": "account_b", "agent-b": "account_a"}), 2),
    (browser_settings(), 1),
    (browser_settings(allowed_origins=["https://app.example/private"]), 2),
    (browser_settings(endpoints={
        "account_a": "http://8.8.8.8:9231",
        "account_b": ENDPOINTS["account_b"],
    }), 2),
])
def test_invalid_browser_scope_or_session_config_fails_before_factory(
        tmp_path, browser, workers):
    factory = FakeBrowserFactory()
    with pytest.raises(ValueError):
        Research(
            tmp_path,
            config=configuration(workers=workers, browser=browser),
            browser_factory=factory,
        )
    assert factory.calls == []


def test_work_phase_enforces_session_ownership_and_reads_new_observations(tmp_path):
    controller = Research(
        tmp_path,
        config=configuration(workers=3),
        runtime_class=BrowserDrivingRuntime,
    )
    adapter = FakeBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.state["criteria"] = copy.deepcopy(CRITERIA)
    controller.deadline = time.monotonic() + 30
    prompts = {
        "agent-a": {"role": "worker", "task": "Inspect account_a"},
        "agent-b": {"role": "worker", "task": "Inspect account_b"},
        "agent-c": {"role": "worker", "task": "Falsify the browser findings independently"},
    }
    try:
        outputs, artifacts = asyncio.run(controller._stage(
            "work",
            DEFAULT_MODELS["worker"],
            prompts,
            WORK_SCHEMA,
            [],
        ))
    finally:
        controller.close()

    runtime = BrowserDrivingRuntime.instances[-1]
    assert runtime.runtime_options["max_tool_calls_per_response"] == 8
    tool_names = {tool["name"] for tool in runtime.registered_tools}
    assert tool_names == {"read_artifact", "submit_result", "browser_inspect"}
    assert set(runtime.cross_session_errors) == set(controller.workers)
    assert {(call["session"], call["name"]) for call in adapter.calls} == {
        ("account_a", "browser_inspect"),
        ("account_b", "browser_inspect"),
    }
    assert len({call["call_id"] for call in adapter.calls}) == 2
    assert set(outputs) == set(controller.workers)
    assert set(artifacts) == set(controller.workers)
    assert len(controller.state["observation_ids"]) == 2
    assert set(controller._browser_observations(1, "work")) == set(controller.state["observation_ids"])
    for agent, value in runtime.read_results.items():
        assert value["origin"] == "browser_observation"
        assert value["content"]["session"] == SESSIONS[agent]
        cited = outputs[agent]["findings"][0]["evidence_ids"][0]
        assert cited in controller.state["observation_ids"]
    for agent, prompt in runtime.prompts.items():
        browser = json.loads(prompt)["context"]["browser"]
        if browser["assigned_session"] is None:
            assert "attempt_budget" not in browser
            continue
        assert browser["attempt_budget"]["actions_used"] == 0
        assert browser["attempt_budget"]["actions_remaining"] == 12
        assert browser["attempt_budget"]["replay_consumes_action"] is True
        assert "redundant inspect" in browser["action_guidance"]


def test_three_worker_browser_loop_falsifies_after_parallel_session_work(tmp_path):
    controller = Research(
        tmp_path,
        config=configuration(workers=3),
        runtime_class=WorkflowRuntime,  # type: ignore[arg-type]
    )
    adapter = FakeBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.deadline = time.monotonic() + 30
    try:
        asyncio.run(controller._loop())
    finally:
        controller.close()

    assert controller.state["status"] == "completed"
    assert [runtime.phase for runtime in WorkflowRuntime.instances] == [
        "plan", "work", "falsify", "synthesize", "review",
    ]

    stages = {runtime.phase: runtime for runtime in WorkflowRuntime.instances}
    assert stages["plan"].runtime_options["max_tokens"] == 8192
    work = stages["work"]
    assert work.runtime_options["max_tool_calls_per_response"] == 8
    assert work.runtime_options["max_tokens"] == 4096
    assert work.agents == ("agent-a", "agent-b")
    assert set(work.prompts) == {"agent-a", "agent-b"}
    assert {tool["name"] for tool in work.registered_tools} == {
        "read_artifact", "submit_result", "browser_inspect",
    }
    assert {(call["session"], call["name"]) for call in adapter.calls} == {
        ("account_a", "browser_inspect"),
        ("account_b", "browser_inspect"),
    }

    falsify = stages["falsify"]
    assert falsify.runtime_options["max_tool_calls_per_response"] == 16
    assert falsify.runtime_options["max_tokens"] == 4096
    assert falsify.agents == ("agent-c",)
    assert set(falsify.prompts) == {"agent-c"}
    assert {tool["name"] for tool in falsify.registered_tools} == {
        "read_artifact", "submit_result",
    }
    falsify_prompt = json.loads(falsify.prompts["agent-c"])
    catalog = falsify_prompt["context"]["artifacts"]
    worker_ids = {
        row["id"] for row in catalog
        if row["label"] in {"round-1/work/agent-a", "round-1/work/agent-b"}
    }
    browser_ids = {
        row["id"] for row in catalog
        if row["origin"] == "browser_observation"
        and row["label"].startswith("round-1/work/browser-")
    }
    expected_reads = worker_ids | browser_ids
    assert len(worker_ids) == 2
    assert len(browser_ids) == 2
    assert set(falsify_prompt["context"]["required_read_artifact_ids"]) == expected_reads
    assert set(falsify.read_artifacts["agent-c"]) == expected_reads
    assert {
        value["origin"] for value in falsify.read_artifacts["agent-c"].values()
    } == {"model_analysis", "browser_observation"}

    synthesis = stages["synthesize"]
    assert synthesis.runtime_options["max_tool_calls_per_response"] == 16
    assert synthesis.runtime_options["max_tokens"] == 8192
    synthesis_prompt = json.loads(synthesis.prompts["agent-a"])
    falsification_ids = {
        row["id"] for row in synthesis_prompt["context"]["artifacts"]
        if row["label"] == "round-1/falsify/agent-c"
    }
    assert len(falsification_ids) == 1
    assert falsification_ids <= set(
        synthesis_prompt["context"]["required_read_artifact_ids"]
    )
    assert falsification_ids <= set(controller.state["history"][0]["artifact_ids"])
    assert stages["review"].runtime_options["max_tokens"] == 8192


def test_two_worker_browser_loop_skips_falsification(tmp_path):
    controller = Research(
        tmp_path,
        config=configuration(workers=2),
        runtime_class=WorkflowRuntime,  # type: ignore[arg-type]
    )
    adapter = FakeBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.deadline = time.monotonic() + 30
    try:
        asyncio.run(controller._loop())
    finally:
        controller.close()

    assert controller.state["status"] == "completed"
    assert [runtime.phase for runtime in WorkflowRuntime.instances] == [
        "plan", "work", "synthesize", "review",
    ]
    assert all(runtime.phase != "falsify" for runtime in WorkflowRuntime.instances)
    work = next(runtime for runtime in WorkflowRuntime.instances if runtime.phase == "work")
    assert work.agents == ("agent-a", "agent-b")
    assert set(work.prompts) == {"agent-a", "agent-b"}

    synthesis = next(
        runtime for runtime in WorkflowRuntime.instances
        if runtime.phase == "synthesize"
    )
    prompt = json.loads(synthesis.prompts["agent-a"])
    expected_reads = {
        row["id"] for row in prompt["context"]["artifacts"]
        if row["label"] in {"round-1/work/agent-a", "round-1/work/agent-b"}
        or row["origin"] == "browser_observation"
    }
    assert len(expected_reads) == 4
    assert set(prompt["context"]["required_read_artifact_ids"]) == expected_reads


def test_unauth_browser_worker_keeps_work_and_gets_exclusive_eval_advertisement(tmp_path):
    endpoints = {**ENDPOINTS, "unauth": "http://172.17.0.1:9235"}
    assignments = {**SESSIONS, "agent-c": "unauth"}
    controller = Research(
        tmp_path,
        config=configuration(workers=3, browser=browser_settings(
            endpoints=endpoints,
            worker_sessions=assignments,
            interaction_enabled=True,
            request_replay_enabled=True,
            max_replays_per_session=2,
            script_eval_sessions=["unauth"],
        )),
        runtime_class=WorkflowRuntime,  # type: ignore[arg-type]
    )
    adapter = FakeBrowserAdapter(tmp_path / "browser")
    adapter.runtime_tools = lambda: [
        {
            "name": "browser_inspect", "description": "Inspect assigned session",
            "inputSchema": {"type": "object", "additionalProperties": False,
                            "properties": {"session": {"type": "string", "enum": list(endpoints)}},
                            "required": ["session"]},
        },
        {
            "name": "browser_evaluate", "description": "Evaluate only in unauth session",
            "inputSchema": {"type": "object", "additionalProperties": False,
                            "properties": {"session": {"type": "string", "enum": ["unauth"]},
                                           "code": {"type": "string"}},
                            "required": ["session", "code"]},
        },
    ]
    controller.browser = adapter
    controller.deadline = time.monotonic() + 30
    try:
        asyncio.run(controller._loop())
    finally:
        controller.close()

    phases = [runtime.phase for runtime in WorkflowRuntime.instances]
    assert phases == ["plan", "work", "synthesize", "review"]
    work = next(runtime for runtime in WorkflowRuntime.instances if runtime.phase == "work")
    assert isinstance(work.registered_tools, dict)
    for agent, assigned in assignments.items():
        tools = work.registered_tools[agent]
        names = [tool["name"] for tool in tools]
        assert ("browser_evaluate" in names) is (agent == "agent-c")
        for tool in tools:
            session = tool["inputSchema"]["properties"].get("session")
            if session:
                assert session["enum"] == [assigned]
        prompt = json.loads(work.prompts[agent])
        capabilities = prompt["context"]["browser"]["capabilities"]
        assert ("browser_evaluate" in capabilities) is (agent == "agent-c")
    work_artifacts = [
        controller.get_artifact(ref)["label"] for ref in controller.state["history"][0]["artifact_ids"]
    ]
    assert "round-1/work/agent-c" in work_artifacts
    assert "round-1/falsify/agent-c" not in work_artifacts


@pytest.mark.parametrize(
    "urls, expected_unique_call_ids, expected_executions",
    [
        (["https://app.example/same", "https://app.example/same"], 1, 1),
        (["https://app.example/first", "https://app.example/changed"], 2, 2),
    ],
    ids=["same-intent-deduplicates", "changed-intent-gets-new-id"],
)
def test_browser_call_ids_follow_intent_across_phase_recovery(
        tmp_path, urls, expected_unique_call_ids, expected_executions):
    IntentRecoveryRuntime.urls = urls
    controller = Research(
        tmp_path,
        config=configuration(
            workers=2,
            browser=browser_settings(interaction_enabled=True),
        ),
        limits={
            "max_rounds": 1,
            "max_requests": 40,
            "max_minutes": 1,
            "max_phase_attempts": 2,
            "request_timeout_seconds": 5,
        },
        runtime_class=IntentRecoveryRuntime,
    )
    adapter = IntentDedupBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.state["criteria"] = copy.deepcopy(CRITERIA)
    controller.deadline = time.monotonic() + 30
    try:
        outputs, artifacts = asyncio.run(controller._stage(
            "work",
            DEFAULT_MODELS["worker"],
            {"agent-a": {"role": "worker", "task": "Navigate the assigned session."}},
            WORK_SCHEMA,
            [],
        ))
    finally:
        controller.close()

    assert set(outputs) == {"agent-a"}
    assert set(artifacts) == {"agent-a"}
    assert len(IntentRecoveryRuntime.instances) == 2
    assert len(adapter.calls) == 2
    assert len({call["call_id"] for call in adapter.calls}) == expected_unique_call_ids
    assert len(adapter.executions) == expected_executions
    assert all(call["call_id"].startswith("research-") for call in adapter.calls)
    if expected_unique_call_ids == 1:
        assert adapter.calls[0]["intent"] == adapter.calls[1]["intent"]
    else:
        assert adapter.calls[0]["intent"] != adapter.calls[1]["intent"]
    cited = outputs["agent-a"]["findings"][0]["evidence_ids"]
    assert len(cited) == 1
    assert cited[0] in controller.state["observation_ids"]
    assert controller.get_artifact(cited[0])["content"]["data"]["url"] == urls[-1]
    attempts = sorted((tmp_path / "rounds" / "001" / "work").glob("attempt-*"))
    assert [json.loads((path / "attempt.json").read_text())["status"] for path in attempts] == [
        "failed", "completed",
    ]


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("browser_inspect", {"session": "account_a"}),
        ("browser_read_response", {
            "session": "account_a",
            "request_id": "request-from-current-page",
        }),
    ],
    ids=["inspect", "read-response"],
)
def test_read_only_browser_intent_executes_again_in_fresh_phase_attempt(
        tmp_path, tool_name, arguments):
    ReadIntentRecoveryRuntime.tool_name = tool_name
    ReadIntentRecoveryRuntime.arguments = arguments
    controller = Research(
        tmp_path,
        config=configuration(workers=2),
        limits={
            "max_rounds": 1,
            "max_requests": 40,
            "max_minutes": 1,
            "max_phase_attempts": 2,
            "request_timeout_seconds": 5,
        },
        runtime_class=ReadIntentRecoveryRuntime,
    )
    adapter = ReadRefreshBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.state["criteria"] = copy.deepcopy(CRITERIA)
    controller.deadline = time.monotonic() + 30
    try:
        outputs, artifacts = asyncio.run(controller._stage(
            "work",
            DEFAULT_MODELS["worker"],
            {"agent-a": {"role": "worker", "task": "Sample current browser evidence."}},
            WORK_SCHEMA,
            [],
        ))
    finally:
        controller.close()

    assert set(outputs) == {"agent-a"}
    assert set(artifacts) == {"agent-a"}
    assert len(ReadIntentRecoveryRuntime.instances) == 2
    assert len(adapter.calls) == 2
    assert adapter.calls[0]["name"] == adapter.calls[1]["name"] == tool_name
    assert adapter.calls[0]["arguments"] == adapter.calls[1]["arguments"] == arguments
    assert adapter.calls[0]["intent"] == adapter.calls[1]["intent"]
    assert adapter.calls[0]["call_id"] != adapter.calls[1]["call_id"]
    assert adapter.executions == [call["call_id"] for call in adapter.calls]
    assert len(controller.state["observation_ids"]) == 2
    cited = outputs["agent-a"]["findings"][0]["evidence_ids"]
    assert len(cited) == 1
    latest = controller.get_artifact(cited[0])["content"]
    assert latest["action"] == tool_name.removeprefix("browser_")
    assert latest["sequence"] == 2
    attempts = sorted((tmp_path / "rounds/001/work").glob("attempt-*"))
    assert [json.loads((path / "attempt.json").read_text())["status"] for path in attempts] == [
        "failed", "completed",
    ]


def test_uncertain_browser_action_becomes_non_evidence_and_work_inspects_state(tmp_path):
    controller = Research(
        tmp_path,
        config=configuration(
            workers=2,
            browser=browser_settings(interaction_enabled=True),
        ),
        limits={
            "max_rounds": 1,
            "max_requests": 20,
            "max_minutes": 1,
            "max_phase_attempts": 1,
            "request_timeout_seconds": 5,
        },
        runtime_class=UncertainActionRuntime,
    )
    adapter = UncertainActionBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.state["criteria"] = copy.deepcopy(CRITERIA)
    controller.deadline = time.monotonic() + 30
    try:
        outputs, artifacts = asyncio.run(controller._stage(
            "work",
            DEFAULT_MODELS["worker"],
            {"agent-a": {"role": "worker", "task": "Try the action and verify current state."}},
            WORK_SCHEMA,
            [],
        ))
    finally:
        controller.close()

    assert set(outputs) == {"agent-a"}
    assert set(artifacts) == {"agent-a"}
    assert len(UncertainActionRuntime.instances) == 1
    runtime = UncertainActionRuntime.instances[0]
    uncertain = runtime.uncertain_result
    assert uncertain["accepted"] is False
    assert uncertain["outcome"] == "uncertain"
    assert uncertain["session"] == "account_a"
    assert uncertain["action"] == "browser_click"
    assert "artifact_id" not in uncertain
    assert "evidence_id" not in uncertain
    assert "Do not repeat this intent" in uncertain["next_action"]
    assert "browser_inspect" in uncertain["next_action"]
    assert "current state" in uncertain["next_action"]
    assert "adapter-private-detail" not in json.dumps(uncertain)
    assert len(json.dumps(uncertain)) < 1_000

    assert [call["name"] for call in adapter.calls] == [
        "browser_click", "browser_inspect",
    ]
    assert runtime.inspection_result["origin"] == "browser_observation"
    assert len(controller.state["observation_ids"]) == 1
    assert outputs["agent-a"]["unresolved"] == [
        "The click outcome itself remains uncertain."
    ]
    attempt = tmp_path / "rounds" / "001" / "work" / "attempt-001" / "attempt.json"
    metadata = json.loads(attempt.read_text())
    assert metadata["status"] == "completed"
    assert "failure_code" not in metadata
    assert metadata["completed_after_runtime_failure"] is False


def test_unavailable_response_read_is_normal_non_evidence_and_work_continues(tmp_path):
    controller = Research(
        tmp_path,
        config=configuration(workers=2),
        limits={
            "max_rounds": 1,
            "max_requests": 20,
            "max_minutes": 1,
            "max_phase_attempts": 1,
            "request_timeout_seconds": 5,
        },
        runtime_class=UnavailableReadRuntime,
    )
    adapter = UnavailableReadBrowserAdapter(tmp_path / "browser")
    controller.browser = adapter
    controller.state["criteria"] = copy.deepcopy(CRITERIA)
    controller.deadline = time.monotonic() + 30
    try:
        outputs, artifacts = asyncio.run(controller._stage(
            "work",
            DEFAULT_MODELS["worker"],
            {"agent-a": {"role": "worker", "task": "Read one response, then continue."}},
            WORK_SCHEMA,
            [],
        ))
    finally:
        controller.close()

    assert set(outputs) == {"agent-a"}
    assert set(artifacts) == {"agent-a"}
    runtime = UnavailableReadRuntime.instances[0]
    unavailable = runtime.unavailable_result
    assert unavailable["accepted"] is False
    assert unavailable["outcome"] == "unavailable"
    assert unavailable["session"] == "account_a"
    assert unavailable["action"] == "browser_read_response"
    assert "artifact_id" not in unavailable
    assert "evidence_id" not in unavailable
    assert "Do not retry" in unavailable["next_action"]
    assert "adapter-private-detail" not in json.dumps(unavailable)
    assert [call["name"] for call in adapter.calls] == [
        "browser_read_response", "browser_inspect",
    ]
    assert len(controller.state["observation_ids"]) == 1
    assert outputs["agent-a"]["unresolved"] == [
        "The expired response body was unavailable."
    ]
    attempt = tmp_path / "rounds" / "001" / "work" / "attempt-001" / "attempt.json"
    metadata = json.loads(attempt.read_text())
    assert metadata["status"] == "completed"
    assert "failure_code" not in metadata


def test_run_preflights_before_models_and_closes_browser_afterward(tmp_path):
    events = StopAfterPreflightRuntime.events
    factory = FakeBrowserFactory(events=events)
    controller = Research(
        tmp_path,
        config=configuration(),
        runtime_class=StopAfterPreflightRuntime,
        browser_factory=factory,
    )
    try:
        result = asyncio.run(controller.run())
    finally:
        controller.close()

    assert result["status"] == "blocked"
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["endpoints"] == ENDPOINTS
    assert call["artifact_dir"] == tmp_path / "browser"
    assert isinstance(call["policy"], BrowserPolicy)
    assert call["policy"].allowed_origins == frozenset({"https://app.example"})
    assert events.index("browser_preflight") < events.index("runtime_start")
    assert events[-1] == "browser_close"
    assert factory.adapter.preflight_count == 1
    assert factory.adapter.close_count == 1
    preflight = result["browser"]["preflight"]
    assert preflight["status"] == "connected"
    assert preflight["sessions"] == ["account_a", "account_b"]
    assert isinstance(preflight["connected_at"], str)


def test_preflight_failure_closes_adapter_and_never_constructs_model_runtime(tmp_path):
    events = StopAfterPreflightRuntime.events
    factory = FakeBrowserFactory(events=events, fail_preflight=True)
    controller = Research(
        tmp_path,
        config=configuration(),
        runtime_class=StopAfterPreflightRuntime,
        browser_factory=factory,
    )
    try:
        result = asyncio.run(controller.run())
    finally:
        controller.close()

    assert result["status"] == "blocked"
    assert factory.adapter.preflight_count == 1
    assert factory.adapter.close_count == 1
    assert StopAfterPreflightRuntime.instances == []
    assert events == ["browser_factory", "browser_preflight", "browser_close"]


def test_cli_dry_run_builds_unauthenticated_evaluation_tier(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(SystemExit) as exited:
        cli.main([
            "research", "Inspect the authorized test accounts",
            "--workers", "3",
            "--browser-cdp", ENDPOINTS["account_a"],
            "--browser-cdp", ENDPOINTS["account_b"],
            "--browser-unauth-cdp", "http://127.0.0.1:9234",
            "--browser-origin", "https://app.example",
            "--run-dir", str(run_dir),
            "--dry-run",
        ])
    assert exited.value.code == 0
    browser = json.loads((run_dir / "research.json").read_text())["config"]["browser"]
    assert browser["endpoints"]["unauth"] == "http://127.0.0.1:9234"
    assert browser["worker_sessions"]["agent-c"] == "unauth"
    assert browser["script_eval_sessions"] == ["unauth"]


def test_cli_unauthenticated_evaluation_tier_requires_three_workers(tmp_path):
    with pytest.raises(SystemExit) as exited:
        cli.main([
            "research", "Inspect the authorized test accounts",
            "--workers", "2",
            "--browser-cdp", ENDPOINTS["account_a"],
            "--browser-cdp", ENDPOINTS["account_b"],
            "--browser-unauth-cdp", "http://127.0.0.1:9234",
            "--browser-origin", "https://app.example",
            "--run-dir", str(tmp_path / "run"),
            "--dry-run",
        ])
    assert exited.value.code == 1


@pytest.mark.parametrize("eval_sessions", [["account_a"], ["account_b"], ["absent"]])
def test_script_evaluation_can_never_designate_an_authenticated_session(tmp_path, eval_sessions):
    with pytest.raises(ValueError):
        Research(tmp_path, config=configuration(
            browser=browser_settings(script_eval_sessions=eval_sessions)))
