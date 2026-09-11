"""Bounded Chrome DevTools Protocol access for authorized browser validation.

The adapter deliberately exposes a small vocabulary rather than arbitrary CDP or
JavaScript.  Every model-visible operation has a durable call id, a per-browser
budget, an exact-origin policy, and a content-addressed, redacted observation.
Browser profiles remain isolated by requiring one CDP endpoint per session.

This module is intentionally stdlib-only.  ``WebSocketCDPTransport`` implements
the small RFC 6455 subset Chrome needs; tests can inject a transport factory.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
import base64
import hashlib
import http.client
import inspect as python_inspect
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import ssl
import struct
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .schema import atomic_json, canonical, now


class BrowserCDPError(RuntimeError):
    """The bounded CDP adapter or transport could not complete an operation."""


class BrowserPolicyError(BrowserCDPError, ValueError):
    """An operation was rejected before it reached the browser."""


class BrowserReadUnavailable(BrowserPolicyError):
    """A valid side-effect-free response read could not be served by Chrome."""


class _BrowserTextTargetError(BrowserPolicyError):
    """A fixed visible-text lookup safely resolved to zero or multiple nodes."""


class BrowserRecoveryRequired(BrowserCDPError):
    """A durable browser call may have run and must not be replayed blindly."""


_SENSITIVE_KEYS = re.compile(
    r"(?:authorization|proxy.authorization|cookie|set.cookie|password|passwd|"
    r"secret|api.?key|access.?token|refresh.?token|session.?token|csrf|xsrf)", re.I
)
_SENSITIVE_QUERY_KEYS = re.compile(
    r"(?:token|code|key|secret|password|passwd|auth|signature|sig|session|csrf|xsrf)", re.I
)
_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}")
_OPAQUE = re.compile(r"(?<![A-Za-z0-9_-])(?=[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-]))"
                     r"(?=[A-Za-z0-9_-]*[A-Za-z])(?=[A-Za-z0-9_-]*[0-9])[A-Za-z0-9_-]+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(\b(?:authorization|cookie|password|passwd|secret|api.?key|access.?token|"
    r"refresh.?token|session.?token|csrf|xsrf)\b[\"']?\s*[:=]\s*[\"']?)"
    r"([^\"'\s,;&<>]{1,2048})"
)
_CALL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ACTIONS = frozenset({
    "inspect", "read_response", "navigate", "click", "click_text", "type", "fetch", "replay_request",
    "evaluate",
})
_READABLE_RESOURCE_TYPES = frozenset({"Document", "Script", "XHR", "Fetch"})
_RESPONSE_READ_REJECTION = (
    "Response body could not be read safely; choose another completed observed request_id."
)
_CLICK_TEXT_REJECTION = (
    "Visible text did not resolve to exactly one visible link or button; choose a unique target."
)
_STATEFUL_SESSION_REJECTION = (
    "Script evaluation requires a designated session holding no cookies or browser storage for an allowed origin."
)
_MAX_CLICK_TEXT_CHARS = 300
_FETCH_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_FETCH_AUTH_MODES = frozenset({"none", "cookies", "auto_bearer"})
_FORBIDDEN_REQUEST_HEADERS = frozenset({
    "authorization", "cookie", "host", "origin", "referer", "content-length", "connection",
})


def _host_text(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _normalize_origin(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise BrowserPolicyError("Allowed origins must be non-empty strings")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise BrowserPolicyError("Allowed origins must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BrowserPolicyError("Allowed origins cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise BrowserPolicyError("Allowed origins cannot contain a path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BrowserPolicyError("Allowed origin has an invalid port") from exc
    host = parsed.hostname.lower()
    default = 80 if parsed.scheme.lower() == "http" else 443
    suffix = "" if port in {None, default} else f":{port}"
    return f"{parsed.scheme.lower()}://{_host_text(host)}{suffix}"


def _url_origin(value: str, *, allow_blank: bool = False) -> str:
    if allow_blank and value == "about:blank":
        return value
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise BrowserPolicyError("URL must be an absolute http or https URL")
    if parsed.username or parsed.password:
        raise BrowserPolicyError("URLs containing credentials are forbidden")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BrowserPolicyError("URL has an invalid port") from exc
    host = parsed.hostname.lower()
    default = 80 if parsed.scheme.lower() == "http" else 443
    suffix = "" if port in {None, default} else f":{port}"
    return f"{parsed.scheme.lower()}://{_host_text(host)}{suffix}"


def _normalize_endpoint(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise BrowserPolicyError("CDP endpoints must be non-empty strings")
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise BrowserPolicyError("CDP endpoints must use http or https")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise BrowserPolicyError("CDP endpoints cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise BrowserPolicyError("CDP endpoints cannot contain a path")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() != "localhost":
            raise BrowserPolicyError("CDP endpoint must be localhost or a private numeric address")
    else:
        if not (address.is_private or address.is_loopback or address.is_link_local):
            raise BrowserPolicyError("Public CDP endpoints are forbidden")
    try:
        port = parsed.port
    except ValueError as exc:
        raise BrowserPolicyError("CDP endpoint has an invalid port") from exc
    if port is None:
        port = 80 if parsed.scheme.lower() == "http" else 443
    return f"{parsed.scheme.lower()}://{_host_text(parsed.hostname.lower())}:{port}"


def _rewrite_websocket_url(discovered: str, endpoint: str) -> str:
    """Route Chrome's advertised loopback websocket through the configured bridge."""
    source, bridge = urlsplit(discovered), urlsplit(_normalize_endpoint(endpoint))
    if source.scheme not in {"ws", "wss"} or not source.path.startswith("/devtools/"):
        raise BrowserCDPError("Chrome returned an invalid DevTools websocket URL")
    scheme = "wss" if bridge.scheme == "https" else "ws"
    return urlunsplit((scheme, bridge.netloc, source.path, source.query, ""))


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return value
        pairs = [(key, "[REDACTED]" if _SENSITIVE_QUERY_KEYS.search(key) else item)
                 for key, item in parse_qsl(parsed.query, keep_blank_values=True)]
        host = _host_text(parsed.hostname.lower())
        if parsed.port:
            host += f":{parsed.port}"
        return urlunsplit((parsed.scheme.lower(), host, parsed.path, urlencode(pairs), ""))
    except (TypeError, ValueError):
        return "[REDACTED_URL]"


def _redact_text(value: str, limit: int) -> str:
    value = _BEARER.sub("[REDACTED_CREDENTIAL]", value)
    value = _JWT.sub("[REDACTED_TOKEN]", value)
    value = _EMAIL.sub("[REDACTED_EMAIL]", value)
    value = _OPAQUE.sub("[REDACTED_OPAQUE]", value)
    value = _SECRET_ASSIGNMENT.sub(lambda match: match.group(1) + "[REDACTED]", value)
    return value if len(value) <= limit else value[:limit] + "…[TRUNCATED]"


def _utf8_prefix(value: str, byte_limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value, False
    return encoded[:byte_limit].decode("utf-8", errors="ignore"), True


def redact(value: Any, *, string_limit: int = 12_000, depth: int = 0) -> Any:
    """Bound and redact model-visible browser data before it reaches disk."""
    if depth > 8:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        candidate = _redact_url(value) if value.startswith(("http://", "https://")) else value
        return _redact_text(candidate, string_limit)
    if isinstance(value, Mapping):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                result["[TRUNCATED_ITEMS]"] = True
                break
            label = _redact_text(str(key), 200)
            result[label] = "[REDACTED]" if _SENSITIVE_KEYS.search(label) else redact(
                item, string_limit=string_limit, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple, deque)):
        items = list(value)
        bounded = [redact(item, string_limit=string_limit, depth=depth + 1) for item in items[:128]]
        if len(items) > 128:
            bounded.append("[TRUNCATED_ITEMS]")
        return bounded
    return _redact_text(str(value), string_limit)


@dataclass(frozen=True)
class BrowserPolicy:
    """Explicit capability and volume limits for two browser sessions."""

    allowed_origins: frozenset[str]
    interaction_enabled: bool = False
    request_replay_enabled: bool = False
    max_actions_per_session: int = 24
    max_replays_per_session: int = 2
    max_text_chars: int = 2_000
    max_selector_chars: int = 500
    max_observation_chars: int = 65_536
    max_response_body_bytes: int = 65_536
    max_script_chars: int = 4_000
    script_eval_sessions: frozenset[str] = frozenset()
    command_timeout: float = 10.0
    navigation_timeout: float = 15.0

    def __post_init__(self):
        if isinstance(self.allowed_origins, (str, bytes)):
            raise BrowserPolicyError("allowed_origins must be a collection of exact origins")
        origins = frozenset(_normalize_origin(item) for item in self.allowed_origins)
        if not origins:
            raise BrowserPolicyError("At least one exact allowed origin is required")
        object.__setattr__(self, "allowed_origins", origins)
        for name, lower, upper in (
            ("max_actions_per_session", 1, 100),
            ("max_replays_per_session", 0, 10),
            ("max_text_chars", 1, 10_000),
            ("max_selector_chars", 1, 2_000),
            ("max_observation_chars", 500, 100_000),
            ("max_response_body_bytes", 1, 1_048_576),
            ("max_script_chars", 1, 20_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise BrowserPolicyError(f"{name} must be an integer in [{lower}, {upper}]")
        if isinstance(self.script_eval_sessions, (str, bytes)):
            raise BrowserPolicyError("script_eval_sessions must be a collection of session names")
        if any(not isinstance(name, str) or not 1 <= len(name) <= 64
               for name in self.script_eval_sessions):
            raise BrowserPolicyError("script_eval_sessions entries must be bounded session names")
        object.__setattr__(self, "script_eval_sessions", frozenset(self.script_eval_sessions))
        for name in ("command_timeout", "navigation_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.1 <= value <= 120:
                raise BrowserPolicyError(f"{name} must be in [0.1, 120] seconds")

    def allows(self, url: str, *, blank: bool = False) -> bool:
        try:
            origin = _url_origin(url, allow_blank=blank)
        except BrowserPolicyError:
            return False
        return (blank and origin == "about:blank") or origin in self.allowed_origins

    def require_url(self, url: str) -> str:
        if not isinstance(url, str) or len(url) > 8_000:
            raise BrowserPolicyError("URL exceeds its bound")
        if _url_origin(url) not in self.allowed_origins:
            raise BrowserPolicyError("URL origin is outside the exact allowlist")
        return url


EventHandler = Callable[[Mapping[str, Any]], Awaitable[None] | None]


class WebSocketCDPTransport:
    """Small correlated JSON CDP client over a direct Chrome websocket."""

    MAX_DISCOVERY_BYTES = 128 * 1024
    MAX_MESSAGE_BYTES = 16 * 1024 * 1024

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, timeout: float):
        self.reader, self.writer, self.timeout = reader, writer, timeout
        self._sequence = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._handlers: dict[str, list[EventHandler]] = {}
        self._send_lock = asyncio.Lock()
        self._handler_tasks: set[asyncio.Task] = set()
        self._closed = False
        self._reader_task = asyncio.create_task(self._read_messages())

    @classmethod
    async def connect(cls, endpoint: str, *, timeout: float = 10.0):
        endpoint = _normalize_endpoint(endpoint)
        discovered = await asyncio.wait_for(
            asyncio.to_thread(cls._discover, endpoint, timeout), timeout + 1)
        websocket = _rewrite_websocket_url(discovered, endpoint)
        parsed = urlsplit(websocket)
        ssl_context = ssl.create_default_context() if parsed.scheme == "wss" else None
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(parsed.hostname, port, ssl=ssl_context), timeout)
        try:
            await asyncio.wait_for(cls._handshake(reader, writer, parsed), timeout)
            return cls(reader, writer, timeout)
        except BaseException:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
            raise

    @classmethod
    def _discover(cls, endpoint: str, timeout: float) -> str:
        parsed = urlsplit(endpoint)
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
        try:
            connection.request("GET", "/json/version", headers={"Accept": "application/json"})
            response = connection.getresponse()
            body = response.read(cls.MAX_DISCOVERY_BYTES + 1)
            if response.status != 200:
                raise BrowserCDPError(f"CDP discovery returned HTTP {response.status}")
            if len(body) > cls.MAX_DISCOVERY_BYTES:
                raise BrowserCDPError("CDP discovery response exceeded its bound")
            value = json.loads(body)
            websocket = value.get("webSocketDebuggerUrl")
            if not isinstance(websocket, str):
                raise BrowserCDPError("CDP discovery omitted webSocketDebuggerUrl")
            return websocket
        except BrowserCDPError:
            raise
        except Exception as exc:
            raise BrowserCDPError(f"CDP discovery failed: {exc}") from exc
        finally:
            connection.close()

    @staticmethod
    async def _handshake(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, parsed):
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        default = 443 if parsed.scheme == "wss" else 80
        host = _host_text(parsed.hostname)
        if parsed.port and parsed.port != default:
            host += f":{parsed.port}"
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        writer.write(request.encode("ascii"))
        await writer.drain()
        raw = await reader.readuntil(b"\r\n\r\n")
        if len(raw) > 64 * 1024:
            raise BrowserCDPError("Websocket handshake response exceeded its bound")
        lines = raw.decode("latin-1").split("\r\n")
        if not lines or " 101 " not in f" {lines[0]} ":
            raise BrowserCDPError(f"Websocket upgrade failed: {lines[0] if lines else 'empty response'}")
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key_name, value = line.split(":", 1)
                headers[key_name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        if headers.get("sec-websocket-accept") != expected:
            raise BrowserCDPError("Websocket upgrade returned an invalid accept key")

    def add_event_handler(self, method: str, handler: EventHandler):
        self._handlers.setdefault(method, []).append(handler)

    async def command(self, method: str, params: Mapping[str, Any] | None = None,
                      *, session_id: str | None = None) -> Mapping[str, Any]:
        if self._closed:
            raise BrowserCDPError("CDP transport is closed")
        self._sequence += 1
        request_id = self._sequence
        message: dict[str, Any] = {"id": request_id, "method": method, "params": dict(params or {})}
        if session_id:
            message["sessionId"] = session_id
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send_frame(0x1, canonical(message).encode("utf-8"))
            response = await asyncio.wait_for(asyncio.shield(future), self.timeout)
        except asyncio.TimeoutError as exc:
            raise BrowserCDPError(f"CDP command timed out: {method}") from exc
        finally:
            self._pending.pop(request_id, None)
        if "error" in response:
            error = response["error"]
            raise BrowserCDPError(f"CDP {method} failed: {error.get('message', 'unknown error')}")
        result = response.get("result", {})
        if not isinstance(result, Mapping):
            raise BrowserCDPError(f"CDP {method} returned a malformed result")
        return result

    async def _send_frame(self, opcode: int, payload: bytes):
        if len(payload) > self.MAX_MESSAGE_BYTES:
            raise BrowserCDPError("Outgoing websocket message exceeded its bound")
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = bytes((first, 0x80 | length))
        elif length <= 0xFFFF:
            header = bytes((first, 0x80 | 126)) + struct.pack("!H", length)
        else:
            header = bytes((first, 0x80 | 127)) + struct.pack("!Q", length)
        mask = secrets.token_bytes(4)
        masked = bytes(item ^ mask[index % 4] for index, item in enumerate(payload))
        async with self._send_lock:
            self.writer.write(header + mask + masked)
            await self.writer.drain()

    async def _read_frame(self) -> tuple[bool, int, bytes]:
        first, second = await self.reader.readexactly(2)
        final, opcode, masked = bool(first & 0x80), first & 0x0F, bool(second & 0x80)
        if masked:
            raise BrowserCDPError("Chrome sent a masked server frame")
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", await self.reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await self.reader.readexactly(8))[0]
        if length > self.MAX_MESSAGE_BYTES:
            raise BrowserCDPError("Incoming websocket message exceeded its bound")
        return final, opcode, await self.reader.readexactly(length)

    def _schedule_handler(self, handler: EventHandler, params: Mapping[str, Any]):
        async def invoke():
            result = handler(params)
            if python_inspect.isawaitable(result):
                await result
        task = asyncio.create_task(invoke())
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    async def _read_messages(self):
        fragments = bytearray()
        fragmented_opcode: int | None = None
        failure = "CDP websocket closed"
        try:
            while True:
                final, opcode, payload = await self._read_frame()
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    await self._send_frame(0xA, payload)
                    continue
                if opcode == 0xA:
                    continue
                if opcode in {0x1, 0x2}:
                    if fragmented_opcode is not None:
                        raise BrowserCDPError("Nested fragmented websocket message")
                    if final:
                        complete = payload
                    else:
                        fragments = bytearray(payload)
                        fragmented_opcode = opcode
                        continue
                elif opcode == 0x0 and fragmented_opcode is not None:
                    fragments.extend(payload)
                    if len(fragments) > self.MAX_MESSAGE_BYTES:
                        raise BrowserCDPError("Fragmented websocket message exceeded its bound")
                    if not final:
                        continue
                    complete = bytes(fragments)
                    opcode, fragmented_opcode, fragments = fragmented_opcode, None, bytearray()
                else:
                    raise BrowserCDPError("Unsupported websocket frame")
                if opcode != 0x1:
                    continue
                message = json.loads(complete.decode("utf-8"))
                if message.get("id") in self._pending:
                    future = self._pending[message["id"]]
                    if not future.done():
                        future.set_result(message)
                elif isinstance(message.get("method"), str):
                    params = message.get("params", {})
                    if isinstance(params, Mapping):
                        for handler in self._handlers.get(message["method"], ()):
                            self._schedule_handler(handler, params)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            failure = f"CDP websocket failed: {exc}"
        finally:
            self._closed = True
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(BrowserCDPError(failure))

    async def close(self):
        if not self._closed:
            with suppress(Exception):
                await self._send_frame(0x8, b"")
        self._closed = True
        self.writer.close()
        with suppress(Exception):
            await self.writer.wait_closed()
        self._reader_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._reader_task
        tasks = list(self._handler_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


TransportFactory = Callable[[str, float], Awaitable[Any]]


@dataclass
class _Session:
    name: str
    endpoint: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transport: Any = None
    target_id: str | None = None
    cdp_session_id: str | None = None
    actions_started: int = 0
    replays_started: int = 0
    sequence: int = 0
    requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    logs: deque = field(default_factory=lambda: deque(maxlen=32))
    handler_tasks: set[asyncio.Task] = field(default_factory=set)


_INSPECT_EXPRESSION = r"""(() => {
  const compact = (value, limit) => String(value || '').replace(/\s+/g, ' ').trim().slice(0, limit);
  const visible = (node) => {
    const style = getComputedStyle(node), box = node.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && box.width > 0 && box.height > 0;
  };
  const take = (selector, limit, map) => Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, limit).map(map);
  return {
    url: location.href,
    title: compact(document.title, 500),
    headings: take('h1,h2,h3', 40, n => ({level:n.tagName.toLowerCase(), text:compact(n.innerText, 500)})),
    links: take('a[href]', 60, n => ({text:compact(n.innerText || n.getAttribute('aria-label'), 300), href:n.href})),
    buttons: take('button,[role="button"]', 60, n => ({text:compact(n.innerText || n.getAttribute('aria-label'), 300), disabled:!!n.disabled})),
    controls: take('input,textarea,select', 60, n => ({tag:n.tagName.toLowerCase(), type:(n.type || ''), name:compact(n.name, 200), label:compact(n.getAttribute('aria-label'), 300), disabled:!!n.disabled})),
    visible_text: compact(document.body && document.body.innerText, 12000)
  };
})()"""

_PAGE_STATE_EXPRESSION = "({url: location.href, readyState: document.readyState})"

# This fixed function is evaluated without model-supplied source.  The target is
# passed separately to Runtime.callFunctionOn, so it cannot alter the program.
# Primitive sentinel values keep zero/ambiguous matches distinguishable from a
# returned DOM node without exposing page text through the adapter.
_CLICK_TEXT_RESOLVER_EXPRESSION = r"""((target) => {
  const compact = (value) => String(value || '').replace(/\s+/g, ' ').trim();
  const nodes = document.querySelectorAll('a,button,[role="button"]');
  if (nodes.length > 1000) return -1;
  let match = null;
  for (const node of nodes) {
    const style = getComputedStyle(node), box = node.getBoundingClientRect();
    const visible = style.visibility !== 'hidden' && style.display !== 'none' &&
      style.opacity !== '0' && box.width > 0 && box.height > 0;
    if (!visible) continue;
    const inner = compact(node.innerText);
    const aria = compact(node.getAttribute('aria-label'));
    if (inner !== target && aria !== target) continue;
    if (match !== null) return -1;
    match = node;
  }
  return match === null ? 0 : match;
})"""
_CLICK_TEXT_CALL_FUNCTION = "function(target) { return this(target); }"

# Model input is passed as Runtime.callFunctionOn values, never interpolated into
# source.  Tokens remain inside the isolated page: auto_bearer may select a JWT
# from session storage, local storage, or a readable cookie, but never returns it.
_FETCH_FUNCTION_EXPRESSION = r"""(async (url, method, headers, body, authMode, bodyLimit) => {
  const options = {
    method,
    headers: Object.assign({}, headers),
    credentials: authMode === 'none' ? 'omit' : 'include',
    redirect: 'follow',
    cache: 'no-store'
  };
  let authApplied = false;
  if (authMode === 'auto_bearer') {
    const jwt = /eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}/;
    const candidates = [];
    const collect = (key, value) => {
      const text = String(value || '');
      const match = text.match(jwt);
      if (match) candidates.push({key: String(key || ''), token: match[0]});
    };
    for (const store of [sessionStorage, localStorage]) {
      for (let i = 0; i < Math.min(store.length, 256); i++) {
        const key = store.key(i);
        collect(key, store.getItem(key));
      }
    }
    for (const pair of String(document.cookie || '').split(';')) {
      const cut = pair.indexOf('=');
      if (cut > 0) collect(pair.slice(0, cut).trim(), decodeURIComponent(pair.slice(cut + 1).trim()));
    }
    candidates.sort((a, b) => {
      const rank = (item) => /id.?token/i.test(item.key) ? 0 : /access.?token/i.test(item.key) ? 1 : 2;
      return rank(a) - rank(b);
    });
    if (candidates.length) {
      options.headers.Authorization = 'Bearer ' + candidates[0].token;
      authApplied = true;
    }
  }
  if (!['GET', 'HEAD'].includes(method) && body !== '') options.body = body;
  const response = await fetch(url, options);
  const text = method === 'HEAD' ? '' : await response.text();
  const responseHeaders = {};
  let count = 0;
  for (const [key, value] of response.headers.entries()) {
    if (count++ >= 64) break;
    responseHeaders[key] = value;
  }
  return {
    url: response.url,
    status: response.status,
    status_text: response.statusText,
    redirected: response.redirected,
    response_type: response.type,
    headers: responseHeaders,
    body: text.slice(0, bodyLimit),
    total_body_characters: text.length,
    truncated_in_page: text.length > bodyLimit,
    auth_mode: authMode,
    bearer_applied: authApplied
  };
})"""
_FETCH_CALL_FUNCTION = "function(url, method, headers, body, authMode, bodyLimit) { return this(url, method, headers, body, authMode, bodyLimit); }"


class BrowserCDPAdapter:
    """Two-session, durable, policy-limited facade over Chrome CDP."""

    JOURNAL_SCHEMA_VERSION = 1
    OBSERVATION_SCHEMA_VERSION = 1

    def __init__(self, endpoints: Mapping[str, str], artifact_dir: str | os.PathLike,
                 policy: BrowserPolicy, *, transport_factory: TransportFactory | None = None):
        if not isinstance(endpoints, Mapping) or not 2 <= len(endpoints) <= 3:
            raise BrowserPolicyError("Two or three isolated browser sessions are required")
        normalized = {str(name): _normalize_endpoint(endpoint) for name, endpoint in endpoints.items()}
        if any(not name or len(name) > 64 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
               for name in normalized):
            raise BrowserPolicyError("Browser session names are invalid")
        if policy.script_eval_sessions - set(normalized):
            raise BrowserPolicyError("Script evaluation was designated for an unknown browser session")
        if len(set(normalized.values())) != len(normalized):
            raise BrowserPolicyError("Each browser session must use a unique CDP endpoint")
        self.policy = policy
        self.artifact_dir = Path(artifact_dir)
        self.observation_dir = self.artifact_dir / "observations"
        self.journal_path = self.artifact_dir / "browser-actions.json"
        self.sessions = {name: _Session(name, endpoint) for name, endpoint in normalized.items()}
        self._factory = transport_factory or self._default_factory
        self._start_lock = asyncio.Lock()
        self._journal_lock = asyncio.Lock()
        self._started = False
        self._journal = self._load_journal()
        self._restore_counters()

    async def _default_factory(self, endpoint: str, timeout: float):
        return await WebSocketCDPTransport.connect(endpoint, timeout=timeout)

    def _load_journal(self) -> dict[str, Any]:
        if not self.journal_path.exists():
            return {"schema_version": self.JOURNAL_SCHEMA_VERSION, "records": {}}
        try:
            value = json.loads(self.journal_path.read_text())
        except Exception as exc:
            raise BrowserRecoveryRequired(f"Cannot read browser action journal: {exc}") from exc
        if value.get("schema_version") != self.JOURNAL_SCHEMA_VERSION or not isinstance(value.get("records"), dict):
            raise BrowserRecoveryRequired("Browser action journal has an unsupported schema")
        normalized = False
        for record in value["records"].values():
            if not isinstance(record, dict) or record.get("status") not in {
                    "started", "completed", "rejected", "uncertain"}:
                raise BrowserRecoveryRequired("Browser action journal is malformed")
            if (record.get("action") == "read_response"
                    and (record.get("status") in {"started", "uncertain"}
                         or (record.get("status") == "rejected"
                             and record.get("reason") == _RESPONSE_READ_REJECTION
                             and record.get("error_kind") != "read_unavailable"))):
                record.update(status="rejected", reason=_RESPONSE_READ_REJECTION,
                              error_kind="read_unavailable", updated_at=now())
                record.pop("error_type", None)
                normalized = True
        if normalized:
            atomic_json(self.journal_path, value)
        return value

    def _restore_counters(self):
        for record in self._journal["records"].values():
            session = self.sessions.get(record.get("session"))
            if session is None:
                continue
            session.actions_started += 1
            if record.get("action") == "replay_request":
                session.replays_started += 1
            result = record.get("result", {})
            sequence = result.get("sequence", 0) if isinstance(result, Mapping) else 0
            if type(sequence) is int:
                session.sequence = max(session.sequence, sequence)

    async def start(self):
        async with self._start_lock:
            if self._started:
                return self
            connected = []
            try:
                for session in self.sessions.values():
                    session.transport = await self._factory(session.endpoint, self.policy.command_timeout)
                    connected.append(session)
                    await self._attach(session)
                self._started = True
                return self
            except BaseException:
                for session in reversed(connected):
                    with suppress(Exception):
                        await session.transport.close()
                    session.transport = None
                raise

    async def _attach(self, session: _Session):
        targets = await session.transport.command("Target.getTargets")
        candidates = []
        for target in targets.get("targetInfos", []):
            if not isinstance(target, Mapping) or target.get("type") != "page":
                continue
            url = target.get("url", "")
            if url == "about:blank" or self.policy.allows(url):
                candidates.append(target)
        target = next((item for item in candidates if self.policy.allows(item.get("url", ""))), None)
        target = target or (candidates[0] if candidates else None)
        if target is None:
            target_id = (await session.transport.command("Target.createTarget", {"url": "about:blank"})).get("targetId")
        else:
            target_id = target.get("targetId")
        if not isinstance(target_id, str) or not target_id:
            raise BrowserCDPError("Chrome did not provide a usable page target")
        attached = await session.transport.command("Target.attachToTarget", {"targetId": target_id, "flatten": True})
        session_id = attached.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise BrowserCDPError("Chrome did not provide a target session id")
        session.target_id, session.cdp_session_id = target_id, session_id
        session.transport.add_event_handler("Network.requestWillBeSent", lambda params: self._network_request(session, params))
        session.transport.add_event_handler("Network.responseReceived", lambda params: self._network_response(session, params))
        session.transport.add_event_handler("Network.loadingFinished", lambda params: self._network_finished(session, params))
        session.transport.add_event_handler("Network.loadingFailed", lambda params: self._network_failed(session, params))
        session.transport.add_event_handler("Runtime.consoleAPICalled", lambda params: self._console_event(session, params))
        session.transport.add_event_handler("Log.entryAdded", lambda params: self._log_event(session, params))
        session.transport.add_event_handler("Fetch.requestPaused", lambda params: self._fetch_paused(session, params))
        enable_commands = [
            ("Page.enable", {}), ("DOM.enable", {}), ("Runtime.enable", {}),
            ("Network.enable", {"maxTotalBufferSize": 1_000_000, "maxResourceBufferSize": 200_000}),
            ("Log.enable", {}),
        ]
        if self.policy.interaction_enabled:
            enable_commands.append((
                "Fetch.enable",
                {"patterns": [{"urlPattern": "*", "resourceType": "Document", "requestStage": "Request"}]},
            ))
        for method, params in enable_commands:
            await session.transport.command(method, params, session_id=session_id)

    async def preflight(self) -> Mapping[str, Any]:
        """Connect and attach without reading DOM, page text, cookies, or storage."""
        await self.start()
        return {"schema_version": 1, "sessions": [
            {"session": item.name, "endpoint": item.endpoint, "connected": bool(item.cdp_session_id)}
            for item in self.sessions.values()
        ]}

    def _event_for_session(self, session: _Session, params: Mapping[str, Any]) -> bool:
        event_session = params.get("sessionId")
        return event_session in {None, session.cdp_session_id}

    def _network_request(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        request, request_id = params.get("request", {}), params.get("requestId")
        if not isinstance(request, Mapping) or not isinstance(request_id, str):
            return
        url, method = request.get("url", ""), str(request.get("method", "")).upper()
        resource_type = str(params.get("type", ""))
        origin = None
        with suppress(BrowserPolicyError):
            origin = _url_origin(url)
        session.requests[request_id] = {
            "request_id": request_id,
            "url": _redact_url(url),
            "origin": origin,
            "method": method,
            "resource_type": resource_type,
            "eligible": origin in self.policy.allowed_origins and method in {"GET", "HEAD"} and resource_type == "XHR",
            "readable": False,
            "response_seen": False,
            "finished": False,
        }
        while len(session.requests) > 256:
            session.requests.pop(next(iter(session.requests)))

    def _network_response(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        request_id, response = params.get("requestId"), params.get("response", {})
        if isinstance(request_id, str) and request_id in session.requests and isinstance(response, Mapping):
            request = session.requests[request_id]
            response_url = response.get("url", request.get("url", ""))
            response_origin = None
            with suppress(BrowserPolicyError):
                response_origin = _url_origin(response_url)
            resource_type = str(params.get("type") or request.get("resource_type", ""))
            same_allowed_scope = (request.get("origin") in self.policy.allowed_origins
                                  and response_origin in self.policy.allowed_origins)
            request.update({
                "status": response.get("status"),
                "mime_type": _redact_text(str(response.get("mimeType", "")), 200),
                "response_origin": response_origin,
                "resource_type": resource_type,
                "response_seen": True,
                "readable": same_allowed_scope and resource_type in _READABLE_RESOURCE_TYPES,
            })
            if not same_allowed_scope:
                request["eligible"] = False

    def _network_finished(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        request_id = params.get("requestId")
        if isinstance(request_id, str) and request_id in session.requests:
            encoded = params.get("encodedDataLength")
            session.requests[request_id].update({
                "finished": True,
                "encoded_data_length": encoded if isinstance(encoded, (int, float)) and encoded >= 0 else None,
            })

    def _network_failed(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        request_id = params.get("requestId")
        if isinstance(request_id, str) and request_id in session.requests:
            session.requests[request_id].update({"finished": False, "failed": True, "readable": False})

    def _console_event(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        values = []
        for arg in params.get("args", [])[:10]:
            if isinstance(arg, Mapping):
                values.append(arg.get("value", arg.get("description", "")))
        session.logs.append(redact({"source": "console", "level": params.get("type"), "values": values},
                                   string_limit=1_000))

    def _log_event(self, session: _Session, params: Mapping[str, Any]):
        if not self._event_for_session(session, params):
            return
        entry = params.get("entry", {})
        if isinstance(entry, Mapping):
            session.logs.append(redact({"source": entry.get("source"), "level": entry.get("level"),
                                        "text": entry.get("text"), "url": entry.get("url")},
                                       string_limit=1_000))

    async def _fetch_paused(self, session: _Session, params: Mapping[str, Any]):
        request_id, request = params.get("requestId"), params.get("request", {})
        if not isinstance(request_id, str) or not isinstance(request, Mapping):
            return
        url = request.get("url", "")
        method = "Fetch.continueRequest" if self.policy.allows(url, blank=True) else "Fetch.failRequest"
        arguments = {"requestId": request_id}
        if method == "Fetch.failRequest":
            arguments["errorReason"] = "BlockedByClient"
        with suppress(Exception):
            await session.transport.command(method, arguments, session_id=session.cdp_session_id)

    def _session(self, name: str) -> _Session:
        try:
            return self.sessions[name]
        except KeyError as exc:
            raise BrowserPolicyError("Unknown browser session") from exc

    def budget_status(self, name: str) -> Mapping[str, Any]:
        """Return bounded counters for attempt-specific worker guidance."""
        session = self._session(name)
        return {
            "actions_used": session.actions_started,
            "actions_remaining": max(
                0, self.policy.max_actions_per_session - session.actions_started),
            "replays_used": session.replays_started,
            "replays_remaining": max(
                0, self.policy.max_replays_per_session - session.replays_started),
            "currently_eligible_replay_requests": sum(
                request.get("eligible") is True for request in session.requests.values()),
            "replay_consumes_action": True,
            "navigate_and_click_return_snapshots": True,
        }

    def _selector(self, value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > self.policy.max_selector_chars:
            raise BrowserPolicyError("CSS selector is empty or exceeds its bound")
        if any(ord(character) < 32 for character in value):
            raise BrowserPolicyError("CSS selector contains control characters")
        return value

    @staticmethod
    def _click_text(value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > _MAX_CLICK_TEXT_CHARS:
            raise BrowserPolicyError("Visible-text target is empty or exceeds its bound")
        if any(ord(character) < 32 and not character.isspace() for character in value):
            raise BrowserPolicyError("Visible-text target contains control characters")
        normalized = " ".join(value.split())
        if not normalized or len(normalized) > _MAX_CLICK_TEXT_CHARS:
            raise BrowserPolicyError("Visible-text target is empty or exceeds its bound")
        return normalized

    def _prepare(self, action: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if action not in _ACTIONS or not isinstance(arguments, Mapping):
            raise BrowserPolicyError("Unknown browser action")
        allowed_keys = {
            "inspect": set(), "read_response": {"request_id"},
            "navigate": {"url"}, "click": {"selector"},
            "click_text": {"text"}, "type": {"selector", "text"},
            "fetch": {"url", "method", "headers", "body", "auth_mode"},
            "replay_request": {"request_id"},
            "evaluate": {"code"},
        }[action]
        if set(arguments) != allowed_keys:
            raise BrowserPolicyError(f"{action} requires exactly: {', '.join(sorted(allowed_keys)) or 'no arguments'}")
        if action == "evaluate":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Script evaluation is disabled by policy")
            code = arguments["code"]
            if not isinstance(code, str) or not code or len(code) > self.policy.max_script_chars:
                raise BrowserPolicyError("Script source is empty or exceeds its bound")
            if "\x00" in code:
                raise BrowserPolicyError("Script source contains a NUL byte")
            return {"code": code}
        if action == "navigate":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Navigation is disabled by policy")
            return {"url": self.policy.require_url(arguments["url"])}
        if action == "click":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Click is disabled by policy")
            return {"selector": self._selector(arguments["selector"])}
        if action == "click_text":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Click is disabled by policy")
            return {"text": self._click_text(arguments["text"])}
        if action == "type":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Typing is disabled by policy")
            text = arguments["text"]
            if not isinstance(text, str) or not text or len(text) > self.policy.max_text_chars:
                raise BrowserPolicyError("Typed text is empty or exceeds its bound")
            if "\x00" in text:
                raise BrowserPolicyError("Typed text contains a NUL byte")
            return {"selector": self._selector(arguments["selector"]), "text": text}
        if action == "fetch":
            if not self.policy.interaction_enabled:
                raise BrowserPolicyError("Authenticated fetch is disabled by policy")
            method = arguments["method"]
            if not isinstance(method, str) or method.upper() not in _FETCH_METHODS:
                raise BrowserPolicyError("Fetch method is unsupported")
            method = method.upper()
            auth_mode = arguments["auth_mode"]
            if auth_mode not in _FETCH_AUTH_MODES:
                raise BrowserPolicyError("Fetch auth_mode is unsupported")
            headers = arguments["headers"]
            if not isinstance(headers, Mapping) or len(headers) > 24:
                raise BrowserPolicyError("Fetch headers must be an object with at most 24 fields")
            clean_headers = {}
            for key, value in headers.items():
                if (not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,100}", key)
                        or key.lower() in _FORBIDDEN_REQUEST_HEADERS):
                    raise BrowserPolicyError("Fetch contains a forbidden or invalid header name")
                if (not isinstance(value, str) or len(value) > 2_000
                        or any(ord(character) < 32 and character != "\t" for character in value)):
                    raise BrowserPolicyError("Fetch contains an invalid header value")
                clean_headers[key] = value
            body = arguments["body"]
            if not isinstance(body, str) or len(body) > self.policy.max_text_chars or "\x00" in body:
                raise BrowserPolicyError("Fetch body exceeds its bound or contains a NUL byte")
            if method in {"GET", "HEAD"} and body:
                raise BrowserPolicyError("GET and HEAD fetches require an empty body")
            return {"url": self.policy.require_url(arguments["url"]), "method": method,
                    "headers": clean_headers, "body": body, "auth_mode": auth_mode}
        if action in {"read_response", "replay_request"}:
            if action == "replay_request" and not self.policy.request_replay_enabled:
                raise BrowserPolicyError("XHR replay is disabled by policy")
            request_id = arguments["request_id"]
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 200:
                raise BrowserPolicyError("Request id is invalid")
            return {"request_id": request_id}
        return {}

    async def execute(self, session_name: str, action: str, arguments: Mapping[str, Any],
                      *, call_id: str) -> Mapping[str, Any]:
        if not isinstance(call_id, str) or not _CALL_ID.fullmatch(call_id):
            raise BrowserPolicyError("call_id must be a stable bounded identifier")
        session = self._session(session_name)
        if action == "evaluate" and session_name not in self.policy.script_eval_sessions:
            raise BrowserPolicyError("Script evaluation is not enabled for this session")
        prepared = self._prepare(action, arguments)
        digest_arguments = dict(prepared)
        if "text" in digest_arguments:
            digest_arguments["text"] = {"sha256": hashlib.sha256(
                digest_arguments["text"].encode("utf-8")).hexdigest(), "length": len(digest_arguments["text"])}
        intent_digest = hashlib.sha256(canonical({"session": session_name, "action": action,
                                                   "arguments": digest_arguments}).encode()).hexdigest()
        await self.start()
        async with session.lock:
            # Keep the state gate and model-authored execution in the same
            # per-session critical section. Another worker action must not be
            # able to establish a login between the gate and evaluation.
            if action == "evaluate":
                await self._require_stateless_session(session)
            async with self._journal_lock:
                existing = self._journal["records"].get(call_id)
                if existing:
                    if existing.get("intent_digest") != intent_digest:
                        raise BrowserPolicyError("call_id was already used for a different browser intent")
                    if existing.get("status") == "completed":
                        return dict(existing["result"])
                    if existing.get("status") == "rejected":
                        if (existing.get("action") == "read_response"
                                and existing.get("error_kind") == "read_unavailable"):
                            raise BrowserReadUnavailable(
                                existing.get("reason", _RESPONSE_READ_REJECTION))
                        raise BrowserPolicyError(existing.get("reason", "Browser response read was rejected"))
                    raise BrowserRecoveryRequired("Browser action outcome is uncertain; it was not retried")
                if action == "read_response":
                    request = session.requests.get(prepared["request_id"])
                    if not request or not request.get("readable"):
                        raise BrowserPolicyError(
                            "Only observed exact-allowlisted Document/Script/XHR/Fetch responses can be read")
                    if not request.get("finished") or request.get("failed"):
                        raise BrowserPolicyError("Observed response has not completed successfully")
                if action == "replay_request":
                    request = session.requests.get(prepared["request_id"])
                    if not request or not request.get("eligible"):
                        raise BrowserPolicyError(
                            "Only observed same-origin GET/HEAD XHR requests can be replayed")
                if session.actions_started >= self.policy.max_actions_per_session:
                    raise BrowserPolicyError("Browser session action budget is exhausted")
                if action == "replay_request" and session.replays_started >= self.policy.max_replays_per_session:
                    raise BrowserPolicyError("Browser session replay budget is exhausted")
                session.actions_started += 1
                if action == "replay_request":
                    session.replays_started += 1
                record = {"session": session_name, "action": action, "intent_digest": intent_digest,
                          "status": "started", "started_at": now()}
                self._journal["records"][call_id] = record
                atomic_json(self.journal_path, self._journal)
            try:
                data = await self._perform(session, action, prepared)
                session.sequence += 1
                result = self._write_observation(session, action, data)
            except BaseException as exc:
                safe_text_rejection = (
                    action == "click_text" and isinstance(exc, _BrowserTextTargetError)
                )
                async with self._journal_lock:
                    record["status"] = (
                        "rejected" if action == "read_response" or safe_text_rejection else "uncertain"
                    )
                    record["updated_at"] = now()
                    if action == "read_response":
                        record["reason"] = _RESPONSE_READ_REJECTION
                        record["error_kind"] = "read_unavailable"
                        record.pop("error_type", None)
                    elif safe_text_rejection:
                        record["reason"] = _CLICK_TEXT_REJECTION
                        record["error_kind"] = "target_not_unique"
                        record.pop("error_type", None)
                    else:
                        record["error_type"] = type(exc).__name__
                    atomic_json(self.journal_path, self._journal)
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
                if action == "read_response":
                    raise BrowserReadUnavailable(_RESPONSE_READ_REJECTION) from None
                if safe_text_rejection:
                    raise BrowserPolicyError(_CLICK_TEXT_REJECTION) from None
                raise BrowserRecoveryRequired(
                    "Browser action may have run; its durable call id was marked uncertain") from exc
            async with self._journal_lock:
                record["status"] = "completed"
                record["completed_at"] = now()
                record["result"] = result
                atomic_json(self.journal_path, self._journal)
            return dict(result)

    async def inspect(self, session: str, *, call_id: str):
        return await self.execute(session, "inspect", {}, call_id=call_id)

    async def navigate(self, session: str, url: str, *, call_id: str):
        return await self.execute(session, "navigate", {"url": url}, call_id=call_id)

    async def read_response(self, session: str, request_id: str, *, call_id: str):
        return await self.execute(session, "read_response", {"request_id": request_id}, call_id=call_id)

    async def click(self, session: str, selector: str, *, call_id: str):
        return await self.execute(session, "click", {"selector": selector}, call_id=call_id)

    async def click_text(self, session: str, text: str, *, call_id: str):
        return await self.execute(session, "click_text", {"text": text}, call_id=call_id)

    async def type_text(self, session: str, selector: str, text: str, *, call_id: str):
        return await self.execute(session, "type", {"selector": selector, "text": text}, call_id=call_id)

    async def fetch(self, session: str, url: str, method: str, headers: Mapping[str, str],
                    body: str, auth_mode: str, *, call_id: str):
        return await self.execute(session, "fetch", {
            "url": url, "method": method, "headers": headers, "body": body, "auth_mode": auth_mode,
        }, call_id=call_id)

    async def replay_request(self, session: str, request_id: str, *, call_id: str):
        return await self.execute(session, "replay_request", {"request_id": request_id}, call_id=call_id)

    async def _command(self, session: _Session, method: str, params: Mapping[str, Any] | None = None):
        return await session.transport.command(method, params or {}, session_id=session.cdp_session_id)

    async def _evaluate(self, session: _Session, expression: str) -> Any:
        response = await self._command(session, "Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": False,
            "userGesture": False, "includeCommandLineAPI": False,
        })
        if response.get("exceptionDetails"):
            raise BrowserCDPError("Fixed browser inspection expression failed")
        result = response.get("result", {})
        if not isinstance(result, Mapping) or "value" not in result:
            raise BrowserCDPError("Fixed browser inspection returned no value")
        return result["value"]

    async def _holds_origin_state(self, session: _Session) -> bool:
        """Fail closed on cookies or browser storage for an allowed origin."""
        try:
            page = await self._evaluate(session, _PAGE_STATE_EXPRESSION)
            if not isinstance(page, Mapping) or not self.policy.allows(page.get("url", "")):
                return True
            response = await self._command(
                session, "Network.getCookies", {"urls": sorted(self.policy.allowed_origins)})
            cookies = response.get("cookies")
            if not isinstance(cookies, list) or cookies:
                return True
            for origin in sorted(self.policy.allowed_origins):
                for is_local in (True, False):
                    response = await self._command(session, "DOMStorage.getDOMStorageItems", {
                        "storageId": {"securityOrigin": origin, "isLocalStorage": is_local},
                    })
                    entries = response.get("entries")
                    if not isinstance(entries, list) or entries:
                        return True
                response = await self._command(
                    session, "IndexedDB.requestDatabaseNames", {"securityOrigin": origin})
                names = response.get("databaseNames")
                if not isinstance(names, list) or names:
                    return True
                response = await self._command(
                    session, "CacheStorage.requestCacheNames", {"securityOrigin": origin})
                caches = response.get("caches")
                if not isinstance(caches, list) or caches:
                    return True
                response = await self._command(
                    session, "Storage.getUsageAndQuota", {"origin": origin})
                usage = response.get("usage")
                breakdown = response.get("usageBreakdown")
                if (isinstance(usage, bool) or not isinstance(usage, (int, float)) or usage < 0
                        or not isinstance(breakdown, list)):
                    return True
                for item in breakdown:
                    if (not isinstance(item, Mapping)
                            or not isinstance(item.get("storageType"), str)
                            or isinstance(item.get("usage"), bool)
                            or not isinstance(item.get("usage"), (int, float))
                            or item["usage"] < 0):
                        return True
                    if item["usage"] > 0:
                        return True
            return False
        except BrowserCDPError:
            return True

    async def _require_stateless_session(self, session: _Session) -> None:
        """Refuse model-authored script on a session that holds any origin state."""
        # Re-checked on every call rather than cached: a session that logs in
        # after an earlier check must not keep script access.
        if await self._holds_origin_state(session):
            raise BrowserPolicyError(_STATEFUL_SESSION_REJECTION)

    async def _script_eval(self, session: _Session, code: str) -> Mapping[str, Any]:
        """Run model-authored source in a designated stateless session only."""
        response = await self._command(session, "Runtime.evaluate", {
            "expression": code, "returnByValue": True, "awaitPromise": True,
            "userGesture": False, "includeCommandLineAPI": False,
        })
        result = response.get("result", {})
        if not isinstance(result, Mapping):
            raise BrowserCDPError("Script evaluation returned a malformed result")
        evaluation: dict[str, Any] = {
            "type": result.get("type"), "subtype": result.get("subtype"),
            "description": redact(result.get("description"), string_limit=1_000),
        }
        if "value" in result:
            try:
                serialized = canonical(result["value"])
            except Exception:
                serialized = json.dumps(str(result["value"]))
            text, truncated = _utf8_prefix(
                _redact_text(serialized, self.policy.max_observation_chars),
                self.policy.max_observation_chars)
            evaluation["value"] = text
            evaluation["truncated"] = truncated
        payload: dict[str, Any] = {"evaluation": evaluation}
        if response.get("exceptionDetails"):
            payload["exception"] = redact(response["exceptionDetails"], string_limit=2_000)
        return payload

    async def _snapshot(self, session: _Session) -> Mapping[str, Any]:
        value = await self._evaluate(session, _INSPECT_EXPRESSION)
        if not isinstance(value, Mapping):
            raise BrowserCDPError("Browser inspection returned a malformed snapshot")
        url = value.get("url", "")
        if not self.policy.allows(url):
            raise BrowserPolicyError("Browser left the exact origin allowlist")
        requests = list(session.requests.values())[-40:]
        return {"page": value, "network": requests, "logs": list(session.logs)}

    async def _perform(self, session: _Session, action: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if action == "inspect":
            return await self._snapshot(session)
        if action == "evaluate":
            return await self._script_eval(session, arguments["code"])
        if action == "read_response":
            request = session.requests[arguments["request_id"]]
            response = await self._command(
                session, "Network.getResponseBody", {"requestId": arguments["request_id"]})
            body, encoded = response.get("body"), response.get("base64Encoded", False)
            if not isinstance(body, str) or type(encoded) is not bool:
                raise BrowserCDPError("Chrome returned a malformed response body")
            if encoded:
                try:
                    raw = base64.b64decode(body, validate=True)
                except Exception as exc:
                    raise BrowserCDPError("Chrome returned invalid base64 response data") from exc
            else:
                try:
                    raw = body.encode("utf-8")
                except UnicodeError as exc:
                    raise BrowserCDPError("Response body is not valid UTF-8 text") from exc
            total_decoded_bytes = len(raw)
            raw_prefix = raw[:self.policy.max_response_body_bytes]
            text = raw_prefix.decode("utf-8", errors="replace")
            text = _redact_text(text, max(len(text), self.policy.max_response_body_bytes))
            text, redaction_truncated = _utf8_prefix(text, self.policy.max_response_body_bytes)
            return {"response": {
                "request": {key: request.get(key) for key in
                            ("request_id", "url", "method", "resource_type", "status", "mime_type")},
                "body": text,
                "returned_body_bytes": len(text.encode("utf-8")),
                "total_decoded_bytes": total_decoded_bytes,
                "truncated": total_decoded_bytes > len(raw_prefix) or redaction_truncated,
                "was_base64_encoded": encoded,
            }}
        if action == "navigate":
            response = await self._command(session, "Page.navigate", {"url": arguments["url"]})
            if response.get("errorText"):
                raise BrowserCDPError(f"Navigation failed: {response['errorText']}")
            deadline = asyncio.get_running_loop().time() + self.policy.navigation_timeout
            state: Any = None
            while asyncio.get_running_loop().time() < deadline:
                state = await self._evaluate(session, _PAGE_STATE_EXPRESSION)
                if isinstance(state, Mapping) and state.get("readyState") in {"interactive", "complete"}:
                    break
                await asyncio.sleep(0.1)
            if not isinstance(state, Mapping) or not self.policy.allows(state.get("url", "")):
                raise BrowserPolicyError("Navigation did not settle on an allowed origin")
            return {"navigation": {"url": state["url"], "ready_state": state.get("readyState")},
                    **await self._snapshot(session)}
        if action == "fetch":
            function_response = await self._command(session, "Runtime.evaluate", {
                "expression": _FETCH_FUNCTION_EXPRESSION,
                "returnByValue": False,
                "awaitPromise": False,
                "userGesture": False,
                "includeCommandLineAPI": False,
            })
            if function_response.get("exceptionDetails"):
                raise BrowserCDPError("Fixed authenticated fetch function failed to initialize")
            function = function_response.get("result", {})
            object_id = function.get("objectId") if isinstance(function, Mapping) else None
            if (not isinstance(function, Mapping) or function.get("type") != "function"
                    or not isinstance(object_id, str) or not object_id):
                raise BrowserCDPError("Fixed authenticated fetch function was not created")
            try:
                response = await self._command(session, "Runtime.callFunctionOn", {
                    "objectId": object_id,
                    "functionDeclaration": _FETCH_CALL_FUNCTION,
                    "arguments": [{"value": arguments[key]} for key in
                                  ("url", "method", "headers", "body", "auth_mode")]
                                 + [{"value": self.policy.max_response_body_bytes}],
                    "returnByValue": True,
                    "awaitPromise": True,
                    "userGesture": False,
                    "silent": True,
                })
            finally:
                with suppress(Exception):
                    await self._command(session, "Runtime.releaseObject", {"objectId": object_id})
            if response.get("exceptionDetails"):
                raise BrowserCDPError("Fixed authenticated fetch failed")
            result = response.get("result", {})
            value = result.get("value") if isinstance(result, Mapping) else None
            if not isinstance(value, Mapping):
                raise BrowserCDPError("Fixed authenticated fetch returned a malformed result")
            final_url = value.get("url", "")
            if not self.policy.allows(final_url):
                raise BrowserPolicyError("Fetch redirected outside the exact origin allowlist")
            body = value.get("body", "")
            if not isinstance(body, str):
                raise BrowserCDPError("Fixed authenticated fetch returned a malformed body")
            body, body_truncated = _utf8_prefix(body, self.policy.max_response_body_bytes)
            return {"fetch": {
                "request": {"url": arguments["url"], "method": arguments["method"],
                            "auth_mode": arguments["auth_mode"],
                            "header_names": sorted(arguments["headers"])},
                "response": {"url": final_url, "status": value.get("status"),
                             "status_text": value.get("status_text"),
                             "redirected": value.get("redirected"),
                             "response_type": value.get("response_type"),
                             "headers": value.get("headers", {}), "body": body,
                             "returned_body_bytes": len(body.encode("utf-8")),
                             "total_body_characters": value.get("total_body_characters"),
                             "truncated": bool(value.get("truncated_in_page")) or body_truncated,
                             "bearer_applied": bool(value.get("bearer_applied"))},
            }}
        if action in {"click", "click_text", "type"}:
            if action == "click_text":
                node_id, node = await self._resolve_text_node(session, arguments["text"])
            else:
                node_id, node = await self._resolve_node(session, arguments["selector"])
            attributes = dict(zip(node.get("attributes", [])[::2], node.get("attributes", [])[1::2]))
            if action == "type":
                tag = str(node.get("nodeName", "")).lower()
                field_type = str(attributes.get("type", "text")).lower()
                editable = attributes.get("contenteditable", "").lower() in {"", "true"} and "contenteditable" in attributes
                if tag not in {"input", "textarea"} and not editable:
                    raise BrowserPolicyError("Typing target is not an editable control")
                if field_type in {"password", "file", "hidden"}:
                    raise BrowserPolicyError("Typing into sensitive or non-text controls is forbidden")
                await self._command(session, "DOM.focus", {"nodeId": node_id})
                for event_type in ("keyDown", "keyUp"):
                    await self._command(session, "Input.dispatchKeyEvent", {
                        "type": event_type, "key": "a", "code": "KeyA", "modifiers": 2,
                    })
                await self._command(session, "Input.dispatchKeyEvent", {"type": "keyDown", "key": "Backspace", "code": "Backspace"})
                await self._command(session, "Input.dispatchKeyEvent", {"type": "keyUp", "key": "Backspace", "code": "Backspace"})
                await self._command(session, "Input.insertText", {"text": arguments["text"]})
                interaction = {"kind": "type", "selector_sha256": hashlib.sha256(
                                   arguments["selector"].encode()).hexdigest(),
                               "tag": tag, "inserted_characters": len(arguments["text"])}
            else:
                await self._command(session, "DOM.scrollIntoViewIfNeeded", {"nodeId": node_id})
                model = await self._command(session, "DOM.getBoxModel", {"nodeId": node_id})
                quad = model.get("model", {}).get("content") or model.get("model", {}).get("border")
                if not isinstance(quad, list) or len(quad) != 8:
                    raise BrowserCDPError("Click target has no usable box")
                x = sum(quad[0::2]) / 4
                y = sum(quad[1::2]) / 4
                await self._command(session, "Input.dispatchMouseEvent", {
                    "type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
                await self._command(session, "Input.dispatchMouseEvent", {
                    "type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
                if action == "click_text":
                    interaction = {
                        "kind": "click_text",
                        "target_text_sha256": hashlib.sha256(
                            arguments["text"].encode("utf-8")).hexdigest(),
                        "target_text_length": len(arguments["text"]),
                        "tag": str(node.get("nodeName", "")).lower(),
                    }
                else:
                    interaction = {"kind": "click", "selector_sha256": hashlib.sha256(
                                       arguments["selector"].encode()).hexdigest(),
                                   "tag": str(node.get("nodeName", "")).lower()}
            await asyncio.sleep(0)
            if action == "type":
                # A page can reflect typed text into its DOM or console.  Do not
                # capture either in the same durable observation.
                return {"interaction": interaction}
            return {"interaction": interaction, **await self._snapshot(session)}
        if action == "replay_request":
            request = session.requests.get(arguments["request_id"])
            if not request or not request.get("eligible"):
                raise BrowserPolicyError("Only observed same-origin GET/HEAD XHR requests can be replayed")
            await self._command(session, "Network.replayXHR", {"requestId": arguments["request_id"]})
            return {"replay": {key: request.get(key) for key in
                               ("request_id", "url", "method", "resource_type")},
                    **await self._snapshot(session)}
        raise BrowserPolicyError("Unknown browser action")

    async def _resolve_node(self, session: _Session, selector: str) -> tuple[int, Mapping[str, Any]]:
        root = await self._command(session, "DOM.getDocument", {"depth": 0, "pierce": False})
        root_id = root.get("root", {}).get("nodeId")
        if not isinstance(root_id, int):
            raise BrowserCDPError("DOM root is unavailable")
        found = await self._command(session, "DOM.querySelector", {"nodeId": root_id, "selector": selector})
        node_id = found.get("nodeId")
        if not isinstance(node_id, int) or node_id <= 0:
            raise BrowserPolicyError("CSS selector matched no element")
        described = await self._command(session, "DOM.describeNode", {"nodeId": node_id, "depth": 0})
        node = described.get("node", {})
        if not isinstance(node, Mapping) or node.get("nodeType") != 1:
            raise BrowserPolicyError("CSS selector did not resolve to an element")
        return node_id, node

    async def _resolve_text_node(self, session: _Session, target: str) -> tuple[int, Mapping[str, Any]]:
        resolver_response = await self._command(session, "Runtime.evaluate", {
            "expression": _CLICK_TEXT_RESOLVER_EXPRESSION,
            "returnByValue": False,
            "awaitPromise": False,
            "userGesture": False,
            "includeCommandLineAPI": False,
        })
        if resolver_response.get("exceptionDetails"):
            raise BrowserCDPError("Fixed visible-text resolver failed")
        resolver = resolver_response.get("result", {})
        resolver_id = resolver.get("objectId") if isinstance(resolver, Mapping) else None
        if (not isinstance(resolver, Mapping) or resolver.get("type") != "function"
                or not isinstance(resolver_id, str) or not resolver_id):
            raise BrowserCDPError("Fixed visible-text resolver was not created")
        try:
            match_response = await self._command(session, "Runtime.callFunctionOn", {
                "objectId": resolver_id,
                "functionDeclaration": _CLICK_TEXT_CALL_FUNCTION,
                "arguments": [{"value": target}],
                "returnByValue": False,
                "awaitPromise": False,
                "userGesture": False,
                "silent": True,
            })
        finally:
            with suppress(Exception):
                await self._command(session, "Runtime.releaseObject", {"objectId": resolver_id})
        if match_response.get("exceptionDetails"):
            raise BrowserCDPError("Fixed visible-text lookup failed")
        match = match_response.get("result", {})
        if (isinstance(match, Mapping) and match.get("type") == "number"
                and type(match.get("value")) in {int, float} and match["value"] <= 0):
            raise _BrowserTextTargetError(_CLICK_TEXT_REJECTION)
        object_id = match.get("objectId") if isinstance(match, Mapping) else None
        if (not isinstance(match, Mapping) or match.get("subtype") != "node"
                or not isinstance(object_id, str) or not object_id):
            raise BrowserCDPError("Fixed visible-text lookup returned an invalid target")
        try:
            requested = await self._command(session, "DOM.requestNode", {"objectId": object_id})
        finally:
            with suppress(Exception):
                await self._command(session, "Runtime.releaseObject", {"objectId": object_id})
        node_id = requested.get("nodeId")
        if not isinstance(node_id, int) or node_id <= 0:
            raise BrowserCDPError("Fixed visible-text target has no DOM node")
        described = await self._command(session, "DOM.describeNode", {"nodeId": node_id, "depth": 0})
        node = described.get("node", {})
        if not isinstance(node, Mapping) or node.get("nodeType") != 1:
            raise BrowserCDPError("Fixed visible-text target is not an element")
        raw_attributes = node.get("attributes", [])
        if not isinstance(raw_attributes, list) or len(raw_attributes) % 2:
            raise BrowserCDPError("Fixed visible-text target has malformed attributes")
        attributes = dict(zip(raw_attributes[::2], raw_attributes[1::2]))
        tag = str(node.get("nodeName", "")).lower()
        if tag not in {"a", "button"} and str(attributes.get("role", "")).lower() != "button":
            raise BrowserCDPError("Fixed visible-text target is not a link or button")
        return node_id, node

    def _write_observation(self, session: _Session, action: str, data: Mapping[str, Any]) -> dict[str, Any]:
        safe_data = redact(data, string_limit=self.policy.max_observation_chars)
        if action in {"read_response", "fetch"} and isinstance(safe_data, dict):
            response = (safe_data.get("response") if action == "read_response"
                        else safe_data.get("fetch", {}).get("response"))
            if isinstance(response, dict) and isinstance(response.get("body"), str):
                body, body_truncated = _utf8_prefix(
                    response["body"], self.policy.max_response_body_bytes)
                response["body"] = body
                response["returned_body_bytes"] = len(body.encode("utf-8"))
                response["truncated"] = bool(response.get("truncated")) or body_truncated
        observation = {
            "schema_version": self.OBSERVATION_SCHEMA_VERSION,
            "origin": "browser_observation",
            "session": session.name,
            "action": action,
            "sequence": session.sequence,
            "captured_at": now(),
            "data": safe_data,
        }
        artifact_id = hashlib.sha256(canonical(observation).encode("utf-8")).hexdigest()
        path = self.observation_dir / f"{artifact_id}.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if canonical(existing) != canonical(observation):
                raise BrowserRecoveryRequired("Content-addressed browser artifact collision")
        else:
            atomic_json(path, observation)
        return {"schema_version": 1, "origin": "browser_observation", "artifact_id": artifact_id,
                "evidence_id": artifact_id, "session": session.name, "action": action,
                "sequence": session.sequence}

    def get_observation(self, artifact_id: str) -> Mapping[str, Any]:
        """Read one redacted observation and verify its content address."""
        if (not isinstance(artifact_id, str) or len(artifact_id) != 64
                or any(character not in "0123456789abcdef" for character in artifact_id)):
            raise BrowserPolicyError("Invalid browser observation id")
        path = self.observation_dir / f"{artifact_id}.json"
        try:
            value = json.loads(path.read_text())
        except Exception as exc:
            raise BrowserCDPError("Browser observation is missing or unreadable") from exc
        actual = hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
        if actual != artifact_id or value.get("origin") != "browser_observation":
            raise BrowserCDPError("Browser observation integrity check failed")
        return value

    def tool_specs(self) -> list[dict[str, Any]]:
        sessions = list(self.sessions)
        # Provider call IDs are supplied by the host runtime to dispatch_tool;
        # models never choose their durable idempotency key.
        base = {"session": {"type": "string", "enum": sessions}}
        definitions = [
            ("browser_inspect", "Read a bounded, redacted snapshot from an authorized browser.", {}),
            ("browser_read_response",
             "Read a capped, redacted body only when the latest inspection lists this request_id with "
             "readable=true and finished=true.",
             {"request_id": {"type": "string", "maxLength": 200}}),
        ]
        if self.policy.interaction_enabled:
            definitions.extend([
                ("browser_navigate", "Navigate to an exact-allowlisted URL.",
                 {"url": {"type": "string", "maxLength": 8000}}),
                ("browser_click", "Click one CSS-selected element in an authorized test session.",
                 {"selector": {"type": "string", "maxLength": self.policy.max_selector_chars}}),
                ("browser_click_text",
                 "Click exactly one visible link or button whose normalized text or aria-label exactly matches.",
                 {"text": {"type": "string", "maxLength": _MAX_CLICK_TEXT_CHARS}}),
                ("browser_type", "Replace text in a non-password editable control.",
                 {"selector": {"type": "string", "maxLength": self.policy.max_selector_chars},
                  "text": {"type": "string", "maxLength": self.policy.max_text_chars}}),
                ("browser_fetch",
                 "Issue a bounded exact-allowlisted fetch inside this browser session. Cookies stay in the "
                 "browser; auto_bearer may apply a session JWT without revealing it. Use none for a clean "
                 "unauthenticated control, cookies for cookie auth, or auto_bearer for app-token auth.",
                 {"url": {"type": "string", "maxLength": 8000},
                  "method": {"type": "string", "enum": sorted(_FETCH_METHODS)},
                  "headers": {"type": "object", "maxProperties": 24,
                              "additionalProperties": {"type": "string", "maxLength": 2000}},
                  "body": {"type": "string", "maxLength": self.policy.max_text_chars},
                  "auth_mode": {"type": "string", "enum": sorted(_FETCH_AUTH_MODES)}}),
            ])
        if self.policy.request_replay_enabled and self.policy.max_replays_per_session:
            definitions.append((
                "browser_replay_request",
                "Replay only a request_id whose latest inspection lists eligible=true; bounded to observed "
                "same-origin GET/HEAD XHR.",
                {"request_id": {"type": "string", "maxLength": 200}},
            ))
        eval_sessions = sorted(set(self.policy.script_eval_sessions) & set(sessions))
        if eval_sessions:
            definitions.append((
                "browser_evaluate",
                "Evaluate model-authored JavaScript in a designated session that holds no cookies, DOM storage, "
                "IndexedDB, Cache Storage, or persistent storage for an allowed origin. Never advertised for an "
                "authenticated session; state checks fail closed and results are redacted and capped.",
                {"code": {"type": "string", "maxLength": self.policy.max_script_chars}},
            ))
        result = []
        for name, description, extra in definitions:
            properties = {**base, **extra}
            if name == "browser_evaluate":
                properties["session"] = {"type": "string", "enum": eval_sessions}
            result.append({"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "additionalProperties": False,
                               "properties": properties, "required": list(properties)}}})
        return result

    def runtime_tools(self) -> list[dict[str, Any]]:
        """Return the descriptor shape accepted by Runtime.create_workers()."""
        return [{"name": item["function"]["name"],
                 "description": item["function"]["description"],
                 "inputSchema": item["function"]["parameters"]}
                for item in self.tool_specs()]

    async def dispatch_tool(self, name: str, arguments: Mapping[str, Any], *,
                            call_id: str) -> Mapping[str, Any]:
        action_by_name = {"browser_inspect": "inspect", "browser_navigate": "navigate",
                          "browser_read_response": "read_response",
                          "browser_click": "click", "browser_click_text": "click_text",
                          "browser_type": "type",
                          "browser_fetch": "fetch",
                          "browser_evaluate": "evaluate",
                          "browser_replay_request": "replay_request"}
        if name not in action_by_name or not isinstance(arguments, Mapping):
            raise BrowserPolicyError("Unknown browser tool")
        payload = dict(arguments)
        try:
            session = payload.pop("session")
        except KeyError as exc:
            raise BrowserPolicyError("Browser tool requires session") from exc
        return await self.execute(session, action_by_name[name], payload, call_id=call_id)

    async def close(self):
        for session in self.sessions.values():
            tasks = list(session.handler_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if session.transport is not None:
                with suppress(Exception):
                    await session.transport.close()
                session.transport = None
        self._started = False

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *_):
        await self.close()
