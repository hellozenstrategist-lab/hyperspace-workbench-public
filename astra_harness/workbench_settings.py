"""Private provider preferences for the local visual Workbench assistant."""
from __future__ import annotations

import copy
import json
import os
import re

from .schema import atomic_json


DEFAULTS = {"provider": "chatgpt", "chatgpt": {"model": ""},
            "openrouter": {"model": "", "max_tokens": 2048},
            "privacy": {"include_source": True, "include_run": True}}


def model_id(value, provider):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError("Model ID must be bounded text")
    if value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value):
        raise ValueError("Model ID contains unsupported characters")
    if "://" in value or (provider == "openrouter" and
            (value.lower().split(":", 1)[0] in {"auto", "openrouter/auto", "openrouter/free"} or ":online" in value.lower())):
        raise ValueError("Automatic model routing and web-enabled models are unavailable")
    return value


def _fields(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Unsupported assistant setting")


class WorkbenchSettings:
    def __init__(self, reader):
        self.reader = reader
        self.directory = reader.root / "data" / "workbench-chat"
        self.path = self.directory / "settings.json"
        self.secret_path = self.directory / "credentials.json"
        self.value = copy.deepcopy(DEFAULTS)
        self.session_secret = None
        self.saved_secret = None
        self.error = None
        try:
            content, truncated = reader._read(self.path, 16384)
            if truncated:
                raise ValueError("Saved settings exceed their bound")
            self.value = self._validated(json.loads(content))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            self.error = "Saved assistant settings are unavailable"
        try:
            content, truncated = reader._read(self.secret_path, 8192)
            value = json.loads(content)
            _fields(value, {"openrouter"})
            if truncated:
                raise ValueError("Saved credential exceeds its bound")
            self.saved_secret = self._secret(value.get("openrouter"))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError):
            self.error = "Saved assistant credential is unavailable"

    @staticmethod
    def _secret(value):
        if not isinstance(value, str) or not 1 <= len(value) <= 4096 or any(not 33 <= ord(character) <= 126 for character in value):
            raise ValueError("Credential must be nonempty bounded text without whitespace")
        return value

    def _validated(self, body):
        _fields(body, DEFAULTS)
        value = copy.deepcopy(self.value)
        if "provider" in body:
            if not isinstance(body["provider"], str) or body["provider"] not in {"chatgpt", "openrouter"}:
                raise ValueError("Unknown assistant provider")
            value["provider"] = body["provider"]
        for provider in ("chatgpt", "openrouter"):
            incoming = body.get(provider, {})
            allowed = {"model"} if provider == "chatgpt" else {"model", "max_tokens", "secret", "remember_secret", "clear_secret"}
            _fields(incoming, allowed)
            if "model" in incoming:
                value[provider]["model"] = model_id(incoming["model"], provider)
            if "max_tokens" in incoming:
                if type(incoming["max_tokens"]) is not int or not 256 <= incoming["max_tokens"] <= 8192:
                    raise ValueError("OpenRouter output limit must be 256 to 8192 tokens")
                value[provider]["max_tokens"] = incoming["max_tokens"]
        privacy = body.get("privacy", {})
        _fields(privacy, DEFAULTS["privacy"])
        if any(type(value) is not bool for value in privacy.values()):
            raise ValueError("Privacy preferences must be boolean")
        value["privacy"].update(privacy)
        return value

    def credential(self):
        for secret, source in ((self.session_secret, "session"), (self.saved_secret, "saved"),
                               (os.environ.get("OPENROUTER_API_KEY"), "environment")):
            if secret:
                try:
                    return self._secret(secret), source
                except ValueError:
                    continue
        return None, "none"

    def public(self):
        value = copy.deepcopy(self.value)
        secret, source = self.credential()
        value["openrouter"].update(credential_present=bool(secret), credential_source=source)
        if self.error:
            value["error"] = self.error
        return value

    def update(self, body):
        value = self._validated(body)
        incoming = body.get("openrouter", {})
        for field in ("remember_secret", "clear_secret"):
            if field in incoming and type(incoming[field]) is not bool:
                raise ValueError("Credential controls must be boolean")
        secret = self._secret(incoming["secret"]) if "secret" in incoming else None
        if incoming.get("clear_secret") and (secret or incoming.get("remember_secret")):
            raise ValueError("Clear and replace credentials in separate settings changes")
        if incoming.get("remember_secret") and secret is None:
            raise ValueError("Remembering requires explicitly supplied credentials")
        self.reader._safe(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.reader._safe(self.path)
        self.reader._safe(self.secret_path)
        if incoming.get("clear_secret") or (secret is not None and not incoming.get("remember_secret")):
            self.secret_path.unlink(missing_ok=True)
            self.saved_secret = None
            self.session_secret = None
        if secret is not None:
            if incoming.get("remember_secret"):
                atomic_json(self.secret_path, {"openrouter": secret})
                os.chmod(self.secret_path, 0o600)
                self.saved_secret = secret
                self.session_secret = None
            else:
                self.session_secret = secret
        atomic_json(self.path, value)
        os.chmod(self.path, 0o600)
        self.value = value
        self.error = None
        return self.public()
