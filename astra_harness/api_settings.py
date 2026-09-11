"""Private OpenRouter settings; public command results never contain credentials."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        # A redirected account check must never forward the credential elsewhere.
        return None


urlopen = build_opener(_NoRedirect()).open


def validate_openrouter_model(model):
    if (not isinstance(model, str) or not 3 <= len(model) <= 200 or "/" not in model
            or any(char.isspace() or ord(char) < 32 for char in model)
            or any(not part for part in model.split("/"))):
        raise ValueError("Choose an OpenRouter model ID in provider/model form")
    return model


def _path():
    value = os.environ.get("HYPERSPACE_HARNESS_CONFIG")
    return Path(value).expanduser().absolute() if value else Path.home() / ".config/hyperspace-harness/openrouter.json"


def _private(info, mode, label):
    if (stat.S_IMODE(info.st_mode) != mode or
            (hasattr(os, "getuid") and info.st_uid != os.getuid())):
        raise ValueError(f"{label} must be owned by you with permissions {mode:o}")


def _parent(path, *, create=False):
    if create:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.parent.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("OpenRouter settings directory must not be a symlink")
    _private(info, 0o700, "OpenRouter settings directory")


def _key(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096 or any(char.isspace() or not 33 <= ord(char) <= 126 for char in value):
        raise ValueError("OpenRouter API key must be nonempty printable text without whitespace")
    return value


def _saved():
    path = _path()
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("OpenRouter settings must be a regular file, not a symlink")
    _parent(path)
    _private(info, 0o600, "OpenRouter settings file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        actual = os.fstat(stream.fileno())
        _private(actual, 0o600, "OpenRouter settings file")
        if not stat.S_ISREG(actual.st_mode) or actual.st_size > 16384:
            raise ValueError("OpenRouter settings file is invalid or too large")
        try:
            data = json.loads(stream.read(16385))
        except (ValueError, UnicodeError):
            raise ValueError("OpenRouter settings contain invalid JSON") from None
    if not isinstance(data, dict) or set(data) - {"version", "api_key", "model"} or data.get("version") != 1:
        raise ValueError("OpenRouter settings have an unsupported format")
    if data.get("api_key") is not None:
        _key(data["api_key"])
    if data.get("model") is not None:
        validate_openrouter_model(data["model"])
    return data


def _write(data):
    path = _path()
    _parent(path, create=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".openrouter-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump({"version": 1, **data}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def openrouter_settings():
    """Internal credential lookup; environment overrides saved fields independently."""
    saved = _saved()
    key = os.environ.get("OPENROUTER_API_KEY") or saved.get("api_key")
    model = os.environ.get("OPENROUTER_MODEL") or saved.get("model")
    return {"api_key": _key(key) if key else None,
            "model": validate_openrouter_model(model) if model else None,
            "key_source": "environment" if os.environ.get("OPENROUTER_API_KEY") else "saved" if key else "missing",
            "model_source": "environment" if os.environ.get("OPENROUTER_MODEL") else "saved" if model else "missing"}


def api_status():
    """Return effective model and credential presence without exposing the key."""
    settings = openrouter_settings()
    return {"model": settings["model"], "key_configured": bool(settings["api_key"]),
            "key_source": settings["key_source"], "model_source": settings["model_source"]}


def api_setup(model=None):
    """Save an existing credential or ask the human through a hidden terminal prompt."""
    if model is not None:
        validate_openrouter_model(model)
    settings = openrouter_settings()
    key = settings["api_key"]
    if not key:
        if not sys.stdin.isatty():
            raise ValueError("Run hyperspace api setup in an interactive terminal, or set OPENROUTER_API_KEY in the environment; do not put the key in command arguments")
        key = _key(getpass.getpass("OpenRouter API key (hidden): "))
    data = {"api_key": key}
    selected = model if model is not None else settings["model"]
    if selected is None and sys.stdin.isatty():
        selected = validate_openrouter_model(input("OpenRouter model (provider/model): ").strip())
    if selected:
        data["model"] = selected
    _write(data)
    return api_status()


def api_set_model(model):
    """Update the saved model without reading a secret into command arguments."""
    model = validate_openrouter_model(model)
    data = _saved()
    data["model"] = model
    _write(data)
    return api_status()


def api_check():
    """Check the key with a bounded account request; never return provider bodies."""
    settings = openrouter_settings()
    if not settings["api_key"]:
        raise ValueError("No OpenRouter API key configured; run hyperspace api setup in a terminal")
    request = Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + settings["api_key"], "Accept": "application/json"})
    try:
        with urlopen(request, timeout=15) as response:
            status = response.status
    except HTTPError as exc:
        status = exc.code
        exc.close()
    except (URLError, OSError, TimeoutError):
        return {"key_valid": None, "connection": "failed", "http_status": None}
    return {"key_valid": True if status == 200 else False if status in {401, 403} else None,
            "connection": "reached", "http_status": status}
