"""Explicit runtime selection. Credentials are never configuration or run data."""
from __future__ import annotations

import os

from .codex_runtime import MODEL, SUPPORTED_MODELS, Runtime
from .schema import AGENTS, worker_ids


def runtime_options(provider="codex", model=None):
    if provider not in {"codex", "openrouter"}:
        raise ValueError("Runtime provider must be codex or openrouter")
    if provider == "openrouter":
        model = model or os.environ.get("OPENROUTER_MODEL")
        if not model:
            from .api_settings import openrouter_settings
            model = openrouter_settings()["model"]
        if not isinstance(model, str) or not model.strip() or "/" not in model or len(model) > 200:
            raise ValueError("Choose an OpenRouter model with hyperspace api setup, --model provider/model, or OPENROUTER_MODEL")
        if any(char.isspace() for char in model):
            raise ValueError("OpenRouter model ID must not contain whitespace")
    else:
        model = model or MODEL
        if model not in SUPPORTED_MODELS:
            raise ValueError("Unsupported Codex model: " + str(model))
    return provider, model


def resolve_model(provider="codex", model=None):
    return runtime_options(provider, model)[1]


def create_runtime(state_path, artifacts_dir, on_event, on_tool, *, agents=AGENTS,
                   provider="codex", model=None, codex_class=None):
    agents = tuple(agents)
    if agents != worker_ids(len(agents)):
        raise ValueError("Use the canonical worker IDs agent-a through the configured count")
    provider, model = runtime_options(provider, model)
    if provider == "openrouter":
        from .openrouter_runtime import OpenRouterRuntime
        return OpenRouterRuntime(state_path, artifacts_dir, on_event, on_tool, agents=agents, model=model)
    # Keep the old default constructor compatible with three-worker integrations.
    kwargs = {}
    if agents != AGENTS:
        kwargs["agents"] = agents
    if model != MODEL:
        kwargs["model"] = model
    return (codex_class or Runtime)(state_path, artifacts_dir, on_event, on_tool, **kwargs)
