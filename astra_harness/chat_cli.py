"""Launch the Codex terminal UI using saved OpenRouter settings by default."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .api_settings import api_status, openrouter_settings, validate_openrouter_model


HARNESS_CONTEXT = """You are the user's interactive coding assistant for the Hyperspace harness.
Use the native Codex tools to inspect and edit project files, run commands, and test requested changes. Preserve existing work and follow the user's instructions and workspace approval policy.
The harness supports at most five research workers per run. Keep the total concurrent research workers at five or fewer; do not evade this limit through nested groups or concurrent harness runs. The chat assistant manages this work; the bounded mission workers only receive knowledge tools.
Read README.md and docs/chat-cli.md as needed. Relevant implementation: astra_harness/cli.py, mission.py, coordinator.py, runtime_factory.py, codex_runtime.py, openrouter_runtime.py, api_settings.py, router.py, attention.py. Example tasks are example_mission.json and example_mission_five.json; reports and ledgers are under runs/.
Use ./harness api status to inspect redacted OpenRouter configuration. Use ./harness api model provider/model to change the saved research model. The runtime reads saved credentials itself. Do not open, print, paste, or copy credential files or keys into prompts, code, logs, or command arguments. Initial key entry belongs to the human running hyperspace api setup --model provider/model in a terminal. ./harness api check checks the saved key without a generation request.
The chat uses the existing ChatGPT/Codex login. A research run uses OpenRouter only when explicitly selected with --provider openrouter; choose --model or the saved model. Read ./harness mission --help or ./harness run --help before launching work. Model calls consume the selected account's allowance. Notifications require the user's explicit request. Worker lifetime overlap, HTTP request overlap, and actual provider generation concurrency are different measurements.
Wait for the user's request when no initial task is supplied. Do not start workers solely because this context describes them.
For a bounded review loop use ./harness research 'starting prompt', or --prompt-file FILE. It uses an explicit GLM head, up to three DeepSeek workers, and Qwen QC through OpenRouter, independently of the chat model. Read ./harness research --help. It analyzes supplied --evidence files and model reasoning. A new run can explicitly attach exactly two private CDP endpoints with --browser-cdp plus exact --browser-origin allowlists; only the first two workers receive isolated sessions and fixed bounded browser tools, including exact-allowlisted in-session fetches with cookie or non-exporting automatic bearer authentication. No shell, model-supplied JavaScript, or external search is exposed. Qwen can reject incomplete results and request specific revisions. Completion requires its structured acceptance of every fixed criterion. Limits pause the run; uncertainty and missing external evidence block it. Research status uses ./harness status --run-dir DIR; resume a checkpoint with ./harness research --resume --run-dir DIR. Keep failed and completed evidence intact.
"""


def chat_command(args, root):
    """Build an argv list; caller-supplied prompts never pass through a shell."""
    executable = shutil.which("codex")
    if not executable:
        raise ValueError("Codex CLI is not installed or is missing from PATH; install it and run codex login")
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Chat workspace must be an existing directory")
    resume = getattr(args, "resume", None)
    command = [executable] + (["resume"] if resume else [])
    command += ["--cd", str(root), "--sandbox", "workspace-write", "--ask-for-approval", "on-request"]
    provider = getattr(args, "provider", "openrouter")
    context = HARNESS_CONTEXT
    model = getattr(args, "model", None)
    if provider == "openrouter":
        model = model or api_status()["model"]
        if not model:
            raise ValueError("Configure an OpenRouter model with hyperspace api setup --model provider/model")
        validate_openrouter_model(model)
        context = context.replace("The chat uses the existing ChatGPT/Codex login.",
                                  "The chat uses OpenRouter with the user's saved API model.")
        for setting in (
            'model_provider="hyperspace_openrouter"',
            'model_providers.hyperspace_openrouter.name="OpenRouter"',
            'model_providers.hyperspace_openrouter.base_url="https://openrouter.ai/api/v1"',
            'model_providers.hyperspace_openrouter.env_key="HYPERSPACE_CHAT_API_KEY"',
            'model_providers.hyperspace_openrouter.wire_api="responses"',
            'model_providers.hyperspace_openrouter.requires_openai_auth=false',
            'model_providers.hyperspace_openrouter.supports_websockets=false',
            'web_search="disabled"',
            'shell_environment_policy.exclude=["HYPERSPACE_CHAT_API_KEY", "OPENROUTER_API_KEY"]',
        ):
            command += ["--config", setting]
        if getattr(args, "search", False):
            raise ValueError("--search requires --provider codex")
    else:
        command += ["--config", 'model_provider="openai"', "--config", 'forced_login_method="chatgpt"']
    command += ["--config", "developer_instructions=" + json.dumps(context, ensure_ascii=False)]
    if model is not None:
        if not isinstance(model, str) or not model or len(model) > 200 or model.startswith("-") or any(char.isspace() or ord(char) < 32 for char in model):
            raise ValueError("Chat model must be a nonempty model ID")
        command += ["--model", model]
    for flag in ("search", "no_alt_screen"):
        if getattr(args, flag, False):
            command.append("--" + flag.replace("_", "-"))
    for directory in getattr(args, "add_dir", []) or []:
        selected = Path(directory).expanduser().resolve()
        if not selected.is_dir():
            raise ValueError("Each --add-dir must name an existing directory")
        command += ["--add-dir", str(selected)]
    prompt = getattr(args, "prompt", None)
    if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 64000 or "\0" in prompt):
        raise ValueError("Chat prompt must contain at most 64000 characters and no NUL bytes")
    if resume is True or resume == "last":
        if prompt is not None:
            raise ValueError("With --resume last, enter your follow-up after the session opens, or provide an explicit session ID")
        command.append("--last")
    elif resume and resume != "pick":
        if not isinstance(resume, str) or not resume or len(resume) > 200 or "\0" in resume:
            raise ValueError("Resume requires last, pick, or a Codex session ID/name")
        command += ["--", resume]
        if prompt is not None:
            command.append(prompt)
    elif resume == "pick":
        if prompt is not None:
            raise ValueError("With --resume pick, enter your follow-up after selecting the session")
    elif prompt is not None:
        command += ["--", prompt]
    return command


def launch_chat(args, root):
    """Hand the real terminal to Codex; tests replace subprocess.run entirely."""
    command = chat_command(args, root)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError("Interactive chat needs a terminal. Run hyperspace chat in your terminal")
    try:
        options = {}
        if getattr(args, "provider", "openrouter") == "openrouter":
            key = openrouter_settings()["api_key"]
            if not key:
                raise ValueError("Run hyperspace api setup in your terminal to configure your OpenRouter key")
            options["env"] = {**os.environ, "HYPERSPACE_CHAT_API_KEY": key}
        result = subprocess.run(command, cwd=str(Path(root).expanduser().resolve()), check=False, **options)
    except KeyboardInterrupt:
        return 130
    return result.returncode
