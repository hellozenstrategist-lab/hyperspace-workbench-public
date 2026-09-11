from types import SimpleNamespace
from unittest.mock import patch

import pytest

from astra_harness import chat_cli


def args(**changes):
    return SimpleNamespace(**{"prompt": None, "resume": None, "model": None, "search": False,
                              "no_alt_screen": False, "add_dir": [], "provider": "codex", **changes})


def test_default_chat_uses_saved_openrouter_model(tmp_path):
    selected = args(provider="openrouter")
    del selected.provider  # Same default as a bare hyperspace launch.
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"), \
            patch.object(chat_cli, "api_status", return_value={"model": "z-ai/glm-5.3-flash"}):
        command = chat_cli.chat_command(selected, tmp_path)
    assert command[command.index("--model") + 1] == "z-ai/glm-5.3-flash"
    assert 'model_provider="hyperspace_openrouter"' in command
    assert 'forced_login_method="chatgpt"' not in command
    assert 'model_providers.hyperspace_openrouter.wire_api="responses"' in command


def test_openrouter_key_only_passed_privately_to_child(tmp_path):
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"), \
            patch.object(chat_cli, "openrouter_settings", return_value={"api_key": "test-private"}), \
            patch.object(chat_cli.sys.stdin, "isatty", return_value=True), \
            patch.object(chat_cli.sys.stdout, "isatty", return_value=True), \
            patch.object(chat_cli.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
        assert chat_cli.launch_chat(args(provider="openrouter", model="z-ai/glm-5.3-flash"), tmp_path) == 0
    assert "test-private" not in str(run.call_args.args[0])
    assert run.call_args.kwargs["env"]["HYPERSPACE_CHAT_API_KEY"] == "test-private"
    assert 'shell_environment_policy.exclude=["HYPERSPACE_CHAT_API_KEY", "OPENROUTER_API_KEY"]' in run.call_args.args[0]


def test_parser_chat_defaults_to_openrouter():
    from astra_harness.cli import build_parser
    assert build_parser().parse_args(["chat"]).provider == "openrouter"


def test_native_argv_keeps_chatgpt_login_and_harness_context(tmp_path):
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"):
        command = chat_cli.chat_command(args(prompt="Inspect `literal` $(not-a-shell)"), tmp_path)
    assert command[0] == "/native/codex"
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--ask-for-approval") + 1] == "on-request"
    assert 'forced_login_method="chatgpt"' in command
    assert any("at most five research workers" in item for item in command)
    assert command[-2:] == ["--", "Inspect `literal` $(not-a-shell)"]
    assert "--model" not in command


def test_optional_flags_and_terminal_are_forwarded_without_shell(tmp_path):
    selected = args(model="example/alternate-model", search=True, no_alt_screen=True, add_dir=[str(tmp_path)])
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"), \
            patch.object(chat_cli.sys.stdin, "isatty", return_value=True), \
            patch.object(chat_cli.sys.stdout, "isatty", return_value=True), \
            patch.object(chat_cli.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run:
        assert chat_cli.launch_chat(selected, tmp_path) == 7
    command = run.call_args.args[0]
    assert "--search" in command and "--no-alt-screen" in command
    assert command[command.index("--model") + 1] == "example/alternate-model"
    assert command[command.index("--add-dir") + 1] == str(tmp_path)
    assert run.call_args.kwargs == {"cwd": str(tmp_path), "check": False}


@pytest.mark.parametrize("resume, tail", [(True, ["--last"]), ("last", ["--last"]), ("session-name", ["--", "session-name"])])
def test_resume_native_session_selection(tmp_path, resume, tail):
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"):
        command = chat_cli.chat_command(args(resume=resume), tmp_path)
    assert command[:2] == ["/native/codex", "resume"]
    assert command[-len(tail):] == tail


def test_resume_last_prompt_requires_explicit_session_for_initial_followup(tmp_path):
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"), pytest.raises(ValueError, match="after the session opens"):
        chat_cli.chat_command(args(resume="last", prompt="continue"), tmp_path)


def test_resume_picker_does_not_misinterpret_prompt_as_session(tmp_path):
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"):
        command = chat_cli.chat_command(args(resume="pick"), tmp_path)
        assert "resume" in command and "--last" not in command
        with pytest.raises(ValueError, match="after selecting"):
            chat_cli.chat_command(args(resume="pick", prompt="continue"), tmp_path)


def test_nonterminal_and_missing_native_cli_do_not_launch(tmp_path):
    with patch.object(chat_cli.shutil, "which", return_value=None), pytest.raises(ValueError, match="not installed"):
        chat_cli.chat_command(args(), tmp_path)
    with patch.object(chat_cli.shutil, "which", return_value="/native/codex"), \
            patch.object(chat_cli.sys.stdin, "isatty", return_value=False), \
            patch.object(chat_cli.subprocess, "run", side_effect=AssertionError("must not launch")), \
            pytest.raises(ValueError, match="needs a terminal"):
        chat_cli.launch_chat(args(), tmp_path)
