"""CLI selection never starts models, Docker, or notifications in these tests."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from astra_harness.cli import build_parser, run_options, endpoint
from astra_harness.schema import worker_ids


def test_new_runs_default_to_five_with_notifications_off():
    args = build_parser().parse_args(["run"])
    assert run_options(args)["worker_count"] == 5
    assert args.photon == "off"


def test_existing_three_worker_run_keeps_count_and_explicit_override_is_visible():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "manifest.json").write_text(json.dumps({"worker_count": 3, "model": "example/default-model"}))
        args = build_parser().parse_args(["run", "--run-dir", directory])
        assert run_options(args, root)["worker_count"] == 3
        args.workers = 5
        assert run_options(args, root)["worker_count"] == 5


def test_openrouter_model_explicit_or_environment_and_no_key_in_options():
    parser = build_parser()
    with patch.dict(os.environ, {"OPENROUTER_MODEL": "example/research", "OPENROUTER_API_KEY": "test-secret"}):
        options = run_options(parser.parse_args(["run", "--provider", "openrouter", "--workers", "5"]))
        assert options == {"worker_count": 5, "provider": "openrouter", "model": "example/research"}
        assert "test-secret" not in json.dumps(options)
        options = run_options(parser.parse_args(["run", "--provider", "openrouter", "--model", "example/other"]))
        assert options["model"] == "example/other"
    # A developer's real saved settings must not supply this missing-model case.
    with patch.dict(os.environ, {}, clear=True), \
            patch("astra_harness.api_settings.openrouter_settings", return_value={"model": None}), \
            pytest.raises(ValueError, match="model"):
        run_options(parser.parse_args(["run", "--provider", "openrouter"]))


@pytest.mark.parametrize("count", [True, False, 0, 1, 6, 8, 9, 3.0, "5"])
def test_worker_bound_is_strict(count):
    with pytest.raises(ValueError):
        worker_ids(count)


def test_local_geometry_does_not_start_a_service():
    with patch("astra_harness.cli.subprocess.run", side_effect=AssertionError("No Docker")):
        assert endpoint("local") is None


def test_chat_and_api_commands_parse_without_model_calls():
    parser = build_parser()
    chat = parser.parse_args(["chat", "Set up the research API", "--resume", "last"])
    assert chat.prompt == "Set up the research API"
    assert chat.resume == "last"
    args = parser.parse_args(["api", "setup", "--model", "example/research"])
    assert args.api_command == "setup"
    assert args.model == "example/research"
    args = parser.parse_args(["api", "status"])
    assert args.api_command == "status"


def test_worker_cap_cannot_be_overridden_from_command_line():
    for count in ("6", "8", "100"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["run", "--workers", count])


def test_saved_api_setup_supplies_runtime_model_and_key_without_network(tmp_path):
    from astra_harness.api_settings import api_setup
    from astra_harness.runtime_factory import create_runtime
    settings_path = tmp_path / "private-settings" / "openrouter.json"
    environment = {"HYPERSPACE_HARNESS_CONFIG": str(settings_path),
                   "OPENROUTER_API_KEY": "dummy-key-for-local-tests", "OPENROUTER_MODEL": "example/research"}
    with patch.dict(os.environ, environment, clear=True):
        assert api_setup()["key_configured"]
    with patch.dict(os.environ, {"HYPERSPACE_HARNESS_CONFIG": str(settings_path)}, clear=True):
        options = run_options(build_parser().parse_args(["run", "--provider", "openrouter"]))
        assert options["model"] == "example/research"

        async def event(*_args, **_kwargs):
            pass

        async def verify():
            runtime = create_runtime(tmp_path / "run" / "runtime_state.json", tmp_path / "run" / "runtime",
                                     event, event, agents=worker_ids(5), provider="openrouter")
            try:
                with patch.object(runtime, "_http", side_effect=AssertionError("No network in local preflight")):
                    manifest = await runtime.start()
                assert manifest["worker_count"] == 5
                assert manifest["model"] == "example/research"
                assert "dummy-key-for-local-tests" not in runtime.state_path.read_text()
            finally:
                await runtime.close()

        asyncio.run(verify())
