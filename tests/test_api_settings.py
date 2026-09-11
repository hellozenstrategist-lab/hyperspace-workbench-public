import json
import stat
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from astra_harness import api_settings as api


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = tmp_path / "private" / "openrouter.json"
    monkeypatch.setenv("HYPERSPACE_HARNESS_CONFIG", str(path))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    return path


def test_hidden_setup_private_permissions_and_redacted_status(settings_path):
    with patch.object(api.sys.stdin, "isatty", return_value=True), patch.object(api.getpass, "getpass", return_value="fake-test-key"):
        result = api.api_setup("provider/model")
    assert json.loads(settings_path.read_text())["api_key"] == "fake-test-key"
    assert stat.S_IMODE(settings_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(settings_path.parent.stat().st_mode) == 0o700
    assert result == {"model": "provider/model", "key_configured": True, "key_source": "saved", "model_source": "saved"}
    assert "fake-test-key" not in json.dumps(api.api_status())


def test_noninteractive_setup_requires_key_and_does_not_create_file(settings_path):
    with patch.object(api.sys.stdin, "isatty", return_value=False), pytest.raises(ValueError, match="interactive terminal"):
        api.api_setup()
    assert not settings_path.exists()


def test_bare_interactive_setup_prompts_for_key_and_model(settings_path):
    with patch.object(api.sys.stdin, "isatty", return_value=True), \
            patch.object(api.getpass, "getpass", return_value="fake-test-key"), \
            patch("builtins.input", return_value="provider/model"):
        result = api.api_setup()
    assert result["model"] == "provider/model"
    assert result["key_configured"] is True


def test_model_update_reuses_saved_key_without_prompt(settings_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-saved-key")
    api.api_setup("provider/old")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-environment-override")
    with patch.object(api.getpass, "getpass", side_effect=AssertionError("must not prompt")):
        result = api.api_set_model("provider/new")
    assert json.loads(settings_path.read_text())["api_key"] == "fake-saved-key"
    assert api.openrouter_settings()["api_key"] == "fake-environment-override"
    assert result["model"] == "provider/new"
    assert result["key_source"] == "environment"
    monkeypatch.setenv("OPENROUTER_MODEL", "provider/env")
    assert api.api_status()["model"] == "provider/env"


def test_model_can_be_configured_before_initial_key(settings_path):
    result = api.api_set_model("provider/model")
    assert result["key_configured"] is False
    assert "api_key" not in json.loads(settings_path.read_text())


@pytest.mark.parametrize("model", ["", "model", "/model", "provider/", "provider/ bad", "provider/\nmodel", None])
def test_invalid_models_do_not_write(settings_path, model):
    with pytest.raises(ValueError):
        api.api_set_model(model)
    assert not settings_path.exists()


def test_insecure_file_and_symlink_refused(settings_path, tmp_path):
    api.api_set_model("provider/model")
    settings_path.chmod(0o644)
    with pytest.raises(ValueError, match="600"):
        api.api_status()
    settings_path.chmod(0o600)
    saved = settings_path.read_bytes()
    settings_path.unlink()
    target = tmp_path / "target.json"
    target.write_bytes(saved)
    target.chmod(0o600)
    settings_path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        api.api_set_model("provider/changed")
    assert target.read_bytes() == saved


def test_insecure_directory_refused(settings_path):
    api.api_set_model("provider/model")
    settings_path.parent.chmod(0o755)
    with pytest.raises(ValueError, match="700"):
        api.api_status()


def test_key_check_has_fixed_destination_and_redacts_response(settings_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-secret-check")
    response = type("Response", (), {"status": 200, "__enter__": lambda self: self, "__exit__": lambda *args: None})()
    with patch.object(api, "urlopen", return_value=response) as opened:
        result = api.api_check()
    request = opened.call_args.args[0]
    assert request.full_url == "https://openrouter.ai/api/v1/key"
    assert request.get_header("Authorization") == "Bearer fake-secret-check"
    assert opened.call_args.kwargs["timeout"] == 15
    assert result == {"key_valid": True, "connection": "reached", "http_status": 200}
    assert "fake-secret-check" not in json.dumps(result)


@pytest.mark.parametrize("error, expected", [(HTTPError("redacted", 401, "fake-secret", {}, None), False),
                                            (HTTPError("redacted", 429, "fake-secret", {}, None), None),
                                            (URLError("fake-secret"), None)])
def test_key_check_failure_never_returns_exception_text(settings_path, monkeypatch, error, expected):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-secret")
    with patch.object(api, "urlopen", side_effect=error):
        result = api.api_check()
    assert result["key_valid"] is expected
    assert "fake-secret" not in json.dumps(result)


def test_missing_key_check_has_no_network_call(settings_path):
    with patch.object(api, "urlopen", side_effect=AssertionError("must not call")), pytest.raises(ValueError, match="api setup"):
        api.api_check()


def test_account_check_never_follows_redirect_with_credential():
    assert api._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.com") is None
