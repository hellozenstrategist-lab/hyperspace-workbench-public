import pytest
from astra_harness.hyperspace_backend import HyperspaceBackend


@pytest.mark.parametrize("endpoint", ["8.8.8.8:50051", "example.com:50051", "0.0.0.0:50051", "http://127.0.0.1:50051", "user:password@127.0.0.1:50051", "127.0.0.1:50051/path", "127.0.0.1:50051?query", "127.0.0.1:50051#fragment"])
def test_backend_rejects_nonlocal_or_ambiguous_endpoints_before_opening_state(endpoint, tmp_path):
    state = tmp_path / "must-not-exist.sqlite"
    with pytest.raises(ValueError):
        HyperspaceBackend(endpoint, state_path=state)
    assert not state.exists()


def test_backend_refuses_crosshost_http_setup(tmp_path):
    with pytest.raises(ValueError, match="same local host"):
        HyperspaceBackend("127.0.0.1:50051", state_path=tmp_path / "unused.sqlite", http_endpoint="http://192.168.1.1:50050")
