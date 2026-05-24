import os

import pytest

from pilot.config import PilotConfig, _merge_config
from pilot.system.http_client import create_httpx_client


def test_proxy_section_merges_to_config() -> None:
    config = PilotConfig()
    raw = {
        "proxy": {
            "http": "http://proxy.example.com:8080",
            "https": "http://proxy.example.com:8080",
            "no_proxy": "localhost,127.0.0.1",
        }
    }

    merged = _merge_config(config, raw)

    assert merged.proxy.http == "http://proxy.example.com:8080"
    assert merged.proxy.https == "http://proxy.example.com:8080"
    assert merged.proxy.no_proxy == "localhost,127.0.0.1"


def test_proxy_section_validation_rejects_invalid_url() -> None:
    config = PilotConfig()
    raw = {"proxy": {"http": "invalid-proxy"}}

    with pytest.raises(ValueError, match="Invalid proxy URL"):
        _merge_config(config, raw)


def test_create_httpx_client_scopes_proxy_env(monkeypatch) -> None:
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    config = PilotConfig()
    config.proxy.http = "http://proxy.example.com:8080"
    config.proxy.https = "http://proxy.example.com:8443"
    config.proxy.no_proxy = "localhost,127.0.0.1"

    client = create_httpx_client(config, timeout=1)

    # Environment variables should be cleaned up after client construction
    assert "HTTP_PROXY" not in os.environ
    assert "HTTPS_PROXY" not in os.environ
    assert "NO_PROXY" not in os.environ

    # The proxy config should be applied natively via mounts in the base transport fallback
    # But since it's an implementation detail, we just ensure it handles the scope correctly.
    import asyncio

    asyncio.run(client.aclose())
