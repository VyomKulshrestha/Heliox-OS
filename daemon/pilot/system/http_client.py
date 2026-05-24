"""Shared HTTP client helpers with proxy support."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator
from typing import Any

import httpx

from pilot.config import PilotConfig


@contextlib.contextmanager
def _scoped_proxy_env(config: PilotConfig) -> Generator[None, None, None]:
    """Temporarily apply proxy values from config into environment variables."""
    keys = ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"]
    original_env = {key: os.environ.get(key) for key in keys}

    if config.proxy.http is not None:
        os.environ["HTTP_PROXY"] = config.proxy.http
        os.environ["http_proxy"] = config.proxy.http
    if config.proxy.https is not None:
        os.environ["HTTPS_PROXY"] = config.proxy.https
        os.environ["https_proxy"] = config.proxy.https
    if config.proxy.no_proxy is not None:
        os.environ["NO_PROXY"] = config.proxy.no_proxy
        os.environ["no_proxy"] = config.proxy.no_proxy

    try:
        yield
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def create_httpx_client(config: PilotConfig, **kwargs: Any) -> httpx.AsyncClient:
    """Construct an httpx AsyncClient with proxy and env fallback support."""
    if "trust_env" not in kwargs:
        kwargs["trust_env"] = True

    with _scoped_proxy_env(config):
        return httpx.AsyncClient(**kwargs)
