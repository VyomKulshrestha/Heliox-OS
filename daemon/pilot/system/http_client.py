"""Shared HTTP client helpers with proxy support."""

from __future__ import annotations

import os
from typing import Any

import httpx

from pilot.config import PilotConfig


def _apply_proxy_env(config: PilotConfig) -> None:
    """Apply proxy values from config into environment variables."""
    if config.proxy.http is not None:
        os.environ["HTTP_PROXY"] = config.proxy.http
        os.environ["http_proxy"] = config.proxy.http
    if config.proxy.https is not None:
        os.environ["HTTPS_PROXY"] = config.proxy.https
        os.environ["https_proxy"] = config.proxy.https
    if config.proxy.no_proxy is not None:
        os.environ["NO_PROXY"] = config.proxy.no_proxy
        os.environ["no_proxy"] = config.proxy.no_proxy


def _build_proxy_map(config: PilotConfig) -> dict[str, str] | None:
    proxies: dict[str, str] = {}
    if config.proxy.http:
        proxies["http://"] = config.proxy.http
    if config.proxy.https:
        proxies["https://"] = config.proxy.https
    return proxies or None


def create_httpx_client(config: PilotConfig, **kwargs: Any) -> httpx.AsyncClient:
    """Construct an httpx AsyncClient with proxy and env fallback support."""
    _apply_proxy_env(config)
    if "trust_env" not in kwargs:
        kwargs["trust_env"] = True

    proxies = _build_proxy_map(config)
    if proxies is not None:
        kwargs["proxies"] = proxies

    return httpx.AsyncClient(**kwargs)
