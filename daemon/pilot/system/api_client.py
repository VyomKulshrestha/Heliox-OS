"""API Integration Layer — connect to REST APIs and popular services.

Make HTTP requests, call GitHub/Slack/Discord APIs, send emails,
and interact with custom webhooks — all from natural language.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import smtplib
import socket
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from urllib.parse import urljoin, urlsplit

from pilot.actions import SUPPORTED_HTTP_METHODS
from pilot.config import PilotConfig
from pilot.system.http_client import create_httpx_client

logger = logging.getLogger("pilot.system.api_client")

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SENSITIVE_REDIRECT_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization"})
_BODY_HEADERS = frozenset({"content-length", "content-type", "transfer-encoding"})
_MAX_REDIRECTS = 5


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


def _is_public_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


async def _validate_outbound_url(url: str, *, allow_private_network: bool) -> None:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Outbound URLs must use http:// or https://")
    if not parsed.hostname:
        raise ValueError("Outbound URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials must be supplied as request authentication, not embedded in the URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Outbound URL contains an invalid port") from exc

    if allow_private_network:
        return

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Private-network URL blocked; explicit approval is required")

    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        literal_address = None
    if literal_address is not None:
        if not literal_address.is_global:
            raise ValueError("Private-network URL blocked; explicit approval is required")
        return

    try:
        resolved = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            port or (443 if parsed.scheme.lower() == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve outbound hostname {hostname!r}") from exc

    addresses = {entry[4][0] for entry in resolved}
    if not addresses or any(not _is_public_address(address) for address in addresses):
        raise ValueError("Private-network URL blocked; explicit approval is required")


def _redirect_method(method: str, status_code: int) -> str:
    if status_code == 303 and method != "HEAD":
        return "GET"
    if status_code in {301, 302} and method == "POST":
        return "GET"
    return method


async def _request_with_validated_redirects(
    client: Any,
    method: str,
    url: str,
    *,
    allow_private_network: bool,
    request_kwargs: dict[str, Any] | None = None,
) -> Any:
    current_method = method
    current_url = url
    kwargs = dict(request_kwargs or {})
    if "headers" in kwargs:
        kwargs["headers"] = dict(kwargs["headers"])

    for redirect_count in range(_MAX_REDIRECTS + 1):
        await _validate_outbound_url(current_url, allow_private_network=allow_private_network)
        response = await client.request(current_method, current_url, **kwargs)
        location = response.headers.get("location")
        if response.status_code not in _REDIRECT_STATUSES or not location:
            return response
        if redirect_count == _MAX_REDIRECTS:
            raise ValueError(f"Outbound request exceeded {_MAX_REDIRECTS} redirects")

        next_url = urljoin(str(response.url), location)
        if _origin(next_url) != _origin(current_url):
            kwargs.pop("auth", None)
            headers = kwargs.get("headers")
            if headers:
                kwargs["headers"] = {
                    key: value for key, value in headers.items() if key.lower() not in _SENSITIVE_REDIRECT_HEADERS
                }

        next_method = _redirect_method(current_method, response.status_code)
        if next_method != current_method:
            kwargs.pop("content", None)
            kwargs.pop("json", None)
            headers = kwargs.get("headers")
            if headers:
                kwargs["headers"] = {key: value for key, value in headers.items() if key.lower() not in _BODY_HEADERS}

        kwargs.pop("params", None)
        current_method = next_method
        current_url = next_url

    raise AssertionError("redirect loop must return or raise")


# ── Generic REST API ─────────────────────────────────────────────────


class HttpStatusError(RuntimeError):
    """The server answered with HTTP 4xx or 5xx, so the request did not do what was asked.

    The message names only the scheme and host: a webhook URL carries its secret in the path.
    """

    def __init__(self, method: str, url: str, status: int, reason: str = "", body: str = "") -> None:
        scheme, host, _port = _origin(url)
        excerpt = " ".join((body or "").split())[:300]
        reason = f" {reason}" if reason else ""
        message = f"{method} to {scheme}://{host} failed with HTTP {status}{reason}"
        super().__init__(f"{message}: {excerpt}" if excerpt else message)
        self.status = status


async def api_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict | str | None = None,
    params: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
    timeout: int = 30,
    allow_private_network: bool = False,
) -> str:
    """Make a generic HTTP request.

    method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
    """
    method = method.upper()
    if method not in SUPPORTED_HTTP_METHODS:
        supported = ", ".join(sorted(SUPPORTED_HTTP_METHODS))
        raise ValueError(f"Unsupported HTTP method {method!r}; expected one of: {supported}")
    async with create_httpx_client(PilotConfig.load(), timeout=timeout, follow_redirects=False, verify=True) as client:
        kwargs: dict[str, Any] = {"params": params}
        if headers:
            kwargs["headers"] = headers
        if auth:
            kwargs["auth"] = auth
        if body:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["content"] = body
                kwargs.setdefault("headers", {})["Content-Type"] = "application/json"

        resp = await _request_with_validated_redirects(
            client,
            method,
            url,
            allow_private_network=allow_private_network,
            request_kwargs=kwargs,
        )

        if resp.status_code >= 400:
            raise HttpStatusError(
                method, str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""), resp.text
            )

        result = {
            "status": resp.status_code,
            "url": str(resp.url),
            "headers": dict(resp.headers),
        }

        # Parse response body
        try:
            result["body"] = resp.json()
        except (json.JSONDecodeError, ValueError):
            text = resp.text
            if len(text) > 5000:
                text = text[:5000] + f"... ({len(resp.text):,} chars total)"
            result["body"] = text

        return json.dumps(result, indent=2, ensure_ascii=False)


# ── GitHub API ───────────────────────────────────────────────────────


async def github_api(
    endpoint: str,
    method: str = "GET",
    body: dict | None = None,
    token: str | None = None,
) -> str:
    """Call the GitHub API.

    endpoint: e.g., '/user/repos', '/repos/owner/repo/issues'
    Uses GITHUB_TOKEN env var if token not provided.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else endpoint

    return await api_request(method, url, headers=headers, body=body)


async def github_list_repos(username: str | None = None) -> str:
    """List repositories for a user or the authenticated user."""
    if username:
        return await github_api(f"/users/{username}/repos?per_page=30&sort=updated")
    return await github_api("/user/repos?per_page=30&sort=updated")


async def github_create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str = "",
    labels: list[str] | None = None,
) -> str:
    """Create a GitHub issue."""
    payload: dict[str, Any] = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    return await github_api(
        f"/repos/{owner}/{repo}/issues",
        method="POST",
        body=payload,
    )


async def github_list_prs(owner: str, repo: str, state: str = "open") -> str:
    """List pull requests for a repository."""
    return await github_api(f"/repos/{owner}/{repo}/pulls?state={state}&per_page=30")


# ── Email ────────────────────────────────────────────────────────────


async def send_email(
    to: str,
    subject: str,
    body: str,
    smtp_server: str | None = None,
    smtp_port: int = 587,
    username: str | None = None,
    password: str | None = None,
    from_addr: str | None = None,
    html: bool = False,
) -> str:
    """Send an email via SMTP.

    Uses environment variables SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS
    if not provided.
    """
    smtp_server = smtp_server or os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", str(smtp_port)))
    username = username or os.environ.get("SMTP_USER", "")
    password = password or os.environ.get("SMTP_PASS", "")
    from_addr = from_addr or username

    if not username or not password:
        raise ValueError(
            "The email was not sent: SMTP credentials are not configured. Set SMTP_USER and SMTP_PASS for the daemon."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to

    if html:
        msg.attach(MIMEText(body, "html"))
    else:
        msg.attach(MIMEText(body, "plain"))

    accepted: dict[str, Any] = {}

    def _do() -> None:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(username, password)
            accepted["refused"] = server.sendmail(from_addr, to, msg.as_string())

    try:
        await asyncio.to_thread(_do)
    except Exception as e:
        if "refused" not in accepted:
            raise RuntimeError(f"The email to {to} was not sent: {e}") from e
        # The server already took the message; only closing the connection failed. Reporting a failure
        # here would invite a retry that sends it twice.
        logger.warning("SMTP connection closed with an error after the message was accepted: %s", type(e).__name__)
    refused = accepted.get("refused") or {}
    if refused:
        return f"Email sent to {to}: {subject} (the server refused: {', '.join(sorted(refused))})"
    return f"Email sent to {to}: {subject}"


# ── Webhook ──────────────────────────────────────────────────────────


async def send_webhook(
    url: str,
    payload: dict,
    method: str = "POST",
    headers: dict[str, str] | None = None,
) -> str:
    """Send a webhook/callback to any URL."""
    return await api_request(method, url, headers=headers, body=payload)


# ── Slack (Webhook) ──────────────────────────────────────────────────


async def send_slack_message(
    message: str,
    webhook_url: str | None = None,
    channel: str | None = None,
) -> str:
    """Send a message to Slack via webhook.

    Uses SLACK_WEBHOOK_URL env var if not provided.
    """
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        raise ValueError(
            "The Slack message was not sent: no webhook URL is configured. Set SLACK_WEBHOOK_URL or give webhook_url."
        )

    payload: dict[str, str] = {"text": message}
    if channel:
        payload["channel"] = channel

    return await api_request("POST", webhook_url, body=payload)


# ── Discord (Webhook) ───────────────────────────────────────────────


async def send_discord_message(
    message: str,
    webhook_url: str | None = None,
) -> str:
    """Send a message to Discord via webhook.

    Uses DISCORD_WEBHOOK_URL env var if not provided.
    """
    webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        raise ValueError(
            "The Discord message was not sent: no webhook URL is configured. "
            "Set DISCORD_WEBHOOK_URL or give webhook_url."
        )

    return await api_request("POST", webhook_url, body={"content": message})


# ── Web Scraping ─────────────────────────────────────────────────────


async def scrape_url(
    url: str,
    selector: str | None = None,
    extract: str = "text",
    *,
    allow_private_network: bool = False,
) -> str:
    """Fetch a URL and extract content.

    selector: CSS selector to target specific elements (requires BeautifulSoup)
    extract: 'text', 'html', 'links', 'tables'
    """
    async with create_httpx_client(
        PilotConfig.load(),
        timeout=30,
        follow_redirects=False,
        verify=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Pilot/0.3)"},
    ) as client:
        resp = await _request_with_validated_redirects(
            client,
            "GET",
            url,
            allow_private_network=allow_private_network,
        )
        if resp.status_code >= 400:
            raise HttpStatusError("GET", str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""))
        html = resp.text

    if extract == "html":
        return html[:10000]

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        if selector:
            elements = soup.select(selector)
            if extract == "text":
                return "\n".join(el.get_text(strip=True) for el in elements[:50])
            elif extract == "html":
                return "\n".join(str(el) for el in elements[:50])

        if extract == "links":
            links = []
            for a in soup.find_all("a", href=True)[:100]:
                links.append({"text": a.get_text(strip=True)[:100], "href": a["href"]})
            return json.dumps(links, indent=2)

        if extract == "tables":
            tables = []
            for table in soup.find_all("table")[:10]:
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    rows.append(cells)
                tables.append(rows)
            return json.dumps(tables, indent=2)

        text = soup.get_text(separator="\n", strip=True)
        return text[:10000]

    except ImportError:
        # Fallback without BeautifulSoup
        import re

        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:10000]
