"""API Integration Layer — connect to REST APIs and popular services.

Make HTTP requests, call GitHub/Slack/Discord APIs, send emails,
and interact with custom webhooks — all from natural language.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger("pilot.system.api_client")


# ── Generic REST API ─────────────────────────────────────────────────


async def api_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict | str | None = None,
    params: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
    timeout: int = 30,
) -> str:
    """Make a generic HTTP request.

    method: GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS
    """
    method = method.upper()
    last_error = None

    # Try with SSL verification first, fallback without if cert issues
    for verify in (True, False):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, verify=verify) as client:
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

                resp = await client.request(method, url, **kwargs)

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

        except Exception as e:
            last_error = e
            if verify and "CERTIFICATE_VERIFY_FAILED" in str(e):
                continue  # Retry without SSL
            raise

    raise last_error  # type: ignore[misc]


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
        return "ERROR: SMTP credentials not configured. Set SMTP_USER and SMTP_PASS environment variables."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to

    if html:
        msg.attach(MIMEText(body, "html"))
    else:
        msg.attach(MIMEText(body, "plain"))

    def _do():
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(username, password)
            server.sendmail(from_addr, to, msg.as_string())

    try:
        await asyncio.to_thread(_do)
        return f"Email sent to {to}: {subject}"
    except Exception as e:
        return f"Failed to send email: {e}"


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
        return "ERROR: Slack webhook URL not configured. Set SLACK_WEBHOOK_URL."

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
        return "ERROR: Discord webhook URL not configured. Set DISCORD_WEBHOOK_URL."

    return await api_request("POST", webhook_url, body={"content": message})


# ── Web Scraping ─────────────────────────────────────────────────────


async def scrape_url(
    url: str,
    selector: str | None = None,
    extract: str = "text",
) -> str:
    """Fetch a URL and extract content.

    selector: CSS selector to target specific elements (requires BeautifulSoup)
    extract: 'text', 'html', 'links', 'tables'
    """
    async with httpx.AsyncClient(
        timeout=30, follow_redirects=True, verify=False, headers={"User-Agent": "Mozilla/5.0 (compatible; Pilot/0.3)"}
    ) as client:
        resp = await client.get(url)
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
