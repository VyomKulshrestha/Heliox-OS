"""API Integration Layer — connect to REST APIs and popular services.

Make HTTP requests, call GitHub/Slack/Discord APIs, send emails,
and interact with custom webhooks — all from natural language.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pilot.actions import SUPPORTED_HTTP_METHODS, OutputWithNotice
from pilot.security.secret_urls import mask_if_secret, refuse_masked
from pilot.system import outbound
from pilot.system.mail_readback import message_id_note, new_message_id

logger = logging.getLogger("pilot.system.api_client")


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port


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
        # What was found, said so a repair round does not ask the same thing again: a page that is not there, or a
        # refusal, gives the same answer to the same request.
        hint = _STATUS_HINTS.get(status, "")
        if hint:
            message += f" ({hint})"
        super().__init__(f"{message}: {excerpt}" if excerpt else message)
        self.status = status
        self.excerpt = excerpt

    @classmethod
    def with_reason(cls, error: HttpStatusError, text: str) -> HttpStatusError:
        """The same failure with a plain sentence for the person in place of the service's raw answer."""
        replaced = cls.__new__(cls)
        RuntimeError.__init__(replaced, text)
        replaced.status = error.status
        replaced.excerpt = error.excerpt
        return replaced


def _connection_refused(error: HttpStatusError, service_name: str, what: str) -> HttpStatusError:
    """A 401, 403 or 404 from a chat webhook or GitHub means the stored secret no longer works: say where to fix it."""
    from pilot.system import connections

    if error.status not in {401, 403, 404, 410}:
        return error
    return HttpStatusError.with_reason(
        error,
        f"{connections.TITLE[service_name]} refused the {what} (HTTP {error.status}), so nothing was sent: it was "
        f"removed, revoked or has expired. Replace it in {connections.WHERE[service_name]}."
        + (f" The service answered: {error.excerpt}" if error.excerpt else ""),
    )


def _page_found(soup: Any, size: int) -> str:
    """What a page that a selector missed does hold (``page_outline``), read from the parsed page."""
    title = soup.title.get_text() if soup.title else ""
    ids = [str(tag.get("id")) for tag in soup.find_all(id=True)][:200]
    classes = [str(name) for tag in soup.find_all(class_=True) for name in (tag.get("class") or [])]
    return page_outline(title, ids, classes, size)


def page_outline(title: str, ids: list[str], classes: list[str], size: int) -> str:
    """What a page that a selector missed does hold, in a sentence a planner can choose another selector from: its
    title and the most used element ids and classes. Never the page's text."""
    from collections import Counter

    title = " ".join(str(title or "").split())[:80]
    named = [f"#{name}" for name in ids[:5]] + [f".{name}" for name, _count in Counter(classes).most_common(8)]
    parts = [f"The page loaded ({size:,} characters"]
    parts.append(f", title {title!r})" if title else ")")
    sentence = "".join(parts) + "."
    if named:
        sentence += f" It has elements such as {', '.join(named)}."
    else:
        sentence += " It has no element ids or classes; its content may be drawn by a script, so try the whole text."
    return sentence


_STATUS_HINTS = {
    404: "the server has nothing at this address; the same request will get the same answer",
    410: "the server says this address is gone for good; the same request will get the same answer",
    401: "the server wants a sign-in this request did not carry; sending it again unchanged will be refused again",
    403: "the server refused this request; sending it again unchanged will be refused again",
}


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
    for name, value in (headers or {}).items():
        # A resumed plan whose header secret could not be restored holds its masked form: never send that.
        refuse_masked(str(value), what=f"{name} header")
    json_body = body if isinstance(body, dict) and body else None
    content = body if body and not isinstance(body, dict) else None
    sent_headers = dict(headers or {})
    if content is not None:
        sent_headers["Content-Type"] = "application/json"
    resp = await outbound.fetch(
        method,
        url,
        allow_private_network=allow_private_network,
        headers=sent_headers or None,
        params=params,
        json=json_body,
        content=content,
        auth=auth,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise HttpStatusError(method, str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""), resp.text)

    result = {
        "status": resp.status_code,
        "url": mask_if_secret(str(resp.url)),
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
    The token comes from the GitHub connection (Settings, else ``GITHUB_TOKEN``; pilot.system.connections). A token
    given here is used as given; one hidden when the plan was kept is replaced by the connected token when they
    match. Without any token a GET goes out unauthenticated (public data only) and any other method fails, saying
    where to connect GitHub.
    """
    from pilot.system import connections

    stored = (await connections.service("github")).get(connections.GITHUB_TOKEN)
    token = connections.choose(token, stored, service_name="github", what="GitHub token")
    if not token and method.upper() != "GET":
        raise ValueError(f"Nothing was sent to GitHub: {connections.not_connected('github')}")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com{endpoint}" if endpoint.startswith("/") else endpoint
    if token and urlsplit(url).hostname != "api.github.com":
        # The connected token belongs to api.github.com only; a full URL elsewhere never receives it.
        raise ValueError("Nothing was sent: api_github calls api.github.com only. Use api_request for other hosts.")

    try:
        return await api_request(method, url, headers=headers, body=body)
    except HttpStatusError as exc:
        if exc.status == 401 and token:
            raise _connection_refused(exc, "github", "token") from None
        if exc.status in {401, 403, 404} and not token:
            raise HttpStatusError.with_reason(
                exc,
                f"{exc}. GitHub answers this way for private data when no token is sent. "
                f"{connections.not_connected('github')}",
            ) from None
        raise


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

    Without explicit credentials it sends from the one mail account (pilot.system.connections.mail_account): the
    Email settings with the app password from the credential store, else ``SMTP_SERVER``, ``SMTP_PORT``,
    ``SMTP_USER`` and ``SMTP_PASS``. The email agent sends from the same account.
    """
    from pilot.system import connections

    source = "the given account"
    if not username or not password:
        account = await connections.mail_account()
        if account is None:
            raise ValueError(
                f"The email was not sent: no mail account is connected. Set up the account in {connections.WHERE['mail']}."
            )
        smtp_server, smtp_port = account.host, account.port
        username, password = account.username, account.password
        source = connections.WHERE["mail"] if account.source == connections.SETTINGS else "SMTP_USER and SMTP_PASS"
    smtp_server = smtp_server or "smtp.gmail.com"
    from_addr = from_addr or username

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to
    # The verifier finds the sender's copy in Sent by this Message-ID (pilot.system.mail_readback).
    message_id = new_message_id(from_addr)
    msg["Message-ID"] = message_id
    msg["Date"] = email.utils.formatdate(localtime=True)

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

    from pilot.system.system_guard import check_mail

    check_mail(smtp_server, smtplib.SMTP)
    try:
        await asyncio.to_thread(_do)
    except Exception as e:
        if "refused" not in accepted:
            if isinstance(e, smtplib.SMTPAuthenticationError):
                raise RuntimeError(
                    f"The email to {to} was not sent: the mail server refused the sign-in for {username}. Check the "
                    f"app password in {source}."
                ) from None
            raise RuntimeError(f"The email to {to} was not sent: {e}") from e
        # The server already took the message; only closing the connection failed. Reporting a failure
        # here would invite a retry that sends it twice.
        logger.warning("SMTP connection closed with an error after the message was accepted: %s", type(e).__name__)
    refused = accepted.get("refused") or {}
    if refused:
        # A success for the accepted recipients (failing would invite a resend to them); the refused ones go
        # to the person through ActionResult.notice, into the reply and the task receipt.
        notice = f"The mail server refused the email for: {', '.join(sorted(refused))}"
        return OutputWithNotice(f"Email sent to {to}: {subject}{message_id_note(message_id)}. {notice}.", notice)
    return f"Email sent to {to}: {subject}{message_id_note(message_id)}"


# ── Webhook ──────────────────────────────────────────────────────────


async def send_webhook(
    url: str,
    payload: dict,
    method: str = "POST",
    headers: dict[str, str] | None = None,
) -> str:
    """Send a webhook/callback to any URL."""
    return await api_request(method, url, headers=headers, body=payload)


def _with_query(url: str, **values: str) -> str:
    """The URL with these query values set, keeping the others (Discord's thread_id)."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(values)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _chat_receipt(answer: str, webhook_url: str, sent_at: float) -> str:
    """What a chat webhook answered, without the webhook URL or the response headers.

    A webhook URL carries its secret token in the path, so the receipt names the host only. It keeps the
    status, the send time and, for Discord, the id of the message created, which the post-condition check
    uses to read the message back.
    """
    try:
        parsed = json.loads(answer)
    except (TypeError, ValueError):
        parsed = {}
    body = parsed.get("body") if isinstance(parsed, dict) else None
    receipt: dict[str, Any] = {
        "status": parsed.get("status") if isinstance(parsed, dict) else None,
        "host": _origin(webhook_url)[1],
        "sent_at": round(sent_at, 3),
    }
    if isinstance(body, dict):
        for key, name in (("id", "message_id"), ("channel_id", "channel_id"), ("ok", "ok"), ("error", "error")):
            if body.get(key) is not None:
                receipt[name] = body[key]
    elif body is not None:
        receipt["body"] = str(body)[:200]
    return json.dumps(receipt, ensure_ascii=False)


# ── Slack (Webhook) ──────────────────────────────────────────────────


async def send_slack_message(
    message: str,
    webhook_url: str | None = None,
    channel: str | None = None,
) -> str:
    """Send a message to Slack via webhook.

    The webhook comes from the Slack connection (Settings, else ``SLACK_WEBHOOK_URL``; pilot.system.connections)
    when the plan gives none. Returns a receipt (status, host, send time, Slack's answer) without the webhook URL.
    An incoming webhook gives no message id; the post-condition check reads the message back only when a bot token
    and channel ID are connected too (see pilot.agents.postconditions.messaging).
    """
    from pilot.system import connections

    stored = (await connections.service("slack")).get(connections.SLACK_WEBHOOK)
    webhook_url = connections.choose(webhook_url, stored, service_name="slack", what="Slack webhook address")
    if not webhook_url:
        raise ValueError(f"The Slack message was not sent: {connections.not_connected('slack')}")

    payload: dict[str, str] = {"text": message}
    if channel:
        payload["channel"] = channel

    sent_at = time.time()
    try:
        answer = await api_request("POST", webhook_url, body=payload)
    except HttpStatusError as exc:
        raise _connection_refused(exc, "slack", "webhook address") from None
    return _chat_receipt(answer, webhook_url, sent_at)


# ── Discord (Webhook) ───────────────────────────────────────────────


async def send_discord_message(
    message: str,
    webhook_url: str | None = None,
) -> str:
    """Send a message to Discord via webhook.

    The webhook comes from the Discord connection (Settings, else ``DISCORD_WEBHOOK_URL``) when the plan gives
    none. Returns a receipt (status, host, send time and the id of the message Discord created) without the
    webhook URL.
    """
    from pilot.system import connections

    stored = (await connections.service("discord")).get(connections.DISCORD_WEBHOOK)
    webhook_url = connections.choose(webhook_url, stored, service_name="discord", what="Discord webhook address")
    if not webhook_url:
        raise ValueError(f"The Discord message was not sent: {connections.not_connected('discord')}")

    sent_at = time.time()
    # wait=true makes Discord answer with the message it created, so its id can be read back later.
    try:
        answer = await api_request("POST", _with_query(webhook_url, wait="true"), body={"content": message})
    except HttpStatusError as exc:
        raise _connection_refused(exc, "discord", "webhook address") from None
    return _chat_receipt(answer, webhook_url, sent_at)


# ── Web Scraping ─────────────────────────────────────────────────────


SCRAPE_CHAR_LIMIT = 10_000
SCRAPE_ELEMENT_LIMIT = 50
SCRAPE_LINK_LIMIT = 100
SCRAPE_TABLE_LIMIT = 10


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
    resp = await outbound.fetch(
        "GET",
        url,
        allow_private_network=allow_private_network,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Pilot/0.3)"},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise HttpStatusError("GET", str(resp.url), resp.status_code, getattr(resp, "reason_phrase", ""))
    html = resp.text

    from urllib.parse import urlsplit

    from pilot.system.read_limits import with_cut

    if extract == "html" and not selector:
        return with_cut(html[:SCRAPE_CHAR_LIMIT], min(len(html), SCRAPE_CHAR_LIMIT), len(html), "characters of HTML")

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        if selector and extract in ("text", "html"):
            elements = soup.select(selector)
            if not elements:
                host = urlsplit(url).hostname or "the page"
                raise LookupError(
                    f"No element matches the selector {selector!r} on {host}. {_page_found(soup, len(html))} "
                    "Asking for the same selector again will find nothing again."
                )
            shown = elements[:SCRAPE_ELEMENT_LIMIT]
            if extract == "text":
                joined = "\n".join(el.get_text(strip=True) for el in shown)
            else:
                joined = "\n".join(str(el) for el in shown)
            return with_cut(joined, len(shown), len(elements), "matching elements")

        if extract == "links":
            anchors = soup.find_all("a", href=True)
            links = []
            for a in anchors[:SCRAPE_LINK_LIMIT]:
                links.append({"text": a.get_text(strip=True)[:100], "href": a["href"]})
            return with_cut(json.dumps(links, indent=2), len(links), len(anchors), "links")

        if extract == "tables":
            found = soup.find_all("table")
            tables = []
            for table in found[:SCRAPE_TABLE_LIMIT]:
                rows = []
                for tr in table.find_all("tr"):
                    cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                    rows.append(cells)
                tables.append(rows)
            return with_cut(json.dumps(tables, indent=2), len(tables), len(found), "tables")

        text = soup.get_text(separator="\n", strip=True)
        return with_cut(text[:SCRAPE_CHAR_LIMIT], min(len(text), SCRAPE_CHAR_LIMIT), len(text), "characters of text")

    except ImportError:
        # Fallback without BeautifulSoup: plain text only, so a selector or a links or tables request cannot be met.
        import re

        if selector or extract in ("links", "tables", "html"):
            raise RuntimeError(
                "Extracting by selector, links, tables or HTML needs BeautifulSoup (pip install beautifulsoup4)"
            ) from None
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return with_cut(text[:SCRAPE_CHAR_LIMIT], min(len(text), SCRAPE_CHAR_LIMIT), len(text), "characters of text")
