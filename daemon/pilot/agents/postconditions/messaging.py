"""Post-conditions for chat messages sent through Slack and Discord webhooks.

Both sends are irreversible: a MISMATCH stops the run for the person to look (``server.py`` never repeats an
irreversible action after a failed post-condition), so a message is never sent twice by this check. Only an
observed difference is a MISMATCH: a read path that cannot look (no token, no network, an odd answer) is reported
as not verified with its reason and never fails the send. Nothing here raises on purpose.

Slack: an incoming webhook answers ``ok`` and gives no message id. That answer is a receipt, not a read-back,
and is reported as such. When a bot token and a channel id are connected (Settings > Connections > Slack, else
``SLACK_BOT_TOKEN`` and ``SLACK_CHANNEL_ID``; pilot.system.connections; the token needs the ``channels:history``
scope, or ``groups:history`` for a private channel, and the app must be a member of the channel), the check reads
the channel with
``conversations.history`` from the send time and looks for the text.

Discord: the handler posts with ``?wait=true``, so Discord answers with the message it created. The check reads
that message back with ``GET /webhooks/{id}/{token}/messages/{message_id}``.

A webhook URL holds its secret token in the path. No detail or log line here contains it: only the host.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from pilot.actions import ActionResult, ActionType, ApiRequestParams
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.agents.unconfirmed import Kind
from pilot.system import connections

logger = logging.getLogger("pilot.agents.postconditions.messaging")

SLACK_HISTORY_URL = "https://slack.com/api/conversations.history"
# Slack's clock and this computer's clock can differ; read a little before the recorded send time.
SLACK_CLOCK_MARGIN_SECONDS = 30.0
# A message can take a moment to show in the channel's history; look again before calling it missing.
SLACK_READ_ATTEMPTS = 3
SLACK_READ_PAUSE_SECONDS = 1.5
HTTP_TIMEOUT_SECONDS = 8.0

_DISCORD_WEBHOOK_PATH = re.compile(r"^(?P<prefix>/api(?:/v\d+)?)/webhooks/(?P<id>\d+)/(?P<token>[^/]+)/?$")
_UNKNOWN_DISCORD_WEBHOOK = 10015


class _Unreachable(Exception):
    """The provider could not be asked (no network, name not resolved, refused address)."""


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def _http_get(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None):
    """GET through the same outbound checks as the senders; returns (status, parsed JSON or None).

    Raises ``_Unreachable`` when the request could not be made. Its message names the exception type only,
    because a transport error can quote the URL, and a Discord URL holds the webhook token.
    """
    import httpx

    from pilot.system import outbound

    try:
        response = await outbound.fetch(
            "GET",
            url,
            allow_private_network=False,
            headers=dict(headers or {}),
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except (httpx.TransportError, OSError, ValueError) as exc:
        raise _Unreachable(type(exc).__name__) from None
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body


def _receipt(output: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads(output or "")
    except ValueError:
        return None
    return receipt if isinstance(receipt, dict) else None


def _status(receipt: dict[str, Any]) -> int | None:
    status = receipt.get("status")
    return status if isinstance(status, int) else None


def _same_text(sent: str, found: str) -> bool:
    """Equal after the escaping and trimming chat services apply (Slack writes &, < and > as entities)."""

    def clean(text: str) -> str:
        return " ".join(html.unescape(text or "").split())

    return clean(sent) == clean(found)


def _when(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%H:%M:%S UTC")


# ── Slack ────────────────────────────────────────────────────────────


@register(ActionType.API_SLACK, method="conditional_slack_read_back_postcondition", timeout=20.0)
async def slack_message_in_channel(result: ActionResult) -> Outcome:
    """Slack's ``ok`` is a receipt; with a bot token the channel's history is read afresh for the text."""
    params: ApiRequestParams = result.action.parameters  # type: ignore[assignment]
    receipt = _receipt(result.output)
    if receipt is None:
        return unobserved(
            "Slack's answer was not recorded, so the message cannot be looked for", kind=Kind.COULD_NOT_LOOK
        )
    status = _status(receipt)
    if status is None or not 200 <= status < 300:
        return mismatch(f"Slack refused the message: HTTP {status}, so nothing was posted")
    answer = receipt.get("body", "ok" if receipt.get("ok") is True else receipt.get("error", ""))
    if str(answer).strip() != "ok":
        return mismatch(f"Slack answered HTTP {status} but not 'ok' ({str(answer).strip()[:60]!r})")

    slack = await connections.service("slack")
    configured_webhook = slack.get(connections.SLACK_WEBHOOK)
    try:
        webhook = connections.choose(
            params.webhook_url, configured_webhook, service_name="slack", what="Slack webhook address"
        )
    except ValueError:
        webhook = ""
    if not urlsplit(webhook).path.startswith("/services/"):
        return unobserved(
            "Slack accepted the message (receipt, no read-back): a workflow webhook decides the text it posts",
            kind=Kind.RECEIPT_ONLY,
        )
    token = slack.get(connections.SLACK_BOT_TOKEN)
    channel = slack.get(connections.SLACK_CHANNEL)
    if not token or not channel:
        return unobserved(
            "Slack accepted the message (receipt, no read-back); add a bot token and the channel ID in "
            f"{connections.WHERE['slack']} to have it read back",
            kind=Kind.RECEIPT_ONLY,
        )
    if webhook != configured_webhook:
        return unobserved(
            "Slack accepted the message (receipt, no read-back): the plan used its own webhook, which may "
            f"post to a channel other than the one connected in {connections.WHERE['slack']}",
            kind=Kind.RECEIPT_ONLY,
        )
    sent_at = receipt.get("sent_at")
    if not isinstance(sent_at, int | float):
        return unobserved(
            "Slack accepted the message (receipt); its send time was not recorded", kind=Kind.RECEIPT_ONLY
        )

    oldest = float(sent_at) - SLACK_CLOCK_MARGIN_SECONDS
    query = {"channel": channel, "oldest": f"{oldest:.6f}", "inclusive": "true", "limit": "200"}
    read = 0
    for attempt in range(SLACK_READ_ATTEMPTS):
        try:
            code, body = await _http_get(SLACK_HISTORY_URL, headers={"Authorization": f"Bearer {token}"}, params=query)
        except _Unreachable as exc:
            return unobserved(
                f"Slack accepted the message (receipt); slack.com could not be reached ({exc})",
                kind=Kind.COULD_NOT_LOOK,
            )
        if code != 200 or not isinstance(body, dict):
            return unobserved(
                f"Slack accepted the message (receipt); reading it back answered HTTP {code}", kind=Kind.COULD_NOT_LOOK
            )
        if body.get("ok") is not True:
            error = str(body.get("error") or "unknown_error")[:60]
            return unobserved(
                f"Slack accepted the message (receipt); reading channel {channel} failed with {error!r} "
                "(the token needs channels:history or groups:history and the app must be in the channel)",
                kind=Kind.COULD_NOT_LOOK,
            )
        messages = body.get("messages")
        if not isinstance(messages, list):
            return unobserved(
                "Slack accepted the message (receipt); the channel history came back without messages",
                kind=Kind.RECEIPT_ONLY,
            )
        read = len(messages)
        for message in messages:
            if isinstance(message, dict) and _same_text(params.message, str(message.get("text", ""))):
                return verified(f"Slack channel {channel} shows the message (ts {message.get('ts', '?')})")
        if attempt + 1 < SLACK_READ_ATTEMPTS:
            await _pause(SLACK_READ_PAUSE_SECONDS)
    return mismatch(
        f"Slack accepted the message, but channel {channel} shows no message with its text since "
        f"{_when(float(sent_at))} ({read} message(s) read). Look at the channel before sending it again"
    )


# ── Discord ──────────────────────────────────────────────────────────


def _discord_message_url(webhook: str, message_id: str) -> tuple[str, dict[str, str], str] | None:
    """The read-back URL for a message of this webhook, the query to keep (thread_id) and the host."""
    parts = urlsplit(webhook)
    match = _DISCORD_WEBHOOK_PATH.match(parts.path)
    if parts.scheme != "https" or not parts.hostname or match is None:
        return None
    query = {key: value for key, value in parse_qsl(parts.query) if key == "thread_id"}
    path = f"{match['prefix']}/webhooks/{match['id']}/{match['token']}/messages/{message_id}"
    return f"https://{parts.netloc}{path}", query, parts.hostname


@register(ActionType.API_DISCORD, method="discord_message_read_back_postcondition", timeout=20.0)
async def discord_message_in_channel(result: ActionResult) -> Outcome:
    """The message Discord created is read back through the webhook's own message endpoint."""
    params: ApiRequestParams = result.action.parameters  # type: ignore[assignment]
    receipt = _receipt(result.output)
    if receipt is None:
        return unobserved(
            "Discord's answer was not recorded, so the message cannot be looked for", kind=Kind.COULD_NOT_LOOK
        )
    status = _status(receipt)
    if status is None or not 200 <= status < 300:
        return mismatch(f"Discord refused the message: HTTP {status}, so nothing was posted")
    message_id = str(receipt.get("message_id") or "")
    if not message_id.isdigit():
        return unobserved(
            "Discord accepted the message (receipt) but gave no message id to read back", kind=Kind.RECEIPT_ONLY
        )
    stored = (await connections.service("discord")).get(connections.DISCORD_WEBHOOK)
    try:
        webhook = connections.choose(params.webhook_url, stored, service_name="discord", what="Discord webhook address")
    except ValueError:
        webhook = ""
    target = _discord_message_url(webhook, message_id)
    if target is None:
        return unobserved(
            f"Discord accepted message {message_id} (receipt); the webhook address is not a Discord webhook "
            "address, so it cannot be read back",
            kind=Kind.RECEIPT_ONLY,
        )
    url, query, host = target
    try:
        code, body = await _http_get(url, params=query or None)
    except _Unreachable as exc:
        return unobserved(
            f"Discord accepted message {message_id} (receipt); {host} could not be reached ({exc})",
            kind=Kind.COULD_NOT_LOOK,
        )
    if code == 200:
        if not isinstance(body, dict) or str(body.get("id")) != message_id:
            return unobserved(
                f"Discord accepted message {message_id} (receipt); {host} answered the read-back without it",
                kind=Kind.RECEIPT_ONLY,
            )
        found = str(body.get("content", ""))
        if _same_text(params.message, found):
            return verified(f"Discord message {message_id} on {host} has the text that was sent")
        return mismatch(
            f"Discord message {message_id} on {host} exists but its text differs from what was sent "
            f"({len(params.message)} characters sent, {len(found)} found)"
        )
    if code == 404:
        if isinstance(body, dict) and body.get("code") == _UNKNOWN_DISCORD_WEBHOOK:
            return unobserved(
                f"Discord accepted message {message_id} (receipt); the webhook no longer exists, "
                "so the message cannot be read back",
                kind=Kind.RECEIPT_ONLY,
            )
        return mismatch(
            f"Discord accepted message {message_id}, but {host} has no such message from this webhook "
            "(HTTP 404). Look at the channel before sending it again"
        )
    return unobserved(
        f"Discord accepted message {message_id} (receipt); reading it back answered HTTP {code}",
        kind=Kind.COULD_NOT_LOOK,
    )
