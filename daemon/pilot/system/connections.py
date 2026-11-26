"""Where mail, Slack, Discord and GitHub connections come from, in one place.

Each connection is read in this order:

1. what the person connected in Settings, kept in the operating system's credential store (Windows Credential
   Manager, macOS Keychain, a Secret Service keyring on Linux) through ``pilot.security.vault.KeyVault``;
2. the environment variable older setups use (``SLACK_WEBHOOK_URL``, ``DISCORD_WEBHOOK_URL``, ``GITHUB_TOKEN``,
   ``SMTP_*``), unchanged for people who already set one;
3. nothing: the action fails with a sentence that names the Settings place to connect it.

A service takes all its values from one source: once any Slack value is saved in Settings, the Slack values in the
environment are ignored, so a webhook from one workspace is never read back with a token from another.

Mail has one account for both sending paths: the Email settings (host, port, user name) with the app password in the
credential store under ``email.password_provider``. The executor's ``api_send_email`` and the email agent use the
same account.

A plan never needs a secret: the senders take the stored connection when the plan gives none. A plan that still
carries one keeps working; a value hidden when the plan was kept (``pilot.security.secret_urls``) is matched against
the stored connection and replaced by it when their masked forms agree.

On Linux without a Secret Service (a headless or minimal desktop), the credential store is unavailable: Settings
cannot save a connection and says so, nothing is written to a file instead, and environment variables still work.

No function here logs or returns a secret value except to the sender that uses it.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from pilot.security.secret_urls import is_masked, mask_secret, mask_url, unrestored_reason

logger = logging.getLogger("pilot.system.connections")

# Credential store names (pilot.security.vault.KNOWN_PROVIDERS lists them for clean-up on uninstall).
SLACK_WEBHOOK = "slack_webhook"
SLACK_BOT_TOKEN = "slack_bot_token"
SLACK_CHANNEL = "slack_channel_id"
DISCORD_WEBHOOK = "discord_webhook"
GITHUB_TOKEN = "github_token"

SERVICE_NAMES: dict[str, tuple[str, ...]] = {
    "slack": (SLACK_WEBHOOK, SLACK_BOT_TOKEN, SLACK_CHANNEL),
    "discord": (DISCORD_WEBHOOK,),
    "github": (GITHUB_TOKEN,),
}
STORE_NAMES = frozenset(name for names in SERVICE_NAMES.values() for name in names)

ENVIRONMENT: dict[str, str] = {
    SLACK_WEBHOOK: "SLACK_WEBHOOK_URL",
    SLACK_BOT_TOKEN: "SLACK_BOT_TOKEN",
    SLACK_CHANNEL: "SLACK_CHANNEL_ID",
    DISCORD_WEBHOOK: "DISCORD_WEBHOOK_URL",
    GITHUB_TOKEN: "GITHUB_TOKEN",
}
MAIL_ENVIRONMENT = ("SMTP_USER", "SMTP_PASS")

WHERE = {
    "mail": "Settings > Email integration",
    "slack": "Settings > Connections > Slack",
    "discord": "Settings > Connections > Discord",
    "github": "Settings > Connections > GitHub",
}
TITLE = {"mail": "Mail", "slack": "Slack", "discord": "Discord", "github": "GitHub"}

SETTINGS = "settings"
ENVIRONMENT_SOURCE = "environment"

_config: Any = None
_vault: Any = None


def configure(config: Any, vault: Any) -> None:
    """Give the lookups the daemon's settings and credential store. Called once at start-up by the server.

    Until then (a library caller, a test) only environment variables are read.
    """
    global _config, _vault
    _config, _vault = config, vault


def not_connected(service: str) -> str:
    return f"{TITLE[service]} is not connected. Connect it in {WHERE[service]}."


async def _stored(name: str) -> str:
    """The value saved in the credential store, or "" (none saved, no store, or the store could not be read)."""
    if _vault is None:
        return ""
    try:
        value = await _vault.get_key(name)
    except Exception as exc:  # noqa: BLE001 - a locked or broken store reads as "nothing saved", with a log line
        logger.warning("The credential store could not read %s: %s", name, type(exc).__name__)
        return ""
    return (value or "").strip()


def _environment(name: str) -> str:
    return os.environ.get(ENVIRONMENT[name], "").strip()


@dataclass(frozen=True)
class Found:
    """A connection's values and where they came from ("settings", "environment" or "")."""

    values: dict[str, str]
    source: str

    def get(self, name: str) -> str:
        return self.values.get(name, "")


async def service(name: str) -> Found:
    """All values of one service, from the credential store if any is saved there, else from the environment."""
    names = SERVICE_NAMES[name]
    stored = {key: await _stored(key) for key in names}
    if any(stored.values()):
        return Found(stored, SETTINGS)
    environment = {key: _environment(key) for key in names}
    if any(environment.values()):
        return Found(environment, ENVIRONMENT_SOURCE)
    return Found({key: "" for key in names}, "")


def environment_set(name: str) -> bool:
    """Whether an environment variable for this service is set (its value is never returned)."""
    if name == "mail":
        return all(os.environ.get(variable, "").strip() for variable in MAIL_ENVIRONMENT)
    return any(_environment(key) for key in SERVICE_NAMES[name])


def choose(plan_value: str | None, stored: str, *, service_name: str, what: str) -> str:
    """The value a step uses: the stored one when the plan gives none or gives the stored one hidden.

    A literal value in the plan is used as given (older plans keep working). A hidden value that is not the stored
    one cannot be recovered: the step fails before anything is sent.
    """
    given = (plan_value or "").strip()
    if not given:
        return stored
    if not is_masked(given):
        return given
    if stored and given in {mask_url(stored), mask_secret(stored)}:
        return stored
    why = unrestored_reason(given)
    raise ValueError(
        f"The {what} in this plan was hidden when the plan was kept and is not the one connected in "
        f"{WHERE[service_name]}, so nothing was sent. " + (why or f"Connect it there, or ask again with the {what}.")
    )


# ── Mail ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MailAccount:
    host: str
    port: int
    username: str
    password: str
    source: str

    def __repr__(self) -> str:  # the password never reaches a log line through a repr
        return f"MailAccount(host={self.host!r}, port={self.port}, username={self.username!r}, source={self.source!r})"


def _mail_settings() -> Any:
    settings = getattr(_config, "email", None)
    if settings is None or not getattr(settings, "enabled", False):
        return None
    if not getattr(settings, "smtp_host", "") or not getattr(settings, "username", ""):
        return None
    return settings


async def mail_account() -> MailAccount | None:
    """The account both mail paths send from: Settings with its saved app password, else ``SMTP_*``."""
    settings = _mail_settings()
    if settings is not None:
        password = await _stored(settings.password_provider or "email")
        if password:
            return MailAccount(
                host=settings.smtp_host,
                port=int(settings.smtp_port or 587),
                username=settings.username,
                password=password,
                source=SETTINGS,
            )
    username = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "")
    if username and password:
        try:
            port = int(os.environ.get("SMTP_PORT", "587"))
        except ValueError:
            port = 587
        return MailAccount(
            host=os.environ.get("SMTP_SERVER", "").strip() or "smtp.gmail.com",
            port=port,
            username=username,
            password=password,
            source=ENVIRONMENT_SOURCE,
        )
    return None


async def mail_sender() -> str:
    """The address ``api_send_email`` sends from, for the Sent-folder read-back; "" when no account is connected."""
    account = await mail_account()
    return account.username if account is not None else ""


# ── Checking what a person enters in Settings ────────────────────────

_SLACK_HOSTS = ("hooks.slack.com", "hooks.slack-gov.com")
_DISCORD_HOSTS = ("discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com")
_DISCORD_PATH = re.compile(r"^/api(/v\d+)?/webhooks/\d+/[A-Za-z0-9_-]+/?$")
_SLACK_CHANNEL = re.compile(r"^[CGD][A-Z0-9]{6,}$")


def check_value(name: str, value: str) -> str:
    """Why this value cannot be saved under ``name``, or "" when it can. Never quotes the value."""
    text = value.strip()
    if not text:
        return "Enter a value first."
    if any(character.isspace() for character in text):
        return "The value contains spaces or line breaks; paste it again without them."
    if name in {SLACK_WEBHOOK, DISCORD_WEBHOOK}:
        parts = urlsplit(text)
        if parts.scheme != "https" or not parts.hostname:
            return "A webhook address starts with https://."
        host = parts.hostname.lower()
        if name == SLACK_WEBHOOK and (
            host not in _SLACK_HOSTS or not parts.path.startswith(("/services/", "/triggers/", "/workflows/"))
        ):
            return "This is not a Slack webhook address (https://hooks.slack.com/services/...)."
        if name == DISCORD_WEBHOOK and (host not in _DISCORD_HOSTS or not _DISCORD_PATH.match(parts.path)):
            return "This is not a Discord webhook address (https://discord.com/api/webhooks/...)."
        return ""
    if name == SLACK_BOT_TOKEN:
        return "" if text.startswith("xoxb-") else "A Slack bot token starts with xoxb-."
    if name == SLACK_CHANNEL:
        return (
            ""
            if _SLACK_CHANNEL.match(text)
            else "A Slack channel ID looks like C0123ABCD (open the channel's details)."
        )
    if name == GITHUB_TOKEN:
        return "" if 20 <= len(text) <= 255 else "This does not look like a GitHub token."
    return "Unknown connection value."


# ── What Settings shows: never a secret ──────────────────────────────


def _url_hint(url: str) -> dict[str, str]:
    parts = urlsplit(url)
    return {"host": (parts.hostname or "").lower(), "last4": url[-4:] if len(url) > 12 else ""}


async def status() -> dict[str, Any]:
    """Whether each connection is set and where from. Hosts, last four characters and the channel ID only."""
    slack = await service("slack")
    discord = await service("discord")
    github = await service("github")
    account = await mail_account()
    settings = getattr(_config, "email", None)
    connections: dict[str, Any] = {
        "mail": {
            "connected": account is not None,
            "source": account.source if account is not None else "",
            "account": account.username if account is not None else "",
            "environment_also_set": environment_set("mail") and account is not None and account.source == SETTINGS,
            "settings_enabled": bool(getattr(settings, "enabled", False)),
        },
        "slack": {
            "connected": bool(slack.get(SLACK_WEBHOOK)),
            "source": slack.source if slack.get(SLACK_WEBHOOK) else "",
            "webhook": _url_hint(slack.get(SLACK_WEBHOOK)) if slack.get(SLACK_WEBHOOK) else None,
            "read_back": bool(slack.get(SLACK_BOT_TOKEN) and slack.get(SLACK_CHANNEL)),
            "bot_token": bool(slack.get(SLACK_BOT_TOKEN)),
            "channel_id": slack.get(SLACK_CHANNEL),
            "environment_also_set": slack.source == SETTINGS and environment_set("slack"),
        },
        "discord": {
            "connected": bool(discord.get(DISCORD_WEBHOOK)),
            "source": discord.source,
            "webhook": _url_hint(discord.get(DISCORD_WEBHOOK)) if discord.get(DISCORD_WEBHOOK) else None,
            "environment_also_set": discord.source == SETTINGS and environment_set("discord"),
        },
        "github": {
            "connected": bool(github.get(GITHUB_TOKEN)),
            "source": github.source,
            "environment_also_set": github.source == SETTINGS and environment_set("github"),
        },
    }
    available = bool(getattr(_vault, "available", False))
    return {
        "status": "ok",
        "available": available,
        "backend": getattr(_vault, "backend_name", "") if _vault is not None else "",
        "message": ""
        if available
        else (
            "This computer has no usable credential store (Windows Credential Manager, macOS Keychain or a Secret "
            "Service keyring), so connections cannot be saved here. Environment variables still work."
        ),
        "connections": connections,
    }
