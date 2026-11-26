"""Post-conditions for mail and calendar actions, read back over a new session with the provider.

Mail (``api_send_email``, ``email_reply``): the SMTP server accepting a message is only a receipt. The check signs in
to the sender's mailbox over a new IMAP session and looks for the message's Message-ID in the Sent folder
(``pilot.system.mail_readback``). It runs only when reading mail is set up in Settings (IMAP host, user name and the
app password in the operating-system credential store) for the same account that sent; otherwise the send stays a
receipt and is reported as not verified. A copy that is not there is a MISMATCH only for providers known to file
every SMTP message in Sent (Gmail, Microsoft); elsewhere it is "not verified". Nothing here sends, appends or moves
mail, so a check can never send a message twice.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pilot.actions import ActionResult, ActionType, ApiRequestParams, EmailParams
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

logger = logging.getLogger("pilot.agents.postconditions.mail_calendar")

# Up to four searches with 14 s of pauses (mail_readback.SEARCH_PAUSES_SECONDS), each IMAP step limited to 10 s.
MAIL_TIMEOUT_SECONDS = 60.0

_config: Any = None
_vault: Any = None


def configure_read_back(config: Any, vault: Any) -> None:
    """Give the checks the daemon's settings and credential store. Called once at start-up by the server."""
    global _config, _vault
    _config, _vault = config, vault


# ── Mail ─────────────────────────────────────────────────────────────────────

_RECEIPT = "The mail server accepted the message (an SMTP receipt)"


def _sender(result: ActionResult) -> str:
    """The account the message went out from, as the handler chose it."""
    params = result.action.parameters
    if isinstance(params, EmailParams):
        return params.username or (getattr(getattr(_config, "email", None), "username", "") or "")
    if isinstance(params, ApiRequestParams):
        # pilot.system.api_client.send_email signs in to SMTP as SMTP_USER and sends from it.
        return os.environ.get("SMTP_USER", "")
    return ""


async def _imap_account(sender: str) -> tuple[Any, str]:
    """(account, why there is none). Reads the same settings and credential store the email agent uses."""
    from pilot.system.mail_readback import ImapAccount

    settings = getattr(_config, "email", None)
    if settings is None or _vault is None:
        return None, "mail read-back is not set up in this process"
    if not settings.enabled or not settings.imap_host or not settings.username:
        return None, "reading mail (IMAP) is not set up in Settings, so the copy in Sent was not looked for"
    if sender and sender.strip().casefold() != settings.username.strip().casefold():
        return None, "the message went out from a different account than the one set up for reading mail"
    try:
        password = await _vault.get_key(settings.password_provider or "email")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read the mail password for the Sent read-back: %s", type(exc).__name__)
        return None, "the saved mail password could not be read from the credential store"
    if not password:
        return None, "no mail app password is saved in the operating-system credential store"
    return (
        ImapAccount(
            host=settings.imap_host,
            username=settings.username,
            password=password,
            sent_mailbox=getattr(settings, "sent_mailbox", "") or "",
        ),
        "",
    )


async def _sent_copy_outcome(result: ActionResult, *, original: str = "", is_reply: bool = False) -> Outcome:
    from pilot.system import mail_readback

    message_id = mail_readback.message_id_from_output(result.output)
    if not message_id:
        return unobserved(f"{_RECEIPT}; the send named no Message-ID, so its copy in Sent could not be looked up")
    account, why = await _imap_account(_sender(result))
    if account is None:
        return unobserved(f"{_RECEIPT}; {why}. Not verified.")
    copy = await mail_readback.find_sent_copy(account, message_id)
    if copy.state == "unavailable":
        return unobserved(f"{_RECEIPT}; its copy in Sent could not be checked: {copy.reason}. Not verified.")
    if copy.state == "absent":
        looked = f'no copy with Message-ID {message_id} is in "{copy.folder}" after {copy.searches} searches'
        if account.host.casefold() in mail_readback.FILES_SMTP_MAIL_IN_SENT:
            return mismatch(
                f"{_RECEIPT}, but {looked} over {copy.waited_seconds:g} s, and {account.host} files every message it "
                "sends there. Heliox did not send it again: look in Sent before sending it again."
            )
        return unobserved(
            f"{_RECEIPT}; {looked}, and {account.host} may not keep a copy of mail sent over SMTP. Not verified; "
            "Heliox did not send it again."
        )
    where = f'"{copy.folder}" ({copy.found_by})'
    if not is_reply:
        return verified(f"Message {message_id} is in the sender's Sent folder {where}")
    if not original:
        return verified(
            f"The reply {message_id} is in the sender's Sent folder {where}; the plan named no original message, "
            "so there was nothing to thread it under"
        )
    if copy.in_reply_to != original:
        return mismatch(
            f"The reply {message_id} is in {where}, but its In-Reply-To is {copy.in_reply_to or 'missing'} instead "
            f"of {original}, so it will not thread under the original"
        )
    if original not in copy.references.split():
        return mismatch(
            f"The reply {message_id} is in {where}, but its References do not name {original}, so some mail "
            "programs will not thread it under the original"
        )
    return verified(f"The reply {message_id} is in {where} and threads under {original} (In-Reply-To and References)")


@register(ActionType.API_SEND_EMAIL, method="conditional_sent_copy_postcondition", timeout=MAIL_TIMEOUT_SECONDS)
async def sent_copy_present(result: ActionResult) -> Outcome:
    """The message is in the sender's Sent folder, found by its Message-ID over a new IMAP session."""
    return await _sent_copy_outcome(result)


@register(ActionType.EMAIL_REPLY, method="conditional_threaded_reply_postcondition", timeout=MAIL_TIMEOUT_SECONDS)
async def threaded_reply_present(result: ActionResult) -> Outcome:
    """The reply is in Sent and its In-Reply-To and References name the original's Message-ID, read on the copy."""
    from pilot.system.mail_readback import in_reply_to_from_output

    params = result.action.parameters
    named = params.in_reply_to.strip() if isinstance(params, EmailParams) else ""
    # The plan's own Message-ID wins; a plan that named the original by its number in the inbox learns the ID from
    # the handler, which read it from the server before sending.
    original = named or in_reply_to_from_output(result.output)
    return await _sent_copy_outcome(result, original=original, is_reply=True)
