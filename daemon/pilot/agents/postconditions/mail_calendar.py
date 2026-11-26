"""Post-conditions for mail and calendar actions, read back over a new session with the provider.

Mail (``api_send_email``, ``email_reply``): the SMTP server accepting a message is only a receipt. The check signs in
to the sender's mailbox over a new IMAP session and looks for the message's Message-ID in the Sent folder
(``pilot.system.mail_readback``). It runs only when reading mail is set up in Settings (IMAP host, user name and the
app password in the operating-system credential store) for the same account that sent; otherwise the send stays a
receipt and is reported as not verified. A copy that is not there is a MISMATCH only for providers known to file
every SMTP message in Sent (Gmail, Microsoft); elsewhere it is "not verified". Nothing here sends, appends or moves
mail, so a check can never send a message twice.

Calendar (``calendar_create_event``, ``calendar_delete_event``): a fresh CalDAV client (new session, same settings and
saved password as the calendar agent) asks the server for the event by its UID (``pilot.system.calendar_readback``).
A server that cannot be reached, or settings that are missing, give "not verified", never a pass.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
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

_RECEIPT = "The mail server accepted the message"


async def _sender(result: ActionResult) -> str:
    """The account the message went out from, as the handler chose it."""
    params = result.action.parameters
    if isinstance(params, EmailParams):
        return params.username or (getattr(getattr(_config, "email", None), "username", "") or "")
    if isinstance(params, ApiRequestParams):
        # pilot.system.api_client.send_email sends from the connected mail account (Settings, else SMTP_USER).
        from pilot.system import connections

        return await connections.mail_sender()
    return ""


async def _imap_account(sender: str) -> tuple[Any, str]:
    """(account, why there is none). Reads the same settings and credential store the email agent uses."""
    from pilot.system.mail_readback import ImapAccount

    settings = getattr(_config, "email", None)
    if settings is None or _vault is None:
        return None, "mail read-back is not set up in this process"
    if not settings.enabled or not settings.imap_host or not settings.username:
        return None, "reading mail (IMAP) is not set up in Settings"
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
        return unobserved(
            f"{_RECEIPT}, but the send named no Message-ID, so Heliox could not look for its copy in Sent."
        )
    account, why = await _imap_account(await _sender(result))
    if account is None:
        return unobserved(f"{_RECEIPT}, but Heliox could not check its copy in Sent: {why}.")
    copy = await mail_readback.find_sent_copy(account, message_id)
    if copy.state == "unavailable":
        return unobserved(f"{_RECEIPT}, but Heliox could not check its copy in Sent: {copy.reason}.")
    if copy.state == "absent":
        looked = f'no copy with Message-ID {message_id} is in "{copy.folder}" after {copy.searches} searches'
        if account.host.casefold() in mail_readback.FILES_SMTP_MAIL_IN_SENT:
            return mismatch(
                f"{_RECEIPT}, but {looked} over {copy.waited_seconds:g} s, and {account.host} files every message it "
                "sends there. Heliox did not send it again: look in Sent before sending it again."
            )
        return unobserved(
            f"{_RECEIPT}, but {looked}, and {account.host} may not keep a copy of mail sent this way. Heliox "
            "did not send it again."
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


# ── Calendar ─────────────────────────────────────────────────────────────────

CALENDAR_TIMEOUT_SECONDS = 30.0


async def _calendar_account() -> tuple[Any, str]:
    """(account, why there is none). Reads the same settings and credential store the calendar agent uses."""
    from urllib.parse import urlparse

    from pilot.system.calendar_readback import CalendarAccount

    settings = getattr(_config, "calendar", None)
    if settings is None or _vault is None:
        return None, "calendar read-back is not set up in this process"
    if not settings.enabled or not settings.caldav_url or not settings.caldav_username:
        return None, "CalDAV is not set up in Settings"
    parsed = urlparse(settings.caldav_url)
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
        return None, "the CalDAV address is not HTTPS, so Heliox will not sign in to it"
    provider = settings.caldav_password_provider
    try:
        password = await _vault.get_key(provider) if provider else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read the CalDAV password for the read-back: %s", type(exc).__name__)
        return None, "the saved CalDAV password could not be read from the credential store"
    if not password:
        return None, "no CalDAV password is saved in the operating-system credential store"
    return CalendarAccount(settings.caldav_url, settings.caldav_username, password), ""


async def _read_calendar(calendar_id: str | None, look: Callable[[Any], Any]) -> tuple[Any, str]:
    """Run ``look(calendar)`` on a calendar opened by a new client, in a worker thread. (answer, why it could not)."""
    from pilot.system.calendar_readback import open_calendar

    account, why = await _calendar_account()
    if account is None:
        return None, why

    def read() -> Any:
        return look(open_calendar(account, calendar_id))

    try:
        return await asyncio.to_thread(read), ""
    except Exception as exc:  # noqa: BLE001
        # The server or the network did not answer the read: that says nothing about the event either way.
        logger.info("CalDAV read-back failed: %s", type(exc).__name__)
        if isinstance(exc, LookupError):
            return None, str(exc)
        return None, f"the CalDAV server did not answer ({type(exc).__name__})"


@register(
    ActionType.CALENDAR_DELETE_EVENT,
    method="caldav_event_absence_postcondition",
    timeout=CALENDAR_TIMEOUT_SECONDS,
)
async def event_absent(result: ActionResult) -> Outcome:
    """No event with the UID remains on the server, asked by a fresh CalDAV client (for a series, every occurrence)."""
    from pilot.system.calendar_readback import event_by_uid

    params = result.action.parameters
    uid = str(getattr(params, "event_uid", "") or "").strip()
    if not uid:
        raise ValueError("Calendar delete has no event UID to observe")

    def still_there(calendar: Any) -> bool:
        return event_by_uid(calendar, uid) is not None

    answer, why = await _read_calendar(getattr(params, "calendar_id", None), still_there)
    if answer is None:
        return unobserved(f"Heliox could not check that event {uid} is gone from the calendar, because {why}.")
    if answer:
        return mismatch(f"Event {uid} is still on the CalDAV server after the delete")
    return verified(f"No event with UID {uid} remains on the CalDAV server")


@register(
    ActionType.CALENDAR_CREATE_EVENT,
    method="caldav_event_fields_postcondition",
    timeout=CALENDAR_TIMEOUT_SECONDS,
)
async def event_matches(result: ActionResult) -> Outcome:
    """The event is on the server with the requested summary, start and end, read by its UID with a fresh client."""
    from datetime import datetime

    from pilot.system import calendar_readback as readback

    params = result.action.parameters
    uid = readback.event_uid_from_output(result.output)
    if not uid:
        return unobserved(
            "The calendar server accepted the event, but the create named no UID, so Heliox could not read it back."
        )
    summary = str(getattr(params, "summary", "") or "")
    start = datetime.fromisoformat(str(getattr(params, "start", "")))
    end_text = getattr(params, "end", None)
    end = datetime.fromisoformat(end_text) if end_text else None

    def look(calendar: Any) -> tuple[str, Any]:
        event = readback.event_by_uid(calendar, uid)
        if event is not None:
            return "uid", readback.series_component(event)
        return "similar", readback.events_like(calendar, summary, start)

    when = readback.describe(start)
    looks = len(readback.LOOK_PAUSES_SECONDS)
    waited = 0.0
    for pause in readback.LOOK_PAUSES_SECONDS:
        if pause:
            await readback._pause(pause)
            waited += pause
        answer, why = await _read_calendar(getattr(params, "calendar_id", None), look)
        if answer is None:
            return unobserved(f"Heliox could not read the new event back from the calendar, because {why}.")
        kind, found = answer
        if kind == "uid" or found:
            break
    if kind == "similar":
        if found:
            # Some servers keep an event under a UID of their own; the event itself is what was asked for.
            return verified(f'"{summary}" is on the calendar at {when}, kept under the UID {found[0]}')
        # A slow server can still be filing it: "not there yet" is no reason to create a second event.
        return unobserved(
            f'The calendar server accepted "{summary}" at {when}, but it was not found there after {looks} looks over '
            f"{waited:g} s. Heliox did not create it again; check the calendar before adding it again."
        )
    if found is None:
        return mismatch(f"Event {uid} on the calendar server holds no event data")
    differences = []
    found_summary = str(found.get("summary", ""))
    if found_summary != summary:
        differences.append(f'summary "{found_summary}" instead of "{summary}"')
    found_start = readback.property_time(found, "dtstart")
    if not readback.same_moment(start, found_start):
        differences.append(f"start {readback.describe(found_start)} instead of {when}")
    if end is not None:
        found_end = readback.property_time(found, "dtend")
        if not readback.same_moment(end, found_end):
            differences.append(f"end {readback.describe(found_end)} instead of {readback.describe(end)}")
    if differences:
        # This action is in RECONCILE_NOT_RETRY_ACTIONS, so the run stops here and the person sees this text.
        return mismatch(
            f'The calendar now holds "{found_summary}" with {"; ".join(differences)}. Heliox did not create it '
            "again: correct or delete that event yourself."
        )
    return verified(f'Event {uid} "{summary}" is on the calendar at {when} as requested')
