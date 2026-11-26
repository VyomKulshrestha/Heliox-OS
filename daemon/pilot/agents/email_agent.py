"""Email Agent — reads and sends emails via IMAP/SMTP using App Passwords.

Specialises in:
  - Fetching unread emails securely over IMAP (SSL)
  - Summarising email threads using the model router
  - Drafting and sending replies via SMTP (STARTTLS)

Security model
--------------
The daemon resolves the user's app password from the operating-system keyring
only when an IMAP/SMTP connection is opened.  It is never written into a plan,
history record, configuration file, or log.  Explicit ``EmailParams``
credentials remain supported only for direct library consumers that construct
this agent without the daemon's config and vault dependencies.

All outbound send/reply actions are tagged ``requires_confirmation=True`` so
the security gate will always prompt the user before anything is transmitted.
"""

from __future__ import annotations

import asyncio
import email as email_lib
import email.header
import email.utils
import imaplib
import json
import logging
import platform
import re
import smtplib
import ssl
import subprocess
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType, CalendarParams, EmailParams
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent
from pilot.system.mail_readback import message_id_note, new_message_id

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride
    from pilot.security.vault import KeyVault

logger = logging.getLogger("pilot.agents.email_agent")

EMAIL_ACTION_TYPES: set[ActionType] = {
    ActionType.EMAIL_FETCH,
    ActionType.EMAIL_SUMMARIZE,
    ActionType.EMAIL_REPLY,
    ActionType.API_SEND_EMAIL,
    ActionType.CALENDAR_FETCH,
    ActionType.CALENDAR_RECONCILE,
}

# Maximum body length forwarded to the LLM to avoid token bloat
_MAX_BODY_CHARS = 2_000


class EmailAgent(BaseAgent):
    """Specialist agent for reading and sending emails via IMAP/SMTP."""

    def __init__(
        self,
        model_router: ModelRouter,
        config: PilotConfig | None = None,
        vault: KeyVault | None = None,
    ) -> None:
        super().__init__(role=AgentRole.COMMUNICATION, model_router=model_router)
        self._config = config
        self._vault = vault

    # ── Capabilities ──────────────────────────────────────────────────────────

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.EMAIL_FETCH,
                description="Fetch unread emails from an IMAP mailbox using an App Password",
                requires_confirmation=False,
            ),
            AgentCapability(
                action_type=ActionType.EMAIL_SUMMARIZE,
                description="Summarise a list of fetched emails using the LLM",
                requires_confirmation=False,
            ),
            AgentCapability(
                action_type=ActionType.EMAIL_REPLY,
                description="Draft and send a reply to an email via SMTP",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.API_SEND_EMAIL,
                description="Compose and send a new email via SMTP",
                requires_confirmation=True,
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_FETCH,
                description="Parse upcoming calendar events embedded as ICS in fetched emails",
                requires_confirmation=False,
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_RECONCILE,
                description=(
                    "Cross-reference calendar events with emails; fire OS notifications "
                    "for scheduling conflicts or missing meeting links"
                ),
                requires_confirmation=False,
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the EMAIL AGENT for Heliox OS. "
            "You securely connect to the user's mail provider using IMAP (for reading) "
            "and SMTP (for sending) with App Passwords — never the user's main password. "
            "When summarising emails, be concise: one sentence per email. "
            "When drafting replies, match the tone of the original message and keep "
            "the reply professional unless instructed otherwise. "
            "You ALWAYS confirm with the user before sending any email."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in EMAIL_ACTION_TYPES

    # ── Task dispatcher ───────────────────────────────────────────────────────

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        """Dispatch each email action to the appropriate handler.

        AgentOrchestrator gates these direct IMAP/SMTP calls and applies the
        caller's scope override before dispatch.
        """
        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        results: list[ActionResult] = []
        for action in my_actions:
            params = action.parameters

            try:
                if action.action_type in {
                    ActionType.EMAIL_FETCH,
                    ActionType.EMAIL_SUMMARIZE,
                    ActionType.EMAIL_REPLY,
                    ActionType.API_SEND_EMAIL,
                }:
                    if not isinstance(params, EmailParams):
                        result = ActionResult(action=action, success=False, error="EmailParams required")
                    elif action.action_type == ActionType.EMAIL_SUMMARIZE:
                        result = await self._summarize_emails(action, params)
                    else:
                        resolved = await self._with_saved_credentials(params)
                        if action.action_type == ActionType.EMAIL_FETCH:
                            result = await self._fetch_emails(action, resolved)
                        else:
                            result = await self._send_email(action, resolved)
                elif action.action_type in {ActionType.CALENDAR_FETCH, ActionType.CALENDAR_RECONCILE}:
                    if not isinstance(params, CalendarParams):
                        result = ActionResult(action=action, success=False, error="CalendarParams required")
                    elif action.action_type == ActionType.CALENDAR_FETCH:
                        result = await self._fetch_calendar_events(action, params)
                    else:
                        result = await self._reconcile_calendar(action, params)
                else:
                    result = ActionResult(
                        action=action,
                        success=False,
                        error=f"Unhandled action type: {action.action_type}",
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("EmailAgent error on %s", action.action_type)
                result = ActionResult(action=action, success=False, error=str(exc))

            results.append(result)

        duration_ms = int((time.time() - start) * 1000)
        self._record_task(duration_ms, all(r.success for r in results))
        self.status = AgentStatus.IDLE
        return results

    async def _with_saved_credentials(self, params: EmailParams) -> EmailParams:
        """Resolve mail credentials without putting a password into a plan."""
        if self._config is None or self._vault is None:
            # Compatibility for direct library users; the Heliox daemon always
            # injects config + vault through agent auto-registration.
            return params
        if not self._config.email.enabled:
            raise ValueError("Email integration is disabled in Settings")
        provider = self._config.email.password_provider or "email"
        password = await self._vault.get_key(provider)
        if not password:
            raise ValueError("No email app password is saved in the operating-system keyring")
        return params.model_copy(
            update={
                "imap_host": self._config.email.imap_host,
                "smtp_host": self._config.email.smtp_host,
                "smtp_port": self._config.email.smtp_port,
                "username": self._config.email.username,
                "app_password": password,
            }
        )

    async def test_connection(self) -> dict[str, Any]:
        """Authenticate to IMAP without reading or changing any messages."""
        try:
            params = await self._with_saved_credentials(EmailParams())
            if not params.imap_host or not params.username or not params.app_password:
                raise ValueError("IMAP host, username, and a saved app password are required")
            context = ssl.create_default_context()
            mail = imaplib.IMAP4_SSL(params.imap_host, ssl_context=context)
            try:
                mail.login(params.username, params.app_password)
            finally:
                try:
                    mail.logout()
                except Exception:  # noqa: BLE001
                    pass
            return {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Email connection test failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    # ── IMAP: fetch unread emails ─────────────────────────────────────────────

    async def _fetch_emails(self, action: Any, params: EmailParams) -> ActionResult:
        """Connect to IMAP over SSL and fetch unread messages."""
        if not params.imap_host or not params.username or not params.app_password:
            return ActionResult(
                action=action,
                success=False,
                error="imap_host, username, and app_password are required for EMAIL_FETCH",
            )

        logger.info("Connecting to IMAP host %s as %s", params.imap_host, params.username)

        ssl_context = ssl.create_default_context()
        try:
            mail = imaplib.IMAP4_SSL(params.imap_host, ssl_context=ssl_context)
        except Exception as exc:
            return ActionResult(action=action, success=False, error=f"IMAP connection failed: {exc}")

        try:
            mail.login(params.username, params.app_password)
            mail.select(params.mailbox)

            # Search for unseen messages
            status, data = mail.search(None, "UNSEEN")
            if status != "OK":
                return ActionResult(action=action, success=False, error="IMAP SEARCH failed")

            uid_list = data[0].split()
            unseen_total = len(uid_list)
            # Respect the caller's limit, newest first
            uid_list = uid_list[-params.max_emails :][::-1]

            emails: list[dict[str, str]] = []
            unreadable = 0
            for uid in uid_list:
                # BODY.PEEK[] leaves the message unread on the server; (RFC822) would set \Seen on every fetch.
                fetch_status, msg_data = mail.fetch(uid, "(BODY.PEEK[])")
                if fetch_status != "OK" or not msg_data or not msg_data[0]:
                    unreadable += 1
                    continue

                raw = msg_data[0][1]
                if not isinstance(raw, bytes):
                    unreadable += 1
                    continue

                msg = email_lib.message_from_bytes(raw)
                body = _extract_body(msg)

                entry = {
                    "uid": uid.decode(),
                    "message_id": msg.get("Message-ID", ""),
                    "from": msg.get("From", ""),
                    "to": msg.get("To", ""),
                    "subject": msg.get("Subject", "(no subject)"),
                    "date": msg.get("Date", ""),
                    "body": body[:_MAX_BODY_CHARS],
                }
                calendar = _extract_calendar(msg, body)
                if calendar:
                    # Whole, never cut: calendar_fetch parses these, and the body cut is for display only.
                    entry["calendar"] = calendar
                if len(body) > _MAX_BODY_CHARS:
                    # Said inside the message, so the next step knows the body is partial.
                    entry["body_cut"] = f"Only the first {_MAX_BODY_CHARS:,} of {len(body):,} characters are included."
                emails.append(entry)

                if params.mark_as_read:
                    mail.store(uid, "+FLAGS", "\\Seen")

            mail.logout()
        except imaplib.IMAP4.error as exc:
            return ActionResult(action=action, success=False, error=f"IMAP error: {exc}")

        logger.info("Fetched %d unread email(s) from %s", len(emails), params.mailbox)
        notes = []
        if unseen_total > len(uid_list):
            notes.append(f"Only the newest {len(uid_list)} of {unseen_total} unread messages were fetched.")
        if unreadable:
            notes.append(f"{unreadable} message(s) could not be fetched and are not included.")
        cut_bodies = sum(1 for entry in emails if "body_cut" in entry)
        if cut_bodies:
            notes.append(f"{cut_bodies} message body(ies) were cut at {_MAX_BODY_CHARS:,} characters.")
        # An object, so the next step sees what was left out as well as the messages.
        output = json.dumps(
            {
                "messages": emails,
                "unread_total": unseen_total,
                "fetched": len(emails),
                "left_out": {"older_unread": unseen_total - len(uid_list), "could_not_fetch": unreadable},
                "notes": notes,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ActionResult(action=action, success=True, output=output, notice=" ".join(notes) or None)

    # ── LLM: summarise emails ─────────────────────────────────────────────────

    async def _summarize_emails(self, action: Any, params: EmailParams) -> ActionResult:
        """Use the model router to produce a concise summary of fetched emails."""
        if not params.emails_json:
            return ActionResult(
                action=action,
                success=False,
                error="emails_json must contain the JSON output from EMAIL_FETCH",
            )

        try:
            emails, notes = _fetched_messages(params.emails_json)
        except ValueError as exc:
            return ActionResult(action=action, success=False, error=f"Invalid emails_json: {exc}")

        if not emails:
            return ActionResult(action=action, success=True, output=_with_notes("No emails to summarise.", notes))

        if self._model is None:
            # Fallback: plain-text list when no LLM is available
            lines = [
                f"{i + 1}. From: {e.get('from', '')} | Subject: {e.get('subject', '')} | Date: {e.get('date', '')}"
                for i, e in enumerate(emails)
            ]
            return ActionResult(action=action, success=True, output=_with_notes("\n".join(lines), notes))

        # Build a compact prompt
        email_block = "\n\n".join(
            f"[{i + 1}] From: {e.get('from', '')}\n"
            f"Subject: {e.get('subject', '')}\n"
            f"Date: {e.get('date', '')}\n"
            f"Body: {e.get('body', '')[:_MAX_BODY_CHARS]}"
            for i, e in enumerate(emails)
        )
        prompt = (
            "You are an email assistant. Summarise each of the following emails in one "
            "sentence. Number each summary to match the original.\n\n"
            f"{email_block}"
        )

        try:
            summary = await self._model.complete(prompt)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(action=action, success=False, error=f"LLM summarisation failed: {exc}")

        return ActionResult(action=action, success=True, output=_with_notes(summary, notes))

    # ── SMTP: send / reply ────────────────────────────────────────────────────

    async def _send_email(self, action: Any, params: EmailParams) -> ActionResult:
        """Draft (optionally via LLM) and send an email over SMTP with STARTTLS."""
        if not params.smtp_host or not params.username or not params.app_password:
            return ActionResult(
                action=action,
                success=False,
                error="smtp_host, username, and app_password are required to send email",
            )

        original: dict[str, str] = {}
        if action.action_type == ActionType.EMAIL_REPLY:
            try:
                original = await asyncio.to_thread(_original_message, params)
            except (OSError, imaplib.IMAP4.error, ValueError) as exc:
                return ActionResult(
                    action=action,
                    success=False,
                    error=f"Could not read the message being replied to, so nothing was sent: {exc}",
                )
            recipient = params.to or original.get("reply_to", "") or original.get("from", "")
            if not recipient and "@" in params.reply_to_uid:  # earlier plans put the address here
                recipient = params.reply_to_uid
        else:
            recipient = params.to or params.reply_to_uid  # reply_to_uid doubles as To: for API_SEND_EMAIL
        if not recipient:
            return ActionResult(action=action, success=False, error="No recipient specified (set 'to')")

        subject = params.subject
        if not subject and original.get("subject"):
            subject = original["subject"]
            if not subject.casefold().startswith("re:"):
                subject = f"Re: {subject}"

        body = params.reply_body

        # If no body provided, ask the LLM to draft one
        if not body and self._model is not None:
            draft_prompt = (
                f"Draft a professional email reply.\n"
                f"To: {recipient}\n"
                f"Subject: {subject}\n"
                f"Write only the email body, no greeting header."
            )
            try:
                body = await self._model.complete(draft_prompt)
            except Exception as exc:  # noqa: BLE001
                return ActionResult(action=action, success=False, error=f"LLM draft failed: {exc}")

        if not body:
            return ActionResult(
                action=action,
                success=False,
                error="reply_body is empty and no LLM is available to draft a reply",
            )

        # Build the MIME message
        msg = MIMEMultipart("alternative")
        msg["From"] = params.username
        msg["To"] = recipient
        msg["Subject"] = subject or "Re: (no subject)"
        # The verifier finds the sender's copy in Sent by this Message-ID (pilot.system.mail_readback).
        message_id = new_message_id(params.username)
        msg["Message-ID"] = message_id
        msg["Date"] = email_lib.utils.formatdate(localtime=True)
        original_id = original.get("message_id", "")
        if original_id:
            # A reply threads under the original in every mail program only with both headers (RFC 5322 3.6.4).
            references = original.get("references", "").split()
            msg["In-Reply-To"] = original_id
            msg["References"] = " ".join([*references, original_id] if original_id not in references else references)
        msg.attach(MIMEText(body, "plain", "utf-8"))

        logger.info(
            "Sending email via %s:%d from %s to %s",
            params.smtp_host,
            params.smtp_port,
            params.username,
            recipient,
        )

        from pilot.system.system_guard import check_mail

        check_mail(params.smtp_host, smtplib.SMTP)
        ssl_context = ssl.create_default_context()
        try:
            with smtplib.SMTP(params.smtp_host, params.smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ssl_context)
                server.ehlo()
                server.login(params.username, params.app_password)
                server.sendmail(params.username, [recipient], msg.as_string())
        except smtplib.SMTPException as exc:
            return ActionResult(action=action, success=False, error=f"SMTP error: {exc}")

        logger.info("Email sent successfully to %s", recipient)
        return ActionResult(
            action=action,
            success=True,
            output=f"Email sent to {recipient} with subject '{msg['Subject']}'{message_id_note(message_id, original_id)}",
        )

    # ── Calendar: fetch events from ICS in emails ─────────────────────────────

    async def _fetch_calendar_events(self, action: Any, params: CalendarParams) -> ActionResult:
        """Parse VEVENT records from ICS blocks embedded in fetched email bodies."""
        if not params.emails_json:
            return ActionResult(
                action=action,
                success=False,
                error="emails_json is required for CALENDAR_FETCH",
            )

        try:
            events, notes = _parse_ics_from_emails(params.emails_json)
        except ValueError as exc:
            return ActionResult(action=action, success=False, error=f"Invalid emails_json: {exc}")
        now_ts = time.time()
        cutoff_ts = now_ts + params.lookahead_hours * 3600
        upcoming = [e for e in events if now_ts <= e["dtstart_ts"] <= cutoff_ts]
        outside = len(events) - len(upcoming)
        if outside:
            notes.append(
                f"{outside} event(s) start outside the next {params.lookahead_hours} hours and are not listed "
                "(set lookahead_hours to include them)."
            )

        logger.info(
            "CALENDAR_FETCH: %d event(s) in next %dh",
            len(upcoming),
            params.lookahead_hours,
        )
        return ActionResult(
            action=action,
            success=True,
            output=json.dumps(
                {
                    "events": upcoming,
                    "lookahead_hours": params.lookahead_hours,
                    "found": len(events),
                    "outside_window": outside,
                    "notes": notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            notice=" ".join(notes) or None,
        )

    # ── Calendar: reconcile events with emails ────────────────────────────────

    async def _reconcile_calendar(self, action: Any, params: CalendarParams) -> ActionResult:
        """Detect conflicts / missing meeting links and fire OS-native notifications."""
        if not params.emails_json:
            return ActionResult(
                action=action,
                success=False,
                error="emails_json is required for CALENDAR_RECONCILE",
            )

        try:
            events, notes = _parse_ics_from_emails(params.emails_json)
        except ValueError as exc:
            return ActionResult(action=action, success=False, error=f"Invalid emails_json: {exc}")
        now_ts = time.time()
        cutoff_ts = now_ts + params.lookahead_hours * 3600
        upcoming = [e for e in events if now_ts <= e["dtstart_ts"] <= cutoff_ts]
        if len(events) > len(upcoming):
            notes.append(
                f"{len(events) - len(upcoming)} event(s) start outside the next {params.lookahead_hours} hours "
                "and were not checked."
            )

        issues: list[dict[str, Any]] = []

        if params.check_conflicts:
            for a, b in _find_conflicts(upcoming):
                issue = {
                    "type": "conflict",
                    "severity": "high",
                    "title": "Scheduling Conflict Detected",
                    "detail": (f'"{a["summary"]}" and "{b["summary"]}" overlap. Check your calendar.'),
                    "event_uids": [a["uid"], b["uid"]],
                }
                issues.append(issue)
                if params.notify:
                    _send_os_notification(issue["title"], issue["detail"])

        if params.check_missing_links:
            for e in upcoming:
                if not _has_meeting_link(e):
                    issue = {
                        "type": "missing_link",
                        "severity": "medium",
                        "title": "Missing Meeting Link",
                        "detail": f'"{e["summary"]}" has no video-call link.',
                        "event_uids": [e["uid"]],
                    }
                    issues.append(issue)
                    if params.notify:
                        _send_os_notification(issue["title"], issue["detail"])

        summary = f"Reconciled {len(upcoming)} upcoming event(s); found {len(issues)} issue(s)."
        logger.info(summary)
        return ActionResult(
            action=action,
            success=True,
            output=json.dumps(
                {"events_checked": len(upcoming), "issues": issues, "summary": summary, "notes": notes},
                ensure_ascii=False,
                indent=2,
            ),
            notice=" ".join(notes) or None,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _original_message(params: EmailParams) -> dict[str, str]:
    """Headers of the message being replied to: its Message-ID (to thread under), References, sender and subject.

    From ``in_reply_to`` when the plan names the original's Message-ID (email_fetch reports it as ``message_id``).
    Otherwise read over IMAP from message ``reply_to_uid`` of ``mailbox``, the number email_fetch reported, with the
    folder opened read-only and BODY.PEEK so the original is not marked as read. Empty when the plan names no
    original. Raises ValueError (or an IMAP or network error) when the original cannot be read: a reply that would
    not thread is not sent.
    """
    if params.in_reply_to.strip():
        return {"message_id": params.in_reply_to.strip()}
    number = params.reply_to_uid.strip()
    if not number or "@" in number:
        return {}
    if not number.isdigit():
        raise ValueError(f"reply_to_uid {number!r} is not a message number from email_fetch")
    if not params.imap_host:
        raise ValueError("no IMAP host is set up to read it")
    connection = imaplib.IMAP4_SSL(params.imap_host, ssl_context=ssl.create_default_context(), timeout=30)
    try:
        connection.login(params.username, params.app_password)
        status, _ = connection.select(params.mailbox, readonly=True)
        if status != "OK":
            raise ValueError(f"the folder {params.mailbox} could not be opened")
        status, data = connection.fetch(
            number, "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID REFERENCES FROM REPLY-TO SUBJECT)])"
        )
        raw = next(
            (
                part[1]
                for part in data or []
                if isinstance(part, tuple) and len(part) > 1 and isinstance(part[1], bytes)
            ),
            None,
        )
        if status != "OK" or raw is None:
            raise ValueError(f"message {number} is not in {params.mailbox}")
        headers = email_lib.message_from_bytes(raw)
        message_id = str(headers.get("Message-ID", "") or "").strip()
        if not message_id:
            raise ValueError(f"message {number} has no Message-ID to reply under")
        subject = str(headers.get("Subject", "") or "")
        return {
            "message_id": message_id,
            "references": " ".join(str(headers.get("References", "") or "").split()),
            "reply_to": email_lib.utils.parseaddr(str(headers.get("Reply-To", "") or ""))[1],
            "from": email_lib.utils.parseaddr(str(headers.get("From", "") or ""))[1],
            "subject": str(email_lib.header.make_header(email_lib.header.decode_header(subject))) if subject else "",
        }
    finally:
        try:
            connection.logout()
        except Exception:  # noqa: BLE001
            pass


def _fetched_messages(emails_json: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The messages and notes of an email_fetch output: the object it writes, or a bare list of messages.

    Raises ValueError when the text is not JSON of either shape.
    """
    try:
        data = json.loads(emails_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return [m for m in data["messages"] if isinstance(m, dict)], [str(n) for n in data.get("notes") or []]
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)], []
    raise ValueError("expected the output of email_fetch: an object with messages, or a list of messages")


def _with_notes(text: str, notes: list[str]) -> str:
    """``text`` followed by what the fetch left out, so a summary never reads as covering all mail when it did not."""
    return text if not notes else text + "\n\n" + "\n".join(f"Note: {note}" for note in notes)


def _extract_body(msg: email_lib.message.Message) -> str:
    """Extract plain-text body from a (possibly multipart) email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


# ── ICS / Calendar helpers ────────────────────────────────────────────────────

_MEETING_LINK_PATTERNS: list[str] = [
    "zoom.us/j/",
    "zoom.us/s/",
    "meet.google.com/",
    "teams.microsoft.com/l/meetup",
    "teams.live.com/",
    "meet.jit.si/",
    "webex.com/",
    "bluejeans.com/",
    "gotomeeting.com/",
]


def _extract_calendar(msg: email_lib.message.Message, body: str) -> list[str]:
    """Every invitation in the message, whole: text/calendar parts (attached or inline) and VEVENT blocks in the
    plain-text body, which is read before any cut."""
    found: list[str] = []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_type() == "text/calendar":
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                found.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
    if not found and "BEGIN:VEVENT" in body:
        found.append(body)
    return found


def _parse_ics_from_emails(emails_json: str) -> tuple[list[dict[str, Any]], list[str]]:
    """VEVENT records from the fetched messages, and notes on what could not be read whole.

    Reads each message's whole ``calendar`` parts when email_fetch kept them, else its body. A body that was cut
    with an invitation started but not finished is said in the notes. Raises ValueError for text that is not an
    email_fetch output.
    """
    emails, notes = _fetched_messages(emails_json)
    events: list[dict[str, Any]] = []
    incomplete = 0
    for em in emails:
        texts = list(em.get("calendar") or []) or [em.get("body", "") + "\n" + em.get("subject", "")]
        for text in texts:
            for match in re.finditer(r"BEGIN:VEVENT(.+?)END:VEVENT", text, re.DOTALL):
                evt = _parse_vevent_block(match.group(1))
                if evt:
                    events.append(evt)
            if not em.get("calendar") and text.count("BEGIN:VEVENT") > text.count("END:VEVENT"):
                incomplete += 1
    if incomplete:
        notes = [
            *notes,
            f"{incomplete} invitation(s) were cut short in the message body and could not be read; "
            "fetch the mail again to read them whole.",
        ]
    return events, list(notes)


def _parse_vevent_block(block: str) -> dict[str, Any] | None:
    """Parse a raw VEVENT block into a structured dict."""
    # Unfold RFC 5545 folded lines (CRLF + space/tab = continuation)
    unfolded = re.sub(r"\r?\n[ \t]", "", block)

    props: dict[str, str] = {}
    for line in unfolded.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.split(";")[0].strip().upper()  # drop params like TZID=...
        props[key] = value.strip()

    dtstart = _parse_ics_datetime(props.get("DTSTART", ""))
    if not dtstart:
        return None
    dtend = _parse_ics_datetime(props.get("DTEND", ""))

    start_ts = dtstart.timestamp()
    end_ts = dtend.timestamp() if dtend else start_ts + 3600

    return {
        "uid": props.get("UID", ""),
        "summary": props.get("SUMMARY", "Untitled Event"),
        "description": props.get("DESCRIPTION", ""),
        "location": props.get("LOCATION", ""),
        "url": props.get("URL", ""),
        "dtstart": dtstart.isoformat(),
        "dtend": dtend.isoformat() if dtend else "",
        "dtstart_ts": start_ts,
        "dtend_ts": end_ts,
    }


def _parse_ics_datetime(s: str) -> datetime | None:
    """Parse ICS datetime strings: YYYYMMDDTHHMMSSZ, YYYYMMDDTHHMMSS, YYYYMMDD."""
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%d"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc) if fmt.endswith("Z") else dt
        except ValueError:
            continue
    return None


def _has_meeting_link(event: dict[str, Any]) -> bool:
    """Return True if the event contains a recognised video-call URL."""
    haystack = " ".join(
        [
            event.get("summary", ""),
            event.get("description", ""),
            event.get("location", ""),
            event.get("url", ""),
        ]
    ).lower()
    return any(pat in haystack for pat in _MEETING_LINK_PATTERNS)


def _find_conflicts(
    events: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return pairs of events whose time ranges overlap."""
    conflicts = []
    for i, a in enumerate(events):
        for b in events[i + 1 :]:
            # Overlap condition: a starts before b ends AND a ends after b starts
            if a["dtstart_ts"] < b["dtend_ts"] and a["dtend_ts"] > b["dtstart_ts"]:
                conflicts.append((a, b))
    return conflicts


_WINDOWS_BALLOON = (
    "Add-Type -AssemblyName System.Windows.Forms; "
    "$n = New-Object System.Windows.Forms.NotifyIcon; "
    "$n.Icon = [System.Drawing.SystemIcons]::Information; "
    "$n.Visible = $true; "
    "$n.ShowBalloonTip(6000, $HelioxArgs['title'], $HelioxArgs['body'], [System.Windows.Forms.ToolTipIcon]::Info);"
)


def _send_os_notification(title: str, body: str) -> None:
    """Fire an OS-native desktop notification (best-effort, never raises)."""
    # The body holds calendar event titles, which anyone who sends the person an invitation chooses: it is data for
    # every system's notifier, never part of the code that shows it (pilot.system.script_values).
    from pilot.system.notify import notification_command
    from pilot.system.script_values import powershell_values

    system = platform.system()
    try:
        if system == "Linux":
            subprocess.run(["notify-send", "--app-name=Heliox OS", "--", title, body], timeout=5, check=False)
        elif system == "Darwin":
            subprocess.run(notification_command("macos", title, body), timeout=5, check=False)
        elif system == "Windows":
            # Windows Forms balloon tip — no extra packages required
            with powershell_values(_WINDOWS_BALLOON, {"title": title, "body": body}) as (args, env):
                subprocess.run(args, env=env, timeout=10, check=False)
    except Exception:  # noqa: BLE001
        logger.debug("OS notification failed", exc_info=True)
