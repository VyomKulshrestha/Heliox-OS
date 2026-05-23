"""Email Agent — reads and sends emails via IMAP/SMTP using App Passwords.

Specialises in:
  - Fetching unread emails securely over IMAP (SSL)
  - Summarising email threads using the model router
  - Drafting and sending replies via SMTP (STARTTLS)

Security model
--------------
Credentials are passed per-action inside ``EmailParams`` and are **never**
persisted to disk by this agent.  Users should store their App Password in
the Heliox OS encrypted vault and inject it at call time.

All outbound send/reply actions are tagged ``requires_confirmation=True`` so
the security gate will always prompt the user before anything is transmitted.
"""

from __future__ import annotations

import email as email_lib
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

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

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

    def __init__(self, model_router: ModelRouter) -> None:
        super().__init__(role=AgentRole.COMMUNICATION, model_router=model_router)

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
    ) -> list[ActionResult]:
        """Dispatch each email action to the appropriate handler."""
        start = time.time()
        self.status = AgentStatus.BUSY

        my_actions = [a for a in plan.actions if self.can_handle(a.action_type)]
        if not my_actions:
            self.status = AgentStatus.IDLE
            return []

        results: list[ActionResult] = []
        for action in my_actions:
            params = action.parameters
            if not isinstance(params, EmailParams):
                results.append(
                    ActionResult(
                        action=action,
                        success=False,
                        error="EmailAgent requires EmailParams",
                    )
                )
                continue

            try:
                if action.action_type == ActionType.EMAIL_FETCH:
                    result = await self._fetch_emails(action, params)
                elif action.action_type == ActionType.EMAIL_SUMMARIZE:
                    result = await self._summarize_emails(action, params)
                elif action.action_type in (ActionType.EMAIL_REPLY, ActionType.API_SEND_EMAIL):
                    result = await self._send_email(action, params)
                elif action.action_type == ActionType.CALENDAR_FETCH:
                    if not isinstance(params, CalendarParams):
                        result = ActionResult(action=action, success=False, error="CalendarParams required")
                    else:
                        result = await self._fetch_calendar_events(action, params)
                elif action.action_type == ActionType.CALENDAR_RECONCILE:
                    if not isinstance(params, CalendarParams):
                        result = ActionResult(action=action, success=False, error="CalendarParams required")
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
            # Respect the caller's limit, newest first
            uid_list = uid_list[-params.max_emails :][::-1]

            emails: list[dict[str, str]] = []
            for uid in uid_list:
                fetch_status, msg_data = mail.fetch(uid, "(RFC822)")
                if fetch_status != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw = msg_data[0][1]
                if not isinstance(raw, bytes):
                    continue

                msg = email_lib.message_from_bytes(raw)
                body = _extract_body(msg)

                emails.append(
                    {
                        "uid": uid.decode(),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "subject": msg.get("Subject", "(no subject)"),
                        "date": msg.get("Date", ""),
                        "body": body[:_MAX_BODY_CHARS],
                    }
                )

                if params.mark_as_read:
                    mail.store(uid, "+FLAGS", "\\Seen")

            mail.logout()
        except imaplib.IMAP4.error as exc:
            return ActionResult(action=action, success=False, error=f"IMAP error: {exc}")

        output = json.dumps(emails, ensure_ascii=False, indent=2)
        logger.info("Fetched %d unread email(s) from %s", len(emails), params.mailbox)
        return ActionResult(action=action, success=True, output=output)

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
            emails: list[dict[str, str]] = json.loads(params.emails_json)
        except json.JSONDecodeError as exc:
            return ActionResult(action=action, success=False, error=f"Invalid emails_json: {exc}")

        if not emails:
            return ActionResult(action=action, success=True, output="No emails to summarise.")

        if self._model is None:
            # Fallback: plain-text list when no LLM is available
            lines = [
                f"{i + 1}. From: {e.get('from', '')} | Subject: {e.get('subject', '')} | Date: {e.get('date', '')}"
                for i, e in enumerate(emails)
            ]
            return ActionResult(action=action, success=True, output="\n".join(lines))

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

        return ActionResult(action=action, success=True, output=summary)

    # ── SMTP: send / reply ────────────────────────────────────────────────────

    async def _send_email(self, action: Any, params: EmailParams) -> ActionResult:
        """Draft (optionally via LLM) and send an email over SMTP with STARTTLS."""
        if not params.smtp_host or not params.username or not params.app_password:
            return ActionResult(
                action=action,
                success=False,
                error="smtp_host, username, and app_password are required to send email",
            )

        recipient = params.to or params.reply_to_uid  # reply_to_uid doubles as To: for API_SEND_EMAIL
        if not recipient:
            return ActionResult(action=action, success=False, error="No recipient specified (set 'to')")

        body = params.reply_body

        # If no body provided, ask the LLM to draft one
        if not body and self._model is not None:
            draft_prompt = (
                f"Draft a professional email reply.\n"
                f"To: {recipient}\n"
                f"Subject: {params.subject}\n"
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
        msg["Subject"] = params.subject or "Re: (no subject)"
        msg.attach(MIMEText(body, "plain", "utf-8"))

        logger.info(
            "Sending email via %s:%d from %s to %s",
            params.smtp_host,
            params.smtp_port,
            params.username,
            recipient,
        )

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
            output=f"Email sent to {recipient} with subject '{msg['Subject']}'",
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

        events = _parse_ics_from_emails(params.emails_json)
        now_ts = time.time()
        cutoff_ts = now_ts + params.lookahead_hours * 3600
        upcoming = [e for e in events if now_ts <= e["dtstart_ts"] <= cutoff_ts]

        logger.info(
            "CALENDAR_FETCH: %d event(s) in next %dh",
            len(upcoming),
            params.lookahead_hours,
        )
        return ActionResult(
            action=action,
            success=True,
            output=json.dumps(upcoming, ensure_ascii=False, indent=2),
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

        events = _parse_ics_from_emails(params.emails_json)
        now_ts = time.time()
        cutoff_ts = now_ts + params.lookahead_hours * 3600
        upcoming = [e for e in events if now_ts <= e["dtstart_ts"] <= cutoff_ts]

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
                {"events_checked": len(upcoming), "issues": issues, "summary": summary},
                ensure_ascii=False,
                indent=2,
            ),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _parse_ics_from_emails(emails_json: str) -> list[dict[str, Any]]:
    """Extract VEVENT records from ICS blocks embedded in email bodies."""
    try:
        emails: list[dict[str, str]] = json.loads(emails_json)
    except (json.JSONDecodeError, TypeError):
        return []

    events: list[dict[str, Any]] = []
    for em in emails:
        text = em.get("body", "") + "\n" + em.get("subject", "")
        for match in re.finditer(r"BEGIN:VEVENT(.+?)END:VEVENT", text, re.DOTALL):
            evt = _parse_vevent_block(match.group(1))
            if evt:
                events.append(evt)
    return events


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


def _send_os_notification(title: str, body: str) -> None:
    """Fire an OS-native desktop notification (best-effort, never raises)."""
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.run(
                ["notify-send", "--app-name=Heliox OS", title, body],
                timeout=5,
                check=False,
            )
        elif system == "Darwin":
            script = f'display notification "{body}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], timeout=5, check=False)
        elif system == "Windows":
            # Windows Forms balloon tip — no extra packages required
            safe_title = title.replace("'", "''")
            safe_body = body.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$n = New-Object System.Windows.Forms.NotifyIcon; "
                "$n.Icon = [System.Drawing.SystemIcons]::Information; "
                "$n.Visible = $true; "
                f"$n.ShowBalloonTip(6000, '{safe_title}', '{safe_body}', "
                "[System.Windows.Forms.ToolTipIcon]::Info);"
            )
            subprocess.run(["powershell", "-Command", ps], timeout=10, check=False)
    except Exception:  # noqa: BLE001
        logger.debug("OS notification failed", exc_info=True)
