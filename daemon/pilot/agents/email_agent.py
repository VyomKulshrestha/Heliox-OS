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
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

from pilot.actions import ActionPlan, ActionResult, ActionType, EmailParams
from pilot.agents.base_agent import AgentCapability, AgentRole, AgentStatus, BaseAgent

if TYPE_CHECKING:
    from pilot.models.router import ModelRouter

logger = logging.getLogger("pilot.agents.email_agent")

EMAIL_ACTION_TYPES: set[ActionType] = {
    ActionType.EMAIL_FETCH,
    ActionType.EMAIL_SUMMARIZE,
    ActionType.EMAIL_REPLY,
    ActionType.API_SEND_EMAIL,
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
