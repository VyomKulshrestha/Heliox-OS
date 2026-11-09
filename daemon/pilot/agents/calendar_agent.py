"""Calendar agent for local .ics parsing and CalDAV integration."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import caldav
import icalendar

from pilot.actions import Action, ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, BaseAgent
from pilot.agents.registry import auto_register

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride
    from pilot.security.vault import KeyVault

logger = logging.getLogger("pilot.agents.calendar_agent")


@auto_register
class CalendarAgent(BaseAgent):
    """Specialist agent for managing calendar events (local .ics and remote CalDAV)."""

    def __init__(
        self,
        model_router: ModelRouter,
        config: PilotConfig,
        vault: KeyVault,
    ) -> None:
        super().__init__(role=AgentRole.CALENDAR, model_router=model_router)
        self._config = config
        self._vault = vault

    def get_capabilities(self) -> list[AgentCapability]:
        return [
            AgentCapability(
                action_type=ActionType.CALENDAR_PARSE,
                description="Parse local .ics files to extract events",
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_SYNC,
                description="Sync with a remote CalDAV calendar",
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_CREATE_EVENT,
                description="Create a new event in the calendar",
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_LIST_EVENTS,
                description="List upcoming events from the calendar",
            ),
            AgentCapability(
                action_type=ActionType.CALENDAR_DELETE_EVENT,
                description="Delete an event from the calendar",
            ),
        ]

    def get_system_prompt(self) -> str:
        return (
            "You are the CALENDAR AGENT for Heliox OS. "
            "You manage time-based events by parsing local .ics files and "
            "integrating with CalDAV servers. You can list, create, and delete events."
        )

    def can_handle(self, action_type: ActionType) -> bool:
        return action_type in {
            ActionType.CALENDAR_PARSE,
            ActionType.CALENDAR_SYNC,
            ActionType.CALENDAR_CREATE_EVENT,
            ActionType.CALENDAR_LIST_EVENTS,
            ActionType.CALENDAR_DELETE_EVENT,
        }

    async def handle_task(
        self,
        user_input: str,
        plan: ActionPlan,
        context: dict[str, Any] | None = None,
        scope_override: TaskScopeOverride | None = None,
    ) -> list[ActionResult]:
        # AgentOrchestrator gates these direct CalDAV calls and applies the
        # caller's scope override before dispatch.
        results = []
        for action in plan.actions:
            if not self.can_handle(action.action_type):
                continue

            payload = action.parameters.model_dump() if hasattr(action.parameters, "model_dump") else {}

            if action.action_type == ActionType.CALENDAR_PARSE:
                res = await self._handle_parse(action, payload)
            elif action.action_type == ActionType.CALENDAR_SYNC:
                res = await self._handle_sync(action, payload)
            elif action.action_type == ActionType.CALENDAR_CREATE_EVENT:
                res = await self._handle_create_event(action, payload)
            elif action.action_type == ActionType.CALENDAR_LIST_EVENTS:
                res = await self._handle_list_events(action, payload)
            elif action.action_type == ActionType.CALENDAR_DELETE_EVENT:
                res = await self._handle_delete_event(action, payload)
            else:
                res = ActionResult(action=action, success=False, error="Unsupported action")

            results.append(res)

        return results

    async def _handle_parse(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        import json

        explicit_path = str(payload.get("file_path") or "").strip()
        paths = [explicit_path] if explicit_path else list(self._config.calendar.ics_files)
        if not paths:
            return ActionResult(action=action, success=False, error="No .ics file or configured calendar source")

        events, sources, warnings = self._read_local_calendars(paths)
        if not sources:
            return ActionResult(action=action, success=False, error="; ".join(warnings))
        return ActionResult(
            action=action,
            success=True,
            output=json.dumps({"events": events, "sources": sources, "warnings": warnings}),
        )

    @staticmethod
    def _event_from_component(component: Any, *, source: str, calendar_id: str = "") -> dict[str, Any]:
        start = component.get("dtstart")
        end = component.get("dtend")
        return {
            "uid": str(component.get("uid", "")),
            "summary": str(component.get("summary", "")),
            "start": start.dt.isoformat()
            if start and hasattr(start.dt, "isoformat")
            else str(start.dt if start else ""),
            "end": end.dt.isoformat() if end and hasattr(end.dt, "isoformat") else str(end.dt) if end else None,
            "description": str(component.get("description", "")),
            "calendar_id": calendar_id,
            "source": source,
        }

    def _read_local_calendars(self, paths: list[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        events: list[dict[str, Any]] = []
        sources: list[str] = []
        warnings: list[str] = []
        for configured_path in paths:
            path = Path(configured_path).expanduser()
            try:
                if path.suffix.lower() != ".ics":
                    raise ValueError("calendar source must use the .ics extension")
                calendar = icalendar.Calendar.from_ical(path.read_bytes())
                source = str(path.resolve())
                sources.append(source)
                for component in calendar.walk():
                    if component.name == "VEVENT":
                        events.append(self._event_from_component(component, source=source))
            except Exception as exc:
                warning = f"{path}: {exc}"
                warnings.append(warning)
                logger.warning("Failed to read local calendar source: %s", warning)
        return events, sources, warnings

    async def _get_caldav_client(self):
        if not self._config.calendar.enabled:
            raise ValueError("CalDAV integration is disabled in Settings")

        url = self._config.calendar.caldav_url
        username = self._config.calendar.caldav_username
        password = ""

        if self._config.calendar.caldav_password_provider:
            password = await self._vault.get_key(self._config.calendar.caldav_password_provider) or ""

        if not url or not username or not password:
            raise ValueError("CalDAV configuration is incomplete (URL, username, and saved password are required)")

        parsed_url = urlparse(url)
        is_loopback = parsed_url.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed_url.scheme != "https" and not (parsed_url.scheme == "http" and is_loopback):
            raise ValueError("CalDAV URL must use HTTPS (plain HTTP is allowed only for localhost testing)")

        client = caldav.DAVClient(url=url, username=username, password=password)
        return client

    async def test_connection(self) -> dict[str, Any]:
        """Verify the saved CalDAV credentials without changing any events."""
        try:
            client = await self._get_caldav_client()
            calendars = client.principal().calendars()
            return {
                "status": "ok",
                "calendars": [str(getattr(calendar, "name", "")) for calendar in calendars],
            }
        except Exception as exc:
            logger.warning("CalDAV connection test failed: %s", exc)
            return {"status": "error", "message": str(exc), "calendars": []}

    @staticmethod
    def _select_calendar(client: Any, calendar_id: str | None = None) -> Any:
        calendars = client.principal().calendars()
        if not calendars:
            raise ValueError("The CalDAV account has no calendars")
        if not calendar_id:
            return calendars[0]
        for calendar in calendars:
            if str(getattr(calendar, "name", "")) == calendar_id:
                return calendar
        raise ValueError(f"Calendar not found: {calendar_id}")

    async def _handle_sync(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        try:
            import json

            client = await self._get_caldav_client()
            principal = client.principal()
            calendars = principal.calendars()
            return ActionResult(
                action=action, success=True, output=json.dumps({"calendars": [c.name for c in calendars]})
            )
        except Exception as e:
            logger.error(f"CalDAV sync failed: {e}")
            return ActionResult(action=action, success=False, error=str(e))

    async def _handle_create_event(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        # Simplified implementation
        summary = payload.get("summary")
        start = payload.get("start")
        end = payload.get("end")

        if not all([summary, start]):
            return ActionResult(action=action, success=False, error="Missing summary or start time")

        try:
            client = await self._get_caldav_client()
            calendar = self._select_calendar(client, payload.get("calendar_id"))
            calendar.save_event(
                dtstart=datetime.fromisoformat(start),
                dtend=datetime.fromisoformat(end) if end else None,
                summary=summary,
            )
            return ActionResult(action=action, success=True, output="Event created")
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            return ActionResult(action=action, success=False, error=str(e))

    async def _handle_list_events(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        import json

        parsed_events, sources, warnings = await self.collect_events(payload.get("calendar_id"))
        if not sources:
            error = "; ".join(warnings) or "No local .ics files are configured and CalDAV is disabled"
            return ActionResult(action=action, success=False, error=error)
        return ActionResult(
            action=action,
            success=True,
            output=json.dumps({"events": parsed_events, "sources": sources, "warnings": warnings}),
        )

    async def collect_events(self, calendar_id: str | None = None) -> tuple[list[dict[str, Any]], list[str], list[str]]:
        """Every event in the local .ics files and, when it is set up, the CalDAV calendar. Read-only."""
        parsed_events, sources, warnings = self._read_local_calendars(list(self._config.calendar.ics_files))
        if self._config.calendar.enabled:
            try:
                client = await self._get_caldav_client()
                calendar = self._select_calendar(client, calendar_id)
                calendar_name = str(getattr(calendar, "name", ""))
                sources.append(f"caldav:{calendar_name}")
                for event in calendar.events():
                    remote_calendar = icalendar.Calendar.from_ical(event.data)
                    for component in remote_calendar.walk():
                        if component.name == "VEVENT":
                            parsed_events.append(
                                self._event_from_component(
                                    component,
                                    source="caldav",
                                    calendar_id=calendar_name,
                                )
                            )
            except Exception as exc:
                warning = f"CalDAV: {exc}"
                warnings.append(warning)
                logger.warning("Failed to list remote calendar events: %s", exc)
        return parsed_events, sources, warnings

    async def _handle_delete_event(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        event_uid = str(payload.get("event_uid") or "").strip()
        if not event_uid:
            return ActionResult(action=action, success=False, error="Missing event_uid")
        try:
            client = await self._get_caldav_client()
            calendar = self._select_calendar(client, payload.get("calendar_id"))
            event = calendar.event_by_uid(event_uid)
            if event is None:
                return ActionResult(action=action, success=False, error="Calendar event was not found")
            event.delete()
            return ActionResult(action=action, success=True, output="Event deleted")
        except Exception as e:
            logger.error("Failed to delete event: %s", e)
            return ActionResult(action=action, success=False, error=str(e))
