"""Calendar agent for local .ics parsing and CalDAV integration."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import caldav
import icalendar

from pilot.actions import ActionPlan, ActionResult, ActionType
from pilot.agents.base_agent import AgentCapability, AgentRole, BaseAgent
from pilot.agents.registry import auto_register

if TYPE_CHECKING:
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter
    from pilot.security.gateway import TaskScopeOverride
    from pilot.security.vault import Vault

logger = logging.getLogger("pilot.agents.calendar_agent")


@auto_register
class CalendarAgent(BaseAgent):
    """Specialist agent for managing calendar events (local .ics and remote CalDAV)."""

    def __init__(
        self,
        model_router: ModelRouter,
        config: PilotConfig,
        vault: Vault,
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
        # Note: this agent talks to caldav/icalendar directly rather than
        # through the shared Executor, so it isn't gateway-scoped today —
        # scope_override is accepted for interface consistency with
        # BaseAgent.handle_task but has nothing to apply it to here.
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
        file_path = payload.get("file_path")
        if not file_path:
            return ActionResult(action=action, success=False, error="Missing file_path")

        try:
            import json

            with open(file_path, "rb") as f:
                cal = icalendar.Calendar.from_ical(f.read())
                events = []
                for component in cal.walk():
                    if component.name == "VEVENT":
                        events.append(
                            {
                                "summary": str(component.get("summary")),
                                "start": component.get("dtstart").dt.isoformat()
                                if hasattr(component.get("dtstart").dt, "isoformat")
                                else str(component.get("dtstart").dt),
                                "end": (
                                    component.get("dtend").dt.isoformat()
                                    if hasattr(component.get("dtend").dt, "isoformat")
                                    else str(component.get("dtend").dt)
                                )
                                if component.get("dtend")
                                else None,
                                "description": str(component.get("description", "")),
                            }
                        )
                return ActionResult(action=action, success=True, output=json.dumps({"events": events}))
        except Exception as e:
            logger.error(f"Failed to parse .ics file: {e}")
            return ActionResult(action=action, success=False, error=str(e))

    async def _get_caldav_client(self):
        url = self._config.calendar.caldav_url
        username = self._config.calendar.caldav_username
        password = ""

        if self._config.calendar.caldav_password_provider:
            password = self._vault.get_secret(self._config.calendar.caldav_password_provider) or ""

        if not url or not username:
            raise ValueError("CalDAV configuration missing (url/username)")

        client = caldav.DAVClient(url=url, username=username, password=password)
        return client

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
            calendar = client.principal().calendars()[0]  # Use first calendar for now
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
        try:
            import json

            client = await self._get_caldav_client()
            calendar = client.principal().calendars()[0]
            events = calendar.events()
            parsed_events = []
            for event in events:
                ical = icalendar.Calendar.from_ical(event.data)
                for component in ical.walk():
                    if component.name == "VEVENT":
                        parsed_events.append(
                            {
                                "summary": str(component.get("summary")),
                                "start": str(component.get("dtstart").dt),
                            }
                        )
            return ActionResult(action=action, success=True, output=json.dumps({"events": parsed_events}))
        except Exception as e:
            logger.error(f"Failed to list events: {e}")
            return ActionResult(action=action, success=False, error=str(e))

    async def _handle_delete_event(self, action: Action, payload: dict[str, Any]) -> ActionResult:
        # This would normally require an event ID or similar
        return ActionResult(action=action, success=False, error="Delete not fully implemented")
