from unittest.mock import MagicMock, patch

import pytest

from pilot.actions import Action, ActionPlan, ActionType, CalendarParams
from pilot.agents.calendar_agent import CalendarAgent


@pytest.fixture
def mock_router():
    return MagicMock()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.calendar.caldav_url = "http://localhost/caldav"
    config.calendar.caldav_username = "user"
    config.calendar.caldav_password_provider = "cal_pass"
    return config


@pytest.fixture
def mock_vault():
    vault = MagicMock()
    vault.get_secret.return_value = "password"
    return vault


@pytest.mark.asyncio
async def test_calendar_agent_parse(mock_router, mock_config, mock_vault, tmp_path):
    agent = CalendarAgent(mock_router, mock_config, mock_vault)

    # Create a dummy .ics file
    ics_file = tmp_path / "test.ics"
    ics_file.write_text("""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Test Event
DTSTART:20231027T100000Z
DTEND:20231027T110000Z
DESCRIPTION:Test Description
END:VEVENT
END:VCALENDAR""")

    plan = ActionPlan(
        actions=[Action(action_type=ActionType.CALENDAR_PARSE, parameters=CalendarParams(file_path=str(ics_file)))]
    )

    results = await agent.handle_task("parse my calendar", plan)
    assert len(results) == 1
    assert results[0].success
    import json

    output = json.loads(results[0].output)
    assert len(output["events"]) == 1
    assert output["events"][0]["summary"] == "Test Event"


@pytest.mark.asyncio
async def test_calendar_agent_sync_mocked(mock_router, mock_config, mock_vault):
    agent = CalendarAgent(mock_router, mock_config, mock_vault)

    plan = ActionPlan(actions=[Action(action_type=ActionType.CALENDAR_SYNC, parameters=CalendarParams())])

    with patch("caldav.DAVClient") as mock_dav:
        mock_client = MagicMock()
        mock_dav.return_value = mock_client
        mock_principal = MagicMock()
        mock_client.principal.return_value = mock_principal
        mock_calendar = MagicMock()
        mock_calendar.name = "Test Calendar"
        mock_principal.calendars.return_value = [mock_calendar]

        results = await agent.handle_task("sync calendar", plan)
        assert len(results) == 1
        assert results[0].success
        import json

        output = json.loads(results[0].output)
        assert output["calendars"] == ["Test Calendar"]
