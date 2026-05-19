import json
from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, EmailParams
from pilot.agents.email_agent import EmailAgent


def extract_email_body(msg_str: str) -> str:
    from email import policy
    from email.parser import BytesParser

    parsed = BytesParser(policy=policy.default).parsebytes(msg_str.encode())
    return parsed.get_body(preferencelist=("plain")).get_content()


@pytest.fixture
def email_agent():
    mock_router = AsyncMock()
    mock_router.complete.return_value = "Mocked LLM reply."
    agent = EmailAgent(model_router=mock_router)
    # The actual implementation of BaseAgent sets self._model
    agent._model = mock_router
    return agent


@pytest.mark.asyncio
async def test_fetch_emails_mocked(email_agent, imap_mock):
    params = EmailParams(imap_host="imap.fake.com", username="user", app_password="pwd", mailbox="INBOX", max_emails=5)
    action = Action(action_type=ActionType.EMAIL_FETCH, parameters=params, requires_root=False)
    plan = ActionPlan(actions=[action])

    results = await email_agent.handle_task("fetch emails", plan)

    assert len(results) == 1
    result = results[0]
    assert result.success is True

    emails = json.loads(result.output)
    assert len(emails) == 2
    # The results are reversed (newest first) by email_agent.py: uid_list = uid_list[-params.max_emails :][::-1]
    # UIDs: b"2", b"1"
    assert emails[0]["from"] == "another@example.com"
    assert "This is a test email 2" in emails[0]["body"]
    assert emails[1]["from"] == "test@example.com"
    assert "This is a test email 1" in emails[1]["body"]


@pytest.mark.asyncio
async def test_send_email_mocked(email_agent, smtp_mock):
    params = EmailParams(
        smtp_host="smtp.fake.com",
        smtp_port=587,
        username="user",
        app_password="pwd",
        to="test@example.com",
        subject="Hello Response",
        reply_body="This is a mocked reply.",
    )
    action = Action(action_type=ActionType.API_SEND_EMAIL, parameters=params)
    plan = ActionPlan(actions=[action])

    results = await email_agent.handle_task("send email", plan)

    assert len(results) == 1
    result = results[0]
    assert result.success is True

    assert len(smtp_mock) == 1
    assert smtp_mock[0]["to"] == ["test@example.com"]
    assert smtp_mock[0]["from"] == "user"
    assert "This is a mocked reply." in extract_email_body(smtp_mock[0]["msg"])


@pytest.mark.asyncio
async def test_reply_flow(email_agent, imap_mock, smtp_mock):
    # 1. Fetch
    fetch_params = EmailParams(imap_host="imap.fake.com", username="user", app_password="pwd", mailbox="INBOX")
    fetch_action = Action(action_type=ActionType.EMAIL_FETCH, parameters=fetch_params)

    # 2. Summarize
    summarize_params = EmailParams(
        emails_json='[{"from": "test@example.com", "subject": "Hello", "date": "today", "body": "test"}]'
    )
    summarize_action = Action(action_type=ActionType.EMAIL_SUMMARIZE, parameters=summarize_params)

    # 3. Reply
    reply_params = EmailParams(
        smtp_host="smtp.fake.com",
        smtp_port=587,
        username="user",
        app_password="pwd",
        to="test@example.com",
        subject="Re: Hello",
        reply_body="",  # Empty body to trigger LLM
    )
    reply_action = Action(action_type=ActionType.EMAIL_REPLY, parameters=reply_params)

    plan = ActionPlan(actions=[fetch_action, summarize_action, reply_action])

    results = await email_agent.handle_task("full flow", plan)

    assert len(results) == 3
    assert all(r.success for r in results)

    # Verify smtp_mock got the LLM generated reply
    assert len(smtp_mock) == 1
    assert "Mocked LLM reply." in extract_email_body(smtp_mock[0]["msg"])
