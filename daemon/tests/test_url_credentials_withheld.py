"""A password or token inside a web address is taken out before any model sees the words, and the request stops.

The host, port, path and user name stay readable. Every road a request's words take into Heliox is checked: typed and
spoken turns (the conversation layer and the interactive gate), the queue, another app through MCP, a phone's words
already taken out where they arrived, and the autonomous engine that runs triggers' stored words. Fakes only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pilot.agents.autonomous import AutonomousExecutor, JobStatus
from pilot.conversation.layer import ConversationLayer
from pilot.conversation.withheld import forget_recent, holds_url_credential, left_out, withhold
from pilot.server import PilotServer


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_recent()  # values taken out here are not looked for in other tests
    yield
    forget_recent()


TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWX0123456789"

TAKEN_OUT = [
    (
        "Open http://sam:hunter2@host.example:8080/reports?week=3",
        "hunter2",
        "http://sam:",
        "@host.example:8080/reports?week=3",
    ),
    ("Fetch ftp://:pa55word@files.example.com/a.txt", "pa55word", "ftp://:", "@files.example.com/a.txt"),
    ("Get https://sam:p%40ss%21w0rd@host.example/", "p%40ss%21w0rd", "https://sam:", "@host.example/"),
    (
        f"Clone https://{TOKEN}:x-oauth-basic@github.com/me/repo.git",
        TOKEN,
        "https://",
        ":x-oauth-basic@github.com/me/repo.git",
    ),
    (f"Clone https://{TOKEN}@github.com/me/repo.git", TOKEN, "https://", "@github.com/me/repo.git"),
    ("Open https://sam:12345678@host.example:443/x", "12345678", "https://sam:", "@host.example:443/x"),
    ("Open https://sam:4471@host.example/x", "4471", "https://sam:", "@host.example/x"),
]

NOT_A_CREDENTIAL = [
    "Open https://example.com/users/@sam?to=sam@example.com",
    "Clone ssh://git@github.com/me/repo.git",
    "Email sam@example.com about the report",
    "Open https://sam@host.example/profile",
]


@pytest.mark.parametrize(("words", "secret", "before", "after"), TAKEN_OUT)
def test_the_secret_is_taken_out_and_the_rest_stays_readable(words, secret, before, after):
    found = withhold(words)
    assert found is not None and secret not in found.text, found
    assert before in found.text and after in found.text, found.text
    assert secret in found.values  # matched as one whole value, digits only or not
    assert holds_url_credential(words) and holds_url_credential(found.text)


@pytest.mark.parametrize("words", NOT_A_CREDENTIAL)
def test_an_at_sign_that_is_not_a_credential_is_left_alone(words):
    assert withhold(words) is None and not holds_url_credential(words)


def test_a_digit_only_password_is_taken_whole_even_beside_a_port():
    found = withhold("the password for https://sam:4471@host.example:8080/ is that one")
    assert found is not None and "4471" not in found.text and ":8080/" in found.text


# ── every road ──


async def _layer_says(words: str, source: str) -> list[str]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    class _Never:
        local, streams, name, provider = False, True, "fake", "fake"

        async def stream(self, system, messages):
            raise AssertionError("no model sees a request holding a password")
            yield ""  # pragma: no cover

    result = await ConversationLayer(sink=sink, model=lambda: _Never()).take(words, source=source)
    assert result.done and result.intent is None
    return said


@pytest.mark.parametrize("source", ["typed", "voice"])
async def test_a_turn_stops_with_the_sentence_for_how_it_arrived(source):
    words = "Open http://sam:hunter2@host.example:8080/reports"
    said = await _layer_says(left_out(words) if source == "typed" else words, source)
    assert len(said) == 1 and "hunter2" not in said[0]
    assert ("typed" in said[0]) is (source == "typed") and ("aloud" in said[0]) is (source == "voice")


class _Socket:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(message)


@pytest.mark.parametrize(("source", "typed"), [("text", True), ("chat", True), ("phone", False), ("mcp", False)])
async def test_the_interactive_gate_stops_it_before_anything_is_planned_or_kept(source, typed):
    socket = _Socket()
    words = "Open http://sam:hunter2@host.example:8080/reports"
    if source == "phone":
        words = left_out(words)  # the phone's words arrive already taken out (pilot.air_handoff)
    params, stopped = await PilotServer._secret_left_out(SimpleNamespace(), {"input": words, "source": source}, socket)
    assert stopped is not None and "hunter2" not in params["input"] and "hunter2" not in str(socket.sent)
    assert ("you typed" in stopped["message"]) is typed and "hunter2" not in stopped["message"]


async def test_the_queue_never_keeps_it():
    server = SimpleNamespace(_durable_tasks=object(), _queue_shutdown=False)
    response = await PilotServer._queue_interactive_request(
        server, {"input": "Open http://sam:hunter2@host.example/", "source": "text"}
    )
    assert response["status"] == "error" and "hunter2" not in response["message"]


def test_another_app_through_mcp_is_stopped():
    with pytest.raises(ValueError) as stopped:
        PilotServer._mcp_task_input({"input": "Open http://sam:hunter2@host.example/"})
    assert "hunter2" not in str(stopped.value) and "in that request" in str(stopped.value)
    assert PilotServer._mcp_task_input({"input": "my door code is 2580, open the door app"}).count("2580") == 0


async def test_a_trigger_s_stored_words_never_run_with_it():
    engine = AutonomousExecutor(planner=None, executor=None, verifier=None, decomposer=None)
    job = await engine.submit("Fetch http://sam:hunter2@host.example/feed.xml", source="trigger")
    assert job.status is JobStatus.FAILED and "hunter2" not in job.goal and "hunter2" not in job.result_summary
    assert job.job_id not in engine._active_tasks
