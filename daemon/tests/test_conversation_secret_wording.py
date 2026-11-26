"""A request holding a secret is stopped with a sentence that fits how it arrived and says what to do instead.

In the live run of 26 November a typed request with a bearer token was stopped with "I don't use a PIN, password or
code said aloud", and the web address's host and port had been taken out as keys too. No model, no network.
"""

from __future__ import annotations

import pytest

from pilot.conversation.layer import SECRET_NOT_USED, ConversationLayer, secret_not_used
from pilot.conversation.withheld import forget_recent, left_out, withhold
from pilot.system.connections import WHERE

TOKEN = "live10-test-token-7f3a9c21e5"
WEBHOOK = (
    'Send a POST request to http://127.0.0.1:8797/hook with the JSON body {"event": "live10", "ok": true} and the '
    f"header Authorization: Bearer {TOKEN}"
)


@pytest.fixture(autouse=True)
def _nothing_remembered():
    forget_recent()  # values taken out here are not looked for in other tests
    yield
    forget_recent()


def test_an_address_and_port_are_not_taken_for_a_secret():
    found = withhold(WEBHOOK)
    assert found is not None and TOKEN not in found.text
    assert "http://127.0.0.1:8797/hook" in found.text


def test_digits_beside_a_password_in_an_address_are_still_taken_out():
    found = withhold("the password for http://user:4471@host.example/ is that")
    assert found is not None and "4471" not in found.text


async def _take(words: str, source: str) -> list[str]:
    said: list[str] = []

    async def sink(text, channel, key):
        said.append(text)
        return "spoken"

    class _Never:
        local, streams, name, provider = False, True, "fake", "fake"

        async def stream(self, system, messages):
            raise AssertionError("no model is asked about a request holding a secret")
            yield ""  # pragma: no cover

    layer = ConversationLayer(sink=sink, model=lambda: _Never())
    typed_words = left_out(words) if source == "typed" else words  # the server takes it out where typed words arrive
    result = await layer.take(typed_words, source=source)
    assert result.done and result.intent is None
    return said


async def test_a_typed_request_with_a_token_says_typed_and_what_to_do():
    said = await _take(WEBHOOK, "typed")
    assert len(said) == 1 and TOKEN not in said[0]
    assert "aloud" not in said[0] and "typed" in said[0]
    assert "doesn't take passwords, keys or tokens inside a request" in said[0]


@pytest.mark.parametrize(
    ("words", "service"),
    [
        (f"Post 'deploy done' to Slack with the token xoxb-{TOKEN}", "slack"),
        (f"Send 'hi' to my Discord webhook https://discord.com/api/webhooks/123456789/{TOKEN}", "discord"),
        (f"Open an issue on GitHub with the token ghp_{TOKEN}abcdefghij", "github"),
        (f"Send an email to sam@example.com, my mail password is {TOKEN}", "mail"),
    ],
)
async def test_a_service_with_a_place_in_settings_is_pointed_there(words, service):
    said = await _take(words, "typed")
    assert len(said) == 1 and WHERE[service] in said[0], said
    assert TOKEN not in said[0]


async def test_spoken_keeps_its_own_wording():
    said = await _take("type my PIN 4471 into the bank form", "voice")
    assert said == [SECRET_NOT_USED]
    spoken = secret_not_used("post it to Slack with the token [left out: a key]", typed=False)
    assert (
        spoken.startswith("I don't use a PIN, password or code said aloud") and "Settings, Connections, Slack" in spoken
    )


def test_the_sentence_never_quotes_the_words():
    sentence = secret_not_used("pay [left out: a card number] to the plumber", typed=True)
    assert "plumber" not in sentence and "card number you typed" in sentence
