"""What a model's history keeps of each offer (pilot.personal.wiring.offer_in_history), through deliver_offer: an
offer built from what Heliox knows about the person keeps its words only when
pilot.conversation.adapters.personal_in_prompt says personal lines may go to that model (a model on this computer, or
conversation.share_profile "auto"); otherwise a marker naming the kind and nothing else. The routine record's offers
keep their words only for a model on this computer, under every value of the switch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.personal.offers import INTRO, TIMELINE, Offer
from pilot.personal.wiring import OFFERED_HERE, PROFILE_OFFER_MARKERS, deliver_offer

TEXTS = {
    "habit": "You're on a 4-day streak for water and haven't logged it today. Want me to log it?",
    "stale_goal": 'Your goal "launch the podcast" has been quiet for a while. Want me to remind you?',
    "goal_step": 'On "launch the podcast", your next step is book a guest.',
    "goal_progress": 'That sounds like progress on your goal "launch the podcast". Want me to note it?',
    "todos": 'You still have 3 things on your list; the first is "call Ana". Want me to read the list?',
    "routine": "It's nearly 6 pm, when you stop work. Want me to wrap up the day?",
    "usual_app": "You usually open Outlook about now. Want me to open it?",
}


async def history_of(offer: Offer, *, local: bool, share: str) -> str:
    config = PilotConfig()
    config.conversation.share_profile = share
    kept: list[str] = []
    layer = SimpleNamespace(memory=SimpleNamespace(add_heliox=kept.append), _model=lambda: SimpleNamespace(local=local))
    server = SimpleNamespace(
        config=config, _broadcast_notification=AsyncMock(), _voice_listener=None, _conversation=layer
    )
    await deliver_offer(server, offer)
    assert len(kept) == 1
    return kept[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(TEXTS))
@pytest.mark.parametrize(
    ("local", "share", "words"),
    [(True, "auto", True), (True, "off", True), (False, "auto", True), (False, "off", False)],
    ids=["local-auto", "local-off", "cloud-auto", "cloud-off"],
)
async def test_each_profile_offer(kind, local, share, words):
    offer = Offer(key=f"k:{kind}", kind=kind, text=TEXTS[kind], request="x", why="y")
    kept = await history_of(offer, local=local, share=share)
    assert kept == (TEXTS[kind] if words else PROFILE_OFFER_MARKERS[kind])


@pytest.mark.asyncio
@pytest.mark.parametrize("share", ["auto", "off"])
@pytest.mark.parametrize("local", [True, False])
async def test_the_routine_records_offer(local, share):
    offer = Offer(key="learned:open:outlook", kind="learned_routine", text="You often ask me to open outlook around "
                  "nine. Want me to do it now?", request="open outlook", why="")  # fmt: skip
    assert await history_of(offer, local=local, share=share) == (offer.text if local else OFFERED_HERE)


@pytest.mark.asyncio
@pytest.mark.parametrize("offer", [INTRO, TIMELINE], ids=["get_to_know", "timeline"])
async def test_offers_that_hold_nothing_about_the_person_keep_their_words(offer):
    assert await history_of(offer, local=False, share="off") == offer.text


def test_markers_name_the_kind_and_nothing_else():
    for kind, marker in PROFILE_OFFER_MARKERS.items():
        assert marker.startswith("(made an offer on this computer, ") and marker.endswith(")")
        assert not any(word in marker for word in ("podcast", "Outlook", "water", "Ana", "6 pm")), kind
