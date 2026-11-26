"""An offer is built from an imperative form of what would be done, never by splicing a question in.

In the live11talk replay, "How do I check my disk space?" got "Do you want me to how do I check my disk space?". No
model, no network.
"""

from __future__ import annotations

import pytest

from pilot.conversation.handover import HandoverIntent
from pilot.conversation.layer import offer_sentence


def _intent(request: str, say: str = "") -> HandoverIntent:
    return HandoverIntent(verb="check", object="", words=request, request=request, source="model", say=say)


@pytest.mark.parametrize(
    ("request_text", "say", "offer"),
    [
        ("how do I check my disk space?", "", "Do you want me to do that?"),
        ("How do I check my disk space?", "I'll check your disk space.", "Do you want me to check your disk space?"),
        ("What's taking up space on my drive?", "", "Do you want me to do that?"),
        ("Is my disk nearly full?", "Let me check the disk usage.", "Do you want me to check the disk usage?"),
        (
            "How do I open the registry editor?",
            "I'll open the registry editor.",
            "Do you want me to open the registry editor?",
        ),
        ("Check my disk space", "", "Do you want me to check your disk space?"),
        ("Open the registry editor", "", "Do you want me to open the registry editor?"),
        ("Could you open Notepad?", "", "Do you want me to open Notepad?"),
        ("Can I delete this?", "", "Do you want me to do that?"),
        ("", "", "Do you want me to do that?"),
    ],
)
def test_the_offer(request_text, say, offer):
    assert offer_sentence(_intent(request_text, say)) == offer


def test_still():
    intent = _intent("Delete the thesis drafts folder")
    assert offer_sentence(intent, still=True) == "Do you still want me to delete the thesis drafts folder?"
    assert offer_sentence(_intent("why would I?"), still=True) == "Do you still want me to do that?"
