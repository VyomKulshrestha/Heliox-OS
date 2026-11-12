"""'What can I say?' in two spoken sentences, three examples that work for this person now, and "more"."""

from __future__ import annotations

import pytest

from pilot.conversation.help import SpokenHelp, examples


def test_two_sentences_three_examples_and_more():
    help_ = SpokenHelp(clock=lambda: 0.0)
    first = help_.first(setup="claude", live=True, platform="win32")
    assert first == (
        'You can say things like "open Notepad", "remind me in 20 minutes to stretch", or '
        '"why might my build be slow?". Say "more" for others.'
    )
    assert help_.more("More.", setup="claude", live=True, platform="win32").startswith("You can say things like")


@pytest.mark.parametrize(
    ("platform", "app"), [("win32", "open Notepad"), ("darwin", "open Notes"), ("linux", "open the terminal")]
)
def test_the_app_named_exists_on_this_system(platform, app):
    assert examples(setup="none", live=False, platform=platform)[0] == app


def test_with_no_model_no_question_is_offered_and_without_live_no_live_example():
    offered = examples(setup="none", live=False, platform="win32")
    assert "why might my build be slow?" not in offered and "why did you say that?" not in offered


def test_more_only_right_after_help_and_it_runs_out_politely():
    now = [0.0]
    help_ = SpokenHelp(clock=lambda: now[0])
    assert help_.more("more", setup="none", live=False) is None  # nothing asked yet
    help_.first(setup="none", live=False)
    pages = [help_.more("more", setup="none", live=False) for _ in range(4)]
    assert pages[-1].startswith("That's most of it")
    now[0] += 500
    assert help_.more("more", setup="none", live=False) is None  # too long ago
