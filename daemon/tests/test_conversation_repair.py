"""A sentence repaired as it is said (pilot.conversation.repair): only the repair is acted on, never both halves, and
what does not clearly read as a repair is left as said. The real path, for every kind of person, is the session "a
sentence repaired as it is said: only the repair is acted on" (sessions.json)."""

from __future__ import annotations

import pytest

from pilot.conversation.repair import resolve


@pytest.mark.parametrize(
    ("said", "meant"),
    [
        ("open, no, close Spotify", "close Spotify"),  # a new verb: the whole new request
        ("open Spotify, sorry, close Spotify", "close Spotify"),
        ("mute, I mean unmute", "unmute"),
        ("close, uh, open Spotify", "open Spotify"),  # a pause, then a new verb
        ("open no close spotify", "close spotify"),  # no commas, "no" right before a verb
        ("open, no, close, no, minimize Spotify", "minimize Spotify"),  # the last repair wins
        ("open Notepad, no, Spotify", "open Spotify"),  # the object replaced
        ("open Notepad no Spotify", "open Spotify"),
        ("delete the logs, no wait, the screenshots", "delete the screenshots"),
        ("turn off Wi-Fi, no, Bluetooth", "turn off Bluetooth"),
        ("search for flights to Lisbon, no, Porto", "search for flights to Porto"),  # the last place replaced
        ("email Ana about the invoice, no, about the report", "email Ana about the report"),
        ("delete the files in Downloads, no, in Desktop", "delete the files in Desktop"),
        ("set a timer for ten, no, fifteen minutes", "set a timer for fifteen minutes"),  # the number replaced
        ("remind me at four, no, five to call mum", "remind me at five to call mum"),
        ("what time is it, no, what day is it", "what day is it"),
        ("open Spotify, no", ""),  # taken back: nothing is acted on
    ],
)
def test_only_the_repair_is_acted_on(said, meant):
    assert resolve(said) == meant


@pytest.mark.parametrize(
    "said",
    [
        "No.",
        "No, stop.",  # a denial or a stop, never a repair
        "Wait, stop",
        "I said no, close it",
        "set the volume to no more than fifty",  # "no more than" is not a marker
        "open Spotify, um, please",  # a pause, not a repair
        "open Spotify",
        "what is the capital of Peru",
    ],
)
def test_what_is_not_a_repair_is_left_as_said(said):
    assert resolve(said) is None
