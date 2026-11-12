"""Routing decisions for addressed utterances, on a table drawn from ordinary use and from what the owner's
microphone actually produced (the daemon log of rehearsals in August and November 2026, and the real-speech cases
in test_voice_real_speech.py). Text only.

"talk" means the utterance goes to the fast model, which answers or asks (conversation or unclear); what matters
most is that nothing misheard or ambiguous reaches the planner as an action.
"""

from __future__ import annotations

import time

import pytest

from pilot.conversation.router import TurnKind, route

TALK = {TurnKind.CONVERSATION, TurnKind.UNCLEAR}

# (utterance, expected) where expected is a TurnKind or "talk".
OWNER_TRANSCRIPTS = [
    (". open github.", TurnKind.ACTION),
    (", show system information.", TurnKind.ACTION),
    ("true system information", "talk"),
    ("it's all it", "talk"),
    ("we were interested in", "talk"),
    ("launch it", TurnKind.UNCLEAR),
    ("launch", TurnKind.UNCLEAR),
    ("open", TurnKind.UNCLEAR),
    ("Open", TurnKind.UNCLEAR),
    ("yes", "talk"),
    ("open the screen", TurnKind.ACTION),
    ("who is responding to all these things", TurnKind.CONVERSATION),
    ("open translation", TurnKind.ACTION),
    ("who was this all about", TurnKind.CONVERSATION),
    ("how do you", TurnKind.CONVERSATION),
    ("how to do it", TurnKind.CONVERSATION),
    ("Open, close, close, close.", TurnKind.UNCLEAR),
    ("open up", TurnKind.UNCLEAR),
    ("what", "talk"),
    ("Okay.", "talk"),
    ("i'm john", "talk"),
    ("is that the end", TurnKind.CONVERSATION),
    ("how many people", TurnKind.CONVERSATION),
    ("what happened", TurnKind.CONVERSATION),
    ("Heliox.", TurnKind.UNCLEAR),
    ("go on, go on, go on", TurnKind.UNCLEAR),
    ("where we used to", "talk"),
    ("search, search, search", TurnKind.UNCLEAR),
    ("close", TurnKind.UNCLEAR),
    ("when you change the code", TurnKind.CONVERSATION),
    ("go ahead and open", TurnKind.UNCLEAR),
    ("are you a boss here", TurnKind.CONVERSATION),
    ("is it", TurnKind.CONVERSATION),
    ("Open, open, open, open", TurnKind.UNCLEAR),
    ("Click, click, click", TurnKind.UNCLEAR),
    ("Let's try and take a look.", "talk"),
    ("Let's begin.", "talk"),
    ("Go ahead and look for your own", TurnKind.UNCLEAR),
    ("GitHub, Google, Google, Google, Google, Google, Google, Google", TurnKind.UNCLEAR),
    ("Google, Google, Google.", TurnKind.UNCLEAR),
    ("Hello. Hello", TurnKind.CONVERSATION),
    ("good afternoon", TurnKind.CONVERSATION),
    ("what can I say", TurnKind.LOCAL),
    ("what can I say to you", TurnKind.LOCAL),
    ("Where we store home", "talk"),
    ("Stop", TurnKind.CONTROL),
    ("No.", TurnKind.CONTROL),
    ("Skip the backup.", TurnKind.CONTROL),
    ("Undo that", TurnKind.CONTROL),
    ("remind me that the launch call is Monday, the 12th of November at 10.", TurnKind.ACTION),
    ("The works remind me that the launch call is Monday, the twelfth of November at ten", "talk"),
]

ORDINARY_USE = [
    # requests to act
    ("Open Notepad", TurnKind.ACTION),
    ("open notepad please", TurnKind.ACTION),
    ("Can you open my downloads folder?", TurnKind.ACTION),
    ("Could you please close Spotify", TurnKind.ACTION),
    ("Set a timer for ten minutes", TurnKind.ACTION),
    ("Remind me to call Priya at four", TurnKind.ACTION),
    ("Find the invoice from October", TurnKind.ACTION),
    ("search the web for flights to Lisbon", TurnKind.ACTION),
    ("Turn on dark mode", TurnKind.ACTION),
    ("Take a screenshot", TurnKind.ACTION),
    ("zip the launch folder", TurnKind.ACTION),
    ("Delete the old backups in my downloads", TurnKind.ACTION),
    ("Send an email to Sam saying I'm running late", TurnKind.ACTION),
    ("play some music", TurnKind.ACTION),
    ("Mute", TurnKind.ACTION),
    ("I need you to move the screenshots into a new folder", TurnKind.ACTION),
    ("go ahead and rename the report to final", TurnKind.ACTION),
    ("Install the latest Python", TurnKind.ACTION),
    ("check my calendar for tomorrow", TurnKind.ACTION),
    ("How much disk space do I have left?", TurnKind.ACTION),
    ("Is Spotify running?", TurnKind.ACTION),
    ("what's the battery level", TurnKind.ACTION),
    ("Look up the weather in Pune", TurnKind.ACTION),
    ("please organize my desktop", TurnKind.ACTION),
    ("schedule a meeting with Priya on Friday at three", TurnKind.ACTION),
    # conversation
    ("How are you?", TurnKind.CONVERSATION),
    ("What do you think about this error?", TurnKind.CONVERSATION),
    ("Explain what you just did", TurnKind.CONVERSATION),
    ("Why did that fail?", TurnKind.CONVERSATION),
    ("What's the difference between a process and a thread?", TurnKind.CONVERSATION),
    ("Tell me a joke", TurnKind.CONVERSATION),
    ("What time is it?", TurnKind.CONVERSATION),
    ("thanks, that was quick", TurnKind.CONVERSATION),
    ("that's not what I meant", TurnKind.CONVERSATION),
    ("I'm tired today", TurnKind.CONVERSATION),
    ("Can you explain recursion?", TurnKind.CONVERSATION),
    ("What should I work on next?", TurnKind.CONVERSATION),
    ("How do I undo a git commit?", TurnKind.CONVERSATION),
    ("What did you just do?", TurnKind.CONVERSATION),
    ("Who are you?", TurnKind.CONVERSATION),
    ("Do you remember my name?", TurnKind.CONVERSATION),
    ("Good morning", TurnKind.CONVERSATION),
    ("I think the build is broken", TurnKind.CONVERSATION),
    ("the meeting got moved to three", TurnKind.CONVERSATION),
    ("Describe what's on my screen", TurnKind.CONVERSATION),
    ("Show me how to make a pivot table", TurnKind.CONVERSATION),
    ("Let's take a look at it together", "talk"),
    # control and local
    ("wait", TurnKind.CONTROL),
    ("cancel", TurnKind.CONTROL),
    ("never mind", TurnKind.CONTROL),
    ("hold on", TurnKind.CONTROL),
    ("no, not that one", "talk"),  # a correction that refers to the conversation
    ("don't send it", TurnKind.CONTROL),
    ("that's enough", TurnKind.CONTROL),
    ("What's on my to-do list?", TurnKind.LOCAL),
    ("What did you do today?", TurnKind.LOCAL),
    ("Say that again", TurnKind.LOCAL),
    ("what reminders do I have", TurnKind.LOCAL),
    ("help", TurnKind.LOCAL),
]

# A control phrase that continues as an instruction is routed by the whole sentence.
CONTROL_THEN_INSTRUCTION = [
    ("That's enough for the backup folder, now zip it", TurnKind.ACTION),
    ("stop there and then open the report", TurnKind.ACTION),
    ("hold on, delete the old backups instead", TurnKind.ACTION),
    ("wait, not that folder, the other one", TurnKind.CONVERSATION),
    ("no, the second one", TurnKind.CONVERSATION),
    ("enough of that, what's next on my list", TurnKind.CONVERSATION),
    ("okay so open notepad", TurnKind.ACTION),
    ("Stop it now", TurnKind.CONTROL),
    ("enough of this", TurnKind.CONTROL),
    ("Wait, stop", TurnKind.CONTROL),
]

TABLE = OWNER_TRANSCRIPTS + ORDINARY_USE + CONTROL_THEN_INSTRUCTION


def test_the_table_is_big_enough():
    assert len(TABLE) >= 80


@pytest.mark.parametrize(("text", "expected"), TABLE)
def test_routing(text, expected):
    decided = route(text)
    if expected == "talk":
        assert decided.kind in TALK, decided
    else:
        assert decided.kind is expected, decided


def test_nothing_misheard_reaches_the_planner_as_an_action():
    garbled = [text for text, expected in OWNER_TRANSCRIPTS if expected in ("talk", TurnKind.UNCLEAR)]
    assert all(route(text).kind is not TurnKind.ACTION for text in garbled)


@pytest.mark.parametrize("text", ["open it", "send that to Priya", "delete those", "open it again"])
def test_a_pronoun_is_resolved_by_the_conversation_or_asked_about(text):
    assert route(text, conversation_open=True).kind is TurnKind.CONVERSATION
    assert route(text, conversation_open=False).kind in (TurnKind.UNCLEAR, TurnKind.ACTION)
    assert route("open it", conversation_open=False).kind is TurnKind.UNCLEAR


def test_the_action_carries_its_verb_and_target():
    decided = route("Could you open my launch folder")
    assert (decided.kind, decided.verb, decided.target) == (TurnKind.ACTION, "open", "my launch folder")


def test_routing_is_fast():
    started = time.perf_counter()
    for _ in range(20):
        for text, _expected in TABLE:
            route(text)
    per_call_ms = (time.perf_counter() - started) * 1000 / (20 * len(TABLE))
    assert per_call_ms < 25.0  # about 0.02 ms when idle; the bound only catches a pathological pattern


def test_the_zip_request_names_what_the_control_phrase_named():
    decided = route("That's enough for the backup folder, now zip it")
    assert (decided.verb, decided.target) == ("zip", "the backup folder")


def test_routing_depends_only_on_the_words_and_the_conversation_state():
    # No clock, model or I/O is consulted: the same input gives the same route every time, under any load.
    for text, _expected in TABLE:
        assert len({(route(text).kind, route(text).reason) for _ in range(3)}) == 1
