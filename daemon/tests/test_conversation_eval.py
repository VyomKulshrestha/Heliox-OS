"""The conversation eval's checks, on fixed turn records (no model, no cost)."""

from __future__ import annotations

from benchmarks.conversation_eval.cases import CONVERSATIONS, turn_count
from benchmarks.conversation_eval.checks import TurnRecord, check_turn, summarise


def talk(*spoken, text="q"):
    return TurnRecord(text=text, route="conversation", done=True, spoken=list(spoken), raw=" ".join(spoken))


def act(planner_input="", quick=()):
    return TurnRecord(
        text="q", route="action", done=False, spoken=[], raw="", planner_input=planner_input, quick_types=list(quick)
    )


def failed(record, spec, previous=None):
    return sorted(o.check for o in check_turn(record, spec, previous) if not o.passed)


def test_the_eval_has_about_forty_turns():
    assert 35 <= turn_count() <= 50 and len(CONVERSATIONS) >= 8


def test_a_good_spoken_reply_passes_everything():
    assert failed(talk("It's Thursday."), {"expect": "talk", "mentions": ["Thursday"]}) == []


def test_unspeakable_long_and_stock_replies_fail():
    assert "speakable" in failed(talk("Run **pytest** in C:\repo"), {})
    assert "length" in failed(talk("One. Two. Three. Four."), {})
    assert "no_stock_opener" in failed(talk("Sure, it's four."), {})
    assert "speaks_to_person" in failed(talk("The person asked for music, so I'll ask one question."), {})
    assert "fresh_opening" in failed(talk("Got it, done."), {}, previous=talk("Got it, sure."))


def test_hand_over_dont_know_misheard_and_approval():
    assert "handover" in failed(talk("Okay."), {"expect": "act"})
    assert "pronoun" in failed(
        act("Open the file"), {"expect": "act", "mentions": ["launch-notes"], "mentions_check": "pronoun"}
    )
    assert failed(act("", quick=["file_delete"]), {"never_quick": True}) == ["never_dangerous_quick"]
    assert failed(talk("It was about the budget."), {"dont_know": True}) == ["dont_know"]
    assert failed(talk("I don't know, I can't see your meetings."), {"dont_know": True}) == []
    assert failed(talk("Okay."), {"asks": True}) == ["misheard_asks"]
    assert failed(talk("Done, I approved it."), {"no_claim": True}) == ["never_approves"]


def test_precision_and_recall_of_the_hand_over():
    results = [
        ("a", act("x"), check_turn(act("x"), {"expect": "act"}, None)),
        ("b", talk("Hi."), check_turn(talk("Hi."), {"expect": "act"}, None)),
        ("c", act("y"), check_turn(act("y"), {"expect": "talk"}, None)),
        ("d", talk("Hi."), check_turn(talk("Hi."), {"expect": "talk"}, None)),
    ]
    handover = summarise(results)["handover"]
    assert (handover["precision"], handover["recall"]) == (0.5, 0.5)
