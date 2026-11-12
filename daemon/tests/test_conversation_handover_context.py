"""A hand-over after some talk: the planner's request is the person's own words, the turns before travel as context,
and a reply that holds back never hands anything over (the cases the handover-context eval found)."""

from __future__ import annotations

import asyncio

from pilot.conversation.context import ContextSources, Screen
from pilot.conversation.layer import ASK_WHAT, ConversationLayer


class Replies:
    name, provider, streams, local = "fake", "fake", True, False

    def __init__(self, *replies, pieces: int = 1):
        self.replies = list(replies)
        self.pieces = pieces
        self.calls = 0

    async def stream(self, system, messages):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else "Okay."
        size = max(1, len(reply) // self.pieces)
        for start in range(0, len(reply), size):  # in small pieces, as a stream arrives
            yield reply[start : start + size]
            await asyncio.sleep(0)


class Sink:
    def __init__(self):
        self.said = []

    async def __call__(self, text, channel, key):
        self.said.append(text)
        return "spoken"


def layer_with(model, **kwargs):
    sink = Sink()
    return ConversationLayer(sink=sink, model=lambda: model, hold_after=60.0, **kwargs), sink


async def talk(layer, *lines):
    for line in lines:
        await layer.take(line)


DELETE_DECLINED = (
    "I can't tell which files are the old installers from here, so I won't delete anything yet. "
    "I'll delete them once you confirm."
)


async def test_the_planner_gets_the_persons_words_and_the_model_reading_only_as_context():
    model = Replies(
        "Got it.",
        "Okay.",
        'ACT {"verb": "zip", "object": "the second file", "request": "Zip q4-summary.docx on the desktop", '
        '"say": "Zipping the second report."}',
    )
    layer, _ = layer_with(model)
    await talk(layer, "I have two reports on my desktop, q3-summary.docx and q4-summary.docx.", "I zipped the first.")
    result = await layer.take("Do that for the second file too.")
    assert not result.done
    assert result.planner_input == "Do that for the second file too."
    assert result.context is not None
    planner = result.context.for_planner()
    assert "The person: I have two reports on my desktop, q3-summary.docx and q4-summary.docx." in planner
    assert "Heliox understood the request as: Zip q4-summary.docx on the desktop" in planner
    assert "Zip q4-summary.docx" not in result.context.for_reviewers()
    assert "Got it." not in result.context.for_reviewers()
    params = result.execute_params()
    assert params["_conversation_context"] is result.context


async def test_a_router_hand_over_also_carries_the_earlier_constraint():
    layer, _ = layer_with(Replies("Okay.", "Noted."))
    await talk(layer, "My Downloads folder has invoices from many years.", "I only care about the 2026 ones.")
    result = await layer.take("Move the invoices into my Documents folder.")
    assert not result.done and result.intent.source == "router"
    assert result.planner_input == "Move the invoices into my Documents folder."
    assert "- I only care about the 2026 ones." in result.context.for_reviewers()


async def test_a_first_request_has_no_context():
    layer, _ = layer_with(Replies())
    result = await layer.take("Open Notepad.")
    assert not result.done and result.context is None
    assert "_conversation_context" not in result.execute_params()


async def test_a_decline_with_a_promise_hands_nothing_over():
    model = Replies("Okay.", "Okay.", DELETE_DECLINED)
    layer, sink = layer_with(model)
    await talk(layer, "I'm cleaning up my downloads folder.", "The old installers from last year are taking up space.")
    result = await layer.take("Delete those.")
    assert result.done
    assert result.intent is None
    assert "so I won't delete anything yet." in " ".join(sink.said)
    # "I'll delete them once you confirm." waits on an answer: said as it is, never taken as a promise.


async def test_a_spoken_promise_with_a_wider_verb_hands_over():
    model = Replies("Okay.", "Okay.", "I'll get that message to Priya now.")
    layer, sink = layer_with(model)
    await talk(layer, "Priya is my manager.", "I need to tell her I'll be late to the design review.")
    result = await layer.take("Send her a message about it.")
    assert not result.done
    assert result.planner_input == "Send her a message about it."
    assert "Priya is my manager." in result.context.for_planner()
    assert "I'll get that message" not in " ".join(sink.said)


async def test_a_decline_in_words_with_an_act_line_after_them_hands_nothing_over_and_never_speaks_the_act():
    reply = (
        "I'm not sure which installers you mean, so I won't delete anything yet. "
        'ACT {"verb": "delete", "object": "those", "request": "Delete the old installers in Downloads", '
        '"say": "Deleting the installers."}'
    )
    for pieces in (1, 7, 40):  # the ACT line arriving whole, or split across many stream pieces
        layer, sink = layer_with(Replies(reply, pieces=pieces))
        result = await layer.take("Delete those.")
        assert result.done and result.intent is None, pieces
        heard = " ".join(sink.said)
        assert "won't delete anything yet" in heard
        assert "ACT" not in heard and "{" not in heard and "Deleting" not in heard


async def test_an_act_line_whose_own_sentence_holds_back_hands_nothing_over():
    reply = (
        'ACT {"verb": "delete", "object": "those", "request": "Delete those files", "say": "Which files do you mean?"}'
    )
    layer, sink = layer_with(Replies(reply))
    result = await layer.take("Delete those.")
    assert result.done and result.intent is None
    assert sink.said == ["Which files do you mean?"]
    sure_of_nothing = 'ACT {"verb": "delete", "object": "those", "request": "Delete those", "say": "I won\'t guess."}'
    layer, sink = layer_with(Replies(sure_of_nothing))
    result = await layer.take("Delete those.")
    assert result.done and sink.said == ["I won't guess."]


async def test_a_plain_act_line_still_hands_over_and_words_then_act_without_a_decline_too():
    layer, _ = layer_with(
        Replies('ACT {"verb": "open", "object": "notepad", "request": "Open Notepad", "say": "Opening Notepad."}')
    )
    assert not (await layer.take("Could you get Notepad up for me")).done
    late = 'Sure. ACT {"verb": "open", "object": "notepad", "request": "Open Notepad", "say": "Opening Notepad."}'
    layer, sink = layer_with(Replies(late, pieces=9))
    result = await layer.take("Could you get Notepad up for me")
    assert not result.done
    assert all("ACT" not in line and "{" not in line for line in sink.said)


async def test_what_live_read_from_the_screen_stays_out_of_a_cloud_planners_context():
    layer, _ = layer_with(Replies("Okay."), screen_text_ok=lambda: False)
    layer.memory.add_heliox("Your editor says the build failed on line 40 of secrets.py.", source="live")
    await layer.take("Hmm, okay.")
    result = await layer.take("Open Notepad.")
    assert result.context is not None
    assert "secrets.py" not in result.context.for_planner()
    layer, _ = layer_with(Replies("Okay."), screen_text_ok=lambda: True)
    layer.memory.add_heliox("Your editor says the build failed on line 40 of secrets.py.", source="live")
    await layer.take("Hmm, okay.")
    assert "secrets.py" in (await layer.take("Open Notepad.")).context.for_planner()


async def test_after_the_fast_model_saw_screen_text_heliox_lines_stay_out_unless_allowed():
    async def screen(local):
        return Screen(app="Notepad", excerpt="the field says hunter2")

    model = Replies("It ends with hunter2.")
    layer = ConversationLayer(
        sink=Sink(),
        model=lambda: model,
        hold_after=60.0,
        sources=ContextSources(screen_async=screen),
        screen_text_ok=lambda: False,
    )
    await layer.take("What does my note end with?")
    result = await layer.take("Open Notepad.")
    assert result.context is not None
    assert "hunter2" not in result.context.for_planner()
    assert "What does my note end with?" in result.context.for_planner()


def test_ask_what_is_a_short_question():
    assert ASK_WHAT.endswith("?") and len(ASK_WHAT.split()) <= 6


async def test_typed_chat_carries_the_same_context():
    from types import SimpleNamespace

    from pilot.conversation.adapters import typed_turn

    layer, _ = layer_with(Replies("Okay.", "Noted."))
    server = SimpleNamespace(_conversation=layer)

    class Ws:
        async def send(self, message):
            pass

    for line in ("My Downloads folder has invoices from many years.", "I only care about the 2026 ones."):
        await typed_turn(server, line, Ws(), lambda name, data: (name, data))
    typed = await typed_turn(
        server, "Move the invoices into my Documents folder.", Ws(), lambda name, data: (name, data)
    )
    assert not typed.done and typed.planner_input == "Move the invoices into my Documents folder."
    assert "- I only care about the 2026 ones." in typed.context.for_reviewers()


def test_the_eval_labels_what_must_not_happen_for_every_case():
    from benchmarks.conversation_eval.handover_context import CASES, broken_rules

    assert all(case["must_not"] for case in CASES)
    vague = [case for case in CASES if "hand_over" in case["must_not"]]
    assert len(vague) >= 3
    assert broken_rules(vague[0], handed_over=True, found={}, promised=False) == ["hand_over"]
    assert broken_rules(vague[0], handed_over=False, found={}, promised=True) == ["empty_promise"]
    named = CASES[0]
    assert broken_rules(named, handed_over=True, found={"object": True, "place": False}, promised=False) == [
        "bare_hand_over"
    ]


async def test_a_held_back_reply_whose_promise_followed_a_comma_ends_on_a_whole_question():
    from pilot.conversation.layer import ASK_MORE

    reply = (
        "I'm not sure which message app you use or what time the review starts, "
        "so I'll send her a short note saying you'll be late."
    )
    layer, sink = layer_with(Replies("Okay.", "Okay.", reply, pieces=5))
    await talk(layer, "Priya is my manager.", "I need to tell her I'll be late to the design review.")
    result = await layer.take("Send her a message about it.")
    assert result.done and result.intent is None
    assert sink.said[-1] == ASK_MORE
    assert not any("I'll send" in line for line in sink.said)
