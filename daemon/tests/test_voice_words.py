"""Stop and deny in the person's language, and a stop that never becomes a request.

The measured clips are synthetic (espeak-ng in Hindi, Spanish, German and Mandarin, Windows' Helena voice in
Spanish), as Whisper wrote them on 13 November with the English vocabulary prompt: no recording of a person. Every
clip that was said as "no" or "stop" must end as a denial, a stop, or nothing while a task runs or an approval
waits, never as a request. No model, no audio device."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.config import PilotConfig
from pilot.server import VOICE_DENIALS, VOICE_STOP_PHRASES, PendingConfirmation, PilotServer
from pilot.system import voice_words
from pilot.system.addressing import Addressing, Heard, plausibly_addressed


@pytest.fixture(autouse=True)
def nothing_detected_yet(monkeypatch):
    monkeypatch.setattr(voice_words, "_detected", "")


CLIPS = json.loads((Path(__file__).parent / "data" / "voice_stop_words_synthetic.json").read_text("utf-8"))["clips"]


def server_outcome(text: str, language: str, *, question_open: bool = False) -> str:
    """What the voice turn does with ``text`` while a task runs or an approval waits: "deny", "stop", "nothing"
    (not acted on), "queued" (a one-step command, after the task) or "request" (handed on as a change)."""
    normalized = voice_words.normalize(text)
    if voice_words.is_denial(normalized, language):
        return "deny"
    if voice_words.is_stop(normalized, language):
        return "stop"
    owner = SimpleNamespace(_voice_listener=SimpleNamespace(question_open=question_open))
    held = PilotServer._short_word_while_busy(owner, text, normalized, language)
    if held == "unknown" and PilotServer._quick_command(text):
        return "queued"
    return "nothing" if held else "request"


def follow_up_outcome(text: str, language: str) -> str:
    """The addressing rule, a task running, the conversation open, nobody asked a question; what it takes goes on to
    the voice turn (server_outcome)."""
    rule = Addressing(clock=lambda: 100.0)
    rule.task_open = True
    rule.language = language
    decision = rule.decide(Heard(text=text, started=100.0, ended=101.0, voiced_seconds=0.8))
    if not decision.act:
        return "nothing"
    if decision.why == "follow-up: control":
        return "control"
    return server_outcome(text, language)


@pytest.mark.parametrize(
    "clip", CLIPS, ids=lambda c: f"{c['lang']}-{c['kind']}-{c['synth']}-{c['whisper']}-{c['language_setting']}"
)
def test_no_measured_no_or_stop_ever_becomes_a_request(clip):
    language = clip["lang"]
    assert server_outcome(clip["heard"], language) in {"deny", "stop", "nothing"}
    assert follow_up_outcome(clip["heard"], language) in {"control", "deny", "stop", "nothing"}


@pytest.mark.parametrize(
    ("language", "said"),
    [
        ("hi", "नहीं।"), ("hi", "nahin"), ("hi", "रुको"), ("hi", "Ruko."), ("hi", "बस करो"),
        ("es", "No."), ("es", "Para."), ("es", "¡Basta!"), ("es", "Detente."), ("es", "Stop."),
        ("de", "Nein."), ("de", "Stopp!"), ("de", "Halt."), ("de", "Hör auf."), ("de", "Stop."),
        ("zh", "不要。"), ("zh", "停。"), ("zh", "停下"), ("zh", "別說了"), ("zh", "Stop."),
        ("fr", "Non."), ("fr", "Arrête !"), ("it", "Basta."), ("pt", "Não."), ("ja", "ストップ"), ("ja", "止めて"),
    ],
)  # fmt: skip
def test_each_language_s_stop_and_deny_words_as_whisper_writes_them(language, said):
    assert voice_words.is_control_word(said, language)
    assert plausibly_addressed(said, question_open=False, language=language) == "control"
    assert server_outcome(said, language) in {"deny", "stop"}


def test_english_is_always_included_and_the_english_lists_are_unchanged():
    for language in ("en", "hi", "es", "de", "zh", "xx"):
        assert voice_words.is_denial("No.", language) and voice_words.is_stop("Stop.", language)
        assert voice_words.is_stop("That's enough.", language)
    assert VOICE_DENIALS == voice_words.ENGLISH_DENIALS and "not that" in VOICE_DENIALS
    assert VOICE_STOP_PHRASES == voice_words.ENGLISH_STOPS and "hold on" in VOICE_STOP_PHRASES
    # another language's words count only for that language, and only as a whole utterance
    assert not voice_words.is_stop("Para.", "en") and not voice_words.is_stop("para mañana", "es")


def test_the_voice_language_follows_the_setting_or_for_auto_the_system():
    assert voice_words.voice_language("hi") == "hi"
    assert voice_words.voice_language("zh-CN") == "zh"
    assert voice_words.voice_language("auto", "German_Germany") == "de"
    assert voice_words.voice_language("auto", "es_ES.UTF-8") == "es"
    assert voice_words.voice_language("auto", "C.UTF-8") == "en"
    assert voice_words.voice_language("auto", "") == "en"
    # "auto" follows what Whisper last detected, when that is not English
    voice_words.note_detected("hi")
    assert voice_words.voice_language("auto", "English_India") == "hi"
    voice_words.note_detected("en")
    assert voice_words.voice_language("auto", "German_Germany") == "de"
    assert voice_words.has_words_for("de") and not voice_words.has_words_for("sw")


def test_whisper_gets_the_english_vocabulary_only_for_english():
    assert voice_words.transcription_prompt("en") == voice_words.ENGLISH_PROMPT
    assert voice_words.transcription_prompt("auto", "English_United States") == voice_words.ENGLISH_PROMPT
    for language, system in (("hi", None), ("zh", None), ("auto", "German_Germany"), ("auto", "es_ES")):
        prompt = voice_words.transcription_prompt(language, system)
        assert prompt == voice_words.NAMES_PROMPT
        assert "Heliox" in prompt and not any(word in prompt.casefold() for word in ("send", "stop", "save", "deny"))
    # "auto" after a detection: English only when English was detected, whatever the system
    voice_words.note_detected("zh")
    assert voice_words.transcription_prompt("auto", "English_United States") == voice_words.NAMES_PROMPT
    voice_words.note_detected("en")
    assert voice_words.transcription_prompt("auto", "German_Germany") == voice_words.ENGLISH_PROMPT
    assert voice_words.transcription_prompt("en") == voice_words.ENGLISH_PROMPT  # the wake read, always


@pytest.mark.parametrize(
    ("said", "language", "expected"),
    [
        ("Send.", "en", "nothing"),  # said by anyone while a task runs: never a command
        ("Save.", "zh", "nothing"),
        ("Open Chrome.", "en", "queued"),  # a one-step command: after the task, never into it
        ("Okay.", "en", "nothing"),  # an acknowledgement: nothing to say back either
        ("Actually, use Chrome.", "en", "request"),  # a correction that starts like one still goes through
        ("Use the other folder instead.", "en", "request"),  # four words or more: a change, as before
        ("明天早上九点提醒我开会", "zh", "request"),  # a whole sentence in Mandarin is not "short"
        ("Stop.", "en", "deny"),
        ("Undo that.", "en", "nothing_blocked_by_rule_is_wrong"),
    ],
)
def test_the_short_word_rule_while_a_task_runs(said, language, expected):
    outcome = server_outcome(said, language)
    if expected == "nothing_blocked_by_rule_is_wrong":
        assert outcome == "request"  # handled by the undo road, never held back as an unknown short word
    else:
        assert outcome == expected


def test_an_answer_to_heliox_s_question_still_goes_through():
    assert server_outcome("The blue one.", "en", question_open=True) == "request"
    rule = Addressing(clock=lambda: 100.0)
    rule.task_open = True
    rule.heliox_ended_at = 99.5
    rule.question_open = True
    decision = rule.decide(Heard(text="The blue one.", started=100.0, ended=101.0, voiced_seconds=0.8))
    assert decision.act


def test_words_are_counted_in_scripts_without_spaces():
    assert voice_words.spoken_word_count("停。") == 1
    assert voice_words.spoken_word_count("别说了") == 3
    assert voice_words.spoken_word_count("明天早上九点提醒我开会") > 3
    assert voice_words.spoken_word_count("नहीं") == 1
    assert voice_words.normalize("नहीं।") == "नहीं"


@pytest.mark.asyncio
async def test_a_short_unknown_word_never_touches_a_waiting_approval_and_the_stop_word_denies_it():
    config = PilotConfig()
    config.voice.language = "zh"
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_voice_response = AsyncMock()
    server._handle_interject = AsyncMock()
    pending = PendingConfirmation(plan_id="plan-1", event=asyncio.Event())
    server._pending_confirms["plan-1"] = pending
    from pilot.system.voice import voice_turn_addressed

    voice_turn_addressed.set("wake")
    await server._voice_command_dispatch("Send.")
    payload = server._broadcast_notification.await_args.args[1]
    assert payload["status"] == "not_acted_on" and not pending.resolved
    server._handle_interject.assert_not_awaited()
    said = server._speak_voice_response.await_args.args[0]
    assert "Deny" in said
    await server._voice_command_dispatch("Save.")  # said once per approval
    assert server._speak_voice_response.await_count == 1

    server._handle_confirm = AsyncMock(return_value={"status": "ok"})
    await server._voice_command_dispatch("停。")
    confirm = server._handle_confirm.await_args.args[0]
    assert confirm["plan_id"] == "plan-1" and confirm["confirmed"] is False
