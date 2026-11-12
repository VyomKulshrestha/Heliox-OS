"""Personal sessions (benchmarks/conversation_eval/personal_sessions.json): what a person tells Heliox, across a restart
of the session server, for every kind of person, with what the model is sent and what is stored checked directly.

A session that fails because of another workstream's code is listed in KNOWN with the cause; it runs as a strict
expected failure, so it shows up the moment that code changes."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.conversation_eval.sessions import KINDS, PERSONAL_PACK, load_pack, run_session
from pilot.conversation.adapters import _profile_lines
from pilot.conversation.context import ContextSources, ConversationMemory, assemble, never_sent
from pilot.conversation.layer import PERSONAL_WORD_CAPS, SPOKEN_WORD_CAP, ConversationLayer
from pilot.conversation.planner_context import build_context, people_meant
from pilot.workflows.user_profile import UserProfile

SESSIONS = load_pack(PERSONAL_PACK)
DAEMON = Path(__file__).resolve().parents[1]

_EVERYONE: dict[str, str] = {}
KNOWN: dict[tuple[str, str], str] = {(name, kind): cause for name, cause in _EVERYONE.items() for kind in KINDS}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("session", SESSIONS, ids=[s["name"] for s in SESSIONS])
def test_a_personal_session(session, kind, request):
    cause = KNOWN.get((session["name"], kind))
    if cause:
        request.applymarker(pytest.mark.xfail(reason=cause, strict=True))
    _, failures = asyncio.run(run_session(session, kind))
    assert failures == [], "\n".join(failures)


def test_the_pack_covers_what_the_round_asked_for():
    names = " ".join(s["name"] for s in SESSIONS)
    for topic in ("told once", "corrected", "forgotten", "not learned", "what a model is sent"):
        assert topic in names, topic
    assert all(any(t.get("event") == "restart" for t in s["turns"]) for s in SESSIONS)


# ── what a model is sent about the person ──


@pytest.fixture
def profile(tmp_path):
    async def nothing(*_args, **_kwargs):
        return None

    store = UserProfile(tmp_path / "profile.json", remember=nothing, forget=nothing)

    async def fill():
        await store.set_name("Sam")
        await store.set_style("length", "short")
        await store.add_person("Priya", "manager")
        await store.add("about", "I live in Pune")
        await store.add("about", "I'm training for a marathon")

    asyncio.run(fill())
    # Stored by an older version, before the profile refused it (pilot.personal.never_kept): still never sent.
    path = tmp_path / "profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["about"].append({"id": "older00001", "text": "my salary is 90k", "at": 1.0})
    path.write_text(json.dumps(data), encoding="utf-8")
    return store


def test_a_cloud_model_is_sent_only_what_the_answer_needs(profile):
    server = SimpleNamespace(_user_profile=profile)
    peru = _profile_lines(server, "what's the capital of Peru?", local=False)
    assert peru == ["Call them Sam.", "Answer style, length: short"]
    manager = " ".join(_profile_lines(server, "what does my manager like for lunch?", local=False))
    assert "Priya" in manager and "Pune" not in manager and "marathon" not in manager
    home = " ".join(_profile_lines(server, "where should I eat near home tonight?", local=False))
    assert "Pune" in home and "Priya" not in home


def test_a_model_on_this_computer_may_be_sent_more(profile):
    server = SimpleNamespace(_user_profile=profile)
    lines = " ".join(_profile_lines(server, "what's the capital of Peru?", local=True))
    assert "Priya" in lines and "Pune" in lines and "marathon" in lines


def test_nothing_never_kept_is_sent_whatever_is_stored(profile):
    server = SimpleNamespace(_user_profile=profile)
    for local in (True, False):
        sources = ContextSources(profile_lines=lambda text="", local=local: _profile_lines(server, text, local=local))
        block, _ = assemble(sources, ConversationMemory(), text="what about my salary?")
        assert "90k" not in block and "salary" not in block
    assert never_sent("About them: I have a dentist appointment on Friday")
    assert never_sent("About them: the wifi password is hunter2")
    assert not never_sent("About them: I write code for a living")


def test_an_extra_line_does_not_repeat_the_profile_or_carry_what_is_never_sent():
    extra = (
        "Also relevant about the person: Answer style, length: short. Person they know: Priya is my manager. "
        "About them: my salary is 90k."
    )
    sources = ContextSources(profile_lines=lambda text="": ["Answer style, length: short"], extra=[lambda: extra])
    block, _ = assemble(sources, ConversationMemory(), text="my manager")
    assert block.count("Answer style") == 1 and "Priya" in block and "90k" not in block


# ── the person's answer length, and the people a hand-over means ──


@pytest.mark.parametrize(("length", "cap"), [("short", 15), ("detailed", 60), ("", SPOKEN_WORD_CAP)])
def test_the_spoken_cap_follows_the_answer_style(length, cap):
    async def sink(*_args):
        return "spoken"

    layer = ConversationLayer(sink=sink, model=lambda: None, sources=ContextSources(answer_length=lambda: length))
    assert layer._word_cap() == cap == PERSONAL_WORD_CAPS.get(length, SPOKEN_WORD_CAP)
    block, _ = assemble(layer.sources, ConversationMemory(), text="why is the sky blue?")
    assert ("short answers" in block) is (length == "short")
    assert ("more detail" in block) is (length == "detailed")


def test_a_hand_over_says_who_my_manager_is():
    assert people_meant("message my manager that I'm late", {"manager": "Priya", "sister": "Ana"}) == (
        "Their manager is Priya.",
    )
    context = build_context(
        [], words="message my manager that I'm late", screen_text_ok=False, people={"manager": "Priya"}
    )
    assert context is not None and "Their manager is Priya." in context.for_planner()
    assert build_context([], words="open notepad", screen_text_ok=False, people={"manager": "Priya"}) is None


# ── the model never claims to have saved or changed something ──


def test_the_prompt_only_says_the_record_is_the_truth():
    """The blanket "you cannot save anything" made the model deny a save that happened; the line is conditional."""
    from pilot.conversation.style import SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT):
        assert "You cannot save" not in prompt
        assert '("done there: ...")' in prompt and "never contradict it" in prompt


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _layer():
    async def sink(*_args):
        return "spoken"

    clock = Clock()
    return ConversationLayer(sink=sink, model=lambda: None, clock=clock), clock


@pytest.mark.parametrize(
    ("words", "starts"),
    [
        ("No, call me Sam", "I didn't change your name."),
        ("don't interrupt me while I'm writing", "I didn't change when I speak up."),
        ("give me more detail from now on", "I didn't change how long my answers are."),
        ("don't forget I'm vegan", "I didn't keep that."),
        ("from now on answer in French", "I didn't change anything."),
    ],
)
def test_a_request_to_keep_or_change_that_no_handler_took_gets_one_true_sentence(words, starts):
    layer, _clock = _layer()
    assert (layer._kept_or_changed(words, None) or "").startswith(starts)


@pytest.mark.parametrize(
    "words",
    ["call me later about it", "remember when we went to Paris?", "remember to buy milk", "why is the sky blue?"],
)
def test_other_words_still_go_to_the_model(words):
    layer, _clock = _layer()
    assert layer._kept_or_changed(words, None) is None


def test_did_you_get_that_is_answered_from_the_record():
    layer, clock = _layer()
    saved = ("Nice to meet you, Sam. I'll call you that.", ("remembered that",), clock())
    assert layer._kept_or_changed("Did you get that right?", saved) == (
        "Yes, I remembered that. Nice to meet you, Sam. I'll call you that."
    )
    nothing = ('I didn\'t keep that. Say "remember that" and then the fact.', (), clock())
    assert layer._kept_or_changed("did you save that?", nothing) == "No, nothing was saved or changed."
    answered = ("It's nine a m.", (), clock())
    assert layer._kept_or_changed("did you get that?", answered) is None  # not about a save: the usual road
    clock.t += 61.0
    assert layer._kept_or_changed("did you get that?", saved) is None  # too long ago to be about that turn


def test_a_turn_answered_in_code_tells_a_cloud_model_what_was_done_without_the_data():
    from pilot.server import ANSWERED_ON_THIS_COMPUTER, answered_here

    assert answered_here(()) == ANSWERED_ON_THIS_COMPUTER
    assert answered_here(("remembered that",)) == "(answered on this computer; done there: remembered that)"


@pytest.mark.parametrize(
    ("words", "told"),
    [
        ("No, call me Sam.", True),
        ("stay quiet unless I ask", True),
        ("remember that I like tea", True),
        ("from now on answer in French", True),
        ("keep your answers short", True),
        ("remember to buy milk", False),
        ("call me later about it", False),
        ("remember when we went to Paris?", False),
        ("what do you remember about me?", False),
        ("what's the capital of Peru?", False),
    ],
)
def test_words_asking_to_keep_or_change_something_that_reach_the_model_are_marked_as_not_done(words, told):
    """Heliox's code takes these before the model; when they reach it, nothing was saved, and the turn says so."""
    from pilot.conversation.context import NOTHING_SAVED

    block, _ = assemble(ContextSources(), ConversationMemory(), text=words)
    assert (NOTHING_SAVED in block) is told


# ── every Heliox folder is a scratch one (pilot.real_folders) ──


def test_every_heliox_folder_is_under_the_drivers_scratch_root():
    from benchmarks.conversation_eval import sessions as S
    from pilot.real_folders import heliox_folders, refuse_real_folders

    assert refuse_real_folders("this test") == "test root"
    assert all(path.resolve().is_relative_to(S.TEST_ROOT.resolve()) for path in heliox_folders().values())


def test_a_session_folder_outside_the_root_is_refused(tmp_path):
    from benchmarks.conversation_eval import sessions as S

    with pytest.raises(RuntimeError, match="not inside"):
        S._isolate(tmp_path / "outside")


def test_a_paid_run_copies_the_persons_model_and_conversation_settings_into_its_own_config(monkeypatch, tmp_path):
    """The person's file is read, never built into a config; the run's config lives under the scratch root, so its save
    cannot reach theirs."""
    import pilot.config as config_module
    import pilot.real_folders as real_folders
    from benchmarks.conversation_eval import OWNER_CONVERSATION_FIELDS
    from benchmarks.conversation_eval import sessions as S
    from pilot.config import PilotConfig

    theirs = tmp_path / ".config" / "heliox-os" / "config.toml"
    theirs.parent.mkdir(parents=True)
    theirs.write_text(
        '[model]\nprovider = "cloud"\ncloud_provider = "claude"\nbudget_monthly_limit_usd = 25.0\n\n'
        '[conversation]\nclaude_model = "claude-haiku-5-5"\nmax_reply_tokens = 300\nshare_screen_text = "on"\n'
        'provisional_answers = "on"\n\n[security]\nroot_enabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(real_folders, "_owner_folder", lambda variable, fallback: tmp_path / fallback / "heliox-os")
    before = theirs.stat().st_mtime_ns
    config = PilotConfig()
    copied = real_folders.apply_owner_settings(config, ("model", "conversation"), only=OWNER_CONVERSATION_FIELDS)
    assert "conversation.claude_model" in copied and "model.cloud_provider" in copied
    assert config.model.budget_monthly_limit_usd == 25.0 and config.conversation.max_reply_tokens == 300
    assert config.security.root_enabled is False  # only [model] and [conversation]
    # A pack's own privacy and feature settings are never inherited: only the fields the constant names.
    assert config.conversation.share_screen_text == "auto" and config.conversation.provisional_answers == "auto"
    assert "conversation.share_screen_text" not in copied
    assert config_module.CONFIG_FILE.resolve().is_relative_to(S.TEST_ROOT.resolve())  # where its save would go
    assert theirs.stat().st_mtime_ns == before


def _child(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "HELIOX_TEST_ROOT"}  # as a script run by hand would start
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=DAEMON, timeout=120
    )


def test_run_as_a_script_the_driver_takes_a_scratch_root_before_pilot_reads_a_path():
    done = _child(
        "import benchmarks.conversation_eval.sessions as S, pilot.config as c\n"
        "from pilot.real_folders import refuse_real_folders\n"
        "print(S.TEST_ROOT)\nprint(refuse_real_folders('check'))\nprint(c.CONFIG_FILE)\n"
    )
    assert done.returncode == 0, done.stderr
    root, kind, config_file = done.stdout.split("\n")[:3]
    assert kind == "test root" and config_file.startswith(root)


def test_run_as_a_script_after_pilot_read_the_real_paths_the_driver_refuses():
    done = _child("import pilot.config\nimport benchmarks.conversation_eval.sessions\n")
    assert done.returncode != 0 and "must run before pilot.config is imported" in done.stderr
