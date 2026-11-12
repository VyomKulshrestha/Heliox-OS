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


def test_the_model_is_told_it_cannot_save_and_how_the_person_can():
    from pilot.conversation.style import SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, TYPED_SYSTEM_PROMPT):
        assert "You cannot save or remember anything for later" in prompt
        assert "never say it was" in prompt and '"remember that ..."' in prompt and "Settings" in prompt
        assert "(answered on this computer; done there: ...)" in prompt  # how it is told what Heliox did


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


# ── the session driver owns every Heliox folder ──


def test_every_heliox_path_is_inside_the_drivers_test_root():
    from benchmarks.conversation_eval import sessions as S

    assert S.outside(S.TEST_ROOT) == {}
    assert {
        "CONFIG_FILE",
        "RESTRICTIONS_FILE",
        "DATA_DIR",
        "STATE_DIR",
        "CACHE_DIR",
        "RUNTIME_DIR",
        "SETUP_DIR",
    } <= set(S.config_paths())


def test_the_driver_refuses_a_session_when_a_path_is_outside(monkeypatch, tmp_path):
    import pilot.config as config_module
    from benchmarks.conversation_eval import sessions as S

    elsewhere = tmp_path / "somewhere-else"
    monkeypatch.setattr(config_module, "CONFIG_FILE", elsewhere / "config.toml")
    with pytest.raises(RuntimeError, match="CONFIG_FILE"):
        S.refuse_outside(S.TEST_ROOT)
    with pytest.raises(RuntimeError, match="refusing to start a session"):
        S._isolate(S.TEST_ROOT / "a-session")
    assert elsewhere / "config.toml" == config_module.CONFIG_FILE  # nothing was moved before refusing


def test_a_session_folder_outside_the_root_is_refused(tmp_path):
    from benchmarks.conversation_eval import sessions as S

    with pytest.raises(RuntimeError, match="not inside"):
        S._isolate(tmp_path / "outside")


def test_a_paid_run_copies_only_what_it_needs_into_its_own_config(tmp_path):
    """The person's file is read once, read-only; the config built from it lives in the test root, so its save cannot
    reach theirs."""
    import pilot.config as config_module
    from benchmarks.conversation_eval import sessions as S

    theirs = tmp_path / "config.toml"
    theirs.write_text(
        '[model]\nprovider = "cloud"\ncloud_provider = "claude"\nbudget_monthly_limit_usd = 25.0\n'
        'cloud_model = "claude-x"\n\n[conversation]\nclaude_model = "claude-haiku-5-5"\n\n'
        "[security]\nroot_enabled = true\n",
        encoding="utf-8",
    )
    before = theirs.stat().st_mtime_ns
    settings = S.owner_settings(theirs)
    assert set(settings) == {"model", "conversation"}  # nothing else is copied (no security, voice, Live...)
    config = S.config_from_owner(settings)
    assert config.model.cloud_provider == "claude" and config.model.budget_monthly_limit_usd == 25.0
    assert config.conversation.claude_model == "claude-haiku-5-5"
    assert config.security.root_enabled is False
    assert config_module.CONFIG_FILE.resolve().is_relative_to(S.TEST_ROOT.resolve())  # where its save would go
    assert theirs.stat().st_mtime_ns == before


def _child(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "HELIOX_TEST_ROOT"}  # as a script run by hand would start
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=DAEMON, timeout=120
    )


def test_run_as_a_script_the_driver_makes_its_own_marked_root_before_pilot_reads_a_path():
    done = _child(
        "import benchmarks.conversation_eval.sessions as S, pilot.config as c, pathlib\n"
        "print(S.TEST_ROOT)\nprint(S.outside(S.TEST_ROOT))\nprint((S.TEST_ROOT / c.TEST_ROOT_MARKER).is_file())\n"
        "print(c.CONFIG_FILE)\n"
    )
    assert done.returncode == 0, done.stderr
    root, stray, marked, config_file = done.stdout.split("\n")[:4]
    assert "heliox-sessions-root-" in root and stray == "{}" and marked == "True"
    assert config_file.startswith(root)


def test_run_as_a_script_after_pilot_read_the_real_paths_the_driver_refuses():
    done = _child("import pilot.config\nimport benchmarks.conversation_eval.sessions\n")
    assert done.returncode != 0 and "refusing to run sessions" in done.stderr
