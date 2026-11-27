"""A trigger whose job is to tell the person is a notice, not a job.

Third live run, 27 November: "Tell me when watched.txt changes" was stored with no steps and the words "Tell the
person that <full path> has changed"; on fire it started an autonomous job that planned a desktop notify whose body
was the full path. The person asked to be told: the engine's own notice (shown and spoken, as a watcher's is) is the
telling. Fakes only: the files are in a temporary folder, no job starts, no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from pilot.actions import Action, ActionType, EmptyParams, FileParams
from pilot.agents.planner import SYSTEM_PROMPT, Planner
from pilot.config import PilotConfig
from pilot.system import triggers


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> triggers.TriggerEngine:
    fresh = triggers.TriggerEngine()
    fresh.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", fresh)
    return fresh


@pytest.mark.parametrize(
    ("words", "only"),
    [
        ("Tell the person that C:\\work\\watched.txt has changed", True),
        ("Let me know", True),
        ("alert me", True),
        ("Notify the user", True),
        ("Tell me and copy it to the backup folder", False),
        ("Append the line 'seen' to fired.log", False),
        ("", False),
    ],
)
def test_words_that_only_ask_to_be_told(words: str, only: bool) -> None:
    assert triggers.tells_only(words) is only


async def test_a_tell_me_trigger_is_stored_as_a_notice(engine, tmp_path: Path) -> None:
    watched = str(tmp_path / "watched.txt")
    answer = json.loads(
        await triggers.trigger_create(
            "watched-txt-change", "file_modified", {"path": watched}, f"Tell the person that {watched} has changed"
        )
    )
    assert (answer["notify_only"], answer["action_command"], answer["steps"]) == (True, "", [])
    await engine.stop()
    again = triggers.TriggerEngine()
    again.attach_store(tmp_path / "triggers.json")
    [loaded] = again._triggers.values()
    assert loaded.enabled and loaded.notify_only and loaded.disabled_reason == ""


def _server():
    from pilot.server import PilotServer

    server = PilotServer(PilotConfig())
    server._running = True
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._autonomous = SimpleNamespace(submit=AsyncMock(side_effect=AssertionError("no job for a notice")))
    return server


def _told(server) -> list[str]:
    return [c.args[1]["text"] for c in server._broadcast_notification.await_args_list if c.args[0] == "watch_fired"]


async def test_a_notice_fire_says_the_files_name_and_starts_no_job(engine, tmp_path: Path) -> None:
    made = engine.create_trigger("w", "file_modified", {"path": str(tmp_path / "watched.txt")}, "", notify_only=True)
    server = _server()
    await server._dispatch_reactive_trigger(made)
    assert _told(server) == ["watched.txt changed."]
    server._speak_companion_text.assert_awaited_once()
    assert str(tmp_path) not in _told(server)[0]  # the name, never the full path


async def test_a_trigger_stored_before_with_tell_words_is_a_notice_too(engine, tmp_path: Path) -> None:
    watched = str(tmp_path / "watched.txt")
    made = engine.create_trigger("w", "file_modified", {"path": watched}, f"Tell the person that {watched} has changed")
    made.notify_only = False  # as the third run stored it
    server = _server()
    await server._dispatch_reactive_trigger(made)
    assert _told(server) == ["watched.txt changed."]


async def test_two_triggers_on_one_file_each_say_which_fired(engine, tmp_path: Path) -> None:
    watched, log = str(tmp_path / "watched.txt"), str(tmp_path / "fired.log")
    notice = engine.create_trigger("change", "file_modified", {"path": watched}, "", notify_only=True)
    step = Action(
        action_type=ActionType.FILE_WRITE, target=log, parameters=FileParams(path=log, content="seen", append=True)
    )
    adder = engine.create_trigger(
        "log", "file_modified", {"path": watched}, "add seen", steps=[step], steps_approved=True
    )
    from pilot.result_sentences import trigger_fire_sentence

    assert trigger_fire_sentence(notice, "notice") == "watched.txt changed (the trigger set to tell you)."
    assert trigger_fire_sentence(adder, "ran").startswith(
        'The trigger that watches watched.txt and is set to add "seen" to fired.log fired'
    )


def test_the_planner_makes_tell_me_a_notice_whatever_the_model_wrote() -> None:
    watched = "C:\\work\\watched.txt"
    words = Planner._normalize_params(
        ActionType.TRIGGER_CREATE,
        {"name": "w", "trigger_type": "file_modified", "condition": {"path": watched},
         "action_command": f"Tell the person that {watched} has changed"},
        "w",
    )  # fmt: skip
    assert (words["notify_only"], words["action_command"], words["steps"]) == (True, "", [])
    notify = Planner._normalize_params(
        ActionType.TRIGGER_CREATE,
        {"name": "w", "trigger_type": "file_modified", "condition": {"path": watched}, "action_command": "notify",
         "steps": [{"action_type": "notify", "parameters": {"title": "Changed", "message": watched}}]},
        "w",
    )  # fmt: skip
    assert notify["notify_only"] is True and notify["steps"] == []
    job = Planner._normalize_params(
        ActionType.TRIGGER_CREATE,
        {"name": "w", "trigger_type": "file_modified", "condition": {"path": watched},
         "action_command": "Append seen to fired.log"},
        "w",
    )  # fmt: skip
    assert not job.get("notify_only")


def test_the_prompt_shows_the_notice_form_in_its_fixed_part() -> None:
    prompt = SYSTEM_PROMPT.format(os="Windows", path_style="Windows", home="C:\\Users\\me")
    assert '"condition": {"threshold": 90},\n  "notify_only": true}' in prompt
    assert '"tell me when", "let me know' in prompt
