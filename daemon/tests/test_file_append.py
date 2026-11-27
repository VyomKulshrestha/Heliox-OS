"""Appending is one operation: file_write with "append": true, end to end.

Second live run, 27 November: "Append the line X to note.txt" was planned as a file_read then a file_write around
{PREV_OUTPUT}. It worked when the file existed, but the reply always said "1 step could not be checked" (the read was
"changed later" by the write), a trigger whose job was "add a line to fired.log" stored the two steps and its first
fire failed because fired.log did not exist yet, the added line began with a blank line, and two steps cost more
model output than one. Fakes only: files live in a temporary folder, nothing is sent, no model is called.
"""

from __future__ import annotations

import codecs
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    FileParams,
    TriggerParams,
)
from pilot.agents.postconditions import PRECHECK
from pilot.agents.postconditions.files import file_has_content
from pilot.config import PilotConfig
from pilot.system import triggers
from pilot.system.filesystem import appended_bytes, file_append

NL = os.linesep.encode()


def _append_action(path: Path, content: str = "Also renew the parking permit.") -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE,
        target=str(path),
        parameters=FileParams(path=str(path), content=content, append=True),
    )


# ── The bytes an append adds ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("existing", "text", "added"),
    [
        (b"", "first", b"first" + NL),  # a new or empty file: no line break before the line
        (b"one\n", "two", b"two\n"),  # ends with a line break: none added before
        (b"one", "two", NL + b"two" + NL),  # no line break anywhere: one added, the platform's
        (b"one\r\ntwo", "three", b"\r\nthree\r\n"),  # Windows line endings are kept
        (b"one\n", "\n\ntwo\n\n", b"two\n"),  # never a blank line before or after the new line
        (b"one\n", "two\nthree", b"two\nthree\n"),  # several lines keep the file's line ending
        (codecs.BOM_UTF8 + "größe\n".encode(), "maß", "maß\n".encode()),  # UTF-8 with its mark
        (codecs.BOM_UTF16_LE + "one\r\n".encode("utf-16-le"), "zwei", "zwei\r\n".encode("utf-16-le")),
        (b"caf\xe9\n", "plain ascii", b"plain ascii\n"),  # 8-bit text takes a plain ASCII line as it is
    ],
    ids=["new", "ends-with-break", "no-break", "crlf", "blank-lines", "two-lines", "utf8-bom", "utf16", "latin1"],
)
def test_the_added_bytes_keep_one_line_break_and_the_files_own_style(existing: bytes, text: str, added: bytes) -> None:
    assert appended_bytes(existing, text) == added


def test_text_that_cannot_be_written_in_the_files_encoding_is_refused() -> None:
    with pytest.raises(ValueError, match="not UTF-8 text"):
        appended_bytes(b"caf\xe9\n", "naïve")
    with pytest.raises(ValueError, match="no text to add"):
        appended_bytes(b"one\n", "\n \n")


async def test_an_append_makes_a_missing_file_and_its_folder(tmp_path: Path) -> None:
    target = tmp_path / "new" / "fired.log"
    output = await file_append(str(target), "seen a change")
    assert target.read_bytes() == b"seen a change" + NL
    assert output.startswith(f"Appended 1 line to {target} (appended sha256 ")
    assert output.endswith("; the file was new")
    await file_append(str(target), "seen a change")
    assert target.read_bytes() == (b"seen a change" + NL) * 2


# ── The check ────────────────────────────────────────────────────────


async def _appended(tmp_path: Path, start: bytes = b"Buy milk.\n") -> tuple[Path, ActionResult]:
    note = tmp_path / "note.txt"
    note.write_bytes(start)
    action = _append_action(note)
    return note, ActionResult(action=action, success=True, output=await file_append(str(note), "Also renew it."))


async def test_the_check_reads_the_file_again_and_finds_the_added_line_at_its_end(tmp_path: Path) -> None:
    _note, result = await _appended(tmp_path)
    outcome = await file_has_content(result)
    assert outcome.passed is True and "the rest is as it was" in outcome.detail


async def test_a_file_that_changed_afterwards_is_said_as_it_is(tmp_path: Path) -> None:
    note, result = await _appended(tmp_path)
    note.write_bytes(b"Buy oat milk.\n" + note.read_bytes().split(b"\n", 1)[1])  # the beginning changed
    kept = await file_has_content(result)
    assert kept.passed is True and "it also changed elsewhere since" in kept.detail
    note.write_bytes(note.read_bytes() + b"someone else's line\n")  # the end changed
    lost = await file_has_content(result)
    assert lost.passed is False and "no longer ends with the lines that were added" in lost.detail


async def test_an_append_is_never_skipped_by_the_check_before_it(tmp_path: Path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("Also renew the parking permit.\n", encoding="utf-8")
    before = ActionResult(action=_append_action(note), success=True, executed=False, skip_reason=PRECHECK)
    assert (await file_has_content(before)).passed is False  # it runs, although the file already ends with the line


# ── Undo, the review class, the sentences ────────────────────────────


async def test_undo_puts_back_the_earlier_text_or_removes_the_file_the_append_made(tmp_path: Path) -> None:
    from pilot.config import UndoConfig
    from pilot.workflows.undo_journal import UndoJournal

    journal = UndoJournal(tmp_path / "undo-store", config=UndoConfig())
    note, new = tmp_path / "note.txt", tmp_path / "logs" / "fired.log"
    note.write_bytes(b"Buy milk.")
    for action in (_append_action(note), _append_action(new, "seen a change")):
        token = await journal.before(action, task_key="task-1")
        output = await file_append(action.parameters.path, action.parameters.content)
        await journal.after(token, ActionResult(action=action, success=True, output=output))
    assert note.read_bytes() == b"Buy milk." + NL + b"Also renew the parking permit." + NL

    preview = await journal.preview("task-1")
    assert preview is not None
    await journal.undo("task-1", preview["preview_token"])

    assert note.read_bytes() == b"Buy milk."
    assert not new.exists()


def test_an_append_in_the_persons_folders_runs_beside_its_review(tmp_path: Path) -> None:
    from pilot.agents.review_timing import ReviewTiming, kept_copy_failed_sentence, review_timing

    note = tmp_path / "notes" / "note.txt"
    note.parent.mkdir()
    note.write_text("Buy milk.\n", encoding="utf-8")
    undo = SimpleNamespace(enabled=True, max_task_mb=1)
    plan = ActionPlan(actions=[_append_action(note)])
    assert review_timing(plan, card_anyway=False, undo=undo, home=tmp_path) is ReviewTiming.BESIDE
    off = SimpleNamespace(enabled=False, max_task_mb=1)
    assert review_timing(plan, card_anyway=False, undo=off, home=tmp_path) is ReviewTiming.BEFORE
    assert kept_copy_failed_sentence(_append_action(note)) == (
        "I could not keep a copy of the earlier file, so I did not add the line to it."
    )


def test_a_trigger_that_appends_is_said_as_adding_the_line() -> None:
    from pilot.result_sentences import steps_job

    step = _append_action(Path("C:/work/fired.log"), "seen a change").model_dump(mode="json")
    assert steps_job([step]) == 'add "seen a change" to fired.log'


# ── The planner ──────────────────────────────────────────────────────


def _parsed(actions: list[dict]) -> Any:
    from pilot.agents.planner import Planner

    return Planner.__new__(Planner)._parse_response(
        json.dumps({"explanation": "Add the line.", "actions": actions}), "Append the line to note.txt"
    )


def test_a_read_then_a_write_that_only_adds_at_the_end_is_folded_into_one_append() -> None:
    path = "C:\\work\\notes\\note.txt"
    plan = _parsed(
        [
            {"action_type": "file_read", "target": path, "parameters": {"path": path}},
            {
                "action_type": "file_write",
                "target": path,
                "parameters": {"path": path, "content": "{PREV_OUTPUT}\nAlso renew the permit."},
                "use_previous_output": True,
            },
        ]
    )
    assert plan.error is None
    [step] = plan.actions
    assert step.action_type is ActionType.FILE_WRITE and step.use_previous_output is False
    assert (step.parameters.path, step.parameters.content, step.parameters.append) == (
        path,
        "Also renew the permit.",
        True,
    )


def test_a_read_whose_output_a_later_step_uses_is_kept() -> None:
    path = "C:\\work\\notes\\note.txt"
    plan = _parsed(
        [
            {"action_type": "file_read", "target": path, "parameters": {"path": path}},
            {"action_type": "file_write", "target": path, "parameters": {"path": path, "content": "{PREV_OUTPUT}\nx"}},
            {"action_type": "code_execute", "parameters": {"code": "print(PREV_OUTPUT)"}, "use_previous_output": True},
        ]
    )
    assert [a.action_type for a in plan.actions][:2] == [ActionType.FILE_READ, ActionType.FILE_WRITE]


def test_a_triggers_append_step_is_one_append() -> None:
    from pilot.agents.planner import _trigger_steps

    [step] = _trigger_steps(
        [{"action_type": "file_append", "parameters": {"path": "C:\\w\\fired.log", "content": "x"}}]
    )
    assert step.action_type is ActionType.FILE_WRITE and step.parameters.append is True


def test_the_prompt_teaches_the_one_step_append_in_its_fixed_part() -> None:
    from pilot.agents.planner import SYSTEM_PROMPT

    prompt = SYSTEM_PROMPT.format(os="Windows", path_style="Windows", home="C:\\Users\\me")
    assert '"content": "new line", "append": true}' in prompt
    assert '"content": "seen a change",\n  "append": true}' in prompt  # the trigger example's step
    assert "file_append" not in prompt


# ── A trigger that appends, through the engine ───────────────────────


class _HandlerExecutor:
    """The shared Executor as far as file_write goes: the real handler, nothing else."""

    def __init__(self) -> None:
        from tests.test_failed_actions_are_failures import _executor

        self._real = _executor(ActionType.FILE_WRITE, "_exec_file_write")
        self._real._last_output = self._real._largest_output = ""

    async def execute(self, plan: ActionPlan, **_kwargs: Any) -> list[ActionResult]:
        return [await self._real._execute_single(action, snapshot_id=None) for action in plan.actions]


async def test_a_trigger_that_appends_works_on_its_first_fire_with_a_new_file(tmp_path: Path, monkeypatch) -> None:
    from pilot.server import PilotServer

    engine = triggers.TriggerEngine()
    engine.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", engine)
    fired_log = tmp_path / "work" / "fired.log"
    watched = tmp_path / "watched.txt"
    watched.write_text("x", encoding="utf-8")
    made = engine.create_trigger(
        "watched-txt-log",
        "file_modified",
        {"path": str(watched)},
        "Add the line 'watched.txt changed' to fired.log",
        steps=[_append_action(fired_log, "watched.txt changed")],
        steps_approved=True,
    )
    server = PilotServer(PilotConfig())
    server._running = True
    server._executor = _HandlerExecutor()
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._task_receipts = None

    await server._dispatch_reactive_trigger(made)
    await server._dispatch_reactive_trigger(engine.get_trigger(made.id))

    assert fired_log.read_bytes() == (b"watched.txt changed" + NL) * 2  # no leading blank line, one break each
    told = [c.args[1]["text"] for c in server._broadcast_notification.await_args_list if c.args[0] == "watch_fired"]
    assert len(told) == 2 and all("did not" not in sentence for sentence in told), told


# ── "Delete both of them" ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "count", "picked"),
    [
        ("Delete both of them", 2, [0, 1]),
        ("both", 2, [0, 1]),
        ("both", 3, None),  # "both" of three is not a pick
        ("delete all of them", 3, [0, 1, 2]),
        ("the first one", 2, [0]),
        ("Just the second.", 2, [1]),
        ("the last one", 3, [2]),
        ("neither", 2, []),
        ("keep both", 2, []),
        ("none of them", 3, []),
        ("delete all my notes", 2, None),  # about something else
        ("delete note.txt", 2, None),  # names something
        ("the fourth one", 2, None),  # not offered
        ("yes", 2, None),
    ],
)
def test_a_reply_that_picks_from_the_list_heliox_gave(reply: str, count: int, picked: list[int] | None) -> None:
    from pilot.agents.request_match import picked_from

    assert picked_from(reply, count) == picked


@pytest.fixture
def two_triggers(tmp_path: Path, monkeypatch) -> tuple[triggers.TriggerEngine, list[triggers.Trigger]]:
    engine = triggers.TriggerEngine()
    engine.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", engine)
    path = str(tmp_path / "watched.txt")
    made = [
        engine.create_trigger("watched-txt-log", "file_modified", {"path": path}, "add a line to fired.log"),
        engine.create_trigger("watched-txt-change", "file_modified", {"path": path}, "tell me"),
    ]
    return engine, made


async def test_both_of_them_after_the_which_question_deletes_exactly_those_two(two_triggers) -> None:
    engine, made = two_triggers
    with pytest.raises(triggers.TriggerNotFound, match="2 triggers match"):
        await triggers.trigger_delete("the trigger that watches watched.txt")

    expanded = triggers.pick_request("Delete both of them")
    assert expanded is not None
    request, note = expanded
    assert request == f"Delete the triggers with the ids {made[0].id} and {made[1].id}."
    assert note.startswith('Deleting "watched-txt-log"')

    from pilot.agents.planner import Planner

    plan = await Planner.__new__(Planner).plan(request)  # made in code: no model is asked
    assert [(a.action_type, a.parameters.trigger_id) for a in plan.actions] == [
        (ActionType.TRIGGER_DELETE, made[0].id),
        (ActionType.TRIGGER_DELETE, made[1].id),
    ]
    for action in plan.actions:
        await triggers.trigger_delete(action.parameters.trigger_id)
    assert engine.list_triggers() == []


async def test_the_list_after_a_name_that_matched_nothing_is_an_offer_too(two_triggers) -> None:
    """The run's order: which one? then a name that does not exist, then "delete both of them"."""
    _engine, made = two_triggers
    with pytest.raises(triggers.TriggerNotFound, match="No trigger matches"):
        await triggers.trigger_delete("nonexistent-trigger-42")
    assert triggers.picked_triggers("Delete both of them") == made  # never the name that does not exist


async def test_the_first_one_and_neither(two_triggers) -> None:
    _engine, made = two_triggers
    with pytest.raises(triggers.TriggerNotFound):
        await triggers.trigger_delete("watched.txt")
    assert triggers.pick_request("the second one") == (
        f"Delete the trigger with the id {made[1].id}.",
        'Deleting "watched-txt-change" (file modified: watched.txt)',
    )
    assert triggers.pick_request("neither") is None
    assert triggers.kept_sentence("neither") == "Okay, I've kept both triggers; nothing was deleted."
    assert triggers.picked_triggers("both") is None  # the offer is used up


async def test_a_question_the_model_asked_is_remembered_as_the_offer(two_triggers) -> None:
    from pilot.agents.planner import _note_trigger_question

    _engine, made = two_triggers
    _note_trigger_question("Delete the trigger that watches watched.txt")
    assert triggers.picked_triggers("delete both of them") == made


async def test_an_old_offer_is_not_a_pick(two_triggers, monkeypatch) -> None:
    engine, _made = two_triggers
    clock = [1000.0]
    monkeypatch.setattr(triggers.time, "monotonic", lambda: clock[0])
    engine.offer([t.id for t in _made])
    clock[0] += triggers.OFFER_SECONDS + 1
    assert triggers.picked_triggers("delete both of them") is None


async def test_the_server_expands_the_pick_before_the_request_is_routed(two_triggers) -> None:
    from pilot.server import PilotServer

    _engine, made = two_triggers
    with pytest.raises(triggers.TriggerNotFound):
        await triggers.trigger_delete("watched.txt")
    server = PilotServer(PilotConfig())
    assert (await server._expand_request("Delete both of them"))[0].startswith("Delete the triggers with the ids")
    assert await server._trigger_pick_request("neither") == "Okay, I've kept both triggers; nothing was deleted."


# ── A trigger's steps that would need approval when asked for directly get the card ──


@pytest.mark.parametrize(
    "step",
    [
        Action(
            action_type=ActionType.FILE_DELETE, target="C:\\w\\old.log", parameters=FileParams(path="C:\\w\\old.log")
        ),
        Action(
            action_type=ActionType.API_SEND_EMAIL,
            target="pat@example.com",
            parameters=ApiRequestParams(to="pat@example.com", subject="Changed", message="watched.txt changed"),
        ),
    ],
    ids=["delete", "send"],
)
def test_a_trigger_whose_step_needs_approval_gets_the_card_with_the_fires_line(step: Action) -> None:
    from pilot.server import _approval_effect

    assert step.requires_confirmation or step.is_irreversible  # asked for directly, it would need approval
    create = Action(
        action_type=ActionType.TRIGGER_CREATE,
        target="watch",
        parameters=TriggerParams(name="watch", trigger_type="file_modified", steps=[step]),
    )
    assert create.requires_confirmation is True
    assert _approval_effect(create).startswith("Each time this trigger fires, I'll ")
    plain = create.model_copy(update={"parameters": TriggerParams(name="watch", steps=[_append_action(Path("a.log"))])})
    assert plain.requires_confirmation is False  # an append alone needs no card, as the second run showed
