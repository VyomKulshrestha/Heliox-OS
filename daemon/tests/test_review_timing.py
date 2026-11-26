"""The companion's review holds up only a plan that could hurt and would run without a card.

Fake planners, executors, verifiers and models; the companion's clock is moved only by its own polls. Nothing is
written, run or sent.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pilot.actions import (
    READ_ONLY_ACTIONS,
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    ClipboardParams,
    FileParams,
    KeyboardParams,
    MouseParams,
    PowerParams,
    VerificationResult,
    VolumeParams,
    WindowParams,
)
from pilot.agents.execution_companion import ExecutionCompanion
from pilot.agents.review_timing import (
    BEFORE_LIMIT_SECONDS,
    KEPT_COPY_FAILED_SKIP,
    UNANSWERED_REASON,
    ReviewTiming,
    kept_copy_required,
    require_kept_copies,
    review_timing,
)
from pilot.agents.step_stakes import _EXPLICIT, STAKES, Stake
from tests.test_execute_tracked import _FakePermissionChecker, _FakeWs, _server_ready_for_handle_execute

NOTE = os.path.abspath(os.path.join("work", "timing", "note_timing.txt"))


def _write() -> Action:
    return Action(
        action_type=ActionType.FILE_WRITE, target=NOTE, parameters=FileParams(path=NOTE, content="timing check.")
    )


def _delete() -> Action:
    return Action(action_type=ActionType.FILE_DELETE, target=NOTE, parameters=FileParams(path=NOTE))


def _read() -> Action:
    return Action(action_type=ActionType.FILE_READ, target=NOTE, parameters=FileParams(path=NOTE))


def _type() -> Action:
    return Action(action_type=ActionType.KEYBOARD_TYPE, target="Notepad", parameters=KeyboardParams(text="hello"))


def _plan(*actions: Action) -> ActionPlan:
    return ActionPlan(actions=list(actions), explanation="plan")


# ── the rule, from what the plan carries ──

UNDO = SimpleNamespace(enabled=True, max_task_mb=1)


@pytest.fixture(autouse=True)
def _home_is_the_working_folder(monkeypatch):
    """The server-level tests write (in fakes) under the working folder; it stands in for the home folder."""
    monkeypatch.setattr("pilot.agents.review_timing._home", lambda: Path(os.getcwd()))


def _file_action(kind: ActionType, path: Path, destination: Path | None = None) -> Action:
    params = FileParams(path=str(path), destination=str(destination) if destination else None, content="x")
    return Action(action_type=kind, target=str(path), parameters=params)


def _timing(*actions: Action, home: Path, undo=UNDO, **options) -> ReviewTiming:
    return review_timing(_plan(*actions), card_anyway=False, undo=undo, home=home, **options)


def test_reads_and_new_files_in_the_persons_folders_run_beside_their_review(tmp_path):
    source = tmp_path / "notes" / "a.txt"
    source.parent.mkdir()
    source.write_text("a")
    read = _file_action(ActionType.FILE_READ, source)
    write_new = _file_action(ActionType.FILE_WRITE, tmp_path / "notes" / "new.txt")
    copy_new = _file_action(ActionType.FILE_COPY, source, tmp_path / "notes" / "copy.txt")
    move_into_folder = _file_action(ActionType.FILE_MOVE, source, tmp_path / "notes" / "moved.txt")
    assert _timing(read, write_new, copy_new, move_into_folder, home=tmp_path) is ReviewTiming.BESIDE


def test_an_overwrite_runs_beside_only_while_the_journal_can_keep_the_file_it_replaces(tmp_path):
    small = tmp_path / "small.txt"
    small.write_bytes(b"x" * 1000)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (1024 * 1024 + 1))  # past the 1 MB undo limit of this task
    assert _timing(_file_action(ActionType.FILE_WRITE, small), home=tmp_path) is ReviewTiming.BESIDE
    assert _timing(_file_action(ActionType.FILE_WRITE, big), home=tmp_path) is ReviewTiming.BEFORE
    source = tmp_path / "source.txt"
    source.write_text("s")
    assert _timing(_file_action(ActionType.FILE_COPY, source, big), home=tmp_path) is ReviewTiming.BEFORE
    # Two overwrites that each fit but not together: the limit is for the whole task.
    half = tmp_path / "half.bin"
    half.write_bytes(b"x" * (600 * 1024))
    other = tmp_path / "other.bin"
    other.write_bytes(b"x" * (600 * 1024))
    writes = (_file_action(ActionType.FILE_WRITE, half), _file_action(ActionType.FILE_WRITE, other))
    assert _timing(*writes, home=tmp_path) is ReviewTiming.BEFORE


def test_undo_off_keeps_nothing_so_every_change_waits(tmp_path):
    off = SimpleNamespace(enabled=False, max_task_mb=512)
    write = _file_action(ActionType.FILE_WRITE, tmp_path / "new.txt")
    assert _timing(write, home=tmp_path, undo=off) is ReviewTiming.BEFORE


def test_a_folder_in_the_way_or_a_folder_copied_waits(tmp_path):
    inner = tmp_path / "folder" / "inner"
    inner.mkdir(parents=True)
    source = tmp_path / "a.txt"
    source.write_text("a")
    copy_into = _file_action(ActionType.FILE_COPY, source, inner)
    assert _timing(copy_into, home=tmp_path) is ReviewTiming.BESIDE
    # A folder where the copy would land: the journal refuses it ("the destination folder already exists").
    (inner / "a.txt").mkdir()
    assert _timing(copy_into, home=tmp_path) is ReviewTiming.BEFORE
    assert _timing(_file_action(ActionType.FILE_WRITE, inner), home=tmp_path) is ReviewTiming.BEFORE
    # A whole folder copied (many files): not this class.
    folder_copy = _file_action(ActionType.FILE_COPY, tmp_path / "folder", tmp_path / "copy")
    assert _timing(folder_copy, home=tmp_path) is ReviewTiming.BEFORE


def test_a_move_across_drives_waits(tmp_path):
    source = tmp_path / "a.txt"
    source.write_text("a")
    move = _file_action(ActionType.FILE_MOVE, source, tmp_path / "b.txt")
    same = _timing(move, home=tmp_path, device_of=lambda path: 1)
    across = _timing(move, home=tmp_path, device_of=lambda path: 1 if path == source else 2)
    assert (same, across) == (ReviewTiming.BESIDE, ReviewTiming.BEFORE)


def test_outside_the_persons_own_folders_waits_even_where_the_journal_could_restore_it(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    inside = home / "notes.txt"
    inside.write_text("n")
    # A program folder or another user's folder: not under the home folder.
    outside = tmp_path / "Program Files" / "App" / "settings.ini"
    assert _timing(_file_action(ActionType.FILE_WRITE, outside), home=home) is ReviewTiming.BEFORE
    share = Path("//server/share/notes.txt")  # a network share
    assert _timing(_file_action(ActionType.FILE_WRITE, share), home=home) is ReviewTiming.BEFORE
    removable = _file_action(ActionType.FILE_COPY, inside, tmp_path / "usb" / "notes.txt")
    assert _timing(removable, home=home) is ReviewTiming.BEFORE
    assert _timing(_file_action(ActionType.FILE_WRITE, inside), home=home) is ReviewTiming.BESIDE
    # Heliox's own folders inside the home folder: the journal refuses to keep a copy there.
    own = _timing(_file_action(ActionType.FILE_WRITE, inside), home=home, private=lambda path: True)
    assert own is ReviewTiming.BEFORE


def test_a_clipboard_write_waits_because_its_earlier_content_can_be_lost(tmp_path):
    clipboard = Action(
        action_type=ActionType.CLIPBOARD_WRITE, target="clipboard", parameters=ClipboardParams(content="x")
    )
    assert _timing(clipboard, home=tmp_path) is ReviewTiming.BEFORE


def test_steps_with_nothing_at_stake_run_beside_their_review_whatever_the_journal_keeps(tmp_path):
    volume = Action(action_type=ActionType.VOLUME_SET, target="volume", parameters=VolumeParams(level=30))
    focus = Action(action_type=ActionType.WINDOW_FOCUS, target="Notepad", parameters=WindowParams(title="Notepad"))
    page = Action(
        action_type=ActionType.BROWSER_NAVIGATE,
        target="https://example.com",
        parameters=BrowserParams(url="https://example.com"),
    )
    off = SimpleNamespace(enabled=False, max_task_mb=0)  # undo off: these do not depend on it
    assert _timing(volume, focus, page, home=tmp_path, undo=off) is ReviewTiming.BESIDE


@pytest.mark.parametrize(
    "action",
    [
        Action(action_type=ActionType.WINDOW_CLOSE, target="Notepad", parameters=WindowParams(title="Notepad")),
        Action(action_type=ActionType.BROWSER_REFRESH, target="page", parameters=BrowserParams()),
        Action(action_type=ActionType.MOUSE_SCROLL, target="screen", parameters=MouseParams()),
        Action(action_type=ActionType.POWER_LOCK, target="system", parameters=PowerParams()),
    ],
)
def test_closing_reloading_a_wheel_over_a_program_or_locking_waits(tmp_path, action):
    assert _timing(action, home=tmp_path) is ReviewTiming.BEFORE


def test_every_action_type_is_in_exactly_one_class_with_a_reason():
    # A new action type without an entry fails here: decide what is at stake in it.
    assert set(STAKES) == set(ActionType)
    explicit_reads = {action for action, (stake, _reason) in _EXPLICIT.items() if stake is Stake.READ}
    assert not (set(_EXPLICIT) - explicit_reads) & READ_ONLY_ACTIONS  # a read is never also another class
    assert all(reason.strip() for _stake, reason in STAKES.values())
    journal_kept = {action for action, (stake, _reason) in STAKES.items() if stake is Stake.JOURNAL_KEEPS}
    assert journal_kept == {ActionType.FILE_WRITE, ActionType.FILE_COPY, ActionType.FILE_MOVE}


def test_only_journal_kept_steps_are_marked_for_the_executor(tmp_path):
    write = _file_action(ActionType.FILE_WRITE, tmp_path / "a.txt")
    volume = Action(action_type=ActionType.VOLUME_SET, target="volume", parameters=VolumeParams(level=30))
    require_kept_copies([write, volume])
    try:
        # The content does not matter: the executor may put an earlier step's output into it.
        assert kept_copy_required(
            write.model_copy(update={"parameters": write.parameters.model_copy(update={"content": "y"})})
        )
        assert not kept_copy_required(volume)
    finally:
        require_kept_copies(())
    assert not kept_copy_required(write)


async def _write_with_a_failing_copy(tmp_path, *, required: bool):
    from pilot.agents.executor import Executor
    from pilot.config import PilotConfig
    from pilot.security.audit import AuditLogger
    from pilot.security.gateway import InvocationSource
    from pilot.security.permissions import PermissionChecker
    from pilot.security.validator import ActionValidator
    from pilot.workflows.undo_journal import UndoJournal

    config = PilotConfig()
    executor = Executor(
        config, ActionValidator(config), PermissionChecker(config), AuditLogger(audit_file=tmp_path / "audit.jsonl")
    )
    journal = UndoJournal(tmp_path / "undo-store", config=config.undo)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    journal._keep_copy = full_disk  # the journal cannot keep its copy of the earlier file
    executor.set_undo_journal(journal)
    path = tmp_path / "note.txt"
    path.write_text("earlier", encoding="utf-8")
    write = Action(
        action_type=ActionType.FILE_WRITE, target=str(path), parameters=FileParams(path=str(path), content="new")
    )
    plan = ActionPlan(actions=[write], raw_input="write the note")
    require_kept_copies([write] if required else [])
    try:
        results = await executor.execute(
            plan, invocation_source=InvocationSource.INTERACTIVE, user_confirmed=True, plan_id="plan-k"
        )
    finally:
        require_kept_copies(())
    return results[0], path.read_text(encoding="utf-8"), journal


@pytest.mark.asyncio
async def test_a_step_run_beside_its_review_does_not_overwrite_when_the_journal_cannot_keep_the_earlier_file(
    tmp_path,
):
    result, content, _journal = await _write_with_a_failing_copy(tmp_path, required=True)

    assert content == "earlier"  # nothing was changed
    assert result.success is False and result.executed is False and result.skip_reason == KEPT_COPY_FAILED_SKIP
    assert result.error == "Heliox could not keep a copy of the earlier file, so it did not overwrite it."


@pytest.mark.asyncio
async def test_every_other_step_keeps_the_journals_rule_and_runs_without_undo(tmp_path):
    result, content, journal = await _write_with_a_failing_copy(tmp_path, required=False)

    assert result.success is True and content == "new"
    preview = await journal.preview("plan-k")
    assert preview is not None and preview["not_undoable"]


def test_a_delete_or_typing_into_another_app_waits(tmp_path):
    assert _timing(_file_action(ActionType.FILE_DELETE, tmp_path / "a.txt"), home=tmp_path) is ReviewTiming.BEFORE
    assert _timing(_type(), home=tmp_path) is ReviewTiming.BEFORE


@pytest.mark.parametrize(
    ("kind", "keys", "beside"),
    [
        (ActionType.KEYBOARD_PRESS, {"key": "playpause"}, True),
        (ActionType.KEYBOARD_PRESS, {"key": "volumeup"}, True),
        (ActionType.KEYBOARD_PRESS, {"key": "stop"}, True),
        (ActionType.KEYBOARD_HOTKEY, {"keys": ["volumedown", "volumemute"]}, True),
        (ActionType.KEYBOARD_PRESS, {"key": "enter"}, False),
        (ActionType.KEYBOARD_HOTKEY, {"keys": ["ctrl", "volumeup"]}, False),
        (ActionType.KEYBOARD_HOTKEY, {"keys": ["ctrl", "s"]}, False),
    ],
)
def test_a_key_press_of_only_media_or_volume_keys_runs_beside_its_review(tmp_path, kind, keys, beside):
    """Play or pause, next, previous, stop and the volume keys go to the media player, not into the program in
    front: nothing at stake. Every other key press waits for the review, as typing does."""
    press = Action(action_type=kind, target="media", parameters=KeyboardParams(**keys))
    expected = ReviewTiming.BESIDE if beside else ReviewTiming.BEFORE
    assert _timing(press, home=tmp_path) is expected


def test_a_plan_with_a_card_is_reviewed_while_the_card_is_open(tmp_path):
    plan = _plan(_file_action(ActionType.FILE_DELETE, tmp_path / "a.txt"))
    assert review_timing(plan, card_anyway=True, undo=UNDO, home=tmp_path) is ReviewTiming.WITH_CARD


# ── fakes ──


class _GatedModel:
    """Answers only when released: the order of events is decided by the test, not by a clock."""

    def __init__(self, reply: str = '{"decision": "CONTINUE", "reason": "Matches the request."}') -> None:
        self.reply = reply
        self.calls = 0
        self.answered = 0
        self.release = asyncio.Event()

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        await self.release.wait()
        self.answered += 1
        return self.reply


class _NeverAnswers:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt, **kwargs):
        self.calls += 1
        await asyncio.Event().wait()


class _PolledClock:
    """The companion's limit clock, moved on only by its own polls."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.waited = 0.0

    def __call__(self) -> float:
        return self.now

    async def wait(self, tasks, timeout=None):
        self.now += timeout
        self.waited += timeout
        await asyncio.sleep(0)
        done = {task for task in tasks if task.done()}
        return done, set(tasks) - done


class _Executor:
    def __init__(self) -> None:
        self.ran = 0

    async def execute(self, plan, **kwargs):
        self.ran += 1
        return [ActionResult(action=action, success=True, output="ok") for action in plan.actions]


class _Verifier:
    async def verify(self, plan, results):
        return VerificationResult(passed=True, independently_observed_actions=list(range(len(results))))


def _server(plan: ActionPlan, companion: ExecutionCompanion, executor: _Executor, *, card: bool = False):
    server = _server_ready_for_handle_execute(executor, plan)
    server._verifier = _Verifier()
    server._execution_companion = companion
    server._permission_checker = _FakePermissionChecker(requires_confirmation=card)
    return server


def _sent(ws: _FakeWs, method: str) -> list[dict]:
    return [m["params"] for m in map(json.loads, ws.sent) if m.get("method") == method]


# ── a note: runs at once, reviewed beside ──


@pytest.mark.asyncio
async def test_a_note_runs_and_replies_before_its_review_answers_and_a_concern_comes_after_as_a_remark():
    model = _GatedModel(reply='{"decision": "WARN", "reason": "The note went to a folder you did not name."}')
    executor = _Executor()
    server = _server(_plan(_write()), ExecutionCompanion(model), executor)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "write a note"}, ws), 10)

    assert result["status"] == "success" and executor.ran == 1
    assert model.calls == 1 and model.answered == 0  # the reply went out while the review was still waiting
    assert _sent(ws, "companion_plan_review") == []

    model.release.set()
    await asyncio.gather(*list(server._companion_follow_up_tasks), return_exceptions=True)

    review = _sent(ws, "companion_plan_review")
    assert len(review) == 1 and review[0]["beside_run"] is True and review[0]["decision"] == "WARN"
    remark = [params for method, params in server.test_broadcasts if method == "companion_plan_intervention"]
    assert len(remark) == 1
    # Not "WARN", "REVISE" or "STOP": the window shows it as a remark and nothing speaks a stop for a finished task.
    assert remark[0]["decision"] == "REMARK" and remark[0]["original_decision"] == "WARN"


# ── a risky plan without a card: waits, short limit, a card when unanswered ──


@pytest.mark.asyncio
async def test_a_risky_plan_whose_review_does_not_answer_asks_the_person_first(monkeypatch):
    clock = _PolledClock()
    model = _NeverAnswers()
    companion = ExecutionCompanion(model, wait=clock.wait, limit_clock=clock)
    executor = _Executor()
    server = _server(_plan(_type()), companion, executor)
    asked: list[dict] = []

    async def _confirm(plan_id, plan, ws, **kwargs):
        asked.append(kwargs)
        return False, set(), set(range(len(plan.actions)))

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "type hello in notepad"}, ws), 10)

    assert clock.waited == pytest.approx(BEFORE_LIMIT_SECONDS)  # its short limit, counted by its own polls
    assert len(asked) == 1 and UNANSWERED_REASON in asked[0]["reason"]
    assert asked[0]["force_all_actions"] is True  # no step needed approval by itself: all of them are asked
    assert executor.ran == 0 and result["status"] == "cancelled"


@pytest.mark.asyncio
async def test_a_risky_plan_whose_review_answers_runs_without_a_card(monkeypatch):
    model = _GatedModel()
    model.release.set()
    executor = _Executor()
    server = _server(_plan(_type()), ExecutionCompanion(model), executor)

    async def _confirm(*args, **kwargs):
        raise AssertionError("no card for a reviewed plan that needs none")

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "type hello in notepad"}, ws), 10)

    assert result["status"] == "success" and executor.ran == 1
    assert _sent(ws, "companion_plan_review")[0]["status"] == "reviewed"


# ── a plan with a card: reviewed while the card is open ──


@pytest.mark.asyncio
async def test_the_card_opens_before_the_review_answers_and_a_stop_still_holds_after_approval(monkeypatch):
    model = _GatedModel(reply='{"decision": "STOP", "reason": "This deletes a file you asked to keep."}')
    executor = _Executor()
    server = _server(_plan(_delete()), ExecutionCompanion(model), executor, card=True)
    seen_on_card: list[int] = []

    async def _confirm(plan_id, plan, ws, **kwargs):
        seen_on_card.append(model.answered)
        server._pending_confirms[plan_id] = object()  # the card is open while the review answers
        model.release.set()
        for _ in range(500):  # counted turns of the loop until the concern is shown beside the card
            if any(method == "companion_plan_intervention" for method, _params in server.test_broadcasts):
                break
            await asyncio.sleep(0)
        server._pending_confirms.pop(plan_id, None)
        return True, {0}, {0}

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)
    ws = _FakeWs()

    result = await asyncio.wait_for(server._handle_execute({"input": "delete the note"}, ws), 10)

    assert seen_on_card == [0]  # the card did not wait for the review
    concern = [params for method, params in server.test_broadcasts if method == "companion_plan_intervention"]
    assert concern and concern[0]["while_approval_open"] is True
    assert result["status"] == "blocked_by_companion" and executor.ran == 0


@pytest.mark.asyncio
async def test_a_denied_card_cancels_its_review(monkeypatch):
    model = _GatedModel()
    executor = _Executor()
    server = _server(_plan(_delete()), ExecutionCompanion(model), executor, card=True)

    async def _confirm(plan_id, plan, ws, **kwargs):
        await asyncio.sleep(0)
        return False, set(), {0}

    monkeypatch.setattr(server, "_wait_for_confirmation", _confirm)

    result = await asyncio.wait_for(server._handle_execute({"input": "delete the note"}, _FakeWs()), 10)

    assert result["status"] == "cancelled" and executor.ran == 0
    assert model.answered == 0 and all(task.done() for task in server._companion_follow_up_tasks)


# ── the limit's message ──


@pytest.mark.asyncio
async def test_a_review_that_only_rode_along_says_so_instead_of_naming_the_configured_limit(caplog):
    clock = _PolledClock()
    companion = ExecutionCompanion(_NeverAnswers(), timeout_seconds=5.0, wait=clock.wait, limit_clock=clock)
    first = await companion.review("type hello", _plan(_type()), others_running=lambda: False)
    assert first.status == "unavailable"

    others_until = clock.now + 0.1
    with caplog.at_level("WARNING", logger="pilot.agents.execution_companion"):
        second = await companion.review("type hello", _plan(_type()), others_running=lambda: clock() < others_until)

    assert second.status == "unavailable"
    message = caplog.records[-1].getMessage()
    assert "had not answered when the other plan reviews finished, after 0.1 s" in message
    assert "limit 5 s" not in message
