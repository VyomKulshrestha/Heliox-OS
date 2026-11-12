"""Whole sessions, end to end: a person's exchange driven through every piece the way the listener drives it.

  python -m benchmarks.conversation_eval.sessions [--only NAME] [--kind claude|openai|local|none] [--out FILE.json]

Real: the server's voice dispatch, the listener's addressing rule (pilot.system.addressing, on an injected clock), the
conversation layer, the quick road, the planner road with its approvals and permission check, Heliox Live's speech
checks and replies, the person's profile, to-dos and reminders (in a fresh temporary folder). Replaced: the
microphone and recognisers (each turn arrives as words with a start time and a length), the speaker (what would be
said is recorded, and the clock moves on by its length), the fast model (each provider's own wire format over a mock
transport, replying with the text the session gives for that turn), the planner (plans the session gives) and the
executor (stub tools that succeed and say what they did). No audio, no network, no real model.

The pack is sessions.json beside this file: each session is a list of turns; each turn says what the person says,
when, and what must and must not happen (see the pack's "how_to_read"). Every session runs for four kinds of person:
a Claude key, an OpenAI-compatible endpoint, a local model only, and no model at all.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
import tempfile
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pilot.conversation.layer import SPOKEN_WORD_CAP

PACK = Path(__file__).with_name("sessions.json")
KINDS = ("claude", "openai", "local", "none")
NOW = datetime(2026, 11, 12, 9, 0)  # a Thursday
WORDS_PER_SECOND = 2.5
DEFAULT_GAP = 1.0
REPLY_WORD_LIMIT = SPOKEN_WORD_CAP  # one rule, the layer's own (pilot.conversation.layer)
WAIT_SIGNS = {"One moment.", "Let me think.", "Give me a second.", "Loading the local model, one moment."}
DEFAULT_OUTPUTS = {
    "battery_info": "=== Battery ===\n  Charge: 80%\n  Status: Charging",
    "volume_set": "Volume set to 60%",
    "keyboard_press": "Pressed volumeup",
    "volume_mute": "Muted",
    "open_application": "Opened",
    "disk_usage": "=== Disk ===\n  C:\\ (C:\\) - 512.0 GB total, 392.0 GB used (76.6%), fstype=NTFS",
    "file_delete": "Deleted",
    "file_move": "Moved",
    "file_list": "3 files",
}


# ── what happened ──


@dataclass
class Said:
    seq: int
    turn: int
    text: str
    channel: str
    over_the_person: bool


@dataclass
class Shown:
    seq: int
    turn: int
    method: str
    params: dict[str, Any]


@dataclass
class Ran:
    turn: int
    action_type: str
    target: str


@dataclass
class Record:
    said: list[Said] = field(default_factory=list)
    shown: list[Shown] = field(default_factory=list)
    ran: list[Ran] = field(default_factory=list)
    taken: dict[int, str] = field(default_factory=dict)  # turn -> why it was taken
    not_taken: dict[int, str] = field(default_factory=dict)
    model_calls: dict[int, int] = field(default_factory=dict)
    withdrawals: dict[int, str] = field(default_factory=dict)  # a continuation turn -> what the server said to it
    ended: dict[int, float] = field(default_factory=dict)  # when the person stopped talking, per taken turn
    tasks: dict[int, set[Any]] = field(default_factory=dict)  # every asyncio task a turn spawned, by that turn
    stuck: list[str] = field(default_factory=list)  # every bounded wait that ran out, and on what
    seq: int = 0
    turn: int = -1

    def next(self) -> int:
        self.seq += 1
        return self.seq


# The turn a piece of work belongs to. Set in the turn's dispatch task; asyncio copies it into every task that one
# spawns (speech said later, Live's end-of-turn delivery, an acknowledgement), so what is said or shown is the turn's
# by origin, not by when it happens to run. -1: outside any turn.
SESSION_TURN: ContextVar[int] = ContextVar("session_turn", default=-1)
END_SECONDS = 5.0  # real time a cancelled dispatch may take to end when the session is over
SETTLE_SECONDS = 8.0  # real time a turn's own tasks may take to finish (a delayed acknowledgement waits 2.5 s)


def _turn_of(record: Record) -> int:
    turn = SESSION_TURN.get()
    return turn if turn >= 0 else record.turn


class Clock:
    def __init__(self) -> None:
        self.t = 10_000.0

    def __call__(self) -> float:
        return self.t


# ── the replaced parts ──


class Listener:
    """What the server asks of the continuous listener, over the real addressing rule."""

    def __init__(self, addressing: Any, config: Any) -> None:
        self.addressing = addressing
        self.config = config
        self.is_running = True
        self.in_call = ""
        self.user_is_speaking = False
        self.last_detected_language = "en"
        self.waited_for_floor = 0

    async def wait_for_floor(self, max_seconds: float = 30.0) -> float:
        if self.user_is_speaking:
            self.waited_for_floor += 1
            self.user_is_speaking = False  # the person finishes their sentence; then Heliox may speak
            return 0.6
        return 0.0

    def suppress_wake_free_commands(self) -> None:
        return None

    def resume_wake_free_commands(self) -> None:
        return None

    def arm_follow_up_window(self, seconds: float | None = None, *, asked: bool = False) -> bool:
        configured = self.config.voice.follow_up_window_seconds if seconds is None else seconds
        rule = self.addressing
        rule.follow_up_seconds = max(0.0, min(float(configured), 120.0))
        rule.heliox_finished(asked=asked)
        return bool(rule.follow_up_seconds and rule.conversation_open and rule.heliox_ended_at is not None)

    def clear_follow_up_window(self) -> None:
        self.addressing.close("cleared")

    def note_not_understood(self) -> None:
        self.addressing.note_not_understood()

    def note_interruption(self, *args: Any, **kwargs: Any) -> None:
        return None


class Speaker:
    """The speech coordinator: records what would be said and moves the clock on by how long it takes to say."""

    def __init__(self, record: Record, clock: Clock, listener: Listener) -> None:
        self.record, self.clock, self.listener = record, clock, listener

    async def speak(self, text: str, channel: Any = "final_answer", dedupe_key: str = "", recorder: Any = None):
        from pilot.system.companion_speech import SpeechChannel, SpeechOutcome

        resolved = SpeechChannel(str(channel))
        self.record.said.append(
            Said(
                self.record.next(),
                _turn_of(self.record),
                text.strip(),
                resolved.value,
                self.listener.user_is_speaking,
            )
        )
        self.clock.t += max(0.4, len(text.split()) / WORDS_PER_SECOND)
        return SpeechOutcome("spoken", resolved, "")

    def status(self) -> dict[str, Any]:
        return {"active": False}

    async def stop_all(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _wire(provider: str, text: str) -> bytes:
    pieces = [text[i : i + 9] for i in range(0, len(text), 9)] or [""]
    if provider == "ollama":
        lines = [json.dumps({"message": {"content": p}, "done": False}) for p in pieces]
        lines.append(json.dumps({"message": {"content": ""}, "done": True, "prompt_eval_count": 300, "eval_count": 30}))
        return ("\n".join(lines) + "\n").encode()
    if provider == "claude":
        events = [{"type": "message_start", "message": {"usage": {"input_tokens": 300}}}]
        events += [{"type": "content_block_delta", "delta": {"type": "text_delta", "text": p}} for p in pieces]
        events.append({"type": "message_delta", "usage": {"output_tokens": 30}})
        return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
    events = [{"choices": [{"delta": {"content": p}}]} for p in pieces]
    events.append({"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 30}})
    return ("".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def fast_model(kind: str, replies: dict[str, Any], record: Record) -> Any:
    """The kind of person's fast model, answering with the current turn's scripted reply in its own wire format."""
    if kind == "none":
        return None
    import httpx

    from pilot.conversation.fast_model import HttpFastModel

    provider = {"claude": "claude", "openai": "openai", "local": "ollama"}[kind]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 1})
        turn = _turn_of(record)
        record.model_calls[turn] = record.model_calls.get(turn, 0) + 1
        reply = replies.get(turn)
        text = reply if isinstance(reply, str) else "Okay."
        return httpx.Response(200, content=_wire(provider, text), headers={"content-type": "text/event-stream"})

    async def key() -> str:
        return "session-key"

    return HttpFastModel(
        provider=provider,
        name={"claude": "claude-haiku-5-5", "openai": "gpt-4o-mini", "ollama": "qwen3.5:4b"}[provider],
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        api_key=None if provider == "ollama" else key,
        base_url="http://127.0.0.1:11434" if provider == "ollama" else "",
    )


def _configure(config: Any, kind: str) -> None:
    config.security.dry_run = False
    config.typed_decision.enabled = False
    config.voice.barge_in_enabled = False
    config.live.enabled = True
    if kind == "claude":
        config.model.provider, config.model.cloud_provider = "cloud", "claude"
    elif kind == "openai":
        config.model.provider, config.model.cloud_provider = "cloud", "openai"
    elif kind == "local":
        config.model.provider = "ollama"
    else:
        config.model.provider = "none"


async def build_server(kind: str, session: dict[str, Any], folder: Path, record: Record, clock: Clock) -> Any:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from pilot.actions import ActionPlan, ActionResult, VerificationResult
    from pilot.agents.execution_companion import CompanionReview
    from pilot.config import PilotConfig
    from pilot.conversation import layer_for
    from pilot.live.engine import LiveCompanion
    from pilot.security.permissions import PermissionChecker
    from pilot.server import PilotServer
    from pilot.system.addressing import Addressing
    from pilot.workflows.focus import FocusSession
    from pilot.workflows.notes import NoteStore
    from pilot.workflows.quick_answers import QuickAnswers
    from pilot.workflows.reminders import ReminderStore
    from pilot.workflows.routines import ShortcutStore
    from pilot.workflows.todos import TodoStore
    from pilot.workflows.user_profile import UserProfile

    config = PilotConfig()
    _configure(config, kind)
    server = PilotServer(config)
    listener = Listener(Addressing(clock=clock), config)
    server._voice_listener = listener
    server._speech_coordinator = Speaker(record, clock, listener)

    async def broadcast(method: str, params: Any = None) -> None:
        record.shown.append(Shown(record.next(), _turn_of(record), method, dict(params or {})))

    server._broadcast_notification = broadcast
    server._model_router = object()  # the model stack is up (the layer is on); which model is the kind's own
    server._todos = TodoStore(folder / "todos.json")
    server._notes = NoteStore(folder / "notes.json")
    server._quick_answers = QuickAnswers(now=lambda: NOW)
    server._user_profile = UserProfile(
        folder / "profile.json", remember=AsyncMock(return_value=None), forget=AsyncMock()
    )
    for line in session.get("profile", []):
        with contextlib.suppress(Exception):
            server._user_profile.add(line)
    server._reminders = ReminderStore(
        folder / "reminders.json", deliver=AsyncMock(), deliver_missed=AsyncMock(), clock=lambda: NOW.timestamp()
    )
    server._wall_clock = lambda: NOW  # the session's morning, not the machine's
    server._shortcuts = ShortcutStore(folder / "shortcuts.json", reserved=server._phrase_reserved)
    server._focus = FocusSession(folder / "focus.json", on_end=server._on_focus_end)

    plans: dict[str, list[dict[str, Any]]] = session.get("plans", {})
    outputs = {**DEFAULT_OUTPUTS, **session.get("outputs", {})}
    last_request: list[str] = []

    async def plan(user_input: str, **kwargs: Any) -> Any:
        from pilot.actions import Action

        if kind == "none":
            return ActionPlan(actions=[], explanation="Planning that needs a language model, and none is set up.")
        lowered = user_input.casefold()
        for key, actions in plans.items():
            if key.casefold() in lowered:
                return ActionPlan(actions=[Action.model_validate(a) for a in actions], explanation=key)
        return ActionPlan(actions=[], explanation=session.get("planner_says", "I'm not sure what to do there."))

    async def execute(run_plan: Any, **kwargs: Any) -> list[Any]:
        results = []
        for action in run_plan.actions:
            kind_of = action.action_type.value
            record.ran.append(Ran(_turn_of(record), kind_of, str(action.target or "")))
            results.append(ActionResult(action=action, success=True, output=outputs.get(kind_of, "Done")))
        return results

    class Companion:
        async def review(self, user_input: str, plan: Any, *, screen_context: str = "", others_running: Any = None):
            return CompanionReview(decision="CONTINUE", reason="ok")

    server._planner = SimpleNamespace(plan=plan)
    server._executor = SimpleNamespace(execute=execute)
    server._reflector = AsyncMock()
    server._reflector.get_improvement_context.return_value = ""
    server._multi_agent = MagicMock()
    server._multi_agent.get_routing_summary.return_value = {"assigned_agents": []}
    server._permission_checker = PermissionChecker(config)
    server._memory = AsyncMock()  # long-term memory: recorded into nowhere
    server._verifier = SimpleNamespace(verify=AsyncMock(return_value=VerificationResult(passed=True, details=["ok"])))
    # Receipts are written and read as they would be; what "do that again" repeats is the last request (below).
    server._task_receipts = SimpleNamespace(
        finalize=AsyncMock(return_value=None),
        list=AsyncMock(return_value=[]),
        answer_async=AsyncMock(return_value=None),
    )
    server._checkpoint_store = AsyncMock()
    server._orchestrator = None
    server._execution_companion = Companion()

    original_execute = server._handle_execute

    async def handle_execute(params: dict[str, Any], ws: Any, *args: Any, **kwargs: Any) -> Any:
        if str(params.get("input", "")).casefold() != "do that again":
            last_request[:] = [str(params.get("input", ""))]
        return await original_execute(params, ws, *args, **kwargs)

    server._handle_execute = handle_execute

    async def latest_receipt(*, with_actions: bool = False) -> dict[str, Any] | None:
        return {"receipt": {"request": last_request[0]}} if last_request else None

    server._latest_receipt = latest_receipt

    server._live = LiveCompanion(
        settings=lambda: config.live,
        observe=AsyncMock(return_value=None),
        emit=broadcast,
        speak=server._live_speak,
        user_speaking=lambda: listener.user_is_speaking,
        heliox_speaking=lambda: False,
        relevance_path=folder / "live_relevance.json",
        timing_path=folder / "live_timing.json",
        clock=clock,
        today=lambda: NOW.date(),
        wallclock=lambda: NOW,
        reminders=lambda: server._reminders.list(),
    )

    layer = layer_for(server)
    model = fast_model(kind, {i: t.get("model") for i, t in enumerate(session["turns"])}, record)
    layer._model = lambda: model
    layer._prepare = lambda text: None  # no speech synthesis is prepared: there is no voice here, and no audio
    server._voice_turns._clock = clock  # the withdraw window runs on the session's clock
    return server


# ── running a session ──


def _words(text: str) -> int:
    return len(text.split())


async def run_session(session: dict[str, Any], kind: str, real_model: Any = None) -> tuple[Record, list[str]]:
    """``real_model``: a real fast model to answer instead of the scripted replies (the --real run)."""
    from pilot.system.addressing import Heard
    from pilot.system.voice import voice_turn_addressed, voice_turn_id

    record = Record()
    clock = Clock()
    with tempfile.TemporaryDirectory(prefix="heliox-session-") as tmp:
        folder = Path(tmp)
        patches = [*_isolate(folder), _track_tasks(record)]
        try:
            server = await build_server(kind, session, folder, record, clock)
            if real_model is not None:
                from pilot.conversation import layer_for

                layer_for(server)._model = lambda: _counting(real_model, record)
            listener: Listener = server._voice_listener
            pending: list[asyncio.Task[Any]] = []

            def winding_down() -> bool:
                """A task is ending (stopped or denied by this turn) and has not yet given the floor back."""
                return bool(server._interactive_request_active and not server._pending_confirms)

            for index, turn in enumerate(session["turns"]):
                record.turn = index
                event = turn.get("event")
                if event == "call":
                    listener.in_call = str(turn.get("app", "Teams"))
                    continue
                if event == "call_end":
                    listener.in_call = ""
                    continue
                gap = float(turn.get("gap", DEFAULT_GAP))
                clock.t += gap
                if server._last_spoken_at != float("-inf"):
                    server._last_spoken_at -= gap  # the server's own clock is real; age what it last said instead
                text = turn["say"]
                if turn.get("continues"):
                    # The listener heard the rest of a sentence it had ended at a pause: it asks the server to
                    # withdraw the first half's dispatch for the joined sentence, and dispatches the rest alone
                    # when that is too late or unknown (pilot.system.voice, ContinuousVoiceListener._withdraw).
                    first = index - 1
                    joined = f"{session['turns'][first]['say'].rstrip(' ,.')} {text.lstrip()}"
                    joined = joined.removeprefix("Hey Heliox, ").removeprefix("Hey Heliox ")
                    record.taken[index] = "continuation"
                    asked_before = set(server._pending_confirms)
                    # The listener joins only words that start within seconds of the end of the person's own turn:
                    # the gap counts from there, while Heliox, if it had begun to answer, yields at once.
                    after_first = clock.t
                    clock.t = record.ended.get(first, clock.t) + gap
                    token = SESSION_TURN.set(index)
                    try:
                        result = await server.withdraw_voice_turn(f"v{first}", joined, gap=gap)
                    finally:
                        SESSION_TURN.reset(token)
                    clock.t = max(clock.t, after_first)
                    record.withdrawals[index] = result.status
                    if result.status != "withdrawn":

                        async def rest(words: str = text, turn_index: int = index) -> None:
                            SESSION_TURN.set(turn_index)
                            voice_turn_addressed.set("follow_up")
                            voice_turn_id.set(f"v{turn_index}")
                            await server._voice_command_dispatch(words.strip(" \t,.:;!?-"))

                        token = SESSION_TURN.set(index)
                        try:
                            task = asyncio.ensure_future(rest())
                        finally:
                            SESSION_TURN.reset(token)
                        await _settle(
                            record,
                            index,
                            task,
                            lambda before=asked_before: set(server._pending_confirms) - before,
                            winding_down,
                        )
                        (task.result() if task.done() else pending.append(task))
                    else:
                        await _settle(
                            record,
                            index,
                            None,
                            lambda before=asked_before: set(server._pending_confirms) - before,
                            winding_down,
                        )
                    continue
                voiced = float(turn.get("voiced", max(0.6, _words(text) / WORDS_PER_SECOND)))
                started = clock.t
                clock.t += voiced
                record.ended[index] = clock.t
                listener.addressing.task_open = bool(server._interactive_request_active or server._pending_confirms)
                decision = listener.addressing.decide(
                    Heard(
                        text=text,
                        started=started,
                        ended=clock.t,
                        voiced_seconds=voiced,
                        by_button=bool(turn.get("button")),
                        in_call=bool(listener.in_call),
                    )
                )
                if not (decision.act or decision.say_sorry):
                    record.not_taken[index] = decision.why
                    continue
                record.taken[index] = decision.why
                addressed = (
                    "doubtful_follow_up"
                    if decision.say_sorry
                    else decision.why
                    if decision.why in {"wake", "button"}
                    else "follow_up"
                )
                command = decision.command if decision.act else text.strip(" \t,.:;!?-")
                listener.user_is_speaking = bool(turn.get("still_talking"))

                async def dispatch(words: str = command, how: str = addressed, turn_index: int = index) -> None:
                    SESSION_TURN.set(turn_index)
                    voice_turn_addressed.set(how)
                    voice_turn_id.set(f"v{turn_index}")  # the listener numbers its dispatches the same way
                    await server._voice_command_dispatch(words)

                asked_before = set(server._pending_confirms)
                token = SESSION_TURN.set(index)
                try:
                    task = asyncio.ensure_future(dispatch())
                finally:
                    SESSION_TURN.reset(token)
                await _settle(
                    record,
                    index,
                    task,
                    lambda before=asked_before: set(server._pending_confirms) - before,
                    winding_down,
                )
                if task.done():
                    task.result()
                else:
                    pending.append(task)  # waiting on the visible approval dialog, which nobody presses here
            record.turn = len(session["turns"])
            for task in pending:
                task.cancel()
                done, _ = await asyncio.wait({task}, timeout=END_SECONDS)
                if not done:  # a task told to stop that does not end is a failure to name, not a hang
                    record.stuck.append(
                        f"a dispatch still running {END_SECONDS:.0f} s after it was cancelled at the end of the session "
                        f"({_describe(task)}; {_server_state(server)})"
                    )
        finally:
            for undo in patches:
                undo()
    return record, check_session(session, kind, record)


async def _settle(record: Record, index: int, dispatch: Any, asked: Any, winding_down: Any = lambda: False) -> None:
    """Wait until everything turn ``index`` set going has finished, or is waiting on the approval dialog after
    saying so, and no earlier task this turn ended is still winding down. Decided by the turns' own tasks and the
    server's state, never by a fixed sleep."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SETTLE_SECONDS
    running: set[Any] = set()
    while loop.time() < deadline:
        running = {task for task in record.tasks.get(index, set()) if not task.done() and not _is_timer(task)}
        if not running and not winding_down():
            return
        if asked() and any(s.turn == index and "approv" in s.text.casefold() for s in record.said):
            return  # the plan waits for the person's click; the rest of this turn waits with it
        await asyncio.sleep(0.005)
    record.stuck.append(
        f"turn {index + 1} still had work running after {SETTLE_SECONDS:.0f} s: "
        + "; ".join(_describe(task) for task in sorted(running, key=id)[:4])
        + (" (and a task it ended had not given the floor back)" if winding_down() else "")
    )


# Work a turn starts that is meant to outlive it: a pomodoro's or a focus session's own timer. Not turn work.
TIMERS = {"_wait_and_end"}


def _is_timer(task: Any) -> bool:
    frames = task.get_stack(limit=8)
    return any(frame.f_code.co_name in TIMERS for frame in frames)


def _describe(task: Any) -> str:
    """Where a task is waiting, for a failure message: its name and the innermost frame."""
    frames = task.get_stack(limit=8) if not task.done() else []
    where = f"{frames[-1].f_code.co_name} at line {frames[-1].f_lineno}" if frames else "finished"
    return f"{task.get_name()} in {where}"


def _server_state(server: Any) -> str:
    return (
        f"interactive task active: {bool(server._interactive_request_active)}, "
        f"approvals waiting: {sorted(server._pending_confirms)}"
    )


def _track_tasks(record: Record) -> Any:
    """Record every task by the turn it was created in (the context it starts with); gives back how to undo it."""
    loop = asyncio.get_running_loop()
    before = loop.get_task_factory()

    def factory(loop: Any, coro: Any, **kwargs: Any) -> Any:
        task = before(loop, coro, **kwargs) if before is not None else asyncio.Task(coro, loop=loop, **kwargs)
        context = kwargs.get("context")
        turn = context.get(SESSION_TURN, -1) if context is not None else SESSION_TURN.get()
        if turn >= 0:
            record.tasks.setdefault(turn, set()).add(task)
        return task

    loop.set_task_factory(factory)
    return lambda: loop.set_task_factory(before)


def _isolate(folder: Path) -> list[Any]:
    """Point the server's data folder at the session's temporary folder; give back how to undo it."""
    import pilot.server as server_module

    undo = []
    for module, name in ((server_module, "DATA_DIR"),):
        before = getattr(module, name)
        setattr(module, name, folder)
        undo.append(lambda m=module, n=name, b=before: setattr(m, n, b))
    return undo


# ── checks ──


def _has(text: str, option: str) -> bool:
    return any(part.casefold() in text.casefold() for part in option.split("|"))


def _expect(turn: dict[str, Any], kind: str) -> dict[str, Any]:
    expect = dict(turn.get("expect", {}))
    if kind == "none":
        expect.update(turn.get("expect_none", {}))
    if kind in ("claude", "openai", "local"):
        expect.update(turn.get("expect_model", {}))
    return expect


STOCK = re.compile(
    r"^(?:okay\.|one moment\.|let me think\.|give me a second\.|sorry\?|on it\.|got it\.|"
    r"i can't answer that without a language model\.|not without the language model, sorry\.)$",
    re.I,
)


def check_session(session: dict[str, Any], kind: str, record: Record) -> list[str]:
    """Every way the session went wrong for this kind of person, one line each ("" list: it passed)."""
    failures: list[str] = [f"never finished: {line}" for line in record.stuck]
    for index, turn in enumerate(session["turns"]):
        if turn.get("event"):
            continue
        expect = _expect(turn, kind)
        said = [s for s in record.said if s.turn == index]
        heard = " ".join(s.text for s in said)
        where = f"turn {index + 1} ({turn['say']!r})"
        taken = index in record.taken
        if "taken" in expect and taken != expect["taken"]:
            why = record.taken.get(index) or record.not_taken.get(index, "")
            failures.append(f"{where}: {'taken' if taken else 'not taken'} ({why}); expected the opposite")
        for option in expect.get("has", []):
            if not _has(heard, option):
                failures.append(f"{where}: expected {option!r} in what was said; heard {heard!r}")
        for option in expect.get("not", []):
            if _has(heard, option):
                failures.append(f"{where}: {option!r} must not be said; heard {heard!r}")
        if expect.get("silent") and said:
            failures.append(f"{where}: expected silence; heard {heard!r}")
        ran = [r.action_type for r in record.ran if r.turn == index]
        for action in expect.get("runs", []):
            if action not in ran:
                failures.append(f"{where}: expected {action} to run; ran {ran}")
        if "runs_nothing" in expect and expect["runs_nothing"] and ran:
            failures.append(f"{where}: expected nothing to run; ran {ran}")
        asked = any(s.method == "confirm_required" and s.turn == index for s in record.shown)
        if "approval" in expect and asked != expect["approval"]:
            failures.append(f"{where}: approval {'asked' if asked else 'not asked'}; expected the opposite")
        if "withdrawal" in expect and record.withdrawals.get(index) != expect["withdrawal"]:
            failures.append(
                f"{where}: the first half was {record.withdrawals.get(index, 'not asked')}; "
                f"expected {expect['withdrawal']}"
            )
        if "model" in expect:
            called = record.model_calls.get(index, 0) > 0
            if called != expect["model"]:
                failures.append(f"{where}: the model was {'called' if called else 'not called'}; expected the opposite")
        reply = " ".join(s.text for s in said if s.text not in WAIT_SIGNS)  # "One moment." is not the reply
        if not turn.get("detail") and _words(reply) > REPLY_WORD_LIMIT:
            failures.append(f"{where}: {_words(reply)} words said without being asked for detail: {reply!r}")
    everything = [line.text.casefold() for line in record.said]
    for phrase in session.get("said_once", []):
        times = sum(phrase.casefold() in line for line in everything)
        if times != 1:
            failures.append(f"session: {phrase!r} said {times} times; expected once")
    for action, times in session.get("runs_count", {}).items():
        ran = sum(r.action_type == action for r in record.ran)
        if ran != times:
            failures.append(f"session: {action} ran {ran} times; expected {times}")
    for action in session.get("never_runs", []):
        ran = [r for r in record.ran if r.action_type == action]
        if ran:
            failures.append(f"session: {action} ran (turn {ran[0].turn + 1}), which must never happen")
    previous: Said | None = None
    for line in record.said:
        if line.over_the_person:
            failures.append(f"turn {line.turn + 1}: said while the person was mid-turn: {line.text!r}")
        repeat_ok = 0 <= line.turn < len(session["turns"]) and session["turns"][line.turn].get("repeat")
        if previous is not None and line.text == previous.text and STOCK.match(line.text) and not repeat_ok:
            failures.append(f"turn {line.turn + 1}: the stock phrase {line.text!r} said twice running")
        previous = line
    # Spoken: once a later turn has been answered aloud, an earlier turn says nothing more (the person would hear
    # the end of an old request after the answer to a new one). Shown: each turn's items keep their own order. A
    # turn's work is the turn's by its id (SESSION_TURN), so a task that ends during a later turn stays its own.
    latest = -1
    for line in sorted(record.said, key=lambda item: item.seq):
        if line.turn < latest:
            failures.append(f"output order: turn {line.turn + 1} spoke after turn {latest + 1}: {line.text!r}")
            break
        latest = max(latest, line.turn)
    for turn in {item.turn for item in record.shown}:
        seqs = [item.seq for item in record.shown if item.turn == turn]
        if seqs != sorted(seqs):
            failures.append(f"output order: turn {turn + 1}'s screen updates came out of order")
    return failures


# ── the run ──


def load_pack(path: Path = PACK) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["sessions"]


class _counting:
    """A real model that counts its calls per turn in the record, as the stub does."""

    def __init__(self, model: Any, record: Record) -> None:
        self._model, self._record = model, record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def stream(self, system: str, messages: list[dict[str, Any]]) -> Any:
        turn = _turn_of(self._record)
        self._record.model_calls[turn] = self._record.model_calls.get(turn, 0) + 1
        return self._model.stream(system, messages)


async def run_all(sessions: list[dict[str, Any]], kinds: tuple[str, ...], real_model: Any = None) -> dict[str, Any]:
    rows = []
    for session in sessions:
        for kind in kinds:
            record, failures = await run_session(session, kind, real_model)
            rows.append(
                {
                    "session": session["name"],
                    "kind": kind,
                    "passed": not failures,
                    "failures": failures,
                    "said": [(s.turn + 1, s.text) for s in record.said],
                    "ran": [(r.turn + 1, r.action_type) for r in record.ran],
                }
            )
    table = {kind: f"{sum(r['passed'] for r in rows if r['kind'] == kind)}/{len(sessions)}" for kind in kinds}
    return {"table": table, "rows": rows}


async def _real_run(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """The Claude road against the real fast model, once, with the budget tracker on; scored by the same rules."""
    from pilot.config import DB_FILE, PilotConfig
    from pilot.conversation.fast_model import build_fast_model
    from pilot.models.budget_tracker import BudgetTracker
    from pilot.security.vault import KeyVault

    config = PilotConfig.load()
    config.save = lambda: None
    if config.model.cloud_provider != "claude":
        raise SystemExit("--real runs the Claude road only; the configured cloud provider is not Claude")
    budget = BudgetTracker(config.model, str(DB_FILE))
    await budget.initialize()
    before = budget._monthly_cost
    rows: list[dict[str, Any]] = []
    calls = 0
    firsts: list[float | None] = []  # seconds to the first text, per call (None: no text came)
    name = ""
    for session in sessions:
        # A fresh model per session, as a person's sessions are minutes apart: a slow answer's rest in one session
        # must not answer the next session's turns.
        model = build_fast_model(config, vault=KeyVault(config), router=None, budget=budget)
        rows += (await run_all([session], ("claude",), real_model=model))["rows"]
        calls += len(getattr(model, "calls", []))
        firsts += [
            round(c.first_text_at - c.started, 2) if c.first_text_at is not None else None
            for c in getattr(model, "calls", [])
        ]
        name = getattr(model, "name", "")
        await asyncio.sleep(0.3)  # the last call's usage is recorded as its stream closes
    summary = {"table": {"claude": f"{sum(r['passed'] for r in rows)}/{len(sessions)}"}, "rows": rows}
    summary["spent_usd"] = round(budget._monthly_cost - before, 5)
    summary["calls"] = calls
    summary["first_text_seconds"] = firsts
    summary["model"] = name
    await budget.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.conversation_eval.sessions")
    parser.add_argument("--only", default="")
    parser.add_argument("--kind", default="")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--real",
        action="store_true",
        help="the Claude road only, against the configured Claude fast model; costs money, recorded in the budget",
    )
    args = parser.parse_args(argv)
    sessions = [s for s in load_pack() if args.only in s["name"]]
    kinds = (args.kind,) if args.kind else KINDS
    if args.real:
        summary = asyncio.run(_real_run(sessions))
    else:
        summary = asyncio.run(run_all(sessions, kinds))
    print("sessions passed:", summary["table"])
    if "spent_usd" in summary:
        print("budget tracker:", summary["spent_usd"], "USD over", summary["calls"], "calls")
    for row in summary["rows"]:
        for failure in row["failures"]:
            print(f"[{row['kind']}] {row['session']}: {failure}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["KINDS", "NOW", "PACK", "check_session", "load_pack", "run_all", "run_session", "date"]
