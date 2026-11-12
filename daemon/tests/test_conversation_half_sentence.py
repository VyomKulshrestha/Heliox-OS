"""A reply to half a sentence (scratchpad bench_r5_split_cases.txt: a first half that sounds finished, "Remind me to
call the bank." | "tomorrow morning", had its reply chosen before the withdrawal arrived).

- Nothing that changes anything starts while the person is audibly going on. In silence a change runs at once,
  except a destructive or not undoable plan, which waits out the continuation window.
- Too late: Heliox says what it did, and offers the undo where one exists.
- A withdrawn first half leaves no trace: not in the conversation, not as the last thing said, not on screen, not in
  what Live compares against.
- The joined sentence is answered as one sentence, never as a correction of its first half.

On the real server, with a fake listener that holds what the real one exposes (its last turn, whether it hears or
is recognising speech, the continuation it decided)."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.config import PilotConfig, UndoConfig
from pilot.conversation import layer_for
from pilot.conversation import turns as turns_module
from pilot.conversation.fast_model import HttpFastModel
from pilot.live import spoken
from pilot.live.held import Turns
from pilot.server import PilotServer
from pilot.system.companion_speech import SpeechChannel, SpeechOutcome
from pilot.system.voice import voice_turn_addressed, voice_turn_id
from pilot.workflows.todos import TodoStore
from pilot.workflows.undo_journal import UndoJournal


class Listener:
    """What the real listener exposes about the turn it just dispatched and what it hears next."""

    def __init__(self, turn_id: str = "v1") -> None:
        self.is_running = False  # Heliox's own speech paths leave it alone
        self.user_is_speaking = False
        self._busy = False
        self._continuation_of = ("", "", "")
        self._last_turn = (time.monotonic(), "", turn_id)
        self._recorder = SimpleNamespace(speech_started_at=None, _frame_captured_at=None)

    def goes_on(self) -> None:
        """The person starts speaking again 0.4 s after the turn ended."""
        self.user_is_speaking = True
        self._recorder.speech_started_at = self._last_turn[0] + 0.4

    def decided_continuation(self, first: str, turn_id: str, rest: str) -> None:
        self.user_is_speaking = False
        self._continuation_of = (first, turn_id, rest)
        self._last_turn = None


class Rig:
    def __init__(self, reply: str = "Whiskers is a good name.") -> None:
        self.server = server = PilotServer(PilotConfig())
        self.heard: list[str] = []
        self.shown: list[tuple[str, dict]] = []
        self.ran: list[list[str]] = []
        self.planned: list[dict] = []
        self.requests: list[dict] = []
        self.speaking = asyncio.Event()
        self.listener = Listener()
        server._voice_listener = self.listener

        async def speak(text, channel="final_answer", dedupe_key="", recorder=None):
            self.heard.append(text)
            self.speaking.set()
            if str(dedupe_key).startswith("conv:") and "Whiskers" in text:
                await asyncio.sleep(10)  # the first half's reply is still playing when the rest arrives
            return SpeechOutcome("spoken", SpeechChannel(str(channel)), "")

        async def broadcast(kind, payload):
            self.shown.append((kind, dict(payload)))

        server._speech_coordinator.speak = speak
        server._broadcast_notification = broadcast
        server._model_router = object()

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            self.requests.append(body)
            asked = body["messages"][-1]["content"].casefold()
            text = "Mochi suits a sleepy cat." if "sleep" in asked else reply
            event = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
            return httpx.Response(
                200, content=f"data: {json.dumps(event)}\n\n".encode(), headers={"content-type": "text/event-stream"}
            )

        async def key() -> str:
            return "k"

        model = HttpFastModel(
            provider="claude",
            name="claude-haiku-5-5",
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            api_key=key,
        )
        layer_for(server)._model = lambda: model

        class Executor:
            async def execute(_self, plan, **kwargs):
                self.ran.append([a.action_type.value for a in plan.actions])
                return []

        server._executor = Executor()

    def plans(self, kind: ActionType) -> None:
        """The planner: a plan of one ``kind`` action for every request, sent straight to the executor."""

        async def execute(params, ws, *args, **kwargs):
            self.planned.append(params)
            action = Action.model_construct(action_type=kind, target="draft.txt")
            await self.server._execute_tracked(ActionPlan.model_construct(actions=[action]))
            return {"status": "success", "message": "Done.", "results": []}

        self.server._handle_execute = execute

    def dispatch(self, words: str, turn_id: str = "v1") -> asyncio.Task:
        async def run() -> None:
            voice_turn_addressed.set("wake")
            voice_turn_id.set(turn_id)
            await self.server._voice_command_dispatch(words)

        return asyncio.ensure_future(run())

    async def withdraw(self, first: str, rest: str, gap: float = 0.4) -> str:
        joined = f"{first.rstrip('.')} {rest}"
        self.listener.decided_continuation(first, "v1", rest)
        result = await self.server.withdraw_voice_turn("v1", joined, gap=gap)
        await asyncio.gather(*getattr(self.server, "_background_dispatches", set()))
        return result.status


async def wait_until(condition, seconds: float = 3.0) -> None:
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


# ── nothing that changes anything starts while the person is audibly going on ──


async def test_a_change_continued_while_its_plan_waits_never_runs():
    """ "Rename the report." | "to the March version": the person goes on while the first half's plan is at the
    executor's door. It waits for the listener, and the continuation withdraws it."""
    rig = Rig()
    rig.plans(ActionType.FILE_MOVE)

    async def execute(params, ws, *args, **kwargs):
        rig.planned.append(params)
        if len(rig.planned) == 1:
            rig.listener.goes_on()  # the person goes on while the plan is being made
        action = Action.model_construct(action_type=ActionType.FILE_MOVE, target="report.txt")
        await rig.server._execute_tracked(ActionPlan.model_construct(actions=[action]))
        return {"status": "success", "message": "Done.", "results": []}

    rig.server._handle_execute = execute
    first = rig.dispatch("Rename the report.")
    await wait_until(lambda: rig.planned)
    await asyncio.sleep(0.2)
    assert rig.ran == []  # held while the person is heard
    assert await rig.withdraw("Rename the report.", "to the March version") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    assert [p["input"] for p in rig.planned] == ["Rename the report.", "Rename the report to the March version"]
    assert rig.ran == [["file_move"]]  # once, for the whole sentence


async def test_a_first_half_heard_going_on_waits_before_any_road_takes_it():
    """The person is already speaking again when the first half reaches the server (the bench's 0.4 to 0.55 s
    gaps): it is not planned, answered or asked about until the listener has decided."""
    rig = Rig()
    rig.plans(ActionType.FILE_DELETE)
    rig.listener.goes_on()
    first = rig.dispatch("Delete the draft.")
    await asyncio.sleep(0.3)
    assert rig.planned == [] and rig.heard == []
    assert await rig.withdraw("Delete the draft.", "of the old report") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    assert [p["input"] for p in rig.planned] == ["Delete the draft of the old report"]
    assert rig.ran == [["file_delete"]]


async def test_a_change_made_in_code_waits_while_the_person_is_heard(tmp_path):
    rig = Rig()
    rig.server._todos = TodoStore(tmp_path / "todos.json")
    rig.listener.goes_on()
    first = rig.dispatch("add milk to my list")
    await asyncio.sleep(0.3)
    assert rig.server._todos.list() == []  # not added while the person is still talking
    rig.listener.user_is_speaking = False  # what they said was not a continuation
    async with asyncio.timeout(3.0):
        await first
    assert [item["text"] for item in rig.server._todos.list()] == ["milk"]


async def test_speech_that_is_not_a_continuation_lets_the_first_half_run():
    rig = Rig()
    rig.plans(ActionType.FILE_MOVE)
    rig.listener.goes_on()
    first = rig.dispatch("Rename the report.")
    await asyncio.sleep(0.2)
    assert rig.ran == []
    rig.listener.user_is_speaking = False  # what they said was for someone else: not taken, not a continuation
    async with asyncio.timeout(3.0):
        await first
    assert rig.ran == [["file_move"]]


# ── in silence a change runs at once; a destructive one waits out the window ──


@pytest.mark.parametrize("kind", [ActionType.FILE_MOVE, ActionType.VOLUME_SET, ActionType.OPEN_APPLICATION])
async def test_in_silence_a_change_runs_at_once(kind):
    rig = Rig()
    rig.plans(kind)
    started = time.monotonic()
    await rig.dispatch("Rename the report.")
    assert rig.ran == [[kind.value]]
    assert time.monotonic() - started < 1.0  # the window (3 s) is still open; it does not wait for it


async def test_in_silence_a_change_made_in_code_runs_at_once(tmp_path):
    rig = Rig()
    rig.server._todos = TodoStore(tmp_path / "todos.json")
    started = time.monotonic()
    await rig.dispatch("add milk to my list")
    assert [item["text"] for item in rig.server._todos.list()] == ["milk"]
    assert time.monotonic() - started < 1.0


async def test_in_silence_a_destructive_plan_waits_out_the_window(monkeypatch):
    monkeypatch.setattr(turns_module, "CONTINUATION_SECONDS", 0.4)
    rig = Rig()
    rig.plans(ActionType.FILE_DELETE)
    started = time.monotonic()
    await rig.dispatch("Delete the draft.")
    assert rig.ran == [["file_delete"]]
    assert time.monotonic() - started >= 0.35


async def test_a_destructive_plan_continued_in_the_window_never_runs():
    """ "Delete the draft." | "of the old report", the rest begun after the plan was ready."""
    rig = Rig()
    rig.plans(ActionType.FILE_DELETE)
    first = rig.dispatch("Delete the draft.")
    await wait_until(lambda: rig.planned)
    await asyncio.sleep(0.2)
    assert rig.ran == []  # waiting out the window, in silence
    rig.listener.goes_on()
    assert await rig.withdraw("Delete the draft.", "of the old report") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    assert rig.ran == [["file_delete"]]


def test_destructive_or_not_undoable_comes_from_the_risk_classification():
    def plan(kind):
        return [Action.model_construct(action_type=kind, target="x", requires_root=False)]

    assert turns_module.must_wait_out_window(plan(ActionType.FILE_DELETE))  # destructive
    assert turns_module.must_wait_out_window(plan(ActionType.SHELL_COMMAND))  # would need an approval
    assert turns_module.must_wait_out_window(plan(ActionType.API_SEND_EMAIL))  # not undoable
    assert not turns_module.must_wait_out_window(plan(ActionType.VOLUME_SET))
    assert not turns_module.must_wait_out_window(plan(ActionType.FILE_MOVE))
    assert not turns_module.changes_something(plan(ActionType.OPEN_APPLICATION))


async def test_a_turn_not_from_the_listener_does_not_wait():
    rig = Rig()
    rig.listener._last_turn = (time.monotonic(), "", "v7")  # the listener's last turn is another one
    rig.plans(ActionType.FILE_DELETE)
    started = time.monotonic()
    await rig.dispatch("Delete the draft.")
    assert rig.ran == [["file_delete"]] and time.monotonic() - started < 1.0


# ── too late: Heliox says what it did, with the undo where one exists ──


async def test_too_late_offers_the_undo_and_the_undo_works(tmp_path):
    """ "Save the note." runs in silence, then the person goes on: too late. Heliox says what it did and that undo
    takes it back; "undo" then opens the undo for that task, and undoing restores the file."""
    rig = Rig()
    journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    rig.server._undo_journal = journal
    note = tmp_path / "note.txt"
    note.write_text("before", encoding="utf-8")

    class Executor:
        async def execute(_self, plan, **kwargs):
            rig.ran.append([a.action_type.value for a in plan.actions])
            for action in plan.actions:
                token = await journal.before(action, task_key="task-v1")
                note.write_text("after", encoding="utf-8")
                await journal.after(token, ActionResult(action=action, success=True, output="saved"))
            return []

    rig.server._executor = Executor()
    write = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(note),
        parameters=FileParams(path=str(note), content="after"),
    )

    async def execute(params, ws, *args, **kwargs):
        rig.planned.append(params)
        if len(rig.planned) == 1:
            await rig.server._execute_tracked(ActionPlan(actions=[write]))
        return {"status": "success", "message": "Done.", "results": []}

    rig.server._handle_execute = execute
    await rig.dispatch("Save the note.")
    assert note.read_text(encoding="utf-8") == "after"

    rig.listener.goes_on()
    result = await rig.server.withdraw_voice_turn("v1", "Save the note as a draft", gap=0.4)
    assert result.status == "too_late"
    assert result.said.endswith("Say undo to take it back."), result.said
    assert result.said in rig.heard

    rig.listener._last_turn = None
    await rig.dispatch("undo", turn_id="v2")
    offers = [payload for kind, payload in rig.shown if kind == "undo_offer"]
    assert offers and offers[-1]["task_key"] == "task-v1"
    preview = await journal.preview("task-v1")
    await journal.undo("task-v1", preview["preview_token"])
    assert note.read_text(encoding="utf-8") == "before"


async def test_too_late_without_an_undo_does_not_offer_one(tmp_path):
    rig = Rig()
    rig.server._undo_journal = UndoJournal(tmp_path / "undo", config=UndoConfig())
    rig.server._todos = TodoStore(tmp_path / "todos.json")
    await rig.dispatch("add milk to my list")
    rig.listener.goes_on()
    result = await rig.server.withdraw_voice_turn("v1", "add milk to my list for the party", gap=0.4)
    assert result.status == "too_late"
    assert "undo" not in result.said.casefold(), result.said  # a to-do has no undo; it is not offered


# ── a cut reply leaves no trace, and the joined sentence is one sentence ──


async def test_a_cut_reply_to_the_first_half_leaves_no_trace():
    rig = Rig()
    rig.listener._last_turn = None  # nothing to wait for: the reply starts, as in the bench
    first = rig.dispatch("What's a good name for a cat.")
    await rig.speaking.wait()
    assert rig.heard == ["Whiskers is a good name."]  # being said when the rest of the sentence arrives
    rig.listener._last_turn = (time.monotonic(), "", "v1")
    assert await rig.withdraw("What's a good name for a cat.", "that likes to sleep") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)

    conversation = layer_for(rig.server).memory.recent()
    said = [(turn.role, turn.text) for turn in conversation]
    assert said == [
        ("user", "What's a good name for a cat that likes to sleep"),
        ("assistant", "Mochi suits a sleepy cat."),
    ]  # one sentence and its answer: no first half, no cut-off reply
    joined_request = rig.requests[-1]["messages"]
    assert "Whiskers" not in json.dumps(joined_request)
    assert sum(1 for m in joined_request if m["role"] == "user") == 1  # asked as one sentence, not a correction
    assert rig.server._last_spoken == "Mochi suits a sleepy cat."  # "say that again" is about the whole sentence


async def test_a_withdrawn_first_half_is_not_the_last_thing_said():
    rig = Rig()
    rig.listener._last_turn = None
    first = rig.dispatch("What's a good name for a cat.")
    await rig.speaking.wait()
    assert rig.server._last_spoken == "Whiskers is a good name."

    async def held_back(_words):  # the joined sentence not yet taken: the state it would start from
        return None

    rig.server._voice_turn = held_back
    assert await rig.withdraw("What's a good name for a cat.", "that likes to sleep") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    assert rig.server._last_spoken == ""
    assert await rig.server._say_again_request("say that again") == "I haven't said anything yet."
    assert layer_for(rig.server).memory.recent() == []


async def test_a_withdrawn_request_leaves_the_screen_and_the_planner_sees_one_sentence():
    rig = Rig()
    rig.plans(ActionType.FILE_DELETE)
    first = rig.dispatch("Delete the draft.")
    await wait_until(lambda: rig.planned)
    rig.listener.goes_on()
    assert await rig.withdraw("Delete the draft.", "of the old report") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    kinds = [kind for kind, _ in rig.shown]
    assert ("voice_withdrawn", {"command": "Delete the draft.", "turn_id": "v1"}) in rig.shown
    assert kinds.index("voice_withdrawn") < max(i for i, k in enumerate(kinds) if k == "voice_command")
    joined_context = rig.planned[-1].get("_conversation_context")
    lines = joined_context.lines if joined_context is not None else ()
    assert not any(text == "Delete the draft." for _who, text in lines)


async def test_live_does_not_compare_the_joined_sentence_with_its_first_half():
    rig = Rig()
    compared = Turns()

    class Live:
        turns = compared

        def hold(self, text):
            compared.add(spoken.to_written(text))
            return None

        async def handle_reply(self, _text):
            return None

        def __getattr__(self, _name):  # every other hook: Live has nothing to say
            return lambda *_a, **_k: None

    rig.server._live = Live()
    rig.plans(ActionType.FILE_DELETE)
    first = rig.dispatch("Delete the invoice for 400.")
    await wait_until(lambda: rig.planned)
    rig.listener.goes_on()
    assert await rig.withdraw("Delete the invoice for 400.", "and the receipt") == "withdrawn"
    await asyncio.gather(first, return_exceptions=True)
    texts = [turn.text for turn in compared.turns]
    assert texts == [spoken.to_written("Delete the invoice for 400 and the receipt")], texts


@pytest.mark.parametrize("window", [True, False])
def test_the_wait_reads_only_the_listener(window):
    listener = Listener("v3")
    assert turns_module.continuation_possible(listener, "v3", window=True)
    assert not turns_module.continuation_possible(listener, "v3", window=False)
    assert not turns_module.continuation_possible(listener, "v2", window=window)  # not its last turn
    assert not turns_module.continuation_possible(None, "v3", window=window)
    listener.goes_on()
    assert turns_module.continuation_possible(listener, "v3", window=window)
    listener._recorder.speech_started_at = listener._last_turn[0] - 1.0  # its own words, before its end
    assert turns_module.continuation_possible(listener, "v3", window=window) is window
    listener.decided_continuation("x", "v3", "y")
    assert turns_module.continuation_possible(listener, "v3", window=window)
