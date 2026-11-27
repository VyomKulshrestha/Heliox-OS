"""The live run of 26 November, one finding at a time, where the fix series meet: the run's own request, from the
router to the reply, with fakes (a fake planner answer, fake handlers), asserting what is shown and spoken.

Each series tested its own finding. This file holds the regression net across them: the router's catalog net hands
the request to the planner, the plan check refuses a plan for another turn, the sentences say what a handler really
returned, failures are said in words, the claim check leaves Heliox's own notices alone, and the review timing lets a
harmless plan run beside its review. Nothing here runs a real handler, a model, a browser or the system store.
"""

from __future__ import annotations

import json

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, ApiRequestParams, FileParams, TriggerParams
from pilot.reply_rules import ear_faults, screen_faults

WORK = r"C:\Users\me\work"


def _said(action: Action, output: str):
    from pilot.result_sentences import step_said

    return step_said(ActionResult(action=action, success=True, output=output))


def _plain(text: str) -> None:
    assert text.strip()
    assert ear_faults(text) == [], text
    for raw in ("{", "Errno", "WinError", "sha256", "\\\\"):
        assert raw not in text, text


# ── 1. The planner planned an earlier turn ───────────────────────────────────────────────────────────────────────


def test_1_a_plan_for_the_previous_turns_file_is_refused_and_said_plainly() -> None:
    from pilot.agents.request_match import check_plan, first_refusal, is_refusal

    copy = Action(
        action_type=ActionType.FILE_COPY,
        target=WORK + r"\missing.txt",
        parameters=FileParams(path=WORK + r"\missing.txt", destination=WORK + r"\copy.txt"),
    )
    mismatch = check_plan(f"Delete {WORK}\\note_backup.txt", [copy])
    assert mismatch is not None
    sentence = first_refusal(mismatch, f"Delete {WORK}\\note_backup.txt")
    assert is_refusal(sentence) and sentence.startswith("I stopped before doing anything")


@pytest.mark.asyncio
async def test_1_the_refusal_is_the_reply_shown_and_spoken_and_the_claim_check_keeps_it() -> None:
    from pilot.conversation import work_claims
    from tests.test_reply_never_empty import _Executor, _Planner, _server, _ws

    refusal = (
        "I stopped before doing anything: the plan I made would copy missing.txt to copy.txt, which is not what you "
        "asked. Nothing was changed."
    )
    planner = _Planner(ActionPlan(actions=[], error=refusal))
    executor = _Executor()
    server = _server(planner, executor)

    response = await server._handle_execute({"input": "Delete note_backup.txt"}, _ws(server))

    assert executor.runs == 0 and planner.calls == 1  # final: never planned again
    assert "I stopped before doing anything" in response["message"]
    assert response["spoken"].strip() and ear_faults(response["spoken"]) == []
    kept, dropped = work_claims.without_unbacked(response["message"], {"known": True, "queued": []})
    assert dropped == [] and kept == response["message"]


# ── 2. Requests the catalog names were answered as talk ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "words",
    [
        "Append the line 'hello' to notes.txt",
        "Compute the sha256 hash of notes.txt",
        "Post 'build finished' to Slack",
        "Watch the file notes.txt",
        "Fetch https://example.com and tell me the title",
    ],
)
def test_2_requests_the_catalog_names_reach_the_planner(words: str) -> None:
    from pilot.conversation.router import TurnKind, route

    assert route(words).kind is not TurnKind.CONVERSATION


# ── 3 and 4. Battery and memory went to the monitor agent; a 404 crashed the request ────────────────────────────


def test_3_the_battery_is_said_from_the_readers_text() -> None:
    from pilot.actions import SystemInfoParams

    battery = Action(action_type=ActionType.BATTERY_INFO, target="battery", parameters=SystemInfoParams())
    said = _said(battery, "=== Battery ===\n  Charge: 68%\n  Status: Plugged in\n  Time remaining: N/A")
    assert said.spoken == "The battery is at 68 percent, charging."
    memory = Action(action_type=ActionType.MEMORY_USAGE, target="memory", parameters=SystemInfoParams())
    said = _said(memory, "=== Memory ===\n  Total: 16.0 GB\n  Used:  12.0 GB (75.0%)\n  Free: 4.0 GB")
    assert said.spoken == "12 of 16 gigabytes of memory are in use, 75 percent."


def test_4_an_http_404_is_a_failure_said_in_words() -> None:
    from pilot.failure_sentences import failure_sentence

    get = Action(
        action_type=ActionType.API_REQUEST,
        target="https://example.com/missing",
        parameters=ApiRequestParams(method="GET", url="https://example.com/missing"),
    )
    shown, spoken = failure_sentence(
        get,
        "GET to https://example.com failed with HTTP 404 Not Found: the server has nothing at this address; the same "
        "request will get the same answer",
    )
    _plain(spoken)
    assert shown.strip()


# ── 5. Raw handler JSON was the reply ───────────────────────────────────────────────────────────────────────────


def test_5_a_write_a_trigger_and_a_delete_are_said_without_their_raw_output() -> None:
    write = Action(
        action_type=ActionType.FILE_WRITE,
        target=WORK + r"\notes\note.txt",
        parameters=FileParams(path=WORK + r"\notes\note.txt", content="hello"),
    )
    _plain(_said(write, f"Written 5 bytes to {WORK}\\notes\\note.txt (sha256 {'ab12' * 16}, 5 bytes on disk)").spoken)

    entry = {
        "id": "cb220aa1",
        "name": "notes_watch",
        "trigger_type": "file_modified",
        "condition": {"path": WORK + r"\notes.txt"},
        "action_command": "add a line to log.txt",
        "steps": [
            {"action_type": "file_read", "target": "log.txt", "parameters": {"path": WORK + r"\log.txt"}},
            {
                "action_type": "file_write",
                "target": "log.txt",
                "parameters": {"path": WORK + r"\log.txt", "content": "{PREV_OUTPUT}\nseen a change"},
                "use_previous_output": True,
            },
        ],
        "steps_approved": True,
    }
    create = Action(action_type=ActionType.TRIGGER_CREATE, target="notes_watch", parameters=TriggerParams())
    said = _said(create, json.dumps(entry))
    assert said.spoken == 'I\'ll watch notes.txt and add "seen a change" to log.txt when it changes.'
    delete = Action(action_type=ActionType.TRIGGER_DELETE, target="notes", parameters=TriggerParams())
    removed = {**entry, "watched": "notes.txt", "deleted": True, "text": "Deleted the trigger (id cb220aa1)."}
    assert _said(delete, json.dumps(removed)).spoken == "I've stopped watching notes.txt."


# ── 6. A failure was said as raw system text, and asked again three times ───────────────────────────────────────


def test_6_a_permission_failure_is_final_and_said_in_words() -> None:
    from pilot.failure_sentences import failure_sentence, is_final

    write = Action(
        action_type=ActionType.FILE_WRITE,
        target=WORK + r"\readonly\blocked.txt",
        parameters=FileParams(path=WORK + r"\readonly\blocked.txt", content="hello"),
    )
    error = f"[Errno 13] Permission denied: '{WORK}\\readonly\\blocked.txt'"
    _shown, spoken = failure_sentence(write, error, platform="windows")
    _plain(spoken)
    assert is_final(write, error, "write hello to blocked.txt", platform="windows")


# ── 7. Learned failures asked for approval ──────────────────────────────────────────────────────────────────────


def test_7_copies_that_failed_without_changing_anything_ask_for_nothing() -> None:
    from pilot.security.outcome_evidence import Outcome, OutcomeEvidence

    copy = Action(
        action_type=ActionType.FILE_COPY,
        target=WORK + r"\a.txt",
        parameters=FileParams(path=WORK + r"\a.txt", destination=WORK + r"\b.txt"),
    )
    evidence = OutcomeEvidence(clock=lambda: 1_000_000.0)
    for _ in range(4):
        evidence.record(copy, Outcome.FAILED_UNCHANGED, cause="the file was missing")
    assert evidence.assess(copy).score == 0.0


# ── 8. Triggers ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_8_a_guessed_trigger_type_is_resolved_and_a_fire_is_said_in_first_person() -> None:
    from pilot.result_sentences import trigger_fire_sentence
    from pilot.system.triggers import resolve_trigger_type

    assert resolve_trigger_type("file_change") == "file_modified"
    trigger = {
        "trigger_type": "file_modified",
        "condition": {"path": WORK + r"\notes.txt"},
        "action_command": "tell me",
    }
    sentence = trigger_fire_sentence(trigger, "ran")
    assert sentence == "The trigger that watches notes.txt fired, and I did what it asks: tell you."
    _plain(sentence)


def test_8_a_talk_reply_that_repeats_a_fire_is_not_taken_back() -> None:
    from pilot.conversation import work_claims

    fired = 'The trigger that watches notes.txt fired, and I did what it asks: add "seen" to log.txt.'
    reply = "I've appended the line to log.txt."
    assert work_claims.unbacked(reply, {"known": True, "queued": []}) == "done"
    assert work_claims.unbacked(reply, {"known": True, "queued": [], "said": [fired]}) == ""


# ── 9 and 10. The credential store, and a typed secret ──────────────────────────────────────────────────────────


def test_9_the_suite_never_reaches_this_computers_credential_store() -> None:
    from pilot.security import credential_store

    assert credential_store.mode() in {"file", "none"}


def test_10_a_typed_secret_is_refused_in_typed_words() -> None:
    from pilot.conversation.withheld import left_out, secret_not_used

    words = left_out("use the token ghp_fakefakefakefakefake0000 to list my repos")
    sentence = secret_not_used(words, typed=True)
    assert "said aloud" not in sentence and "ghp_" not in sentence
    assert sentence.startswith("I left out the key you typed")
    assert screen_faults(sentence) == []


# ── 11 and 13. The search folder and the log tail ───────────────────────────────────────────────────────────────


def test_11_a_content_search_takes_its_folder_under_either_name() -> None:
    from pilot.agents.planner import Planner

    params = Planner._normalize_params(ActionType.FILE_SEARCH_CONTENT, {"path": WORK, "query": "budget"}, "")
    assert params["directory"] == WORK and params["search_text"] == "budget"


def test_13_a_log_tail_is_a_read_of_its_last_lines() -> None:
    from pilot.agents.planner import Planner

    params = Planner._parse_parameters(ActionType.FILE_READ, {"path": WORK + r"\app.log", "tail_lines": 5})
    assert params.tail_lines == 5


# ── 12. A blank reply ───────────────────────────────────────────────────────────────────────────────────────────


def test_12_a_reply_is_never_empty() -> None:
    from pilot.task_reply import with_reply

    reply = with_reply({"status": "partial_failure", "message": "", "explanation": "", "results": []})
    assert reply["message"].strip() and reply["spoken"].strip()
    assert screen_faults(reply["message"]) == [] and ear_faults(reply["spoken"]) == []


# ── 14. The plan review decided the turn's time ─────────────────────────────────────────────────────────────────


def test_14_a_note_in_the_persons_folders_runs_beside_its_review(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    from pilot.agents.review_timing import ReviewTiming, review_timing

    note = tmp_path / "notes" / "note.txt"
    write = Action(action_type=ActionType.FILE_WRITE, target=str(note), parameters=FileParams(path=str(note)))
    undo = SimpleNamespace(enabled=True, max_task_mb=50)
    timing = review_timing(ActionPlan(actions=[write]), card_anyway=False, undo=undo, home=tmp_path)
    assert timing is ReviewTiming.BESIDE


# ── The run's requests through the execute handler, with a fake planner answer and fake handlers ────────────────


class _Outputs:
    """Each step returns the output, or fails with the error, given for its action type: the handler's own form."""

    def __init__(self, outputs: dict[ActionType, str], errors: dict[ActionType, str] | None = None) -> None:
        self.outputs, self.errors, self.runs = outputs, errors or {}, 0

    async def execute(self, plan, **kwargs):
        self.runs += 1
        return [
            ActionResult(
                action=action,
                success=action.action_type not in self.errors,
                output=self.outputs.get(action.action_type, ""),
                error=self.errors.get(action.action_type),
            )
            for action in plan.actions
        ]


@pytest.mark.asyncio
async def test_5_and_8_a_trigger_made_through_the_handler_is_said_not_dumped() -> None:
    from tests.test_reply_never_empty import _assert_said, _Planner, _server, _ws

    create = Action(
        action_type=ActionType.TRIGGER_CREATE,
        target="notes_watch",
        parameters=TriggerParams(
            name="notes_watch",
            trigger_type="file_modified",
            condition={"path": WORK + r"\notes.txt"},
            action_command="tell me",
        ),
    )
    entry = {
        "id": "cb220aa1",
        "name": "notes_watch",
        "trigger_type": "file_modified",
        "condition": {"path": WORK + r"\notes.txt"},
        "action_command": "tell me",
    }
    server = _server(_Planner(ActionPlan(actions=[create])), _Outputs({ActionType.TRIGGER_CREATE: json.dumps(entry)}))

    response = await server._handle_execute({"input": f"Tell me when {WORK}\notes.txt changes"}, _ws(server))

    assert response["status"] == "success"
    assert response["spoken"] == "I'll tell you when notes.txt changes."
    assert "cb220aa1" not in response["spoken"] and "{" not in response["message"]
    _assert_said(response)


@pytest.mark.asyncio
async def test_4_a_404_through_the_handler_is_a_failure_said_in_words() -> None:
    from tests.test_reply_never_empty import _assert_said, _Planner, _server, _ws

    url = "https://example.com/missing"
    get = Action(action_type=ActionType.API_REQUEST, target=url, parameters=ApiRequestParams(method="GET", url=url))
    error = (
        f"GET to {url} failed with HTTP 404 Not Found: the server has nothing at this address; the same request "
        "will get the same answer"
    )
    executor = _Outputs({}, {ActionType.API_REQUEST: error})
    server = _server(_Planner(ActionPlan(actions=[get])), executor)

    response = await server._handle_execute({"input": f"Fetch {url} and tell me the title"}, _ws(server))

    assert response["status"] != "success"
    assert executor.runs == 1  # the same request gets the same answer: not asked again
    _assert_said(response)


@pytest.mark.asyncio
async def test_10_a_typed_secret_never_reaches_the_planner_or_the_reply() -> None:
    from tests.test_reply_never_empty import _Planner, _server, _ws

    class _Seen(_Planner):
        def __init__(self, *plans: ActionPlan) -> None:
            super().__init__(*plans)
            self.inputs: list[str] = []

        async def plan(self, user_input, *args, **kwargs):
            self.inputs.append(str(user_input))
            return await super().plan(user_input, *args, **kwargs)

    planner = _Seen(ActionPlan(actions=[], explanation="I need the token to list your repositories."))
    executor = _Outputs({})
    server = _server(planner, executor)

    response = await server._handle_execute(
        {"input": "use the token ghp_fakefakefakefakefake0000 to list my repos"}, _ws(server)
    )

    assert executor.runs == 0
    assert all("ghp_" not in seen for seen in planner.inputs)
    assert "ghp_" not in json.dumps(response, default=str)
    assert "said aloud" not in response["message"] and response["message"].strip()


# ── The second live run (27 November), R1 to R9: the request through the router, the planner and the handlers ──
#
# The planner here is the real one, with a scripted model answering in place of a provider, so the plan checks of
# round twelve (a value kept in its step, a repair that may not escalate, a pick decided in code) run as they do in
# the daemon. The handlers are fakes that answer in each handler's own shape.

W2 = "C:\\w"
_PROCESSES = (
    "PID=  4321  CPU= 12.5%  MEM=  3.1%  STATUS=running       chrome.exe\n"
    "PID=   812  CPU=  0.4%  MEM=  0.9%  STATUS=running       explorer.exe\n"
    "PID=  9001  CPU=  1.1%  MEM=  1.2%  STATUS=sleeping      Code.exe"
)


class _ScriptedModel:
    """Answers each planner call with the next plan; the last one again when they run out."""

    def __init__(self, *answers: dict) -> None:
        self.answers = [json.dumps(answer) for answer in answers]
        self.sent: list[list[dict]] = []

    async def generate(self, messages, **_kwargs) -> str:
        self.sent.append([dict(m) for m in messages])
        return self.answers[min(len(self.sent), len(self.answers)) - 1]


class _Handlers:
    """Each step answers with the output given for its type, or fails with the next error queued for it."""

    def __init__(self, outputs: dict | None = None, errors: dict | None = None) -> None:
        self.outputs = outputs or {}
        self.errors = {kind: list(queue) for kind, queue in (errors or {}).items()}
        self.ran: list[Action] = []

    async def execute(self, plan, **kwargs):
        results = []
        for action in plan.actions:
            self.ran.append(action)
            queue = self.errors.get(action.action_type) or []
            error = queue.pop(0) if queue else None
            output = "" if error else self.outputs.get(action.action_type, "done")
            results.append(ActionResult(action=action, success=error is None, output=output, error=error))
        return results


class _Checks:
    """The check after the run: a failed step fails, and so does the first run of a type in ``mismatch``."""

    def __init__(self, mismatch: tuple = ()) -> None:
        self.mismatch, self.rounds = set(mismatch), 0

    async def verify(self, plan, results):
        from pilot.actions import VerificationResult

        failed, details = [], []
        for index, result in enumerate(results):
            bad = not result.success or (result.action.action_type in self.mismatch and self.rounds == 0)
            reason = result.error if not result.success else "the copy's content differs from the source"
            details.append(
                f"Action {index} ({result.action.action_type.value}): " + (f"FAILED - {reason}" if bad else "VERIFIED")
            )
            if bad:
                failed.append(index)
        self.rounds += 1
        return VerificationResult(
            passed=not failed,
            failed_actions=failed,
            details=details,
            independently_observed_actions=[i for i in range(len(results)) if i not in failed],
        )


def _step(kind: str, target: str, **params) -> dict:
    return {"action_type": kind, "target": target, "parameters": params}


async def _drive(request: str, *answers: dict, handlers: _Handlers, confirm: bool = False, mismatch: tuple = ()):
    """The request through _handle_execute with the real planner on a scripted model: (response, model, cards)."""
    from pilot.agents.planner import Planner
    from tests.test_reply_never_empty import _approvals, _server, _ws

    class _NoMemory:
        async def get_context(self, *_a, **_k) -> str:
            return ""

        async def get_history(self, *_a, **_k) -> list:
            return []

    model = _ScriptedModel(*answers)
    server = _server(Planner(model, _NoMemory()), handlers, confirm=confirm)  # type: ignore[arg-type]
    server._verifier = _Checks(mismatch)
    ws = _ws(server, confirmation=True if confirm else None)
    response = await server._handle_execute({"input": request}, ws)
    return response, model, _approvals(ws)


def _first_sentence(text: str) -> str:
    import re

    return re.split(r"(?<=[.!?])\s", str(text or "").strip(), maxsplit=1)[0]


@pytest.mark.asyncio
async def test_r1_what_is_running_is_a_process_list_said_by_its_programs() -> None:
    from pilot.conversation.router import TurnKind, route

    request = "What programs are running right now?"
    assert route(request).kind is not TurnKind.CONVERSATION
    handlers = _Handlers({ActionType.PROCESS_LIST: _PROCESSES})
    response, model, _cards = await _drive(
        request, {"actions": [_step("process_list", "processes")]}, handlers=handlers
    )
    assert [a.action_type for a in handlers.ran] == [ActionType.PROCESS_LIST]
    assert "process_list" in json.dumps(model.sent[0])  # the prompt's reading section names it
    assert response["spoken"] == "3 programs are running; the biggest are Chrome, Code and Explorer."
    assert "Operating System" not in response["message"]


@pytest.mark.asyncio
async def test_r2_a_temperature_no_action_reads_is_said_honestly_and_nothing_runs() -> None:
    sentence = "I can't read the processor's temperature on this computer."
    handlers = _Handlers()
    response, model, _cards = await _drive(
        "What is the CPU temperature?", {"actions": [], "explanation": sentence}, handlers=handlers
    )
    assert handlers.ran == []
    assert sentence in json.dumps(model.sent[0])  # the prompt holds the honest sentence for what nothing reads
    assert response["message"] == sentence and response["spoken"] == sentence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_words", "kind", "output", "spoken"),
    [
        (
            "What is the volume level? Just tell me, don't change it.",
            "volume_get",
            "Volume is at 40.",
            "Volume is at 40.",
        ),
        ("How bright is the screen?", "brightness_get", "Brightness: 70%", "Brightness is at seventy percent."),
    ],
)
async def test_r3_volume_and_brightness_are_read_not_answered_by_talk(request_words, kind, output, spoken) -> None:
    from pilot.conversation.router import TurnKind, route

    assert route(request_words).kind is not TurnKind.CONVERSATION
    handlers = _Handlers({ActionType(kind): output})
    response, _model, _cards = await _drive(request_words, {"actions": [_step(kind, kind)]}, handlers=handlers)
    assert [a.action_type.value for a in handlers.ran] == [kind]
    assert response["spoken"] == spoken


@pytest.mark.asyncio
async def test_r5_a_plan_that_drops_the_selector_is_planned_again_and_the_selector_is_used() -> None:
    request = "Scrape https://example.com and extract the text of the elements matching '.price-table td.amount'"
    dropped = {"actions": [_step("api_scrape", "https://example.com", url="https://example.com", selector="body")]}
    kept = {
        "actions": [
            _step("api_scrape", "https://example.com", url="https://example.com", selector=".price-table td.amount")
        ]
    }
    handlers = _Handlers({ActionType.API_SCRAPE: json.dumps({"matches": ["$12.00", "$15.00"], "count": 2})})
    response, model, _cards = await _drive(request, dropped, kept, handlers=handlers)
    assert len(model.sent) == 2  # the first plan was refused in code and planned again once
    assert [a.parameters.selector for a in handlers.ran] == [".price-table td.amount"]
    assert response["status"] == "success"
    assert "body" not in response["spoken"]


@pytest.mark.asyncio
async def test_r6_an_append_planned_as_a_read_and_a_write_runs_as_one_step_and_is_checked() -> None:
    note = W2 + "\\notes\\note.txt"
    pair = {
        "actions": [
            _step("file_read", note, path=note),
            {**_step("file_write", note, path=note, content="{PREV_OUTPUT}\n## Done"), "use_previous_output": True},
        ]
    }
    appended = f"Appended 1 line to {note} (appended sha256 {'ab' * 32}, 8 bytes; file sha256 {'cd' * 32}, 40 bytes)"
    handlers = _Handlers({ActionType.FILE_WRITE: appended})
    response, _model, _cards = await _drive(f"Append the line '## Done' to {note}", pair, handlers=handlers)
    assert [(a.action_type, a.parameters.append, a.parameters.content) for a in handlers.ran] == [
        (ActionType.FILE_WRITE, True, "## Done")
    ]
    assert response["spoken"] == "Added a line to note.txt in notes."
    assert "could not be checked" not in response["message"]


@pytest.mark.asyncio
async def test_r6_both_of_them_after_the_which_question_deletes_the_two_listed(tmp_path, monkeypatch) -> None:
    from pilot.system import triggers

    engine = triggers.TriggerEngine()
    engine.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", engine)
    watched = str(tmp_path / "watched.txt")
    made = [
        engine.create_trigger("watched-txt-log", "file_modified", {"path": watched}, "add a line to fired.log"),
        engine.create_trigger("watched-txt-change", "file_modified", {"path": watched}, "tell me"),
    ]
    with pytest.raises(triggers.TriggerNotFound):
        await triggers.trigger_delete("nonexistent-trigger-42")  # the run's name that matched nothing

    class _TriggerHandlers(_Handlers):
        async def execute(self, plan, **kwargs):
            results = []
            for action in plan.actions:
                self.ran.append(action)
                output = await triggers.trigger_delete(action.parameters.trigger_id)
                results.append(ActionResult(action=action, success=True, output=output))
            return results

    handlers = _TriggerHandlers()
    response, model, _cards = await _drive("Delete both of them", {"actions": []}, handlers=handlers)
    assert model.sent == []  # decided in code: no model is asked which ones
    assert [a.parameters.trigger_id for a in handlers.ran] == [t.id for t in made]
    assert engine.list_triggers() == []
    assert response["spoken"] == "I've removed 2 triggers; both watched watched.txt."


def test_r7_a_tail_a_scraped_title_and_a_delete_of_something_gone_are_said_as_what_they_are() -> None:
    from pilot.result_sentences import step_said

    log = W2 + "\\logs\\app.log"
    tail = Action(action_type=ActionType.FILE_READ, target=log, parameters=FileParams(path=log, tail_lines=5))
    lines = "\n".join(f"line {n}" for n in range(196, 201))
    said = step_said(ActionResult(action=tail, success=True, output=lines), request="show the last 5 lines of app.log")
    assert said.spoken.startswith("The last 5 lines of app.log")
    assert "has 5 lines" not in said.spoken
    title = Action(
        action_type=ActionType.API_SCRAPE,
        target="https://example.com",
        parameters=ApiRequestParams(url="https://example.com", selector="title"),
    )
    said = step_said(ActionResult(action=title, success=True, output="Example Domain"))
    assert "Example Domain" in said.spoken and "Found 1 match" not in said.spoken
    gone = W2 + "\\archive\\gone.txt"
    delete = Action(action_type=ActionType.FILE_DELETE, target=gone, parameters=FileParams(path=gone))
    absent = ActionResult(
        action=delete,
        success=True,
        output=f"There was no file at {gone}, so nothing was deleted",
        skip_reason="already_absent",
        executed=False,
    )
    said = step_said(absent)
    assert said.spoken == "There was no gone.txt in archive, so nothing was deleted."
    assert W2 not in said.shown


def test_r9_one_mismatched_copy_reported_by_several_paths_is_counted_once() -> None:
    from pilot.security.outcome_evidence import Outcome, OutcomeEvidence

    copy = Action(
        action_type=ActionType.FILE_COPY,
        target=W2 + "\\big\\a.bin",
        parameters=FileParams(path=W2 + "\\a.bin", destination=W2 + "\\big\\a.bin"),
    )
    evidence = OutcomeEvidence(clock=lambda: 1_000_000.0)
    for _ in range(3):  # the check, a late check and a repair's re-run of the same step in one task
        evidence.record(copy, Outcome.MISMATCH, task="t1")
    reason = evidence.assess(copy, task="t2").reason.casefold()
    assert "three" not in reason and "six" not in reason


# ── Item c of round twelve: what happens after a failure, for each kind the rules treat apart ──
#
# Final failures end the run with no repair round; a repair may not switch to a shell command; a repair of a missing
# file the planner guessed may try the right one. No case shows two approval cards or runs a shell step.

_SHELL_REPAIR = {
    "explanation": "Work around it.",
    "actions": [_step("shell_command", "cmd", command="cmd", args=["/c", "echo"])],
}
_PUSH = _step("git_push", W2 + "\\repo", repo_path=W2 + "\\repo", remote="origin", branch="main")
_AFTER_A_FAILURE = [
    (
        "a rejected push",
        f"Push the main branch of the git repository at {W2}\\repo to origin",
        _PUSH,
        {ActionType.GIT_PUSH: ["! [rejected]        main -> main (fetch first)\nerror: failed to push some refs"]},
        True,
        (),
        1,
        "I couldn't push main: origin has commits you don't have yet.",
    ),
    (
        "a 404",
        "Fetch https://example.com/missing and tell me the title",
        _step("api_request", "https://example.com/missing", url="https://example.com/missing", method="GET"),
        {ActionType.API_REQUEST: ["GET to https://example.com failed with HTTP 404 Not Found: <!doctype html><p>x"]},
        False,
        (),
        1,
        "Example.com has no page at that address.",
    ),
    (
        "a 503",
        "Fetch https://example.com/status and tell me what it says",
        _step("api_request", "https://example.com/status", url="https://example.com/status", method="GET"),
        {ActionType.API_REQUEST: ["GET to https://example.com failed with HTTP 503 Service Unavailable: busy"] * 4},
        False,
        (),
        2,
        "Example.com is overloaded or having trouble right now.",
    ),
    (
        "a copy whose check found a mismatch",
        f"Copy {W2}\\a.txt to {W2}\\b.txt",
        _step("file_copy", W2 + "\\a.txt", path=W2 + "\\a.txt", destination=W2 + "\\b.txt"),
        {},
        False,
        (ActionType.FILE_COPY,),
        2,
        "That didn't work for b.txt: the check afterwards found something else.",
    ),
    (
        "a write into a read-only folder",
        f"Write hello to {W2}\\readonly\\x.txt",
        _step("file_write", W2 + "\\readonly\\x.txt", path=W2 + "\\readonly\\x.txt", content="hello"),
        {ActionType.FILE_WRITE: [f"[Errno 13] Permission denied: '{W2}\\readonly\\x.txt' (read-only file system)"]},
        False,
        (),
        1,
        "I couldn't write x.txt in readonly: that place is read-only.",
    ),
    (
        "a scrape whose selector matches nothing",
        "Scrape https://example.com and extract the text of the elements matching '.price-table td.amount'",
        _step("api_scrape", "https://example.com", url="https://example.com", selector=".price-table td.amount"),
        {ActionType.API_SCRAPE: ["No element matches the selector '.price-table td.amount' on https://example.com"]},
        False,
        (),
        1,
        "Nothing on example.com matches .price-table td.amount.",
    ),
    (
        "a missing source the request named",
        f"Copy {W2}\\notes.txt to {W2}\\backup\\notes.txt",
        _step("file_copy", W2 + "\\notes.txt", path=W2 + "\\notes.txt", destination=W2 + "\\backup\\notes.txt"),
        {ActionType.FILE_COPY: [f"[Errno 2] No such file or directory: '{W2}\\notes.txt'"]},
        False,
        (),
        1,
        "I couldn't find notes.txt in w.",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _AFTER_A_FAILURE, ids=[case[0] for case in _AFTER_A_FAILURE])
async def test_what_the_person_sees_after_each_kind_of_failure(case) -> None:
    _label, request, first, errors, confirm, mismatch, model_calls, lead = case
    handlers = _Handlers(errors=errors)
    response, model, cards = await _drive(
        request, {"actions": [first]}, _SHELL_REPAIR, handlers=handlers, confirm=confirm, mismatch=mismatch
    )
    assert response["status"] == "partial_failure"
    assert len(model.sent) == model_calls  # a final failure asks the model nothing more
    assert [a.action_type for a in handlers.ran] == [ActionType(first["action_type"])]  # never the shell repair
    assert cards <= 1
    assert _first_sentence(response["spoken"]) == lead
    assert ear_faults(response["spoken"]) == [] and "<" not in response["message"]
    if model_calls > 1:  # the repair round was planned and refused in code
        assert "I did not try again: the retry would run a shell command" in response["spoken"]


@pytest.mark.asyncio
async def test_a_missing_source_the_planner_guessed_may_be_repaired_with_the_right_file() -> None:
    guessed = _step("file_copy", W2 + "\\notes.txt", path=W2 + "\\notes.txt", destination=W2 + "\\backup\\notes.txt")
    right = _step("file_copy", W2 + "\\notes.md", path=W2 + "\\notes.md", destination=W2 + "\\backup\\notes.md")
    handlers = _Handlers(errors={ActionType.FILE_COPY: [f"[Errno 2] No such file or directory: '{W2}\\notes.txt'"]})
    response, model, cards = await _drive(
        f"Copy my notes file in {W2} to {W2}\\backup", {"actions": [guessed]}, {"actions": [right]}, handlers=handlers
    )
    assert len(model.sent) == 2 and cards == 0
    assert [a.parameters.path for a in handlers.ran] == [W2 + "\\notes.txt", W2 + "\\notes.md"]
    assert response["status"] == "success" and response["spoken"] == "Copied notes.md into backup."


# ── Item d of round twelve: a read question answered in the first sentence, from each system's own output ──

_WHOAMI = "USER INFORMATION\n----------------\n\nUser Name    SID\n============ =============\ndesk-1\\me    S-1-5-21-1-2-3-1001"
_NET_USER = (
    "User accounts for \\\\DESK-1\n\n-------------------------------------------------------------------------------\n"
    "Administrator            DefaultAccount           Guest\nme                       WDAGUtilityAccount\n"
    "The command completed successfully."
)
_LSBLK = "NAME   SIZE TYPE MOUNTPOINT FSTYPE\nsda    477G disk\n├─sda1 512M part /boot/efi vfat\nsdb   28.9G disk"
_GET_DISK = (
    "Number Friendly Name   Serial Number HealthStatus OperationalStatus Total Size Partition Style\n"
    "------ -------------   ------------- ------------ ----------------- ---------- ---------------\n"
    "0      NVMe SSD 1TB    0000_0001.    Healthy      Online             1000.00 GB GPT"
)


@pytest.mark.parametrize(
    ("kind", "params", "output", "request_words", "spoken"),
    [
        ("user_info", {}, "uid=1000(me) gid=1000(me) groups=1000(me),27(sudo)", "which user am I", "You're logged in as me."),
        ("user_info", {}, _WHOAMI, "which user am I", "You're logged in as me."),
        ("user_list", {}, "me\nguest", "which accounts", "This computer has 2 user accounts: me and guest."),
        ("user_list", {}, _NET_USER, "which accounts", "This computer has 5 user accounts: Administrator, DefaultAccount, Guest and 2 more."),
        ("disk_list", {}, _LSBLK, "which drives", "This computer has 2 disks."),
        ("disk_list", {}, _GET_DISK, "which drives", "This computer has 1 disk."),
        ("env_get", {"name": "PATH"}, "PATH=/usr/local/bin:/usr/bin:/bin", "what is in my PATH", "PATH holds 3 folders; the list is on screen."),
        ("network_info", {}, "=== Network ===\n  Wi-Fi (UP):\n    IPv4: 192.168.1.5\n  Ethernet (DOWN):", "what is my IP address", "Your IP address is 192.168.1.5, on Wi-Fi."),
        ("system_info", {"categories": ["os"]}, "=== Operating System ===\n  system: Windows\n  release: 11", "what operating system am I running", "This computer runs Windows 11."),
    ],
)  # fmt: skip
def test_d_a_read_question_is_answered_in_its_first_sentence(kind, params, output, request_words, spoken) -> None:
    from pilot.actions import VerificationResult
    from pilot.response_contract import success_reply

    action = Action.model_validate({"action_type": kind, "target": kind, "parameters": params})
    result = ActionResult(action=action, success=True, output=output)
    said = success_reply(
        ActionPlan(actions=[action], raw_input=request_words), [result], VerificationResult(passed=True), dry_run=False
    )
    assert _first_sentence(said.spoken) == spoken
    assert screen_faults(said.shown) == []


def test_b_a_quoted_value_with_a_double_quote_in_it_is_found_in_its_step() -> None:
    from pilot.agents.request_match import check_plan

    note = "C:/w/note.txt"
    append = Action(
        action_type=ActionType.FILE_WRITE,
        target=note,
        parameters=FileParams(path=note, content='see a/b "now"', append=True),
    )
    assert check_plan("Append the line 'see a/b \"now\"' to C:/w/note.txt", [append]) is None
    assert check_plan("Append the line 'see a/b \"later\"' to C:/w/note.txt", [append]).rule == "value_dropped"


@pytest.mark.parametrize(
    "sentence",
    [
        "I couldn't find .env in work. Check the name, or tell me where it is.",
        "Nothing on example.com matches .price-table td.amount. The page loaded; what it does hold is on screen.",
        "On example.com, main .title reads: Example Domain.",
        "Found 2 matches for .txt in work: note.txt and old.txt.",
        "The last 5 lines of app.log; the newest is: done.",
        "It's 9:40 right now.",
    ],
)
def test_voice_a_name_or_selector_with_a_leading_dot_is_not_a_hole(sentence: str) -> None:
    from pilot.reply_rules import hole

    assert hole(sentence) == ""


@pytest.mark.parametrize("sentence", ["And it's now, so a morning session may be tight.", "Your report is due ."])
def test_voice_a_real_hole_is_still_found(sentence: str) -> None:
    from pilot.reply_rules import hole

    assert hole(sentence)


# ── The third live run (round thirteen): its new faults through the router, the planner and the handlers ──────

_LOCAL = "http://127.0.0.1:8797/busy"
_UPTIME = (
    "=== System Time ===\n  Current Local Time: Tuesday, March 3, 2026 4:05:09 PM\n  Timezone: Newfoundland Standard Time\n"
    "  Up for: 2 day(s), 3 hour(s), 0 minute(s)"
)
_OS = "=== Operating System ===\n  system: Windows\n  release: 11\n  hostname: studio-pc\n  machine: AMD64"


async def _drive_gateway(source: str):
    """A GET to this computer through the server with the real gateway: (response, executor runs, cards)."""
    from unittest.mock import MagicMock

    from pilot.security.gateway import AgentGateway
    from tests.test_reply_never_empty import _approvals, _Executor, _Planner, _server, _ws

    get = Action(
        action_type=ActionType.API_REQUEST,
        target=_LOCAL,
        parameters=ApiRequestParams(method="GET", url=_LOCAL, allow_private_network=True),
    )
    executor = _Executor()
    server = _server(_Planner(ActionPlan(actions=[get], explanation="GET it.")), executor, confirm=True)
    server._agent_gateway = AgentGateway(server.config, MagicMock())
    ws = _ws(server, confirmation=True)
    params = {"input": f"Make an HTTP GET request to {_LOCAL}", **({"source": source} if source else {})}
    response = await server._handle_execute(params, ws)
    return response, executor.runs, _approvals(ws)


@pytest.mark.asyncio
async def test_r13_a_spoken_local_get_is_refused_with_no_card_in_one_sentence() -> None:
    response, runs, cards = await _drive_gateway("voice")
    assert (runs, cards) == (0, 0)
    assert response["denied_kind"] == "gateway_denied"
    sentence = "I didn't call 127.0.0.1: my safety rules don't allow that from here, so nothing was sent. Asking again won't change it."
    assert response["spoken"] == sentence
    assert response["message"].startswith(sentence)
    assert "integration_agent" not in response["message"] and "tier=" not in response["message"]


@pytest.mark.asyncio
async def test_r13_the_same_get_typed_asks_once_and_runs() -> None:
    response, runs, cards = await _drive_gateway("")
    assert (runs, cards) == (1, 1)
    assert response["status"] == "success"
    assert "safety rules" not in response["spoken"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_words", "categories", "output", "spoken"),
    [
        ("How long has this computer been running?", ["time"], _UPTIME, "This computer has been up for 2 days and 3 hours."),
        ("What is this computer's host name?", ["os"], _OS, "This computer is called studio-pc."),
    ],
)  # fmt: skip
async def test_r13_uptime_and_the_host_name_are_read_and_answered(request_words, categories, output, spoken) -> None:
    from pilot.conversation.router import TurnKind, route

    assert route(request_words).kind is not TurnKind.CONVERSATION
    handlers = _Handlers({ActionType.SYSTEM_INFO: output})
    response, model, _cards = await _drive(
        request_words, {"actions": [_step("system_info", "system", categories=categories)]}, handlers=handlers
    )
    assert "system_info" in json.dumps(model.sent[0])
    assert response["spoken"] == spoken
    assert response["message"].splitlines()[0] == spoken


@pytest.mark.asyncio
async def test_r13_the_ram_total_is_the_total() -> None:
    memory = "=== Memory ===\n  Total: 16.0 GB\n  Used:  12.0 GB (75.0%)\n  Free:  4.0 GB"
    handlers = _Handlers({ActionType.MEMORY_USAGE: memory})
    response, _model, _cards = await _drive(
        "How much RAM does this computer have?", {"actions": [_step("memory_usage", "memory")]}, handlers=handlers
    )
    assert response["spoken"] == "This computer has 16.0 gigabytes of memory; 12.0 are in use."


@pytest.mark.asyncio
async def test_r13_a_tell_me_trigger_is_a_notice_made_said_and_fired_by_the_file(tmp_path, monkeypatch) -> None:
    from pilot.result_sentences import trigger_fire_sentence
    from pilot.system import triggers

    engine = triggers.TriggerEngine()
    engine.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", engine)
    watched = str(tmp_path / "watched.txt")

    class _MakeTriggers(_Handlers):
        async def execute(self, plan, **kwargs):
            results = []
            for action in plan.actions:
                self.ran.append(action)
                p = action.parameters
                output = await triggers.trigger_create(
                    p.name, p.trigger_type, p.condition, p.action_command, notify_only=p.notify_only
                )
                results.append(ActionResult(action=action, success=True, output=output))
            return results

    answer = {
        "actions": [
            _step(
                "trigger_create",
                "watched_notice",
                name="watched_notice",
                trigger_type="file_modified",
                condition={"path": watched},
                action_command="Tell me when it changes",
            )
        ]
    }
    handlers = _MakeTriggers()
    response, _model, _cards = await _drive(f"Tell me when {watched} changes", answer, handlers=handlers)
    assert [a.parameters.notify_only for a in handlers.ran] == [True]  # made a notice whatever the model wrote
    assert response["spoken"] == "I'll tell you when watched.txt changes."
    [made] = engine.list_triggers()
    assert trigger_fire_sentence(made, "notice") == "watched.txt changed."


@pytest.mark.asyncio
async def test_r13_deleting_one_of_two_triggers_on_a_file_says_what_remains(tmp_path, monkeypatch) -> None:
    from pilot.system import triggers

    engine = triggers.TriggerEngine()
    engine.attach_store(tmp_path / "triggers.json")
    monkeypatch.setattr(triggers, "_engine", engine)
    watched = str(tmp_path / "watched.txt")
    notice = engine.create_trigger("watched_notice", "file_modified", {"path": watched}, "tell me", notify_only=True)
    engine.create_trigger("watched_log", "file_modified", {"path": watched}, "add a line to fired.log")

    class _DeleteTriggers(_Handlers):
        async def execute(self, plan, **kwargs):
            results = []
            for action in plan.actions:
                self.ran.append(action)
                output = await triggers.trigger_delete(action.parameters.trigger_id)
                results.append(ActionResult(action=action, success=True, output=output))
            return results

    answer = {"actions": [_step("trigger_delete", notice.id, trigger_id=notice.id)]}
    response, _model, _cards = await _drive(
        f"Delete the trigger with the id {notice.id}.", answer, handlers=_DeleteTriggers()
    )
    assert response["spoken"] == (
        "I've removed the trigger that tells you when watched.txt changes. The other one still watches it."
    )
    assert [t["name"] for t in engine.list_triggers()] == ["watched_log"]


@pytest.mark.asyncio
async def test_r13_a_refused_retry_is_said_after_the_failure() -> None:
    copy = _step("file_copy", W2 + "\\a.txt", path=W2 + "\\a.txt", destination=W2 + "\\b.txt")
    response, model, cards = await _drive(
        f"Copy {W2}\\a.txt to {W2}\\b.txt",
        {"actions": [copy]},
        _SHELL_REPAIR,
        handlers=_Handlers(),
        mismatch=(ActionType.FILE_COPY,),
    )
    assert len(model.sent) == 2 and cards == 0
    first, _, rest = response["spoken"].partition(". ")
    assert first == "That didn't work for b.txt: the check afterwards found something else"
    assert "I did not try again" in rest


@pytest.mark.parametrize(("reason", "speaking"), [("no sound", "alba"), ("no sound", ""), ("missing", "alba")])
def test_r13_a_voice_put_aside_is_told_in_the_first_person_and_passes_every_check(reason, speaking) -> None:
    from pilot.conversation import work_claims
    from pilot.reply_rules import hole
    from pilot.system.local_tts_worker import aside_sentence

    sentence = aside_sentence("michael", reason, speaking, 7_000_000)
    assert "Heliox" not in sentence and "I'll" in sentence
    assert hole(sentence) == "" and ear_faults(sentence) == [] and screen_faults(sentence) == []
    assert work_claims.unbacked(sentence, {"known": True, "queued": []}) == ""
