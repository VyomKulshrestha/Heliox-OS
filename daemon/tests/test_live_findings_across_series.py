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
    said = _said(memory, "=== Memory ===\n  Total: 15.7 GB\n  Used:  9.9 GB (63.2%)\n  Free: 5.0 GB")
    assert said.spoken == "9.9 of 15.7 gigabytes of memory are in use, 63 percent."


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
    assert response["spoken"] == "I'll watch notes.txt and tell you when it changes."
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
