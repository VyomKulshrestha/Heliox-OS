"""A talk reply never claims work that does not exist (pilot.conversation.work_claims), and the talk model is told
the truth about what runs (pilot.conversation.adapters._task_state, context.describe_task).

The claims below are the talk model's own sentences from the live run of 26 November (NEW\\live10\\a10 and main), where
nothing ran; then other wordings; then ordinary talk that must never be taken for a claim. Fake models and sinks
only: no model, network or audio.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pilot.conversation import work_claims
from pilot.conversation.adapters import _ANSWERING, _task_state
from pilot.conversation.context import NOTHING_RUNNING, ContextSources, ConversationMemory, build_messages
from pilot.conversation.layer import NOTHING_RUNNING_ASK, NOTHING_RUNNING_SAID, ConversationLayer

NOTHING = {"known": True, "queued": []}

RECORDED_CLAIMS = [
    'Heliox is already working on posting "live10 check, please ignore" to Slack, and it\'s planning the safest way '
    "to do it now.",
    "It will show on screen once it's sent.",
    "Heliox is planning that post to Discord now, so it's on its way.",
    "The append to note.txt is already in progress, so I'll leave that running.",
    'Heliox is already appending "Also renew the parking permit." to that note, so I\'ll leave that step to the '
    "planner.",
    "Heliox is still working on the hash for the archive folder, and it's checking note_b, not note_backup.txt.",
    "Once the planner finishes, the result will appear on screen.",
    "Heliox is still on the hash, but it's working on the notes folder rather than note.txt, so the result may not "
    "match the file you asked about.",
    "Your earlier request to delete note_backup.txt is also still open.",
    "Heliox is already planning the git status check for the repository at ...\\live10\\work\\repo, and the result "
    "will show on screen when it finishes.",
    'Your earlier requests, the last five lines of app.log and the "quarterly budget" search, are still queued '
    "behind it.",
    "Heliox is already checking that now, and the result will appear on screen once it finishes.",
    "Heliox is still working on your last request, listing the programs that are running.",
    "I can't see the list from here, so it will appear on screen when it's ready.",
    "Heliox is planning a safe read-only check for it now, and the result will show on screen when it's ready.",
    "Heliox is planning a read-only check of the volume level now, and the figure will show on screen once it's done.",
    "I'll have Heliox check it with a read-only step, and the result will appear on screen once it's planned and run.",
    'Heliox is already setting up that watch on watched.txt, and it will append "seen a change" to fired.log on each '
    "modification.",
    "I'll show the trigger on screen once it's active.",
    "Heliox is already on that fetch of https://example.com, and the page title will show on screen once it's back.",
    "The commit, branch and push requests are queued after it.",
    "That scrape is already running: Heliox is planning the safest way to get the text from https://example.com.",
    "Heliox is still working on that now; the planner is choosing a safe way to read your disk space.",
    "The task on screen is still working on the 404 check you asked for.",
]

OTHER_CLAIMS = [
    "I'm on it.",
    "I'm checking your clipboard now.",
    "Heliox is fetching the page for you.",
    "Your post to Slack is queued.",
    "That's in progress.",
    "The hash is being worked on.",
    "The result will pop up on your screen shortly.",
    "I've sent it to Discord.",
    "I have appended the line to note.txt.",
    "That's been posted.",
    "Heliox has copied the file.",
    "Let me check the disk for you.",
    "I'll ask the planner to read your clipboard.",
    "I'm going to fetch that page now.",
    "Heliox will show it on the screen when it's done.",
    "Once it's done, it will show up here.",
]

ORDINARY_TALK = [
    "I'm doing well, thanks.",
    "That's interesting.",
    "It's written in Python.",
    "It's raining in Pune today.",
    "Canberra is the capital of Australia.",
    "You copied the file yesterday, so it should still be there.",
    "Good.",
    "I'm not sure which file you mean.",
    "I'm thinking about it.",
    "That's something to keep in mind.",
    "Git keeps every version of your files, so nothing is lost.",
    "Posting to Slack reaches the whole channel.",
    "Do you want me to check the disk?",
    "Should I post it to Slack?",
    "I can check that for you if you like.",
    "I can't read your clipboard from this chat.",
    "A hash is a short fingerprint of a file.",
    "The store is still open until nine.",
    "Your package should be there tomorrow.",
    "That was quick.",
    "It's working now, nice.",
    "I'm sorry, I didn't catch that.",
    "The meeting is at three.",
]


@pytest.mark.parametrize("sentence", RECORDED_CLAIMS)
def test_every_recorded_claim_is_caught_when_nothing_runs(sentence):
    assert work_claims.unbacked(sentence, NOTHING), sentence


@pytest.mark.parametrize("sentence", OTHER_CLAIMS)
def test_other_wordings_of_a_claim_are_caught(sentence):
    assert work_claims.unbacked(sentence, NOTHING), sentence


@pytest.mark.parametrize("sentence", ORDINARY_TALK)
def test_ordinary_talk_is_never_a_claim(sentence):
    assert work_claims.claim(sentence) == "", sentence


def test_a_claim_about_the_task_that_really_runs_is_said():
    copying = {"known": True, "running": "Copy C:\\work\\a.txt to C:\\work\\b.txt", "queued": []}
    assert not work_claims.unbacked("Heliox is still copying it.", copying)
    assert not work_claims.unbacked("I'm still on it.", copying)
    assert not work_claims.unbacked("The copy of a.txt will show on screen once it finishes.", copying)


def test_a_claim_about_another_task_than_the_one_running_is_not():
    status = {"known": True, "running": "What's the git status of C:\\work\\repo?", "queued": []}
    assert work_claims.unbacked("The append to note.txt is already in progress.", status)
    assert not work_claims.unbacked("The git status check is still running.", status)


def test_a_queued_request_backs_a_claim_that_it_waits():
    queued = {"known": True, "queued": ["Post the release notes to Slack"]}
    assert not work_claims.unbacked("Your Slack post is queued.", queued)
    assert work_claims.unbacked("Your Discord post is queued.", queued)


def test_a_finished_task_backs_a_claim_that_it_was_done():
    done = {"known": True, "queued": [], "last_result": "success: Copied 3 files to C:\\backup"}
    assert not work_claims.unbacked("I've copied them.", done)
    assert work_claims.unbacked("I've copied them.", NOTHING)


def test_typed_text_keeps_its_other_sentences():
    kept, dropped = work_claims.without_unbacked(
        "I can't read your clipboard from this chat. Heliox is planning a check now.", NOTHING
    )
    assert kept.strip() == "I can't read your clipboard from this chat." and len(dropped) == 1


def test_the_check_is_quick():
    import time

    started = time.perf_counter()
    for _ in range(50):
        for sentence in RECORDED_CLAIMS + ORDINARY_TALK:
            work_claims.unbacked(sentence, NOTHING)
    per = (time.perf_counter() - started) / (50 * (len(RECORDED_CLAIMS) + len(ORDINARY_TALK)))
    assert per < 0.005  # about 0.02 ms when idle; the bound only catches a pathological pattern


# ── the truth the talk model is told ──


class _Runtime:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def status(self):
        return self.snapshot


def _server(*, interaction_id="i-1", user_input="What is on my clipboard?", queued=()):
    return SimpleNamespace(
        _interaction_runtime=_Runtime(
            {"active": True, "interaction_id": interaction_id, "user_input": user_input, "message": "Planning"}
        ),
        _interactive_request_active=True,
        _pending_confirms={},
        _active_interaction_id=interaction_id,
        _queued_execute_requests=list(queued),
        _queued_reserved_task_id="",
    )


def test_the_turn_being_answered_is_never_reported_as_running():
    server = _server()
    token = _ANSWERING.set("i-1")
    try:
        state = _task_state(server, None)
    finally:
        _ANSWERING.reset(token)
    assert "running" not in state and state["known"] is True and state["queued"] == []


def test_another_request_that_runs_is_reported():
    state = _task_state(_server(interaction_id="other"), None)
    assert state["running"] == "What is on my clipboard?"


def test_queued_requests_are_reported():
    server = _server(queued=[("t1", "token", {"input": "Post the notes to Slack"})])
    token = _ANSWERING.set("i-1")
    try:
        state = _task_state(server, None)
    finally:
        _ANSWERING.reset(token)
    assert state["queued"] == ["Post the notes to Slack"]


def test_the_context_says_nothing_runs_when_nothing_does():
    memory = ConversationMemory()
    sources = ContextSources(task_state=lambda: dict(NOTHING))
    messages = build_messages("What do you think about git?", sources, memory)
    assert NOTHING_RUNNING in messages[-1]["content"]
    busy = ContextSources(task_state=lambda: {"known": True, "queued": ["Post the notes to Slack"]})
    content = build_messages("hi", busy, ConversationMemory())[-1]["content"]
    assert NOTHING_RUNNING not in content and "Waiting in Heliox's queue: Post the notes to Slack" in content


# ── the layer: a claim is never said, and the person's request is not lost ──


class _Model:
    local, streams, name, provider = False, True, "fake", "fake"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def stream(self, system, messages):
        self.calls += 1
        yield self.text


def _layer(text, state=None):
    said: list[str] = []

    async def sink(sentence, channel, key):
        said.append(sentence)
        return "spoken"

    model = _Model(text)
    layer = ConversationLayer(
        sink=sink, model=lambda: model, sources=ContextSources(task_state=lambda: dict(state or NOTHING))
    )
    return layer, said, model


@pytest.mark.parametrize("source", ["typed", "voice"])
async def test_a_reply_that_claims_work_is_not_said_and_a_question_gets_the_truth(source):
    layer, said, _ = _layer("Heliox is already checking that, and the result will show on screen once it finishes.")
    result = await layer.take("What happened to the weather this week?", source=source)
    assert result.done and result.intent is None
    assert said == [NOTHING_RUNNING_SAID]


@pytest.mark.parametrize("source", ["typed", "voice"])
async def test_a_request_the_reply_claimed_goes_to_the_planner(source):
    layer, said, _ = _layer("Heliox is planning that now; it will show on screen once it's done.")
    layer.memory.add_user("I wrote the release notes")
    layer.memory.add_heliox("Nice.")
    result = await layer.take("post them to slack", source=source)
    assert not result.done and result.planner_input == "post them to slack"
    assert not any("planning" in line.casefold() or "on screen" in line.casefold() for line in said), said


async def test_an_unclear_turn_with_a_claim_is_asked_about():
    layer, said, _ = _layer("I'm on it.")
    result = await layer.take("hmm, the clipboard thing", source="typed")
    assert result.done
    assert said in ([NOTHING_RUNNING_SAID], [NOTHING_RUNNING_ASK])


async def test_the_other_sentences_of_the_reply_are_kept():
    layer, said, _ = _layer("Canberra. Heliox is checking the atlas now.")
    result = await layer.take("What's the capital of Australia?", source="typed")
    assert result.done and "".join(said).strip() == "Canberra."


async def test_a_claim_about_the_running_task_is_said():
    running = {"known": True, "running": "Copy C:\\work\\a.txt to C:\\work\\b.txt", "queued": []}
    layer, said, _ = _layer("Heliox is still copying it.", running)
    result = await layer.take("how's it going?", source="typed")
    assert result.done and "".join(said).strip() == "Heliox is still copying it."


async def test_a_read_question_reaches_the_planner_without_a_model_call():
    layer, said, model = _layer("Heliox is checking your clipboard now.")
    result = await layer.take("What is on my clipboard? Just read it, don't change it.", source="typed")
    assert not result.done and model.calls == 0 and result.intent is not None and result.intent.verb == "check"
    assert result.quick is None  # the clipboard is not on the quick road; the planner reads it


async def test_a_read_naming_a_path_with_disk_in_it_is_not_a_quick_disk_reading():
    layer, _said, _model = _layer("")
    result = await layer.take("What is the SHA-256 hash of C:\\disk\\space\\notes.txt?", source="typed")
    assert not result.done and result.quick is None


async def test_a_disk_question_still_takes_the_quick_road():
    layer, _said, _model = _layer("")
    result = await layer.take("How much free disk space is there?", source="typed")
    assert not result.done and result.quick is not None
