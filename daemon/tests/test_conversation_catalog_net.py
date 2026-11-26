"""The action catalog's net under the router's rules (pilot.conversation.catalog_words, router.catalog_net).

In the live run of 26 November these typed requests were routed as talk, and the talk model then said the work was
under way when no task existed. Each must reach the planner now; genuine conversation must stay talk; a word on its
own that names something Heliox could act on is asked about. Text only, no model.
"""

from __future__ import annotations

import re

import pytest

from pilot.actions import READ_ONLY_ACTIONS, ActionType
from pilot.conversation import catalog_words
from pilot.conversation.router import CATALOG_IMPERATIVE_REASON, READ_REASON, TurnKind, normalize, route

WORK = "C:\\Users\\someone\\AppData\\Local\\Temp\\run\\work"

# The run's typed wordings that ended as talk (NEW\live10\a10\cases), with what they ask for.
RUN_REQUESTS = [
    ("Post 'live10 check, please ignore' to Slack", "post"),
    ("Post 'live10 check, please ignore' to Discord", "post"),
    (f"Append the line 'Also renew the parking permit.' to {WORK}\\notes\\note.txt", "append"),
    (f"What is the SHA-256 hash of {WORK}\\archive\\note_backup.txt?", "check"),
    (f"Compute the SHA-256 hash of the file {WORK}\\notes\\note.txt", "compute"),
    (f"What's the git status of the repository at {WORK}\\repo?", "check"),
    (f"Stage README.md in the git repository at {WORK}\\repo", "stage"),
    ("How much free disk space is there?", "check"),
    ("What programs are running right now?", "check"),
    ("What is on my clipboard? Just read it, don't change it.", "check"),
    ("What is the volume level? Just tell me, don't change it.", "check"),
    ("What is the display brightness? Just tell me, don't change it.", "check"),
    (
        f"Watch the file {WORK}\\trigger\\watched.txt and every time it is modified, append the line 'seen a change' "
        f"to {WORK}\\trigger\\fired.log",
        "watch",
    ),
    ("Fetch https://example.com with an HTTP GET request and tell me the page title", "fetch"),
    (
        "Do an HTTP GET request to https://example.com/this-page-does-not-exist-404 and tell me the status code",
        "check",
    ),
    (
        "Scrape https://example.com and extract the text of every element matching the CSS selector "
        "'.price-table td.amount'",
        "scrape",
    ),
    ("Get the text of the elements matching the CSS selector '.price-table td.amount' from https://example.com", "get"),
]

# Other wordings of the same requests.
PARAPHRASES = [
    "Can you post 'deploy done' to Slack?",
    "Could you append 'call mum' to C:\\notes\\todo.txt",
    "Would you post the release notes to Discord",
    "tell me the volume level",
    "give me the git status of ~/projects/site",
    "I'd like to know how much disk space is left",
    "what's my battery at?",
    "clipboard contents please",
    "disk space?",
    "how full is my drive right now",
    "What's the md5 checksum of report.pdf?",
    "is my wifi connected right now?",
    "Monitor the folder /home/sam/inbox and tell me when a file appears",
    "hash C:\\data\\backup.zip",
    "What's using my memory right now?",
]

# Genuine conversation that names the same things: it stays talk.
STAYS_TALK = [
    "I copied the file yesterday.",
    "What do you think about git?",
    "Should I post this to Slack?",
    "Should we post the update to Discord?",
    "What is a clipboard?",
    "What is git?",
    "How does disk space get used up so fast?",
    "I posted the update to Discord this morning.",
    "Is it worth hashing every file I download?",
    "Why is my laptop so slow lately?",
    "Tell me about Slack.",
    "Do you know what SHA-256 is?",
    "My battery died during the meeting yesterday.",
    "The volume on that video was way too loud.",
    "How do I check my disk space?",
    "Which is better, Slack or Discord?",
    "Can I use Discord without an account?",
    "What does git stash do?",
    "I'm thinking of fetching groceries later.",
    "Watch out, the build is flaky today.",
    "Who invented the clipboard?",
    "Remember when the disk filled up last week?",
    "I think my memory is getting worse.",
    "What's the capital of Australia?",
    "What time is it?",
    "Get some rest, you've earned it",
    "stop worrying about it",
]

# A word on its own that names something Heliox could act on: Heliox asks, it never guesses.
ASKED_ABOUT = ["Slack.", "git", "the webhook", "post it", "hash that", "watch it"]


@pytest.mark.parametrize(("text", "verb"), RUN_REQUESTS)
def test_the_run_requests_reach_the_planner(text, verb):
    decided = route(text)
    assert decided.kind is TurnKind.ACTION, decided
    assert decided.verb == verb, decided


@pytest.mark.parametrize("text", PARAPHRASES)
def test_other_wordings_reach_the_planner(text):
    assert route(text).kind is TurnKind.ACTION, route(text)


@pytest.mark.parametrize("text", STAYS_TALK)
def test_genuine_talk_stays_talk(text):
    assert route(text).kind is TurnKind.CONVERSATION, route(text)


@pytest.mark.parametrize("text", ASKED_ABOUT)
def test_a_bare_name_is_asked_about(text):
    assert route(text).kind is TurnKind.UNCLEAR, route(text)


def test_a_pronoun_after_a_catalog_verb_is_resolved_by_the_conversation():
    # As for the router's own verbs ("send that to Priya"): with earlier turns, the talk model resolves "it" and
    # writes the ACT line; with none, "post it" names nothing and Heliox asks.
    assert route("post it to slack", conversation_open=True).kind is TurnKind.CONVERSATION
    assert route("post it", conversation_open=True).kind is TurnKind.CONVERSATION
    assert route("post it", conversation_open=False).kind is TurnKind.UNCLEAR


def test_the_reasons_say_which_net_caught_it():
    assert route("What is on my clipboard?").reason == READ_REASON
    assert route("Append a line to C:\\a.txt").reason == CATALOG_IMPERATIVE_REASON


# ── the contract: the net is built from the catalog, and a new action cannot be forgotten ──


def test_every_action_type_is_named_or_says_why_not():
    named, unnamed = set(catalog_words.NAMED), set(catalog_words.UNNAMED)
    assert not named & unnamed, sorted(a.value for a in named & unnamed)
    missing = set(ActionType) - named - unnamed
    assert not missing, f"decide how a person asks for: {sorted(a.value for a in missing)}"
    assert all(reason.strip() for reason in catalog_words.UNNAMED.values())


def test_every_named_action_has_words():
    empty = [
        kind.value for kind, words in catalog_words.NAMED.items() if not (words.subjects or words.topics or words.verbs)
    ]
    assert not empty, empty


def test_every_named_read_action_has_a_question_that_reaches_the_planner():
    for kind, words in catalog_words.NAMED.items():
        if kind not in READ_ONLY_ACTIONS or not (words.subjects or words.topics):
            continue
        assert words.asked, f"{kind.value}: give a question a person would ask"
        decided = route(words.asked)
        assert decided.kind is TurnKind.ACTION, (kind.value, words.asked, decided)


def test_every_subject_is_caught_in_a_question_about_this_computer():
    for kind, words in catalog_words.NAMED.items():
        if kind not in READ_ONLY_ACTIONS:
            continue
        for subject in words.subjects:
            plain = re.sub(r"\(\?:([^|)]+)[^)]*\)\??", r"\1", subject).replace("?", "").replace("\\w+", "five")
            question = f"What's the {plain} right now?"
            assert route(question).kind is TurnKind.ACTION, (kind.value, question, route(question))


def test_the_patterns_compile_and_read_normalized_words():
    for words in catalog_words.NAMED.values():
        for fragment in (*words.subjects, *words.topics):
            re.compile(fragment)
            assert fragment == fragment.casefold(), fragment
    assert catalog_words.STRONG.search(normalize("What's on my CLIPBOARD?"))


def test_a_path_holding_disk_is_not_read_as_a_disk_question_by_the_quick_road():
    decided = route("What is the SHA-256 hash of C:\\disk\\space\\notes.txt?")
    assert decided.kind is TurnKind.ACTION and decided.target == "sha 256"
