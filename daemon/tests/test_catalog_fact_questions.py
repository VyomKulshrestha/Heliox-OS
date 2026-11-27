"""Questions about one fact inside a composite read reach a read, never talk.

Third live run: "What is this computer's name?" was answered by the talk model ("I can't see the computer's name
from here") although system_info returns the host name. A composite read (system_info, network_info, user_info,
disk_usage) answers many small questions; each fact it holds is asked about here in the words people use, and each
question must become a task (the quick road or the planner). Text only, no model."""

from __future__ import annotations

import pytest

from pilot.actions import ActionType
from pilot.agents.read_catalog import READS
from pilot.conversation.router import TurnKind, route

A = ActionType

FACTS: dict[ActionType, dict[str, tuple[str, ...]]] = {
    A.SYSTEM_INFO: {
        "host name": ("What is this computer's name?", "What's my computer's hostname?", "What is this PC called?"),
        "operating system and version": (
            "Which version of Windows am I on?",
            "What OS is this computer running?",
            "Which macOS version do I have?",
        ),
        "processor model": ("What processor does this computer have?", "Which CPU is in my laptop?"),
        "processor cores": ("How many CPU cores do I have?", "How many cores does my processor have?"),
        "total memory": ("How much RAM does this computer have?", "How much memory is installed in my PC?"),
        "uptime": ("How long has this computer been running?", "When did my computer last restart?"),
        "time zone": ("What time zone is this computer set to?", "Which time zone is my laptop in?"),
        "architecture": ("Is this a 64-bit computer?", "What architecture is my machine?"),
    },
    A.NETWORK_INFO: {
        "IP address": ("What's my local IP address?", "What's my IPv6 address?"),
        "adapters": ("Which network adapters do I have?", "Is my ethernet connected?", "Is my wifi adapter up?"),
    },
    A.USER_INFO: {
        "user name": ("What's my user name?", "What is my username on this computer?"),
        "groups and rights": ("Which groups am I in on this computer?", "Do I have admin rights on this computer?"),
    },
    A.DISK_USAGE: {
        "drive size": ("How big is my hard drive?", "How large is my C drive?"),
        "space used and free": ("How much space is used on my C drive?", "How much free space is on drive D?"),
    },
}


def test_each_composite_read_names_its_facts_in_the_planners_table() -> None:
    answers = {read.action: read.answers.casefold() for read in READS}
    assert "host name" in answers[A.SYSTEM_INFO] and "processor" in answers[A.SYSTEM_INFO]
    assert "time zone" in answers[A.SYSTEM_INFO] and "64-bit" in answers[A.SYSTEM_INFO]
    assert "ip address" in answers[A.NETWORK_INFO]


@pytest.mark.parametrize(
    ("kind", "fact", "question"),
    [
        (kind, fact, question)
        for kind, facts in FACTS.items()
        for fact, questions in facts.items()
        for question in questions
    ],
)
def test_a_question_about_a_fact_becomes_a_task(kind: ActionType, fact: str, question: str) -> None:
    for open_conversation in (False, True):
        decided = route(question, conversation_open=open_conversation)
        assert decided.kind is TurnKind.ACTION, (kind.value, fact, question, open_conversation, decided)
