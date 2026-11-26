"""The quick road answers only the question asked: a read runs there when the words name that state and nothing the
read does not cover (pilot.conversation.catalog_words.asks_only_for). Otherwise the planner gets the request.

"What is the CPU temperature right now?" read as CPU usage, and "How much memory does my graphics card have?" read as
system memory, were confident wrong answers. Text only, no model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from pilot.actions import ActionType
from pilot.conversation.quick import from_words
from pilot.conversation.router import _LOOKUP_QUESTION, TurnKind, catalog_net, normalize, route

STAYS_QUICK = [
    ("How much battery is left?", ActionType.BATTERY_INFO),
    ("what's my battery level", ActionType.BATTERY_INFO),
    ("What's my disk space?", ActionType.DISK_USAGE),
    ("How much free disk space is there?", ActionType.DISK_USAGE),
    ("How much disk space do I have left?", ActionType.DISK_USAGE),
    ("memory usage right now", ActionType.MEMORY_USAGE),
    ("How much memory is in use right now?", ActionType.MEMORY_USAGE),
    ("how much ram do I have free", ActionType.MEMORY_USAGE),
    ("What's the CPU usage?", ActionType.CPU_USAGE),
    ("how busy is the cpu right now", ActionType.CPU_USAGE),
    ("What's the CPU load at the moment?", ActionType.CPU_USAGE),
    ("How much storage space is left on this laptop?", ActionType.DISK_USAGE),
]

GOES_TO_THE_PLANNER = [
    "What is the CPU temperature right now?",
    "How much memory does my graphics card have?",
    "What's the GPU memory usage?",
    "How fast is the CPU fan spinning?",
    "What's the CPU usage per core?",
    "How much memory is Chrome using?",
    "How much space is left on drive D?",
    "How much space does my Downloads folder take?",
    "What was the battery level yesterday?",
    "What's the average CPU usage today?",
    "Is my memory usage higher than yesterday?",
    "What is the battery health?",
]


def _quick(text: str):
    decided = route(text)
    if decided.kind is not TurnKind.ACTION:
        return None  # talk: no reading is taken at all
    return from_words(normalize(text), decided.verb, decided.target)


@pytest.mark.parametrize(("text", "kind"), STAYS_QUICK)
def test_a_plain_read_stays_on_the_quick_road(text, kind):
    assert route(text).kind is TurnKind.ACTION
    quick = _quick(text)
    assert quick is not None and quick.plan.actions[0].action_type is kind, quick


@pytest.mark.parametrize("text", GOES_TO_THE_PLANNER)
def test_a_qualified_read_goes_to_the_planner(text):
    assert _quick(text) is None


def _replay_lines() -> list[str]:
    path = Path(__file__).with_name("test_conversation_catalog_net.py")
    spec = importlib.util.spec_from_file_location("catalog_net_tables", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["catalog_net_tables"] = module
    spec.loader.exec_module(module)
    lines = [text for text, _verb in module.RUN_REQUESTS]
    return lines + module.PARAPHRASES + module.STAYS_TALK + module.ASKED_ABOUT + STAYS_QUICK_TEXTS + GOES_TO_THE_PLANNER


STAYS_QUICK_TEXTS = [text for text, _kind in STAYS_QUICK]


def test_the_old_lookup_rule_and_the_catalog_net_agree():
    for text in _replay_lines():
        words = normalize(text)
        if _LOOKUP_QUESTION.match(words):
            netted = catalog_net(text, words, words)
            assert netted is not None and netted.kind is TurnKind.ACTION, (text, netted)
