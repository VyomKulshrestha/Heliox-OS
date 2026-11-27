"""The planner's read actions come from one table that says what each answers and what it is not for.

In the second live run "what programs are running?", "what is the CPU temperature?" and "how much memory is in use?"
were planned as system_info, and "what is the volume level?" was answered "no tool can read it": the prompt named
none of process_list, memory_usage, volume_get. A replay of 31 read questions landed on the right action 42 times in
93 before the table (scratchpad live12read_scripts/replay_reads.py). Text only, no model."""

from __future__ import annotations

import asyncio
import sys
import types

from pilot.actions import READ_ONLY_ACTIONS, ActionType
from pilot.agents.planner import SYSTEM_PROMPT
from pilot.agents.read_catalog import CANNOT_READ, READS, reads_for_prompt

A = ActionType

# Every action that reads this computer's state: each must be in the table, so the prompt names it.
STATE_READS = {
    A.PROCESS_LIST,
    A.PROCESS_INFO,
    A.MEMORY_USAGE,
    A.CPU_USAGE,
    A.DISK_USAGE,
    A.DIRECTORY_SIZE,
    A.DISK_LIST,
    A.NETWORK_INFO,
    A.WIFI_LIST,
    A.BATTERY_INFO,
    A.VOLUME_GET,
    A.BRIGHTNESS_GET,
    A.CLIPBOARD_READ,
    A.WINDOW_LIST,
    A.USER_INFO,
    A.USER_LIST,
    A.SERVICE_STATUS,
    A.SCHEDULE_LIST,
    A.SYSTEM_HEALTH_REVIEW,
    A.SYSTEM_INFO,
}

# Pairs a model confuses: the first entry's "not for" names the second.
CONFUSABLE = [
    (A.PROCESS_LIST, A.MEMORY_USAGE),
    (A.PROCESS_LIST, A.CPU_USAGE),
    (A.MEMORY_USAGE, A.PROCESS_LIST),
    (A.DISK_USAGE, A.DIRECTORY_SIZE),
    (A.DISK_USAGE, A.DISK_LIST),
    (A.DIRECTORY_SIZE, A.DISK_USAGE),
    (A.NETWORK_INFO, A.WIFI_LIST),
    (A.WIFI_LIST, A.NETWORK_INFO),
    (A.WINDOW_LIST, A.PROCESS_LIST),
    (A.PROCESS_INFO, A.PROCESS_LIST),
    (A.SYSTEM_INFO, A.PROCESS_LIST),
    (A.SYSTEM_INFO, A.MEMORY_USAGE),
    (A.VOLUME_GET, A.VOLUME_SET),
    (A.BRIGHTNESS_GET, A.BRIGHTNESS_SET),
]


def test_every_state_read_is_in_the_table_once_and_reads_only() -> None:
    kinds = [read.action for read in READS]
    assert set(kinds) == STATE_READS and len(kinds) == len(set(kinds))
    assert all(kind in READ_ONLY_ACTIONS for kind in kinds)


def test_the_prompt_names_every_read_from_the_table() -> None:
    rendered = SYSTEM_PROMPT.format(os="Windows", path_style="Windows (C:\\Users\\...)", home="C:\\Users\\me")
    for read in READS:
        assert f"- {read.action.value} — Params: " in rendered, read.action
    assert "<<READ_ACTIONS>>" not in SYSTEM_PROMPT
    assert "use the system_info actions" not in rendered


def test_confusable_reads_say_which_neighbour_to_take() -> None:
    by_kind = {read.action: read for read in READS}
    for first, second in CONFUSABLE:
        assert second.value in by_kind[first].instead, (first.value, second.value)


def test_questions_no_action_answers_have_the_sentence_the_person_gets() -> None:
    text = reads_for_prompt()
    assert "I can't read the processor's temperature on this computer." in text
    assert all(sentence in text for _what, sentence in CANNOT_READ)
    assert "return no actions" in text and "system information is not a temperature" in text


def test_the_clock_section_says_how_long_the_computer_has_been_on(monkeypatch) -> None:
    from pilot.system import sysinfo

    fake = types.SimpleNamespace(boot_time=lambda: 0.0)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    text = asyncio.run(sysinfo._time_info())
    assert "=== System Time ===" in text and "Up for: " in text and " day(s), " in text
