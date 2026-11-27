"""The actions that read this computer's state, what each answers, and how to tell neighbours apart.

The planner's prompt lists them from this table (pilot.agents.planner), so a read action cannot be left out of it
again. In the second live run "what programs are running?" and "what is the CPU temperature?" were both planned as
system_info, and "what is the volume level?" was answered "no tool can read it": the prompt described none of
process_list, volume_get or brightness_get, and said system_info was for "queries about the system". Each entry
says what the action answers and which neighbour to take for a nearby question (system information against the
running programs, the computer's memory against one program's memory, the disks against one folder's size, the
network's addresses against a speed test).

Questions no action answers are listed too, with the sentence the person gets. Nothing here runs: the table is text
for the prompt, and tests/test_read_catalog.py checks each example against the planner's own parameter models.
"""

from __future__ import annotations

from dataclasses import dataclass

from pilot.actions import ActionType

A = ActionType


@dataclass(frozen=True, slots=True)
class Read:
    action: ActionType
    params: str  # one or more examples, separated by " / ", exactly as the planner accepts them ("{}" for none)
    answers: str  # what the action returns, in the words a person asks with
    instead: str = ""  # the neighbour to use for a nearby question


READS: tuple[Read, ...] = (
    Read(
        A.PROCESS_LIST,
        '{} / {"name": "chrome"}',
        "the programs running now, each with its process id, CPU and memory share: what is running, whether an app "
        'is running ("is Spotify running"), which apps use the most memory or CPU, how much memory one app uses',
        "the computer's total memory (memory_usage) or processor load (cpu_usage)",
    ),
    Read(
        A.PROCESS_INFO,
        '{"pid": 1234}',
        "one process by its process id: its name, state, memory and CPU",
        'a program by name is process_list with "name"',
    ),
    Read(
        A.MEMORY_USAGE,
        "{}",
        "the computer's memory (RAM) in use and free, in total",
        "one program's memory is process_list",
    ),
    Read(
        A.CPU_USAGE,
        "{}",
        "how busy the processor is now, in total and per core",
        "never a temperature or fan speed (see QUESTIONS NO ACTION ANSWERS)",
    ),
    Read(
        A.DISK_USAGE,
        "{}",
        "each drive's size, used and free space",
        "the size of one folder is directory_size; the drives that are connected is disk_list",
    ),
    Read(
        A.DIRECTORY_SIZE,
        '{"path": "C:\\\\Users\\\\me\\\\Downloads"}',
        "the total size of one folder and what is in it",
        "free space on a drive is disk_usage",
    ),
    Read(
        A.DISK_LIST,
        "{}",
        "the disks and drives connected (USB drives too), with their sizes and file systems",
        "free space is disk_usage",
    ),
    Read(
        A.NETWORK_INFO,
        "{}",
        "this computer's network adapters (Wi-Fi, ethernet), their IP addresses (IPv4 and IPv6) and whether each is up",
        "never a speed test; nearby Wi-Fi networks are wifi_list",
    ),
    Read(A.WIFI_LIST, "{}", "the Wi-Fi networks in range", "this computer's own addresses are network_info"),
    Read(A.BATTERY_INFO, "{}", "the battery's charge, whether it is plugged in, and the time left"),
    Read(A.VOLUME_GET, "{}", "the sound volume level now, and whether it is muted", "changing it is volume_set"),
    Read(
        A.BRIGHTNESS_GET,
        "{}",
        "the screen's brightness level now",
        "changing it is brightness_set",
    ),
    Read(A.CLIPBOARD_READ, "{}", "the text on the clipboard now, without changing it"),
    Read(
        A.WINDOW_LIST,
        "{}",
        "the windows open on the desktop, with their titles",
        "programs without a window are process_list",
    ),
    Read(
        A.USER_INFO,
        "{}",
        "the account this computer is used with now: its user name, its groups and whether it is an administrator",
    ),
    Read(A.USER_LIST, "{}", "every user account on this computer"),
    Read(A.SERVICE_STATUS, '{"name": "nginx"}', "whether one system service is running, stopped or disabled"),
    Read(A.SCHEDULE_LIST, "{}", "the scheduled tasks set up on this computer"),
    Read(
        A.SYSTEM_HEALTH_REVIEW,
        "{}",
        "a check of what may be wrong now (processor load, memory, the fullest disk, battery, the busiest programs), "
        "with advice",
    ),
    Read(
        A.SYSTEM_INFO,
        '{"categories": ["os"]} / {}',
        'an overview, by category: "os" (the computer\'s name or host name, the operating system and its version, '
        'the processor model, 64-bit or 32-bit), "time" (the clock, the time zone and how long the computer has been '
        'on), "cpu" (cores and load), "memory" (how much RAM it has and how much is used), "disk", "network", '
        '"battery", or "all"; ask only for the categories the question needs',
        "a single reading has its own action above (memory_usage, cpu_usage, disk_usage, network_info, battery_info); "
        "never use it for what is running (process_list) or for a temperature",
    ),
)

# Questions no action answers, with the sentence the person gets. Return no actions and that sentence.
CANNOT_READ: tuple[tuple[str, str], ...] = (
    ("the processor's temperature", "I can't read the processor's temperature on this computer."),
    ("a graphics card's temperature", "I can't read the graphics card's temperature on this computer."),
    ("fan speeds", "I can't read the fan speeds on this computer."),
    ("the internet connection's speed (a speed test)", "I can't run a speed test on this computer."),
    ("the power the computer draws in watts", "I can't read how much power this computer draws."),
)


def reads_for_prompt() -> str:
    """The prompt's READING THIS COMPUTER section: one entry per read action, then the questions none answers."""
    lines = []
    for read in READS:
        line = f"- {read.action.value} — Params: {read.params}. Answers {read.answers}."
        if read.instead:
            line += f" Not for: {read.instead}."
        lines.append(line)
    lines.append(
        "- QUESTIONS NO ACTION ANSWERS: "
        + "; ".join(f"{what} (say: {sentence})" for what, sentence in CANNOT_READ)
        + ". For these, and any reading no action above returns, return no actions and that one sentence in "
        '"explanation". Never answer with a nearby reading instead: system information is not a temperature.'
    )
    return "\n".join(lines)
