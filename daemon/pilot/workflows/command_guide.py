"""The guide to what Heliox answers or does right away, without planning.

Every example here is a sentence the person can say or type as it is. A test checks
that each one matches the code that handles it, so the guide cannot drift from what
Heliox actually understands. The command palette (Ctrl+K) and "what can I say?" both
read from this list.
"""

from __future__ import annotations

import re
from typing import Any

HELP_REQUEST = re.compile(
    r"^(?:hey heliox |heliox )?(?:what can i say|what (?:voice )?commands (?:are there|can i use|do you know)"
    r"|show (?:me )?(?:the |your )?(?:voice )?commands|open the command palette)$"
)

GUIDE: list[dict[str, Any]] = [
    {
        "group": "Undo",
        "about": "Open the undo preview for recent tasks; every undo still needs your click.",
        "examples": ["undo that", "undo the last two tasks"],
    },
    {
        "group": "Your day",
        "about": "Answered from your receipts, activity timeline and undo history on this computer.",
        "examples": [
            "what did you do today",
            "what was I working on this morning",
            "when did I last open the quarterly report",
            "when did I ask you to email the invoice",
            "where did you put my report",
        ],
    },
    {
        "group": "Reminders and timers",
        "about": "Kept across restarts and spoken when due.",
        "examples": [
            "remind me in 20 minutes to stretch",
            "remind me tomorrow at 9 am to send the report",
            "set a timer for 10 minutes",
            "what are my reminders",
            "cancel the reminder about rent",
        ],
    },
    {
        "group": "Notes",
        "about": "Short notes kept on this computer.",
        "examples": ["note that the plumber comes Tuesday", "read my notes", "what did I note about the plumber"],
    },
    {
        "group": "Focus",
        "about": "Holds reminders and the brief; approvals still come through.",
        "examples": ["focus for 45 minutes", "stop focusing"],
    },
    {
        "group": "Setups",
        "about": "Remember which apps you have open and bring them back.",
        "examples": ["save this as my coding setup", "open my coding setup", "what setups do I have"],
    },
    {
        "group": "Routines and shortcuts",
        "about": "Repeat a finished task on a schedule, or give a long command a short phrase.",
        "examples": [
            "do that every weekday at 8:30 am",
            "do that again",
            "when I say start work, open VS Code and Slack",
        ],
    },
    {
        "group": "Usage nudges",
        "about": "A once-a-day heads-up from the activity timeline; nothing is blocked.",
        "examples": ["tell me if I spend more than an hour on YouTube today", "what nudges do I have"],
    },
    {
        "group": "Settings",
        "about": "Everyday settings only; security and approval settings never change by voice.",
        "examples": ["turn on the activity timeline", "send me a daily brief at 7 pm", "read the brief aloud"],
    },
]

SPOKEN_HELP = (
    "Besides doing things on your computer, I handle these right away: undo, reminders and timers, notes, "
    "focus, your setups, routines and shortcuts, and questions about your day, like what did you do today. "
    "The command palette is open on screen with examples; you can open it any time with Control K."
)


def guide() -> list[dict[str, Any]]:
    return [dict(group, examples=list(group["examples"])) for group in GUIDE]


def all_examples() -> list[str]:
    return [example for group in GUIDE for example in group["examples"]]
