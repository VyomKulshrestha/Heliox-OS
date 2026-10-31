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
        "group": "Suggestions",
        "about": "One timely nudge from your goals, habits and list; ask any time, or turn on gentle check-ins.",
        "examples": ["what should I do now", "any suggestions", "turn on suggestions"],
    },
    {
        "group": "Your assistant",
        "about": "Name Heliox and shape its tone; the tone carries into both quick replies and the model's voice.",
        "examples": [
            "call yourself Nova",
            "what's your name",
            "be more playful",
            "be strictly professional",
            "introduce yourself",
            "go back to normal",
        ],
    },
    {
        "group": "Goals",
        "about": "Long-lived goals Heliox keeps; your next steps show in the morning and the Today tab.",
        "examples": [
            "start a goal: launch the podcast",
            "add to my podcast goal: recorded episode one",
            "set the next step on the podcast to book a guest",
            "what's my next step on the podcast",
            "what are my goals",
            "I finished the podcast goal",
        ],
    },
    {
        "group": "About you",
        "about": "What you tell Heliox to remember about you, kept on this computer and used to personalize it.",
        "examples": [
            "call me Vyom",
            "remember that I'm a founder",
            "my manager is Sarah",
            "I prefer short answers",
            "what do you know about me",
            "forget everything about me",
        ],
    },
    {
        "group": "Start and end the day",
        "about": "Reminders, your list, streaks and anything wrong with the computer, then the day's summary.",
        "examples": ["good morning", "what's my day look like", "good night"],
    },
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
            "remind me every hour to stand up",
            "remind me every weekday at 9 to check email",
            "remind me to submit my timesheet when I open Outlook",
            "what are my reminders",
            "cancel the reminder about rent",
            "stop reminding me to stand up",
        ],
    },
    {
        "group": "Notes",
        "about": "Short notes kept on this computer.",
        "examples": ["note that the plumber comes Tuesday", "read my notes", "what did I note about the plumber"],
    },
    {
        "group": "To-do list",
        "about": "Kept on this computer; open items show in the Today tab and the daily brief.",
        "examples": [
            "add call the bank to my list",
            "what's on my list",
            "mark call the bank as done",
            "remove call the bank from my list",
            "clear finished items",
        ],
    },
    {
        "group": "Focus",
        "about": "Holds reminders and the brief; approvals still come through.",
        "examples": [
            "focus for 45 minutes",
            "start a pomodoro",
            "focus every weekday from 9 to 12",
            "what is my focus schedule",
            "during focus, tell me if I open YouTube",
            "how long did I focus this week",
            "what are my distractions",
            "stop focusing",
        ],
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
        "group": "Tell me when",
        "about": "Watches until something happens, then tells you once; it never changes anything.",
        "examples": [
            "tell me when Chrome closes",
            "let me know when my download finishes",
            "tell me when my battery is full",
            "tell me when my computer calms down",
            "what are you watching for",
            "stop watching Chrome",
        ],
    },
    {
        "group": "Quick answers",
        "about": "Sums, percentages, conversions, dates and world time, worked out exactly on this computer.",
        "examples": [
            "what's 15 percent of 240",
            "what's 20 percent off 80",
            "convert 5 miles to km",
            "what is 70 fahrenheit in celsius",
            "how many days until Christmas",
            "what day is December 25",
            "what time is it in Tokyo",
            "how long until 5 pm",
        ],
    },
    {
        "group": "Your files",
        "about": "From folder listings and file dates in your own folders; nothing is opened.",
        "examples": [
            "what did I download today",
            "what was the last file I downloaded",
            "where is my budget spreadsheet",
        ],
    },
    {
        "group": "This computer",
        "about": "A quick reading of processor, memory and disks; closing anything still asks first.",
        "examples": [
            "why is my computer slow",
            "what's using my memory",
            "what's using my CPU",
            "how much disk space do I have",
        ],
    },
    {
        "group": "Clipboard history",
        "about": "Off until you turn it on; kept in memory for a few hours, never passwords or keys.",
        "examples": ["what did I copy earlier", "copy the second one back", "clear my clipboard history"],
    },
    {
        "group": "Snippets",
        "about": "Save what you copied under a name and put it back on the clipboard later; never passwords.",
        "examples": [
            "save that as my signature",
            "copy my signature",
            "what snippets do I have",
            "read what I copied",
        ],
    },
    {
        "group": "Habits",
        "about": "Log a habit by voice and see counts and streaks; kept on this computer.",
        "examples": [
            "log a workout",
            "I drank a glass of water",
            "how many workouts this week",
            "what's my workout streak",
            "show my habits",
        ],
    },
    {
        "group": "Helpers",
        "about": "Passwords go straight to the clipboard and are never kept; chance and the stopwatch run in code.",
        "examples": [
            "generate a password",
            "flip a coin",
            "roll a die",
            "pick a number between 1 and 100",
            "choose between pizza or sushi",
            "start a stopwatch",
            "how long has it been",
            "stop the stopwatch",
        ],
    },
    {
        "group": "Usage nudges",
        "about": "A once-a-day heads-up from the activity timeline; nothing is blocked.",
        "examples": ["tell me if I spend more than an hour on YouTube today", "what nudges do I have"],
    },
    {
        "group": "Privacy",
        "about": "Remove a stretch of the activity timeline; Settings → Your data shows everything Heliox keeps.",
        "examples": ["forget my activity from the last hour"],
    },
    {
        "group": "Voice",
        "about": "Hear the last answer again, or see everything you can say.",
        "examples": ["say that again", "what can I say"],
    },
    {
        "group": "Settings",
        "about": "Everyday settings only; security and approval settings never change by voice.",
        "examples": [
            "turn on the activity timeline",
            "turn on clipboard history",
            "send me a daily brief at 7 pm",
            "read the brief aloud",
        ],
    },
]

SPOKEN_HELP = (
    "Besides doing things on your computer, I handle these right away: undo, reminders and timers, notes, "
    "your to-do list, focus, your setups, routines and shortcuts, and questions about your day, like what did you do today. "
    "The command palette is open on screen with examples; you can open it any time with Control K."
)


def guide() -> list[dict[str, Any]]:
    return [dict(group, examples=list(group["examples"])) for group in GUIDE]


def all_examples() -> list[str]:
    return [example for group in GUIDE for example in group["examples"]]
