"""Three questions a person would ask, for every read action the catalog names, reach the planner or the quick road.

In the second live run "How bright is the screen?" stayed talk although brightness_get reads it. The catalog's own
contract checked one question per read action; this checks three more, in other words than the catalog's, and that
none of them ends as talk. Text only, no model."""

from __future__ import annotations

import pytest

from pilot.actions import READ_ONLY_ACTIONS, ActionType
from pilot.conversation import catalog_words
from pilot.conversation.router import TurnKind, route

A = ActionType

QUESTIONS: dict[ActionType, tuple[str, str, str]] = {
    A.FILE_READ: (
        "Show me what's in C:\\work\\note.txt",
        "Read C:\\work\\todo.txt to me",
        "What does the file C:\\work\\plan.md say?",
    ),
    A.FILE_LIST: (
        "What's in my Downloads folder?",
        "Show me the files on my desktop",
        "Which files are in C:\\work\\notes?",
    ),
    A.DIRECTORY_SUMMARY: (
        "Summarize my Documents folder",
        "Give me an overview of what's in C:\\work",
        "What kinds of files are in my Pictures folder?",
    ),
    A.DIRECTORY_SIZE: (
        "How big is my Downloads folder?",
        "How much space does C:\\work take up?",
        "What's the size of my Videos folder?",
    ),
    A.FILE_HASH: (
        "Give me the checksum of C:\\work\\setup.exe",
        "What's the md5 of C:\\work\\a.zip?",
        "Hash C:\\work\\note.txt with sha256",
    ),
    A.GIT_STATUS: (
        "Do I have uncommitted changes in C:\\work\\repo?",
        "What's changed in my repo?",
        "Is my git repository clean?",
    ),
    A.GIT_DIFF: (
        "Show me the diff of my repo",
        "What did I change in C:\\work\\repo since the last commit?",
        "What are my unstaged changes in the repo?",
    ),
    A.GIT_LOG: (
        "Show me the recent commits in C:\\work\\repo",
        "Who made the last commit in my repo?",
        "What's the commit history of my repo?",
    ),
    A.PACKAGE_SEARCH: (
        "Is there a package called ripgrep I can install?",
        "Can I get vlc from the package manager?",
        "Search the packages for ffmpeg",
    ),
    A.SERVICE_STATUS: (
        "Is the nginx service running?",
        "Is the print spooler service up?",
        "What state is the docker service in?",
    ),
    A.PROCESS_LIST: (
        "Is Spotify running?",
        "Which apps are open in the background?",
        "What processes are using the most CPU?",
    ),
    A.PROCESS_INFO: (
        "How much memory is process 4242 using?",
        "Tell me about the process with PID 1234",
        "What is process 880 doing?",
    ),
    A.CLIPBOARD_READ: ("What did I just copy?", "Read my clipboard", "What text is on the clipboard?"),
    A.SYSTEM_INFO: (
        "What operating system am I running?",
        "What are my computer's specs?",
        "How long has my computer been up?",
    ),
    A.SYSTEM_HEALTH_REVIEW: (
        "Is my computer healthy?",
        "Run a health check on my computer",
        "Is anything wrong with my PC right now?",
    ),
    A.DISK_USAGE: (
        "How full is my hard drive?",
        "How much storage do I have left?",
        "Am I running out of disk space?",
    ),
    A.MEMORY_USAGE: ("How much RAM am I using?", "Is my memory almost full?", "How much free memory do I have?"),
    A.CPU_USAGE: ("How busy is the processor?", "What's my CPU load right now?", "Is the CPU maxed out?"),
    A.NETWORK_INFO: ("What's my IP address?", "Am I connected to the internet?", "Which network am I on?"),
    A.BATTERY_INFO: ("How much charge is left?", "Is my laptop charging?", "What's my battery level?"),
    A.SCHEDULE_LIST: (
        "What scheduled tasks are set up?",
        "Show me my scheduled jobs",
        "Which tasks are scheduled to run on this computer?",
    ),
    A.ENV_GET: (
        "What is my PATH?",
        "What's the value of the JAVA_HOME environment variable?",
        "Show me the HOME environment variable",
    ),
    A.ENV_LIST: (
        "List my environment variables",
        "What environment variables do I have?",
        "Show all the environment variables",
    ),
    A.WINDOW_LIST: ("Which windows do I have open?", "List the open windows", "What apps have windows open right now?"),
    A.VOLUME_GET: ("How loud is my sound right now?", "What's the volume at?", "Is my volume turned up?"),
    A.BRIGHTNESS_GET: ("How bright is the screen?", "What's my screen brightness?", "Is my display dimmed?"),
    A.WIFI_LIST: ("Which Wi-Fi networks can I see?", "List nearby wireless networks", "What wifi networks are around?"),
    A.DISK_LIST: ("What drives are plugged in?", "List my disks", "Is my USB drive connected?"),
    A.USER_LIST: ("Who has an account on this computer?", "List the user accounts", "Which users are on this PC?"),
    A.USER_INFO: (
        "Which user am I logged in as?",
        "What account am I using?",
        "Am I an administrator on this computer?",
    ),
    A.REGISTRY_READ: (
        "What's in the registry key HKCU\\Software\\Example?",
        "Read the registry value for my wallpaper",
        "Show me the registry entry for startup programs",
    ),
    A.LOG_ANALYZE: (
        "Are there errors in C:\\work\\app.log?",
        "Show me the last 20 lines of C:\\work\\app.log",
        "What went wrong according to the log file C:\\work\\app.log?",
    ),
    A.MOUSE_POSITION: ("Where is my mouse pointer?", "What's the cursor position?", "Where's the mouse right now?"),
    A.BROWSER_EXTRACT: (
        "What does https://example.com say?",
        "Read the text on https://example.com",
        "Get the page text of https://example.com",
    ),
    A.BROWSER_EXTRACT_TABLE: (
        "Get the table from https://example.com/prices",
        "What's in the table on https://example.com?",
        "Extract the table on https://example.com",
    ),
    A.BROWSER_EXTRACT_LINKS: (
        "List the links on https://example.com",
        "What links are on https://example.com?",
        "Which pages does https://example.com link to?",
    ),
    A.BROWSER_LIST_TABS: ("Which tabs do I have open?", "List my browser tabs", "What browser tabs are open?"),
    A.BROWSER_PAGE_INFO: (
        "What's the title of https://example.com?",
        "What page am I on in the browser?",
        "What's the page title of https://example.com?",
    ),
    A.TRIGGER_LIST: ("What triggers do I have set up?", "Show my triggers", "Which watchers are active?"),
    A.CALENDAR_LIST_EVENTS: (
        "What's on my calendar tomorrow?",
        "Do I have meetings on Friday on my calendar?",
        "Show me my calendar for next week",
    ),
    A.CALENDAR_FETCH: ("What's on my calendar today?", "Check my calendar", "Any events on my calendar this week?"),
    A.API_SCRAPE: (
        "Scrape https://example.com for the prices",
        "Extract the headlines from https://example.com",
        "What is the text of the elements matching '.price' on https://example.com?",
    ),
    A.EMAIL_FETCH: ("Do I have new emails?", "Check my inbox", "Any unread messages in my inbox?"),
    A.MCP_LIST_TOOLS: ("Which MCP tools can you use?", "List my MCP servers", "What MCP tools are connected?"),
}


def _named_read_actions() -> list[ActionType]:
    return [
        kind
        for kind, words in catalog_words.NAMED.items()
        if kind in READ_ONLY_ACTIONS and (words.subjects or words.topics)
    ]


def test_every_named_read_action_has_three_questions_here() -> None:
    missing = [kind.value for kind in _named_read_actions() if kind not in QUESTIONS]
    assert not missing, missing


@pytest.mark.parametrize(
    ("kind", "question"), [(kind, question) for kind, questions in QUESTIONS.items() for question in questions]
)
def test_the_question_becomes_a_task_and_not_talk(kind: ActionType, question: str) -> None:
    for open_conversation in (False, True):
        decided = route(question, conversation_open=open_conversation)
        assert decided.kind is TurnKind.ACTION, (kind.value, question, open_conversation, decided)
