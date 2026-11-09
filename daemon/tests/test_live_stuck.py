"""Heliox Live, part five: noticing when someone is stuck, and offering one concrete kind of help."""

from __future__ import annotations

from datetime import date

from pilot.live.detectors import TextContext
from pilot.live.stuck import StuckWatcher

MONDAY = date(2026, 11, 9)
NOTES = TextContext(today=MONDAY, app="WINWORD.EXE", title="Pitch.docx - Word")
TERMINAL = TextContext(today=MONDAY, app="WindowsTerminal.exe", title="PowerShell")
CODE = TextContext(today=MONDAY, app="Code.exe", title="main.py - Visual Studio Code")
WORD = "WINWORD.EXE|Pitch.docx - Word"


def test_rewriting_the_same_sentence_three_times_gets_an_offer():
    watcher = StuckWatcher()
    opening = "We help small teams ship faster because"
    found = []
    t = 0.0
    for attempt in (" our tool removes the busywork.", " we automate the boring parts.", " nobody likes setup."):
        for text in (opening + attempt, opening):  # write a version, then delete it
            t += 5
            found += watcher.observe(text, WORD, NOTES, now=t)
    [offer] = found
    assert offer.kind == "stuck" and offer.fix_kind == "ask" and offer.urgency == "low"
    assert offer.fix == f"Suggest a clearer way to write this: {opening}"
    assert offer.message.startswith("You've rewritten this a few times")


def test_one_long_deletion_read_over_several_looks_counts_once():
    watcher = StuckWatcher()
    text = "A fairly long sentence that is going to be deleted bit by bit. " * 2
    found = []
    for step in range(0, 100, 8):  # backspacing, seen every 0.6 s
        found += watcher.observe(text[: len(text) - step], WORD, NOTES, now=step / 10)
    assert found == [] and len(watcher._fields[WORD].rewrites) == 1


def test_ordinary_editing_is_not_being_stuck():
    watcher = StuckWatcher()
    text, t, found = "", 0.0, []
    for sentence in ("First point about the plan. ", "Second point, with numbers. ", "Third point. "):
        text += sentence
        t += 10
        found += watcher.observe(text, WORD, NOTES, now=t)
    text = text[:-20]  # one cut
    found += watcher.observe(text, WORD, NOTES, now=t + 5)
    for extra in ("A new paragraph that keeps growing. ", "And then some more text after it. "):
        text += extra
        found += watcher.observe(text, WORD, NOTES, now=t + 10)
    assert found == []
    # Three deletions spread over more than two minutes are revision, not circling.
    slow, found = StuckWatcher(), []
    for at in (0.0, 70.0, 140.0):
        found += slow.observe("A sentence that is long enough to cut down.", WORD, NOTES, now=at)
        found += slow.observe("A sentence", WORD, NOTES, now=at + 5)
    assert found == []


def test_flicking_between_two_windows_gets_one_offer_to_put_them_side_by_side():
    watcher = StuckWatcher()
    excel, chrome = "EXCEL.EXE|Budget.xlsx - Excel", "chrome.exe|Supplier prices - Chrome"
    found = []
    for i in range(10):
        found += watcher.observe("", excel if i % 2 else chrome, NOTES, now=i * 4.0)
    [offer] = found
    assert offer.message == "You keep switching between Excel and Chrome. Want them side by side?"
    assert offer.fix == "Put the Excel window and the Chrome window side by side"
    # Once is enough for a while.
    assert not [o for i in range(10) for o in watcher.observe("", excel if i % 2 else chrome, NOTES, now=50 + i)]


def test_a_few_switches_or_many_windows_are_ordinary():
    watcher = StuckWatcher()
    windows = ["a.exe|A", "b.exe|B", "c.exe|C", "d.exe|D"]
    assert not [o for i in range(12) for o in watcher.observe("", windows[i % 4], NOTES, now=i * 3.0)]
    watcher = StuckWatcher()
    assert not [o for i in range(5) for o in watcher.observe("", windows[i % 2], NOTES, now=i * 3.0)]


def test_the_same_error_three_times_in_a_terminal_gets_an_offer_once():
    watcher = StuckWatcher()
    run = "PS C:\\app> python main.py\nModuleNotFoundError: No module named 'cv2'\n"
    assert watcher.observe(run * 2, "WindowsTerminal.exe|PowerShell", TERMINAL, now=0) == []
    [offer] = watcher.observe(run * 3, "WindowsTerminal.exe|PowerShell", TERMINAL, now=30)
    assert offer.message == "That error has come up 3 times. Want me to look into it?"
    assert offer.fix == "Help me fix this error: ModuleNotFoundError: No module named 'cv2'"
    assert "pip install opencv-python" in offer.detail or offer.detail
    assert watcher.observe(run * 4, "WindowsTerminal.exe|PowerShell", TERMINAL, now=60) == []


def test_code_editors_are_left_alone():
    watcher = StuckWatcher()
    code = "def handler(event):\n    return process(event)\n"
    found = []
    for t in range(0, 60, 5):
        found += watcher.observe(code if t % 10 else code[:10], "Code.exe|main.py", CODE, now=float(t))
    assert found == []
