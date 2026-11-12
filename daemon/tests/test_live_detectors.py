"""Heliox Live detectors added after the first release: broad deletes, impossible dates and times, weekdays next to
numeric and relative dates, percentages with a connective, and API keys named by their issuer.

Precision first: each rule has at least as many look-alikes it must leave alone as slips it must catch."""

from __future__ import annotations

from datetime import date

import pytest

from pilot.live.detectors import Correction, TextContext, detect

TUESDAY = date(2026, 11, 10)  # 11 November 2026 is a Wednesday


def terminal(text: str) -> list[Correction]:
    return detect(text, TextContext(today=TUESDAY, app="WindowsTerminal.exe", title="PowerShell"))


def chat(text: str, today: date = TUESDAY, hour: int | None = 11) -> list[Correction]:
    return detect(text, TextContext(today=today, app="slack.exe", title="#team - Slack", hour=hour))


def kinds(found: list[Correction]) -> list[str]:
    return [c.kind for c in found]


# ── recursive deletes aimed at something broad ──


@pytest.mark.parametrize(
    ("line", "title", "evidence"),
    [
        (
            "PS C:\\Users\\me> Remove-Item -Recurse -Force C:\\Users\\me\\Documents",
            "That deletes the whole Documents folder",
            "Remove-Item -Recurse -Force C:\\Users\\me\\Documents",
        ),
        ("C:\\Users\\me>del /s /q C:\\Users\\me\\Documents", "That deletes the whole Documents folder", None),
        ("rd /s /q C:\\", "That deletes everything on C:", "rd /s /q C:\\"),
        ("rd /s /q %USERPROFILE%", "That deletes your home folder", None),
        ('rmdir /s /q "C:\\Program Files"', "That deletes a system folder", None),
        (
            "Remove-Item -Path $env:USERPROFILE\\Downloads\\* -Recurse -Force",
            "That deletes everything in your Downloads folder",
            None,
        ),
        ("$ rm -rf ~/", "That deletes your home folder", "rm -rf ~/"),
        ("$ rm -rf ~/*", "That deletes everything in your home folder", None),
        ("$ rm -rf ~/Documents", "That deletes your Documents folder", None),
        ("me@box:~$ rm -r $HOME/Downloads/", "That deletes your Downloads folder", "rm -r $HOME/Downloads/"),
        ("$ sudo rm -rf /home/sam", "That deletes a whole user folder", None),
        ("$ rm -rf /usr", "That deletes a system folder", None),
        ("cd ~ && rm -rf ~/Desktop", "That deletes your Desktop folder", None),
    ],
)
def test_a_recursive_delete_of_something_broad_is_urgent(line, title, evidence):
    [c] = terminal(line)
    assert c.kind == "risky_command" and c.urgency == "urgent"
    assert c.title == title
    if evidence:
        assert c.evidence == evidence


def test_a_broad_powershell_delete_offers_the_whatif_version():
    [c] = terminal("PS C:\\Users\\me> Remove-Item -Recurse -Force C:\\Users\\me\\Documents")
    assert c.message == "Nothing goes to the Recycle Bin."
    assert c.fix == "Remove-Item -Recurse -Force C:\\Users\\me\\Documents -WhatIf" and c.fix_kind == "command"
    [unix] = terminal("$ rm -rf ~/Documents")
    assert unix.message == "rm has no undo." and unix.fix == ""


@pytest.mark.parametrize(
    "line",
    [
        "$ rm -rf ./build",
        "$ rm -rf ~/launch",  # one named project folder: specific, and often meant
        "$ rm -rf ~/projects/app/node_modules",
        "$ rm -rf ~/Downloads/old-installers",
        "$ rm -rf /usr/local/lib/node_modules",
        "$ rm -rf $HOME/.cache",
        "$ rm -ri ~/Documents",  # asks before each file
        "$ rm ~/Downloads/setup.exe",  # not recursive
        "$ ls -R ~",
        "PS C:\\Users\\me> Remove-Item -Recurse -Force C:\\Users\\me\\launch",
        "PS C:\\repo> Remove-Item -Recurse -Force .\\dist",
        'Remove-Item -Recurse -Force "C:\\Users\\me\\Documents\\My Project"',
        "Remove-Item -Recurse -Force C:\\Users\\me\\Documents -WhatIf",
        "Remove-Item -Recurse C:\\Users\\me\\Documents -Confirm",
        "Remove-Item C:\\Users\\me\\Documents\\notes.txt",
        "Get-ChildItem -Recurse C:\\Users\\me\\Documents",
        "del /s /q C:\\Users\\me\\Documents\\*.tmp",
        "del C:\\Users\\me\\Documents\\report.docx",
        "rd /s /q build",
        "rd /s /q C:\\Users\\me\\Documents\\old",
        "dir /s C:\\",
        "Copy-Item -Recurse C:\\Users\\me\\Documents D:\\backup",
        "git branch -D release",  # deliberate after a squash merge; the branch is in the reflog
    ],
)
def test_routine_and_specific_deletes_stay_silent(line):
    assert terminal(line) == [], line


def test_a_delete_mentioned_mid_sentence_in_a_chat_is_not_a_command():
    assert chat("never run Remove-Item -Recurse C:\\Users\\me\\Documents on the shared box") == []
    assert chat("I cleaned up with rd /s /q build earlier") == []


# ── dates that do not exist ──


@pytest.mark.parametrize(
    ("text", "title", "message"),
    [
        ("Meeting on 31 November", "There's no 31 November", "November has 30 days."),
        ("It's due on February 30", "There's no 30 February", "February has 28 days."),
        ("Can we do the 31st of June?", "There's no 31 June", "June has 30 days."),
        ("Payroll runs on Sept 31.", "There's no 31 September", "September has 30 days."),
        ("Book it for 29 February", "There's no 29 February 2027", "2027 isn't a leap year."),
        (
            "Launch is 29 February 2027",
            "There's no 29 February 2027",
            "2027 isn't a leap year.",
        ),
        ("Friday 29 February", "There's no 29 February 2027", "2027 isn't a leap year."),
    ],
)
def test_a_date_that_does_not_exist(text, title, message):
    [c] = chat(text)
    assert (c.kind, c.title, c.message) == ("weekday_date", title, message)


@pytest.mark.parametrize(
    "text",
    [
        "Meeting on 30 November",
        "February 28 works",
        "31 December is the cut-off",
        "Launch is 29 February 2028",
        "May 31 is a Monday",
        "There's no 31 November, so let's say 1 December",
        "November 31 doesn't exist in the calendar",
        "The script rejects invalid dates like 30 February",
        "We're off 30-31 November",  # a range ending on a day that does not exist is not worth a card
        "Order 31 Nov-2 units",
    ],
)
def test_real_dates_and_talk_about_impossible_ones_stay_silent(text):
    assert chat(text) == [], text


def test_a_leap_day_is_fine_when_the_coming_february_has_one():
    assert chat("Book it for 29 February", today=date(2027, 11, 10)) == []
    assert chat("Book it for 29 February", today=date(2028, 1, 20)) == []


def test_impossible_dates_in_code_and_terminals_are_test_data():
    assert detect("assert parse('31 November') is None", TextContext(today=TUESDAY, app="Code.exe")) == []
    assert terminal("echo 'due 31 November'") == []


# ── a weekday next to a numeric date ──


def test_a_weekday_next_to_a_numeric_date_that_reads_only_one_way():
    [c] = chat("Deadline is Thursday 11/11")
    assert c.title == "11 November is a Wednesday" and c.message == "You wrote Thursday."
    assert c.evidence == "Thursday 11/11" and c.fix == "Wednesday 11/11" and c.fix_kind == "replace"
    [later] = chat("Kickoff is Thursday 13/11.")
    assert later.message == "You wrote Thursday." and later.fix == "Friday 13/11"
    [year] = chat("Due Thursday 11/11/2026.")
    assert year.fix == "Wednesday 11/11/2026"


def test_a_numeric_date_that_misses_the_weekday_both_ways():
    [c] = chat("Tuesday 4/12 at 3")
    assert c.title == "4/12 isn't a Tuesday"
    assert c.message == "4 December is a Friday and 12 April is a Monday."
    assert c.fix == ""


@pytest.mark.parametrize(
    "text",
    [
        "Deadline is Wed, 11/11",
        "Friday 4/12 at 3",  # 4 December is a Friday
        "Monday 4/12 at 3",  # 12 April is a Monday
        "Friday 3/4 of the team is out",
        "Monday 1/2 day off",
        "Thursday 2/3 done",
        "we sat 3/4 of the exam",
        "Thursday 3.5 hours",
        "Thursday 3/4/5 split",
        "Wednesday 11/11",
    ],
)
def test_numeric_dates_that_match_or_are_not_dates_stay_silent(text):
    assert chat(text) == [], text


# ── today, tomorrow, yesterday ──


@pytest.mark.parametrize(
    ("text", "title", "fix"),
    [
        ("See you tomorrow, Thursday", "Tomorrow is Wednesday", "tomorrow, Wednesday"),
        ("See you tomorrow (Thursday)!", "Tomorrow is Wednesday", "tomorrow (Wednesday)"),
        ("See you tomorrow, Thursday at 10", "Tomorrow is Wednesday", "tomorrow, Wednesday"),
        ("Busy today (Monday) with reviews", "Today is Tuesday", "today (Tuesday)"),
        ("Remember today is Monday.", "Today is Tuesday", "today is Tuesday"),
        ("Like I said yesterday (Sunday)", "Yesterday was Monday", "yesterday (Monday)"),
    ],
)
def test_a_relative_day_with_the_wrong_weekday(text, title, fix):
    [c] = chat(text)
    assert (c.title, c.fix) == (title, fix)


def test_a_wrong_tomorrow_says_when_the_named_day_is():
    [c] = chat("See you tomorrow, Thursday")
    assert c.message == "You wrote Thursday."
    assert c.detail == "The next Thursday is 12 November."


@pytest.mark.parametrize(
    "text",
    [
        "See you tomorrow, Wednesday",
        "See you tomorrow (Wednesday)",
        "Today is Tuesday",
        "I'm off tomorrow, Thursday and Friday",
        "Can't do today, Monday works better",
        "Not today, Thursday",
        "the day after tomorrow, Thursday",
        "if today is Monday we ship",
        "I woke up thinking today is Monday",
        "Is tomorrow Thursday?",
        "tomorrow is Thursday?",
        "today is Monday's deadline",
        "Monday, today, Thursday",
    ],
)
def test_relative_days_that_match_or_are_not_claims_stay_silent(text):
    assert chat(text) == [], text


def test_after_midnight_tomorrow_still_means_the_day_after_sleep():
    assert chat("See you tomorrow, Tuesday", hour=1) == []  # at 1 a.m. on Tuesday
    assert kinds(chat("See you tomorrow, Tuesday", hour=9)) == ["weekday_date"]


# ── clock times that do not exist ──


@pytest.mark.parametrize(
    ("text", "title", "message", "fix"),
    [
        ("Call at 25:00", "There's no 25:00", "A day ends at 23:59.", ""),
        ("Let's start at 13:75.", "There's no 13:75", "An hour has 60 minutes.", ""),
        ("Open until 24:30", "There's no 24:30", "A day ends at 23:59.", ""),
        ("Meet at 14pm.", "14pm is 2pm", "It mixes the 24-hour and 12-hour clocks.", "2pm"),
        (
            "Meet at 14:30pm",
            "14:30pm is 2:30pm",
            "It mixes the 24-hour and 12-hour clocks.",
            "2:30pm",
        ),
        ("Standup at 13am", "There's no 13am", "It mixes the 24-hour and 12-hour clocks.", ""),
    ],
)
def test_a_clock_time_that_does_not_exist(text, title, message, fix):
    [c] = chat(text)
    assert (c.kind, c.title, c.message, c.fix) == ("weekday_date", title, message, fix)


@pytest.mark.parametrize(
    "text",
    [
        "Call at 23:59",
        "Doors close at 24:00",
        "Meet at 2pm",
        "The video is 25:00 long",
        "Skip to 13:75 in the recording",
        "Look at 1:25:00 into the stream",
        "At 25:00 minutes in, the demo starts",
        "We hired 14 PM interns",
        "The ratio is 16:90",
        "Read John 13:75 tonight",
        "Version 14pm2 is out",
        "Meet at 12pm",
    ],
)
def test_real_times_durations_and_ratios_stay_silent(text):
    assert chat(text) == [], text


# ── percentages with a connective ──


@pytest.mark.parametrize(
    ("text", "fix"),
    [
        ("That is 20% of 150, so 35", "20% of 150, so 30"),
        ("20% of 150 would be 35.", "20% of 150 would be 30"),
        ("Tip: 15% of 80, which is 14", "15% of 80, which is 12"),
        ("10% of 200 -> 30", "10% of 200 -> 20"),
    ],
)
def test_a_percentage_said_with_a_connective(text, fix):
    [c] = chat(text)
    assert c.kind == "arithmetic" and c.fix == fix


@pytest.mark.parametrize(
    "text",
    [
        "That is 20% of 150, so 30",
        "20% of 150, so 35 more seats",
        "20% of 150, so about 35",
        "15% of 47, so 7",  # 7.05, rounded the way people round
        "10% of 200, so 30 + tax",
        "20% of 150 would be 30.",
        "Up 20% of 150 people, so 35 stayed",
        "We cut 5% of 120, which is 6",
    ],
)
def test_percentages_that_are_right_or_are_not_the_claim_stay_silent(text):
    assert chat(text) == [], text


# ── API keys named by their issuer ──


def test_api_keys_are_named_by_their_prefix():
    anthropic = chat("key sk-ant-api03-Zx81mQe4Lr90TuVb27NcPk55aWdYqq")
    assert [c.title for c in anthropic] == ["That looks like an Anthropic API key"]
    assert chat("sk-proj-Zx81mQe4Lr90TuVb27NcPk55aWdY")[0].title == "That looks like an OpenAI API key"
    assert chat("use sk-Zx81mQe4Lr90TuVb27NcPk55aWdY")[0].title == "That looks like an API key"


def test_sk_slugs_are_not_keys():
    assert chat("see sk-hynix-memory-prices-are-rising-again-2026") == []
    assert chat("the sk-learn tutorial covers pipelines") == []


def test_a_wrong_tomorrow_before_a_matching_date_blames_tomorrow_not_the_weekday():
    from pilot.live.detectors import TextContext, detect

    tuesday = TextContext(today=date(2026, 11, 10), hour=12, app="Notepad", title="Notepad", source="focused_field")
    [c] = detect("Demo is tomorrow, Thursday 12 Nov", tuesday)
    assert c.title == "12 November isn't tomorrow"
    assert c.message == "Tomorrow is Wednesday 11 November."
    assert c.detail == "Thursday 12 November is the day after tomorrow."
    # The old fix, "tomorrow, Wednesday", would have made the date wrong: drop the relative word instead.
    assert (c.evidence, c.fix) == ("tomorrow, Thursday", "Thursday")
    [later] = detect("Demo is tomorrow, Thursday 19 Nov at 10", tuesday)
    assert later.detail == "Thursday 19 November is in 9 days."
    # A date that disagrees with its weekday keeps both ordinary cards.
    titles = sorted(c.title for c in detect("Demo is tomorrow, Thursday 13 Nov", tuesday))
    assert titles == ["13 November is a Friday", "Tomorrow is Wednesday"]


def test_a_weekday_with_a_day_of_the_month_and_no_month():
    from datetime import date

    from pilot.live.detectors import TextContext, detect

    def found(text, source="focused_field"):
        ctx = TextContext(today=date(2026, 11, 9), app="notepad.exe", source=source)
        return [(c.title, c.fix) for c in detect(text, ctx, kinds=("weekday_date",))]

    assert found("Could we do Tuesday the 12th instead?") == [("12 November is a Thursday", "Thursday the 12th")]
    assert found("book it for Tuesday the twelfth", "speech")[0][0] == "The twelfth of November is a Thursday"
    for fine in (
        "Thursday the 12th works",  # the coming 12th
        "the outage was on Tuesday the 3rd",  # the last 3rd
        "flights land on Friday the 4th",  # next month's 4th
        "Friday 3 pm",  # no ordinal: not a date
        "see you Tuesday 12 people",
    ):
        assert found(fine) == [], fine
