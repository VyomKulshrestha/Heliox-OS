"""Read handlers that changed state unasked, hid part of their answer or passed a failure off as data. Fakes only."""

from __future__ import annotations

import json

import pytest

from pilot.actions import Action, ActionType, EmailParams

# ── email_fetch ──────────────────────────────────────────────────────


class _Mailbox:
    """An IMAP server that sets \\Seen the way a real one does: on a non-PEEK body fetch, or on STORE +FLAGS."""

    def __init__(self, messages: dict[bytes, bytes]) -> None:
        self.messages = messages
        self.seen: set[bytes] = set()
        self.specs: list[str] = []

    def client(self):
        mailbox = self

        class _IMAP:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                pass

            def login(self, *_args: object) -> None:
                pass

            def select(self, *_args: object) -> None:
                pass

            def search(self, *_args: object):
                unseen = [uid for uid in mailbox.messages if uid not in mailbox.seen]
                return "OK", [b" ".join(unseen)]

            def fetch(self, uid: bytes, spec: str):
                mailbox.specs.append(spec)
                if "PEEK" not in spec.upper():
                    mailbox.seen.add(uid)
                return "OK", [(uid + b" (BODY[] {1}", mailbox.messages[uid])]

            def store(self, uid: bytes, command: str, flag: str) -> None:
                if command == "+FLAGS" and "Seen" in flag:
                    mailbox.seen.add(uid)

            def logout(self) -> None:
                pass

        return _IMAP


def _message(subject: str, body: bytes = b"hello") -> bytes:
    return b"From: a@example.com\r\nSubject: " + subject.encode() + b"\r\n\r\n" + body


async def _fetch(monkeypatch: pytest.MonkeyPatch, mailbox: _Mailbox, **params: object):
    from pilot.agents import email_agent

    monkeypatch.setattr(email_agent.imaplib, "IMAP4_SSL", mailbox.client())
    agent = object.__new__(email_agent.EmailAgent)
    fetch = EmailParams(imap_host="imap.invalid", username="u", app_password="p", **params)
    action = Action(action_type=ActionType.EMAIL_FETCH, target="mail", parameters=fetch)
    return await agent._fetch_emails(action, fetch)


@pytest.mark.asyncio
async def test_email_fetch_leaves_mail_unread_unless_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox = _Mailbox({b"1": _message("one"), b"2": _message("two")})

    result = await _fetch(monkeypatch, mailbox)

    assert result.success is True
    assert mailbox.seen == set()
    assert mailbox.specs and all("BODY.PEEK[]" in spec for spec in mailbox.specs)


@pytest.mark.asyncio
async def test_email_fetch_marks_mail_read_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    mailbox = _Mailbox({b"1": _message("one"), b"2": _message("two")})

    await _fetch(monkeypatch, mailbox, mark_as_read=True)

    assert mailbox.seen == {b"1", b"2"}


@pytest.mark.asyncio
async def test_email_fetch_output_says_what_was_left_out_and_its_readers_take_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pilot.actions import CalendarParams
    from pilot.agents import email_agent

    mailbox = _Mailbox({str(n).encode(): _message(f"m{n}") for n in range(1, 6)})
    result = await _fetch(monkeypatch, mailbox, max_emails=2)

    fetched = json.loads(result.output)
    assert fetched["unread_total"] == 5 and fetched["fetched"] == 2
    assert fetched["left_out"] == {"older_unread": 3, "could_not_fetch": 0}
    assert [m["subject"] for m in fetched["messages"]] == ["m5", "m4"]
    assert "Only the newest 2 of 5 unread messages were fetched." in fetched["notes"]

    agent = object.__new__(email_agent.EmailAgent)
    agent._model = None
    summary_params = EmailParams(emails_json=result.output)
    summary = await agent._summarize_emails(
        Action(action_type=ActionType.EMAIL_SUMMARIZE, parameters=summary_params), summary_params
    )
    assert summary.success and "Subject: m5" in summary.output and "Note: Only the newest 2 of 5" in summary.output

    old_list = json.dumps(fetched["messages"])
    again = EmailParams(emails_json=old_list)
    assert (
        await agent._summarize_emails(Action(action_type=ActionType.EMAIL_SUMMARIZE, parameters=again), again)
    ).success
    calendar = CalendarParams(emails_json=result.output)
    events = await agent._fetch_calendar_events(
        Action(action_type=ActionType.CALENDAR_FETCH, parameters=calendar), calendar
    )
    assert events.success


# ── git_diff ─────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path, monkeypatch: pytest.MonkeyPatch):
    import shutil
    import subprocess

    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    work = tmp_path / "work"
    work.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=work, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "Heliox Test")
    git("config", "user.email", "heliox-test@example.invalid")
    git("config", "commit.gpgsign", "false")
    (work / "a.txt").write_bytes(b"one\n")
    git("add", "a.txt")
    git("commit", "-m", "start")
    return work, git


@pytest.mark.asyncio
async def test_git_diff_of_unstaged_shows_the_staged_changes_when_that_is_all_there_is(repo) -> None:
    from pilot.agents.git_agent import GitAgent

    work, git = repo
    (work / "a.txt").write_bytes(b"one\ntwo\n")
    git("add", "a.txt")
    (work / "new.txt").write_bytes(b"x")
    agent = GitAgent(repo_path=str(work))

    unstaged = await agent._handle_diff({"staged": False})
    assert unstaged["output"].startswith("Nothing unstaged; 1 file(s) staged:")
    assert "+two" in unstaged["output"] and "1 untracked file(s)" in unstaged["output"]
    assert unstaged["diff"] == ""

    (work / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    both = await agent._handle_diff({"staged": True})
    assert "+two" in both["diff"] and "Also 1 file(s) with unstaged changes" in both["output"]


@pytest.mark.asyncio
async def test_git_diff_of_a_clean_tree_says_both_sides_are_empty(repo) -> None:
    from pilot.agents.git_agent import GitAgent

    work, _git = repo
    result = await GitAgent(repo_path=str(work))._handle_diff({"staged": False})
    assert result["output"] == "(no changes: nothing staged or unstaged)"


# ── calendar_fetch ───────────────────────────────────────────────────


def _invite(start, summary: str, description: str = "") -> str:
    stamp = start.strftime("%Y%m%dT%H%M%SZ")
    return (
        "BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\n"
        f"UID:{summary}\r\nSUMMARY:{summary}\r\nDTSTART:{stamp}\r\nDESCRIPTION:{description}\r\n"
        "END:VEVENT\r\nEND:VCALENDAR\r\n"
    )


@pytest.mark.asyncio
async def test_calendar_fetch_reads_the_whole_invite_and_says_what_is_outside_its_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timedelta, timezone
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from pilot.actions import CalendarParams
    from pilot.agents import email_agent

    now = datetime.now(timezone.utc)
    soon = _invite(now + timedelta(hours=2), "soon", "x" * 3000)  # longer than the 2,000-character body cut
    later = _invite(now + timedelta(days=3), "later")
    mail = MIMEMultipart("mixed")
    mail["Subject"] = "Invitation"
    mail.attach(MIMEText("y" * 2500, "plain"))
    mail.attach(MIMEText(soon, "calendar"))
    mail.attach(MIMEText(later, "calendar"))
    mailbox = _Mailbox({b"1": mail.as_bytes()})
    fetched = await _fetch(monkeypatch, mailbox)

    agent = object.__new__(email_agent.EmailAgent)
    params = CalendarParams(emails_json=fetched.output, lookahead_hours=24)
    result = await agent._fetch_calendar_events(
        Action(action_type=ActionType.CALENDAR_FETCH, parameters=params), params
    )

    data = json.loads(result.output)
    assert [e["summary"] for e in data["events"]] == ["soon"]
    assert len(data["events"][0]["description"]) == 3000
    assert data["outside_window"] == 1 and "outside the next 24 hours" in result.notice


@pytest.mark.asyncio
async def test_calendar_fetch_says_an_invite_cut_in_the_body_and_fails_on_text_that_is_not_mail() -> None:
    from pilot.actions import CalendarParams
    from pilot.agents import email_agent

    agent = object.__new__(email_agent.EmailAgent)
    cut = json.dumps([{"subject": "s", "body": "BEGIN:VEVENT\r\nSUMMARY:half", "body_cut": "cut"}])
    params = CalendarParams(emails_json=cut)
    result = await agent._fetch_calendar_events(
        Action(action_type=ActionType.CALENDAR_FETCH, parameters=params), params
    )
    assert result.success and "1 invitation(s) were cut short" in result.notice

    bad = CalendarParams(emails_json="not json")
    failed = await agent._fetch_calendar_events(Action(action_type=ActionType.CALENDAR_FETCH, parameters=bad), bad)
    assert failed.success is False and "Invalid emails_json" in (failed.error or "")


# ── system information without psutil ───────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["memory_usage", "cpu_usage", "disk_usage", "network_info", "battery_info"])
async def test_a_failed_fallback_command_fails_the_read(monkeypatch: pytest.MonkeyPatch, probe: str) -> None:
    import sys
    from unittest.mock import AsyncMock

    from pilot.system import sysinfo
    from pilot.system.platform_detect import Platform

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sysinfo, "CURRENT_PLATFORM", Platform.LINUX)
    monkeypatch.setattr(sysinfo, "run_command", AsyncMock(return_value=(-1, "", "Command not found: lscpu")))

    with pytest.raises(sysinfo.SystemReadError, match="without psutil: Command not found"):
        await getattr(sysinfo, probe)()

    monkeypatch.setattr(sysinfo, "run_command", AsyncMock(return_value=(0, "  ", "")))
    with pytest.raises(sysinfo.SystemReadError, match="answered nothing"):
        await getattr(sysinfo, probe)()

    monkeypatch.setattr(sysinfo, "run_command", AsyncMock(return_value=(0, "a reading", "")))
    assert (await getattr(sysinfo, probe)()).endswith("a reading")


@pytest.mark.asyncio
async def test_system_info_says_which_sections_it_could_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from unittest.mock import AsyncMock

    from pilot.system import sysinfo
    from pilot.system.platform_detect import Platform

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sysinfo, "CURRENT_PLATFORM", Platform.WINDOWS)
    monkeypatch.setattr(sysinfo, "run_powershell", AsyncMock(return_value=(1, "", "Access is denied.")))

    with pytest.raises(sysinfo.SystemReadError, match="Access is denied"):
        await sysinfo.system_info(["memory", "disk"])  # every section failed: no answer at all

    async def memory_only(script: str):
        return (0, "Total: 16 GB", "") if "Win32_OperatingSystem" in script else (1, "", "Access is denied.")

    monkeypatch.setattr(sysinfo, "run_powershell", memory_only)
    text = await sysinfo.system_info(["memory", "disk"])
    assert "Total: 16 GB" in text and "=== Disk ===\n  Not read:" in text
    assert text.notice == "Not read: disk (the system command failed)."


# ── log_analyze ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_analyze_timeline_shows_the_newest_events_and_says_how_many(tmp_path, monkeypatch) -> None:
    import pilot.tools.forensics as forensics
    from pilot.actions import LogAnalyzeParams
    from pilot.agents.executor import Executor

    log = tmp_path / "app.log"
    log.write_bytes(b"x\n")
    events = [{"timestamp": f"t{i:02}", "process": "app", "message": f"event {i}"} for i in range(40)]
    monkeypatch.setattr(forensics, "parse_syslog_file", lambda **_kwargs: events)
    monkeypatch.setattr(forensics, "extract_auth_events", lambda _events: [])
    monkeypatch.setattr(forensics, "detect_spikes", lambda _events: [])
    monkeypatch.setattr(forensics, "correlate_sessions", lambda _events: [])
    executor = object.__new__(Executor)
    executor._model = None
    action = Action(action_type=ActionType.LOG_ANALYZE, parameters=LogAnalyzeParams(log_path=str(log)))

    report = json.loads(await executor._exec_log_analyze(action))

    assert report["timeline"][0].startswith("[t25]") and report["timeline"][-1].startswith("[t39]")
    assert report["events_total"] == 40
    assert report["timeline_note"] == "The timeline shows the newest 15 of 40 events."
    assert "over 40 events" in report["summary"]
