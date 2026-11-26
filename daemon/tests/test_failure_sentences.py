"""Common system failures become a plain sentence that names the thing and says what a person can do, on Windows,
macOS and Linux; the ones asking again cannot change end the run (the terminal list)."""

from __future__ import annotations

import pytest

from pilot.actions import Action, ActionResult, ActionType, ApiRequestParams, FileParams
from pilot.failure_sentences import FINAL_KINDS, FailureKind, classify, failure_sentence, is_final
from pilot.reply_rules import ear_faults

WIN = r"C:\Users\me\work\readonly\blocked.txt"
NIX = "/home/me/work/readonly/blocked.txt"


def _write(path: str) -> Action:
    return Action(action_type=ActionType.FILE_WRITE, target=path, parameters=FileParams(path=path, content="x"))


def _get(url: str = "https://example.com/a") -> Action:
    return Action(action_type=ActionType.API_REQUEST, target=url, parameters=ApiRequestParams(url=url))


@pytest.mark.parametrize(
    ("platform", "error", "kind"),
    [
        # Windows: the C runtime's errno for file calls, WinError for the rest
        ("win32", f"[Errno 13] Permission denied: {WIN!r}", FailureKind.NO_PERMISSION),
        ("win32", f"[WinError 5] Access is denied: {WIN!r}", FailureKind.NO_PERMISSION),
        ("win32", f"[WinError 19] The media is write protected: {WIN!r}", FailureKind.READ_ONLY),
        ("win32", f"[WinError 2] The system cannot find the file specified: {WIN!r}", FailureKind.NOT_FOUND),
        ("win32", f"[WinError 183] Cannot create a file when that file already exists: {WIN!r}",
         FailureKind.ALREADY_EXISTS),
        ("win32", f"[WinError 32] The process cannot access the file because it is being used by another process: "
         f"{WIN!r}", FailureKind.IN_USE),
        ("win32", "[WinError 112] There is not enough space on the disk", FailureKind.DISK_FULL),
        ("win32", "[WinError 206] The filename or extension is too long", FailureKind.PATH_TOO_LONG),
        ("win32", "[WinError 10051] A socket operation was attempted to an unreachable network", FailureKind.NO_NETWORK),
        ("win32", "[Errno 11001] getaddrinfo failed", FailureKind.NAME_NOT_RESOLVED),
        ("win32", "[WinError 10060] A connection attempt failed", FailureKind.TIMED_OUT),
        ("win32", "[WinError 10061] No connection could be made because the target machine actively refused it",
         FailureKind.REFUSED),
        # Linux
        ("linux", f"[Errno 13] Permission denied: {NIX!r}", FailureKind.NO_PERMISSION),
        ("linux", f"[Errno 30] Read-only file system: {NIX!r}", FailureKind.READ_ONLY),
        ("linux", f"[Errno 2] No such file or directory: {NIX!r}", FailureKind.NOT_FOUND),
        ("linux", f"[Errno 17] File exists: {NIX!r}", FailureKind.ALREADY_EXISTS),
        ("linux", f"[Errno 16] Device or resource busy: {NIX!r}", FailureKind.IN_USE),
        ("linux", "[Errno 28] No space left on device", FailureKind.DISK_FULL),
        ("linux", "[Errno 36] File name too long", FailureKind.PATH_TOO_LONG),
        ("linux", "[Errno 101] Network is unreachable", FailureKind.NO_NETWORK),
        ("linux", "[Errno -2] Name or service not known", FailureKind.NAME_NOT_RESOLVED),
        ("linux", "[Errno 110] Connection timed out", FailureKind.TIMED_OUT),
        ("linux", "[Errno 111] Connection refused", FailureKind.REFUSED),
        # macOS: the same names, other numbers
        ("darwin", f"[Errno 13] Permission denied: {NIX!r}", FailureKind.NO_PERMISSION),
        ("darwin", f"[Errno 30] Read-only file system: {NIX!r}", FailureKind.READ_ONLY),
        ("darwin", "[Errno 63] File name too long", FailureKind.PATH_TOO_LONG),
        ("darwin", "[Errno 51] Network is unreachable", FailureKind.NO_NETWORK),
        ("darwin", "[Errno 8] nodename nor servname provided, or not known", FailureKind.NAME_NOT_RESOLVED),
        ("darwin", "[Errno 60] Operation timed out", FailureKind.TIMED_OUT),
        ("darwin", "[Errno 61] Connection refused", FailureKind.REFUSED),
        ("darwin", "[Errno 69] Disc quota exceeded", FailureKind.DISK_FULL),
        # words only
        ("linux", "The mail server refused the sign-in for ann@example.com", FailureKind.SIGN_IN_REFUSED),
        ("win32", "httpx.ConnectTimeout: timed out", FailureKind.TIMED_OUT),
    ],
)  # fmt: skip
def test_each_platforms_numbers_and_words(platform: str, error: str, kind: FailureKind) -> None:
    assert classify(error, platform=platform) is kind


def test_the_same_number_means_what_its_platform_says() -> None:
    # 36 is ENAMETOOLONG on Linux; on macOS it is EINPROGRESS, which is none of the common failures.
    assert classify("[Errno 36] File name too long", platform="linux") is FailureKind.PATH_TOO_LONG
    assert classify("[Errno 36] Operation now in progress", platform="darwin") is None
    assert classify("[Errno 61] Connection refused", platform="linux") is FailureKind.REFUSED  # by its words
    assert classify("[Errno 61] No data available", platform="linux") is None


@pytest.mark.parametrize(
    ("action", "error", "spoken"),
    [
        (_write(WIN), f"[Errno 13] Permission denied: {WIN!r}",
         "I couldn't write blocked.txt in readonly: permission was denied. You can change its permissions, or choose "
         "another place."),
        (_write(NIX), f"[Errno 30] Read-only file system: {NIX!r}",
         "I couldn't write blocked.txt in readonly: that place is read-only. Choose a folder you can write to."),
        (_write(WIN), f"[WinError 32] being used by another process: {WIN!r}",
         "I couldn't write blocked.txt in readonly: another program is using it. Close that program and ask me again."),
        (_write(NIX), "[Errno 28] No space left on device",
         "I couldn't write blocked.txt in readonly: the disk is full. Free some space and ask me again."),
        (_get(), "[Errno -2] Name or service not known",
         "I couldn't reach example.com: that address could not be found. Check its spelling and your connection."),
        (_get(), "[Errno 111] Connection refused",
         "I couldn't reach example.com: the connection was refused. Check that the service is running."),
        (_get(), "ReadTimeout: timed out",
         "I couldn't reach example.com: it took too long to answer. You can ask me again in a moment."),
    ],
)  # fmt: skip
def test_the_sentence_names_the_thing_and_what_to_do(action: Action, error: str, spoken: str) -> None:
    shown, said = failure_sentence(action, error, platform="linux" if "Errno -2" in error else None)
    assert said == spoken
    assert ear_faults(said) == []
    assert "Errno" not in shown and "\\\\" not in shown


def test_an_unknown_failure_is_shown_as_written_and_not_said() -> None:
    shown, said = failure_sentence(_write(WIN), "Process exited with code 125: C:\\\\tmp\\\\x")
    assert shown == "I couldn't write blocked.txt in readonly: Process exited with code 125: C:\\tmp\\x"
    assert said == "I couldn't write blocked.txt in readonly. The details are on screen."


def test_what_is_final() -> None:
    assert {
        FailureKind.NO_PERMISSION,
        FailureKind.READ_ONLY,
        FailureKind.SIGN_IN_REFUSED,
        FailureKind.NOT_FOUND,
    } == FINAL_KINDS
    assert is_final(_write(WIN), f"[Errno 13] Permission denied: {WIN!r}")
    assert is_final(_write(NIX), f"[Errno 30] Read-only file system: {NIX!r}", platform="linux")
    assert not is_final(_write(WIN), "[WinError 32] being used by another process")
    assert not is_final(_get(), "[Errno 110] Connection timed out", platform="linux")
    # Not found is final only when the request named the missing thing; a path the planner guessed may be found.
    missing = f"Source not found: {WIN}"
    assert is_final(_write(WIN), missing, "copy blocked.txt to the desktop")
    assert not is_final(_write(WIN), missing, "copy my draft to the desktop")


def test_the_terminal_list_takes_the_final_failures() -> None:
    from pilot.server import _is_terminal_execution_failure

    denied = ActionResult(action=_write(WIN), success=False, error=f"[Errno 13] Permission denied: {WIN!r}")
    busy = ActionResult(action=_write(WIN), success=False, error="[WinError 32] being used by another process")
    missing = ActionResult(action=_write(WIN), success=False, error=f"File not found: {WIN}")

    assert _is_terminal_execution_failure([denied]) is True
    assert _is_terminal_execution_failure([busy]) is False
    assert _is_terminal_execution_failure([denied, busy]) is False  # the busy one may still be repaired
    assert _is_terminal_execution_failure([missing], "write to blocked.txt") is True
    assert _is_terminal_execution_failure([missing], "write to the usual file") is False
