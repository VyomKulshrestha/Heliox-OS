"""macOS's own voice: a `say` that fails is reported as a failure (once in the log, in plain words), never as
"Spoken". A fake `say` stands in; nothing is spoken and no macOS is needed."""

from __future__ import annotations

import asyncio
import logging

import pytest

from pilot.system import voice


class FakeProcess:
    def __init__(self, code: int, err: bytes = b"") -> None:
        self.returncode = code
        self._err = err

    async def communicate(self):
        return b"", self._err

    def kill(self):
        return None


@pytest.fixture(autouse=True)
def fresh_warning(monkeypatch):
    monkeypatch.setattr(voice, "_SAY_WARNED", False)


def fake_say(monkeypatch, code: int, err: bytes = b""):
    calls = []

    async def create(*cmd, **_kwargs):
        calls.append(cmd)
        return FakeProcess(code, err)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    return calls


@pytest.mark.asyncio
async def test_a_say_that_fails_is_a_failure_said_once(monkeypatch, caplog):
    calls = fake_say(monkeypatch, 1, b"Open output file failed: -50\n")
    with caplog.at_level(logging.WARNING, logger="pilot.system.voice"):
        first = await voice._tts_macos("The backup is done.", None, 170, None)
        second = await voice._tts_macos("Again.", None, 170, None)
    assert first.startswith("TTS failed: say exited with code 1") and "-50" in first
    assert second.startswith("TTS failed") and not first.startswith("Spoken")
    assert calls and calls[0][0] == "say"
    warnings = [r for r in caplog.records if "could not speak" in r.getMessage()]
    assert len(warnings) == 1  # said once, in plain words


@pytest.mark.asyncio
async def test_saving_to_a_file_that_fails_is_not_reported_as_saved(monkeypatch):
    fake_say(monkeypatch, 2)
    result = await voice._tts_macos("Hello.", None, 170, "out.aiff")
    assert result.startswith("TTS failed") and "saved" not in result


@pytest.mark.asyncio
async def test_a_say_that_works_is_spoken(monkeypatch):
    fake_say(monkeypatch, 0)
    assert (await voice._tts_macos("Hello.", "Samantha", 170, None)).startswith("Spoken")


@pytest.mark.asyncio
async def test_say_missing_is_a_failure(monkeypatch):
    async def missing(*_cmd, **_kwargs):
        raise FileNotFoundError("say")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    assert (await voice._tts_macos("Hello.", None, 170, None)).startswith("TTS failed: say could not start")
