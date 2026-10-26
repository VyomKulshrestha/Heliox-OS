"""Watchers: "tell me when Chrome closes", downloads, new files, battery and processor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.config import PilotConfig
from pilot.workflows.watchers import GIVE_UP_AFTER_SECONDS, Probes, WatcherStore, matches_process


class FakeComputer:
    def __init__(self, downloads):
        self.procs = [(10, "chrome.exe", 1.0), (11, "chrome.exe", 2.0), (20, "Code.exe", 3.0)]
        self.dead: set[int] = set()
        self.battery_reading: tuple[float, bool] | None = (50.0, True)
        self.cpu_reading = 80.0
        self.downloads = downloads

    def probes(self) -> Probes:
        return Probes(
            processes=lambda: [row for row in self.procs if row[0] not in self.dead],
            alive=lambda pid, created: pid not in self.dead,
            battery=lambda: self.battery_reading,
            cpu=lambda: self.cpu_reading,
            downloads=self.downloads,
        )


@pytest.fixture
def computer(tmp_path):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    return FakeComputer(downloads)


@pytest.fixture
def clock():
    return MagicMock(return_value=1000.0)


@pytest.fixture
def watchers(computer, clock):
    return WatcherStore(probes=computer.probes(), deliver=AsyncMock(), clock=clock)


def test_matches_process():
    assert matches_process("chrome", "chrome.exe")
    assert matches_process("google chrome", "chrome.exe")
    assert matches_process("vs code", "Code.exe")
    assert matches_process("zoom meeting", "Zoom.exe")
    assert not matches_process("word", "chrome.exe")
    assert not matches_process("", "chrome.exe")


@pytest.mark.asyncio
async def test_app_closing(watchers, computer):
    assert watchers.answer("Tell me when Chrome closes.") == "Okay. I'll tell you when chrome closes."
    assert await watchers.tick() == 0

    computer.dead.add(10)
    assert await watchers.tick() == 0  # one of its processes is still running
    computer.dead.add(11)
    assert await watchers.tick() == 1
    watchers._deliver.assert_awaited_once()
    event = watchers._deliver.await_args.args[0]
    assert event["kind"] == "process" and event["text"] == "Chrome has closed."
    assert watchers.list() == []


def test_app_must_be_running(watchers):
    reply = watchers.answer("let me know when Photoshop quits")
    assert reply == "I can only watch apps that are running, and I can't find photoshop running right now."
    assert watchers.list() == []


@pytest.mark.asyncio
async def test_download_finishing(watchers, computer, clock):
    assert watchers.answer("tell me when my download finishes") == (
        "I don't see a download in progress in your Downloads folder."
    )
    partial = computer.downloads / "report.pdf.crdownload"
    partial.write_bytes(b"x")
    assert (
        watchers.answer("let me know when the download is done") == "Okay. I'll tell you when your download finishes."
    )

    assert await watchers.tick() == 0
    partial.unlink()
    (computer.downloads / "report.pdf").write_bytes(b"done")
    assert await watchers.tick() == 1
    assert watchers._deliver.await_args.args[0]["text"] == "Your download finished: report.pdf."


@pytest.mark.asyncio
async def test_new_file_arriving(watchers, computer):
    (computer.downloads / "old invoice.pdf").write_bytes(b"x")
    assert (
        watchers.answer("tell me when the invoice shows up") == "Okay. I'll tell you when invoice arrives in Downloads."
    )
    assert await watchers.tick() == 0  # the file that was already there does not count

    (computer.downloads / "invoice-october.pdf.part").write_bytes(b"x")
    assert await watchers.tick() == 0  # still downloading
    (computer.downloads / "invoice-october.pdf").write_bytes(b"x")
    assert await watchers.tick() == 1
    assert watchers._deliver.await_args.args[0]["text"] == "invoice-october.pdf is in your Downloads folder."


@pytest.mark.asyncio
async def test_battery_levels(watchers, computer):
    assert watchers.answer("tell me when my battery is full") == "Okay. I'll tell you when the battery is full."
    assert watchers.answer("tell me when my battery drops below 20%") == (
        "Okay. I'll tell you when the battery drops to 20%."
    )
    assert watchers.answer("tell me when the battery reaches 30 percent") == (
        "Okay. I'll tell you when the battery drops to 30%."  # it is at 50% now, so this means going down
    )
    assert await watchers.tick() == 0

    computer.battery_reading = (25.0, False)
    assert await watchers.tick() == 1
    assert watchers._deliver.await_args.args[0]["text"] == "Your battery is at 25% and not charging."

    computer.battery_reading = (99.0, True)
    assert await watchers.tick() == 1
    assert watchers._deliver.await_args.args[0]["text"] == "Your battery is full."


def test_no_battery(watchers, computer):
    computer.battery_reading = None
    assert watchers.answer("tell me when my battery is low") == "I can't read a battery on this computer."


@pytest.mark.asyncio
async def test_processor_settling(watchers, computer):
    assert watchers.answer("tell me when my computer calms down") == (
        "Okay. I'll tell you when the processor stays under 20%."
    )
    computer.cpu_reading = 5.0
    assert await watchers.tick() == 0
    assert await watchers.tick() == 0
    computer.cpu_reading = 50.0
    assert await watchers.tick() == 0  # a spike starts the count again
    computer.cpu_reading = 5.0
    assert [await watchers.tick() for _ in range(3)] == [0, 0, 1]


@pytest.mark.asyncio
async def test_list_cancel_and_give_up(watchers, clock):
    assert watchers.answer("what are you watching for").startswith("I'm not watching for anything")
    watchers.answer("tell me when chrome closes")
    watchers.answer("tell me when the CPU drops below 30 percent")
    assert watchers.answer("what are you watching for") == (
        "I'm waiting until chrome closes; the processor stays under 30%."
    )
    assert watchers.answer("stop watching chrome") == "Okay, I stopped watching."
    assert watchers.answer("stop watching zoom") == "I wasn't watching for zoom."

    clock.return_value = 1000.0 + GIVE_UP_AFTER_SECONDS + 1
    assert await watchers.tick() == 1
    assert "didn't happen within a day" in watchers._deliver.await_args.args[0]["text"]
    assert watchers.list() == []


def test_sentences_left_to_others(watchers):
    for text in (
        "let me know when it's done",
        "tell me when you finish",
        "tell me when it is 5 pm",
        "tell me when Sarah replies",
        "tell me if I spend more than an hour on YouTube today",
        "open chrome",
    ):
        assert watchers.answer(text) is None, text


def test_limit(watchers):
    for _ in range(10):
        watchers.answer("tell me when chrome closes")
    assert watchers.answer("tell me when vs code closes") == "I can watch up to 10 things at once."


@pytest.mark.asyncio
async def test_watch_rpcs_and_delivery(tmp_path, computer):
    from pilot.server import PilotServer

    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    server._broadcast_notification = AsyncMock()
    server._speak_companion_text = AsyncMock()
    server._watchers = WatcherStore(probes=computer.probes(), deliver=server._deliver_watch)
    ws = MagicMock()

    added = await server._handle_watch_add({"text": "Chrome closes"}, ws)
    assert added["status"] == "ok" and added["watcher"]["label"] == "chrome closes"
    assert (await server._handle_watch_add({"text": "Sarah replies"}, ws))["status"] == "error"
    assert [row["label"] for row in (await server._handle_watch_list({}, ws))["watchers"]] == ["chrome closes"]

    computer.dead.update({10, 11})
    await server._watchers.tick()
    server._broadcast_notification.assert_awaited_with(
        "watch_fired",
        {"id": added["watcher"]["id"], "kind": "process", "label": "chrome closes", "text": "Chrome has closed."},
    )
    assert server._speak_companion_text.await_args.args[0] == "Chrome has closed."

    again = await server._handle_watch_add({"text": "tell me when vs code closes"}, ws)
    assert (await server._handle_watch_cancel({"id": again["watcher"]["id"]}, ws))["watchers"] == []
    assert (await server._handle_watch_cancel({"id": "missing"}, ws))["status"] == "error"
