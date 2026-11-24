"""A change of voice: the new one loads at once, the voice the person hears keeps speaking, and the switch comes
between two replies (local_tts_worker._begin_switch, voice_for_reply). Made-up machines and managers only: no worker
process is started, no model loads, the clock is the test's own."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pilot.system import local_tts_worker as ltw
from pilot.system import voice

GB = 1024**3


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Running:
    def poll(self):
        return None


def _manager(clock, *, engine="kokoro_tts", loaded=True, free=8 * GB, lease=False, spoke=True):
    made = ltw.LocalTTSWorkerManager(spawn=MagicMock(), clock=clock)
    if engine:
        made.process, made.engine, made.loaded = Running(), engine, loaded
        made.process_device = made.device
        made.reload_log.append(("load_ended", engine, 0.0))
    made.gpu_lease = lease
    made.free_memory = lambda: free
    made.on_status = MagicMock(name="window status")
    if not spoke:
        made.reload_log.clear()
    made.asked: list[tuple[str, str]] = []
    made._start_prewarm = lambda e, v: made.asked.append((e, v))
    made.closed = 0

    async def close_when_free():
        made.closed += 1

    made.close_when_free = close_when_free
    return made


@pytest.fixture
def machine(monkeypatch):
    clock = Clock()
    old = _manager(clock)
    monkeypatch.setattr(ltw, "_manager", old)
    standbys: list[ltw.LocalTTSWorkerManager] = []

    def make_standby():
        made = _manager(clock, engine="", spoke=False)
        standbys.append(made)
        return made

    ltw.handover_log.clear()
    return SimpleNamespace(clock=clock, old=old, standbys=standbys, make=make_standby)


def _quiet(machine, seconds=ltw.REPLY_GAP_SECONDS + 1):
    machine.old._spoke_until = machine.clock.now
    machine.clock.now += seconds


def test_with_memory_to_spare_the_new_voice_loads_beside_the_one_speaking(machine):
    done = ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    standby = machine.standbys[0]
    assert done == "standby" and standby.asked == [("pocket_tts", "alba")]  # loading at once, not at the next reply
    assert standby.on_status is None  # the window keeps saying the voice is ready
    assert ltw.voice_for_reply("pocket_tts", "alba") == ("kokoro_tts", "af_heart")  # the old one keeps speaking
    assert ltw.manager() is machine.old and machine.old.closed == 0


def test_the_switch_waits_for_the_new_voice_and_for_the_reply_to_end(machine):
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    standby = machine.standbys[0]
    standby.process, standby.engine = Running(), "pocket_tts"
    _quiet(machine)
    assert ltw.finish_handover_if_due() is False  # quiet, but the new voice is still loading
    standby.loaded = True
    machine.old._busy = 1  # a reply is being spoken
    assert ltw.voice_for_reply("pocket_tts", "alba") == ("kokoro_tts", "af_heart")
    machine.old._busy = 0
    machine.old.note_spoken()  # its last piece just ended: still the same reply
    assert ltw.finish_handover_if_due() is False
    machine.clock.now += ltw.REPLY_GAP_SECONDS + 0.1
    assert ltw.voice_for_reply("pocket_tts", "alba") == ("pocket_tts", "alba")
    assert ltw.manager() is standby and ltw.handover() is None
    assert standby.on_status is machine.old.on_status  # the window hears from the new voice from now on
    assert [entry[0] for entry in ltw.handover_log] == ["standby", "switched"]


@pytest.mark.asyncio
async def test_the_old_voice_is_released_after_the_switch(machine):
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    standby = machine.standbys[0]
    standby.process, standby.engine, standby.loaded = Running(), "pocket_tts", True
    _quiet(machine)
    shut: list[str] = []

    async def shutdown():
        shut.append("old")

    machine.old.shutdown = shutdown
    assert ltw.finish_handover_if_due() is True
    for _ in range(3):
        await asyncio.sleep(0)
    assert machine.old.closed == 1 and shut == ["old"]


@pytest.mark.parametrize(
    ("free", "lease", "said"),
    [
        (
            1 * GB,
            False,
            "1.0 GB of memory is free, and loading pocket_tts beside the running voice needs 2.1 GB",
        ),  # tight memory
        (None, False, "the free memory could not be read"),
    ],
)
def test_without_room_for_two_the_old_voice_is_released_first_and_the_log_says_why(machine, caplog, free, lease, said):
    machine.old.free_memory = lambda: free
    machine.old.gpu_lease = lease
    with caplog.at_level("INFO", logger="pilot.system.local_tts_worker"):
        done = ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    assert done == "release_first" and machine.standbys == [] and ltw.handover() is None
    assert said in caplog.text and "the system voice speaks meanwhile" in caplog.text


def test_a_card_that_holds_one_voice_releases_the_old_one_first(machine, caplog):
    """Two voices that both want the card: it holds one, by its lease, whatever the memory."""
    machine.old.engine, machine.old.gpu_lease = "card_engine", True
    machine.old.environ = {}  # this made-up machine shows its card (the test run itself hides every card)
    with caplog.at_level("INFO", logger="pilot.system.local_tts_worker"):
        done = ltw._begin_switch("kokoro_tts", "af_heart", "x", make_standby=machine.make)
    assert done == "release_first" and "the graphics card holds one voice at a time" in caplog.text
    # Pocket runs on the processor: it loads beside a voice on the card, given the memory.
    machine.old.engine = "kokoro_tts"
    assert ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make) == "standby"


@pytest.mark.asyncio
async def test_released_first_the_new_voice_loads_once_the_old_one_is_closed(machine):
    machine.old.free_memory = lambda: 1 * GB
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    for _ in range(3):
        await asyncio.sleep(0)
    assert machine.old.closed == 1 and machine.old.asked == [("pocket_tts", "alba")]


def test_a_standby_that_fails_takes_over_so_the_usual_retry_follows(machine):
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    machine.standbys[0]._failed_until = machine.clock.now + ltw.RETRY_AFTER_SECONDS
    machine.old._busy = 1  # even inside a reply: the old voice never stays for good
    assert ltw.finish_handover_if_due() is True and ltw.manager() is machine.standbys[0]


def test_an_old_voice_that_stopped_hands_over_at_once(machine):
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    machine.old.process = None  # the old worker crashed: neither is ready, so the system voice speaks meanwhile
    assert ltw.voice_for_reply("pocket_tts", "alba") == ("pocket_tts", "alba") and ltw.manager() is machine.standbys[0]


def test_a_second_change_drops_the_first_standby(machine):
    stopped: list[str] = []
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    machine.standbys[0].shutdown = lambda: stopped.append("first") or asyncio.sleep(0)
    ltw._begin_switch("kokoro_tts", "am_michael", "af_heart", make_standby=machine.make)
    assert ltw.handover() is None and stopped == ["first"]  # back to Kokoro: the running worker serves it ("same")


def test_nothing_loads_at_once_when_no_local_voice_has_spoken_this_session(machine, monkeypatch):
    fresh = _manager(machine.clock, engine="", spoke=False)
    monkeypatch.setattr(ltw, "_manager", fresh)
    assert ltw._begin_switch("pocket_tts", "alba", "", make_standby=machine.make) == "later" and fresh.asked == []
    fresh.reload_log.append(("load_ended", "kokoro_tts", 0.0))  # it spoke, then was released while idle
    assert ltw._begin_switch("pocket_tts", "alba", "", make_standby=machine.make) == "load"
    assert fresh.asked == [("pocket_tts", "alba")]


def test_configure_starts_the_switch_only_on_a_change_of_voice(machine, monkeypatch):
    begun: list[tuple[str, str, str]] = []
    monkeypatch.setattr(ltw, "_begin_switch", lambda e, v, f="", **_: begun.append((e, v, f)) or "standby")
    settings = SimpleNamespace(tts_engine="kokoro_tts", tts_voice="af_heart", light_on_small_memory=False)
    ltw.configure_local_voice(settings)
    assert begun == []  # the first settings of a start: nothing to change from
    ltw.configure_local_voice(settings)
    assert begun == []  # the same voice again (another voice setting changed)
    ltw.configure_local_voice(SimpleNamespace(tts_engine="pocket_tts", tts_voice="alba", light_on_small_memory=False))
    assert begun == [("pocket_tts", "alba", "af_heart")]


def test_replies_entry_points_follow_the_voice_that_speaks(machine):
    ltw._begin_switch("pocket_tts", "alba", "af_heart", make_standby=machine.make)
    settings = SimpleNamespace(tts_engine="pocket_tts", tts_voice="alba")
    prewarmed: list[tuple[str, str]] = []
    machine.old.prewarm = lambda e, v: prewarmed.append((e, v))
    ltw.prewarm_local_voice(settings)
    assert prewarmed == [("kokoro_tts", "af_heart")]  # never a third load of the new voice on the old manager
    assert ltw.local_voice_state(settings) == {"voice_loaded": True, "voice_loading": False}


@pytest.mark.asyncio
async def test_a_reply_speaks_with_the_old_voice_while_the_new_one_loads(monkeypatch):
    spoken: list[tuple[str, str]] = []

    async def run(engine, text, voice_name, output_file, *, wait_for_load=False):
        spoken.append((engine, voice_name))

    monkeypatch.setattr(voice, "_run_local_tts_worker", run)
    monkeypatch.setattr(voice, "_speech_config", lambda: SimpleNamespace(voice=SimpleNamespace(
        tts_engine="pocket_tts", tts_voice="alba", system_voice="")))  # fmt: skip
    monkeypatch.setattr(ltw, "voice_for_reply", lambda engine, name: ("kokoro_tts", "af_heart"))
    await voice._speak_impl("It is ten past four.")
    assert spoken == [("kokoro_tts", "af_heart")]


def test_the_load_line_gives_megabytes_in_megabytes():
    assert ltw.load_steps({"waited_s": 0.3, "import_s": 1.62, "read_mb": 506.0}) == (
        "waited_s 0.3 s, import_s 1.6 s, read_mb 506.0 MB"
    )
