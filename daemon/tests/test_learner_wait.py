"""While the learner loads, the risk gate first holds its decision for it, bounded, and says one line once; only a
learner still loading after that gets the extra approval. Without a keyring the snapshot key lives in an owner-only
file, so a machine without one does not replay the whole ledger at every start."""

from __future__ import annotations

import asyncio
import os
import stat

import pytest

from pilot.actions import Action, ActionPlan, ActionType, FileParams
from pilot.config import PilotConfig
from pilot.intelligence.online_learning import VerifiedOnlineLearner
from pilot.system import late_start
from pilot.system.key_file import KeyFileRefused, load_or_create
from pilot.system.late_start import LearnerLoading

LIMIT = 30.0  # failure limit only


def _plan() -> ActionPlan:
    return ActionPlan(
        actions=[Action(action_type=ActionType.FILE_WRITE, target="notes.txt", parameters=FileParams(path="notes.txt"))]
    )


@pytest.fixture
def gate(monkeypatch):
    from pilot.security.risk_gate import RiskGate
    from pilot.security.risk_observation import OsSnapshot

    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(proc_count=100, disk_usage_fraction=0.5, memory_usage_fraction=0.5, disk_path="/"),
    )
    return RiskGate(weights_path="/nonexistent.npz")


async def test_a_saved_state_that_loads_during_the_wait_gives_the_normal_decision(gate, tmp_path):
    from pilot.security.risk_gate import LEARNER_LOADING_REASON

    loading = LearnerLoading()
    loading.expect(0.6)  # a saved state with little after it
    gate.set_online_learner(loading)
    loop = asyncio.get_running_loop()
    said: list[str] = []
    learner = VerifiedOnlineLearner(tmp_path / "state.json")

    def finish_loading():
        gate.set_online_learner(learner)  # as the server does: install, then release the waiting decision
        loading.set_ready()

    def on_wait():  # the moment the gate starts waiting, the load finishes: ordered, not timed
        said.append(late_start.WAITING_LINE)
        loop.call_soon_threadsafe(finish_loading)

    loading.on_wait = on_wait
    risk, reasons = await asyncio.wait_for(
        asyncio.to_thread(gate.evaluate_plan, _plan(), PilotConfig(), source="interactive"), LIMIT
    )
    assert LEARNER_LOADING_REASON not in reasons  # the learner's own (empty) view: no extra approval
    assert said == ["One moment, Heliox is still starting."]


async def test_a_learner_still_loading_after_the_bounded_wait_adds_the_approval(gate, monkeypatch):
    from pilot.agents.destructive_critic import HEURISTIC_RISK_THRESHOLD
    from pilot.security.risk_gate import LEARNER_LOADING_REASON

    monkeypatch.setattr(late_start, "LEARNER_WAIT_SECONDS", 0.2)  # the bound, shortened for the test
    loading = LearnerLoading()  # how long it needs is not known yet: wait the whole bound
    said: list[str] = []
    loading.on_wait = lambda: said.append("line")
    gate.set_online_learner(loading)
    risk, reasons = await asyncio.wait_for(
        asyncio.to_thread(gate.evaluate_plan, _plan(), PilotConfig(), source="interactive"), LIMIT
    )
    assert risk >= HEURISTIC_RISK_THRESHOLD and reasons == [LEARNER_LOADING_REASON]
    assert said == ["line"]  # said once while it waited
    await asyncio.to_thread(gate.evaluate_plan, _plan(), PilotConfig(), source="interactive")
    assert said == ["line"]  # and not again


async def test_a_full_replay_that_cannot_finish_in_the_window_is_not_waited_for(gate):
    from pilot.security.risk_gate import LEARNER_LOADING_REASON

    loading = LearnerLoading()
    loading.expect(3600.0)  # a whole ledger to replay
    said: list[str] = []
    loading.on_wait = lambda: said.append("line")
    gate.set_online_learner(loading)
    risk, reasons = await asyncio.wait_for(
        asyncio.to_thread(gate.evaluate_plan, _plan(), PilotConfig(), source="interactive"), LIMIT
    )
    assert reasons == [LEARNER_LOADING_REASON] and said == []  # straight to the approval, no pointless wait


async def test_on_the_event_loop_itself_the_gate_never_blocks():
    loading = LearnerLoading()
    assert loading.wait_until_ready(0.0, window=60.0) is False  # would hold up the very loading it waits for


class _Ledger:
    def __init__(self, tip):
        self.tip = tip

    async def last_sequence(self):
        return self.tip

    async def list_events(self, *, after_sequence=0, limit=200, **_filters):
        return []


async def test_the_learner_tells_the_gate_how_long_it_needs(tmp_path):
    from pilot.intelligence.online_learning import LOAD_OVERHEAD_SECONDS, REPLAY_EVENTS_PER_SECOND

    estimates: list[float] = []
    learner = VerifiedOnlineLearner(tmp_path / "state.json")
    loaded = await learner.initialize(_Ledger(40_000), expect=estimates.append)
    assert loaded["to_read"] == 40_000
    assert estimates == [LOAD_OVERHEAD_SECONDS + 40_000 / REPLAY_EVENTS_PER_SECOND]


# ── the key file, when there is no keyring ──


def test_the_key_file_is_made_owner_only_and_used_again(tmp_path):
    path = tmp_path / "runtime" / "learner_snapshot_key"
    first = load_or_create(path)
    assert len(first) == 64 and load_or_create(path) == first
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="Windows keeps other users out of the profile folder by its ACLs")
def test_a_key_file_others_can_read_is_refused(tmp_path):
    path = tmp_path / "learner_snapshot_key"
    load_or_create(path)
    os.chmod(path, 0o644)
    with pytest.raises(KeyFileRefused, match="can be read or changed by other users.*chmod 600"):
        load_or_create(path)


def test_a_damaged_key_file_is_refused_not_rewritten(tmp_path):
    path = tmp_path / "learner_snapshot_key"
    path.write_text("short", encoding="utf-8")
    if os.name == "posix":
        os.chmod(path, 0o600)
    with pytest.raises(KeyFileRefused, match="damaged"):
        load_or_create(path)
    assert path.read_text(encoding="utf-8") == "short"


async def test_without_a_keyring_the_server_uses_the_key_file(monkeypatch, tmp_path):
    from pilot.server import KEYS_FOLDER, LEARNER_SNAPSHOT_KEY_FILE, PilotServer

    monkeypatch.setattr("pilot.config.RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr("pilot.server.DATA_DIR", tmp_path / "data")
    server = PilotServer(PilotConfig())
    server._vault = None  # no system keyring
    key = await server._learner_key()
    assert key is not None and (tmp_path / "data" / KEYS_FOLDER / LEARNER_SNAPSHOT_KEY_FILE).exists()
    assert await server._learner_key() == key  # the same key next start: the saved state stays usable
