"""Disk-size lookahead in the risk gate: measured sizes, bounded scans, ordering."""

from __future__ import annotations

import ntpath
import posixpath
import sys

import pytest

from pilot.actions import (
    Action,
    ActionPlan,
    ActionType,
    EmptyParams,
    FileParams,
    ShellCommandParams,
    ShellScriptParams,
)
from pilot.agents.destructive_critic import HEURISTIC_RISK_THRESHOLD, assess_plan_risk, heuristic_risk
from pilot.config import PilotConfig
from pilot.security import risk_gate as risk_gate_module
from pilot.security.risk_gate import UNMEASURED_SIZE_RISK, RiskGate, _label, _shell_created_files
from pilot.security.risk_observation import OsSnapshot, TreeSize, VolumeUsage
from pilot.security.risk_safety import DISK_USAGE_RISK_WEIGHT

GIB = 1024**3
KIB = 1024


def _snapshot(fraction: float) -> OsSnapshot:
    return OsSnapshot(proc_count=100, disk_usage_fraction=fraction, memory_usage_fraction=0.5, disk_path="/sys")


def _gate(monkeypatch, *, fraction: float, total: int, volumes=None) -> RiskGate:
    """A rule-only gate whose every path lives on one fake volume unless
    `volumes` maps a path prefix to another VolumeUsage."""
    monkeypatch.setattr(risk_gate_module, "capture_os_snapshot", lambda: _snapshot(fraction))
    gate = RiskGate(weights_path="/nonexistent.npz")
    system = VolumeUsage(key="sys", root="C:\\", used=round(fraction * total), total=total)

    def lookup(path: str) -> VolumeUsage | None:
        for prefix, volume in (volumes or {}).items():
            if str(path).startswith(prefix):
                return volume
        return system

    gate._volume_of = lookup
    return gate


def _copy(source, destination) -> Action:
    return Action(
        action_type=ActionType.FILE_COPY,
        target=str(source),
        parameters=FileParams(path=str(source), destination=str(destination), recursive=True),
    )


def _plan(*actions: Action) -> ActionPlan:
    return ActionPlan(actions=list(actions), raw_input="test")


def _tree(root, files: int, size: int):
    root.mkdir()
    for index in range(files):
        (root / f"f{index}.bin").write_bytes(b"\0" * size)
    return root


def test_small_file_write_stays_zero_on_the_real_machine(tmp_path):
    gate = RiskGate(weights_path="/nonexistent.npz")
    target = tmp_path / "note.txt"
    action = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(target),
        parameters=FileParams(path=str(target), content="hello"),
    )
    risk, reasons = gate.evaluate_plan(_plan(action), PilotConfig())
    assert risk == 0.0
    assert reasons == []


def test_small_write_on_an_almost_full_disk_is_not_blamed(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, fraction=0.97, total=1000 * GIB)
    action = Action(
        action_type=ActionType.FILE_WRITE,
        target=str(tmp_path / "x.txt"),
        parameters=FileParams(path=str(tmp_path / "x.txt"), content="hello"),
    )
    risk, _ = gate.evaluate_plan(_plan(action), PilotConfig())
    assert risk == 0.0


def test_copy_that_fits_stays_zero_and_records_exact_size(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 3, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.5, total=1_000_000)
    risk, reasons = gate.evaluate_plan(_plan(_copy(source, tmp_path / "out")), PilotConfig())
    assert risk == 0.0
    assert reasons == []
    record = gate.last_evaluation["predictions"][0]["size_estimates"][0]
    assert record["bytes"] == 30 * KIB
    assert record["size_is_exact"] is True


def test_copy_past_the_threshold_names_the_size_and_levels(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 3, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.94, total=1_000_000)
    risk, reasons = gate.evaluate_plan(_plan(_copy(source, tmp_path / "out")), PilotConfig())
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons == ["copying photos (about 30 KB) would take C: from 94% to 97%, past the safe level of 95%"]
    assert gate.last_evaluation["prediction_sources"] == ["size_estimate"]


def test_copy_larger_than_free_space_says_it_does_not_fit(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 3, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.99, total=1_000_000)
    risk, reasons = gate.evaluate_plan(_plan(_copy(source, tmp_path / "out")), PilotConfig())
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons == ["copying photos (about 30 KB) needs more than the 9.8 KB free on C:"]


def test_unfinished_scan_is_not_treated_as_the_size(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    gate._measure_tree_size = lambda path, deadline: TreeSize(
        bytes=6 * GIB, entries=125_000, complete=False, stopped_early=True, files=120_000
    )
    risk, reasons = gate.evaluate_plan(_plan(_copy(tmp_path / "Videos", tmp_path / "Videos-copy")), PilotConfig())
    assert risk == pytest.approx(UNMEASURED_SIZE_RISK)
    assert risk >= HEURISTIC_RISK_THRESHOLD  # adds an approval
    assert reasons == [
        "Videos is too large to measure quickly (at least 6.0 GB in 120,000 files so far). "
        "C: has 265 GB free, so the copy may not fit."
    ]
    record = gate.last_evaluation["predictions"][0]["size_estimates"][0]
    assert record["size_is_exact"] is False


def test_unfinished_scan_whose_lower_bound_crosses_is_a_proven_fill(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, fraction=0.9, total=100 * GIB)
    gate._measure_tree_size = lambda path, deadline: TreeSize(
        bytes=8 * GIB, entries=50_000, complete=False, stopped_early=True
    )
    risk, reasons = gate.evaluate_plan(_plan(_copy(tmp_path / "Videos", tmp_path / "out")), PilotConfig())
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons == [
        "copying Videos (at least 8.0 GB) would take C: from 90% to at least 98%, past the safe level of 95%"
    ]


def test_unfinished_scan_of_a_source_volume_that_cannot_fill_the_target_is_quiet(monkeypatch, tmp_path):
    small_source_volume = VolumeUsage(key="usb", root="E:\\", used=2 * GIB, total=16 * GIB)
    gate = _gate(monkeypatch, fraction=0.5, total=1000 * GIB, volumes={str(tmp_path / "usb"): small_source_volume})
    gate._measure_tree_size = lambda path, deadline: TreeSize(
        bytes=GIB, entries=90_000, complete=False, stopped_early=True
    )
    risk, reasons = gate.evaluate_plan(_plan(_copy(tmp_path / "usb" / "photos", tmp_path / "out")), PilotConfig())
    assert risk == 0.0
    assert reasons == []


def test_bounded_scan_stops_and_reports_a_lower_bound_through_the_gate(monkeypatch, tmp_path):
    source = _tree(tmp_path / "many", 50, KIB)
    monkeypatch.setattr(risk_gate_module, "PLAN_SCAN_BUDGET_SECONDS", 0.0)
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    risk, reasons = gate.evaluate_plan(_plan(_copy(source, tmp_path / "out")), PilotConfig())
    record = gate.last_evaluation["predictions"][0]["size_estimates"][0]
    assert record["size_is_exact"] is False
    assert record["bytes"] < 50 * KIB
    assert risk == pytest.approx(UNMEASURED_SIZE_RISK)
    assert "may not fit" in reasons[0]


def test_missing_or_failing_source_never_raises(monkeypatch, tmp_path):
    gate = _gate(monkeypatch, fraction=0.5, total=1_000_000)
    risk, _ = gate.evaluate_plan(_plan(_copy(tmp_path / "missing", tmp_path / "out")), PilotConfig())
    assert risk == 0.0

    def broken(path, deadline):
        raise PermissionError("denied")

    gate._measure_tree_size = broken
    gate._volume_of = lambda path: (_ for _ in ()).throw(OSError("bad volume"))
    risk, _ = gate.evaluate_plan(_plan(_copy(tmp_path / "x", tmp_path / "out")), PilotConfig())
    assert risk == 0.0


def test_repeated_copies_accumulate_in_order_and_scan_once(monkeypatch, tmp_path):
    source = _tree(tmp_path / "docs", 1, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.80, total=1_000_000)
    calls = []
    real_measure = gate._measure_tree_size

    def counting(path, deadline):
        calls.append(path)
        return real_measure(path, deadline=deadline)

    gate._measure_tree_size = counting
    plan = _plan(*(_copy(source, tmp_path / f"out{i}") for i in range(20)))
    risk, reasons = gate.evaluate_plan(plan, PilotConfig())
    assert len(calls) == 1
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons[0].startswith("copying docs (about 10 KB) would take C: from 9")
    flagged = [prediction["risk_score"] for prediction in gate.last_evaluation["predictions"]]
    assert flagged[0] == 0.0 and flagged[-1] == pytest.approx(DISK_USAGE_RISK_WEIGHT)


def test_deletes_earlier_in_the_plan_do_not_hide_a_later_fill(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 3, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.94, total=1_000_000)
    deletes = [Action(action_type=ActionType.FILE_DELETE, target="/tmp/a", parameters=EmptyParams())] * 10
    risk, reasons = gate.evaluate_plan(_plan(*deletes, _copy(source, tmp_path / "out")), PilotConfig())
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert "from 94% to 97%" in reasons[0]


def test_move_within_one_volume_adds_nothing(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 3, 10 * KIB)
    gate = _gate(monkeypatch, fraction=0.94, total=1_000_000)
    move = Action(
        action_type=ActionType.FILE_MOVE,
        target=str(source),
        parameters=FileParams(path=str(source), destination=str(tmp_path / "moved")),
    )
    risk, _ = gate.evaluate_plan(_plan(move), PilotConfig())
    assert risk == 0.0


def _fsutil_action(path: str = "C:\\big.bin", size: int = 300_000_000_000) -> Action:
    return Action(
        action_type=ActionType.SHELL_SCRIPT,
        target="powershell",
        parameters=ShellScriptParams(script=f"fsutil file createnew {path} {size}"),
    )


@pytest.mark.parametrize("flavour", [ntpath, posixpath], ids=["windows-paths", "posix-paths"])
def test_fsutil_counts_only_where_windows_paths_are_local(monkeypatch, flavour):
    """fsutil is a Windows tool and C:\\big.bin is only a local path there.
    Both flavours run on every OS by passing the path rules explicitly."""
    monkeypatch.setattr(risk_gate_module, "_PATH_FLAVOUR", flavour)
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    risk, reasons = gate.evaluate_plan(_plan(_fsutil_action()), PilotConfig())
    if flavour is ntpath:
        assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
        assert reasons == ["creating big.bin (about 279 GB) needs more than the 265 GB free on C:"]
    else:
        assert risk == 0.0
        assert reasons == []


def test_fsutil_on_the_running_platform(monkeypatch):
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    risk, _ = gate.evaluate_plan(_plan(_fsutil_action()), PilotConfig())
    assert risk == (pytest.approx(DISK_USAGE_RISK_WEIGHT) if sys.platform == "win32" else 0.0)


def test_fallocate_counts_under_posix_paths(monkeypatch):
    monkeypatch.setattr(risk_gate_module, "_PATH_FLAVOUR", posixpath)
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="bash",
        parameters=ShellCommandParams(command="fallocate -l 300G /srv/big.img"),
    )
    risk, reasons = gate.evaluate_plan(_plan(action), PilotConfig())
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons == ["creating big.img (about 300 GB) needs more than the 265 GB free on C:"]


def test_shell_size_parsing_windows_flavour():
    parse = _shell_created_files
    assert parse('fsutil file createnew "D:\\a b.bin" 1048576', None, ntpath) == [("D:\\a b.bin", 1048576)]
    assert parse("fsutil file createnew big.bin 10", "C:\\work", ntpath) == [("C:\\work\\big.bin", 10)]
    # A relative path under a working directory that is not absolute is not guessed.
    assert parse("fsutil file createnew big.bin 10", "relative", ntpath) == []
    assert parse("fallocate -l 10G C:\\x", None, ntpath) == []  # not a Windows command


def test_shell_size_parsing_posix_flavour():
    parse = _shell_created_files
    parsed = parse("fallocate -l 10G /x; fallocate -l 10GB /y; fallocate --length 3MiB /z", None, posixpath)
    assert parsed == [("/x", 10 * GIB), ("/y", 10 * 1000**3), ("/z", 3 * 1024**2)]
    assert parse("fallocate -l 1K data.bin", "/w", posixpath) == [("/w/data.bin", KIB)]
    # A Windows drive path is not a POSIX path, and fsutil does not exist there.
    assert parse('fsutil file createnew "D:\\a b.bin" 1048576', None, posixpath) == []
    assert parse("fallocate -l 1K data.bin", "C:\\work", posixpath) == []
    assert parse("truncate -s 10G /x; dd if=/dev/zero of=/y bs=1M count=10", None, posixpath) == []


@pytest.mark.parametrize(
    ("path", "label"),
    [
        ("C:\\Users\\marcu\\Videos", "Videos"),
        ("C:\\Users\\marcu\\Videos\\", "Videos"),
        ("/home/runner/Videos/", "Videos"),
        ("D:/mixed\\sep/photos", "photos"),
        ("C:\\", "C:"),
        ("/", "/"),
        ("big.bin", "big.bin"),
    ],
)
def test_reason_labels_use_the_last_component_for_both_separators(path, label):
    assert _label(path) == label


def test_size_lookahead_only_adds_caution(monkeypatch, tmp_path):
    """Advisory invariant: a measured small size never removes another warning."""
    config = PilotConfig()
    config.restrictions.protected_folders = [str(tmp_path / "guarded")]
    config.gateway.risk_gate_enabled = True
    source = _tree(tmp_path / "guarded", 1, KIB)
    gate = _gate(monkeypatch, fraction=0.5, total=1_000_000)
    monkeypatch.setattr(risk_gate_module, "_gate", gate)
    plan = _plan(_copy(source, tmp_path / "out"))
    risk, reasons = gate.evaluate_plan(plan, config)
    assert risk == pytest.approx(0.9)
    assert any("protected path" in reason for reason in reasons)
    assessment = assess_plan_risk(plan, config)
    assert assessment.combined_score == max(heuristic_risk(plan), assessment.world_model_score)


def _low_tier_plan(target: str = "/data/report.txt") -> ActionPlan:
    return _plan(
        Action(action_type=ActionType.FILE_WRITE, target=target, parameters=FileParams(path=target, content="x"))
    )


@pytest.mark.asyncio
async def test_gateway_runs_the_assessment_off_the_event_loop(monkeypatch):
    import threading

    from pilot.agents import destructive_critic
    from pilot.security.gateway import AgentGateway
    from pilot.security.permissions import PermissionChecker

    config = PilotConfig()
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fake_assess(plan, cfg, *, source=""):
        seen.append(threading.get_ident())
        return destructive_critic.PlanRiskAssessment(
            heuristic_score=0.0,
            world_model_score=0.5,
            combined_score=0.5,
            reasons=["Videos is too large to measure quickly."],
        )

    monkeypatch.setattr(destructive_critic, "assess_plan_risk", fake_assess)
    gateway = AgentGateway(config, PermissionChecker(config))
    verdict = await gateway._maybe_run_critic(_low_tier_plan(), critic_already_reviewed=False)

    assert seen and seen[0] != loop_thread
    assert verdict["verdict"] == "WARN"
    assert verdict["risk_score"] == 0.5
    assert verdict["issues"] == ["Videos is too large to measure quickly."]


@pytest.mark.asyncio
async def test_gate_failure_in_the_worker_thread_is_handled_as_before(monkeypatch):
    from pilot.security.gateway import AgentGateway, InvocationSource
    from pilot.security.permissions import PermissionChecker

    class _BrokenGate:
        def evaluate_plan(self, *args, **kwargs):
            raise RuntimeError("scan exploded")

    monkeypatch.setattr(risk_gate_module, "get_risk_gate", lambda: _BrokenGate())
    config = PilotConfig()
    config.gateway.risk_gate_enabled = True
    gateway = AgentGateway(config, PermissionChecker(config))
    verdict = await gateway._maybe_run_critic(
        _low_tier_plan(), critic_already_reviewed=False, source=InvocationSource.AUTONOMOUS
    )
    assert verdict["verdict"] == "BLOCK"
    assert verdict["recommendation"] == "World-model assessment is unavailable for a state-changing plan."


def test_overlapping_evaluations_each_read_back_their_own_result(monkeypatch, tmp_path):
    import threading
    import time

    gate = _gate(monkeypatch, fraction=0.5, total=1_000_000)
    real_measure = gate._measure_tree_size

    def slow_measure(path, deadline):
        time.sleep(0.002)  # widen the window in which evaluations could interleave
        return real_measure(path, deadline=deadline)

    gate._measure_tree_size = slow_measure
    source = _tree(tmp_path / "docs", 1, KIB)
    plans = {
        1: _plan(_copy(source, tmp_path / "a")),
        3: _plan(*(_copy(source, tmp_path / f"b{i}") for i in range(3))),
    }
    mismatches: list[tuple[int, object]] = []

    def worker(size: int) -> None:
        for _ in range(25):
            gate.evaluate_plan(plans[size], PilotConfig())
            seen = gate.last_evaluation["action_count"]
            if seen != size:
                mismatches.append((size, seen))

    threads = [threading.Thread(target=worker, args=(size,)) for size in (1, 3, 1, 3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert mismatches == []
    assert len(gate.last_evaluation["predictions"]) == gate.last_evaluation["action_count"]


def test_copying_a_volume_root_uses_the_volume_used_bytes_without_a_scan(monkeypatch, tmp_path):
    import os

    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    scanned: list[str] = []
    gate._measure_tree_size = lambda path, deadline: scanned.append(path)
    root = os.path.abspath(os.sep)
    risk, reasons = gate.evaluate_plan(_plan(_copy(root, tmp_path / "Backup")), PilotConfig())
    assert scanned == []
    assert risk == pytest.approx(DISK_USAGE_RISK_WEIGHT)
    assert reasons[0].endswith("(about 688 GB) needs more than the 265 GB free on C:")
    record = gate.last_evaluation["predictions"][0]["size_estimates"][0]
    assert record["size_is_exact"] is True


def test_ordinary_folder_is_never_sized_from_its_volume(monkeypatch, tmp_path):
    source = _tree(tmp_path / "photos", 2, KIB)
    gate = _gate(monkeypatch, fraction=0.722, total=953 * GIB)
    gate.evaluate_plan(_plan(_copy(source, tmp_path / "out")), PilotConfig())
    assert gate.last_evaluation["predictions"][0]["size_estimates"][0]["bytes"] == 2 * KIB


def test_scan_budget_is_shared_across_the_plan_sources(monkeypatch, tmp_path):
    import time

    gate = _gate(monkeypatch, fraction=0.5, total=1000 * GIB)
    deadlines: list[float] = []

    def fake_measure(path, deadline):
        deadlines.append(deadline)
        return TreeSize(bytes=KIB, entries=1, complete=True, files=1)

    gate._measure_tree_size = fake_measure
    started = time.perf_counter()
    plan = _plan(_copy(tmp_path / "one", tmp_path / "x"), _copy(tmp_path / "two", tmp_path / "y"))
    gate.evaluate_plan(plan, PilotConfig())
    assert len(deadlines) == 2
    assert deadlines[0] == deadlines[1]
    assert deadlines[0] - started <= risk_gate_module.PLAN_SCAN_BUDGET_SECONDS + 0.5


@pytest.mark.asyncio
async def test_scan_budget_is_short_when_called_on_an_event_loop():
    import asyncio

    from pilot.security.risk_gate import _scan_budget_seconds
    from pilot.security.risk_observation import ON_LOOP_SCAN_BUDGET_SECONDS, PLAN_SCAN_BUDGET_SECONDS

    assert _scan_budget_seconds() == ON_LOOP_SCAN_BUDGET_SECONDS
    assert await asyncio.to_thread(_scan_budget_seconds) == PLAN_SCAN_BUDGET_SECONDS
