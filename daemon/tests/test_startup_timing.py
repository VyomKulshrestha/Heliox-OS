"""The daemon's own start-up timings, and the first-run CI check that reads them: times from process start, the
modules loaded, and a failure when a heavy numerical library is loaded before ready."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from pilot.system.startup_timing import HEAVY_LIBRARIES, StartupTimings, heavy_loaded

PLATFORM_REAL = Path(__file__).resolve().parent / "platform_real"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PLATFORM_REAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_each_point_is_timed_from_process_start_with_what_was_loaded(monkeypatch):
    clock = iter([100.0, 103.25, 109.5])
    modules = {"pilot": object(), "numpy": object()}
    timings = StartupTimings(wall=lambda: next(clock), created=99.0, modules=lambda: modules)
    assert timings.mark("listening") == {"seconds": 4.25, "since": "process start", "modules": 2, "heavy": []}
    modules["scipy"] = object()
    assert timings.mark("ready")["heavy"] == ["scipy"]
    monkeypatch.delenv("HELIOX_STARTUP_TIMINGS", raising=False)
    assert timings.snapshot() is None  # off unless asked for
    monkeypatch.setenv("HELIOX_STARTUP_TIMINGS", "1")
    assert set(timings.snapshot()) == {"listening", "ready"}


def test_a_blocked_import_does_not_count_as_loaded():
    assert heavy_loaded({"torch": None, "pandas": object()}) == ["pandas"]
    assert set(HEAVY_LIBRARIES) == {"scipy", "sklearn", "pandas", "river", "torch"}


def test_this_process_has_a_creation_time():
    timings = StartupTimings()
    point = timings.mark("now")
    assert point["since"] == "process start" and point["seconds"] > 0


def test_the_ci_check_reports_the_daemon_s_times_and_fails_on_a_heavy_library_before_ready():
    check = _load("first_run_rpc_check")
    point = {"seconds": 5.0, "since": "process start", "modules": 500, "heavy": []}
    timings = {"listening": point, "ready": {**point, "seconds": 9.0}, "all_started": {**point, "seconds": 31.5}}
    lines, ok = check.timing_checks("first-run", timings, {"neural": "ready"})
    assert ok
    assert (
        lines[0] == "CHECK start-up/first-run/listening: PASS - listening 5.0 s after process start, 500 modules loaded"
    )
    assert "every late part started 31.5 s" in lines[2]
    assert lines[-1].startswith("CHECK start-up/first-run/nothing-heavy-before-ready: PASS")

    timings["ready"] = {**point, "heavy": ["scipy", "river"]}
    lines, ok = check.timing_checks("saved-state", timings, {"neural": "failed"})
    assert not ok
    assert "FAIL - loaded at ready: scipy, river" in "\n".join(lines)
    assert "late-parts: FAIL - not started: neural" in "\n".join(lines)
    _lines, ok = check.timing_checks("full-replay", {}, {})
    assert not ok


def test_the_seeded_ledger_is_the_mix_the_learner_learns_from():
    seed = _load("seed_ledger")
    rows = seed.rows(1_000)
    assert len(rows) == 1_000
    types = {row[1] for row in rows}
    assert {"action_started", "action_completed", "outcome_verified", "suggestion_feedback", "observation"} <= types
    assert len({row[10] for row in rows}) == 1_000  # idempotency keys are unique, as the table requires
    json.loads(rows[0][12])
