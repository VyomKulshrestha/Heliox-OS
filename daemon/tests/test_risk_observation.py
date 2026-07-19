from pilot.security.risk_observation import capture_os_snapshot


def test_capture_returns_sane_fractions():
    snapshot = capture_os_snapshot()
    assert snapshot.proc_count > 0
    assert 0.0 <= snapshot.disk_usage_fraction <= 1.0
    assert 0.0 <= snapshot.memory_usage_fraction <= 1.0
    assert 0.0 <= snapshot.proc_count_normalized <= 1.0


def test_capture_never_raises_on_bad_path():
    # An invalid disk path should degrade to 0.0 disk usage, not raise.
    snapshot = capture_os_snapshot(disk_path="/this/path/does/not/exist/at/all")
    assert snapshot.disk_usage_fraction == 0.0
    assert snapshot.proc_count > 0  # proc_count/memory readings are unaffected


def test_proc_count_normalized_clamps_at_one():
    from pilot.security.risk_observation import OsSnapshot

    snapshot = OsSnapshot(proc_count=10_000, disk_usage_fraction=0.5, memory_usage_fraction=0.5, disk_path="/")
    assert snapshot.proc_count_normalized == 1.0
