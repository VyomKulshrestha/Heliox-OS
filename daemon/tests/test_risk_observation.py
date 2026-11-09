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


def _tree(root, files, size):
    root.mkdir()
    for index in range(files):
        (root / f"f{index}.bin").write_bytes(b"\0" * size)
    (root / "sub").mkdir()
    (root / "sub" / "g.bin").write_bytes(b"\0" * size)
    return root


def _far_future():
    import time

    return time.perf_counter() + 60


def test_tree_size_is_exact_for_a_small_tree(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    root = _tree(tmp_path / "t", 4, 100)
    size = measure_tree_size(str(root), deadline=_far_future())
    assert size is not None
    assert size.bytes == 500
    assert size.complete is True


def test_tree_size_of_a_single_file(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    path = tmp_path / "one.bin"
    path.write_bytes(b"\0" * 1234)
    size = measure_tree_size(str(path), deadline=_far_future())
    assert (size.bytes, size.entries, size.complete) == (1234, 1, True)


def test_tree_scan_stops_at_entry_limit_with_a_lower_bound(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    root = _tree(tmp_path / "t", 20, 100)
    size = measure_tree_size(str(root), deadline=_far_future(), max_entries=5)
    assert size.complete is False
    assert size.stopped_early is True
    assert size.entries == 5
    assert size.bytes <= 500 < 2100


def test_tree_scan_stops_at_deadline(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    root = _tree(tmp_path / "t", 20, 100)
    ticks = iter(range(1000))
    size = measure_tree_size(str(root), deadline=3, clock=lambda: next(ticks))
    assert size.stopped_early is True
    assert size.entries == 3


def test_tree_scan_never_raises(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    assert measure_tree_size(str(tmp_path / "missing"), deadline=_far_future()) is None

    def unreadable(path):
        raise PermissionError(path)

    root = _tree(tmp_path / "t", 2, 10)
    size = measure_tree_size(str(root), deadline=_far_future(), scandir=unreadable)
    assert size.complete is False
    assert size.bytes == 0


def test_tree_scan_does_not_open_files(tmp_path, monkeypatch):
    import builtins

    from pilot.security.risk_observation import measure_tree_size

    root = _tree(tmp_path / "t", 3, 10)

    def no_open(*args, **kwargs):
        raise AssertionError("file contents must not be read")

    monkeypatch.setattr(builtins, "open", no_open)
    assert measure_tree_size(str(root), deadline=_far_future()).bytes == 40


def _make_dir_link(link, target):
    import os
    import sys

    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_tree_scan_does_not_follow_links_out_of_the_tree(tmp_path):
    from pilot.security.risk_observation import measure_tree_size

    outside = _tree(tmp_path / "outside", 5, 10_000)
    root = _tree(tmp_path / "t", 1, 100)
    _make_dir_link(root / "link", outside)
    size = measure_tree_size(str(root), deadline=_far_future())
    assert size.bytes == 200
    assert size.complete is False  # the copy may follow the link; only a lower bound is known

    rooted_at_link = measure_tree_size(str(root / "link"), deadline=_far_future())
    assert (rooted_at_link.bytes, rooted_at_link.complete) == (0, False)


def test_volume_of_uses_nearest_existing_parent(tmp_path):
    from pilot.security.risk_observation import volume_of

    volume = volume_of(str(tmp_path / "not" / "yet" / "there.bin"))
    assert volume is not None
    assert volume.total > 0
    assert 0 <= volume.used <= volume.total
    assert volume == volume_of(str(tmp_path))
