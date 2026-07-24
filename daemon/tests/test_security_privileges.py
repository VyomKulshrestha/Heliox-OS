from pilot.security.privileges import security_runtime_status


def test_root_policy_disabled_is_not_reported_as_unlocked():
    status = security_runtime_status(False, platform_name="win32", process_elevated=True)

    assert status["root_policy_enabled"] is False
    assert "blocked by Heliox policy" in status["detail"]


def test_windows_policy_does_not_claim_os_elevation():
    status = security_runtime_status(True, platform_name="win32", process_elevated=False)

    assert status["root_policy_enabled"] is True
    assert status["process_elevated"] is False
    assert "not running as Administrator" in status["detail"]


def test_elevated_daemon_reports_both_layers():
    status = security_runtime_status(True, platform_name="linux", process_elevated=True)

    assert status["process_elevated"] is True
    assert "allowed by policy" in status["detail"]
    assert "elevated OS privileges" in status["detail"]
