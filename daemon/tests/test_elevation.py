from unittest.mock import MagicMock

import pytest

from pilot.system.elevation import (
    ElevationError,
    replace_existing_daemon,
    request_elevated_restart,
)


def test_windows_elevation_uses_native_runas_with_current_daemon_pid(tmp_path):
    launcher = MagicMock(return_value=42)
    python_exe = tmp_path / "python.exe"

    result = request_elevated_restart(
        replace_pid=4321,
        executable=str(python_exe),
        working_directory=tmp_path,
        platform_name="win32",
        launcher=launcher,
    )

    assert result["status"] == "prompted"
    executable, parameters, working_directory = launcher.call_args.args
    assert executable == str(python_exe.resolve())
    assert parameters == "-m pilot.server --replace-pid 4321"
    assert working_directory == str(tmp_path.resolve())


def test_windows_elevation_reports_uac_denial(tmp_path):
    with pytest.raises(ElevationError, match="did not grant Administrator access"):
        request_elevated_restart(
            replace_pid=4321,
            executable=str(tmp_path / "python.exe"),
            working_directory=tmp_path,
            platform_name="win32",
            launcher=MagicMock(return_value=5),
        )


def test_elevation_is_not_offered_on_other_platforms():
    with pytest.raises(ElevationError, match="only available on Windows"):
        request_elevated_restart(platform_name="linux")


def test_replacement_refuses_to_stop_unrelated_process(monkeypatch):
    process = MagicMock()
    process.cmdline.return_value = ["python", "-m", "unrelated.server"]
    monkeypatch.setattr("psutil.Process", MagicMock(return_value=process))

    with pytest.raises(ElevationError, match="not the Heliox daemon"):
        replace_existing_daemon(4321, platform_name="win32", response_grace_seconds=0)

    process.terminate.assert_not_called()


def test_replacement_stops_only_validated_pilot_daemon(monkeypatch):
    process = MagicMock()
    process.cmdline.return_value = ["python", "-m", "pilot.server"]
    monkeypatch.setattr("psutil.Process", MagicMock(return_value=process))

    replace_existing_daemon(4321, platform_name="win32", response_grace_seconds=0)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=8)
