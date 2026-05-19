import importlib
import sys
import types
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.asyncio


MODULE_NAME = "pilot.system.processes"


class DummyPlatform:
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


@pytest.fixture
def processes_module(monkeypatch):
    platform_detect = types.ModuleType("pilot.system.platform_detect")
    platform_detect.CURRENT_PLATFORM = DummyPlatform.LINUX
    platform_detect.Platform = DummyPlatform
    platform_detect.run_command = AsyncMock(return_value=(0, "", ""))
    platform_detect.run_powershell = AsyncMock(return_value=(0, "", ""))

    monkeypatch.setitem(sys.modules, "pilot.system.platform_detect", platform_detect)
    sys.modules.pop(MODULE_NAME, None)
    module = importlib.import_module(MODULE_NAME)
    return module


class FakePsutilProcess:
    def __init__(self, info):
        self.info = info


class FakeNoSuchProcess(Exception):
    pass


class FakeAccessDenied(Exception):
    pass


class FakeManagedProcess:
    def __init__(self, pid, name="demo"):
        self.pid = pid
        self._name = name
        self.killed = False
        self.terminated = False

    def name(self):
        return self._name

    def kill(self):
        self.killed = True

    def terminate(self):
        self.terminated = True

    def as_dict(self, attrs):
        data = {
            "pid": self.pid,
            "name": self._name,
            "exe": "/usr/bin/demo",
            "cmdline": ["demo", "--flag"],
            "status": "running",
            "cpu_percent": 12.5,
            "memory_percent": 3.4,
            "memory_info": "rss=1234",
            "create_time": 1111.0,
            "num_threads": 4,
            "username": "tester",
        }
        return {k: data[k] for k in attrs}


class BrokenInfoProcess:
    @property
    def info(self):
        raise FakeAccessDenied()


def install_fake_psutil(monkeypatch, *, process_iter=None, process_factory=None):
    fake_psutil = types.SimpleNamespace(
        process_iter=process_iter or (lambda attrs: []),
        Process=process_factory or (lambda pid: FakeManagedProcess(pid)),
        NoSuchProcess=FakeNoSuchProcess,
        AccessDenied=FakeAccessDenied,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    return fake_psutil


def remove_psutil(monkeypatch):
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


async def test_process_list_with_psutil_formats_and_filters(processes_module, monkeypatch):
    processes = [
        FakePsutilProcess(
            {
                "pid": 101,
                "name": "python",
                "cpu_percent": 10.0,
                "memory_percent": 2.5,
                "status": "running",
            }
        ),
        FakePsutilProcess(
            {
                "pid": 202,
                "name": "nginx",
                "cpu_percent": 1.0,
                "memory_percent": 0.5,
                "status": "sleeping",
            }
        ),
        BrokenInfoProcess(),
    ]
    install_fake_psutil(monkeypatch, process_iter=lambda attrs: processes)

    result = await processes_module.process_list(filter_name="py")

    assert "python" in result
    assert "nginx" not in result
    assert "PID=   101" in result
    assert "STATUS=running" in result


async def test_process_list_with_psutil_no_matches_returns_message(processes_module, monkeypatch):
    install_fake_psutil(
        monkeypatch,
        process_iter=lambda attrs: [
            FakePsutilProcess(
                {
                    "pid": 202,
                    "name": "nginx",
                    "cpu_percent": 1.0,
                    "memory_percent": 0.5,
                    "status": "sleeping",
                }
            )
        ],
    )

    result = await processes_module.process_list(filter_name="python")

    assert result == "No matching processes found"


async def test_process_list_windows_fallback_uses_powershell(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.WINDOWS)
    processes_module.run_powershell = AsyncMock(return_value=(0, "Header\nproc1\nproc2", ""))

    result = await processes_module.process_list()

    assert result == "Header\nproc1\nproc2"
    processes_module.run_powershell.assert_awaited_once()


async def test_process_list_linux_fallback_filters_lines(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.LINUX)
    processes_module.run_command = AsyncMock(return_value=(0, "PID CMD\n1 python app.py\n2 nginx\n3 PYTHON worker", ""))

    result = await processes_module.process_list(filter_name="python")

    assert result == "PID CMD\n1 python app.py\n3 PYTHON worker"
    processes_module.run_command.assert_awaited_once_with(["ps", "aux", "--sort=-%mem"])


async def test_process_list_fallback_raises_on_command_error(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.MACOS)
    processes_module.run_command = AsyncMock(return_value=(1, "", "permission denied\n"))

    with pytest.raises(RuntimeError, match="Process list failed: permission denied"):
        await processes_module.process_list()


async def test_process_kill_dispatches_to_pid(processes_module, monkeypatch):
    mock = AsyncMock(return_value="pid path")
    monkeypatch.setattr(processes_module, "_kill_by_pid", mock)

    result = await processes_module.process_kill(pid=55, sig="SIGKILL")

    assert result == "pid path"
    mock.assert_awaited_once_with(55, "SIGKILL")


async def test_process_kill_dispatches_to_name(processes_module, monkeypatch):
    mock = AsyncMock(return_value="name path")
    monkeypatch.setattr(processes_module, "_kill_by_name", mock)

    result = await processes_module.process_kill(name="python")

    assert result == "name path"
    mock.assert_awaited_once_with("python", "SIGTERM")


async def test_process_kill_requires_pid_or_name(processes_module):
    with pytest.raises(ValueError, match="Either pid or name must be provided"):
        await processes_module.process_kill()


async def test_kill_by_pid_uses_psutil_terminate_for_default_signal(processes_module, monkeypatch):
    proc = FakeManagedProcess(77, name="worker")
    install_fake_psutil(monkeypatch, process_factory=lambda pid: proc)

    result = await processes_module._kill_by_pid(77, "SIGTERM")

    assert result == "Sent SIGTERM to process worker (PID 77)"
    assert proc.terminated is True
    assert proc.killed is False


async def test_kill_by_pid_uses_psutil_kill_for_sigkill(processes_module, monkeypatch):
    proc = FakeManagedProcess(88, name="worker")
    install_fake_psutil(monkeypatch, process_factory=lambda pid: proc)

    result = await processes_module._kill_by_pid(88, "9")

    assert result == "Sent 9 to process worker (PID 88)"
    assert proc.killed is True
    assert proc.terminated is False


async def test_kill_by_pid_windows_fallback(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.WINDOWS)
    processes_module.run_command = AsyncMock(return_value=(0, "SUCCESS", ""))

    result = await processes_module._kill_by_pid(999, "SIGTERM")

    assert result == "Killed PID 999"
    processes_module.run_command.assert_awaited_once_with(["taskkill", "/PID", "999", "/F"])


async def test_kill_by_pid_windows_fallback_raises_on_error(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.WINDOWS)
    processes_module.run_command = AsyncMock(return_value=(1, "", "not found\n"))

    with pytest.raises(RuntimeError, match="Kill failed: not found"):
        await processes_module._kill_by_pid(999, "SIGTERM")


async def test_kill_by_pid_non_windows_fallback_uses_os_kill(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.LINUX)
    calls = []
    monkeypatch.setattr(processes_module.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    result = await processes_module._kill_by_pid(123, "SIGKILL")

    assert result == "Sent SIGKILL to PID 123"
    expected_signal = getattr(
        processes_module.signal_module,
        "SIGKILL",
        processes_module.signal_module.SIGTERM,
    )
    assert calls == [(123, expected_signal)]


async def test_kill_by_pid_unknown_signal_falls_back_to_sigterm(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.LINUX)
    calls = []
    monkeypatch.setattr(processes_module.os, "kill", lambda pid, sig: calls.append((pid, sig)))

    result = await processes_module._kill_by_pid(123, "NOT_A_SIGNAL")

    assert result == "Sent NOT_A_SIGNAL to PID 123"
    assert calls == [(123, processes_module.signal_module.SIGTERM)]


async def test_kill_by_name_windows(processes_module, monkeypatch):
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.WINDOWS)
    processes_module.run_command = AsyncMock(return_value=(0, "ok", ""))

    result = await processes_module._kill_by_name("python.exe", "SIGTERM")

    assert result == "Killed processes matching 'python.exe'"
    processes_module.run_command.assert_awaited_once_with(["taskkill", "/IM", "python.exe", "/F"])


async def test_kill_by_name_non_windows_error(processes_module, monkeypatch):
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.LINUX)
    processes_module.run_command = AsyncMock(return_value=(1, "", "no process found\n"))

    with pytest.raises(RuntimeError, match="Kill by name failed: no process found"):
        await processes_module._kill_by_name("python", "SIGTERM")


async def test_process_info_with_psutil_formats_details(processes_module, monkeypatch):
    install_fake_psutil(monkeypatch, process_factory=lambda pid: FakeManagedProcess(pid, name="python"))

    result = await processes_module.process_info(321)

    assert result.startswith("Process info for PID 321:")
    assert "name: python" in result
    assert "username: tester" in result
    assert "cmdline: ['demo', '--flag']" in result


async def test_process_info_windows_fallback(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.WINDOWS)
    processes_module.run_powershell = AsyncMock(return_value=(0, "Id : 321\nName : python", ""))

    result = await processes_module.process_info(321)

    assert result == "Id : 321\nName : python"
    processes_module.run_powershell.assert_awaited_once_with("Get-Process -Id 321 | Format-List *")


async def test_process_info_non_windows_fallback_raises_on_error(processes_module, monkeypatch):
    remove_psutil(monkeypatch)
    monkeypatch.setattr(processes_module, "CURRENT_PLATFORM", DummyPlatform.LINUX)
    processes_module.run_command = AsyncMock(return_value=(1, "", "missing pid\n"))

    with pytest.raises(RuntimeError, match="Process info failed: missing pid"):
        await processes_module.process_info(321)
