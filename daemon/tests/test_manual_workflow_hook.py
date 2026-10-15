"""Manual hook emits only replay-safe event metadata."""

import sys
from types import SimpleNamespace

import pytest

from pilot.workflows.manual_workflow_hook import (
    ManualWorkflowHook,
    _is_heliox_executable,
    _is_heliox_window_title,
)
from pilot.workflows.recorded_workflows import RecordedWorkflowStore


def test_manual_hook_uses_placeholders_and_marks_shortcut_incomplete(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="Desktop task", consent=True, capture_manual=True, manual_consent=True)
    hook = ManualWorkflowHook(store, ignore_foreground=lambda: False)
    button = SimpleNamespace(name="left")
    hook._on_click(10, 20, button, True)
    hook._on_click(10, 20, button, False)
    hook._on_key_press(SimpleNamespace(char="s"))
    hook._on_key_press(SimpleNamespace(char="e"))
    hook._on_key_press(SimpleNamespace(name="tab"))
    hook._on_key_press(SimpleNamespace(char="password"))
    hook._on_click(10, 20, button, True)
    hook._on_click(50, 60, button, False)
    hook._on_key_press(SimpleNamespace(name="ctrl_l"))
    hook._on_key_press(SimpleNamespace(char="c"))
    hook._on_key_release(SimpleNamespace(name="ctrl_l"))
    saved = store.stop(save=True)
    assert saved["placeholders"] == ["1", "3"]
    assert saved["omitted_actions"] == 1
    assert saved["incomplete"] is True
    assert saved["steps"][4]["action_type"] == "mouse_drag"
    assert "password" not in str(saved)
    assert "password" not in str(store._read())


def test_manual_hook_ignores_heliox_foreground(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="Desktop task", consent=True, capture_manual=True, manual_consent=True)
    hook = ManualWorkflowHook(store, ignore_foreground=lambda: True)
    button = SimpleNamespace(name="left")
    hook._on_click(10, 20, button, True)
    hook._on_click(10, 20, button, False)
    hook._on_key_press(SimpleNamespace(char="x"))
    assert store.status()["step_count"] == 0
    store.stop(save=False)


def test_manual_hook_keeps_press_time_feature_until_click_release(tmp_path, monkeypatch) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="Desktop task", consent=True, capture_manual=True, manual_consent=True)
    observed = []
    token = (0, [128] * 256, (0, 0, 100, 100))
    monkeypatch.setattr(store, "manual_press_feature", lambda _pressed_at: token)
    monkeypatch.setattr(store, "record_manual_event", lambda event, **values: observed.append((event, values)))
    hook = ManualWorkflowHook(store, ignore_foreground=lambda: False)
    button = SimpleNamespace(name="left")
    hook._on_click(10, 20, button, True)
    hook._on_click(10, 20, button, False)
    assert observed == [("click", {"x": 10, "y": 20, "button": "left", "press_feature": token})]
    store.stop(save=False)


def test_manual_hook_lifecycle_uses_temporary_listeners(tmp_path, monkeypatch) -> None:
    class Listener:
        def __init__(self, **_callbacks):
            self.alive = False

        def start(self):
            self.alive = True

        def stop(self):
            self.alive = False

        def is_alive(self):
            return self.alive

    monkeypatch.setitem(
        sys.modules,
        "pynput",
        SimpleNamespace(keyboard=SimpleNamespace(Listener=Listener), mouse=SimpleNamespace(Listener=Listener)),
    )
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    hook = ManualWorkflowHook(store, ignore_foreground=lambda: False)
    assert hook.healthy() is False
    hook.start()
    assert hook.healthy() is True
    hook.stop()
    assert hook.healthy() is False


@pytest.mark.parametrize("title", ["Heliox-OS", "Heliox OS", "heliox_os", "HELIOX  -  OS", " heliox-os "])
def test_heliox_window_titles_are_excluded(title) -> None:
    assert _is_heliox_window_title(title) is True


@pytest.mark.parametrize("title", ["Notepad", "Heliox", "Helios OS", "", "OS settings"])
def test_unrelated_window_titles_are_recorded(title) -> None:
    assert _is_heliox_window_title(title) is False


@pytest.mark.parametrize(
    "title",
    [
        "GitHub - VyomKulshrestha/Heliox-OS: Voice-first agent OS - Google Chrome",
        "server.py - Heliox-OS - Visual Studio Code",
        "heliox_os - File Explorer",
        "Heliox OS - Settings",
        "Pull requests \u00b7 VyomKulshrestha/Heliox-OS - Microsoft Edge",
    ],
)
def test_windows_that_only_mention_the_project_are_recorded(title) -> None:
    assert _is_heliox_window_title(title) is False


@pytest.mark.parametrize(
    "image",
    [
        r"C:\Program Files\Heliox-OS\Heliox-OS.exe",
        r"C:\src\pilot\tauri-app\src-tauri\target\debug\heliox-os.exe",
        "/usr/bin/heliox-os",
    ],
)
def test_the_heliox_executable_is_recognised(image) -> None:
    assert _is_heliox_executable(image) is True


@pytest.mark.parametrize(
    "image",
    [r"C:\Windows\explorer.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe", "heliox.exe", ""],
)
def test_other_executables_are_not_heliox(image) -> None:
    assert _is_heliox_executable(image) is False


@pytest.mark.parametrize(
    ("title", "image", "ignored"),
    [
        ("Heliox-OS", r"C:\Program Files\Heliox-OS\Heliox-OS.exe", True),
        ("Heliox-OS", None, True),  # Windows did not name the process: the title alone decides
        ("Heliox-OS", r"C:\Windows\explorer.exe", False),  # a folder named Heliox-OS
        ("server.py - Heliox-OS - Visual Studio Code", r"C:\Program Files\Heliox-OS\Heliox-OS.exe", False),
    ],
)
def test_foreground_check_needs_the_heliox_title_and_process(monkeypatch, title, image, ignored) -> None:
    import pilot.workflows.manual_workflow_hook as hook_module

    looked_up: list[int] = []

    def process_image(window: int) -> str | None:
        looked_up.append(window)
        return image

    monkeypatch.setattr(hook_module.sys, "platform", "win32")
    monkeypatch.setattr(hook_module, "_foreground_window_title", lambda: (42, title))
    monkeypatch.setattr(hook_module, "_window_process_image", process_image)

    assert hook_module._is_heliox_foreground() is ignored
    # The process is looked up only for a window with the Heliox title.
    assert looked_up == ([42] if _is_heliox_window_title(title) else [])


def test_foreground_check_reads_the_real_foreground_window(monkeypatch) -> None:
    import pilot.workflows.manual_workflow_hook as hook_module

    if sys.platform != "win32":
        assert hook_module._is_heliox_foreground() is False
        return
    window, title = hook_module._foreground_window_title()
    assert isinstance(window, int) and isinstance(title, str)
    image = hook_module._window_process_image(window)
    assert image is None or isinstance(image, str)
