"""Manual hook emits only replay-safe event metadata."""

import sys
from types import SimpleNamespace

from pilot.workflows.manual_workflow_hook import ManualWorkflowHook
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
