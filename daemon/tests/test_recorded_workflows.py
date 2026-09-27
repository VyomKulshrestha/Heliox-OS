"""Opt-in recordings stay local, encrypted, bounded, and non-authoritative."""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pilot.actions import (
    Action,
    ActionResult,
    ActionType,
    BrowserParams,
    KeyboardParams,
    MouseParams,
)
from pilot.workflows.recorded_workflows import MODEL_AAD, RecordedWorkflowStore, workflow_prompt


@pytest.mark.asyncio
async def test_recording_requires_consent_and_never_captures_by_default(tmp_path) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {"embedding": [0.5] * 256, "width": 100, "height": 100}

    path = tmp_path / "recordings.enc"
    store = RecordedWorkflowStore(path, key_provider=lambda: b"k" * 32, capture_feature=capture)
    action = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=25, y=75))
    assert await store.before(action, single_action_batch=True) is None
    assert not path.exists()
    with pytest.raises(ValueError, match="consent"):
        store.start(name="Test", consent=False, capture_visual=True)
    store.start(name="Test", consent=True)
    token = await store.before(action, single_action_batch=True)
    await store.after(token, ActionResult(action=action, success=True))
    assert captures == 0
    saved = store.stop(save=True)
    assert saved["sample_count"] == 0
    assert saved["step_count"] == 1
    assert b"mouse_click" not in path.read_bytes()
    assert store.status()["active"] is False


@pytest.mark.asyncio
async def test_visual_samples_require_opt_in_and_never_appear_in_public_record(tmp_path) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {"embedding": [captures / 10] * 256, "width": 100, "height": 100}

    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    store.start(name="Click", consent=True, capture_visual=True)
    action = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=25, y=75))
    token = await store.before(action, single_action_batch=True)
    await store.after(token, ActionResult(action=action, success=True))
    saved = store.stop(save=True)
    assert captures == 2
    assert saved["sample_count"] == 1
    assert "samples" not in saved
    assert "embedding" not in str(store.list())
    assert store._read()[0]["samples"][0]["coords"] == [0.25, 0.75]
    assert store._read()[0]["samples"][0]["before"] == [26] * 256


@pytest.mark.asyncio
async def test_click_outside_captured_monitor_is_not_mislabeled(tmp_path) -> None:
    async def capture() -> dict:
        return {"embedding": [0.5] * 256, "left": 100, "top": 0, "width": 100, "height": 100}

    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    store.start(name="Different monitor", consent=True, capture_visual=True)
    action = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=25, y=75))
    token = await store.before(action, single_action_batch=True)
    await store.after(token, ActionResult(action=action, success=True))
    assert store.stop(save=True)["sample_count"] == 0


@pytest.mark.asyncio
async def test_failed_and_secret_actions_are_not_replayable(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="Safe subset", consent=True)
    secret = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text="password123"))
    token = await store.before(secret, single_action_batch=True)
    await store.after(token, ActionResult(action=secret, success=True))
    click = Action(action_type=ActionType.BROWSER_CLICK, parameters=BrowserParams(selector="#save"))
    token = await store.before(click, single_action_batch=True)
    await store.after(token, ActionResult(action=click, success=False))
    token = await store.before(click, single_action_batch=False)
    await store.after(token, ActionResult(action=click, success=True))
    token = await store.before(click, single_action_batch=True)
    await store.after(token, ActionResult(action=click, success=True))
    saved = store.stop(save=True)
    assert saved["step_count"] == 1
    assert saved["omitted_actions"] == 2
    assert saved["incomplete"] is True
    assert b"password123" not in store.path.read_bytes()
    assert "password123" not in str(store.list())


@pytest.mark.asyncio
async def test_schedule_requires_review_and_claims_due_run_once(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="Dashboard click", consent=True)
    click = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=25, y=75))
    token = await store.before(click, single_action_batch=True)
    await store.after(token, ActionResult(action=click, success=True))
    saved = store.stop(save=True)
    when = datetime.now(UTC) + timedelta(hours=1)
    with pytest.raises(ValueError, match="review"):
        store.schedule(saved["id"], run_at=when.isoformat())
    scheduled = store.schedule(saved["id"], run_at=when.isoformat(), reviewed=True)
    assert scheduled["schedule"]["next_run_at"] == when.isoformat()
    assert store.take_due(now=when - timedelta(seconds=1)) == []
    assert [row["id"] for row in store.take_due(now=when)] == [saved["id"]]
    assert store.take_due(now=when) == []


@pytest.mark.asyncio
async def test_incomplete_workflow_cannot_be_scheduled(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    store.start(name="With a password", consent=True)
    secret = Action(action_type=ActionType.KEYBOARD_TYPE, parameters=KeyboardParams(text="secret"))
    click = Action(action_type=ActionType.BROWSER_CLICK, parameters=BrowserParams(selector="#save"))
    for action in (secret, click):
        token = await store.before(action, single_action_batch=True)
        await store.after(token, ActionResult(action=action, success=True))
    saved = store.stop(save=True)
    with pytest.raises(ValueError, match="Incomplete"):
        store.schedule(
            saved["id"], run_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(), reviewed=True
        )


@pytest.mark.asyncio
async def test_repeat_schedule_survives_restart_but_recording_does_not(tmp_path) -> None:
    path = tmp_path / "recordings.enc"
    store = RecordedWorkflowStore(path, key_provider=lambda: b"k" * 32)
    store.start(name="Review", consent=True)
    action = Action(action_type=ActionType.BROWSER_CLICK, parameters=BrowserParams(selector="#review"))
    token = await store.before(action, single_action_batch=True)
    await store.after(token, ActionResult(action=action, success=True))
    saved = store.stop(save=True)
    first = datetime.now(UTC) + timedelta(hours=1)
    store.schedule(saved["id"], run_at=first.isoformat(), repeat_seconds=86400)

    restarted = RecordedWorkflowStore(path, key_provider=lambda: b"k" * 32)
    assert restarted.status()["active"] is False
    assert restarted.take_due(now=first)[0]["id"] == saved["id"]
    assert restarted.take_due(now=first) == []
    assert restarted.get(saved["id"])["schedule"]["next_run_at"] == (
        first + timedelta(days=1)
    ).isoformat()


def test_tampered_ciphertext_fails_closed(tmp_path) -> None:
    path = tmp_path / "recordings.enc"
    path.write_bytes(b"x" * 40)
    store = RecordedWorkflowStore(path, key_provider=lambda: b"k" * 32)
    with pytest.raises(RuntimeError, match="could not be decrypted"):
        store.prepare()
    assert store.status()["active"] is False


def test_private_model_weights_are_encrypted(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    protected = store.encrypt_model_artifact(b"private model weights")
    assert b"private model weights" not in protected
    assert AESGCM(b"k" * 32).decrypt(protected[:12], protected[12:], MODEL_AAD) == b"private model weights"


def test_manual_actions_need_separate_consent_and_text_is_parameterized(tmp_path) -> None:
    store = RecordedWorkflowStore(tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32)
    with pytest.raises(ValueError, match="Separate consent"):
        store.start(name="Manual", consent=True, capture_manual=True)
    store.start(name="Manual", consent=True, capture_manual=True, manual_consent=True)
    store.record_manual_event("click", x=40, y=60, button="left")
    store.record_manual_event("text")
    store.record_manual_event("text")
    store.record_manual_event("key", name="tab")
    store.record_manual_event("text")
    workflow = store.stop(save=True)
    assert workflow["step_count"] == 4
    assert workflow["placeholders"] == ["1", "3"]
    assert "manual_text" in str(workflow["steps"])
    assert b"password" not in store.path.read_bytes()
    with pytest.raises(ValueError, match="placeholder"):
        workflow_prompt(workflow)
    prompt = workflow_prompt(workflow, text_values={"1": "Project", "3": "Summary"})
    assert "keyboard_type(text='Project')" in prompt
    with pytest.raises(ValueError, match="Text placeholders"):
        store.schedule(
            workflow["id"], run_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat(), reviewed=True
        )


def test_manual_events_suppressed_during_heliox_actions_and_approval(tmp_path) -> None:
    pending = False
    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32,
        allow_manual_event=lambda: not pending,
    )
    store.start(name="Manual", consent=True, capture_manual=True, manual_consent=True)
    with store.suppress_manual_during_heliox_action():
        store.record_manual_event("click", x=40, y=60, button="left")
    pending = True
    store.record_manual_event("click", x=40, y=60, button="left")
    pending = False
    store.record_manual_event("click", x=40, y=60, button="left")
    saved = store.stop(save=True)
    assert saved["step_count"] == 1
    assert saved["overlap_warning"] is True


@pytest.mark.asyncio
async def test_manual_click_visual_pair_is_opt_in_encrypted_and_thread_safe(tmp_path) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {
            "embedding": [min(captures / 10, 1)] * 256,
            "left": 0, "top": 0, "width": 100, "height": 100,
        }

    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    store.start(name="Manual visual", consent=True, capture_manual=True, manual_consent=True,
                capture_visual=True)
    store.start_manual_visual_capture()
    for _ in range(50):
        if captures:
            break
        await asyncio.sleep(0.01)
    assert captures > 0
    press_feature = store.manual_press_feature(time.monotonic())
    assert press_feature is not None
    await asyncio.to_thread(
        store.record_manual_event, "click", x=25, y=75, button="left", press_feature=press_feature
    )
    await store.stop_manual_visual_capture(save=True)
    saved = store.stop(save=True)
    assert saved["sample_count"] == 1
    assert saved["visual_gaps"] == 0
    assert "before" not in str(saved)
    assert b"before" not in store.path.read_bytes()
    sample = store.load_research_samples()[0]["samples"][0]
    assert sample["source"] == "manual"
    assert sample["coords"] == [0.25, 0.75]
    assert sample["before"] == [26] * 256


@pytest.mark.asyncio
async def test_manual_visual_pair_drops_stale_and_interrupted_clicks(tmp_path) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {"embedding": [0.5] * 256, "left": 0, "top": 0,
                "width": 100, "height": 100}

    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    store.start(name="Interrupted", consent=True, capture_manual=True, manual_consent=True,
                capture_visual=True)
    store.start_manual_visual_capture()
    for _ in range(50):
        if captures:
            break
        await asyncio.sleep(0.01)
    assert store.manual_press_feature(time.monotonic() - 5) is None
    store.record_manual_event("click", x=10, y=20, button="left", press_feature=None)
    for _ in range(50):
        if captures > 1:
            break
        await asyncio.sleep(0.01)
    press_feature = store.manual_press_feature(time.monotonic())
    assert press_feature is not None
    store.record_manual_event("click", x=10, y=20, button="left", press_feature=press_feature)
    store.record_manual_event("text")
    await store.stop_manual_visual_capture(save=True)
    saved = store.stop(save=True)
    assert saved["sample_count"] == 0
    assert saved["visual_gaps"] >= 2


@pytest.mark.asyncio
async def test_manual_visual_sampler_never_starts_without_both_opt_ins(tmp_path) -> None:
    captures = 0

    async def capture() -> dict:
        nonlocal captures
        captures += 1
        return {"embedding": [0.5] * 256}

    store = RecordedWorkflowStore(
        tmp_path / "recordings.enc", key_provider=lambda: b"k" * 32, capture_feature=capture
    )
    store.start(name="No visual", consent=True, capture_manual=True, manual_consent=True)
    store.start_manual_visual_capture()
    await asyncio.sleep(0.05)
    assert captures == 0
    await store.stop_manual_visual_capture(save=False)
    store.stop(save=False)
