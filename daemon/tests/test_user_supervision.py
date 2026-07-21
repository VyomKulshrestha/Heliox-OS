"""Tests for pilot.agents.user_supervision.UserSupervisionEngine.

Covers the two independent, advisory-only trigger sources (cognitive
coaching via a fake TribeEngine, risk-pattern detection via the keystroke
hook and OCR snippet) and on_trigger's fan-out, mirroring test_narrator.py's
fake-broadcast/fake-config fixture shape.
"""

from __future__ import annotations

import pytest

import pilot.agents.user_supervision as user_supervision_module
from pilot.agents.user_supervision import UserSupervisionEngine
from pilot.cognitive.tribe_engine import CognitiveSnapshot
from pilot.config import PilotConfig
from pilot.system.input_hook import InputActivitySnapshot


class _Broadcast:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, method: str, params: dict) -> None:
        self.calls.append((method, params))


class _FakeTribe:
    def __init__(self, snapshot: CognitiveSnapshot | None = None):
        self.snapshot = snapshot or CognitiveSnapshot()
        self.calls: list[tuple[str, str]] = []

    async def predict_cognitive_state(self, stimulus_description: str = "", screen_region: str = "full"):
        self.calls.append((stimulus_description, screen_region))
        return self.snapshot


class _FakeScreenState:
    def __init__(self, active_window_title: str = ""):
        self.active_window_title = active_window_title


class _FakeContext:
    def __init__(self, state: _FakeScreenState | None):
        self._state = state

    def current(self):
        return self._state


class _FakeScreenVision:
    def __init__(self, active_window_title: str = "Notepad"):
        self._context = _FakeContext(_FakeScreenState(active_window_title))

    def get_context(self):
        return self._context


class _FakeHook:
    def __init__(self, snapshot: InputActivitySnapshot | None = None):
        self._snapshot = snapshot or InputActivitySnapshot(0.0, 0.0, None, True)

    def snapshot(self) -> InputActivitySnapshot:
        return self._snapshot


def _config(**overrides) -> PilotConfig:
    cfg = PilotConfig()
    cfg.supervision.enabled = True
    for k, v in overrides.items():
        setattr(cfg.supervision, k, v)
    return cfg


def _engine(
    *,
    tribe: _FakeTribe | None = None,
    screen_vision: _FakeScreenVision | None = None,
    hook: _FakeHook | None = None,
    ocr_text: str = "",
    **cfg_overrides,
) -> tuple[UserSupervisionEngine, _Broadcast, _FakeTribe]:
    tribe = tribe or _FakeTribe()
    broadcast = _Broadcast()
    engine = UserSupervisionEngine(
        config=_config(**cfg_overrides),
        tribe_engine=tribe,
        screen_vision=screen_vision or _FakeScreenVision(),
        hook=hook or _FakeHook(),
        broadcast_fn=broadcast,
    )

    async def _fake_screen_ocr(*args, **kwargs) -> str:
        return ocr_text

    user_supervision_module.screen_ocr = _fake_screen_ocr
    return engine, broadcast, tribe


class TestTickHookSignals:
    @pytest.mark.asyncio
    async def test_hook_disabled_skips_keystroke_check(self):
        hook = _FakeHook(InputActivitySnapshot(0.0, 0.0, "destructive_shell_command", True))
        engine, _, _ = _engine(hook=hook, keyboard_mouse_hook_enabled=False, cognitive_coaching_enabled=False)
        result = await engine.tick()
        assert not any(s["kind"] == "risk" and s["source"] == "keystroke" for s in result["signals"])

    @pytest.mark.asyncio
    async def test_hook_enabled_no_match_no_signal(self):
        hook = _FakeHook(InputActivitySnapshot(10.0, 2.0, None, True))
        engine, _, _ = _engine(
            hook=hook, keyboard_mouse_hook_enabled=True, cognitive_coaching_enabled=False, ocr_interval_seconds=999
        )
        result = await engine.tick()
        assert result["signals"] == []

    @pytest.mark.asyncio
    async def test_hook_enabled_with_match_produces_risk_signal(self):
        hook = _FakeHook(InputActivitySnapshot(10.0, 2.0, "destructive_shell_command", True))
        engine, _, _ = _engine(
            hook=hook, keyboard_mouse_hook_enabled=True, cognitive_coaching_enabled=False, ocr_interval_seconds=999
        )
        result = await engine.tick()
        assert result["triggered"] is True
        assert result["signals"] == [{"kind": "risk", "pattern": "destructive_shell_command", "source": "keystroke"}]

    @pytest.mark.asyncio
    async def test_risk_cooldown_suppresses_repeat_within_window(self):
        hook = _FakeHook(InputActivitySnapshot(10.0, 2.0, "destructive_shell_command", True))
        engine, _, _ = _engine(
            hook=hook,
            keyboard_mouse_hook_enabled=True,
            cognitive_coaching_enabled=False,
            ocr_interval_seconds=999,
            risk_cooldown_seconds=300.0,
        )
        first = await engine.tick()
        second = await engine.tick()
        assert first["triggered"] is True
        assert second["triggered"] is False


class TestTickOcrSignals:
    @pytest.mark.asyncio
    async def test_ocr_due_on_first_tick_detects_risk_pattern(self):
        engine, _, _ = _engine(
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=False,
            ocr_text="please run rm -rf / to clean up",
        )
        result = await engine.tick()
        assert result["signals"] == [{"kind": "risk", "pattern": "destructive_shell_command", "source": "ocr"}]

    @pytest.mark.asyncio
    async def test_ocr_not_due_on_second_immediate_tick(self):
        engine, _, _ = _engine(
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=False,
            ocr_text="please run rm -rf / to clean up",
            ocr_interval_seconds=8.0,
        )
        await engine.tick()  # consumes the "due" slot
        second = await engine.tick()
        assert second["signals"] == []

    @pytest.mark.asyncio
    async def test_risk_pattern_detection_disabled_skips_ocr_match(self):
        engine, _, _ = _engine(
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=False,
            risk_pattern_detection_enabled=False,
            ocr_text="please run rm -rf / to clean up",
        )
        result = await engine.tick()
        assert result["signals"] == []

    @pytest.mark.asyncio
    async def test_cognitive_coaching_above_threshold_produces_signal(self):
        tribe = _FakeTribe(CognitiveSnapshot(stress_level=0.9, cognitive_load=0.5, attention_score=0.4))
        engine, _, _ = _engine(
            tribe=tribe,
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=True,
            stress_coaching_threshold=0.75,
        )
        result = await engine.tick()
        assert result["triggered"] is True
        cognitive_signals = [s for s in result["signals"] if s["kind"] == "cognitive"]
        assert len(cognitive_signals) == 1
        assert cognitive_signals[0]["snapshot"]["stress_level"] == 0.9

    @pytest.mark.asyncio
    async def test_cognitive_coaching_below_threshold_no_signal(self):
        tribe = _FakeTribe(CognitiveSnapshot(stress_level=0.1, cognitive_load=0.1))
        engine, _, _ = _engine(tribe=tribe, keyboard_mouse_hook_enabled=False, cognitive_coaching_enabled=True)
        result = await engine.tick()
        assert result["signals"] == []

    @pytest.mark.asyncio
    async def test_coaching_cooldown_suppresses_repeat(self):
        tribe = _FakeTribe(CognitiveSnapshot(stress_level=0.9))
        engine, _, _ = _engine(
            tribe=tribe,
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=True,
            ocr_interval_seconds=0.0,
            coaching_cooldown_seconds=300.0,
        )
        first = await engine.tick()
        second = await engine.tick()
        assert first["triggered"] is True
        assert second["triggered"] is False

    @pytest.mark.asyncio
    async def test_tribe_receives_real_ocr_and_window_title_stimulus(self):
        tribe = _FakeTribe(CognitiveSnapshot(stress_level=0.9))
        screen_vision = _FakeScreenVision(active_window_title="Terminal")
        engine, _, _ = _engine(
            tribe=tribe,
            screen_vision=screen_vision,
            keyboard_mouse_hook_enabled=False,
            cognitive_coaching_enabled=True,
            ocr_text="drop table users",
        )
        await engine.tick()
        assert len(tribe.calls) == 1
        stimulus, region = tribe.calls[0]
        assert "drop table users" in stimulus
        assert "Terminal" in stimulus
        assert region == "user_supervision"

    @pytest.mark.asyncio
    async def test_ocr_failure_does_not_raise_and_skips_ocr_signals(self):
        engine, _, _ = _engine(keyboard_mouse_hook_enabled=False, cognitive_coaching_enabled=False)

        async def _raising_ocr(*args, **kwargs):
            raise RuntimeError("no ocr engine available")

        user_supervision_module.screen_ocr = _raising_ocr

        result = await engine.tick()  # must not raise
        assert result["signals"] == []


class TestOnTrigger:
    @pytest.mark.asyncio
    async def test_dispatches_risk_signal(self):
        engine, broadcast, _ = _engine()
        await engine.on_trigger({"signals": [{"kind": "risk", "pattern": "destructive_sql", "source": "ocr"}]})
        assert len(broadcast.calls) == 1
        method, params = broadcast.calls[0]
        assert method == "supervision_risk_warning"
        assert params["pattern"] == "destructive_sql"
        assert params["source"] == "ocr"

    @pytest.mark.asyncio
    async def test_dispatches_cognitive_signal(self):
        engine, broadcast, _ = _engine()
        snapshot = CognitiveSnapshot(stress_level=0.9, cognitive_load=0.5, attention_score=0.3).to_dict()
        await engine.on_trigger({"signals": [{"kind": "cognitive", "snapshot": snapshot}]})
        assert len(broadcast.calls) == 1
        method, params = broadcast.calls[0]
        assert method == "supervision_cognitive_checkin"
        assert params["stress_level"] == 0.9

    @pytest.mark.asyncio
    async def test_fans_out_multiple_signals_from_one_tick(self):
        engine, broadcast, _ = _engine()
        snapshot = CognitiveSnapshot(stress_level=0.9).to_dict()
        await engine.on_trigger(
            {
                "signals": [
                    {"kind": "risk", "pattern": "destructive_sql", "source": "ocr"},
                    {"kind": "cognitive", "snapshot": snapshot},
                ]
            }
        )
        assert len(broadcast.calls) == 2
        methods = {method for method, _ in broadcast.calls}
        assert methods == {"supervision_risk_warning", "supervision_cognitive_checkin"}


class TestAdvisoryBroadcastsNeverLeakRawContent:
    @pytest.mark.asyncio
    async def test_risk_warning_never_includes_raw_matched_text(self):
        engine, broadcast, _ = _engine()
        await engine.on_risk_pattern_detected("credential_exposure", "keystroke")
        _, params = broadcast.calls[0]
        assert set(params.keys()) == {"pattern", "source", "message"}
        assert "password" not in str(params).lower() or "credential" in params["message"].lower()

    @pytest.mark.asyncio
    async def test_no_broadcast_fn_is_a_safe_noop(self):
        engine = UserSupervisionEngine(
            config=_config(),
            tribe_engine=_FakeTribe(),
            screen_vision=_FakeScreenVision(),
            hook=_FakeHook(),
            broadcast_fn=None,
        )
        await engine.on_risk_pattern_detected("destructive_sql", "ocr")  # must not raise
        await engine.on_cognitive_checkin(CognitiveSnapshot().to_dict())  # must not raise
