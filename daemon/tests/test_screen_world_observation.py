"""The world model may use fresh screen features without exporting pixels."""

from datetime import UTC, datetime, timedelta

import numpy as np

from pilot.agents.screen_vision import ScreenState, ScreenVisionAgent, _world_embedding_from_frame


def test_frame_feature_is_bounded_and_uses_current_pixels() -> None:
    dark = _world_embedding_from_frame(np.zeros((64, 64), dtype=np.float32))
    bright = _world_embedding_from_frame(np.full((64, 64), 255, dtype=np.float32))
    assert len(dark) == len(bright) == 256
    assert set(dark) == {0.0}
    assert set(bright) == {1.0}
    assert _world_embedding_from_frame(np.full((64, 64), np.nan)) == []
    assert _world_embedding_from_frame(np.zeros((32, 32))) == []


def test_world_observation_requires_fresh_running_capture_and_stays_private() -> None:
    agent = ScreenVisionAgent()
    state = ScreenState(
        timestamp=datetime.now(UTC).isoformat(),
        world_embedding=[0.5] * 256,
        screen_bounds=(10, 20, 1920, 1080),
    )
    agent.get_context().add(state)
    assert agent.get_world_observation() is None
    agent._running = True
    observation = agent.get_world_observation()
    assert observation is not None
    assert observation["latent_embedding"] == [0.5] * 256
    assert observation["screen_bounds"] == (10, 20, 1920, 1080)
    assert "world_embedding" not in state.to_dict()
    assert "screen_bounds" not in state.to_dict()
    observation["latent_embedding"][0] = 0.9
    assert state.world_embedding[0] == 0.5
    agent.pause()
    assert agent.get_world_observation() is None
    agent.resume()
    state.timestamp = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    assert agent.get_world_observation() is None


def test_companion_text_context_requires_fresh_running_capture() -> None:
    agent = ScreenVisionAgent()
    state = ScreenState(
        timestamp=datetime.now(UTC).isoformat(),
        active_app="Browser",
        active_window_title="Requested page",
    )
    agent.get_context().add(state)
    assert agent.get_recent_text_context() == ""
    agent._running = True
    assert 'Browser — "Requested page"' in agent.get_recent_text_context()
    agent.pause()
    assert agent.get_recent_text_context() == ""
    agent.resume()
    state.timestamp = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    assert agent.get_recent_text_context() == ""
