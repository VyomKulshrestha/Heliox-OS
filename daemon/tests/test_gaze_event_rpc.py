"""Tests for PilotServer._handle_gaze_event -- the RPC handler that feeds
a frontend gaze reading into the fusion engine."""

import pytest

from pilot.config import PilotConfig
from pilot.multimodal.fusion import MultimodalFusionEngine
from pilot.server import PilotServer


def _server(gaze_enabled: bool = True) -> PilotServer:
    config = PilotConfig()
    config.vision.gaze_tracking_enabled = gaze_enabled
    server = PilotServer(config)
    server._fusion = MultimodalFusionEngine()
    return server


@pytest.mark.asyncio
async def test_returns_error_when_fusion_engine_not_initialized():
    server = PilotServer(PilotConfig())
    result = await server._handle_gaze_event({"region": "left", "confidence": 0.8}, ws=None)
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_ignored_when_gaze_tracking_disabled():
    server = _server(gaze_enabled=False)
    result = await server._handle_gaze_event({"region": "left", "confidence": 0.8}, ws=None)
    assert result["status"] == "ignored"
    assert server._fusion.get_stats()["gaze_buffer_size"] == 0


@pytest.mark.asyncio
async def test_ingests_a_valid_gaze_event():
    server = _server(gaze_enabled=True)
    result = await server._handle_gaze_event({"region": "left", "confidence": 0.8}, ws=None)
    assert result["status"] == "ingested"
    assert server._fusion.get_stats()["gaze_buffer_size"] == 1


@pytest.mark.asyncio
async def test_low_confidence_event_reports_ignored_and_is_not_buffered():
    server = _server(gaze_enabled=True)
    result = await server._handle_gaze_event({"region": "left", "confidence": 0.01}, ws=None)
    assert result == {"status": "ignored", "reason": "confidence_below_threshold"}
    assert server._fusion.get_stats()["gaze_buffer_size"] == 0


@pytest.mark.asyncio
async def test_invalid_region_reports_ignored_and_is_not_buffered():
    server = _server(gaze_enabled=True)
    result = await server._handle_gaze_event({"region": "upper-left", "confidence": 0.8}, ws=None)
    assert result == {"status": "ignored", "reason": "invalid_region"}
    assert server._fusion.get_stats()["gaze_buffer_size"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [-0.1, 1.1, "high", None])
async def test_invalid_confidence_reports_ignored(confidence):
    server = _server(gaze_enabled=True)
    result = await server._handle_gaze_event({"region": "left", "confidence": confidence}, ws=None)
    assert result == {"status": "ignored", "reason": "invalid_confidence"}
    assert server._fusion.get_stats()["gaze_buffer_size"] == 0
