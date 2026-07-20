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
async def test_low_confidence_event_is_ingested_call_but_not_buffered():
    """The handler still reports "ingested" (the call succeeded), but the
    fusion engine itself silently drops sub-threshold readings -- gaze
    never independently rejects/errors on low confidence, it's simply not
    useful signal yet."""
    server = _server(gaze_enabled=True)
    result = await server._handle_gaze_event({"region": "left", "confidence": 0.01}, ws=None)
    assert result["status"] == "ingested"
    assert server._fusion.get_stats()["gaze_buffer_size"] == 0
