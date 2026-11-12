"""Stop and correction while Heliox talks or works, through the real dispatch, coordinator and layer with a silent
speaker: the voice is cut, the model stream is closed and the correction reaches the task. Timings are measured by
benchmarks/conversation_eval/interrupt_timing.py on a quiet machine; here only that each thing happens."""

from __future__ import annotations

import math

import pytest

from benchmarks.conversation_eval.interrupt_timing import reply_once, task_once


@pytest.mark.asyncio
async def test_stop_cuts_the_voice_and_closes_the_model_stream():
    result = await reply_once()
    assert math.isfinite(result["voice_cut_ms"]) and result["voice_cut_ms"] >= 0
    assert math.isfinite(result["stream_closed_ms"]) and result["stream_closed_ms"] >= 0


@pytest.mark.asyncio
async def test_a_correction_cuts_the_narration_and_reaches_the_running_task():
    result = await task_once()
    assert math.isfinite(result["voice_cut_ms"])
    assert math.isfinite(result["correction_reached_ms"])
