"""The suite neither needs nor disturbs the graphics card of the person running it, and loads no model from their
Hugging Face cache (tests/conftest.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pilot.system.local_tts_worker import cuda_hidden


@pytest.mark.skipif(os.environ.get("HELIOX_TEST_GPU") == "1", reason="the card was asked for on purpose")
def test_every_card_is_hidden_here_and_in_every_process_a_test_starts():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and cuda_hidden()
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "'-1'"


@pytest.mark.skipif(os.environ.get("HELIOX_REAL_VOICE_MODELS") == "1", reason="the real models were asked for")
def test_the_hugging_face_cache_is_the_session_folder_and_offline():
    session = Path(os.environ["HELIOX_TEST_ROOT"]).resolve()
    assert session in Path(os.environ["HF_HOME"]).resolve().parents
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        assert name not in os.environ
