"""Semantic end-of-turn detection with Pipecat's Smart Turn v3.2 (BSD-2-Clause, about 8.7 MB, CPU).

Voice activity detection only knows silence, so a person who pauses to think gets cut off, and a person who has
finished still waits out the full silence timeout. Smart Turn listens to the last 8 seconds of the turn (how the
sentence ends, its intonation) and returns the probability that the person is done. The listener asks it after a
short silence: done means answer now, not done means keep listening while they think.

The model downloads once from Hugging Face (pipecat-ai/smart-turn-v3) the first time it is needed and runs on the
CPU in tens of milliseconds. If it cannot load, the listener falls back to plain silence endpointing.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("pilot.system.turn_detector")

REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
COMPLETE_THRESHOLD = 0.5


def window(audio: Any) -> Any:
    """The last 8 seconds of the turn as float32 in [-1, 1], zero-padded at the front (the model's convention)."""
    import numpy as np

    samples = np.asarray(audio)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32)
    size = SAMPLE_RATE * WINDOW_SECONDS
    if samples.shape[0] >= size:
        return samples[-size:]
    return np.concatenate([np.zeros(size - samples.shape[0], dtype=np.float32), samples])


class SmartTurnDetector:
    """Loads lazily and thread-safely; ``probability`` returns None until the model is ready or if it fails."""

    def __init__(self, model_path: str | None = None, session: Any = None) -> None:
        self._model_path = model_path
        self._session = session
        self._features: Any = None
        self._lock = threading.Lock()
        self.error = ""

    @property
    def ready(self) -> bool:
        return self._session is not None

    def load(self) -> bool:
        """Download (first use only) and open the model. Blocking: call it from a worker thread."""
        with self._lock:
            if self._session is not None:
                return True
            try:
                import onnxruntime as ort
                from faster_whisper.feature_extractor import FeatureExtractor

                path = self._model_path
                if path is None:
                    from huggingface_hub import hf_hub_download

                    path = hf_hub_download(REPO, MODEL_FILE)
                options = ort.SessionOptions()
                options.inter_op_num_threads = 1
                options.intra_op_num_threads = 2
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
                self._features = FeatureExtractor(
                    feature_size=80, sampling_rate=SAMPLE_RATE, hop_length=160, chunk_length=WINDOW_SECONDS, n_fft=400
                )
                self.error = ""
                logger.info("Smart Turn end-of-turn model ready")
                return True
            except Exception as error:  # missing package, offline first run, corrupt download
                self.error = f"{type(error).__name__}: {error}"[:200]
                logger.info("Smart Turn unavailable, using silence endpointing: %s", self.error)
                return False

    def features(self, audio: Any) -> Any:
        """Whisper log-mel features exactly as Smart Turn's reference pipeline computes them: (1, 80, 800)."""
        import numpy as np

        if self._features is None:
            from faster_whisper.feature_extractor import FeatureExtractor

            self._features = FeatureExtractor(
                feature_size=80, sampling_rate=SAMPLE_RATE, hop_length=160, chunk_length=WINDOW_SECONDS, n_fft=400
            )
        samples = window(audio)
        normalized = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
        mel = self._features(normalized, padding=0)
        return np.expand_dims(mel[:, : WINDOW_SECONDS * 100].astype(np.float32), 0)

    def probability(self, audio: Any) -> float | None:
        """Probability that the turn is complete, or None when the model is not available."""
        session = self._session
        if session is None:
            return None
        try:
            output = session.run(None, {"input_features": self.features(audio)})
            return float(output[0].ravel()[0])
        except Exception:
            logger.debug("Smart Turn inference failed", exc_info=True)
            return None
