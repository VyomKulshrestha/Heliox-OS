"""Streaming speech recognition with sherpa-onnx (Apache-2.0) and an English streaming Zipformer.

Whisper starts transcribing only after the person stops, then needs a second or more on a laptop CPU. A streaming
transducer decodes while they speak, so the text is ready the moment the turn ends, and partial text can be shown
as a live caption while they are still talking. Measured on a laptop CPU: a 6.6 s clip decoded in 0.37 s as it
played, with nothing left to do at the end.

The model (int8, about 73 MB) downloads once from Hugging Face the first time it is used. It is English-only, so
it is an opt-in engine (``voice.transcription_engine = "streaming"``); Whisper stays the default.

Decoding runs on one worker thread, so the audio loop never waits on it.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger("pilot.system.streaming_asr")

REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
FILES = {
    "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens": "tokens.txt",
}
SAMPLE_RATE = 16000
PARTIAL_EVERY_SECONDS = 0.25


def tidy(text: str) -> str:
    """The model writes in capitals without punctuation; make it read like a sentence."""
    text = " ".join(text.split()).lower()
    if not text:
        return ""
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'(m|ve|ll|d)\b", lambda m: "I'" + m.group(1), text)
    return text[0].upper() + text[1:]


class StreamingTranscriber:
    def __init__(self, recognizer: Any = None, threads: int = 2) -> None:
        self._recognizer = recognizer
        self._threads = threads
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stream: Any = None
        self._partial = ""
        self._last_partial_at = 0.0
        self.on_partial: Callable[[str], None] | None = None
        self.error = ""
        self.last_finish_ms = 0.0

    @property
    def ready(self) -> bool:
        return self._recognizer is not None

    def load(self) -> bool:
        """Download (first use only) and open the model. Blocking: call it from a worker thread."""
        with self._lock:
            if self._recognizer is not None:
                return True
            try:
                import sherpa_onnx
                from huggingface_hub import hf_hub_download

                paths = {key: hf_hub_download(REPO, name) for key, name in FILES.items()}
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=paths["tokens"],
                    encoder=paths["encoder"],
                    decoder=paths["decoder"],
                    joiner=paths["joiner"],
                    num_threads=self._threads,
                    sample_rate=SAMPLE_RATE,
                    feature_dim=80,
                    decoding_method="greedy_search",
                )
                self.error = ""
                logger.info("Streaming speech recognition ready")
                return True
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"[:200]
                logger.info("Streaming recognition unavailable, using Whisper: %s", self.error)
                return False

    # ── one utterance ──

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name="heliox-streaming-asr", daemon=True)
            self._worker.start()

    def begin(self, frames: list[Any]) -> None:
        """A new utterance starts; ``frames`` is the pre-roll already captured."""
        if not self.ready:
            return
        self._ensure_worker()
        self._queue.put(("begin", None))
        for frame in frames:
            self._queue.put(("frame", frame))

    def feed(self, frame: Any) -> None:
        if self.ready:
            self._queue.put(("frame", frame))

    def partial(self) -> str:
        return self._partial

    def finish_future(self) -> Future[str]:
        done: Future[str] = Future()
        if not self.ready:
            done.set_result("")
            return done
        self._ensure_worker()
        self._queue.put(("finish", done))
        return done

    async def finish(self, timeout: float = 3.0) -> str:
        """The final text of the utterance, normally available at once."""
        started = time.perf_counter()
        try:
            text = await asyncio.wait_for(asyncio.wrap_future(self.finish_future()), timeout=timeout)
        except (TimeoutError, Exception):
            text = ""
        self.last_finish_ms = (time.perf_counter() - started) * 1000
        return text

    def _run(self) -> None:
        import numpy as np

        while True:
            kind, payload = self._queue.get()
            try:
                if kind == "begin":
                    self._stream = self._recognizer.create_stream()
                    self._partial = ""
                elif kind == "frame" and self._stream is not None:
                    samples = np.asarray(payload)
                    if samples.dtype == np.int16:
                        samples = samples.astype(np.float32) / 32768.0
                    self._stream.accept_waveform(SAMPLE_RATE, samples.astype(np.float32).ravel())
                    while self._recognizer.is_ready(self._stream):
                        self._recognizer.decode_stream(self._stream)
                    self._maybe_partial()
                elif kind == "finish":
                    text = ""
                    if self._stream is not None:
                        self._stream.input_finished()
                        while self._recognizer.is_ready(self._stream):
                            self._recognizer.decode_stream(self._stream)
                        text = tidy(self._result())
                        self._stream = None
                    payload.set_result(text)
            except Exception as error:
                logger.debug("Streaming recognition step failed", exc_info=True)
                if kind == "finish" and not payload.done():
                    payload.set_exception(error)
                self._stream = None

    def _result(self) -> str:
        result = self._recognizer.get_result(self._stream)
        return result if isinstance(result, str) else getattr(result, "text", "")

    def _maybe_partial(self) -> None:
        now = time.monotonic()
        if now - self._last_partial_at < PARTIAL_EVERY_SECONDS:
            return
        text = tidy(self._result())
        if text and text != self._partial:
            self._partial = text
            self._last_partial_at = now
            if self.on_partial:
                try:
                    self.on_partial(text)
                except Exception:
                    logger.debug("Partial caption callback failed", exc_info=True)
