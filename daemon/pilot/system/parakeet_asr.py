"""NVIDIA Parakeet TDT 0.6b v2 (int8, through sherpa-onnx) as the accurate final pass, on the CPU.

Measured on the owner's voice through a Bluetooth headset in hands-free mode (narrowband, quiet): it kept
"Monday, the twelfth of November at ten" where Whisper base heard "mildly" or "by me", in 0.6 to 0.7 s for a
six-second sentence. It never hears the made-up name "Heliox", so the wake phrase still comes from Whisper, run
alongside it (see merge_wake).

The model is about 640 MB once unpacked and is not shipped: put the folder
``sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8`` (encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx,
tokens.txt; from the sherpa-onnx GitHub releases, tag asr-models) in ``<Heliox data>/models/``, or point
``voice.parakeet_model_dir`` at it, and set ``voice.final_pass = "parakeet"``. Without it Whisper stays the final
pass. sherpa-onnx is imported only when the model is loaded.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.parakeet_asr")

MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
# What it says for a breath or a click between sentences, measured on the rehearsal's noise.
FILLERS = frozenset({"mm hmm", "mm", "hmm", "uh", "um", "kay", "uh huh", "so", "q"})


def model_dir(configured: str, data_dir: Path) -> Path:
    """Where the model is looked for: the configured folder, or ``<data_dir>/models/<MODEL_NAME>``."""
    return Path(configured).expanduser() if configured.strip() else data_dir / "models" / MODEL_NAME


def missing_files(folder: Path) -> list[str]:
    return [name for name in FILES if not (folder / name).is_file()]


def is_filler(text: str) -> bool:
    words = " ".join(re.findall(r"[a-z]+", text.casefold().replace("-", " ")))
    return not words or words in FILLERS


def merge_wake(whisper_text: str, parakeet_text: str, wake_prefix: Any) -> str:
    """Parakeet's words, with the wake phrase taken from Whisper.

    Whisper (with Heliox's vocabulary prompt) finds "Heliox" where Parakeet writes "Dear works" or "Here it was",
    while Parakeet gets the rest right ("Monday, the twelfth of November at ten"). When Whisper heard the wake
    phrase, Parakeet's words before the first word of Whisper's command are its mishearing of that phrase and are
    replaced by it. If the two cannot be lined up, the wake phrase is put in front of Parakeet's words.
    """
    parakeet_text = parakeet_text.strip()
    if not whisper_text.strip() and len(parakeet_text.split()) <= 2:
        # Whisper, which drops what it rates as not speech, heard nothing: a lone "Okay." or "Wait." from Parakeet
        # on the noise between sentences is not the person (measured on the rehearsal).
        return ""
    heard = wake_prefix(whisper_text)
    if not heard:
        return parakeet_text or whisper_text.strip()
    command = whisper_text[len(heard) :].strip(" \t,.:;!?-")
    first = re.findall(r"[a-z']+", command.casefold())[:1]
    words = parakeet_text.split()
    plain = [re.sub(r"[^a-z']", "", word.casefold()) for word in words]
    if first and first[0] in plain[:5]:
        rest = " ".join(words[plain.index(first[0]) :])
        return f"{heard}, {rest}"
    if not command:
        return heard
    # The two cannot be lined up: keep Parakeet's words, which carry the dates and numbers, after the wake phrase.
    return f"{heard}, {parakeet_text}" if parakeet_text else whisper_text.strip()


class ParakeetTranscriber:
    """Loads the model on first use (about 4 s) and keeps it; safe to call from worker threads."""

    def __init__(self, folder: Path, threads: int | None = None) -> None:
        self.folder = folder
        self.threads = threads or max(1, min(8, os.cpu_count() or 4))
        self._recognizer: Any = None
        self._lock = threading.Lock()
        self.error = ""

    @property
    def available(self) -> bool:
        return not missing_files(self.folder)

    def load(self) -> bool:
        with self._lock:
            if self._recognizer is not None:
                return True
            missing = missing_files(self.folder)
            if missing:
                self.error = f"Parakeet model files missing in {self.folder}: {', '.join(missing)}"
                return False
            try:
                import sherpa_onnx

                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(self.folder / "encoder.int8.onnx"),
                    decoder=str(self.folder / "decoder.int8.onnx"),
                    joiner=str(self.folder / "joiner.int8.onnx"),
                    tokens=str(self.folder / "tokens.txt"),
                    model_type="nemo_transducer",
                    num_threads=self.threads,
                )
                self.error = ""
                # One decode of silence: the first real sentence otherwise pays for the runtime's first-run setup.
                import numpy as np

                warm = self._recognizer.create_stream()
                warm.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
                self._recognizer.decode_stream(warm)
                logger.info("Parakeet final pass ready (%s)", self.folder)
                return True
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"[:200]
                logger.warning("Parakeet final pass unavailable: %s", self.error)
                return False

    def transcribe(self, samples: Any) -> str:
        """Text for one utterance (16 kHz mono float32); "" for a filler or when the model cannot load."""
        if not self.load():
            return ""
        import numpy as np

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, np.asarray(samples, dtype=np.float32))
        self._recognizer.decode_stream(stream)
        text = str(stream.result.text).strip()
        return "" if is_filler(text) else text
