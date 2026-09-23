"""Frozen, CPU-only NLI features for an unpromoted scope experiment.

The backbone is public and Apache-2.0 licensed. It is not bundled with Heliox.
No user data is sent to the model host: inference reads previously cached files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BACKBONE = "cross-encoder/nli-MiniLM2-L6-H768"
REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
TOKENIZER_FILE = "tokenizer.json"


def cached_paths() -> tuple[Path, Path]:
    """Require an explicit, pre-cached model rather than downloading at inference."""
    from huggingface_hub import hf_hub_download

    model = hf_hub_download(BACKBONE, ONNX_FILE, revision=REVISION, local_files_only=True)
    tokenizer = hf_hub_download(BACKBONE, TOKENIZER_FILE, revision=REVISION, local_files_only=True)
    return Path(model), Path(tokenizer)


class NliFeatureExtractor:
    def __init__(self, model_path: Path, tokenizer_path: Path, *, max_length: int = 128) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not 8 <= max_length <= 512:
            raise ValueError("max_length must fit the frozen backbone's 512-token input")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    def _direction(self, pairs: list[tuple[str, str]], batch_size: int) -> np.ndarray:
        probabilities: list[np.ndarray] = []
        for offset in range(0, len(pairs), batch_size):
            batch = pairs[offset : offset + batch_size]
            encoded = [self._tokenizer.encode(left, right) for left, right in batch]
            length = max(len(item.ids) for item in encoded)
            input_ids = np.full((len(batch), length), 1, dtype=np.int64)
            attention_mask = np.zeros((len(batch), length), dtype=np.int64)
            for row, item in enumerate(encoded):
                input_ids[row, : len(item.ids)] = item.ids
                attention_mask[row, : len(item.ids)] = 1
            logits = np.asarray(self._session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0])
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            probabilities.append(exp / exp.sum(axis=1, keepdims=True))
        return np.vstack(probabilities) if probabilities else np.empty((0, 3), dtype=np.float32)

    def transform(self, pairs: list[tuple[str, str]], *, batch_size: int = 8) -> np.ndarray:
        """Return forward and reverse contradiction/entailment/neutral scores."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        forward = self._direction(pairs, batch_size)
        reverse = self._direction([(action, request) for request, action in pairs], batch_size)
        return np.hstack((forward, reverse))
