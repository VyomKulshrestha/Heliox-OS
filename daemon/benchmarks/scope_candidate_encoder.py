"""Frozen candidate-conditioned encoder for an offline scope experiment.

This exposes the cached NLI model's pooled representation before its three-way
classifier. No backbone weights are changed or shipped with Heliox. The model
is an experimental scorer, never an execution or approval authority.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from benchmarks.scope_nli_features import REVISION, cached_paths

HIDDEN_OUTPUT = "/classifier/Tanh_output_0"
HIDDEN_SIZE = 768
MAX_LENGTH = 256
CANDIDATES = (
    "The proposed action stays within the user's request.",
    "The proposed action needs additional authorization from the user.",
)


def candidate_prompts(request: str, action: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if not request.strip() or not action.strip():
        raise ValueError("A request and an action are required")
    context = f"User request: {request}\nProposed action: {action}"
    return ((context, CANDIDATES[0]), (context, CANDIDATES[1]))


def candidate_differences(hidden: np.ndarray) -> np.ndarray:
    """The shared scalar head's two-score softmax reduces to this difference."""
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_SIZE or hidden.shape[0] % 2:
        raise ValueError("Expected paired candidate representations")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("Candidate representations must be finite")
    return hidden[0::2] - hidden[1::2]


class CandidateEncoder:
    def __init__(self, *, max_length: int = MAX_LENGTH) -> None:
        import onnx
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not 32 <= max_length <= 512:
            raise ValueError("max_length must be between 32 and 512")
        model_path, tokenizer_path = cached_paths()
        model = onnx.load(str(model_path))
        if not any(HIDDEN_OUTPUT in node.output for node in model.graph.node):
            raise ValueError("Pinned NLI graph has no expected pooled output")
        del model.graph.output[:]
        model.graph.output.append(
            onnx.helper.make_tensor_value_info(HIDDEN_OUTPUT, onnx.TensorProto.FLOAT, ["batch_size", HIDDEN_SIZE])
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        with tempfile.TemporaryDirectory(prefix="heliox-candidate-encoder-") as directory:
            derived = Path(directory) / "pooled.onnx"
            onnx.save(model, str(derived))
            self._session = ort.InferenceSession(str(derived), sess_options=options, providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=max_length)
        self.max_length = max_length

    def transform(self, pairs: list[tuple[str, str]], *, batch_size: int = 16) -> np.ndarray:
        prompts = [prompt for pair in pairs for prompt in candidate_prompts(*pair)]
        return candidate_differences(self.encode_prompts(prompts, batch_size=batch_size * 2))

    def encode_prompts(self, prompts: list[tuple[str, str]], *, batch_size: int = 32) -> np.ndarray:
        """Encode arbitrary context/candidate pairs for a shared decision head."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        features: list[np.ndarray] = []
        for offset in range(0, len(prompts), batch_size):
            encoded = [self._tokenizer.encode(*prompt) for prompt in prompts[offset : offset + batch_size]]
            length = max(len(item.ids) for item in encoded)
            ids = np.full((len(encoded), length), 1, dtype=np.int64)
            mask = np.zeros((len(encoded), length), dtype=np.int64)
            for row, item in enumerate(encoded):
                ids[row, : len(item.ids)] = item.ids
                mask[row, : len(item.ids)] = 1
            hidden = np.asarray(
                self._session.run(
                    None,
                    {
                        "input_ids": ids,
                        "attention_mask": mask,
                    },
                )[0],
                dtype=np.float32,
            )
            features.append(hidden)
        return np.vstack(features) if features else np.empty((0, HIDDEN_SIZE), dtype=np.float32)
