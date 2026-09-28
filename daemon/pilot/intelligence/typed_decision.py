"""Local, non-authoritative typed decisions for research and inspection."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

KINDS = frozenset({"noul", "choice", "score"})
PROMPT_STYLES = frozenset({"candidate_v1", "full_options_v2"})
MAX_OPTIONS = 255
MAX_STATE_CHARS = 16_384
MAX_QUESTION_CHARS = 4_000
MAX_OPTION_CHARS = 1_024
MAX_QUESTIONS = 64
MAX_QUESTION_ID_CHARS = 128
MAX_TOTAL_OPTIONS = 256
MAX_TOTAL_PROMPT_TOKENS = 64_000
MAX_CANDIDATE_MICROBATCH = 16
CONFIDENCE_SEMANTICS = "normalized_distribution_concentration_not_accuracy_probability"


def _candidate_microbatch_size(max_prompt_tokens: int, configured_max: int) -> int:
    """Bound padded GPU work by prompt length while keeping short decisions fast."""
    if max_prompt_tokens < 1 or configured_max < 1:
        raise ValueError("Prompt length and microbatch limit must be positive")
    if max_prompt_tokens > 768:
        return min(configured_max, 2)
    if max_prompt_tokens > 512:
        return min(configured_max, 4)
    return configured_max

# Only base checkpoints with a reviewed, local-only loading path are supported.
PINNED_BACKBONES = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}


def validate_decision(kind: str, question: str, options: list[str]) -> None:
    """Validate a task-agnostic candidate-conditioned decision request."""
    if kind not in KINDS or not isinstance(question, str) or not question.strip():
        raise ValueError("Decision kind and question are required")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Question exceeds {MAX_QUESTION_CHARS} characters")
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"Decision needs 2-{MAX_OPTIONS} options")
    if any(
        not isinstance(option, str)
        or not option.strip()
        or len(option) > MAX_OPTION_CHARS
        for option in options
    ):
        raise ValueError(f"Options must be nonempty strings up to {MAX_OPTION_CHARS} characters")
    if len(options) != len(set(options)):
        raise ValueError("Decision options must be unique")
    if kind == "noul" and options != ["no", "yes"]:
        raise ValueError("Noul options must be ordered as no, yes")


def _render_typed_value(value: Any, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, dict, list)):
        raise ValueError("Instructions and descriptions must be text, an object, or an array")
    try:
        return value if isinstance(value, str) else json.dumps(
            value, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Instructions and descriptions must contain JSON values") from exc


def compile_typed_questions(state: Any, questions: Any) -> list[dict[str, Any]]:
    """Compile TypeSafe-style typed questions into Heliox's neutral candidate API."""
    if not isinstance(state, (str, dict, list)):
        raise ValueError("State must be text, a JSON object, or an array")
    try:
        state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("State must contain JSON values") from exc
    if len(state_json) > MAX_STATE_CHARS:
        raise ValueError(f"State exceeds {MAX_STATE_CHARS} characters")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise ValueError(f"Questions must be an object with 1-{MAX_QUESTIONS} entries")

    compiled = []
    total_options = 0
    for question_id, definition in questions.items():
        if (not isinstance(question_id, str) or not question_id.strip()
                or len(question_id) > MAX_QUESTION_ID_CHARS):
            raise ValueError("Question IDs must be nonempty strings within the size limit")
        if not isinstance(definition, dict) or set(definition) - {"type", "instructions", "criteria"}:
            raise ValueError("Question definitions may contain only type, instructions, and criteria")
        kind = definition.get("type")
        if kind not in KINDS:
            raise ValueError("Question type must be choice, score, or noul")
        question = _render_typed_value(definition.get("instructions"))
        criteria = definition.get("criteria")

        if kind == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                    raise ValueError("Noul criteria must contain true and false descriptions")
                true = _render_typed_value(criteria["true"])
                false = _render_typed_value(criteria["false"])
                question += f"\nYes means: {true}\nNo means: {false}"
            options = ["no", "yes"]
            answer_keys = ["false", "true"]
            legend = None
        elif kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_OPTIONS:
                raise ValueError(f"Choice criteria must contain 2-{MAX_OPTIONS} candidates")
            if any(not isinstance(name, str) for name in criteria):
                raise ValueError("Choice candidate names must be strings")
            answer_keys = list(criteria)
            options = []
            for name, description in criteria.items():
                rendered = _render_typed_value(description, optional=True)
                options.append(name if rendered is None else f"{name}: {rendered}")
            legend = None
        else:
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("Score criteria must contain 2-10 ordered levels")
            answer_keys = [str(index) for index in range(len(criteria))]
            options = [_render_typed_value(level) for level in criteria]
            legend = json.loads(json.dumps(criteria, ensure_ascii=False, allow_nan=False))

        validate_decision(kind, question, options)
        total_options += len(options)
        if total_options > MAX_TOTAL_OPTIONS:
            raise ValueError(f"A typed request may contain at most {MAX_TOTAL_OPTIONS} candidate options")
        compiled.append({
            "id": question_id,
            "kind": kind,
            "question": question,
            "options": options,
            "answer_keys": answer_keys,
            "legend": legend,
        })
    return compiled


def format_typed_answer(record: dict[str, Any], probabilities: list[float] | np.ndarray) -> dict[str, Any]:
    """Format normalized candidate probabilities as a TypeSafe-style answer."""
    values = np.asarray(probabilities, dtype=np.float64)
    options = record["options"]
    keys = record["answer_keys"]
    if (values.ndim != 1 or len(values) != len(options) or len(keys) != len(options)
            or not np.all(np.isfinite(values)) or np.any(values < 0)):
        raise ValueError("Model returned invalid typed-decision probabilities")
    total = float(values.sum())
    if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("Typed-decision probabilities must sum to one")
    values = values / total
    if record["kind"] == "noul":
        return {"type": "noul", "noul": float(values[1])}
    if record["kind"] == "choice":
        selected = int(np.argmax(values))
        uniform = 1.0 / len(values)
        confidence = (float(values.max()) - uniform) / (1.0 - uniform)
        return {
            "type": "choice",
            "choice": keys[selected],
            "probabilities": {key: float(value) for key, value in zip(keys, values, strict=True)},
            "confidence": float(confidence),
            "confidence_semantics": CONFIDENCE_SEMANTICS,
        }
    mode = int(np.argmax(values))
    distance = float(sum(value * abs(index - mode) for index, value in enumerate(values)))
    center = (len(values) - 1) / 2
    uniform_deviation = sum(abs(index - center) for index in range(len(values))) / len(values)
    confidence = max(0.0, 1.0 - distance / uniform_deviation)
    return {
        "type": "score",
        "score": float(np.arange(len(values), dtype=np.float64) @ values),
        "probabilities": {key: float(value) for key, value in zip(keys, values, strict=True)},
        "confidence": float(confidence),
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "legend": record["legend"],
    }


def _validate_prompt_budget(sequences: list[list[int]]) -> None:
    total = sum(len(sequence) for sequence in sequences)
    if total > MAX_TOTAL_PROMPT_TOKENS:
        raise ValueError(
            f"Typed request exceeds the {MAX_TOTAL_PROMPT_TOKENS}-token local scoring budget"
        )


def candidate_prompts(row: dict[str, Any], *, style: str) -> list[str]:
    """Build one neutral scoring prompt per candidate; no task taxonomy is embedded."""
    if style not in PROMPT_STYLES:
        raise ValueError("Unknown typed-decision prompt style")
    state = json.loads(row["state_json"])
    rendered_state = (
        state
        if isinstance(state, str)
        else json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    prefix = f"Context:\n{rendered_state}\n\nQuestion: {row['question']}\n"
    if row["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    if style == "full_options_v2":
        choices = "\n".join(
            f"{index + 1}. {option}" for index, option in enumerate(row["options"])
        )
        prefix += f"Available answers:\n{choices}\n"
    return [
        prefix + f"Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
        for option in row["options"]
    ]


def format_answer(kind: str, options: list[str], logits: np.ndarray, temperature: float) -> dict:
    """Return typed, calibrated scores without converting them into authority."""
    validate_decision(kind, "score this candidate set", options)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(options) or not np.all(np.isfinite(values)):
        raise ValueError("Model returned invalid candidate scores")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Model calibration temperature is invalid")
    scaled = values / temperature
    exp_values = np.exp(scaled - np.max(scaled))
    probabilities = exp_values / exp_values.sum()
    if kind == "noul":
        return {"type": kind, "yes_probability": float(probabilities[1])}
    distribution = {
        option: float(probability)
        for option, probability in zip(options, probabilities, strict=True)
    }
    selected = int(np.argmax(probabilities))
    if kind == "choice":
        return {
            "type": kind,
            "choice": options[selected],
            "probabilities": distribution,
            "max_probability": float(probabilities[selected]),
        }
    return {
        "type": kind,
        "expected_index": float(np.arange(len(probabilities)) @ probabilities),
        "most_likely": options[selected],
        "probabilities": distribution,
    }


class LocalTypedDecisionModel:
    """Lazy GPU-backed scorer that only reads explicitly configured local artifacts."""

    def __init__(self, artifact_dir: str | Path, report_path: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir).expanduser()
        self.report_path = Path(report_path).expanduser()
        self._lock = threading.RLock()
        self._loaded = False
        self._backbone: Any = None
        self._head: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._temperature = 1.0
        self._max_length = 0
        self._prompt_style = ""
        self._microbatch_size = MAX_CANDIDATE_MICROBATCH
        self._identity: dict[str, str] = {}

    def score(self, state: Any, kind: str, question: str, options: list[str]) -> dict:
        """Score dynamic options. Input is not persisted or passed to any action planner."""
        validate_decision(kind, question, options)
        try:
            state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("State must be a JSON-serializable object") from exc
        if len(state_json) > MAX_STATE_CHARS:
            raise ValueError(f"State exceeds {MAX_STATE_CHARS} characters")
        row = {
            "kind": kind,
            "question": question.strip(),
            "options": options,
            "state_json": state_json,
        }

        with self._lock:
            self._load()
            prompts = candidate_prompts(row, style=self._prompt_style)
            sequences = self._encode(prompts)
            _validate_prompt_budget(sequences)
            scores = self._score_sequences(sequences)
            logits = np.concatenate((np.zeros(1, dtype=np.float64), scores[:1])) if kind == "noul" else scores
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "advisory_only": True,
                "artifact_integrity": "sha256_verified",
                "model": self._identity["model"],
                "revision": self._identity["revision"],
                "answer": format_answer(kind, options, logits, self._temperature),
            }

    def score_questions(self, state: Any, questions: Any) -> dict[str, Any]:
        """Score bounded typed questions together in one cached-model GPU scoring call."""
        records = compile_typed_questions(state, questions)
        with self._lock:
            self._load()
            state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
            rows = [
                {
                    "kind": record["kind"],
                    "question": record["question"],
                    "options": record["options"],
                    "state_json": state_json,
                }
                for record in records
            ]
            prompts_by_record = [
                candidate_prompts(row, style=self._prompt_style) for row in rows
            ]
            prompts = [prompt for group in prompts_by_record for prompt in group]
            sequences = self._encode(prompts)
            _validate_prompt_budget(sequences)
            candidate_scores = self._score_sequences(sequences)
            answers: dict[str, dict[str, Any]] = {}
            offset = 0
            for record, record_prompts in zip(records, prompts_by_record, strict=True):
                count = len(record_prompts)
                scores = candidate_scores[offset : offset + count]
                offset += count
                logits = (
                    np.concatenate((np.zeros(1, dtype=np.float64), scores[:1]))
                    if record["kind"] == "noul"
                    else scores
                )
                answer = format_answer(
                    record["kind"], record["options"], logits, self._temperature
                )
                if record["kind"] == "noul":
                    yes_probability = answer["yes_probability"]
                    probabilities = [1.0 - yes_probability, yes_probability]
                else:
                    probabilities = [answer["probabilities"][option] for option in record["options"]]
                answers[record["id"]] = format_typed_answer(record, probabilities)
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "advisory_only": True,
                "artifact_integrity": "sha256_verified",
                "model": self._identity["model"],
                "revision": self._identity["revision"],
                "answers": answers,
            }

    def _load(self) -> None:
        if self._loaded:
            return
        if not self.artifact_dir.is_absolute() or not self.report_path.is_absolute():
            raise ValueError("Artifact and report paths must be absolute")
        if not self.artifact_dir.is_dir() or not self.report_path.is_file():
            raise FileNotFoundError("Configured local decision artifacts were not found")

        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        model = report.get("model")
        revision = report.get("revision")
        if model not in PINNED_BACKBONES or revision != PINNED_BACKBONES[model]:
            raise ValueError("Research report does not match a supported pinned backbone")
        if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("Research report contains an invalid model revision")
        adapter_path = self.artifact_dir / "adapter" / "adapter_model.safetensors"
        head_path = self.artifact_dir / "head.npz"
        for path, key in ((adapter_path, "adapter_sha256"), (head_path, "head_sha256")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing local model artifact: {path.name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not re.fullmatch(r"[0-9a-f]{64}", str(report.get(key, ""))) or digest != report[key]:
                raise ValueError(f"Local research artifact checksum mismatch: {path.name}")

        with np.load(head_path, allow_pickle=False) as saved:
            if str(saved["model_revision"]) != revision:
                raise ValueError("Decision head does not match the pinned backbone revision")
            self._max_length = int(saved["max_length"])
            self._temperature = float(saved["temperature"])
            weights = saved["weights"].copy()
            bias = saved["bias"].copy()
            attention = str(saved["attention_implementation"]) if "attention_implementation" in saved else "eager"
            self._prompt_style = str(saved["prompt_style"]) if "prompt_style" in saved else "candidate_v1"
            model_dtype = str(saved["model_dtype"]) if "model_dtype" in saved else "bfloat16"
        if self._max_length != report.get("max_length") or not 1 <= self._max_length <= 4096:
            raise ValueError("Decision head token limit differs from its report")
        if attention not in {"eager", "sdpa"} or attention != report.get("attention_implementation", "eager"):
            raise ValueError("Decision head attention mode differs from its report")
        if self._prompt_style not in PROMPT_STYLES or self._prompt_style != report.get("prompt_style", "candidate_v1"):
            raise ValueError("Decision head prompt style differs from its report")
        if model_dtype not in {"bfloat16", "float16"} or model_dtype != report.get("model_dtype", "bfloat16"):
            raise ValueError("Decision head precision differs from its report")
        if weights.ndim != 2 or weights.shape[0] != 1 or bias.size != 1:
            raise ValueError("Decision head has an invalid shape")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("Decision head contains non-finite values")

        # Optional research dependencies are imported only after explicit enablement and
        # local artifact validation. Every HF lookup is cache-only; this code never downloads.
        import torch
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this opt-in local research model")
        free_memory, _ = torch.cuda.mem_get_info()
        if free_memory < 4 * 1024**3:
            raise RuntimeError("At least 4 GiB of free CUDA memory is required for local model scoring")
        transformers_logging.disable_progress_bar()
        model_path = snapshot_download(model, revision=revision, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[model_dtype]
        if model == "Qwen/Qwen3.5-2B":
            from transformers import Qwen3_5ForCausalLM

            base = Qwen3_5ForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                device_map={"": 0},
                attn_implementation=attention,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
            )
            backbone = PeftModel.from_pretrained(base.model, self.artifact_dir / "adapter", is_trainable=False)
        else:
            base = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=dtype,
                device_map={"": 0},
                attn_implementation=attention,
            )
            backbone = PeftModel.from_pretrained(base, self.artifact_dir / "adapter", is_trainable=False)
        del base
        head = nn.Linear(weights.shape[1], 1, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(weights).to("cuda"))
            head.bias.copy_(torch.from_numpy(bias).to("cuda"))
        backbone.eval()
        head.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._backbone = backbone
        self._head = head
        self._identity = {"model": model, "revision": revision}
        self._loaded = True

    def _encode(self, prompts: list[str]) -> list[list[int]]:
        sequences = []
        for prompt in prompts:
            rendered = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            token_ids = self._tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if not token_ids or len(token_ids) > self._max_length:
                raise ValueError("Decision exceeds the trained token limit; input was not truncated")
            sequences.append(token_ids)
        return sequences

    def _score_sequences(self, sequences: list[list[int]]) -> np.ndarray:
        torch = self._torch
        candidate_scores = []
        with torch.inference_mode():
            start = 0
            while start < len(sequences):
                proposed = sequences[start : start + self._microbatch_size]
                longest = max(len(ids) for ids in proposed)
                batch_size = _candidate_microbatch_size(longest, self._microbatch_size)
                batch = sequences[start : start + batch_size]
                lengths = torch.tensor([len(ids) for ids in batch], dtype=torch.long, device="cuda")
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids, dtype=torch.long) for ids in batch],
                    batch_first=True,
                    padding_value=0,
                ).to("cuda")
                attention_mask = torch.arange(input_ids.size(1), device="cuda")[None, :] < lengths[:, None]
                hidden = self._backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                last_hidden = hidden[torch.arange(len(batch), device="cuda"), lengths - 1].float()
                candidate_scores.extend(self._head(last_hidden).squeeze(-1).unbind())
                start += len(batch)
            scores = torch.stack(candidate_scores)
        return scores.float().cpu().numpy().astype(np.float64)

    def unload(self) -> None:
        """Release the optional local model and its CUDA allocation."""
        with self._lock:
            torch = self._torch
            self._backbone = None
            self._head = None
            self._tokenizer = None
            self._torch = None
            self._identity = {}
            self._loaded = False
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
