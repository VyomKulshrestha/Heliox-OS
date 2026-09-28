"""Pack typed questions into independent candidate branches for a pointer scorer.

This research-only encoder shares the request state once, while its attention
mask keeps questions and candidates isolated. It does not grant execution
authority. The layout follows public packed-decision research; it does not use
or imitate proprietary model outputs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)
MAX_PACKED_MASK_BYTES = 16 * 1024 * 1024
MAX_PACKED_QUESTIONS = 64
MAX_PACKED_OPTIONS = 255
MAX_PACKED_TOTAL_OPTIONS = 256
MAX_STATE_CHARS = 16_384
MAX_QUESTION_CHARS = 4_000
MAX_OPTION_CHARS = 1_024
_CONTROL_TOKEN = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


@dataclass(frozen=True)
class PackedQuestion:
    """Indices needed to read one question from a packed model pass."""

    question_id: str
    kind: str
    query_index: int
    question_indices: tuple[int, ...]
    branch_end_index: int
    option_indices: tuple[int, ...]
    option_spans: tuple[tuple[int, ...], ...]
    answer_keys: tuple[str, ...]
    options: tuple[str, ...]


@dataclass(frozen=True)
class PackedTypedRequest:
    """Single shared-state token stream and its block-causal attention mask."""

    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    attention_mask: np.ndarray
    questions: tuple[PackedQuestion, ...]
    state_token_count: int


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("State must contain JSON values") from exc


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    # Prevent caller-provided text from being interpreted as one of our delimiters.
    safe_text = _CONTROL_TOKEN.sub(r"<¦\1¦>", text)
    return [int(token_id) for token_id in tokenizer.encode(safe_text, add_special_tokens=False)]


def _marker_ids(tokenizer: Any) -> tuple[int, ...]:
    marker_ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if (
        any(token_id is None for token_id in marker_ids)
        or len(set(marker_ids)) != len(marker_ids)
        or (unknown is not None and unknown in marker_ids)
    ):
        raise ValueError("Tokenizer does not provide distinct packed-decision control tokens")
    return tuple(int(token_id) for token_id in marker_ids)


def pack_typed_questions(
    tokenizer: Any,
    state: Any,
    records: list[dict[str, Any]],
    *,
    max_sequence_length: int,
) -> PackedTypedRequest:
    """Encode many independent typed questions in one shared-state sequence.

    Choice candidates are canonicalized by their rendered text, with answer keys
    carried alongside them, so changing caller order cannot change model input.
    Noul and ordered-score candidates preserve their semantic order. The length
    bound is checked before allocating the quadratic attention mask.
    """
    if not isinstance(max_sequence_length, int) or isinstance(max_sequence_length, bool) or max_sequence_length < 1:
        raise ValueError("Packed sequence limit must be a positive integer")
    if (
        not isinstance(records, list)
        or not 1 <= len(records) <= MAX_PACKED_QUESTIONS
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError(f"Records must contain 1-{MAX_PACKED_QUESTIONS} typed questions")

    state_text = _state_text(state)
    if len(state_text) > MAX_STATE_CHARS:
        raise ValueError(f"State exceeds {MAX_STATE_CHARS} characters")
    marker_ids = _marker_ids(tokenizer)
    state_ids = [marker_ids[0], *_token_ids(tokenizer, state_text)]
    input_ids = list(state_ids)
    position_ids = list(range(len(state_ids)))
    state_indices = tuple(range(len(state_ids)))
    packed_questions: list[PackedQuestion] = []
    branch_masks: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]] = []
    total_options = 0
    question_ids: set[str] = set()

    q_token, option_token, end_option_token = marker_ids[1:4]
    for record in records:
        if set(record) != {"id", "kind", "question", "options", "answer_keys"}:
            raise ValueError("Each typed question must have id, kind, question, options, and answer_keys")
        if (
            not isinstance(record["id"], str)
            or not record["id"]
            or len(record["id"]) > 128
            or record["id"] in question_ids
            or not isinstance(record["kind"], str)
            or record["kind"] not in {"choice", "noul", "score"}
            or not isinstance(record["question"], str)
            or not record["question"].strip()
            or len(record["question"]) > MAX_QUESTION_CHARS
            or not isinstance(record["options"], list)
            or not isinstance(record["answer_keys"], list)
        ):
            raise ValueError("Typed question fields are invalid")
        answer_keys = list(record["answer_keys"])
        options = list(record["options"])
        if (
            len(answer_keys) != len(options)
            or not 2 <= len(options) <= MAX_PACKED_OPTIONS
            or any(not isinstance(key, str) or not key for key in answer_keys)
            or any(
                not isinstance(option, str) or not option.strip() or len(option) > MAX_OPTION_CHARS
                for option in options
            )
            or len(set(options)) != len(options)
            or len(set(answer_keys)) != len(answer_keys)
        ):
            raise ValueError("Packed questions need one answer key for every candidate")
        total_options += len(options)
        if total_options > MAX_PACKED_TOTAL_OPTIONS:
            raise ValueError(f"A packed request may contain at most {MAX_PACKED_TOTAL_OPTIONS} candidates")
        if record["kind"] == "noul" and options != ["no", "yes"]:
            raise ValueError("Noul candidates must be ordered as no, yes")
        question_ids.add(record["id"])
        pairs = list(zip(answer_keys, options, strict=True))
        if record["kind"] == "choice":
            pairs.sort(key=lambda pair: (pair[1].casefold(), pair[0]))

        instruction_ids = [q_token, *_token_ids(tokenizer, record["question"])]
        input_ids.extend(instruction_ids)
        instruction_indices = tuple(range(len(input_ids) - len(instruction_ids), len(input_ids)))
        branch_base_position = len(state_ids) + len(instruction_ids)
        position_ids.extend(range(len(state_ids), branch_base_position))

        option_indices: list[int] = []
        option_spans: list[tuple[int, ...]] = []
        option_position_cursor = branch_base_position
        for _, option in pairs:
            span_ids = [option_token, *_token_ids(tokenizer, option), end_option_token]
            span_start = len(input_ids)
            input_ids.extend(span_ids)
            span = tuple(range(span_start, span_start + len(span_ids)))
            option_spans.append(span)
            option_indices.append(span[-1])

            if record["kind"] == "choice":
                relative_positions = tuple(range(branch_base_position, branch_base_position + len(span_ids)))
            else:
                relative_positions = tuple(range(option_position_cursor, option_position_cursor + len(span_ids)))
                option_position_cursor += len(span_ids)
            position_ids.extend(relative_positions)

        packed_question = PackedQuestion(
            question_id=record["id"],
            kind=record["kind"],
            # The query is question-only: it must not read candidate tokens.
            # Candidate vectors are scored against it after the encoder pass.
            query_index=instruction_indices[-1],
            question_indices=instruction_indices,
            branch_end_index=len(input_ids),
            option_indices=tuple(option_indices),
            option_spans=tuple(option_spans),
            answer_keys=tuple(key for key, _ in pairs),
            options=tuple(option for _, option in pairs),
        )
        packed_questions.append(packed_question)
        branch_masks.append((instruction_indices, tuple(option_spans)))

    if len(input_ids) > max_sequence_length:
        raise ValueError(f"Packed request has {len(input_ids)} tokens; model limit is {max_sequence_length}")
    if len(input_ids) ** 2 > MAX_PACKED_MASK_BYTES:
        raise ValueError(f"Packed attention mask would exceed the {MAX_PACKED_MASK_BYTES}-byte memory budget")

    allowed = np.zeros((len(input_ids), len(input_ids)), dtype=np.bool_)
    for query_index in state_indices:
        allowed[query_index, : query_index + 1] = True

        for instruction_indices, option_spans in branch_masks:
            for instruction_offset, query_index in enumerate(instruction_indices):
                allowed[query_index, state_indices] = True
                allowed[query_index, instruction_indices[: instruction_offset + 1]] = True
            for option_span in option_spans:
                for option_offset, query_index in enumerate(option_span):
                    allowed[query_index, state_indices] = True
                    allowed[query_index, instruction_indices] = True
                    allowed[query_index, option_span[: option_offset + 1]] = True
    return PackedTypedRequest(
        input_ids=tuple(input_ids),
        position_ids=tuple(position_ids),
        attention_mask=allowed,
        questions=tuple(packed_questions),
        state_token_count=len(state_ids),
    )
