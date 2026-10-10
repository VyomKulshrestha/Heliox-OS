from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pilot.config import PilotConfig
from pilot.intelligence.native_typed_decision import (
    LocalNextTokenTypedDecisionModel,
    _encode_prompt_and_labels,
    _probabilities_in_request_order,
    _render_question,
)
from pilot.intelligence.typed_decision import (
    MAX_OPTIONS,
    LocalTypedDecisionModel,
    _candidate_microbatch_size,
    candidate_prompts,
    compile_typed_questions,
    format_answer,
    format_typed_answer,
    validate_decision,
)
from pilot.security.rpc_identity import RpcClientRole
from pilot.server import PilotServer


def test_dynamic_typed_decisions_are_normalized_without_task_specific_rules() -> None:
    validate_decision("choice", "Choose an outcome", ["wait", "ask", "continue"])
    prompts = candidate_prompts(
        {
            "kind": "choice",
            "question": "Choose an outcome",
            "options": ["wait", "ask"],
            "state_json": json.dumps({"request": "read only", "screen": {"title": "Settings"}}),
        },
        style="candidate_v1",
    )
    assert len(prompts) == 2
    assert "Settings" in prompts[0]
    assert "Proposed answer: wait" in prompts[0]
    assert format_answer("noul", ["no", "yes"], np.asarray([-1.0, 1.0]), 1.0)["yes_probability"] > 0.5
    score = format_answer("score", ["low", "mid", "high"], np.asarray([0.0, 1.0, 2.0]), 1.0)
    assert score["most_likely"] == "high"
    assert sum(score["probabilities"].values()) == pytest.approx(1.0)


def test_typed_decision_rejects_oversized_or_ambiguous_requests() -> None:
    with pytest.raises(ValueError, match="ordered as no, yes"):
        validate_decision("noul", "Continue?", ["yes", "no"])
    with pytest.raises(ValueError, match="unique"):
        validate_decision("choice", "Choose", ["same", "same"])
    with pytest.raises(ValueError, match=f"2-{MAX_OPTIONS}"):
        validate_decision("choice", "Choose", [str(index) for index in range(MAX_OPTIONS + 1)])


def test_jev_style_typed_questions_compile_without_leaking_score_indices() -> None:
    questions = {
        "is_safe": {
            "type": "noul",
            "instructions": "Can this be read without changing anything?",
            "criteria": {"true": "Only inspect current state", "false": "It changes state"},
        },
        "route": {
            "type": "choice",
            "instructions": "Choose the best route",
            "criteria": {"cache": "Use verified local data", "live": None},
        },
        "quality": {
            "type": "score",
            "instructions": "Rate the evidence quality",
            "criteria": ["weak", "partial", "strong"],
        },
    }
    compiled = compile_typed_questions({"goal": "inspect"}, questions)

    assert compiled[0]["options"] == ["no", "yes"]
    assert "Yes means: Only inspect current state" in compiled[0]["question"]
    assert "No means: It changes state" in compiled[0]["question"]
    assert compiled[1]["options"] == ["cache: Use verified local data", "live"]
    assert compiled[1]["answer_keys"] == ["cache", "live"]
    assert compiled[2]["options"] == ["weak", "partial", "strong"]
    assert compiled[2]["answer_keys"] == ["0", "1", "2"]
    assert all(not option.startswith(("0: ", "1: ", "2: ")) for option in compiled[2]["options"])

    choice = format_typed_answer(compiled[1], [0.1, 0.9])
    assert choice["choice"] == "live"
    assert choice["confidence"] == pytest.approx(0.8)
    assert choice["confidence_semantics"] == ("normalized_distribution_concentration_not_accuracy_probability")
    assert choice["distribution_semantics"] == ("model_assigned_candidate_mass_not_empirical_correctness_probability")
    score = format_typed_answer(compiled[2], [0.2, 0.5, 0.3])
    assert score["score"] == pytest.approx(1.1)
    assert score["confidence_semantics"] == ("normalized_distribution_concentration_not_accuracy_probability")
    assert score["distribution_semantics"] == ("model_assigned_candidate_mass_not_empirical_correctness_probability")
    assert score["legend"] == ["weak", "partial", "strong"]


def test_jev_style_typed_question_validation_is_bounded_and_strict() -> None:
    optional_criteria = compile_typed_questions({}, {"decision": {"type": "noul", "instructions": "Ready?"}})
    assert optional_criteria[0]["options"] == ["no", "yes"]
    assert optional_criteria[0]["question"] == "Ready?"

    max_choice = {f"option-{index}": None for index in range(MAX_OPTIONS)}
    compiled = compile_typed_questions(
        {}, {"decision": {"type": "choice", "instructions": "Choose", "criteria": max_choice}}
    )
    assert len(compiled[0]["options"]) == 255
    with pytest.raises(ValueError, match=f"2-{MAX_OPTIONS}"):
        compile_typed_questions(
            {},
            {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {f"option-{index}": None for index in range(MAX_OPTIONS + 1)},
                }
            },
        )

    with pytest.raises(ValueError, match="true and false"):
        compile_typed_questions(
            {}, {"decision": {"type": "noul", "instructions": "Ready?", "criteria": {"true": "yes"}}}
        )
    with pytest.raises(ValueError, match="candidate options"):
        compile_typed_questions(
            {},
            {
                f"q{index}": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {f"o{option}": None for option in range(65)},
                }
                for index in range(4)
            },
        )
    with pytest.raises(ValueError, match="probabilities must sum"):
        format_typed_answer(
            {"kind": "choice", "options": ["a", "b"], "answer_keys": ["a", "b"]},
            [0.4, 0.4],
        )


def test_typed_question_batch_submits_all_candidates_in_one_scoring_call(monkeypatch) -> None:
    model = LocalTypedDecisionModel("unused", "unused")
    model._prompt_style = "candidate_v1"
    model._identity = {"model": "local-test", "revision": "pinned-test"}
    model._temperature = 1.0
    monkeypatch.setattr(model, "_load", lambda: None)
    monkeypatch.setattr(model, "_encode", lambda prompts: [[index] for index, _ in enumerate(prompts)])
    calls = []

    def score_sequences(sequences):
        calls.append(len(sequences))
        return np.asarray([1.0] * len(sequences))

    monkeypatch.setattr(model, "_score_sequences", score_sequences)
    result = model.score_questions(
        {"goal": "inspect"},
        {
            "route": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"read": None, "write": None},
            },
            "safe": {
                "type": "noul",
                "instructions": "Is it safe?",
                "criteria": {"true": "read only", "false": "changes state"},
            },
            "quality": {
                "type": "score",
                "instructions": "Rate it",
                "criteria": ["low", "high"],
            },
        },
    )

    assert calls == [5]
    assert result["research_only"] is True
    assert result["execution_authority"] is False
    assert result["answers"]["route"]["choice"] == "read"
    assert result["answers"]["safe"]["noul"] == pytest.approx(1 / (1 + np.exp(-1)))
    assert result["answers"]["safe"]["distribution_semantics"] == (
        "model_assigned_candidate_mass_not_empirical_correctness_probability"
    )
    assert result["answers"]["quality"]["score"] == pytest.approx(0.5)


def test_candidate_microbatch_caps_long_prompts_and_respects_lower_limits() -> None:
    assert _candidate_microbatch_size(512, 8) == 8
    assert _candidate_microbatch_size(513, 8) == 4
    assert _candidate_microbatch_size(768, 8) == 4
    assert _candidate_microbatch_size(769, 8) == 2
    assert _candidate_microbatch_size(1_024, 1) == 1
    assert _candidate_microbatch_size(512, 16) == 16
    assert _candidate_microbatch_size(513, 16) == 4
    assert _candidate_microbatch_size(768, 16) == 4
    assert _candidate_microbatch_size(769, 16) == 2
    with pytest.raises(ValueError, match="must be positive"):
        _candidate_microbatch_size(0, 8)


def test_native_next_token_labels_are_verified_and_choice_order_is_stable() -> None:
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return (
                "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
            )

        def __call__(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [ord(character) for character in text]}

    record = {
        "id": "route",
        "kind": "choice",
        "question": "Choose one",
        "options": ["zebra", "apple"],
        "answer_keys": ["z", "a"],
    }
    prompt_ids, label_ids, scored_keys, labels = _encode_prompt_and_labels(
        CharacterTokenizer(), {"screen": "page"}, record, max_length=4096
    )

    assert prompt_ids
    assert len(label_ids) == len(set(label_ids)) == 2
    assert labels == ["A", "B"]
    assert scored_keys == ["a", "z"]
    rendered = _render_question("{}", record, [("A", "private-a", "apple"), ("B", "private-z", "zebra")])
    assert '"answer":' not in rendered
    assert "private-" not in rendered


def test_native_next_token_reassigns_duplicate_candidate_tokens() -> None:
    class AliasingTokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return (
                "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
            )

        def __call__(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [300 if character in "AB" else ord(character) for character in text]}

    record = {
        "id": "route",
        "kind": "choice",
        "question": "Choose one",
        "options": ["zebra", "apple"],
        "answer_keys": ["z", "a"],
    }
    _, label_ids, scored_keys, labels = _encode_prompt_and_labels(AliasingTokenizer(), "page", record, max_length=4096)

    assert labels == ["A", "C"]
    assert len(label_ids) == len(set(label_ids)) == 2
    assert scored_keys == ["a", "z"]


def test_native_next_token_scores_map_back_to_original_candidate_order() -> None:
    record = {
        "kind": "choice",
        "options": ["zebra", "apple"],
        "answer_keys": ["z", "a"],
    }
    probabilities = _probabilities_in_request_order(
        record,
        ["a", "z"],
        np.asarray([3.0, 0.0]),
        1.0,
    )
    assert probabilities == pytest.approx([0.047425873, 0.952574127])
    result = format_typed_answer(record, probabilities)
    assert result["choice"] == "a"


def test_native_noul_uses_probabilistic_ratings_not_binary_yes_no_labels() -> None:
    class CharacterTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, **_kwargs):
            return (
                "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
            )

        def __call__(self, text, *, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": [ord(character) for character in text]}

    record = {
        "id": "truth",
        "kind": "noul",
        "question": "Is the statement supported?",
        "options": ["no", "yes"],
        "answer_keys": ["false", "true"],
    }
    prompt_ids, label_ids, scored_keys, labels = _encode_prompt_and_labels(
        CharacterTokenizer(), {"statement": "The claim is present."}, record, max_length=4096
    )
    probabilities = _probabilities_in_request_order(
        record,
        scored_keys,
        np.asarray([-8.0] * 8 + [8.0]),
        1.0,
    )

    assert prompt_ids
    assert labels == list("ABCDEFGHI")
    assert len(label_ids) == 9
    assert scored_keys == [str(rating) for rating in range(1, 10)]
    assert probabilities == pytest.approx([0.01, 0.99], abs=1e-6)


def test_native_next_token_model_requires_pinned_model_and_valid_limits() -> None:
    with pytest.raises(ValueError, match="pinned local"):
        LocalNextTokenTypedDecisionModel("untrusted/model")
    with pytest.raises(ValueError, match="token limit"):
        LocalNextTokenTypedDecisionModel("Qwen/Qwen3-0.6B", max_length=32)
    with pytest.raises(ValueError, match="temperature"):
        LocalNextTokenTypedDecisionModel("Qwen/Qwen3-0.6B", temperature=0)


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"])
def test_native_next_token_model_refuses_loading_under_host_memory_pressure(monkeypatch, model_name) -> None:
    torch_module = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1024**3)),
    )
    model = LocalNextTokenTypedDecisionModel(model_name)

    with pytest.raises(RuntimeError, match="available system RAM"):
        model._load()

    torch_module.cuda.is_available.assert_not_called()
    assert model._loaded is False


def test_artifact_verification_fails_before_any_model_or_network_load(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"local candidate")
    head_file = tmp_path / "head.npz"
    np.savez(head_file, model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc")
    report_file = tmp_path / "report.json"
    report_file.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3.5-2B",
                "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                "head_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    model = LocalTypedDecisionModel(tmp_path, report_file)
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.score({}, "choice", "Choose", ["one", "two"])
    assert model._loaded is False


@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"])
@pytest.mark.parametrize("low_host_memory", [False, True])
def test_artifact_load_attaches_adapter_to_decoder_and_preserves_batch_limit(
    monkeypatch, tmp_path: Path, model_name, low_host_memory
):
    from pilot.intelligence.typed_decision import PINNED_BACKBONES

    revision = PINNED_BACKBONES[model_name]
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_file.write_bytes(b"mock adapter")
    head_file = tmp_path / "head.npz"
    np.savez(
        head_file,
        model_revision=revision,
        max_length=128,
        temperature=1.0,
        weights=np.zeros((1, 2), dtype=np.float32),
        bias=np.zeros(1, dtype=np.float32),
    )
    report_file = tmp_path / "report.json"
    report_file.write_text(
        json.dumps(
            {
                "model": model_name,
                "revision": revision,
                "max_length": 128,
                "candidate_microbatch_size": 1,
                "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
                "head_sha256": hashlib.sha256(head_file.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    decoder = MagicMock()
    loader = MagicMock(return_value=SimpleNamespace(model=decoder))
    adapter_loader = MagicMock(return_value=decoder)
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.cuda.mem_get_info.return_value = (8 * 1024**3, 8 * 1024**3)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=MagicMock(return_value="cache"))
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=adapter_loader)))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            Qwen3_5ForCausalLM=SimpleNamespace(from_pretrained=loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=MagicMock()),
            BitsAndBytesConfig=MagicMock(),
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers.utils", SimpleNamespace(logging=MagicMock()))
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=(1 if low_host_memory else 8) * 1024**3)),
    )
    model = LocalTypedDecisionModel(tmp_path, report_file)
    if low_host_memory:
        with pytest.raises(RuntimeError, match="available system RAM"):
            model._load()
        loader.assert_not_called()
        adapter_loader.assert_not_called()
        assert model._loaded is False
        return
    model._load()
    adapter_loader.assert_called_once_with(decoder, adapter, is_trainable=False)
    assert model._loaded is True
    assert model._backbone is decoder
    assert model._microbatch_size == 1


@pytest.mark.parametrize("microbatch_size", [0, -1, True, "1"])
def test_runtime_rejects_invalid_batch_metadata_before_model_load(tmp_path: Path, microbatch_size):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "candidate_microbatch_size": microbatch_size,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="microbatch size"):
        LocalTypedDecisionModel(tmp_path, report)._load()


@pytest.mark.asyncio
async def test_typed_decision_rpc_is_opt_in_and_research_only(monkeypatch, tmp_path: Path) -> None:
    config = PilotConfig()
    server = PilotServer(config)
    client = object()
    server._client_roles[client] = RpcClientRole.MCP_LOCAL
    payload = {
        "state": {"goal": "inspect", "candidate": "collect live evidence"},
        "kind": "choice",
        "question": "Which candidate fits the task?",
        "options": ["collect live evidence", "change configuration"],
    }

    disabled = await server._handle_mcp_typed_decision(payload, client)
    assert disabled["status"] == "disabled"
    assert disabled["execution_authority"] is False

    config.typed_decision.enabled = True
    config.typed_decision.artifact_dir = str(tmp_path / "artifacts")
    config.typed_decision.report_path = str(tmp_path / "report.json")
    invalid = await server._handle_mcp_typed_decision({**payload, "state": None}, client)
    assert invalid["status"] == "error"
    assert invalid["research_only"] is True
    assert invalid["execution_authority"] is False
    calls = []

    class FakeModel:
        def __init__(self, artifact_dir, report_path) -> None:
            calls.append((artifact_dir, report_path))

        def score(self, state, kind, question, options):
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "answer": {"type": "choice", "choice": options[0]},
            }

        def score_questions(self, state, questions):
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "answers": {question_id: {"type": definition["type"]} for question_id, definition in questions.items()},
            }

        def unload(self):
            calls.append("unload")

    import pilot.intelligence.typed_decision as typed_decision_module

    monkeypatch.setattr(typed_decision_module, "LocalTypedDecisionModel", FakeModel)
    result = await server._handle_mcp_typed_decision(payload, client)
    assert result["execution_authority"] is False
    assert result["answer"]["choice"] == "collect live evidence"
    assert calls == [(str(tmp_path / "artifacts"), str(tmp_path / "report.json"))]

    questions = {
        "ready": {
            "type": "noul",
            "instructions": "Is the inspection read only?",
            "criteria": {"true": "No changes", "false": "Changes data"},
        }
    }
    batched = await server._handle_mcp_typed_decision({"state": "inspect", "questions": questions}, client)
    assert batched["execution_authority"] is False
    assert batched["answers"]["ready"]["type"] == "noul"

    config.typed_decision.enabled = False
    await server._handle_mcp_typed_decision(payload, client)
    assert server._typed_decision_engine is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("CUDA is required for this opt-in local research model"),
        ModuleNotFoundError("No module named 'peft'"),
        FileNotFoundError("Configured local decision artifacts were not found"),
    ],
)
async def test_typed_decision_rpc_reports_unavailable_runtime_without_internal_error(
    monkeypatch, tmp_path: Path, failure: Exception
) -> None:
    config = PilotConfig()
    config.typed_decision.enabled = True
    config.typed_decision.artifact_dir = str(tmp_path / "artifacts")
    config.typed_decision.report_path = str(tmp_path / "report.json")
    server = PilotServer(config)
    client = object()
    server._client_roles[client] = RpcClientRole.MCP_LOCAL

    class UnavailableModel:
        def __init__(self, artifact_dir, report_path) -> None:
            pass

        def score(self, state, kind, question, options):
            raise failure

        def unload(self):
            return None

    import pilot.intelligence.typed_decision as typed_decision_module

    monkeypatch.setattr(typed_decision_module, "LocalTypedDecisionModel", UnavailableModel)
    result = await server._handle_mcp_typed_decision(
        {"state": "inspect", "kind": "noul", "question": "Ready?", "options": ["no", "yes"]},
        client,
    )

    assert result["status"] == "unavailable"
    assert result["research_only"] is True
    assert result["execution_authority"] is False
    assert str(failure) in result["message"]


@pytest.mark.asyncio
async def test_native_next_token_mode_needs_no_head_artifact_and_stays_advisory(monkeypatch) -> None:
    config = PilotConfig()
    config.typed_decision.enabled = True
    config.typed_decision.scoring_mode = "native_next_token"
    client = object()
    server = PilotServer(config)
    server._client_roles[client] = RpcClientRole.MCP_LOCAL
    created = []

    class FakeNativeModel:
        def __init__(self, model_id, *, max_length, temperature) -> None:
            created.append((model_id, max_length, temperature))

        def score(self, state, kind, question, options):
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "scoring_method": "native_next_token_v1",
                "calibration": "uncalibrated",
                "answer": {"type": kind, "choice": options[0]},
            }

        def score_questions(self, state, questions):
            return {
                "status": "ok",
                "research_only": True,
                "execution_authority": False,
                "answers": {key: {"type": value["type"]} for key, value in questions.items()},
            }

        def unload(self):
            return None

    import pilot.intelligence.native_typed_decision as native_decision_module

    monkeypatch.setattr(native_decision_module, "LocalNextTokenTypedDecisionModel", FakeNativeModel)
    result = await server._handle_mcp_typed_decision(
        {
            "state": "An untrusted textual context",
            "kind": "choice",
            "question": "Pick",
            "options": ["a", "b"],
        },
        client,
    )

    assert created == [("Qwen/Qwen3-0.6B", 2048, 1.0)]
    assert result["scoring_method"] == "native_next_token_v1"
    assert result["calibration"] == "uncalibrated"
    assert result["execution_authority"] is False


@pytest.mark.asyncio
async def test_typed_decision_config_requires_explicit_paths_and_unloads_on_disable(tmp_path: Path) -> None:
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    invalid = await server._handle_update_config(
        {"section": "typed_decision", "values": {"enabled": True}},
        object(),
    )
    assert invalid["status"] == "error"
    assert config.typed_decision.enabled is False
    config.save.assert_not_called()

    saved = await server._handle_update_config(
        {
            "section": "typed_decision",
            "values": {
                "enabled": True,
                "artifact_dir": str(tmp_path / "artifacts"),
                "report_path": str(tmp_path / "report.json"),
            },
        },
        object(),
    )
    assert saved == {"status": "ok"}
    assert config.typed_decision.enabled is True

    unloaded = []
    server._typed_decision_engine = SimpleNamespace(unload=lambda: unloaded.append(True))
    server._typed_decision_engine_key = (
        "trained_head",
        str(Path(config.typed_decision.artifact_dir).expanduser()),
        str(Path(config.typed_decision.report_path).expanduser()),
        config.typed_decision.model_id,
        config.typed_decision.max_length,
        config.typed_decision.temperature,
    )
    disabled = await server._handle_update_config(
        {"section": "typed_decision", "values": {"enabled": False}},
        object(),
    )
    assert disabled == {"status": "ok"}
    assert unloaded == [True]
    assert server._typed_decision_engine is None


@pytest.mark.asyncio
async def test_native_next_token_config_can_be_enabled_without_training_artifacts() -> None:
    config = PilotConfig()
    config.save = MagicMock()
    server = PilotServer(config)
    invalid = await server._handle_update_config(
        {"section": "typed_decision", "values": {"enabled": True}},
        object(),
    )
    assert invalid["status"] == "error"
    config.save.assert_not_called()

    saved = await server._handle_update_config(
        {
            "section": "typed_decision",
            "values": {
                "enabled": True,
                "scoring_mode": "native_next_token",
                "model_id": "Qwen/Qwen3-0.6B",
            },
        },
        object(),
    )
    assert saved == {"status": "ok"}
    assert config.typed_decision.scoring_mode == "native_next_token"
    assert config.typed_decision.enabled is True


@pytest.mark.asyncio
async def test_native_next_token_config_keeps_cached_model_when_paths_are_empty() -> None:
    config = PilotConfig()
    config.typed_decision.enabled = True
    config.typed_decision.scoring_mode = "native_next_token"
    config.save = MagicMock()
    server = PilotServer(config)
    unloaded = []
    server._typed_decision_engine = SimpleNamespace(unload=lambda: unloaded.append(True))
    server._typed_decision_engine_key = (
        "native_next_token",
        "",
        "",
        config.typed_decision.model_id,
        config.typed_decision.max_length,
        config.typed_decision.temperature,
    )

    result = await server._handle_update_config(
        {"section": "typed_decision", "values": {"temperature": 1.0}},
        object(),
    )

    assert result == {"status": "ok"}
    assert unloaded == []


def test_runtime_entry_points_render_the_offline_training_prompt(monkeypatch) -> None:
    from benchmarks.general_decision import candidate_prompts as offline_candidate_prompts
    from scripts.score_general_decision import prepare_request

    state = {"request": "Summarize only", "proposed_action": "Email the summary"}
    question = "Is this exact proposed action authorized?\n"
    model = LocalTypedDecisionModel("C:/unused", "C:/unused.json")
    model._loaded = True
    model._prompt_style = "candidate_v1"
    model._identity = {"model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
    rendered = []

    def encode(prompts):
        rendered.append(prompts)
        return [[1] for _ in prompts]

    monkeypatch.setattr(model, "_encode", encode)
    monkeypatch.setattr(model, "_score_sequences", lambda sequences: np.zeros(len(sequences)))
    model.score(state, "noul", question, ["no", "yes"])
    model.score_questions(state, {"decision": {"type": "noul", "instructions": question}})
    offline = offline_candidate_prompts(
        prepare_request({"state": state, "kind": "noul", "question": question, "options": ["no", "yes"]}),
        style="candidate_v1",
    )

    assert rendered[0] == rendered[1] == offline


def test_kind_temperatures_must_match_the_report_and_drive_runtime_scaling(monkeypatch) -> None:
    from pilot.intelligence.typed_decision import parse_kind_temperatures

    saved = json.dumps({"noul": 3.0, "choice": 1.5}, sort_keys=True)
    report = {"kind_temperatures": {"noul": 3.0, "choice": 1.5}}
    assert parse_kind_temperatures(saved, report) == {"noul": 3.0, "choice": 1.5}
    assert parse_kind_temperatures(None, {}) == {}
    for bad_saved, bad_report in (
        (None, report),
        (saved, {"kind_temperatures": {"noul": 2.0, "choice": 1.5}}),
        (json.dumps({"verdict": 1.0}), {"kind_temperatures": {"verdict": 1.0}}),
        (json.dumps({"noul": 0.0}), {"kind_temperatures": {"noul": 0.0}}),
        ("not json", report),
    ):
        with pytest.raises(ValueError, match="per-kind temperatures"):
            parse_kind_temperatures(bad_saved, bad_report)

    model = LocalTypedDecisionModel("C:/unused", "C:/unused.json")
    model._loaded = True
    model._prompt_style = "candidate_v1"
    model._temperature = 1.0
    model._kind_temperatures = {"noul": 4.0}
    model._identity = {"model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca"}
    monkeypatch.setattr(model, "_encode", lambda prompts: [[1] for _ in prompts])
    monkeypatch.setattr(model, "_score_sequences", lambda sequences: np.full(len(sequences), 4.0))
    noul = model.score("state", "noul", "Ready?", ["no", "yes"])["answer"]["yes_probability"]
    choice = model.score("state", "choice", "Pick", ["a", "b"])["answer"]["probabilities"]["a"]
    assert noul == pytest.approx(1 / (1 + np.exp(-1.0)))
    assert choice == pytest.approx(0.5)
    batched = model.score_questions("state", {"ready": {"type": "noul", "instructions": "Ready?"}})
    assert batched["answers"]["ready"]["noul"] == pytest.approx(noul)
