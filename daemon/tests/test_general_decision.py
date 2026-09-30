"""General typed-decision data contract and leakage checks."""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pytest

import scripts.train_general_decision as trainer
from benchmarks.general_decision import (
    candidate_prompts,
    correctness_features,
    correctness_metrics,
    fit_correctness_scorer,
    format_answer,
    grouped_mean_bootstrap_delta,
    licensed_row,
    metrics,
    paired_aurc_bootstrap_delta,
    paired_mean_bootstrap_delta,
    predict_correctness,
    probabilities,
    ranked_probability_score,
    select_balanced,
    select_source_holdout,
    selective_risk_metrics,
    uniform_baseline,
    validate_decision,
    validate_row,
    verify_disjoint,
    verify_source_holdout,
)
from benchmarks.scopejudge_general import scope_group_weights
from scripts.train_general_decision import (
    checkpoint_signature,
    general_group_weights,
    require_free_host_ram,
    split_id_group_hash,
    training_row_weights,
)


def row(kind: str = "choice") -> dict:
    return {"id": "item", "group_id": "parent", "source": "source", "kind": kind,
            "state_json": json.dumps({"counter": 3}), "question": "Select a result.",
            "options": ["no", "yes"] if kind == "noul" else ["left", "right", "wait"],
            "target": [0.0, 1.0] if kind == "noul" else [0.0, 1.0, 0.0],
            "metadata_json": json.dumps({"provenance": {"license": "CC0-1.0"}})}


def test_gpu_training_preflight_accepts_sufficient_host_memory() -> None:
    available = int(3.0 * 1024**3)
    assert require_free_host_ram(available, 2.5) == available


def test_gpu_training_preflight_rejects_low_host_memory() -> None:
    with pytest.raises(RuntimeError, match="only 0.61 GiB is available"):
        require_free_host_ram(int(0.61 * 1024**3), 2.5)


@pytest.mark.parametrize(("available", "minimum"), [(-1, 2.5), (1, float("nan")), (1, 0.25)])
def test_gpu_training_preflight_rejects_invalid_values(available: int, minimum: float) -> None:
    with pytest.raises(ValueError, match="Host-memory preflight"):
        require_free_host_ram(available, minimum)


@pytest.mark.parametrize("kind,count", [("noul", 1), ("choice", 3), ("score", 3)])
def test_candidate_prompts_share_context_without_labels(kind: str, count: int) -> None:
    example = row(kind)
    validate_row(example)
    prompts = candidate_prompts(example)
    assert len(prompts) == count
    assert all("counter" in prompt and "Select a result" in prompt for prompt in prompts)
    assert all("parent" not in prompt and "source" not in prompt for prompt in prompts)
    if kind == "noul":
        assert "Proposed answer" not in prompts[0]
    else:
        assert all("Proposed answer: " + option in prompt
                   for prompt, option in zip(prompts, example["options"], strict=True))


def test_full_option_prompts_share_choices_without_labels() -> None:
    example = row("choice")
    prompts = candidate_prompts(example, style="full_options_v2")
    assert len(prompts) == len(example["options"])
    assert all("Available answers:\n1. left\n2. right\n3. wait" in prompt for prompt in prompts)
    assert all("parent" not in prompt and "target" not in prompt for prompt in prompts)
    assert "Available answers:" not in candidate_prompts(example)[0]
    with pytest.raises(ValueError, match="prompt style"):
        candidate_prompts(example, style="unknown")


def test_reject_unverified_rights_and_invalid_targets() -> None:
    example = row()
    assert licensed_row(example)
    example["metadata_json"] = json.dumps({"provenance": {
        "license": "Generated conversations: CC0-1.0; upstream question descriptions: license not verified"
    }})
    assert not licensed_row(example)
    example["target"] = [0.1, 0.8, 0.0]
    with pytest.raises(ValueError, match="sum to one"):
        validate_row(example)


@pytest.mark.parametrize("url", [
    "https://typesafe.ai/evals/example",
    "https://evals.typesafe.ai/customer_service",
    "typesafe.ai/docs/example",
])
def test_training_filter_excludes_provider_origin_sources(url: str) -> None:
    example = row()
    example["metadata_json"] = json.dumps({"provenance": {
        "license": "CC0-1.0", "source_url": url,
    }})
    assert not licensed_row(example)


@pytest.mark.parametrize("url", [
    "https://nottypesafe.ai/evals/example",
    "https://typesafe.ai.attacker.example/evals/example",
    "https://example.org/typesafe.ai/evals/example",
])
def test_training_filter_does_not_reject_unrelated_url_text(url: str) -> None:
    example = row()
    example["metadata_json"] = json.dumps({"provenance": {
        "license": "CC0-1.0", "source_url": url,
    }})
    assert licensed_row(example)


def test_training_filter_searches_nested_provenance_urls() -> None:
    example = row()
    example["metadata_json"] = json.dumps({"provenance": {
        "license": "CC0-1.0",
        "references": [{"uri": "https://docs.typesafe.ai/patterns/fan-out.md"}],
    }})
    assert not licensed_row(example)


def test_group_disjoint_and_balanced_sampling() -> None:
    examples = []
    for source in ("a", "b"):
        for kind in ("choice", "score"):
            for index in range(3):
                examples.append({**row(kind), "id": f"{source}:{kind}:{index}",
                                 "group_id": f"{source}:{kind}:{index}", "source": source})
    first = select_balanced(examples, 8, 12)
    second = select_balanced(examples, 8, 12)
    assert [entry["id"] for entry in first] == [entry["id"] for entry in second]
    assert {(entry["source"], entry["kind"]) for entry in first} == {
        ("a", "choice"), ("a", "score"), ("b", "choice"), ("b", "score")}
    verify_disjoint({"train": first, "test": [examples[-1]]})
    with pytest.raises(ValueError, match="Group leakage"):
        verify_disjoint({"train": first, "test": [first[0]]})


def test_balanced_sampler_is_reproducible_and_balances_categories() -> None:
    examples = [{**row(kind), "id": f"{source}:{kind}:{index}",
                 "group_id": f"{source}:{kind}:{index}", "source": source}
                for source in ("a", "b") for kind in ("choice", "score")
                for index in range(30)]
    first = select_balanced(examples, 9, 19)
    second = select_balanced(examples, 9, 19)
    assert [entry["id"] for entry in first] == [entry["id"] for entry in second]
    counts = Counter((entry["source"], entry["kind"]) for entry in first)
    assert len(first) == 9
    assert len(counts) == 4
    assert max(counts.values()) - min(counts.values()) <= 1


def test_source_holdout_is_deterministic_and_excludes_nonshared_sources() -> None:
    counts = {
        "train": {"alpha": 30, "beta": 20, "gamma": 10, "train-only": 7},
        "calibration": {"alpha": 8, "beta": 6, "gamma": 4},
        "test": {"alpha": 8, "beta": 6, "gamma": 4},
        "ood": {"alpha": 8, "beta": 6, "gamma": 4, "ood-only": 5},
    }
    first = select_source_holdout(counts, 0.2, 17)
    assert first == select_source_holdout(counts, 0.2, 17)
    assert first["common_sources"] == ["alpha", "beta", "gamma"]
    assert len(first["held_out_sources"]) == 1
    assert set(first["seen_sources"]).isdisjoint(first["held_out_sources"])
    assert set(first["seen_sources"] + first["held_out_sources"]) == set(first["common_sources"])
    disabled = select_source_holdout(counts, 0.0, 17)
    assert disabled["held_out_sources"] == []


def test_source_holdout_rejects_invalid_or_unidentifiable_sources() -> None:
    counts = {name: {"alpha": 1} for name in ("train", "calibration", "test", "ood")}
    with pytest.raises(ValueError, match="at least two"):
        select_source_holdout(counts, 0.2, 1)
    with pytest.raises(ValueError, match="fraction"):
        select_source_holdout(counts, 1.0, 1)
    with pytest.raises(ValueError, match="include"):
        select_source_holdout({"train": {}}, 0.2, 1)


def test_source_holdout_verifier_rejects_seen_source_leakage() -> None:
    split_rows = {name: [{**row(), "source": "seen"}]
                  for name in ("train", "calibration", "test", "ood")}
    split_rows["ood"] = [{**row(), "source": "held-out"}]
    verify_source_holdout(split_rows, {"held-out"})
    split_rows["test"] = [{**row(), "source": "held-out"}]
    with pytest.raises(ValueError, match="leaked into test"):
        verify_source_holdout(split_rows, {"held-out"})
    split_rows["test"] = [{**row(), "source": "seen"}]
    with pytest.raises(ValueError, match="OOD rows"):
        verify_source_holdout(split_rows, {"another-source"})


def test_split_hash_binds_sorted_ids_and_group_assignments() -> None:
    items = [{"row": {"id": "b", "group_id": "g2"}},
             {"row": {"id": "a", "group_id": "g1"}}]
    assert split_id_group_hash(items) == split_id_group_hash(list(reversed(items)))
    changed = [{"row": {"id": "b", "group_id": "g1"}}, items[1]]
    assert split_id_group_hash(items) != split_id_group_hash(changed)


def test_variable_option_metrics_and_temperature() -> None:
    examples = [row("noul"), row("choice")]
    logits = [np.array([0.0, 2.0]), np.array([-1.0, 2.0, 0.0])]
    report = metrics(examples, logits, 1.0)
    assert report["top1_accuracy"] == 1.0
    assert set(report["by_kind"]) == {"noul", "choice"}
    assert report["selective_prediction"]["selection_score"] == "maximum_candidate_probability"
    assert report["selective_prediction"]["risk_at_coverage"]["0.5"]["selective_risk"] == 0.0
    assert probabilities(logits[1], 0.5)[1] > probabilities(logits[1], 2.0)[1]
    with pytest.raises(ValueError, match="option counts"):
        metrics(examples, [logits[0], logits[0]], 1.0)


def test_selective_risk_is_order_invariant_for_tied_scores() -> None:
    ordered = selective_risk_metrics([0.9, 0.5, 0.5, 0.1], [True, True, False, False])
    reordered = selective_risk_metrics([0.5, 0.9, 0.1, 0.5], [False, True, False, True])
    assert ordered["aurc"] == pytest.approx(reordered["aurc"])
    assert ordered["risk_at_coverage"] == reordered["risk_at_coverage"]
    assert ordered["aurc"] == pytest.approx(13 / 48)
    assert ordered["risk_at_coverage"]["0.5"]["selective_risk"] == pytest.approx(0.25)
    assert ordered["risk_at_coverage"]["0.8"]["selective_risk"] == pytest.approx(0.5)


def test_selective_risk_rejects_invalid_scores_or_labels() -> None:
    with pytest.raises(ValueError, match="invalid"):
        selective_risk_metrics([], [])
    with pytest.raises(ValueError, match="invalid"):
        selective_risk_metrics([0.4, 1.2], [True, False])
    with pytest.raises(ValueError, match="invalid"):
        selective_risk_metrics([0.4, 0.5], [True, 2])


def test_correctness_features_support_variable_option_counts() -> None:
    uniform = correctness_features(np.full(3, 1 / 3))
    certain = correctness_features(np.asarray([1.0, 0.0]))
    assert uniform == pytest.approx([1 / 3, 0.0, 1.0, np.log(3)])
    assert certain == pytest.approx([1.0, 1.0, 0.0, np.log(2)])
    with pytest.raises(ValueError, match="normalized probability"):
        correctness_features(np.asarray([0.2, 0.2]))


def test_correctness_scorer_learns_a_task_agnostic_selective_score() -> None:
    confidence = np.linspace(0.05, 0.95, 20)
    features = np.column_stack((confidence, confidence / 2, 1 - confidence,
                                np.full_like(confidence, np.log(3))))
    labels = [False] * 10 + [True] * 10
    scorer = fit_correctness_scorer(features, labels)
    predicted = predict_correctness(features, scorer)
    assert predicted.shape == (20,)
    assert np.all(np.diff(predicted) > 0)
    assert predicted[0] < 0.2
    assert predicted[-1] > 0.8
    with pytest.raises(ValueError, match="both labels"):
        fit_correctness_scorer(features[:5], [True] * 5)


def test_correctness_metrics_score_calibration_and_tie_aware_auroc() -> None:
    report = correctness_metrics([0.1, 0.9, 0.4, 0.6], [False, True, False, True])
    assert report["auroc"] == pytest.approx(1.0)
    assert report["brier"] == pytest.approx(0.085)
    assert report["selective_prediction"]["selection_score"] == (
        "estimated_probability_of_top_choice_correct")
    tied = correctness_metrics([0.5, 0.5, 0.1, 0.9], [True, False, False, True])
    assert tied["auroc"] == pytest.approx(0.875)


def test_paired_aurc_bootstrap_is_reproducible_and_paired() -> None:
    raw = [0.1, 0.4, 0.6, 0.9]
    learned = [0.1, 0.6, 0.4, 0.9]
    labels = [False, True, False, True]
    first = paired_aurc_bootstrap_delta(raw, learned, labels, replicates=200, seed=7)
    second = paired_aurc_bootstrap_delta(raw, learned, labels, replicates=200, seed=7)
    assert first == second
    assert first["confidence_level"] == 0.95
    assert len(first["delta_aurc_95_percentile_interval"]) == 2
    with pytest.raises(ValueError, match="inputs are invalid"):
        paired_aurc_bootstrap_delta(raw, learned, labels, replicates=2)


def test_paired_mean_bootstrap_is_reproducible() -> None:
    result = paired_mean_bootstrap_delta([0.8, 0.9, 1.0], [0.5, 0.8, 0.9],
                                         replicates=200, seed=12)
    assert result == paired_mean_bootstrap_delta([0.8, 0.9, 1.0], [0.5, 0.8, 0.9],
                                                  replicates=200, seed=12)
    assert result["mean_delta_95_percentile_interval"][0] > 0
    with pytest.raises(ValueError, match="inputs are invalid"):
        paired_mean_bootstrap_delta([1.0], [], replicates=200)


def test_grouped_mean_bootstrap_is_reproducible_and_clustered() -> None:
    args = (["a", "a", "b", "c"], [1, 0, 1, 0], [0, 0, 0, 0])
    result = grouped_mean_bootstrap_delta(*args, replicates=300, seed=7)
    assert result == grouped_mean_bootstrap_delta(*args, replicates=300, seed=7)
    assert result["independent_groups"] == 3
    assert result["mean_delta"] == 0.5
    assert result["resampling_unit"] == "group_id"


def test_grouped_mean_bootstrap_rejects_invalid_clusters() -> None:
    with pytest.raises(ValueError, match="at least two"):
        grouped_mean_bootstrap_delta(["same", "same"], [1, 0], [0, 0], replicates=100)
    with pytest.raises(ValueError, match="inputs are invalid"):
        grouped_mean_bootstrap_delta(["a"], [1, 0], [0, 0], replicates=100)


def test_ordinal_metrics_report_expected_rank_error() -> None:
    example = row("score")
    example["target"] = [0.0, 0.0, 1.0]
    report = metrics([example], [np.array([0.0, 0.0, 0.0])], 1.0)
    assert report["by_kind"]["score"]["mean_rank_error"] == pytest.approx(1.0)
    assert report["by_kind"]["score"]["mean_ranked_probability_score"] == pytest.approx(5 / 18)


def test_uniform_baseline_uses_no_labels_to_form_predictions() -> None:
    example = row("noul")
    assert uniform_baseline([example])["by_kind"]["noul"]["top1_accuracy"] == 0.0
    assert uniform_baseline([example])["by_kind"]["noul"]["balanced_accuracy"] is None
    example["target"] = [1.0, 0.0]
    assert uniform_baseline([example])["by_kind"]["noul"]["top1_accuracy"] == 1.0


def test_binary_metrics_expose_both_error_directions() -> None:
    negative = row("noul")
    negative["target"] = [1.0, 0.0]
    positive = row("noul")
    report = metrics([negative, positive], [np.array([0.0, 2.0]), np.array([1.0, 0.0])], 1.0)
    assert report["by_kind"]["noul"]["confusion"] == {
        "true_positive": 0, "true_negative": 0, "false_positive": 1, "false_negative": 1}
    assert report["by_kind"]["noul"]["balanced_accuracy"] == 0.0


def test_typed_answers_have_normalized_variable_option_probabilities() -> None:
    yes = format_answer("noul", ["no", "yes"], np.array([0.0, 1.0]))
    assert yes["yes_probability"] > 0.5
    choice = format_answer("choice", ["left", "right", "wait"],
                           np.array([0.0, 2.0, -1.0]))
    assert choice["choice"] == "right"
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)
    score = format_answer("score", ["low", "medium", "high"],
                          np.array([-2.0, 0.0, 2.0]))
    assert 1.0 < score["expected_index"] < 2.0
    with pytest.raises(ValueError, match="Score count"):
        format_answer("choice", ["left", "right"], np.array([1.0]))
    with pytest.raises(ValueError, match="option"):
        format_answer("choice", [["left"], "right"], np.array([0.0, 1.0]))


def test_invalid_option_types_fail_before_duplicate_check() -> None:
    example = row()
    example["options"] = [["left"], "right"]
    with pytest.raises(ValueError, match="options"):
        validate_row(example)
    validate_decision("choice", "Which route?", ["walk", "train", "drive", "cycle"])
    with pytest.raises(ValueError, match="Noul"):
        validate_decision("noul", "Can I?", ["yes", "no"])


def test_checkpoint_signature_binds_exact_tokenized_inputs() -> None:
    args = argparse.Namespace(seed=1, max_length=128, lora_rank=8, attention="sdpa",
                              accumulation=4, lr=2e-5, head_lr=5e-5,
                              brier_weight=0.1, ordinal_weight=0.0,
                              prompt_style="candidate_v1", source_holdout_fraction=0.2,
                              source_holdout_seed=3, general_group_weighting="none")
    encoded = {"train": [{"row": row(), "sequences": [[1, 2, 3]]}]}
    first = checkpoint_signature(args, encoded, {}, None)
    assert first == checkpoint_signature(args, encoded, {}, None)
    args.split_seed = 0
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.split_seed = 1
    assert first == checkpoint_signature(args, encoded, {}, None)
    args.source_holdout_seed = 4
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.source_holdout_seed = 3
    encoded["train"][0]["sequences"] = [[1, 2, 4]]
    assert first != checkpoint_signature(args, encoded, {}, None)
    encoded["train"][0]["sequences"] = [[1, 2, 3]]
    args.ordinal_weight = 0.25
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.ordinal_weight = 0.0
    args.prompt_style = "full_options_v2"
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.prompt_style = "candidate_v1"
    args.dtype = "float16"
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.dtype = "bfloat16"
    assert first == checkpoint_signature(args, encoded, {}, None)
    args.initial_openjev = True
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.initial_openjev = False
    assert first == checkpoint_signature(args, encoded, {}, None)
    args.scope_group_weighting = "inverse_sqrt"
    assert first != checkpoint_signature(args, encoded, {}, None)
    assert checkpoint_signature(args, encoded, {}, None, [1.0, 0.5]) != checkpoint_signature(
        args, encoded, {}, None, [1.0, 0.6])
    weighted = checkpoint_signature(args, encoded, {}, None, [1.0, 0.25])
    args.scope_row_weight = 0.25
    assert weighted != checkpoint_signature(args, encoded, {}, None, [1.0, 0.25])
    args.general_group_weighting = "inverse_sqrt"
    assert first != checkpoint_signature(args, encoded, {}, None)
    args.general_group_weighting = "inverse"
    assert checkpoint_signature(args, encoded, {}, None) != checkpoint_signature(
        argparse.Namespace(**{**vars(args), "general_group_weighting": "inverse_sqrt"}),
        encoded, {}, None)


def test_checkpoint_signature_binds_ordinal_loss_version(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(seed=1, max_length=128, lora_rank=8, attention="sdpa",
                              accumulation=4, lr=2e-5, head_lr=5e-5,
                              brier_weight=0.1, ordinal_weight=0.5,
                              prompt_style="candidate_v1")
    encoded = {"train": [{"row": row("score"), "sequences": [[1, 2, 3]]}]}
    current = checkpoint_signature(args, encoded, {}, None)
    monkeypatch.setattr(trainer, "ORDINAL_LOSS_VERSION", "different_ordinal_objective")
    assert current != checkpoint_signature(args, encoded, {}, None)


def test_scope_weights_reduce_group_dominance_without_changing_class_mass() -> None:
    rows = ([{"group_id": "large", "target": [1.0, 0.0]}] * 9 +
            [{"group_id": "small", "target": [1.0, 0.0]}] +
            [{"group_id": "allowed", "target": [0.0, 1.0]}] * 4)
    weights = scope_group_weights(rows)
    assert sum(weights[:10]) == pytest.approx(10)
    assert sum(weights[10:]) == pytest.approx(4)
    assert sum(weights[:9]) / weights[9] == pytest.approx(3)
    with pytest.raises(ValueError, match="both"):
        scope_group_weights(rows[:10])


def test_scope_loss_weight_only_scales_action_rows() -> None:
    general = [{"row": row()}] * 2
    scope = [{"row": {"group_id": "a", "target": [1.0, 0.0]}},
             {"row": {"group_id": "b", "target": [0.0, 1.0]}}]
    assert training_row_weights(general, scope, group_weighting="none", scope_row_weight=1.0) is None
    assert training_row_weights(general, scope, group_weighting="none", scope_row_weight=0.25) == [
        1.0, 1.0, 0.25, 0.25]
    assert training_row_weights(general, [], group_weighting="none", scope_row_weight=0.25) is None


def test_general_group_weighting_preserves_source_kind_loss_mass() -> None:
    def item(group_id: str, source: str = "source", kind: str = "choice") -> dict:
        return {"row": {"group_id": group_id, "source": source, "kind": kind}}

    rows = [item("episode-a") for _ in range(4)] + [item("episode-b")]
    rows += [item("episode-a", source="other-source")]
    rows += [item("episode-c", kind="noul") for _ in range(2)]
    weights = general_group_weights(rows, "inverse_sqrt")

    assert sum(weights[:5]) == pytest.approx(5.0)
    assert sum(weights[5:6]) == pytest.approx(1.0)
    assert sum(weights[6:]) == pytest.approx(2.0)
    assert weights[0] == pytest.approx(weights[3])
    assert weights[0] < weights[4]


def test_general_inverse_group_weighting_equalizes_episode_mass() -> None:
    rows = [{"row": {"group_id": "repeated", "source": "source", "kind": "choice"}}
            for _ in range(3)]
    rows.append({"row": {"group_id": "single", "source": "source", "kind": "choice"}})

    weights = general_group_weights(rows, "inverse")

    assert sum(weights[:3]) == pytest.approx(weights[3])
    assert sum(weights) == pytest.approx(len(rows))


def test_general_group_weighting_keeps_missing_ids_row_unique() -> None:
    rows = [{"row": {"source": "source", "kind": "choice"}} for _ in range(3)]
    assert general_group_weights(rows, "inverse") == pytest.approx([1.0, 1.0, 1.0])
    assert general_group_weights(rows, "none") == [1.0, 1.0, 1.0]


def test_training_row_weights_applies_general_episode_ablation() -> None:
    general = [{"row": {"group_id": "repeated", "source": "source", "kind": "choice"}}
               for _ in range(3)]
    general.append({"row": {"group_id": "single", "source": "source", "kind": "choice"}})

    weights = training_row_weights(general, [], group_weighting="none", scope_row_weight=1.0,
                                   general_group_weighting="inverse")

    assert weights is not None
    assert sum(weights[:3]) == pytest.approx(weights[3])
    assert sum(weights) == pytest.approx(len(general))


def test_general_group_weighting_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown general group weighting"):
        general_group_weights([], "surprise")


def test_ranked_probability_score_respects_cumulative_distance_for_any_option_count() -> None:
    target = np.array([0.0, 0.0, 1.0])
    near = np.array([0.0, 0.5, 0.5])
    far = np.array([1.0, 0.0, 0.0])
    assert ranked_probability_score(near, target) < ranked_probability_score(far, target)
    assert ranked_probability_score(target, target) == 0
    assert ranked_probability_score(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 1
    many_options = np.zeros(11)
    many_options[-1] = 1.0
    opposite = np.zeros(11)
    opposite[0] = 1.0
    assert ranked_probability_score(opposite, many_options) == 1
    assert ranked_probability_score(np.full(11, 1 / 11), np.full(11, 1 / 11)) == 0


def test_ranked_probability_score_rejects_mismatched_vectors() -> None:
    with pytest.raises(ValueError, match="matching vectors"):
        ranked_probability_score(np.array([0.5, 0.5]), np.array([1.0]))
    with pytest.raises(ValueError, match="matching vectors"):
        ranked_probability_score(np.array([1.0]), np.array([1.0]))
