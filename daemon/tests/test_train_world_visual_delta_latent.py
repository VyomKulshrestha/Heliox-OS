"""Research latent ranking and shared delta encoder contract."""

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.train_world_visual_delta_latent import attach_semantic_features, pair_ranking


def test_pair_ranking_uses_independent_groups() -> None:
    target = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    correct = target.copy()
    wrong = target[[1, 0, 3, 2]].copy()
    wrong[2:] = correct[2:]
    result = pair_ranking(correct, wrong, target, np.array(["a", "a", "b", "b"]))
    assert result["groups"] == 2
    assert result["correct_action_wins"] == 1
    assert result["ties"] == 1
    assert result["mean_mse_advantage"] > 0
    with pytest.raises(ValueError, match="exactly two"):
        pair_ranking(correct, wrong, target, np.array(["a", "a", "a", "b"]))


def test_delta_inverse_receives_shared_encoder_gradient() -> None:
    code = (
        "import torch; from scripts.train_world_visual_delta_latent import "
        "build_delta_latent_model,paired_matching_loss; "
        "model=build_delta_latent_model(torch.nn,torch,(1,16,16),64); "
        "before=model.encode(torch.rand(2,1,16,16)); "
        "after=model.encode(torch.rand(2,1,16,16)); "
        "pred=model(before,torch.rand(2,11)); "
        "assert pred.shape==(2,64); "
        "coords=model.inverse(before,after); "
        "assert coords.shape==(2,2); "
        "assert model.reconstruction_head(before).shape==(2,256); "
        "pair=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True); "
        "assert paired_matching_loss(pair,pair.detach(),torch,1e-8,1)<"
        "paired_matching_loss(pair.flip(0),pair.detach(),torch,1e-8,1); "
        "coords.sum().backward(); "
        "assert model.encoder[0].weight.grad.abs().sum()>0"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if "No module named 'torch'" in result.stderr:
        pytest.skip("Optional torch research dependency is not installed")
    assert result.returncode == 0, result.stderr


def test_semantic_model_requires_context_and_preserves_normalizer() -> None:
    code = (
        "import torch; from scripts.train_world_visual_delta_latent import build_delta_latent_model; "
        "model=build_delta_latent_model(torch.nn,torch,(1,16,16),64,8); "
        "before=model.encode(torch.rand(2,1,16,16)); "
        "assert model(before,torch.rand(2,11),torch.rand(2,8)).shape==(2,64); "
        "assert 'semantic_mean' in model.state_dict(); "
        "assert 'semantic_scale' in model.state_dict(); "
        "assert model.semantic_projection is not None"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if "No module named 'torch'" in result.stderr:
        pytest.skip("Optional torch research dependency is not installed")
    assert result.returncode == 0, result.stderr


def test_semantic_features_reject_visual_row_misalignment(tmp_path) -> None:
    visual = tmp_path / "visual.npz"
    source = tmp_path / "source.json"
    features = tmp_path / "semantic.npz"
    manifest = tmp_path / "semantic.json"
    visual.write_bytes(b"visual")
    source.write_text("{}", encoding="utf-8")
    np.savez_compressed(features, features=np.ones((2, 8), dtype=np.float16),
                        group=np.array(["wrong", "wrong"]), split=np.array(["train", "train"]),
                        coords=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"feature_sha256": digest(features),
                                    "visual_features_sha256": digest(visual),
                                    "source_report_sha256": digest(source),
                                    "feature_dim": 8, "rows": 2}), encoding="utf-8")
    rows = {"train": {"groups": np.array(["same", "same"]),
                      "action": np.zeros((2, 11), dtype=np.float32)},
            "validation": {"groups": np.array([], dtype=str),
                           "action": np.zeros((0, 11), dtype=np.float32)},
            "eval": {"groups": np.array([], dtype=str),
                     "action": np.zeros((0, 11), dtype=np.float32)}}
    with pytest.raises(ValueError, match="misaligned"):
        attach_semantic_features(rows, visual, source, features, manifest)


def test_transfer_rejects_mismatched_weights(tmp_path) -> None:
    model_report = tmp_path / "model.json"
    model_report.write_text(json.dumps({
        "purpose": "research_only_delta_inverse_visual_latent_transition",
        "weights_sha256": "0" * 64,
    }), encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not the pinned weights")
    result = subprocess.run([
        sys.executable, "-m", "scripts.probe_world_visual_delta_latent_transfer",
        "--model-report", str(model_report), "--weights", str(weights),
        "--target-features", str(tmp_path / "unused.npz"),
        "--target-report", str(tmp_path / "unused.json"),
        "--report", str(tmp_path / "out.json"), "--device", "cpu",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Weights do not match the pinned model report" in result.stderr


def test_transfer_requires_matching_semantic_archive(tmp_path) -> None:
    model_report = tmp_path / "semantic-model.json"
    model_report.write_text(json.dumps({
        "purpose": "research_only_delta_inverse_visual_latent_transition",
        "semantic_dim": 8, "weights_sha256": "0" * 64,
    }), encoding="utf-8")
    result = subprocess.run([
        sys.executable, "-m", "scripts.probe_world_visual_delta_latent_transfer",
        "--model-report", str(model_report), "--weights", str(tmp_path / "unused.pt"),
        "--target-features", str(tmp_path / "unused.npz"),
        "--target-report", str(tmp_path / "unused.json"),
        "--report", str(tmp_path / "out.json"), "--device", "cpu",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Target semantic features are required" in result.stderr
