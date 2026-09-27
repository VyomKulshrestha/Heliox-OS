"""Research latent ranking and shared delta encoder contract."""

import json
import subprocess
import sys

import numpy as np
import pytest

from scripts.train_world_visual_delta_latent import pair_ranking


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
