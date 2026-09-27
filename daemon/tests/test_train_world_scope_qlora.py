"""Pure-data guards for the offline QLoRA world task-state experiment."""

import numpy as np
import pytest

from benchmarks.jev_scope_suite import ScopeCase
from scripts import train_world_scope_qlora as probe
from scripts.train_world_scope_qlora import calibration_choice, domain_class_weights


def test_domain_class_weights_balance_both_corpora():
    rows = {"tbac_train": [("a", [], True), ("b", [], False)],
            "scope_train": [("c", [], True), ("d", [], True), ("e", [], False)]}
    weights = domain_class_weights(rows)
    assert sum(weights[domain, row[2]] for domain in ("tbac", "scope")
               for row in rows[f"{domain}_train"]) == pytest.approx(5)
    assert weights["scope", False] == pytest.approx(1.25)


def test_calibration_choice_uses_both_negative_sets():
    calibration = {"tbac": (np.array([False, True]), np.array([0.1, 0.9])),
                   "scope": (np.array([False, True]), np.array([0.7, 0.8]))}
    threshold, auc = calibration_choice(calibration)
    assert threshold > 0.7
    assert auc == {"tbac": 1.0, "scope": 1.0}


def test_openjev_warm_start_uses_candidate_prompt_without_labels():
    case = ScopeCase("example", "Read the status.", "Delete the status file.", False)
    prompt = probe.openjev_scope_prompt(case)
    assert '"user_request": "Read the status."' in prompt
    assert '"proposed_action": "Delete the status file."' in prompt
    assert "Proposed answer" not in prompt
    assert "Yes means:" in prompt and "No means:" in prompt
    assert "gold" not in prompt and "example" not in prompt


def test_openjev_warm_start_rejects_wrong_adapter_hash(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"wrong")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_args, **_kwargs: str(adapter))
    with pytest.raises(ValueError, match="checksum mismatch"):
        probe.openjev_checkpoint_paths()
