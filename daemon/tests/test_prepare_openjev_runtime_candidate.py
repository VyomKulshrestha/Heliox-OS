from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_openjev_runtime_candidate import head_arrays, scorer_provenance


def test_prepare_refuses_to_overwrite_existing_output(tmp_path):
    from scripts.prepare_openjev_runtime_candidate import prepare

    output = tmp_path / "candidate"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare(output)


def test_head_arrays_accepts_finite_scalar_linear_state():
    weight, bias = head_arrays({"weight": [[1.0, -2.0]], "bias": [0.25]})

    np.testing.assert_array_equal(weight, np.asarray([[1.0, -2.0]], dtype=np.float32))
    np.testing.assert_array_equal(bias, np.asarray([0.25], dtype=np.float32))


@pytest.mark.parametrize(
    "state",
    [
        {"weight": [[1.0]], "bias": [0.0], "unexpected": 1},
        {"weight": [1.0], "bias": [0.0]},
        {"weight": [[float("nan")]], "bias": [0.0]},
        {"weight": [[1.0]], "bias": [float("inf")]},
    ],
)
def test_head_arrays_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        head_arrays(state)


def test_prepared_report_contains_provenance_accepted_by_local_scorer(tmp_path):
    import hashlib
    import json

    from scripts.evaluate_qwen_instruct_scope import MODEL_ID, MODEL_REVISION
    from scripts.prepare_openjev_runtime_candidate import BASE_MODEL, BASE_REVISION
    from scripts.score_general_decision import verify_artifact
    from scripts.train_world_scope_qlora import (
        OPENJEV_ADAPTER_SHA256,
        OPENJEV_HEAD_SHA256,
        OPENJEV_REPO,
        OPENJEV_REVISION,
    )

    artifact = tmp_path / "artifact"
    adapter = artifact / "adapter" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"verified adapter fixture")
    head = artifact / "head.npz"
    head.write_bytes(b"verified head fixture")
    report = {
        "schema_version": 1,
        "model": BASE_MODEL,
        "revision": BASE_REVISION,
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
        "head_sha256": hashlib.sha256(head.read_bytes()).hexdigest(),
        **scorer_provenance(OPENJEV_REPO, OPENJEV_REVISION,
                            OPENJEV_ADAPTER_SHA256, OPENJEV_HEAD_SHA256),
    }
    assert (report["model"], report["revision"]) == (MODEL_ID, MODEL_REVISION)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    verified, _ = verify_artifact(artifact, report_path)
    assert verified["initial_checkpoint"]["repo"] == OPENJEV_REPO


def test_scorer_provenance_rejects_incomplete_source():
    with pytest.raises(ValueError, match="provenance must be complete"):
        scorer_provenance("Open-Jev", "", "adapter", "head")
