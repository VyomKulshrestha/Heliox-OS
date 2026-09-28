from __future__ import annotations

import numpy as np
import pytest

from scripts.prepare_openjev_runtime_candidate import head_arrays


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
