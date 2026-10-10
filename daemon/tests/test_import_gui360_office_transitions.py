"""Pinned Office trajectories must remain ordered, nonterminal, and leak-free."""

import io
import json
from collections import Counter

import pytest
from PIL import Image

from scripts.import_aligned_world_trajectories import _file_sha256
from scripts.import_gui360_office_transitions import extract_transitions, import_shards


def _png(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(output, format="PNG")
    return output.getvalue()


def _trajectory(domain: str, *, multi_action: bool = False) -> dict:
    def call(index: int, coordinate: list[int]) -> dict:
        actions = [{"action": "click", "coordinate": coordinate}]
        if multi_action and index == 0:
            actions.append({"action": "click", "coordinate": [800, 700]})
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "function": {"name": "computer", "arguments": {"actions": actions}},
                }
            ],
        }

    messages = [
        {"role": "user", "content": [{"type": "image", "index": 0}]},
        call(0, [100, 200]),
        {"role": "tool", "tool_call_id": "call_0", "content": [{"type": "image", "index": 1}]},
        call(1, [800, 700]),
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "image", "index": 2}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    return {
        "metadata": json.dumps(
            {
                "dims": ["desktop", "use"],
                "others": {
                    "id": f"{domain}_example_1",
                    "source": "vyokky/GUI-360",
                },
            }
        ),
        "messages": json.dumps(messages),
        "images": [{"bytes": _png(color), "path": None} for color in ((10, 20, 30), (80, 90, 100), (150, 160, 170))],
    }


def test_import_accepts_only_aligned_nonterminal_single_click() -> None:
    reasons = Counter()
    rows = extract_transitions(
        _trajectory("excel"),
        split="train",
        expected_domain="excel",
        representation="rgb32",
        reasons=reasons,
    )
    assert len(rows) == 1
    assert rows[0]["state"].shape == (3 * 32 * 32,)
    assert rows[0]["coords"] == (0.1, 0.2)
    assert rows[0]["action"] == 1
    assert reasons["terminal_transition"] == 1
    assert not (rows[0]["state"] == rows[0]["target"]).all()


def test_import_rejects_ambiguous_actions_and_wrong_domain() -> None:
    reasons = Counter()
    assert (
        extract_transitions(
            _trajectory("excel", multi_action=True),
            split="train",
            expected_domain="excel",
            representation="gray16",
            reasons=reasons,
        )
        == []
    )
    assert reasons["unsupported_action"] == 1
    with pytest.raises(ValueError, match="domain"):
        extract_transitions(
            _trajectory("word"),
            split="eval",
            expected_domain="excel",
            representation="rgb32",
            reasons=Counter(),
        )


def test_import_rejects_identical_pre_screen_across_domains(tmp_path) -> None:
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    protocol = {"dataset": "cua-lite/GUI-360", "representation": "rgb32", "minimum_rows_per_split": 1, "shards": []}
    for split, domain in (("train", "excel"), ("validation", "word"), ("eval", "ppt")):
        path = tmp_path / f"{domain}.parquet"
        parquet.write_table(arrow.Table.from_pylist([_trajectory(domain)]), path)
        protocol["shards"].append(
            {
                "path": path.name,
                "sha256": _file_sha256(path),
                "expected_domain": domain,
                "split": split,
            }
        )
    with pytest.raises(ValueError, match="pre-action screen leaked"):
        import_shards(tmp_path, protocol)
    protocol["shards"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed pinned shard"):
        import_shards(tmp_path, protocol)
