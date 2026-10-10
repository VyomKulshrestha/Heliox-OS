import numpy as np
import pytest

from benchmarks.tbac_tool_relevance import RelevanceCase, relevance_request
from scripts.extract_opendec_scope_features import (
    case_signature,
    load_checkpoint,
    packed,
    relevance_label,
    save_checkpoint,
)


def cases():
    return (
        RelevanceCase("tbac-a:1", "Read issue", "Tool: read_issue", True, "tbac-a", "validation"),
        RelevanceCase("tbac-b:2", "Read issue", "Tool: delete_issue", False, "tbac-b", "validation"),
    )


def test_resume_accepts_only_matching_prefix_and_provenance(tmp_path):
    selected = cases()
    signature = case_signature(selected, relevance_request, relevance_label)
    arrays = packed([np.asarray([1.0, 2.0])], [True], [selected[0].id], [0.4])
    partial = tmp_path / "validation.partial.npz"
    source = {"adapter_sha256": "pinned"}
    save_checkpoint(
        partial,
        arrays,
        [{"id": selected[1].id, "tokens": 900}],
        completed=2,
        signature=signature,
        provenance=source,
        max_length=768,
    )
    restored, excluded, completed = load_checkpoint(
        partial, selected, relevance_label, signature=signature, provenance=source, max_length=768
    )
    assert completed == 2
    assert list(restored["ids"]) == [selected[0].id]
    assert excluded == [{"id": selected[1].id, "tokens": 900}]
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(
            partial,
            selected,
            relevance_label,
            signature=signature,
            provenance={"adapter_sha256": "changed"},
            max_length=768,
        )


def test_resume_rejects_changed_labels_or_case_requests(tmp_path):
    selected = cases()
    signature = case_signature(selected, relevance_request, relevance_label)
    partial = tmp_path / "validation.partial.npz"
    save_checkpoint(
        partial,
        packed([np.asarray([1.0, 2.0])], [False], [selected[0].id], [0.4]),
        [],
        completed=1,
        signature=signature,
        provenance={},
        max_length=768,
    )
    with pytest.raises(ValueError, match="contents"):
        load_checkpoint(partial, selected, relevance_label, signature=signature, provenance={}, max_length=768)
    changed = (RelevanceCase("tbac-a:1", "Delete issue", "Tool: read_issue", True, "tbac-a", "validation"), selected[1])
    assert case_signature(changed, relevance_request, relevance_label) != signature
