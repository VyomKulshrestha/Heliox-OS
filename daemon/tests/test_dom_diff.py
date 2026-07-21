"""Tests for the DOM-diff engine (pilot.system.dom_diff).

Covers:
- DomNode.from_dict construction
- DomSnapshot.fingerprint hashing (determinism + sensitivity)
- DomSnapshot.key_set
- diff_dom() Jaccard distance calculation
- diff_dom() added / removed / mutated node detection
- diff_dom() URL and title change weighting
- diff_dom() edge cases (empty snapshots, identical snapshots)
- assert_dom_changed() threshold enforcement
- DomUnchangedError attributes
- DomDiff.summary() and DomDiff.to_dict() output shape
"""

from __future__ import annotations

import pytest

from pilot.system.dom_diff import (
    DomDiff,
    DomNode,
    DomSnapshot,
    DomUnchangedError,
    TargetAssessment,
    assert_dom_changed,
    assess_target,
    diff_dom,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    tag: str = "div",
    id: str = "",
    cls: str = "",
    text: str = "",
    x: int = 0,
    y: int = 0,
    w: int = 100,
    h: int = 50,
    visible: bool = True,
    depth: int = 0,
) -> DomNode:
    """Build a DomNode with a computed key matching the JS nodeHash formula."""
    key = f"{tag}|{id}|{cls}|{text[:40]}"
    return DomNode(
        tag=tag,
        id=id,
        cls=cls,
        text=text,
        depth=depth,
        visible=visible,
        x=x,
        y=y,
        w=w,
        h=h,
        key=key,
    )


def _snap(nodes: list[DomNode], url: str = "https://example.com", title: str = "Test") -> DomSnapshot:
    return DomSnapshot(nodes=nodes, url=url, title=title)


# ---------------------------------------------------------------------------
# DomNode.from_dict
# ---------------------------------------------------------------------------


class TestDomNodeFromDict:
    def test_all_fields_populated(self):
        d = {
            "tag": "button",
            "id": "submit",
            "cls": "btn primary",
            "text": "Click me",
            "depth": 3,
            "visible": True,
            "x": 10,
            "y": 20,
            "w": 80,
            "h": 40,
            "key": "button|submit|btn primary|Click me",
        }
        node = DomNode.from_dict(d)
        assert node.tag == "button"
        assert node.id == "submit"
        assert node.cls == "btn primary"
        assert node.text == "Click me"
        assert node.depth == 3
        assert node.visible is True
        assert node.x == 10
        assert node.y == 20
        assert node.w == 80
        assert node.h == 40
        assert node.key == "button|submit|btn primary|Click me"

    def test_missing_fields_use_defaults(self):
        node = DomNode.from_dict({})
        assert node.tag == ""
        assert node.id == ""
        assert node.visible is False
        assert node.x == 0
        assert node.w == 0


# ---------------------------------------------------------------------------
# DomSnapshot hashing
# ---------------------------------------------------------------------------


class TestDomSnapshotFingerprint:
    def test_empty_snapshot_has_stable_hash(self):
        s = _snap([])
        h1 = s.fingerprint
        h2 = s.fingerprint
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_same_nodes_produce_same_hash(self):
        nodes = [_node("div", id="a"), _node("span", id="b")]
        s1 = _snap(nodes)
        s2 = _snap(nodes)
        assert s1.fingerprint == s2.fingerprint

    def test_different_nodes_produce_different_hash(self):
        s1 = _snap([_node("div", id="a")])
        s2 = _snap([_node("div", id="b")])
        assert s1.fingerprint != s2.fingerprint

    def test_node_order_affects_hash(self):
        """Fingerprint is order-sensitive — DOM order matters."""
        n1 = _node("div", id="a")
        n2 = _node("span", id="b")
        s1 = _snap([n1, n2])
        s2 = _snap([n2, n1])
        assert s1.fingerprint != s2.fingerprint

    def test_text_change_affects_hash(self):
        s1 = _snap([_node("p", text="Hello")])
        s2 = _snap([_node("p", text="World")])
        assert s1.fingerprint != s2.fingerprint


# ---------------------------------------------------------------------------
# DomSnapshot.key_set
# ---------------------------------------------------------------------------


class TestDomSnapshotKeySet:
    def test_returns_set_of_keys(self):
        nodes = [_node("div", id="a"), _node("span", id="b")]
        s = _snap(nodes)
        ks = s.key_set()
        assert isinstance(ks, set)
        assert len(ks) == 2

    def test_duplicate_keys_deduplicated(self):
        # Two nodes with identical keys (same tag/id/cls/text)
        n = _node("div", id="x")
        s = _snap([n, n])
        assert len(s.key_set()) == 1

    def test_node_count_vs_key_set(self):
        nodes = [_node("div", id=str(i)) for i in range(10)]
        s = _snap(nodes)
        assert s.node_count == 10
        assert len(s.key_set()) == 10


# ---------------------------------------------------------------------------
# diff_dom — identical snapshots
# ---------------------------------------------------------------------------


class TestDiffDomIdentical:
    def test_identical_snapshots_score_zero(self):
        nodes = [_node("div", id="a"), _node("p", text="hello")]
        before = _snap(nodes)
        after = _snap(nodes)
        diff = diff_dom(before, after)
        assert diff.change_score == 0.0
        assert diff.changed is False
        assert diff.added == []
        assert diff.removed == []

    def test_empty_snapshots_score_zero(self):
        before = _snap([])
        after = _snap([])
        diff = diff_dom(before, after)
        assert diff.change_score == 0.0

    def test_before_after_counts_recorded(self):
        before = _snap([_node("div")])
        after = _snap([_node("div"), _node("span")])
        diff = diff_dom(before, after)
        assert diff.before_count == 1
        assert diff.after_count == 2


# ---------------------------------------------------------------------------
# diff_dom — Jaccard distance (structural delta)
# ---------------------------------------------------------------------------


class TestDiffDomJaccard:
    def test_completely_different_nodes_max_structural_score(self):
        """No overlap → Jaccard distance = 1.0 → change_score = 0.70 (W_STRUCT)."""
        before = _snap([_node("div", id="a"), _node("span", id="b")])
        after = _snap([_node("p", id="c"), _node("h1", id="d")])
        diff = diff_dom(before, after)
        # structural_delta = 1.0, no url/title change
        assert abs(diff.change_score - 0.70) < 1e-9

    def test_half_overlap_correct_jaccard(self):
        """2 shared, 2 unique each → union=6, intersection=2 → Jaccard=1-2/6=0.667."""
        shared1 = _node("div", id="s1")
        shared2 = _node("span", id="s2")
        before = _snap([shared1, shared2, _node("p", id="b1"), _node("h1", id="b2")])
        after = _snap([shared1, shared2, _node("li", id="a1"), _node("ul", id="a2")])
        diff = diff_dom(before, after)
        expected_structural = 1.0 - (2 / 6)  # ≈ 0.6667
        expected_score = 0.70 * expected_structural
        assert abs(diff.change_score - expected_score) < 1e-6

    def test_one_node_added_to_large_dom(self):
        """Adding 1 node to 99 → union=100, intersection=99 → Jaccard=0.01."""
        nodes = [_node("div", id=str(i)) for i in range(99)]
        before = _snap(nodes)
        after = _snap(nodes + [_node("span", id="new")])
        diff = diff_dom(before, after)
        expected_structural = 1 / 100
        expected_score = 0.70 * expected_structural
        assert abs(diff.change_score - expected_score) < 1e-6
        assert len(diff.added) == 1
        assert diff.added[0].id == "new"

    def test_one_node_removed(self):
        nodes = [_node("div", id=str(i)) for i in range(5)]
        before = _snap(nodes)
        after = _snap(nodes[:-1])
        diff = diff_dom(before, after)
        assert len(diff.removed) == 1
        assert diff.removed[0].id == "4"
        assert diff.change_score > 0.0


# ---------------------------------------------------------------------------
# diff_dom — added / removed node detection
# ---------------------------------------------------------------------------


class TestDiffDomAddedRemoved:
    def test_added_nodes_identified(self):
        before = _snap([_node("div", id="a")])
        after = _snap([_node("div", id="a"), _node("span", id="b"), _node("p", id="c")])
        diff = diff_dom(before, after)
        added_ids = {n.id for n in diff.added}
        assert added_ids == {"b", "c"}

    def test_removed_nodes_identified(self):
        before = _snap([_node("div", id="a"), _node("span", id="b")])
        after = _snap([_node("div", id="a")])
        diff = diff_dom(before, after)
        assert len(diff.removed) == 1
        assert diff.removed[0].id == "b"

    def test_swap_nodes(self):
        before = _snap([_node("div", id="old")])
        after = _snap([_node("div", id="new")])
        diff = diff_dom(before, after)
        assert len(diff.added) == 1
        assert len(diff.removed) == 1
        assert diff.added[0].id == "new"
        assert diff.removed[0].id == "old"


# ---------------------------------------------------------------------------
# diff_dom — mutated node detection (bounding box shift)
# ---------------------------------------------------------------------------


class TestDiffDomMutated:
    def test_node_moved_more_than_5px_detected(self):
        key = "div||cls|text"
        before_node = DomNode("div", "", "cls", "text", 0, True, 10, 10, 100, 50, key)
        after_node = DomNode("div", "", "cls", "text", 0, True, 20, 10, 100, 50, key)  # x shifted 10px
        before = _snap([before_node])
        after = _snap([after_node])
        diff = diff_dom(before, after)
        assert len(diff.mutated) == 1

    def test_node_moved_less_than_5px_not_detected(self):
        key = "div||cls|text"
        before_node = DomNode("div", "", "cls", "text", 0, True, 10, 10, 100, 50, key)
        after_node = DomNode("div", "", "cls", "text", 0, True, 13, 10, 100, 50, key)  # x shifted 3px
        before = _snap([before_node])
        after = _snap([after_node])
        diff = diff_dom(before, after)
        assert len(diff.mutated) == 0

    def test_node_resized_detected(self):
        key = "div||cls|text"
        before_node = DomNode("div", "", "cls", "text", 0, True, 0, 0, 100, 50, key)
        after_node = DomNode("div", "", "cls", "text", 0, True, 0, 0, 200, 50, key)  # width doubled
        before = _snap([before_node])
        after = _snap([after_node])
        diff = diff_dom(before, after)
        assert len(diff.mutated) == 1


# ---------------------------------------------------------------------------
# diff_dom — URL and title change weighting
# ---------------------------------------------------------------------------


class TestDiffDomUrlTitle:
    def test_url_change_adds_020_to_score(self):
        nodes = [_node("div", id="a")]
        before = _snap(nodes, url="https://example.com/page1")
        after = _snap(nodes, url="https://example.com/page2")
        diff = diff_dom(before, after)
        assert diff.url_changed is True
        # structural_delta=0, url_weight=0.20
        assert abs(diff.change_score - 0.20) < 1e-9

    def test_title_change_adds_010_to_score(self):
        nodes = [_node("div", id="a")]
        before = _snap(nodes, title="Page A")
        after = _snap(nodes, title="Page B")
        diff = diff_dom(before, after)
        assert diff.title_changed is True
        assert abs(diff.change_score - 0.10) < 1e-9

    def test_url_and_title_change_combined(self):
        nodes = [_node("div", id="a")]
        before = _snap(nodes, url="https://a.com", title="A")
        after = _snap(nodes, url="https://b.com", title="B")
        diff = diff_dom(before, after)
        # structural=0, url=0.20, title=0.10 → 0.30
        assert abs(diff.change_score - 0.30) < 1e-9

    def test_score_clamped_to_1(self):
        """All three components at max should not exceed 1.0."""
        before = _snap([_node("div", id="a")], url="https://a.com", title="A")
        after = _snap([_node("span", id="b")], url="https://b.com", title="B")
        diff = diff_dom(before, after)
        assert diff.change_score <= 1.0

    def test_same_url_and_title_no_bonus(self):
        nodes = [_node("div", id="a")]
        before = _snap(nodes, url="https://same.com", title="Same")
        after = _snap(nodes, url="https://same.com", title="Same")
        diff = diff_dom(before, after)
        assert diff.url_changed is False
        assert diff.title_changed is False
        assert diff.change_score == 0.0


# ---------------------------------------------------------------------------
# assert_dom_changed — threshold enforcement
# ---------------------------------------------------------------------------


class TestAssertDomChanged:
    def test_passes_when_score_above_threshold(self):
        before = _snap([_node("div", id="a")])
        after = _snap([_node("span", id="b")])  # completely different → score=0.70
        diff = assert_dom_changed(before, after, min_score=0.01)
        assert isinstance(diff, DomDiff)
        assert diff.change_score >= 0.01

    def test_raises_when_score_below_threshold(self):
        nodes = [_node("div", id="a")]
        before = _snap(nodes)
        after = _snap(nodes)  # identical → score=0.0
        with pytest.raises(DomUnchangedError):
            assert_dom_changed(before, after, min_score=0.01)

    def test_raises_exactly_at_threshold_boundary(self):
        """score=0.0 should raise for any min_score > 0."""
        nodes = [_node("p", text="same")]
        before = _snap(nodes)
        after = _snap(nodes)
        with pytest.raises(DomUnchangedError):
            assert_dom_changed(before, after, min_score=0.001)

    def test_passes_at_exact_threshold(self):
        """score >= min_score should not raise."""
        before = _snap([_node("div", id="a")], url="https://a.com")
        after = _snap([_node("div", id="a")], url="https://b.com")
        # url change → score=0.20, threshold=0.20 → should pass
        diff = assert_dom_changed(before, after, min_score=0.20)
        assert diff.change_score >= 0.20

    def test_custom_action_desc_in_error(self):
        nodes = [_node("div")]
        before = _snap(nodes)
        after = _snap(nodes)
        with pytest.raises(DomUnchangedError) as exc_info:
            assert_dom_changed(before, after, min_score=0.01, action_desc="click:#btn")
        assert "click:#btn" in str(exc_info.value)

    def test_default_threshold_is_001(self):
        """Default min_score=0.01 — identical DOM should raise."""
        nodes = [_node("div", id="x")]
        with pytest.raises(DomUnchangedError):
            assert_dom_changed(_snap(nodes), _snap(nodes))


# ---------------------------------------------------------------------------
# DomUnchangedError attributes
# ---------------------------------------------------------------------------


class TestDomUnchangedError:
    def test_carries_diff_object(self):
        nodes = [_node("div")]
        before = _snap(nodes)
        after = _snap(nodes)
        try:
            assert_dom_changed(before, after, min_score=0.01, action_desc="type:#input")
        except DomUnchangedError as e:
            assert isinstance(e.diff, DomDiff)
            assert e.diff.change_score == 0.0
            assert e.action_desc == "type:#input"

    def test_error_message_contains_score(self):
        nodes = [_node("div")]
        before = _snap(nodes)
        after = _snap(nodes)
        with pytest.raises(DomUnchangedError) as exc_info:
            assert_dom_changed(before, after, min_score=0.01)
        assert "0.000" in str(exc_info.value)


# ---------------------------------------------------------------------------
# DomDiff.summary()
# ---------------------------------------------------------------------------


class TestDomDiffSummary:
    def test_no_change_summary(self):
        nodes = [_node("div")]
        diff = diff_dom(_snap(nodes), _snap(nodes))
        assert diff.summary() == "no change detected"

    def test_summary_includes_added_count(self):
        before = _snap([_node("div", id="a")])
        after = _snap([_node("div", id="a"), _node("span", id="b")])
        diff = diff_dom(before, after)
        assert "+1 nodes added" in diff.summary()

    def test_summary_includes_removed_count(self):
        before = _snap([_node("div", id="a"), _node("span", id="b")])
        after = _snap([_node("div", id="a")])
        diff = diff_dom(before, after)
        assert "-1 nodes removed" in diff.summary()

    def test_summary_includes_url_changed(self):
        nodes = [_node("div")]
        diff = diff_dom(
            _snap(nodes, url="https://a.com"),
            _snap(nodes, url="https://b.com"),
        )
        assert "URL changed" in diff.summary()

    def test_summary_includes_change_score(self):
        before = _snap([_node("div", id="a")])
        after = _snap([_node("span", id="b")])
        diff = diff_dom(before, after)
        assert "change_score=" in diff.summary()


# ---------------------------------------------------------------------------
# DomDiff.to_dict()
# ---------------------------------------------------------------------------


class TestDomDiffToDict:
    def test_to_dict_has_required_keys(self):
        nodes = [_node("div", id="a")]
        diff = diff_dom(_snap(nodes), _snap(nodes))
        d = diff.to_dict()
        required = {
            "change_score",
            "changed",
            "url_changed",
            "title_changed",
            "added_count",
            "removed_count",
            "mutated_count",
            "before_node_count",
            "after_node_count",
            "summary",
            "added_sample",
            "removed_sample",
        }
        assert required.issubset(d.keys())

    def test_change_score_rounded_to_4dp(self):
        before = _snap([_node("div", id="a")])
        after = _snap([_node("span", id="b")])
        d = diff_dom(before, after).to_dict()
        # Should be rounded to 4 decimal places
        assert d["change_score"] == round(d["change_score"], 4)

    def test_added_sample_capped_at_5(self):
        before = _snap([])
        after = _snap([_node("div", id=str(i)) for i in range(10)])
        d = diff_dom(before, after).to_dict()
        assert len(d["added_sample"]) <= 5

    def test_counts_match_lists(self):
        before = _snap([_node("div", id="a"), _node("span", id="b")])
        after = _snap([_node("div", id="a"), _node("p", id="c")])
        diff = diff_dom(before, after)
        d = diff.to_dict()
        assert d["added_count"] == len(diff.added)
        assert d["removed_count"] == len(diff.removed)


# ---------------------------------------------------------------------------
# assess_target — pre-execution target assessment
# ---------------------------------------------------------------------------


class TestAssessTargetNoInput:
    def test_no_selector_or_text_is_unmatchable(self):
        result = assess_target(_snap([]))
        assert isinstance(result, TargetAssessment)
        assert result.matchable is False

    def test_empty_text_is_unmatchable(self):
        result = assess_target(_snap([_node("button")]), text="   ")
        assert result.matchable is False


class TestAssessTargetBySelector:
    def test_id_selector_found_and_visible(self):
        snapshot = _snap([_node("button", id="submit-btn")])
        result = assess_target(snapshot, selector="#submit-btn")
        assert result.matchable is True
        assert result.found is True
        assert result.visible is True
        assert result.ambiguous is False

    def test_id_selector_not_found(self):
        snapshot = _snap([_node("button", id="other-btn")])
        result = assess_target(snapshot, selector="#submit-btn")
        assert result.matchable is True
        assert result.found is False
        assert "not found" in result.reason

    def test_class_selector_matches_class_token(self):
        snapshot = _snap([_node("div", cls="btn primary")])
        result = assess_target(snapshot, selector=".primary")
        assert result.found is True

    def test_class_selector_does_not_match_substring_of_another_class(self):
        # "prim" must not match a node whose only class is "primary".
        snapshot = _snap([_node("div", cls="primary")])
        result = assess_target(snapshot, selector=".prim")
        assert result.found is False

    def test_tag_and_id_combined_selector(self):
        snapshot = _snap([_node("button", id="submit-btn"), _node("a", id="submit-btn")])
        result = assess_target(snapshot, selector="button#submit-btn")
        assert result.found is True
        assert result.match_count == 1

    def test_bare_tag_selector(self):
        snapshot = _snap([_node("button"), _node("div")])
        result = assess_target(snapshot, selector="button")
        assert result.found is True
        assert result.match_count == 1

    def test_not_visible_match_reported_as_found_but_not_visible(self):
        snapshot = _snap([_node("button", id="submit-btn", visible=False)])
        result = assess_target(snapshot, selector="#submit-btn")
        assert result.matchable is True
        assert result.found is True
        assert result.visible is False
        assert "not visible" in result.reason

    def test_multiple_visible_matches_are_ambiguous(self):
        snapshot = _snap([_node("button", cls="btn"), _node("button", cls="btn")])
        result = assess_target(snapshot, selector=".btn")
        assert result.matchable is True
        assert result.found is True
        assert result.ambiguous is True
        assert result.match_count == 2

    def test_single_match_is_not_ambiguous(self):
        snapshot = _snap([_node("button", cls="btn")])
        result = assess_target(snapshot, selector=".btn")
        assert result.ambiguous is False


class TestAssessTargetUnmatchableSelectors:
    """Combinators, attribute selectors, and pseudo-classes can't be
    honestly evaluated from a flat node list — must report matchable=False
    rather than silently guessing."""

    def test_descendant_combinator_is_unmatchable(self):
        result = assess_target(_snap([_node("button")]), selector="form button")
        assert result.matchable is False

    def test_child_combinator_is_unmatchable(self):
        result = assess_target(_snap([_node("button")]), selector="form > button")
        assert result.matchable is False

    def test_attribute_selector_is_unmatchable(self):
        result = assess_target(_snap([_node("input")]), selector="input[type='submit']")
        assert result.matchable is False

    def test_pseudo_class_selector_is_unmatchable(self):
        result = assess_target(_snap([_node("button")]), selector="button:has-text('Login')")
        assert result.matchable is False

    def test_empty_selector_is_unmatchable(self):
        result = assess_target(_snap([_node("button")]), selector="")
        assert result.matchable is False


class TestAssessTargetByText:
    def test_visible_text_substring_found(self):
        snapshot = _snap([_node("button", text="Sign In")])
        result = assess_target(snapshot, text="Sign In")
        assert result.matchable is True
        assert result.found is True

    def test_text_match_is_case_insensitive(self):
        snapshot = _snap([_node("button", text="Sign In")])
        result = assess_target(snapshot, text="sign in")
        assert result.found is True

    def test_text_not_found(self):
        snapshot = _snap([_node("button", text="Cancel")])
        result = assess_target(snapshot, text="Sign In")
        assert result.matchable is True
        assert result.found is False

    def test_invisible_text_node_not_counted_as_visible_match(self):
        snapshot = _snap([_node("button", text="Sign In", visible=False)])
        result = assess_target(snapshot, text="Sign In")
        assert result.found is True
        assert result.visible is False
