"""DOM Diff Engine — mathematical before/after DOM comparison for browser actions.

Captures a structural snapshot of the live DOM via Playwright's JS bridge,
then computes a precise diff between two snapshots to determine whether a
browser action actually changed the UI.

Key concepts
------------
- ``DomSnapshot``     — lightweight structural fingerprint of the DOM tree
- ``DomDiff``         — result of comparing two snapshots (added/removed/mutated
                        nodes + a numeric change_score in [0.0, 1.0])
- ``DomUnchangedError`` — raised when change_score < threshold, signalling that
                          the action had no visible effect and self-correction
                          should be attempted
- ``snapshot_dom()``  — async function that captures a snapshot from the live page
- ``diff_dom()``      — pure function that computes a DomDiff from two snapshots

Design goals
------------
- Zero extra dependencies: runs entirely through Playwright's evaluate() bridge
- Fast: only serialises tag, id, class, visible text (≤120 chars), bounding box,
  and a per-subtree hash — not the full innerHTML
- Deterministic: same DOM always produces the same snapshot hash
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("pilot.system.dom_diff")

# ---------------------------------------------------------------------------
# JS injected into the page to build a compact DOM snapshot
# ---------------------------------------------------------------------------

_SNAPSHOT_JS = """
(() => {
    const MAX_NODES = 2000;   // cap to avoid huge payloads on complex pages
    const MAX_TEXT  = 120;    // max chars of visible text per node

    function nodeHash(tag, id, cls, text) {
        return tag + '|' + id + '|' + cls + '|' + text.substring(0, 40);
    }

    function walk(el, nodes, depth) {
        if (nodes.length >= MAX_NODES) return;
        if (el.nodeType !== 1) return;   // elements only

        const tag  = el.tagName.toLowerCase();
        const id   = el.id || '';
        const cls  = (el.className && typeof el.className === 'string')
                     ? el.className.trim().split(/\\s+/).slice(0, 5).join(' ')
                     : '';
        const text = (el.innerText || '').trim().substring(0, MAX_TEXT);
        const rect = el.getBoundingClientRect();
        const visible = rect.width > 0 && rect.height > 0
                        && window.getComputedStyle(el).visibility !== 'hidden'
                        && window.getComputedStyle(el).display !== 'none';

        nodes.push({
            tag, id, cls, text, depth,
            visible,
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            w: Math.round(rect.width),
            h: Math.round(rect.height),
            key: nodeHash(tag, id, cls, text),
        });

        for (const child of el.children) {
            walk(child, nodes, depth + 1);
        }
    }

    const nodes = [];
    walk(document.body || document.documentElement, nodes, 0);
    return nodes;
})()
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DomNode:
    """Lightweight representation of a single DOM element."""

    tag: str
    id: str
    cls: str
    text: str
    depth: int
    visible: bool
    x: int
    y: int
    w: int
    h: int
    key: str  # structural identity key (tag|id|cls|text[:40])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DomNode:
        return cls(
            tag=d.get("tag", ""),
            id=d.get("id", ""),
            cls=d.get("cls", ""),
            text=d.get("text", ""),
            depth=d.get("depth", 0),
            visible=d.get("visible", False),
            x=d.get("x", 0),
            y=d.get("y", 0),
            w=d.get("w", 0),
            h=d.get("h", 0),
            key=d.get("key", ""),
        )


@dataclass
class DomSnapshot:
    """Structural fingerprint of a page's DOM at a point in time."""

    nodes: list[DomNode] = field(default_factory=list)
    url: str = ""
    title: str = ""

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def visible_count(self) -> int:
        return sum(1 for n in self.nodes if n.visible)

    @property
    def fingerprint(self) -> str:
        """SHA-256 of all node keys concatenated — changes if any node changes."""
        raw = "|".join(n.key for n in self.nodes)
        return hashlib.sha256(raw.encode()).hexdigest()

    def key_set(self) -> set[str]:
        """Set of all node identity keys (used for set-difference diffing)."""
        return {n.key for n in self.nodes}


@dataclass
class DomDiff:
    """Result of comparing two DOM snapshots."""

    added: list[DomNode] = field(default_factory=list)  # nodes in after, not in before
    removed: list[DomNode] = field(default_factory=list)  # nodes in before, not in after
    mutated: list[DomNode] = field(default_factory=list)  # nodes whose position/size changed
    change_score: float = 0.0  # 0.0 = identical, 1.0 = completely different
    before_count: int = 0
    after_count: int = 0
    url_changed: bool = False
    title_changed: bool = False

    @property
    def changed(self) -> bool:
        """True if any structural change was detected."""
        return self.change_score > 0.0

    def summary(self) -> str:
        """Human-readable one-line summary of the diff."""
        parts: list[str] = []
        if self.url_changed:
            parts.append("URL changed")
        if self.title_changed:
            parts.append("title changed")
        if self.added:
            parts.append(f"+{len(self.added)} nodes added")
        if self.removed:
            parts.append(f"-{len(self.removed)} nodes removed")
        if self.mutated:
            parts.append(f"~{len(self.mutated)} nodes moved/resized")
        if not parts:
            return "no change detected"
        return f"change_score={self.change_score:.2f} | " + ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for inclusion in ActionResult.output."""
        return {
            "change_score": round(self.change_score, 4),
            "changed": self.changed,
            "url_changed": self.url_changed,
            "title_changed": self.title_changed,
            "added_count": len(self.added),
            "removed_count": len(self.removed),
            "mutated_count": len(self.mutated),
            "before_node_count": self.before_count,
            "after_node_count": self.after_count,
            "summary": self.summary(),
            "added_sample": [{"tag": n.tag, "id": n.id, "text": n.text[:60]} for n in self.added[:5]],
            "removed_sample": [{"tag": n.tag, "id": n.id, "text": n.text[:60]} for n in self.removed[:5]],
        }


class DomUnchangedError(Exception):
    """Raised when a browser action produced no detectable DOM change.

    Attributes
    ----------
    diff:       The DomDiff that triggered the error.
    action_desc: Human-readable description of the action that was attempted.
    """

    def __init__(self, diff: DomDiff, action_desc: str = "") -> None:
        self.diff = diff
        self.action_desc = action_desc
        super().__init__(
            f"DOM unchanged after '{action_desc}' (change_score={diff.change_score:.3f}). Self-correction required."
        )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


async def snapshot_dom(page: Any) -> DomSnapshot:
    """Capture a structural DOM snapshot from a live Playwright page.

    Parameters
    ----------
    page:
        A Playwright ``Page`` object (``playwright.async_api.Page``).

    Returns
    -------
    DomSnapshot
        Lightweight fingerprint of the current DOM state.
    """
    try:
        raw_nodes: list[dict] = await page.evaluate(_SNAPSHOT_JS)
    except Exception as exc:
        logger.warning("DOM snapshot failed (page may be navigating): %s", exc)
        raw_nodes = []

    nodes = [DomNode.from_dict(d) for d in (raw_nodes or [])]

    try:
        url = page.url
        title = await page.title()
    except Exception:
        url = ""
        title = ""

    return DomSnapshot(nodes=nodes, url=url, title=title)


def diff_dom(before: DomSnapshot, after: DomSnapshot) -> DomDiff:
    """Compute a mathematical diff between two DOM snapshots.

    The ``change_score`` is calculated as:

        score = w_struct * structural_delta
              + w_url    * url_changed
              + w_title  * title_changed

    where ``structural_delta`` is the Jaccard distance between the two
    node key-sets, clamped to [0, 1].

    Parameters
    ----------
    before:
        Snapshot taken immediately before the browser action.
    after:
        Snapshot taken immediately after the browser action.

    Returns
    -------
    DomDiff
        Full diff including added/removed/mutated node lists and change_score.
    """
    before_keys = before.key_set()
    after_keys = after.key_set()

    added_keys = after_keys - before_keys
    removed_keys = before_keys - after_keys

    # Build lookup maps for position-change detection
    before_map: dict[str, DomNode] = {n.key: n for n in before.nodes}
    after_map: dict[str, DomNode] = {n.key: n for n in after.nodes}

    added_nodes = [after_map[k] for k in added_keys if k in after_map]
    removed_nodes = [before_map[k] for k in removed_keys if k in before_map]

    # Detect nodes that stayed structurally identical but moved/resized
    mutated_nodes: list[DomNode] = []
    common_keys = before_keys & after_keys
    for key in common_keys:
        b = before_map[key]
        a = after_map[key]
        # Consider a node "mutated" if its bounding box shifted by >5px
        if abs(a.x - b.x) > 5 or abs(a.y - b.y) > 5 or abs(a.w - b.w) > 5 or abs(a.h - b.h) > 5:
            mutated_nodes.append(a)

    # Jaccard distance on node key-sets
    union_size = len(before_keys | after_keys)
    if union_size == 0:
        structural_delta = 0.0
    else:
        intersection_size = len(common_keys)
        structural_delta = 1.0 - (intersection_size / union_size)

    # Weighted change score
    url_changed = before.url != after.url
    title_changed = before.title != after.title

    W_STRUCT = 0.70
    W_URL = 0.20
    W_TITLE = 0.10

    change_score = W_STRUCT * structural_delta + W_URL * float(url_changed) + W_TITLE * float(title_changed)
    change_score = min(1.0, max(0.0, change_score))

    return DomDiff(
        added=added_nodes,
        removed=removed_nodes,
        mutated=mutated_nodes,
        change_score=change_score,
        before_count=before.node_count,
        after_count=after.node_count,
        url_changed=url_changed,
        title_changed=title_changed,
    )


def assert_dom_changed(
    before: DomSnapshot,
    after: DomSnapshot,
    min_score: float = 0.01,
    action_desc: str = "",
) -> DomDiff:
    """Compute the diff and raise ``DomUnchangedError`` if change_score < min_score.

    Parameters
    ----------
    before:
        Pre-action DOM snapshot.
    after:
        Post-action DOM snapshot.
    min_score:
        Minimum change_score required to consider the action successful.
        Default 0.01 (1% of nodes must change).
    action_desc:
        Human-readable description of the action, used in the error message.

    Returns
    -------
    DomDiff
        The computed diff (only returned if change_score >= min_score).

    Raises
    ------
    DomUnchangedError
        If the DOM did not change enough to meet the threshold.
    """
    result = diff_dom(before, after)
    if result.change_score < min_score:
        raise DomUnchangedError(diff=result, action_desc=action_desc)
    return result
