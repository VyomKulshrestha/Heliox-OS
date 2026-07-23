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
- ``assess_target()`` — pure function that checks whether a click/type/select
                        target (CSS selector or text) resolves against an
                        *already-captured* snapshot, before the action runs —
                        a lightweight, structural stand-in for a full
                        generative "predict the outcome" world model (see
                        SECURITY.md's Pre-Execution Target Assessment section
                        for why a generative visual world model wasn't used).

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
import re
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


# ---------------------------------------------------------------------------
# Pre-execution target assessment
# ---------------------------------------------------------------------------

# Matches only the simple, statically-resolvable CSS selector forms this
# module can honestly evaluate against a flat node list: an optional bare
# tag, an optional #id, and zero or more chained .class tokens — e.g.
# "button", "#submit-btn", ".btn.primary", "button#submit.btn.primary".
# Combinators (descendant/child/sibling), attribute selectors, and
# pseudo-classes (":has-text(...)", ":nth-child", etc.) all fail this
# pattern and are deliberately reported as "not matchable" rather than
# guessed at — see TargetAssessment.matchable.
_SIMPLE_SELECTOR_RE = re.compile(
    r"^(?P<tag>[a-zA-Z][a-zA-Z0-9-]*)?"
    r"(?:#(?P<id>[a-zA-Z_-][\w-]*))?"
    r"(?P<classes>(?:\.[a-zA-Z_-][\w-]*)*)$"
)


@dataclass
class TargetAssessment:
    """Result of checking a click/type/select target against a DOM snapshot
    captured immediately before the action would run.

    ``matchable`` is False when the selector uses syntax this module can't
    honestly evaluate from a flat node list (combinators, attribute
    selectors, pseudo-classes) or no selector/text was given at all —
    callers must not treat that as "target is fine", only as "no opinion".
    """

    matchable: bool
    found: bool = False
    visible: bool = False
    ambiguous: bool = False
    match_count: int = 0
    reason: str = ""


def _parse_simple_selector(selector: str) -> tuple[str | None, str | None, list[str]] | None:
    """Parse a bare-tag/#id/.class(es) selector, or None if unsupported."""
    selector = selector.strip()
    if not selector:
        return None
    m = _SIMPLE_SELECTOR_RE.match(selector)
    if not m:
        return None
    tag = m.group("tag").lower() if m.group("tag") else None
    node_id = m.group("id")
    classes_raw = m.group("classes") or ""
    classes = [c for c in classes_raw.split(".") if c]
    if tag is None and node_id is None and not classes:
        return None
    return tag, node_id, classes


def _summarize_matches(matches: list[DomNode], desc: str) -> TargetAssessment:
    if not matches:
        return TargetAssessment(
            matchable=True,
            found=False,
            match_count=0,
            reason=f"{desc} not found in current page — action would likely fail",
        )

    visible_matches = [n for n in matches if n.visible]
    if not visible_matches:
        return TargetAssessment(
            matchable=True,
            found=True,
            visible=False,
            match_count=len(matches),
            reason=f"{desc} found but not visible ({len(matches)} match(es)) — action would likely fail or need scrolling",
        )

    ambiguous = len(visible_matches) > 1
    reason = (
        f"{desc} matches {len(visible_matches)} visible elements — click may hit the wrong one"
        if ambiguous
        else f"{desc} found and visible"
    )
    return TargetAssessment(
        matchable=True,
        found=True,
        visible=True,
        ambiguous=ambiguous,
        match_count=len(visible_matches),
        reason=reason,
    )


def assess_target(snapshot: DomSnapshot, *, selector: str = "", text: str = "") -> TargetAssessment:
    """Check whether a click/type/select target resolves against `snapshot`.

    This is a structural, deterministic stand-in for "predict the outcome
    before executing" — not a generative model. It answers "does this
    target exist, is it visible, is it ambiguous right now", using the
    *current* live DOM (already captured, no extra page evaluation), not a
    simulated future state. See module docstring.

    Parameters
    ----------
    snapshot:
        A DomSnapshot captured immediately before the action would run.
    selector:
        A CSS selector (as used by ``browser_click``/``browser_type``/etc).
        Only simple tag/#id/.class forms can be evaluated — see
        `_parse_simple_selector`.
    text:
        Visible-text substring (as used by ``browser_click_text``).
        Always matchable since it's a plain substring search.

    Returns
    -------
    TargetAssessment
        `matchable=False` if neither argument was usable — callers should
        treat that as "no prediction available", not as a clean bill of
        health.
    """
    if text:
        needle = text.strip().lower()
        if not needle:
            return TargetAssessment(matchable=False, reason="empty text target")
        matches = [n for n in snapshot.nodes if needle in n.text.lower()]
        return _summarize_matches(matches, f"text '{text}'")

    if selector:
        parsed = _parse_simple_selector(selector)
        if parsed is None:
            return TargetAssessment(
                matchable=False,
                reason=(
                    f"selector '{selector}' too complex to statically assess "
                    "(combinators/attributes/pseudo-classes not supported)"
                ),
            )
        tag, node_id, classes = parsed
        matches = [
            n
            for n in snapshot.nodes
            if (tag is None or n.tag == tag)
            and (node_id is None or n.id == node_id)
            and all(c in n.cls.split() for c in classes)
        ]
        return _summarize_matches(matches, f"selector '{selector}'")

    return TargetAssessment(matchable=False, reason="no selector or text given")


# Action types this module knows how to pre-execution-assess -- the 5
# browser interaction actions whose params carry a resolvable selector/text
# target. Shared by SimulationSandbox (dry-run) and Executor (real
# execution) so both react to the exact same signal.
BROWSER_TARGET_ACTION_TYPES = frozenset(
    {"browser_click", "browser_click_text", "browser_type", "browser_select", "browser_fill_form"}
)


async def assess_browser_action_target(action_type: str, action: Any) -> TargetAssessment | None:
    """Run `assess_target()` against the CURRENT live page for one browser
    interaction action, if (and only if) a browser session is already open.

    Returns None -- not a `TargetAssessment` -- when nothing can be
    assessed at all (wrong action type, no active session, snapshot
    failure). Callers must never launch a browser themselves just to run
    this check: a dry-run has to remain a genuine no-op, and real
    execution shouldn't open a browser purely to pre-check a target for an
    action that isn't going to touch the browser anyway.

    Shared by `pilot.agents.sandbox.SimulationSandbox.simulate()` (dry-run)
    and `pilot.agents.executor.Executor.execute()` (real execution) so
    both react to the identical signal.
    """
    if action_type not in BROWSER_TARGET_ACTION_TYPES:
        return None

    from pilot.system.browser import has_active_session, peek_current_dom_snapshot

    if not has_active_session():
        return None

    try:
        snapshot = await peek_current_dom_snapshot()
    except Exception:
        logger.debug("Pre-execution target assessment failed to snapshot DOM", exc_info=True)
        return None
    if snapshot is None:
        return None

    params = getattr(action, "parameters", None) or getattr(action, "params", None)

    if action_type == "browser_fill_form":
        # Multiple targets (one per field, plus an optional submit button)
        # rather than one selector -- assess each and surface the first
        # problem found, or a summary if all resolve.
        fields = dict(getattr(params, "fields", {}) or {}) if params else {}
        submit_selector = getattr(params, "submit_selector", "") if params else ""
        selectors = list(fields.keys()) + ([submit_selector] if submit_selector else [])
        if not selectors:
            return None
        problems = []
        for sel in selectors:
            result = assess_target(snapshot, selector=sel)
            if result.matchable and (not result.found or not result.visible or result.ambiguous):
                problems.append(result.reason)
        if problems:
            return TargetAssessment(matchable=True, found=False, reason="; ".join(problems))
        return TargetAssessment(
            matchable=True,
            found=True,
            visible=True,
            reason=f"all {len(selectors)} form target(s) found and visible",
        )

    selector = getattr(params, "selector", "") if params else ""
    text = getattr(params, "text", "") if action_type == "browser_click_text" and params else ""
    return assess_target(snapshot, selector=selector, text=text)


async def _attempt_action_on_clone(action_type: str, page: Any, params: Any) -> str | None:
    """Best-effort, real attempt at the action against a scratch clone page
    — deliberately NOT the production `pilot.system.browser` action
    functions (those include a self-correction retry loop meant for the
    real, once-only execution; a dry run only needs one honest attempt and
    must never raise). Returns an error string on failure, None on
    apparent success."""
    try:
        if action_type == "browser_fill_form":
            fields = dict(getattr(params, "fields", {}) or {}) if params else {}
            for selector, value in fields.items():
                await page.fill(selector, str(value), timeout=3000)
            submit_selector = getattr(params, "submit_selector", "") if params else ""
            if submit_selector:
                await page.click(submit_selector, timeout=3000)
        elif action_type == "browser_click":
            selector = getattr(params, "selector", "") if params else ""
            await page.click(selector, timeout=3000)
        elif action_type == "browser_click_text":
            text = getattr(params, "text", "") if params else ""
            await page.get_by_text(text).first.click(timeout=3000)
        elif action_type == "browser_type":
            selector = getattr(params, "selector", "") if params else ""
            text = getattr(params, "text", "") if params else ""
            await page.fill(selector, text, timeout=3000)
        elif action_type == "browser_select":
            selector = getattr(params, "selector", "") if params else ""
            value = getattr(params, "value", "") if params else ""
            await page.select_option(selector, value, timeout=3000)
        else:
            return f"unsupported action type for dry run: {action_type}"
        return None
    except Exception as exc:
        return str(exc)[:200]


async def dry_run_action(action_type: str, action: Any) -> DomDiff | None:
    """Actually run a browser action against an isolated CLONE of the real
    page, and return the REAL, measured before/after DOM diff — not a
    prediction. The clone is a new tab in the same `BrowserContext` as the
    real page (`context.new_page()`), so it shares cookies/session state
    (see `pilot.system.browser`'s single-shared-context design), but it is
    a fully separate `Page`/DOM/JS realm: nothing done to the clone is ever
    visible on the user's real tab, and the clone is always closed before
    this function returns.

    Returns None (not a `DomDiff`) whenever nothing could be measured at
    all — wrong action type, no active session, clone/navigation failure —
    same "no prediction available" contract as `assess_browser_action_target()`.
    This function must NEVER raise and must NEVER launch a browser session
    that wasn't already open.
    """
    if action_type not in BROWSER_TARGET_ACTION_TYPES:
        return None

    from pilot.system.browser import get_real_page_for_clone, has_active_session

    if not has_active_session():
        return None

    real_page = await get_real_page_for_clone()
    if real_page is None:
        return None

    clone = None
    try:
        clone = await real_page.context.new_page()
        await clone.goto(real_page.url, wait_until="domcontentloaded", timeout=5000)

        before = await snapshot_dom(clone)
        params = getattr(action, "parameters", None) or getattr(action, "params", None)
        # The attempt's own error string (if any) is intentionally not
        # surfaced separately -- a failed attempt (e.g. selector not found
        # on the clone) naturally produces change_score=0.0 in the diff
        # below, which is itself the honest, useful signal: "this action
        # would produce no measurable effect."
        await _attempt_action_on_clone(action_type, clone, params)
        after = await snapshot_dom(clone)

        return diff_dom(before, after)
    except Exception:
        logger.debug("Dry-run diff failed for %s", action_type, exc_info=True)
        return None
    finally:
        if clone is not None:
            try:
                await clone.close()
            except Exception:
                pass
