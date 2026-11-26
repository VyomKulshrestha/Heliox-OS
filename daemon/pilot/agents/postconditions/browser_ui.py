"""Post-conditions for clicks, tabs, history, hover, scroll and closing the browser, read from the browser afresh.

Heliox drives one browser, Chromium through Playwright (``pilot.system.browser``; ``HELIOX_BROWSER_HEADLESS``
runs the same browser without a window). There is no extension backend and no second browser. Each handler acts
through Playwright's action calls (click, go_back, reload, bring_to_front, close). Each check here reads afterwards
through query calls only: a short read-only script for the address, the document's load time and the scroll
position, a locator read of the target's state, the tab list the driver keeps from the browser's own events, the
tab's history over a separate DevTools session (``Page.getNavigationHistory``), and the operating system's process
table. Nothing here clicks, types, scrolls, navigates or repeats the action.

What a handler recorded before acting (``[PAGE_BEFORE]`` in its output: a hash of the address, the document's load
time, the history entry ids, the scroll position, the tab count, the target's state) is used only as the baseline
to compare with; it is never the evidence. Pages change by themselves, so a mismatch is only returned for something
the action should have produced and did not; any other doubt is "not verified" with its reason. Details never carry
page text, field values or addresses beyond the host.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlsplit

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified
from pilot.agents.unconfirmed import Kind
from pilot.timeouts import bounded

# Each check runs right after its step (postconditions/_in_step.py), so these waits add to the step: kept short.
SETTLE_SECONDS = 1.0
SCROLL_READS = 5
SCROLL_INTERVAL_SECONDS = 0.1
POLL_ATTEMPTS = 8
POLL_INTERVAL_SECONDS = 0.25
LEFT_THE_PAGE = "A later step left this page, so this step cannot be checked now"
MAX_TABS_READ = 30

_REF = re.compile(r"^e\d{1,4}$")
_SNAPSHOT_ID = re.compile(r"^bx\d{1,9}-[0-9a-f]{6}$")


async def _pause(seconds: float) -> None:
    """Wait between two reads; tests replace it so no real time passes."""
    await asyncio.sleep(seconds)


async def _settle(page: Any) -> None:
    """Let a navigation the action started reach DOMContentLoaded, within a bound; then read whatever is there."""
    try:
        await bounded(page.wait_for_load_state("domcontentloaded"), SETTLE_SECONDS)
    except Exception:
        pass


def _host(url: str | None) -> str:
    """The host only: a path or query can carry a token."""
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return "the page"
    if parts.scheme == "about":
        return f"about:{parts.path}"
    return parts.hostname or "the page"


def _origin_and_path(url: str) -> tuple[str, str, str]:
    from pilot.system.browser import _canonical_browser_url

    parts = urlsplit(_canonical_browser_url(url))
    return parts.scheme, parts.netloc, parts.path.rstrip("/") or "/"


def _left_since(before: dict[str, Any] | None, now: dict[str, Any] | None) -> bool:
    """When the check runs after the plan rather than right after its step: the tab shows another page than the one
    recorded before the step (another address or another document), so a later step may have moved it on."""
    from pilot.agents.postconditions._in_step import right_after_step
    from pilot.system.browser import url_fingerprint

    if right_after_step() or not before or not now:
        return False
    other_address = bool(before.get("u")) and url_fingerprint(str(now.get("href") or "")) != before.get("u")
    other_document = before.get("o") is not None and now.get("origin") != before.get("o")
    return other_address or other_document


def _record(result: ActionResult) -> dict[str, Any] | None:
    """What the handler recorded before acting: in ``observation`` (the executor moves it there), else the output."""
    from pilot.system.browser import parse_page_before

    return parse_page_before("\n".join(part for part in (result.output, result.observation) if part))


def _target_selector(result: ActionResult) -> str | None:
    """The selector the handler acted on, rebuilt from the plan's parameters (not from its output)."""
    from pilot.system.browser import click_text_selector

    params = result.action.parameters
    if result.action.action_type == ActionType.BROWSER_CLICK_TEXT:
        text = str(getattr(params, "text", "") or "")
        return click_text_selector(text, bool(getattr(params, "exact", False))) if text else None
    ref = getattr(params, "element_ref", None)
    snapshot_id = getattr(params, "snapshot_id", None)
    if ref and snapshot_id and _REF.fullmatch(ref) and _SNAPSHOT_ID.fullmatch(snapshot_id):
        return f'[data-heliox-ref="{ref}"]'
    return str(getattr(params, "selector", "") or "") or None


# ── Clicks ───────────────────────────────────────────────────────────

_STATE_WORDS = {
    "on": "checked",
    "expanded": "expanded",
    "pressed": "pressed",
    "selected": "selected",
    "open": "open",
    "disabled": "disabled",
}


def _should_toggle(target: dict[str, Any]) -> bool:
    """A click on an enabled checkbox or switch flips it; on an unchecked radio button it checks it."""
    if target.get("disabled"):
        return False
    kind = target.get("kind")
    on = target.get("on")
    if kind in {"checkbox", "switch"}:
        return on is not None
    return kind == "radio" and on in {False, "false"}


@register(
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    method="conditional_click_effect_postcondition",
)
async def click_was_followed_by_an_effect(result: ActionResult) -> Outcome:
    """Something the click should cause is on the page now: a navigation, the target's new state, new elements.

    The click itself (target present, visible, enabled, not covered) is Playwright's actionability check in the
    handler, which raises when it fails, so a missed target is a failed action before this runs.
    """
    from pilot.agents.postconditions._in_step import right_after_step
    from pilot.system import browser
    from pilot.system.dom_diff import snapshot_dom

    before = _record(result)
    if before is None:
        return unobserved(
            "The page was not recorded before the click, so what followed it cannot be judged", kind=Kind.COULD_NOT_LOOK
        )
    page = browser.current_tab()
    if page is None:
        return unobserved(
            "No browser tab is open after the click, so the page could not be read", kind=Kind.COULD_NOT_LOOK
        )
    await _settle(page)
    now = await browser.read_page_state(page)
    if now is None:
        return unobserved(
            "The page could not be read after the click; it may still be loading", kind=Kind.COULD_NOT_LOOK
        )

    if _left_since(before, now) or (
        not right_after_step() and isinstance(before.get("n"), int) and len(browser.open_tabs() or []) > before["n"]
    ):
        # Read after the plan: a navigation or a new tab may be a later step's, not this click's.
        return unobserved(
            "The page changed after this click, and a later step may have changed it", kind=Kind.CHANGED_LATER
        )

    tabs_before = before.get("n")
    tabs_now = browser.open_tabs() or []
    if isinstance(tabs_before, int) and len(tabs_now) > tabs_before:
        # The handler works in a tab its click opened, so the tab read here is that new one.
        return verified(f"The click opened a new tab at {_host(now.get('href'))}")

    new_document = before.get("o") is not None and now.get("origin") != before.get("o")
    new_address = bool(before.get("u")) and browser.url_fingerprint(str(now.get("href"))) != before.get("u")
    if new_document or new_address:
        return verified(f"The click was followed by a navigation; the tab now shows {_host(now.get('href'))}")

    target_before = before.get("t")
    selector = _target_selector(result)
    if selector and isinstance(target_before, dict) and target_before.get("count"):
        target_now = await browser.read_element_state(page, selector)
        if target_now is not None:
            if not target_now.get("count"):
                return verified("The clicked element is no longer on the page; what it belonged to closed or changed")
            changed = [
                word for key, word in _STATE_WORDS.items() if str(target_before.get(key)) != str(target_now.get(key))
            ]
            if changed:
                return verified(f"The clicked element changed: {', '.join(changed)}")
            if _should_toggle(target_before):
                state = "checked" if target_now.get("on") in {True, "true"} else "unchecked"
                return mismatch(
                    f"The {target_before.get('kind')} was clicked but is still {state}; the page did not take the click"
                )

    snapshot = await snapshot_dom(page)
    structure = browser.structure_fingerprint(snapshot)
    if before.get("s") and structure and structure != before.get("s"):
        return verified("The page changed after the click: elements appeared or disappeared")
    return unobserved("The click landed on the element; no change was observed after it", kind=Kind.NOTHING_HAPPENED)


# ── Tabs ─────────────────────────────────────────────────────────────


@register(ActionType.BROWSER_NEW_TAB, method="conditional_new_tab_postcondition")
async def new_tab_is_open(result: ActionResult) -> Outcome:
    """One more tab is open and the newest shows the requested site (or is blank when none was asked)."""
    from pilot.system import browser

    before = _record(result) or {}
    tabs = browser.open_tabs()
    if tabs is None:
        return unobserved("The browser is no longer open, so the new tab could not be read", kind=Kind.COULD_NOT_LOOK)
    count_before = before.get("n")
    if isinstance(count_before, int) and len(tabs) == count_before:
        return mismatch(f"No new tab is open: {len(tabs)} tab(s), as before")
    if not tabs or (isinstance(count_before, int) and len(tabs) < count_before):
        return unobserved("Tabs were closed meanwhile, so the new one cannot be told apart", kind=Kind.CHANGED_LATER)
    newest = tabs[-1]
    await _settle(newest)
    state = await browser.read_page_state(newest)
    if state is None:
        return unobserved("The new tab could not be read; it may still be loading", kind=Kind.COULD_NOT_LOOK)
    href = str(state.get("href") or "")
    requested = str(getattr(result.action.parameters, "url", "") or "")
    if not requested:
        if href == "about:blank":
            return verified("A new blank tab is open")
        return unobserved("The newest tab is not blank; it may be a pop-up a page opened", kind=Kind.CHANGED_LATER)
    if href == "about:blank":
        asked = requested if "://" in requested else "https://" + requested
        return mismatch(f"The new tab is open but blank; {_host(asked)} did not load")
    try:
        if _origin_and_path(href) == _origin_and_path(requested):
            return verified(f"A new tab is open on {_host(href)}")
    except ValueError:
        pass
    return unobserved(
        f"The new tab shows {_host(href)}, not the address asked for; it may have been redirected",
        kind=Kind.CHANGED_LATER,
    )


@register(ActionType.BROWSER_SWITCH_TAB, method="conditional_front_tab_postcondition")
async def switched_tab_is_in_front(result: ActionResult) -> Outcome:
    """The chosen tab is the visible one and the others are hidden, as each tab's document reports it."""
    from pilot.system import browser

    tabs = browser.open_tabs()
    if not tabs:
        return unobserved("No browser tab is open now, so the front tab could not be read", kind=Kind.COULD_NOT_LOOK)
    index = int(getattr(result.action.parameters, "tab_index", -1))
    if not -len(tabs) <= index < len(tabs):
        return unobserved(
            "Tabs were closed meanwhile, so the chosen tab is no longer at that place", kind=Kind.CHANGED_LATER
        )
    if len(tabs) > MAX_TABS_READ:
        return unobserved(f"More than {MAX_TABS_READ} tabs are open; they were not all read", kind=Kind.COULD_NOT_LOOK)
    target = tabs[index]
    visibility: list[str | None] = []
    for tab in tabs:
        state = await browser.read_page_state(tab)
        visibility.append(state.get("visibility") if state else None)
    target_visibility = visibility[tabs.index(target)]
    others = [value for tab, value in zip(tabs, visibility, strict=True) if tab is not target]
    if target_visibility == "hidden":
        return mismatch(f"Tab {index} was brought to the front but its page reports itself hidden")
    if target_visibility == "visible" and all(value == "hidden" for value in others):
        return verified(f"Tab {index} is the visible tab; the other {len(others)} are hidden")
    return unobserved(
        "Every tab reports itself visible (a browser without a window does), so which one is in front cannot be read",
        kind=Kind.NOT_OBSERVABLE,
    )


@register(ActionType.BROWSER_CLOSE_TAB, method="conditional_tab_closed_postcondition")
async def tab_is_closed(result: ActionResult) -> Outcome:
    """The browser reports one tab fewer than before the close."""
    from pilot.system import browser

    before = _record(result)
    tabs = browser.open_tabs()
    if tabs is None:
        return unobserved("The browser is no longer open, so its tabs could not be counted", kind=Kind.COULD_NOT_LOOK)
    if before is None or not isinstance(before.get("n"), int):
        if not tabs:
            return verified("No browser tab is open")
        return unobserved("The tab count before the close was not recorded", kind=Kind.COULD_NOT_LOOK)
    count_before = before["n"]
    if len(tabs) == count_before - 1:
        return verified(f"{len(tabs)} tab(s) remain open, one fewer than before")
    if len(tabs) == count_before:
        return mismatch(f"{len(tabs)} tab(s) are still open, as before the close; no tab was closed")
    return unobserved("Pages opened or closed tabs meanwhile, so the count does not tell", kind=Kind.CHANGED_LATER)


@register(ActionType.BROWSER_CLOSE, method="conditional_browser_exited_postcondition", timeout=20.0)
async def browser_processes_exited(result: ActionResult) -> Outcome:
    """Every process of the browser Heliox started has exited, read from the operating system's process table."""
    before = _record(result)
    processes = (before or {}).get("pids")
    if not processes:
        if str(result.output or "").startswith("No browser was open"):
            return unobserved("No browser was open, so there was nothing to watch close", kind=Kind.COULD_NOT_LOOK)
        return unobserved(
            "The browser's processes could not be found before the close, so their exit was not watched",
            kind=Kind.COULD_NOT_LOOK,
        )
    import psutil

    running: list[int] = []
    for attempt in range(POLL_ATTEMPTS):
        running = [int(pid) for pid, started in processes if _still_running(psutil, int(pid), float(started))]
        if not running:
            return verified(f"All {len(processes)} browser process(es) have exited")
        if attempt + 1 < POLL_ATTEMPTS:
            await _pause(POLL_INTERVAL_SECONDS)
    return mismatch(
        f"{len(running)} of {len(processes)} browser process(es) were still running "
        f"{POLL_ATTEMPTS * POLL_INTERVAL_SECONDS:g} s after the close"
    )


def _still_running(psutil: Any, pid: int, started: float) -> bool:
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - started) > 0.01:
            return False  # the id now belongs to another process
        return process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# ── History ──────────────────────────────────────────────────────────


async def _history_moved(result: ActionResult, *, key: str, word: str) -> Outcome:
    from pilot.system import browser

    before = _record(result)
    if before is None or before.get("cur") is None or before.get(key) is None:
        return unobserved(
            "The tab's place in its history was not recorded before (this browser does not report it)",
            kind=Kind.COULD_NOT_LOOK,
        )
    page = browser.current_tab()
    if page is None:
        return unobserved("No browser tab is open now, so its history could not be read", kind=Kind.COULD_NOT_LOOK)
    await _settle(page)
    history = await browser.read_history(page)
    if history is None:
        return unobserved("The tab's history could not be read from the browser", kind=Kind.COULD_NOT_LOOK)
    current = history["ids"][history["index"]]
    if current == before[key]:
        state = await browser.read_page_state(page)
        where = f", to {_host(state.get('href'))}" if state else ""
        return verified(f"The tab went {word} one page in its history{where}")
    if current == before["cur"]:
        return mismatch(f"The tab is still on the same page of its history; it did not go {word}")
    return unobserved(
        f"The tab is on another page of its history than one step {word}; the page may have redirected",
        kind=Kind.CHANGED_LATER,
    )


@register(ActionType.BROWSER_BACK, method="conditional_history_back_postcondition")
async def went_back(result: ActionResult) -> Outcome:
    """The tab's current history entry is the one before the entry it was on."""
    return await _history_moved(result, key="prev", word="back")


@register(ActionType.BROWSER_FORWARD, method="conditional_history_forward_postcondition")
async def went_forward(result: ActionResult) -> Outcome:
    """The tab's current history entry is the one after the entry it was on."""
    return await _history_moved(result, key="next", word="forward")


@register(ActionType.BROWSER_REFRESH, method="conditional_reload_postcondition")
async def page_was_reloaded(result: ActionResult) -> Outcome:
    """A new document was loaded at the same history entry: the document's load time changed."""
    from pilot.system import browser

    before = _record(result)
    if before is None or before.get("o") is None:
        return unobserved("The page's load time was not recorded before the refresh", kind=Kind.COULD_NOT_LOOK)
    page = browser.current_tab()
    if page is None:
        return unobserved("No browser tab is open now, so the page could not be read", kind=Kind.COULD_NOT_LOOK)
    await _settle(page)
    if before.get("cur") is not None:
        history = await browser.read_history(page)
        if history is not None and history["ids"][history["index"]] != before["cur"]:
            return unobserved(
                "The tab is on another page of its history now; the reload cannot be judged", kind=Kind.CHANGED_LATER
            )
    state = await browser.read_page_state(page)
    if state is None or state.get("origin") is None:
        return unobserved(
            "The page could not be read after the refresh; it may still be loading", kind=Kind.COULD_NOT_LOOK
        )
    if state["origin"] != before["o"]:
        return verified("The page was loaded again: a new document at the same place")
    return mismatch("The page was not reloaded: it is still the document that was there before the refresh")


# ── Hover and scroll ─────────────────────────────────────────────────


@register(ActionType.BROWSER_HOVER, method="conditional_pointer_over_element_postcondition")
async def pointer_is_over_element(result: ActionResult) -> Outcome:
    """The element is in the visible part of the page and the browser reports it under the pointer (:hover)."""
    from pilot.system import browser

    selector = _target_selector(result)
    if not selector:
        return unobserved("The hover named no element to read", kind=Kind.COULD_NOT_LOOK)
    page = browser.current_tab()
    if page is None:
        return unobserved("No browser tab is open now, so the element could not be read", kind=Kind.COULD_NOT_LOOK)
    before = _record(result)
    if before and _left_since(before, await browser.read_page_state(page)):
        return unobserved(LEFT_THE_PAGE, kind=Kind.CHANGED_LATER)
    state = await browser.read_element_state(page, selector)
    if state is None:
        return unobserved("The element could not be read after the hover", kind=Kind.COULD_NOT_LOOK)
    if not state.get("count"):
        return unobserved(
            "The element is no longer on the page, so the hover cannot be judged", kind=Kind.COULD_NOT_LOOK
        )
    if state.get("hovered"):
        return verified("The pointer is over the element")
    if not state.get("in_view"):
        return mismatch("The element is outside the visible part of the page, so the pointer is not over it")
    return unobserved(
        "The element is visible but not under the pointer now; the pointer may have moved since",
        kind=Kind.CHANGED_LATER,
    )


def _same_panel(before: dict[str, Any], now: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The scrolling element under the middle of the window before and now, when it is the same one; else Nones."""
    panel_before, panel_now = before.get("panel"), now.get("panel")
    if (
        isinstance(panel_before, dict)
        and isinstance(panel_now, dict)
        and panel_before.get("key") == panel_now.get("key")
    ):
        return panel_before, panel_now
    return None, None


def _scroll_outcome(direction: str, before: dict[str, Any], now: dict[str, Any]) -> Outcome | None:
    """VERIFIED when the document or the scrolling element under the pointer answers the direction, or the page
    could not move that way to begin with; None when the reading does not show it (yet)."""
    from pilot.system.browser import scroll_end_already_reached

    sign = 1 if direction in {"down", "bottom"} else -1
    word = "down" if sign > 0 else "up"
    y_before, y_now = float(before["y"]), float(now.get("y") or 0)
    if sign * (y_now - y_before) > 0:
        return verified(f"The page moved {word} by {round(abs(y_now - y_before))} px")
    panel_before, panel_now = _same_panel(before, now)
    if panel_before and panel_now:
        moved = float(panel_now.get("top") or 0) - float(panel_before.get("top") or 0)
        if sign * moved > 0:
            return verified(f"The scrolling area moved {word} by {round(abs(moved))} px")
    already = scroll_end_already_reached(direction, before)
    if already:
        return verified(already)
    if direction == "top" and y_now <= 1 and (not panel_now or float(panel_now.get("top") or 0) <= 1):
        return verified("The page is at the top")
    if direction == "bottom" and y_now >= float(now.get("max_y") or 0) - 1:
        if not panel_now or float(panel_now.get("top") or 0) >= float(panel_now.get("max") or 0) - 1:
            return verified("The page is at the bottom")
    return None


def _moved_or_grew(before: dict[str, Any], now: dict[str, Any]) -> bool:
    """Any change of position in either direction, or a page or scrolling area that grew (a feed loading more)."""
    if round(float(now.get("y") or 0)) != round(float(before["y"])):
        return True
    if float(now.get("max_y") or 0) > float(before["max_y"]) + 1:
        return True
    panel_before, panel_now = before.get("panel"), now.get("panel")
    if (panel_before or None) != (panel_now or None) and not (
        isinstance(panel_before, dict)
        and isinstance(panel_now, dict)
        and panel_before.get("key") == panel_now.get("key")
    ):
        return True  # the scrolling area under the pointer appeared, went or was replaced
    if isinstance(panel_before, dict) and isinstance(panel_now, dict):
        if round(float(panel_now.get("top") or 0)) != round(float(panel_before.get("top") or 0)):
            return True
        if float(panel_now.get("max") or 0) > float(panel_before.get("max") or 0) + 1:
            return True
    return False


@register(ActionType.BROWSER_SCROLL, method="conditional_scroll_position_postcondition")
async def page_scrolled(result: ActionResult) -> Outcome:
    """The document or the scrolling element under the pointer moved the way asked, or the page could not move that
    way (already at that end, or it fits the window), read right after the step.

    A scroll that did not move is reported with its reason and never fails the run: a scroll is not worth a repair
    round or a stop, and nothing in a plan says whether the next step depends on how far the page moved.
    """
    from pilot.system import browser

    direction = str(getattr(result.action.parameters, "direction", "down") or "down")
    before = _record(result)
    if before is None or before.get("y") is None or before.get("max_y") is None:
        return unobserved("The scroll position was not recorded before the scroll", kind=Kind.COULD_NOT_LOOK)
    page = browser.current_tab()
    if page is None:
        return unobserved(
            "No browser tab is open now, so the scroll position could not be read", kind=Kind.COULD_NOT_LOOK
        )
    now = await browser.read_page_state(page)
    if _left_since(before, now):
        return unobserved(LEFT_THE_PAGE, kind=Kind.CHANGED_LATER)
    for attempt in range(SCROLL_READS):  # smooth scrolling takes a moment to arrive
        if attempt:
            await _pause(SCROLL_INTERVAL_SECONDS)
            now = await browser.read_page_state(page)
        if now is not None:
            outcome = _scroll_outcome(direction, before, now)
            if outcome is not None:
                return outcome
    if now is None:
        return unobserved("The scroll position could not be read after the scroll", kind=Kind.COULD_NOT_LOOK)
    if _moved_or_grew(before, now):
        return unobserved(
            "The page moved or grew, but not simply the way asked; it may load more as it scrolls",
            kind=Kind.NOT_OBSERVABLE,
        )
    return unobserved(f"The page did not move when scrolled {direction}", kind=Kind.NOTHING_HAPPENED)
