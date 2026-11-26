"""Clicks, tabs, history, hover, scroll and browser close are read back from the browser afresh.

Fakes only: no browser starts and nothing leaves the test. The fake page answers the read-only scripts, locator
reads and DevTools history reads the checks use; the handlers run against it so the record they leave before
acting is the real one. The address carries a token that must never reach a detail.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, BrowserParams
from pilot.agents.postconditions import _in_step, browser_ui, registrations
from pilot.agents.verifier import POSTCONDITION_VERIFIERS, Verifier
from pilot.system import browser, dom_diff

SECRET = "tok-9f8e7d6c5b4a"
CART = f"https://shop.example/cart?session={SECRET}"
DONE = f"https://shop.example/thanks?session={SECRET}"


def _node(tag: str, node_id: str = "", text: str = "", visible: bool = True) -> dict[str, Any]:
    return {"tag": tag, "id": node_id, "cls": "", "text": text, "depth": 1, "visible": visible, "x": 0, "y": 0}


def _checkbox(on: bool | str | None, **extra: Any) -> dict[str, Any]:
    state = {"kind": "checkbox", "on": on, "expanded": None, "pressed": None, "selected": None, "open": None}
    state.update({"disabled": False, "in_view": True, "hovered": False})
    state.update(extra)
    return state


def _button(**extra: Any) -> dict[str, Any]:
    return _checkbox(None, kind="other", **extra)


class _Session:
    def __init__(self, page: _Page) -> None:
        self.page = page

    async def send(self, method: str) -> dict[str, Any]:
        assert method == "Page.getNavigationHistory"
        if self.page.history is None:
            raise RuntimeError("not a Chromium page")
        ids, index = self.page.history
        return {"currentIndex": index, "entries": [{"id": entry, "url": "secret"} for entry in ids]}

    async def detach(self) -> None:
        self.page.sessions_closed += 1


class _PageContext:
    def __init__(self, page: _Page) -> None:
        self.page = page

    async def new_cdp_session(self, page: _Page) -> _Session:
        return _Session(page)


class _Locator:
    def __init__(self, page: _Page, selector: str) -> None:
        self.page = page
        self.selector = selector
        self.first = self

    async def count(self) -> int:
        return 1 if self.selector in self.page.elements else 0

    async def evaluate(self, script: str) -> dict[str, Any]:
        assert script is browser._ELEMENT_STATE_SCRIPT
        return dict(self.page.elements[self.selector])


class _Mouse:
    def __init__(self, page: _Page) -> None:
        self.page = page

    async def move(self, x: float, y: float) -> None:
        self.page.actions.append(f"move:{x},{y}")

    async def wheel(self, dx: float, dy: float) -> None:
        await self.page._act(f"wheel:{dy}")


class _Page:
    """Answers reads; an action runs ``effect`` (what the real page would do) and is logged."""

    def __init__(
        self,
        *,
        href: str = CART,
        origin: float = 1000.0,
        y: float = 0,
        max_y: float = 2000,
        visibility: str = "visible",
        nodes: list[dict[str, Any]] | None = None,
        elements: dict[str, dict[str, Any]] | None = None,
        history: tuple[list[int], int] | None = None,
        effect: Callable[[_Page], None] | None = None,
    ) -> None:
        self.state = {"href": href, "ready": "complete", "visibility": visibility, "origin": origin}
        self.state.update({"x": 0, "y": y, "max_y": max_y, "width": 1280, "height": 720, "panel": None})
        self.mouse = _Mouse(self)
        self.nodes = nodes if nodes is not None else [_node("main", "app"), _node("button", "pay", "Pay")]
        self.elements = elements or {}
        self.history = history
        self.effect = effect
        self.actions: list[str] = []
        self.context = _PageContext(self)
        self.sessions_closed = 0
        self.closed = False
        self.reads = 0

    @property
    def url(self) -> str:
        return self.state["href"]

    async def title(self) -> str:
        return "Checkout"

    def is_closed(self) -> bool:
        return self.closed

    async def evaluate(self, script: str, *args: Any) -> Any:
        if script is browser._PAGE_STATE_SCRIPT:
            self.reads += 1
            return dict(self.state)
        if script is dom_diff._SNAPSHOT_JS:
            return [dict(node) for node in self.nodes]
        self.actions.append(f"script:{script}")
        if self.effect:
            self.effect(self)
        return None

    def locator(self, selector: str) -> _Locator:
        return _Locator(self, selector)

    async def wait_for_load_state(self, state: str = "load", **_: Any) -> None:
        return None

    async def wait_for_timeout(self, timeout: int) -> None:
        return None

    async def _act(self, name: str) -> None:
        self.actions.append(name)
        if self.effect:
            self.effect(self)

    async def click(self, selector: str, **_: Any) -> None:
        await self._act(f"click:{selector}")

    async def hover(self, selector: str, **_: Any) -> None:
        await self._act(f"hover:{selector}")

    async def go_back(self) -> None:
        await self._act("back")

    async def go_forward(self) -> None:
        await self._act("forward")

    async def reload(self) -> None:
        await self._act("reload")

    async def goto(self, url: str, **_: Any) -> None:
        self.state["href"] = url
        self.actions.append("goto")

    async def bring_to_front(self) -> None:
        self.actions.append("front")

    async def close(self) -> None:
        self.closed = True
        self.context_pages.remove(self)

    context_pages: list[_Page]


class _Context:
    def __init__(self, *pages: _Page) -> None:
        self.pages: list[_Page] = list(pages)
        for page in pages:
            page.context_pages = self.pages

    async def new_page(self) -> _Page:
        page = _Page(href="about:blank", nodes=[])
        page.context_pages = self.pages
        self.pages.append(page)
        return page

    async def close(self) -> None:
        self.pages.clear()


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No browser, no driver, no real waiting; the pauses between reads are counted instead."""
    pauses: list[float] = []

    async def pause(seconds: float) -> None:
        pauses.append(seconds)

    monkeypatch.setattr(browser, "_browser_context", None)
    monkeypatch.setattr(browser, "_playwright_instance", None)
    monkeypatch.setattr(browser, "_active_page", None)
    monkeypatch.setattr(browser, "DOM_DIFF_ENABLED", True)
    monkeypatch.setattr(browser_ui, "_pause", pause)
    return pauses


def _open(monkeypatch: pytest.MonkeyPatch, *pages: _Page) -> _Context:
    context = _Context(*pages)
    monkeypatch.setattr(browser, "_browser_context", context)
    return context


def _action(action_type: ActionType, **params: Any) -> Action:
    return Action(action_type=action_type, target="browser", parameters=BrowserParams(**params))


async def _read_right_after(result: ActionResult) -> ActionResult:
    """What the executor does right after a browser step: read the page now and keep the outcome."""
    reading = await _in_step.observe(result)
    if reading:
        result.observation = "\n".join(part for part in (result.observation, reading) if part)
    return result


async def _verify(action: Action, output: str, *, success: bool = True) -> tuple[str, bool, bool]:
    result = ActionResult(action=action, success=success, output=output, error=None if success else "boom")
    await _read_right_after(result)
    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=[action]), [result])
    return verification.details[0], 0 in verification.independently_observed_actions, verification.passed


async def _check(action: Action, output: str) -> browser_ui.Outcome:
    result = await _read_right_after(ActionResult(action=action, success=True, output=output))
    outcome = _in_step.recalled(result)
    assert outcome is not None
    return outcome


async def _check_later(action: Action, output: str) -> browser_ui.Outcome:
    """The check run after the plan, without a reading made right after the step."""
    return await registrations()[action.action_type].check(ActionResult(action=action, success=True, output=output))


def _assert_no_page_secrets(detail: str) -> None:
    assert SECRET not in detail and "cart" not in detail and "thanks" not in detail


# ── Registration ─────────────────────────────────────────────────────

MY_ACTIONS = (
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_CLICK_TEXT,
    ActionType.BROWSER_NEW_TAB,
    ActionType.BROWSER_SWITCH_TAB,
    ActionType.BROWSER_CLOSE_TAB,
    ActionType.BROWSER_CLOSE,
    ActionType.BROWSER_BACK,
    ActionType.BROWSER_FORWARD,
    ActionType.BROWSER_REFRESH,
    ActionType.BROWSER_HOVER,
    ActionType.BROWSER_SCROLL,
)


def test_every_action_is_registered_conditionally_and_never_skips() -> None:
    for action_type in MY_ACTIONS:
        registration = registrations()[action_type]
        assert registration.module == browser_ui.__name__
        assert registration.method.startswith("conditional_")
        assert POSTCONDITION_VERIFIERS[action_type] == registration.method
        assert registration.precheck is False


# ── Clicks ───────────────────────────────────────────────────────────


def _navigate(page: _Page) -> None:
    page.state.update(href=DONE, origin=2000.0)


@pytest.mark.asyncio
async def test_click_followed_by_a_navigation_is_verified_with_the_host_only(monkeypatch) -> None:
    page = _Page(elements={"#pay": _button()}, effect=_navigate)
    _open(monkeypatch, page)
    output = await browser.browser_click("#pay")
    action = _action(ActionType.BROWSER_CLICK, selector="#pay")

    assert page.actions == ["click:#pay"]  # one click, never a second
    label, observed, passed = await _verify(action, output)
    outcome = await _check(action, output)

    assert (label, observed, passed) == ("Action 0 (browser_click): VERIFIED", True, True)
    assert outcome.detail == "The click was followed by a navigation; the tab now shows shop.example"
    _assert_no_page_secrets(output.split(browser.PAGE_BEFORE_MARKER)[1])
    assert page.actions == ["click:#pay"]


@pytest.mark.asyncio
async def test_click_on_a_checkbox_that_flips_is_verified(monkeypatch) -> None:
    page = _Page(elements={"#agree": _checkbox(False)})
    page.effect = lambda p: p.elements["#agree"].update(on=True)
    _open(monkeypatch, page)
    output = await browser.browser_click("#agree")

    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#agree"), output)

    assert outcome == browser_ui.verified("The clicked element changed: checked")


@pytest.mark.asyncio
async def test_click_on_a_checkbox_that_stays_unchecked_is_a_mismatch(monkeypatch) -> None:
    page = _Page(elements={"#agree": _checkbox(False)})
    _open(monkeypatch, page)
    output = await browser.browser_click("#agree")
    action = _action(ActionType.BROWSER_CLICK, selector="#agree")

    label, observed, passed = await _verify(action, output)

    assert label == (
        "Action 0 (browser_click): MISMATCH — The checkbox was clicked but is still unchecked; "
        "the page did not take the click"
    )
    assert observed is True and passed is False
    assert page.actions == ["click:#agree"]


@pytest.mark.asyncio
async def test_click_that_opens_a_dialog_is_verified(monkeypatch) -> None:
    page = _Page(elements={"#more": _button(expanded="false")})
    page.effect = lambda p: p.nodes.append(_node("div", "dialog", "Are you sure?"))
    _open(monkeypatch, page)
    output = await browser.browser_click("#more")

    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#more"), output)

    assert outcome == browser_ui.verified("The page changed after the click: elements appeared or disappeared")


@pytest.mark.asyncio
async def test_click_that_closes_its_own_dialog_is_verified(monkeypatch) -> None:
    page = _Page(elements={"#close": _button()})
    page.effect = lambda p: p.elements.pop("#close")
    _open(monkeypatch, page)
    output = await browser.browser_click("#close")

    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#close"), output)

    assert outcome.passed and outcome.observed
    assert outcome.detail.startswith("The clicked element is no longer on the page")


@pytest.mark.asyncio
async def test_click_with_no_visible_consequence_is_not_verified_and_not_a_mismatch(monkeypatch) -> None:
    page = _Page(elements={"#pay": _button()})
    # A clock ticking on the page changes text, which is not counted as the click's effect.
    page.effect = lambda p: p.nodes.__setitem__(1, _node("button", "pay", "Pay (12:01)"))
    _open(monkeypatch, page)
    output = await browser.browser_click("#pay")
    action = _action(ActionType.BROWSER_CLICK, selector="#pay")

    label, observed, passed = await _verify(action, output)
    outcome = await _check(action, output)

    assert (label, observed, passed) == (
        "Action 0 (browser_click): REPORTED — no independent postcondition "
        "(The click landed on the element; no change was observed after it)",
        False,
        True,
    )
    assert outcome == browser_ui.unobserved("The click landed on the element; no change was observed after it")


@pytest.mark.asyncio
async def test_click_the_handler_could_not_make_is_a_failed_action(monkeypatch) -> None:
    page = _Page(elements={"#pay": _button()})

    async def covered(selector: str, **_: Any) -> None:
        raise TimeoutError("<div class=overlay> intercepts pointer events")

    page.click = covered  # type: ignore[method-assign]
    _open(monkeypatch, page)
    with pytest.raises(TimeoutError):
        await browser.browser_click("#pay")

    label, observed, passed = await _verify(_action(ActionType.BROWSER_CLICK, selector="#pay"), "", success=False)

    assert label == "Action 0 (browser_click): FAILED — boom"
    assert observed is False and passed is False


@pytest.mark.asyncio
async def test_click_without_a_record_from_before_is_not_verified(monkeypatch) -> None:
    page = _Page(elements={"#pay": _button()}, effect=_navigate)
    _open(monkeypatch, page)
    monkeypatch.setattr(browser, "DOM_DIFF_ENABLED", False)
    output = await browser.browser_click("#pay")

    assert output == "Clicked: #pay"
    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#pay"), output)
    assert outcome.observed is False and "not recorded before the click" in outcome.detail


@pytest.mark.asyncio
async def test_click_by_snapshot_ref_reads_the_same_element(monkeypatch) -> None:
    selector = '[data-heliox-ref="e3"]'
    page = _Page(elements={selector: _checkbox("true", kind="switch")})
    page.effect = lambda p: p.elements[selector].update(on="false")
    _open(monkeypatch, page)
    output = await browser.browser_click(selector)
    action = _action(ActionType.BROWSER_CLICK, element_ref="e3", snapshot_id="bx12-abcdef")

    assert (await _check(action, output)) == browser_ui.verified("The clicked element changed: checked")


@pytest.mark.asyncio
async def test_click_text_reads_the_text_target_and_its_navigation(monkeypatch) -> None:
    page = _Page(elements={"text=Place order": _button()}, effect=_navigate)
    _open(monkeypatch, page)
    output = await browser.browser_click_text("Place order")

    outcome = await _check(_action(ActionType.BROWSER_CLICK_TEXT, text="Place order"), output)

    assert page.actions == ["click:text=Place order"]
    assert outcome.passed and outcome.observed and "navigation" in outcome.detail


@pytest.mark.asyncio
async def test_click_after_the_browser_closed_is_not_verified(monkeypatch) -> None:
    page = _Page(elements={"#pay": _button()})
    _open(monkeypatch, page)
    output = await browser.browser_click("#pay")
    monkeypatch.setattr(browser, "_browser_context", None)

    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#pay"), output)

    assert outcome.observed is False and "No browser tab is open" in outcome.detail


# ── Tabs ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_tab_on_the_requested_site_is_verified(monkeypatch) -> None:
    context = _open(monkeypatch, _Page())
    output = await browser.browser_new_tab("shop.example/cart")
    action = _action(ActionType.BROWSER_NEW_TAB, url="shop.example/cart")

    label, observed, _ = await _verify(action, output)

    assert len(context.pages) == 2
    assert label == "Action 0 (browser_new_tab): VERIFIED" and observed
    assert (await _check(action, output)).detail == "A new tab is open on shop.example"


@pytest.mark.asyncio
async def test_new_blank_tab_is_verified(monkeypatch) -> None:
    _open(monkeypatch, _Page())
    output = await browser.browser_new_tab()

    assert (await _check(_action(ActionType.BROWSER_NEW_TAB), output)) == browser_ui.verified("A new blank tab is open")


@pytest.mark.asyncio
async def test_new_tab_that_stayed_blank_or_never_opened_is_a_mismatch(monkeypatch) -> None:
    context = _open(monkeypatch, _Page())
    action = _action(ActionType.BROWSER_NEW_TAB, url="shop.example")

    not_opened = await _check(
        action, f"Opened new tab: https://shop.example\n{browser.PAGE_BEFORE_MARKER} " + '{"n":1}'
    )
    await context.new_page()
    blank = await _check(action, f"Opened new tab: https://shop.example\n{browser.PAGE_BEFORE_MARKER} " + '{"n":1}')

    assert not_opened == browser_ui.mismatch("No new tab is open: 1 tab(s), as before")
    assert blank == browser_ui.mismatch("The new tab is open but blank; shop.example did not load")


@pytest.mark.asyncio
async def test_switch_tab_makes_later_actions_use_that_tab(monkeypatch) -> None:
    first, second = _Page(), _Page(href=DONE)
    _open(monkeypatch, first, second)

    await browser.browser_switch_tab(0)

    assert first.actions == ["front"]
    assert await browser._get_page() is first  # before, the newest tab was used whatever was switched to


@pytest.mark.asyncio
async def test_switch_to_a_tab_that_does_not_exist_fails(monkeypatch) -> None:
    _open(monkeypatch, _Page(), _Page())

    with pytest.raises(LookupError, match="Tab 5 does not exist: 2 tab"):
        await browser.browser_switch_tab(5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("visibility", "expected"),
    [
        (("visible", "hidden"), browser_ui.verified("Tab 0 is the visible tab; the other 1 are hidden")),
        (
            ("hidden", "visible"),
            browser_ui.mismatch("Tab 0 was brought to the front but its page reports itself hidden"),
        ),
        (
            ("visible", "visible"),
            browser_ui.unobserved(
                "Every tab reports itself visible (a browser without a window does), "
                "so which one is in front cannot be read"
            ),
        ),
    ],
)
async def test_switch_tab_reads_which_tab_is_visible(monkeypatch, visibility, expected) -> None:
    _open(monkeypatch, _Page(visibility=visibility[0]), _Page(visibility=visibility[1]))
    output = await browser.browser_switch_tab(0)

    assert (await _check(_action(ActionType.BROWSER_SWITCH_TAB, tab_index=0), output)) == expected


@pytest.mark.asyncio
async def test_closed_tab_is_verified_by_the_tab_count(monkeypatch) -> None:
    first, second = _Page(), _Page(href=DONE)
    context = _open(monkeypatch, first, second)
    browser._active_page = first
    output = await browser.browser_close_tab()
    action = _action(ActionType.BROWSER_CLOSE_TAB)

    label, observed, _ = await _verify(action, output)

    assert context.pages == [second]  # the tab Heliox was acting on, not simply the newest
    assert browser._active_page is None
    assert label == "Action 0 (browser_close_tab): VERIFIED" and observed
    assert SECRET not in label


@pytest.mark.asyncio
async def test_tab_count_unchanged_after_a_close_is_a_mismatch(monkeypatch) -> None:
    _open(monkeypatch, _Page(), _Page())

    outcome = await _check(
        _action(ActionType.BROWSER_CLOSE_TAB), f"Closed tab: x\n{browser.PAGE_BEFORE_MARKER} " + '{"n":2}'
    )

    assert outcome == browser_ui.mismatch("2 tab(s) are still open, as before the close; no tab was closed")


@pytest.mark.asyncio
async def test_closing_a_tab_index_that_does_not_exist_fails_and_closes_nothing(monkeypatch) -> None:
    context = _open(monkeypatch, _Page(), _Page())

    with pytest.raises(LookupError, match="Tab 4 does not exist"):
        await browser.browser_close_tab(4)
    assert len(context.pages) == 2


# ── Closing the browser ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_close_is_verified_when_its_processes_exited(monkeypatch) -> None:
    _open(monkeypatch, _Page())
    monkeypatch.setattr(browser, "browser_process_ids", lambda: [[4242, 1.5], [4243, 1.6]])
    output = await browser.browser_close()
    monkeypatch.setattr(browser_ui, "_still_running", lambda psutil, pid, started: False)

    label, observed, _ = await _verify(_action(ActionType.BROWSER_CLOSE), output)

    assert browser._browser_context is None
    assert label == "Action 0 (browser_close): VERIFIED" and observed


@pytest.mark.asyncio
async def test_browser_still_running_after_close_is_a_mismatch(monkeypatch, _no_real_browser) -> None:
    output = f"Browser closed\n{browser.PAGE_BEFORE_MARKER} " + '{"pids":[[4242,1.5],[4243,1.6]]}'
    monkeypatch.setattr(browser_ui, "_still_running", lambda psutil, pid, started: pid == 4243)

    outcome = await _check(_action(ActionType.BROWSER_CLOSE), output)

    assert outcome == browser_ui.mismatch("1 of 2 browser process(es) were still running 2 s after the close")
    assert len(_no_real_browser) == browser_ui.POLL_ATTEMPTS - 1


@pytest.mark.asyncio
async def test_closing_when_no_browser_was_open_says_so_and_is_not_verified() -> None:
    output = await browser.browser_close()

    assert str(output) == "No browser was open, so nothing was closed" and output.nothing_done is True
    outcome = await _check(_action(ActionType.BROWSER_CLOSE), str(output))
    assert outcome.observed is False


def test_process_liveness_reads_the_process_table() -> None:
    import os

    import psutil

    started = psutil.Process(os.getpid()).create_time()
    assert browser_ui._still_running(psutil, os.getpid(), started) is True
    assert browser_ui._still_running(psutil, os.getpid(), started - 100) is False  # an id reused by another process


# ── History ──────────────────────────────────────────────────────────


def _step(page: _Page, delta: int) -> None:
    ids, index = page.history  # type: ignore[misc]
    page.history = (ids, index + delta)
    page.state["origin"] += 1


@pytest.mark.asyncio
async def test_back_to_the_previous_entry_is_verified(monkeypatch) -> None:
    page = _Page(history=([11, 12, 13], 2), effect=lambda p: _step(p, -1))
    _open(monkeypatch, page)
    output = await browser.browser_back()
    action = _action(ActionType.BROWSER_BACK)

    label, observed, _ = await _verify(action, output)

    assert label == "Action 0 (browser_back): VERIFIED" and observed
    assert (await _check(action, output)).detail == "The tab went back one page in its history, to shop.example"
    assert page.sessions_closed >= 2  # every DevTools session it opened was closed again


@pytest.mark.asyncio
async def test_back_that_stayed_on_the_same_entry_is_a_mismatch(monkeypatch) -> None:
    page = _Page(history=([11, 12], 1))
    _open(monkeypatch, page)
    output = await browser.browser_back()

    outcome = await _check(_action(ActionType.BROWSER_BACK), output)

    assert outcome == browser_ui.mismatch("The tab is still on the same page of its history; it did not go back")


@pytest.mark.asyncio
async def test_back_with_no_earlier_page_fails_instead_of_saying_it_went_back(monkeypatch) -> None:
    page = _Page(history=([11], 0))
    _open(monkeypatch, page)

    with pytest.raises(LookupError, match="no earlier page"):
        await browser.browser_back()
    assert page.actions == []


@pytest.mark.asyncio
async def test_forward_is_verified_and_history_unreadable_is_not(monkeypatch) -> None:
    page = _Page(history=([11, 12], 0), effect=lambda p: _step(p, 1))
    _open(monkeypatch, page)
    forward = await browser.browser_forward()
    assert (await _check(_action(ActionType.BROWSER_FORWARD), forward)).observed is True

    other = _Page(history=None)
    _open(monkeypatch, other)
    back = await browser.browser_back()  # without DevTools history the handler still acts
    outcome = await _check(_action(ActionType.BROWSER_BACK), back)
    assert other.actions == ["back"]
    assert outcome.observed is False and "was not recorded" in outcome.detail


@pytest.mark.asyncio
async def test_refresh_is_verified_by_a_new_document_and_a_mismatch_without_one(monkeypatch) -> None:
    page = _Page(history=([11], 0), effect=lambda p: p.state.update(origin=5000.0))
    _open(monkeypatch, page)
    reloaded = await browser.browser_refresh()
    assert (await _check(_action(ActionType.BROWSER_REFRESH), reloaded)) == browser_ui.verified(
        "The page was loaded again: a new document at the same place"
    )

    stale = _Page(history=([11], 0))
    _open(monkeypatch, stale)
    not_reloaded = await browser.browser_refresh()
    label, observed, _ = await _verify(_action(ActionType.BROWSER_REFRESH), not_reloaded)
    assert label == (
        "Action 0 (browser_refresh): MISMATCH — The page was not reloaded: it is still the document "
        "that was there before the refresh"
    )
    assert observed is True


# ── Hover and scroll ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "label", "observed"),
    [
        (_button(hovered=True), "VERIFIED", True),
        (
            _button(in_view=False),
            "MISMATCH — The element is outside the visible part of the page, so the pointer is not over it",
            True,
        ),
        (
            _button(),
            "REPORTED — no independent postcondition (The element is visible but not under the pointer now; "
            "the pointer may have moved since)",
            False,
        ),
    ],
)
async def test_hover_reads_the_pointer_state(monkeypatch, state, label, observed) -> None:
    _open(monkeypatch, _Page(elements={"#menu": state}))

    got, was_observed, _ = await _verify(_action(ActionType.BROWSER_HOVER, selector="#menu"), "Hovering over: #menu")

    assert got == f"Action 0 (browser_hover): {label}" and was_observed is observed


def _scroll_to(y: float) -> Callable[[_Page], None]:
    return lambda page: page.state.update(y=y)


@pytest.mark.asyncio
async def test_scroll_down_that_moved_is_verified(monkeypatch) -> None:
    page = _Page(y=100, effect=_scroll_to(600))
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down", 500)

    outcome = await _check(_action(ActionType.BROWSER_SCROLL, direction="down"), output)

    assert outcome == browser_ui.verified("The page moved down by 500 px")


@pytest.mark.asyncio
async def test_smooth_scroll_is_waited_for_by_counted_reads(monkeypatch, _no_real_browser) -> None:
    page = _Page(y=0)
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down", 500)
    reads_before = page.reads

    original = page.evaluate

    async def arriving(script: str, *args: Any) -> Any:
        if script is browser._PAGE_STATE_SCRIPT and page.reads - reads_before >= 2:
            page.state["y"] = 500
        return await original(script, *args)

    page.evaluate = arriving  # type: ignore[method-assign]
    outcome = await _check(_action(ActionType.BROWSER_SCROLL, direction="down"), output)

    assert outcome.passed and outcome.observed
    assert len(_no_real_browser) == 2


@pytest.mark.asyncio
async def test_scroll_that_did_not_move_is_reported_and_never_fails_the_run(monkeypatch) -> None:
    _open(monkeypatch, _Page(y=300))
    output = await browser.browser_scroll("up", 200)

    label, observed, passed = await _verify(_action(ActionType.BROWSER_SCROLL, direction="up"), output)

    assert label == (
        "Action 0 (browser_scroll): REPORTED — no independent postcondition (The page did not move when scrolled up)"
    )
    assert observed is False and passed is True


@pytest.mark.asyncio
async def test_a_page_that_fits_the_window_is_not_a_failure(monkeypatch) -> None:
    page = _Page(y=0, max_y=0)
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down")

    assert output.notice == "The page fits in the window, so there was nothing to scroll"
    label, observed, passed = await _verify(_action(ActionType.BROWSER_SCROLL, direction="down"), output)
    assert (label, observed, passed) == ("Action 0 (browser_scroll): VERIFIED", True, True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "y", "notice"),
    [
        ("down", 2000, "The page is already at the bottom, so it did not move"),
        ("bottom", 2000, "The page is already at the bottom, so it did not move"),
        ("up", 0, "The page is already at the top, so it did not move"),
        ("top", 0, "The page is already at the top, so it did not move"),
    ],
)
async def test_a_page_already_at_that_end_is_verified_with_a_notice(monkeypatch, direction, y, notice) -> None:
    _open(monkeypatch, _Page(y=y, max_y=2000))
    output = await browser.browser_scroll(direction)

    assert output.notice == notice
    outcome = await _check(_action(ActionType.BROWSER_SCROLL, direction=direction), output)
    assert outcome == browser_ui.verified(notice)


def _panel(top: int, room: int = 3000) -> dict[str, Any]:
    return {"key": "div#messages@600", "top": top, "max": room}


@pytest.mark.asyncio
async def test_a_page_that_scrolls_inside_a_panel_is_scrolled_there_and_read_there(monkeypatch) -> None:
    page = _Page(y=0, max_y=0)
    page.state["panel"] = _panel(0)
    page.effect = lambda p: p.state.__setitem__("panel", _panel(p.state["panel"]["top"] + 500))
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down", 500)

    assert page.actions == ["move:640,360", "wheel:500"]  # a wheel over the middle, not window.scrollBy
    label, observed, passed = await _verify(_action(ActionType.BROWSER_SCROLL, direction="down"), output)
    assert (label, observed, passed) == ("Action 0 (browser_scroll): VERIFIED", True, True)
    outcome = await _check(_action(ActionType.BROWSER_SCROLL, direction="down"), output)
    assert outcome.detail == "The scrolling area moved down by 500 px"


@pytest.mark.asyncio
async def test_a_panel_that_did_not_move_is_reported_not_failed(monkeypatch) -> None:
    page = _Page(y=0, max_y=0)
    page.state["panel"] = _panel(100)
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down")

    label, observed, passed = await _verify(_action(ActionType.BROWSER_SCROLL, direction="down"), output)

    assert passed is True and observed is False and "did not move" in label


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jump",
    [
        lambda p: p.state.update(y=p.state["y"] - 120, max_y=p.state["max_y"] + 4000),  # a feed loads and re-anchors
        lambda p: p.state.update(max_y=p.state["max_y"] + 4000),  # it grew, the position held
        lambda p: p.state.__setitem__("panel", {"key": "div#feed@700", "top": 0, "max": 900}),  # the list was replaced
    ],
)
async def test_a_page_that_grows_or_jumps_is_not_called_unmoved(monkeypatch, jump) -> None:
    page = _Page(y=400, max_y=3000, effect=jump)
    _open(monkeypatch, page)
    output = await browser.browser_scroll("down")

    outcome = await _check(_action(ActionType.BROWSER_SCROLL, direction="down"), output)

    assert outcome.passed is True and outcome.observed is False
    assert "moved or grew" in outcome.detail


@pytest.mark.asyncio
async def test_unknown_scroll_direction_fails_instead_of_doing_nothing(monkeypatch) -> None:
    page = _Page()
    _open(monkeypatch, page)

    with pytest.raises(ValueError, match="Unknown scroll direction 'sideways'"):
        await browser.browser_scroll("sideways")
    assert page.actions == []


# ── Tabs a click opens, and machine lines kept from people ───────────


@pytest.mark.asyncio
async def test_a_tab_the_click_opened_becomes_the_tab_in_use_and_is_its_consequence(monkeypatch) -> None:
    first = _Page(elements={"#help": _button()})
    context = _open(monkeypatch, first)
    browser._active_page = first  # an earlier explicit switch: before this, a pop-up was no longer followed

    def open_popup(page: _Page) -> None:
        popup = _Page(href=DONE)
        popup.context_pages = context.pages
        context.pages.append(popup)

    first.effect = open_popup
    output = await browser.browser_click("#help")

    assert output.notice == "The click opened a new tab; Heliox is now working in it"
    assert await browser._get_page() is context.pages[1]
    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#help"), str(output))
    assert outcome == browser_ui.verified("The click opened a new tab at shop.example")


@pytest.mark.asyncio
async def test_a_tab_a_page_opens_on_its_own_does_not_take_over(monkeypatch) -> None:
    first = _Page()
    context = _open(monkeypatch, first)
    assert await browser._get_page() is first

    advert = _Page(href=DONE)  # opened by the page's timer, while nothing was clicked
    advert.context_pages = context.pages
    context.pages.append(advert)

    assert await browser._get_page() is first
    output = await browser.browser_click("#anything")
    assert getattr(output, "notice", None) is None
    assert first.actions == ["click:#anything"] and advert.actions == []


@pytest.mark.asyncio
async def test_no_machine_line_reaches_the_reply_and_the_check_still_reads_it(monkeypatch) -> None:
    from pilot.actions import split_machine_lines
    from pilot.agents.verifier import Verifier
    from pilot.response_contract import success_message

    first, second = _Page(elements={"#pay": _button()}), _Page()
    # The click navigates; the scroll moves the page down 500 px.
    first.effect = lambda page: page.state.update(href=DONE, origin=2000.0, y=page.state["y"] + 500)
    _open(monkeypatch, first, second)
    browser._active_page = first
    steps = [
        (_action(ActionType.BROWSER_CLICK, selector="#pay"), lambda: browser.browser_click("#pay")),
        (_action(ActionType.BROWSER_SCROLL, direction="down"), lambda: browser.browser_scroll("down")),
        (_action(ActionType.BROWSER_SWITCH_TAB, tab_index=1), lambda: browser.browser_switch_tab(1)),
    ]

    results = []
    for action, run in steps:  # what the executor does with each step: run it, split its output, read the page
        output = await run()
        if action.action_type == ActionType.BROWSER_CLICK:
            assert browser.PAGE_BEFORE_MARKER in output and "[DOM_DIFF]" in output
        words, machine = split_machine_lines(action.action_type, output)
        result = ActionResult(action=action, success=True, output=words, observation=machine)
        results.append(await _read_right_after(result))
    plan = ActionPlan(actions=[action for action, _ in steps], explanation="Pay, scroll, switch.")
    verification = await Verifier(MagicMock()).verify(plan, results)
    reply = success_message(plan, results, verification, dry_run=False)

    for text in [reply, *(result.output for result in results), *(str(result.model_dump()) for result in results)]:
        assert "[PAGE_BEFORE]" not in text and "[DOM_DIFF]" not in text and "change_score" not in text
    assert verification.details[0] == "Action 0 (browser_click): VERIFIED"  # the check read the record
    assert "Clicked: #pay" in reply


def test_machine_lines_are_split_only_for_browser_handlers_and_keep_a_notice() -> None:
    from pilot.actions import OutputWithNotice, split_machine_lines

    text = "line one\n[DOM_DIFF] {}\n[PAGE_BEFORE] {}"
    assert split_machine_lines(ActionType.FILE_READ, text) == (text, None)
    words, machine = split_machine_lines(ActionType.BROWSER_CLICK, OutputWithNotice(text, "A new tab opened"))
    assert (str(words), words.notice, machine) == ("line one", "A new tab opened", "[DOM_DIFF] {}\n[PAGE_BEFORE] {}")
    assert split_machine_lines(ActionType.BROWSER_CLICK, "Clicked: #a") == ("Clicked: #a", None)


# ── A later step must not make an earlier browser step fail ─────────


@pytest.mark.asyncio
async def test_scroll_then_open_an_article_keeps_the_scroll_verified(monkeypatch) -> None:
    from pilot.response_contract import success_message

    page = _Page(y=0, elements={"#second": _button()})
    _open(monkeypatch, page)
    scroll = _action(ActionType.BROWSER_SCROLL, direction="down")
    click = _action(ActionType.BROWSER_CLICK, selector="#second")

    page.effect = _scroll_to(800)
    scrolled = await _read_right_after(ActionResult(action=scroll, success=True, output=await browser.browser_scroll()))
    page.effect = lambda p: p.state.update(href=DONE, origin=2000.0, y=0)  # the article opens at the top
    clicked = await _read_right_after(
        ActionResult(action=click, success=True, output=await browser.browser_click("#second"))
    )
    plan = ActionPlan(actions=[scroll, click], explanation="Scroll down and open the second article.")
    verification = await Verifier(MagicMock()).verify(plan, [scrolled, clicked])

    assert verification.passed
    assert verification.details == [
        "Action 0 (browser_scroll): VERIFIED",
        "Action 1 (browser_click): VERIFIED",
    ]
    assert "did not move" not in success_message(plan, [scrolled, clicked], verification, dry_run=False)


@pytest.mark.asyncio
async def test_a_check_run_after_the_plan_on_a_page_that_moved_on_is_not_verified_never_a_mismatch(
    monkeypatch,
) -> None:
    page = _Page(y=0, elements={"#menu": _button(hovered=True), "#second": _button()})
    _open(monkeypatch, page)
    scroll_output = await browser.browser_scroll()  # the page did not move yet: a reading now would say so
    hover_output = await browser.browser_hover("#menu")
    click_output = await browser.browser_click("#second")
    page.state.update(href=DONE, origin=2000.0, y=0)  # a later step opened another page

    for action, output in (
        (_action(ActionType.BROWSER_SCROLL, direction="down"), scroll_output),
        (_action(ActionType.BROWSER_HOVER, selector="#menu"), hover_output),
        (_action(ActionType.BROWSER_CLICK, selector="#second"), click_output),
    ):
        outcome = await _check_later(action, output)
        assert outcome.passed and not outcome.observed, (action.action_type, outcome)
        assert "later step" in outcome.detail


@pytest.mark.asyncio
async def test_the_reading_right_after_the_step_is_what_the_verifier_reports(monkeypatch) -> None:
    page = _Page(y=0)
    _open(monkeypatch, page)
    action = _action(ActionType.BROWSER_SCROLL, direction="down")
    result = await _read_right_after(ActionResult(action=action, success=True, output=await browser.browser_scroll()))
    page.state.update(y=900)  # later the page moves by itself; the step's own reading stands

    label, observed, passed = await _verify_stored(result)

    assert label.endswith("(The page did not move when scrolled down)")
    assert observed is False and passed is True


async def _verify_stored(result: ActionResult) -> tuple[str, bool, bool]:
    verification = await Verifier(MagicMock()).verify(ActionPlan(actions=[result.action]), [result])
    return verification.details[0], 0 in verification.independently_observed_actions, verification.passed


def test_a_browser_mismatch_is_not_retried_once_a_later_browser_step_ran() -> None:
    from pilot.actions import VerificationResult
    from pilot.server import _postcondition_failure_requires_reconciliation

    scroll = ActionResult(action=_action(ActionType.BROWSER_SCROLL), success=True)
    click = ActionResult(action=_action(ActionType.BROWSER_CLICK, selector="#second"), success=True)
    failed_scroll = VerificationResult(passed=False, failed_actions=[0])

    assert _postcondition_failure_requires_reconciliation([scroll, click], failed_scroll) is True
    assert _postcondition_failure_requires_reconciliation([scroll], failed_scroll) is False  # same page: retry


@pytest.mark.asyncio
async def test_the_reading_right_after_a_step_is_bounded(monkeypatch) -> None:
    page = _Page()
    _open(monkeypatch, page)

    async def never(*_: Any, **__: Any) -> None:
        import asyncio

        await asyncio.Event().wait()

    page.wait_for_load_state = never  # type: ignore[method-assign]
    monkeypatch.setattr(browser_ui, "SETTLE_SECONDS", 0.01)
    output = await browser.browser_click("#anything")
    outcome = await _check(_action(ActionType.BROWSER_CLICK, selector="#anything"), output)

    assert outcome.observed is False  # no change seen; the settle wait gave up at its bound instead of hanging
