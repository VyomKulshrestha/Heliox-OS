"""A quiet DOM must not cause an already successful browser action to repeat."""

from __future__ import annotations

from typing import Any

import pytest

from pilot.system import browser, dom_diff


class _QuietPage:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str, Any]] = []

    async def click(self, selector: str, **kwargs: Any) -> None:
        self.actions.append(("click", selector, kwargs))

    async def fill(self, selector: str, value: str) -> None:
        self.actions.append(("fill", selector, value))

    async def type(self, selector: str, value: str) -> None:
        self.actions.append(("type", selector, value))

    async def press(self, selector: str, key: str) -> None:
        self.actions.append(("press", selector, key))

    async def select_option(self, selector: str, value: str) -> None:
        self.actions.append(("select", selector, value))

    async def wait_for_timeout(self, timeout: int) -> None:
        pass


class _NoChangeDiff:
    change_score = 0.0

    def summary(self) -> str:
        return "No DOM change"

    def to_dict(self) -> dict[str, float]:
        return {"change_score": self.change_score}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            lambda: browser.browser_click("button[data-label=\"it's done\"]"),
            [("click", "button[data-label=\"it's done\"]", {"button": "left", "click_count": 1, "timeout": 5000})],
        ),
        (
            lambda: browser.browser_click_text("It's done"),
            [("click", "text=It's done", {"timeout": 1000, "no_wait_after": True})],
        ),
        (
            lambda: browser.browser_type("#query", "search", press_enter=True),
            [("fill", "#query", "search"), ("press", "#query", "Enter")],
        ),
        (
            lambda: browser.browser_select("#choice", "a'b"),
            [("select", "#choice", "a'b")],
        ),
        (
            lambda: browser.browser_fill_form({"#email": "person@example.com"}, "#submit"),
            [("fill", "#email", "person@example.com"), ("click", "#submit", {})],
        ),
    ],
)
async def test_successful_action_is_not_repeated_when_dom_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    action: Any,
    expected: list[tuple[str, str, Any]],
) -> None:
    page = _QuietPage()

    async def fake_get_page(tab_index: int = -1) -> _QuietPage:
        return page

    async def fake_snap(current_page: _QuietPage) -> object:
        assert current_page is page
        return object()

    monkeypatch.setattr(browser, "_get_page", fake_get_page)
    monkeypatch.setattr(browser, "_snap", fake_snap)
    monkeypatch.setattr(browser, "DOM_DIFF_ENABLED", True)
    monkeypatch.setattr(dom_diff, "diff_dom", lambda before, after: _NoChangeDiff())

    result = await action()

    assert page.actions == expected
    assert '[DOM_DIFF] {"change_score": 0.0}' in result


@pytest.mark.asyncio
async def test_browser_action_output_does_not_echo_typed_or_form_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = _QuietPage()

    async def fake_get_page(tab_index: int = -1) -> _QuietPage:
        return page

    monkeypatch.setattr(browser, "_get_page", fake_get_page)
    monkeypatch.setattr(browser, "DOM_DIFF_ENABLED", False)

    typed = await browser.browser_type("#password", "private-password")
    submitted = await browser.browser_fill_form({"#token": "private-token"}, "#submit")

    assert "private-password" not in typed
    assert "private-token" not in submitted
    assert typed == "Typed into #password"
    assert submitted == "Filled 1 form field(s) and submitted form"
