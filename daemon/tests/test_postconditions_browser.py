"""Browser navigation and field checks read the tab the action addressed, and judge redirects by the chain."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, BrowserParams
from pilot.agents.postconditions.browser import same_page
from pilot.agents.verifier import Verifier
from pilot.system import browser


class _Locator:
    def __init__(self, value: str) -> None:
        self._value = value

    async def count(self) -> int:
        return 1

    async def is_visible(self) -> bool:
        return True

    async def input_value(self) -> str:
        return self._value


class _Page:
    def __init__(self, url: str, values: dict[str, str] | None = None) -> None:
        self.url = url
        self._values = values or {}

    def locator(self, selector: str) -> _Locator:
        return _Locator(self._values.get(selector, ""))


def _tabs(monkeypatch: pytest.MonkeyPatch, *pages: _Page) -> list[int]:
    asked: list[int] = []

    async def _get_page(tab_index: int = -1):
        asked.append(tab_index)
        return pages[-1] if tab_index < 0 else pages[min(tab_index, len(pages) - 1)]

    monkeypatch.setattr(browser, "_browser_context", object())
    monkeypatch.setattr(browser, "_get_page", _get_page)
    return asked


async def _verify(action: Action, output: str = ""):
    return await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True, output=output)]
    )


def _navigate(url: str, tab_index: int = -1) -> Action:
    return Action(
        action_type=ActionType.BROWSER_NAVIGATE, target=url, parameters=BrowserParams(url=url, tab_index=tab_index)
    )


@pytest.mark.parametrize(
    ("requested", "found", "same"),
    [
        ("example.com", "https://www.example.com/", True),
        ("http://example.com/docs", "https://example.com/docs/", True),
        ("https://example.com/search?q=heliox", "https://example.com/search?q=heliox&lang=en", True),
        ("https://example.com/account", "https://example.com/login?next=/account", False),
        ("https://example.com/account", "https://accounts.example.net/account", False),
        ("https://example.com/search?q=heliox", "https://example.com/search", False),
        ("https://example.com/a", "http://example.com/a", False),  # a downgrade is not the same page
    ],
)
def test_what_counts_as_the_same_page(requested: str, found: str, same: bool) -> None:
    assert same_page(requested, found) is same


@pytest.mark.asyncio
async def test_a_redirect_to_the_canonical_url_verifies_by_the_recorded_chain(monkeypatch) -> None:
    _tabs(monkeypatch, _Page("https://www.example.com/"))
    output = "Navigated to: https://example.com\nRedirect chain: https://example.com/ -> https://www.example.com/"

    verification = await _verify(_navigate("example.com"), output)

    assert verification.passed is True
    assert verification.independently_observed_actions == [0]


@pytest.mark.asyncio
async def test_a_login_wall_is_reported_not_verified_and_never_a_mismatch(monkeypatch) -> None:
    _tabs(monkeypatch, _Page("https://example.com/login?next=%2Faccount&token=hidden"))
    output = "Redirect chain: https://example.com/account -> https://example.com/login"

    verification = await _verify(_navigate("https://example.com/account"), output)

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert "hidden" not in str(verification)


@pytest.mark.asyncio
async def test_a_same_looking_url_without_the_browsers_chain_is_not_verified(monkeypatch) -> None:
    _tabs(monkeypatch, _Page("https://www.example.com/"))

    verification = await _verify(_navigate("https://example.com/"), "Navigated to: https://example.com/")

    assert verification.independently_observed_actions == []


@pytest.mark.asyncio
async def test_navigation_is_read_in_the_addressed_tab(monkeypatch) -> None:
    asked = _tabs(monkeypatch, _Page("https://first.example/"), _Page("https://second.example/"))

    first = await _verify(_navigate("https://first.example/", tab_index=0))
    wrong = await _verify(_navigate("https://second.example/", tab_index=0))

    assert first.independently_observed_actions == [0]
    assert wrong.independently_observed_actions == []
    assert set(asked) == {0}


@pytest.mark.asyncio
async def test_a_field_is_read_in_the_addressed_tab(monkeypatch) -> None:
    asked = _tabs(monkeypatch, _Page("https://a.example/", {"#name": "Ada"}), _Page("https://b.example/"))
    typed = Action(
        action_type=ActionType.BROWSER_TYPE,
        target="#name",
        parameters=BrowserParams(selector="#name", text="Ada", tab_index=0),
    )
    other = Action(
        action_type=ActionType.BROWSER_TYPE,
        target="#name",
        parameters=BrowserParams(selector="#name", text="Grace", tab_index=0),
    )

    verified = await _verify(typed)
    differs = await _verify(other)

    assert verified.independently_observed_actions == [0]
    assert differs.passed is False and differs.independently_observed_actions == [0]
    assert set(asked) == {0}


@pytest.mark.asyncio
async def test_no_browser_session_is_reported(monkeypatch) -> None:
    monkeypatch.setattr(browser, "_browser_context", None)

    verification = await _verify(_navigate("https://example.com/"))

    assert verification.passed is True
    assert verification.independently_observed_actions == []


@pytest.mark.asyncio
async def test_handlers_act_in_the_addressed_tab(monkeypatch) -> None:
    class _Response:
        status = 200

        class request:  # noqa: N801 - the Playwright attribute name
            url = "https://www.example.com/"
            redirected_from = type("R", (), {"url": "https://example.com/", "redirected_from": None})()

    class _ActingPage(_Page):
        def __init__(self, url: str) -> None:
            super().__init__(url)
            self.actions: list[str] = []

        async def goto(self, url: str, **_kwargs: object) -> _Response:
            self.actions.append(f"goto {url}")
            self.url = "https://www.example.com/"
            return _Response()

        async def title(self) -> str:
            return "Example"

        async def fill(self, selector: str, value: str) -> None:
            self.actions.append(f"fill {selector}")

        async def select_option(self, selector: str, value: str) -> None:
            self.actions.append(f"select {selector}")

        async def wait_for_timeout(self, _ms: int) -> None:
            return None

        async def evaluate(self, *_args: object) -> bool:
            return False

    first, last = _ActingPage("about:blank"), _ActingPage("about:blank")
    _tabs(monkeypatch, first, last)
    monkeypatch.setattr(browser, "DOM_DIFF_ENABLED", False)
    backend = browser.PlaywrightBackend()

    output = await backend.navigate("example.com", tab_index=0)
    await backend.type("#q", "x", tab_index=0)
    await backend.select("#s", "y", tab_index=0)
    await backend.fill_form({"#a": "1"}, tab_index=0)

    assert last.actions == []
    assert first.actions == ["goto https://example.com", "fill #q", "select #s", "fill #a"]
    assert "Redirect chain: https://example.com/ -> https://www.example.com/" in output
