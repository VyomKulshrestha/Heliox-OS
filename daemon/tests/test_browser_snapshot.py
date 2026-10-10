"""Browser grounding: page snapshots with refs, stale refs, blockers and secret fields."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import ActionType, BrowserParams
from pilot.agents.executor import Executor
from pilot.agents.planner import Planner
from pilot.security.human_control import agent_initiated
from pilot.system import browser
from pilot.system.browser import (
    SensitiveFieldRefused,
    StaleBrowserRef,
    describe_browser_snapshot,
    resolve_browser_ref,
)

try:
    import playwright.async_api  # noqa: F401
except ImportError:  # optional extra: pip install -e ".[browser]" && playwright install chromium
    HAS_PLAYWRIGHT = False
else:
    HAS_PLAYWRIGHT = True

needs_browser = pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="needs the optional browser extra")

FIXTURE = """<!doctype html><html><head><title>Sign in</title></head><body>
<a href="/help">Help center</a>
<label for="user">Email</label><input id="user" name="email" value="ada@example.com">
<label for="pw">Password</label><input id="pw" type="password" value="hunter2">
<input id="card" name="cardnumber" placeholder="Card number">
<button id="go">Continue</button>
<button style="display:none">Hidden</button>
<div class="g-recaptcha"><button>I am not a robot</button></div>
</body></html>"""


def test_describe_lists_refs_flags_and_blockers():
    text = describe_browser_snapshot(
        "bx1-abcdef",
        {
            "title": "Sign in",
            "url": "https://example.test/login",
            "elements": [
                {"ref": "e1", "role": "button", "name": "Continue", "sensitive": False},
                {"ref": "e2", "role": "password field", "name": "Password", "sensitive": True, "filled": True},
            ],
            "blockers": ["captcha", "login_form"],
        },
    )

    assert 'e1 button "Continue"' in text
    assert "[sensitive: ask the person, filled]" in text
    assert "Blockers: captcha, login_form" in text
    assert "request_human_help" in text
    assert '"snapshot_id": "bx1-abcdef"' in text
    assert "untrusted content" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(("snapshot_id", "ref"), [("bx1-abcdef", "e1']"), ("x", "e1"), ("bx1-abcdef", "")])
async def test_refs_must_look_like_snapshot_output(snapshot_id, ref):
    with pytest.raises(StaleBrowserRef, match="must come from browser_snapshot"):
        await resolve_browser_ref(snapshot_id, ref)


def test_planner_maps_ref_for_browser_actions():
    params = Planner(MagicMock(), MagicMock())._parse_parameters(
        ActionType.BROWSER_CLICK, {"ref": "e4", "snapshot_id": "bx2-abcdef"}
    )

    assert isinstance(params, BrowserParams)
    assert params.element_ref == "e4" and params.snapshot_id == "bx2-abcdef"


@pytest.mark.asyncio
async def test_executor_resolves_refs_through_the_backend():
    backend = SimpleNamespace(
        resolve_ref=AsyncMock(return_value='[data-heliox-ref="e4"]'),
        click=AsyncMock(return_value="Clicked"),
    )
    executor = SimpleNamespace(_get_browser_backend=lambda: backend)
    executor._browser_target = lambda p: Executor._browser_target(executor, p)
    action = SimpleNamespace(parameters=BrowserParams(element_ref="e4", snapshot_id="bx2-abcdef"))

    await Executor._exec_browser_click(executor, action)

    backend.resolve_ref.assert_awaited_once_with("bx2-abcdef", "e4")
    backend.click.assert_awaited_once_with('[data-heliox-ref="e4"]', "left", timeout=10000)


@needs_browser
@pytest.mark.asyncio
async def test_real_page_snapshot_refs_stale_detection_and_secret_guard(monkeypatch):
    monkeypatch.setenv("HELIOX_BROWSER_HEADLESS", "1")
    try:
        page = await browser._get_page()
    except Exception as error:  # browser binaries missing on this host
        pytest.skip(f"Chromium is not available: {error}")
    try:
        await page.set_content(FIXTURE)
        text = await browser.browser_snapshot()
        snapshot_id = text.split()[2]

        assert 'link "Help center"' in text
        assert '"Continue"' in text
        assert "hunter2" not in text and "ada@example.com" not in text  # values never leave the page
        assert "Hidden" not in text and "robot" not in text  # hidden and CAPTCHA elements get no ref
        assert "captcha" in text and "login_form" in text
        password_ref = next(line.split()[0] for line in text.splitlines() if '"Password"' in line)
        assert "sensitive" in next(line for line in text.splitlines() if line.startswith(password_ref + " "))

        selector = await resolve_browser_ref(snapshot_id, "e1")
        assert await page.locator(selector).inner_text() == "Help center"

        await browser.browser_type("#user", "grace@example.com")  # the person's own command
        with agent_initiated():
            with pytest.raises(SensitiveFieldRefused):
                await browser.browser_type("#pw", "secret")
            with pytest.raises(SensitiveFieldRefused):
                await browser.browser_fill_form({"#card": "4111111111111111"})
            await browser.browser_type("#user", "lin@example.com")

        await page.set_content(FIXTURE)  # a reload discards the snapshot
        with pytest.raises(StaleBrowserRef, match="page changed"):
            await resolve_browser_ref(snapshot_id, "e1")
    finally:
        await browser.browser_close()
        os.environ.pop("HELIOX_BROWSER_HEADLESS", None)
