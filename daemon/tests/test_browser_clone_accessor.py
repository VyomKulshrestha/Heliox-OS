"""Tests for pilot.system.browser.get_real_page_for_clone() -- the accessor
dom_diff.dry_run_action() uses to clone the real page into a scratch tab.

Every other dry-run test mocks this accessor rather than exercising it
directly (matching this codebase's established browser.py-boundary mocking
pattern), so this file closes that gap with direct unit coverage of the
accessor itself.
"""

from __future__ import annotations

import pytest

import pilot.system.browser as browser


class TestGetRealPageForClone:
    @pytest.mark.asyncio
    async def test_no_active_session_returns_none(self, monkeypatch):
        monkeypatch.setattr(browser, "_browser_context", None)
        result = await browser.get_real_page_for_clone()
        assert result is None

    @pytest.mark.asyncio
    async def test_active_session_returns_the_real_page(self, monkeypatch):
        sentinel_page = object()

        async def _fake_get_page(tab_index: int = -1):
            return sentinel_page

        monkeypatch.setattr(browser, "_browser_context", object())  # any non-None value
        monkeypatch.setattr(browser, "_get_page", _fake_get_page)

        result = await browser.get_real_page_for_clone()
        assert result is sentinel_page
