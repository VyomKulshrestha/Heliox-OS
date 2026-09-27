"""InfiniteWeb collection keeps public browser branches isolated and truthful."""

import numpy as np
import pytest
from playwright.sync_api import sync_playwright

from scripts import collect_infiniteweb_branches as collector


def test_site_names_and_virtual_assets_are_bounded() -> None:
    assert collector.parse_sites("site_one,site-two") == ("site_one", "site-two")
    assert collector.asset_path("https://infiniteweb.local/site_one/index.html?q=1") == "site_one/index.html"
    for value in ("", "site,site", "../private", "site/other"):
        with pytest.raises(ValueError):
            collector.parse_sites(value)
    for value in (
        "https://example.com/site/index.html",
        "file:///private",
        "https://infiniteweb.local/site/../private",
        "https://infiniteweb.local/site/%2e%2e/private",
    ):
        with pytest.raises(ValueError):
            collector.asset_path(value)


def test_collect_page_requires_two_distinct_outcomes_from_exact_resets() -> None:
    html = b"""<!doctype html><html><head><style>
        body { background: white; margin: 0; }
        button { width: 160px; height: 60px; margin: 10px; }
        #result { height: 500px; }
    </style></head><body>
        <button onclick="document.querySelector('#result').textContent='Alpha';
            document.body.style.background='red'">A</button>
        <button onclick="document.querySelector('#result').textContent='Beta';
            document.body.style.background='blue'">B</button>
        <div id="result">Initial</div>
    </body></html>"""
    assets = collector.PinnedSiteAssets()
    assets._manifest = {"site/index.html": len(html)}
    assets._cache = {"site/index.html": html}
    with sync_playwright() as playwright, playwright.chromium.launch(headless=True) as browser:
        pair, attempts = collector.collect_page(browser, assets, "site/index.html", 2)
    assert attempts == 2
    assert len(pair) == 2
    assert np.array_equal(pair[0]["state"], pair[1]["state"])
    assert not np.array_equal(pair[0]["target"], pair[1]["target"])
    assert pair[0]["semantic"] != pair[1]["semantic"]
    assert {row["action_label"] for row in pair} == {"A", "B"}
    assert all(len(row["state_text"]) <= 2048 and len(row["target_text"]) <= 2048 for row in pair)


def test_focus_only_changes_are_not_collected_as_semantic_outcomes() -> None:
    html = b"""<!doctype html><html><body>
        <button style="width:160px;height:60px">A</button>
        <button style="width:160px;height:60px">B</button>
    </body></html>"""
    assets = collector.PinnedSiteAssets()
    assets._manifest = {"site/index.html": len(html)}
    assets._cache = {"site/index.html": html}
    with sync_playwright() as playwright, playwright.chromium.launch(headless=True) as browser:
        pair, attempts = collector.collect_page(browser, assets, "site/index.html", 2)
    assert attempts == 2
    assert pair == []
