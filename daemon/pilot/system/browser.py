"""Browser Automation — full web browser control via Playwright.

Navigate, click, type, extract data, execute JS, handle forms,
manage tabs, and scrape web content programmatically.

DOM-diff observation
--------------------
Browser actions can:
  1. Snapshot the DOM before the action
  2. Execute the action
  3. Snapshot the DOM after the action
  4. Return a DomDiff summary as evidence for later verification

A small or absent DOM change never authorizes replaying a successful action:
the effect may be external, delayed, or outside the page DOM.

Set ``DOM_DIFF_ENABLED = False`` to disable (e.g. for performance testing).
"""

from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from pilot.system.browser_backend import BrowserBackend

logger = logging.getLogger("pilot.system.browser")

# ---------------------------------------------------------------------------
# DOM-diff configuration
# ---------------------------------------------------------------------------

# Master switch — set to False to skip all DOM diffing
DOM_DIFF_ENABLED: bool = True

# Global browser instance for session persistence
_browser_context: Any = None
_playwright_instance: Any = None

_CLICK_INTENT_GROUPS = (
    frozenset({"learn", "read", "discover", "explore", "details", "detail", "information", "info", "more"}),
    frozenset({"continue", "next", "proceed", "advance"}),
    frozenset({"submit", "send", "confirm", "save", "apply"}),
    frozenset({"close", "cancel", "dismiss", "exit"}),
    frozenset({"login", "log", "signin", "sign"}),
    frozenset({"register", "signup", "join", "create"}),
)
_CLICK_STOP_WORDS = frozenset({"a", "an", "and", "button", "link", "please", "the", "to"})
_INTERACTIVE_SELECTOR = "a, button, [role='button'], [role='link'], input[type='button'], input[type='submit']"


async def _ensure_browser():
    """Lazy-initialize the Playwright browser."""
    global _browser_context, _playwright_instance

    if _browser_context is not None:
        return _browser_context

    from playwright.async_api import async_playwright

    _playwright_instance = await async_playwright().start()
    headless = os.getenv("HELIOX_BROWSER_HEADLESS", "").casefold() in {"1", "true", "yes"}
    browser = await _playwright_instance.chromium.launch(
        headless=headless,  # Visible by default; controlled CI smoke runs opt in to headless mode.
        args=["--disable-blink-features=AutomationControlled"],
    )
    _browser_context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
    )
    return _browser_context


async def _get_page(tab_index: int = -1):
    """Get the current (or specified) page/tab."""
    ctx = await _ensure_browser()
    pages = ctx.pages
    if not pages:
        page = await ctx.new_page()
        return page
    if tab_index < 0:
        return pages[-1]  # Latest tab
    return pages[min(tab_index, len(pages) - 1)]


# ---------------------------------------------------------------------------
# DOM-diff helpers (internal)
# ---------------------------------------------------------------------------


async def _snap(page: Any):
    """Take a DOM snapshot if DOM_DIFF_ENABLED, else return None."""
    if not DOM_DIFF_ENABLED:
        return None
    from pilot.system.dom_diff import snapshot_dom

    return await snapshot_dom(page)


def has_active_session() -> bool:
    """True if a browser context is already open.

    Used by the dry-run sandbox to decide whether it can peek at the
    current page for pre-execution target assessment — it must never
    launch a browser itself (a dry-run simulation has to stay a genuine
    no-op when nothing is already running).
    """
    return _browser_context is not None


async def peek_current_dom_snapshot() -> Any | None:
    """Snapshot the current page's DOM without side effects.

    Returns None if no browser session is open yet, or if DOM_DIFF_ENABLED
    is False — callers must treat None as "no assessment possible", not
    as "page is empty".
    """
    if not has_active_session():
        return None
    page = await _get_page()
    return await _snap(page)


async def get_real_page_for_clone() -> Any | None:
    """Return the real, live page — the ONLY sanctioned use is cloning it
    into a scratch tab for a dry-run (see `pilot.system.dom_diff.dry_run_action`),
    never for reading/mutating it directly. Returns None if no session is
    open yet, same "no assessment possible" contract as
    `peek_current_dom_snapshot()`.
    """
    if not has_active_session():
        return None
    return await _get_page()


def _canonical_browser_url(value: str) -> str:
    """Normalize only browser-safe URL syntax used for exact state checks."""

    candidate = value.strip()
    if not candidate.startswith(("http://", "https://", "file://")):
        candidate = "https://" + candidate
    parsed = urlsplit(candidate)
    scheme = parsed.scheme.casefold()
    netloc = parsed.netloc.casefold()
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    elif scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, parsed.fragment))


async def browser_action_already_satisfied(action_type: str, params: Any) -> tuple[bool, str] | None:
    """Read the live page to prove a conservative browser postcondition.

    ``None`` means the current state cannot prove anything. Clicks, submitted
    forms, append-style typing, and Enter-key actions are deliberately never
    skipped because equality of the visible input state does not prove their
    side effect has already happened.
    """

    if (
        action_type
        not in {
            "browser_navigate",
            "browser_type",
            "browser_select",
            "browser_fill_form",
        }
        or not has_active_session()
    ):
        return None

    try:
        page = await _get_page(int(getattr(params, "tab_index", -1)))
        if action_type == "browser_navigate":
            target = str(getattr(params, "url", "") or "")
            if not target:
                return None
            satisfied = _canonical_browser_url(page.url) == _canonical_browser_url(target)
            return satisfied, f"current URL is {page.url}"

        if action_type == "browser_type":
            if not bool(getattr(params, "clear_first", True)) or bool(getattr(params, "press_enter", False)):
                return None
            fields = {str(getattr(params, "selector", "") or ""): str(getattr(params, "text", "") or "")}
        elif action_type == "browser_select":
            fields = {str(getattr(params, "selector", "") or ""): str(getattr(params, "value", "") or "")}
        else:
            if getattr(params, "submit_selector", None):
                return None
            fields = {
                str(selector): str(value) for selector, value in dict(getattr(params, "fields", {}) or {}).items()
            }

        if not fields or any(not selector for selector in fields):
            return None
        for selector, expected in fields.items():
            locator = page.locator(selector)
            if await locator.count() != 1 or not await locator.is_visible():
                return None
            actual = await locator.input_value()
            if actual != expected:
                return False, f"{selector} has a different value"
        return True, f"all {len(fields)} browser field value(s) already match"
    except Exception:
        logger.debug("Browser step-state check failed for %s", action_type, exc_info=True)
        return None


def _append_diff(base_output: str, before: Any, after: Any, action_desc: str) -> str:
    """Compute diff and append a JSON summary to the action output string."""
    if before is None or after is None:
        return base_output
    from pilot.system.dom_diff import diff_dom

    diff = diff_dom(before, after)
    logger.debug("DOM diff [%s]: %s", action_desc, diff.summary())
    return base_output + f"\n[DOM_DIFF] {json.dumps(diff.to_dict())}"


# ── Navigation ───────────────────────────────────────────────────────


async def browser_navigate(url: str, wait_until: str = "domcontentloaded") -> str:
    """Navigate to a URL."""
    if not url.startswith(("http://", "https://", "file://")):
        url = "https://" + url
    page = await _get_page()
    resp = await page.goto(url, wait_until=wait_until, timeout=30000)
    status = resp.status if resp else "unknown"
    title = await page.title()
    return f"Navigated to: {url}\nTitle: {title}\nStatus: {status}"


async def browser_back() -> str:
    page = await _get_page()
    await page.go_back()
    return f"Navigated back to: {page.url}"


async def browser_forward() -> str:
    page = await _get_page()
    await page.go_forward()
    return f"Navigated forward to: {page.url}"


async def browser_refresh() -> str:
    page = await _get_page()
    await page.reload()
    return f"Refreshed: {page.url}"


# ── Interaction ──────────────────────────────────────────────────────


async def browser_click(
    selector: str,
    button: str = "left",
    click_count: int = 1,
    timeout: int = 5000,
) -> str:
    """Click an element by CSS selector once and record its observed DOM diff.

    Examples: "#submit-btn", "button:has-text('Login')", "a[href='/about']"
    """
    page = await _get_page()
    before = await _snap(page)

    await page.click(selector, button=button, click_count=click_count, timeout=timeout)
    await page.wait_for_timeout(300)
    after = await _snap(page)

    return _append_diff(f"Clicked: {selector}", before, after, f"click:{selector}")


async def browser_click_text(text: str, exact: bool = False) -> str:
    """Click visible text, reasoning over equivalent labels when literal lookup fails.

    The semantic fallback is deliberately bounded: it considers only visible
    interactive controls and proceeds only when one candidate is both
    high-confidence and clearly better than the runner-up.
    """
    page = await _get_page()
    before = await _snap(page)
    matched_text = text
    used_fallback = False

    try:
        if exact:
            await page.click(f"text='{text}'", timeout=1000, no_wait_after=True)
        else:
            await page.click(f"text={text}", timeout=1000, no_wait_after=True)
    except Exception as exc:
        if "timeout" not in str(exc).lower():
            raise
        candidate = await _find_semantic_click_candidate(page, text)
        if candidate is None:
            raise
        candidate_locator, matched_text = candidate
        await candidate_locator.click(timeout=10000, no_wait_after=True)
        used_fallback = True

    await page.wait_for_timeout(300)
    after = await _snap(page)

    output = (
        f"Clicked closest available option: {matched_text} (requested: {text})"
        if used_fallback
        else f"Clicked element with text: {text}"
    )
    return _append_diff(output, before, after, f"click_text:{matched_text}")


def _normalize_click_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _click_text_similarity(requested: str, candidate: str) -> float:
    """Score two UI labels by wording and bounded intent equivalence."""
    requested_norm = _normalize_click_label(requested)
    candidate_norm = _normalize_click_label(candidate)
    if not requested_norm or not candidate_norm:
        return 0.0
    if requested_norm == candidate_norm:
        return 1.0

    requested_tokens = set(requested_norm.split()) - _CLICK_STOP_WORDS
    candidate_tokens = set(candidate_norm.split()) - _CLICK_STOP_WORDS
    union = requested_tokens | candidate_tokens
    overlap = len(requested_tokens & candidate_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, requested_norm, candidate_norm).ratio()
    score = (0.55 * sequence) + (0.45 * overlap)

    for intent_group in _CLICK_INTENT_GROUPS:
        if requested_tokens & intent_group and candidate_tokens & intent_group:
            score = max(score, 0.74 + min(overlap, 0.2))

    if requested_norm in candidate_norm or candidate_norm in requested_norm:
        score = max(score, 0.78)
    return min(score, 1.0)


def _select_semantic_click_candidate(
    requested: str,
    candidates: list[str],
) -> tuple[int, str] | None:
    """Return one unambiguous, high-confidence semantic label match."""
    scored = sorted(
        (
            (_click_text_similarity(requested, label), index, label)
            for index, label in enumerate(candidates)
            if _normalize_click_label(label)
        ),
        reverse=True,
    )
    if not scored or scored[0][0] < 0.65:
        return None

    best_score, best_index, best_label = scored[0]
    if len(scored) > 1:
        second_score, _, second_label = scored[1]
        if second_label != best_label and best_score - second_score < 0.08:
            return None
    return best_index, best_label


async def _find_semantic_click_candidate(page: Any, requested: str) -> tuple[Any, str] | None:
    controls = page.locator(_INTERACTIVE_SELECTOR)
    count = min(await controls.count(), 60)
    visible_locators: list[Any] = []
    labels: list[str] = []

    for index in range(count):
        locator = controls.nth(index)
        if not await locator.is_visible():
            continue
        label = (await locator.inner_text()).strip()
        if not label:
            for attribute in ("aria-label", "title", "value"):
                label = str(await locator.get_attribute(attribute) or "").strip()
                if label:
                    break
        if label:
            visible_locators.append(locator)
            labels.append(label)

    selected = _select_semantic_click_candidate(requested, labels)
    if selected is None:
        available = ", ".join(dict.fromkeys(labels[:8])) or "none"
        raise RuntimeError(f"No unambiguous option matched '{requested}'. Visible options: {available}")
    index, label = selected
    logger.info("browser_click_text: mapped requested label %r to visible option %r", requested, label)
    return visible_locators[index], label


async def browser_type(
    selector: str,
    text: str,
    clear_first: bool = True,
    press_enter: bool = False,
) -> str:
    """Type once into an input field and record its observed DOM diff."""
    page = await _get_page()
    before = await _snap(page)

    if clear_first:
        await page.fill(selector, text)
    else:
        await page.type(selector, text)
    if press_enter:
        await page.press(selector, "Enter")
    await page.wait_for_timeout(200)
    after = await _snap(page)

    return _append_diff(f"Typed into {selector}", before, after, f"type:{selector}")


async def browser_select(selector: str, value: str) -> str:
    """Select an option once and record its observed DOM diff."""
    page = await _get_page()
    before = await _snap(page)

    await page.select_option(selector, value)
    await page.wait_for_timeout(200)
    after = await _snap(page)

    return _append_diff(f"Selected '{value}' in {selector}", before, after, f"select:{selector}")


async def browser_check(selector: str, checked: bool = True) -> str:
    """Check or uncheck a checkbox."""
    page = await _get_page()
    if checked:
        await page.check(selector)
    else:
        await page.uncheck(selector)
    return f"{'Checked' if checked else 'Unchecked'}: {selector}"


async def browser_hover(selector: str) -> str:
    """Hover over an element."""
    page = await _get_page()
    await page.hover(selector)
    return f"Hovering over: {selector}"


async def browser_press_key(key: str) -> str:
    """Press a keyboard key in the browser."""
    page = await _get_page()
    await page.keyboard.press(key)
    return f"Pressed key: {key}"


async def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    """Scroll the page."""
    page = await _get_page()
    if direction == "down":
        await page.evaluate(f"window.scrollBy(0, {amount})")
    elif direction == "up":
        await page.evaluate(f"window.scrollBy(0, -{amount})")
    elif direction == "top":
        await page.evaluate("window.scrollTo(0, 0)")
    elif direction == "bottom":
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    return f"Scrolled {direction} by {amount}px"


# ── Data Extraction ──────────────────────────────────────────────────


async def browser_extract(
    selector: str = "body",
    attribute: str = "innerText",
    multiple: bool = False,
) -> str:
    """Extract data from page elements.

    attribute: 'innerText', 'innerHTML', 'href', 'src', 'value', etc.
    """
    page = await _get_page()

    if multiple:
        elements = await page.query_selector_all(selector)
        values = []
        for el in elements[:50]:  # cap at 50
            val = await el.get_attribute(attribute) if attribute != "innerText" else await el.inner_text()
            if val and val.strip():
                values.append(val.strip())
        # For text extraction, return plain text joined by newlines (not JSON)
        if attribute == "innerText":
            return "\n\n".join(values)
        return json.dumps(values, indent=2)
    else:
        element = await page.query_selector(selector)
        if not element:
            return f"Element not found: {selector}"
        if attribute == "innerText":
            val = await element.inner_text()
        elif attribute == "innerHTML":
            val = await element.inner_html()
        else:
            val = await element.get_attribute(attribute)
        return val or "(empty)"


async def browser_extract_table(selector: str = "table") -> str:
    """Extract a table as JSON."""
    page = await _get_page()

    result = await page.evaluate(f"""
        (() => {{
            const table = document.querySelector('{selector}');
            if (!table) return null;
            const rows = [];
            const headers = [];
            table.querySelectorAll('th').forEach(th => headers.push(th.innerText.trim()));
            table.querySelectorAll('tr').forEach(tr => {{
                const cells = [];
                tr.querySelectorAll('td').forEach(td => cells.push(td.innerText.trim()));
                if (cells.length > 0) rows.push(cells);
            }});
            return {{ headers, rows: rows.slice(0, 100) }};
        }})()
    """)

    if not result:
        return f"Table not found: {selector}"
    return json.dumps(result, indent=2)


async def browser_extract_links() -> str:
    """Extract all links from the current page."""
    page = await _get_page()
    links = await page.evaluate("""
        (() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                links.push({ text: a.innerText.trim().substring(0, 100), href: a.href });
            });
            return links.slice(0, 100);
        })()
    """)
    return json.dumps(links, indent=2)


async def browser_get_page_info() -> str:
    """Get current page information (URL, title, metadata)."""
    page = await _get_page()
    info = {
        "url": page.url,
        "title": await page.title(),
    }
    # Get meta tags
    metas = await page.evaluate("""
        (() => {
            const metas = {};
            document.querySelectorAll('meta').forEach(m => {
                const name = m.getAttribute('name') || m.getAttribute('property') || '';
                const content = m.getAttribute('content') || '';
                if (name && content) metas[name] = content;
            });
            return metas;
        })()
    """)
    info["meta"] = metas
    return json.dumps(info, indent=2)


# ── JavaScript Execution ─────────────────────────────────────────────


async def browser_execute_js(script: str) -> str:
    """Execute arbitrary JavaScript in the browser and return the result."""
    page = await _get_page()
    result = await page.evaluate(script)
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2)
    return str(result) if result is not None else "(undefined)"


# ── Screenshots ──────────────────────────────────────────────────────


async def browser_screenshot(
    output_path: str | None = None,
    full_page: bool = False,
    selector: str | None = None,
) -> str:
    """Take a screenshot of the browser page."""
    page = await _get_page()

    if output_path is None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.expanduser(f"~/Pictures/browser_{ts}.png")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if selector:
        element = await page.query_selector(selector)
        if element:
            await element.screenshot(path=output_path)
        else:
            return f"Element not found for screenshot: {selector}"
    else:
        await page.screenshot(path=output_path, full_page=full_page)

    size = Path(output_path).stat().st_size
    return f"Browser screenshot saved to {output_path} ({size:,} bytes)"


# ── Tab Management ───────────────────────────────────────────────────


async def browser_new_tab(url: str | None = None) -> str:
    """Open a new browser tab."""
    ctx = await _ensure_browser()
    page = await ctx.new_page()
    if url:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        await page.goto(url)
    return f"Opened new tab: {url or 'about:blank'}"


async def browser_close_tab(tab_index: int = -1) -> str:
    """Close a browser tab."""
    ctx = await _ensure_browser()
    pages = ctx.pages
    if not pages:
        return "No tabs to close"
    page = pages[min(tab_index, len(pages) - 1)]
    url = page.url
    await page.close()
    return f"Closed tab: {url}"


async def browser_list_tabs() -> str:
    """List all open browser tabs."""
    ctx = await _ensure_browser()
    tabs = []
    for i, page in enumerate(ctx.pages):
        tabs.append(
            {
                "index": i,
                "url": page.url,
                "title": await page.title(),
            }
        )
    return json.dumps(tabs, indent=2)


async def browser_switch_tab(tab_index: int) -> str:
    """Switch to a specific tab."""
    ctx = await _ensure_browser()
    pages = ctx.pages
    if tab_index >= len(pages):
        return f"Tab {tab_index} doesn't exist (only {len(pages)} tabs open)"
    page = pages[tab_index]
    await page.bring_to_front()
    return f"Switched to tab {tab_index}: {page.url}"


# ── Wait / Sync ──────────────────────────────────────────────────────


async def browser_wait(
    selector: str | None = None,
    timeout: int = 10000,
    state: str = "visible",
) -> str:
    """Wait for an element or a timeout."""
    page = await _get_page()
    if selector:
        await page.wait_for_selector(selector, state=state, timeout=timeout)
        return f"Element ready: {selector} ({state})"
    else:
        await page.wait_for_timeout(timeout)
        return f"Waited {timeout}ms"


async def browser_wait_navigation(timeout: int = 30000) -> str:
    """Wait for navigation to complete."""
    page = await _get_page()
    await page.wait_for_load_state("domcontentloaded", timeout=timeout)
    return f"Navigation complete: {page.url}"


# ── Form Automation ──────────────────────────────────────────────────


async def browser_fill_form(fields: dict[str, str], submit_selector: str | None = None) -> str:
    """Fill each form field once and optionally submit once.

    fields: {"#email": "user@example.com", "#password": "secret", ...}
    """
    page = await _get_page()
    before_form = await _snap(page)

    for selector, value in fields.items():
        await page.fill(selector, value)
        await page.wait_for_timeout(150)

    if submit_selector:
        await page.click(submit_selector)
        await page.wait_for_timeout(300)

    after_form = await _snap(page)
    base = f"Filled {len(fields)} form field(s)" + (" and submitted form" if submit_selector else "")
    return _append_diff(base, before_form, after_form, "fill_form")


# ── Cleanup ──────────────────────────────────────────────────────────


async def browser_close() -> str:
    """Close the browser completely."""
    global _browser_context, _playwright_instance
    if _browser_context:
        await _browser_context.close()
        _browser_context = None
    if _playwright_instance:
        await _playwright_instance.stop()
        _playwright_instance = None
    return "Browser closed"


class PlaywrightBackend(BrowserBackend):
    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> str:
        return await browser_navigate(url, wait_until)

    async def click(self, selector: str, button: str = "left", timeout: int = 5000) -> str:
        return await browser_click(selector, button=button, timeout=timeout)

    async def click_text(self, text: str, exact: bool = False) -> str:
        return await browser_click_text(text, exact)

    async def type(self, selector: str, text: str, clear_first: bool = True, press_enter: bool = False) -> str:
        return await browser_type(selector, text, clear_first, press_enter)

    async def select(self, selector: str, value: str) -> str:
        return await browser_select(selector, value)

    async def hover(self, selector: str) -> str:
        return await browser_hover(selector)

    async def scroll(self, direction: str = "down", amount: int = 500) -> str:
        return await browser_scroll(direction, amount)

    async def extract(self, selector: str = "body", attribute: str = "innerText", multiple: bool = False) -> str:
        return await browser_extract(selector, attribute, multiple)

    async def extract_table(self, selector: str = "table") -> str:
        return await browser_extract_table(selector)

    async def extract_links(self) -> str:
        return await browser_extract_links()

    async def execute_js(self, script: str) -> str:
        return await browser_execute_js(script)

    async def screenshot(
        self, output_path: str | None = None, full_page: bool = False, selector: str | None = None
    ) -> str:
        return await browser_screenshot(output_path, full_page, selector)

    async def fill_form(self, fields: dict[str, str], submit_selector: str | None = None) -> str:
        return await browser_fill_form(fields, submit_selector)

    async def new_tab(self, url: str | None = None) -> str:
        return await browser_new_tab(url)

    async def close_tab(self, tab_index: int = -1) -> str:
        return await browser_close_tab(tab_index)

    async def list_tabs(self) -> str:
        return await browser_list_tabs()

    async def switch_tab(self, tab_index: int) -> str:
        return await browser_switch_tab(tab_index)

    async def back(self) -> str:
        return await browser_back()

    async def forward(self) -> str:
        return await browser_forward()

    async def refresh(self) -> str:
        return await browser_refresh()

    async def wait(self, selector: str | None = None, timeout: int = 10000, state: str = "visible") -> str:
        return await browser_wait(selector, timeout, state)

    async def close(self) -> str:
        return await browser_close()

    async def get_page_info(self) -> str:
        return await browser_get_page_info()
