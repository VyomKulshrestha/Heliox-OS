"""Browser Automation — full web browser control via Playwright.

Navigate, click, type, extract data, execute JS, handle forms,
manage tabs, and scrape web content programmatically.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("pilot.system.browser")

# Global browser instance for session persistence
_browser_context: Any = None
_playwright_instance: Any = None


async def _ensure_browser():
    """Lazy-initialize the Playwright browser."""
    global _browser_context, _playwright_instance

    if _browser_context is not None:
        return _browser_context

    from playwright.async_api import async_playwright

    _playwright_instance = await async_playwright().start()
    browser = await _playwright_instance.chromium.launch(
        headless=False,  # Visible browser for user
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
    """Click an element by CSS selector.

    Examples: "#submit-btn", "button:has-text('Login')", "a[href='/about']"
    """
    page = await _get_page()
    await page.click(selector, button=button, click_count=click_count, timeout=timeout)
    return f"Clicked: {selector}"


async def browser_click_text(text: str, exact: bool = False) -> str:
    """Click an element by its visible text content."""
    page = await _get_page()
    if exact:
        await page.click(f"text='{text}'")
    else:
        await page.click(f"text={text}")
    return f"Clicked element with text: {text}"


async def browser_type(
    selector: str,
    text: str,
    clear_first: bool = True,
    press_enter: bool = False,
) -> str:
    """Type text into an input field."""
    page = await _get_page()
    if clear_first:
        await page.fill(selector, text)
    else:
        await page.type(selector, text)
    if press_enter:
        await page.press(selector, "Enter")
    return f"Typed into {selector}: {text[:80]}"


async def browser_select(selector: str, value: str) -> str:
    """Select an option from a dropdown."""
    page = await _get_page()
    await page.select_option(selector, value)
    return f"Selected '{value}' in {selector}"


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
    """Fill multiple form fields at once.

    fields: {"#email": "user@example.com", "#password": "secret", ...}
    """
    page = await _get_page()
    for selector, value in fields.items():
        await page.fill(selector, value)

    if submit_selector:
        await page.click(submit_selector)

    filled = ", ".join(f"{k}={v[:20]}..." for k, v in fields.items())
    return f"Filled form: {filled}" + (f" and submitted via {submit_selector}" if submit_selector else "")


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
