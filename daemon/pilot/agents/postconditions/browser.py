"""Post-conditions for browser navigation and form fields, read from the live page afresh.

The page is read through the tab the action addressed (``tab_index``; the handlers act on that tab too).
These register without ``precheck``: ``Verifier.check_already_satisfied`` has its own, more conservative
browser path (it never skips an action for a tab other than the default one).

Navigation: the page's URL equal to the requested one is verified. A different URL is verified only when the
browser's own redirect chain, recorded by the handler, led from the requested URL to the page now shown and that
page is the same page: the same host (``www.`` aside), ``http`` raised to ``https``, the same path (a trailing
slash aside) and every query value that was asked for still there. A sign-in wall, another path or another host
is not the requested page: reported, not verified, and never a mismatch (a retry would be sent to the same wall).
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlsplit

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified

_CHAIN = re.compile(r"^Redirect chain: (?P<chain>.+)$", re.MULTILINE)


def _parts(url: str) -> tuple[str, str, str, list[tuple[str, str]]]:
    candidate = url.strip()
    if not candidate.startswith(("http://", "https://", "file://")):
        candidate = "https://" + candidate
    parsed = urlsplit(candidate)
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if parsed.port and parsed.port not in {80, 443}:
        host = f"{host}:{parsed.port}"
    return parsed.scheme.casefold(), host, parsed.path.rstrip("/"), parse_qsl(parsed.query, keep_blank_values=True)


def same_page(requested: str, found: str) -> bool:
    """Whether ``found`` is the page ``requested`` names, after the redirects a site makes to its canonical form."""
    scheme, host, path, query = _parts(requested)
    found_scheme, found_host, found_path, found_query = _parts(found)
    if found_scheme != scheme and (scheme, found_scheme) != ("http", "https"):
        return False
    return found_host == host and found_path == path and all(item in found_query for item in query)


def _redirect_chain(output: str) -> list[str]:
    match = _CHAIN.search(output or "")
    return [step.strip() for step in match["chain"].split(" -> ")] if match else []


@register(ActionType.BROWSER_NAVIGATE, method="conditional_exact_url_postcondition")
async def browser_at_requested_url(result: ActionResult) -> Outcome:
    """The addressed tab shows the requested page, directly or after redirects to the same page."""
    from pilot.system.browser import browser_action_already_satisfied, browser_current_url

    params = result.action.parameters
    requested = str(getattr(params, "url", "") or "")
    observed = await browser_action_already_satisfied("browser_navigate", params)
    if observed is None:
        return unobserved("The page could not be read")
    if observed[0]:
        return verified("Browser URL matches requested navigation")
    current = await browser_current_url(int(getattr(params, "tab_index", -1)))
    if current is None:
        return unobserved("The page's address could not be read")
    requested_host = _parts(requested)[1]
    chain = _redirect_chain(result.output)
    if (
        len(chain) >= 2
        and _parts(chain[0])[:3] == _parts(requested)[:3]
        and _parts(chain[-1])[:3] == _parts(current)[:3]
        and same_page(requested, current)
    ):
        return verified(f"The browser was redirected {len(chain) - 1} time(s) to the same page on {requested_host}")
    current_host = _parts(current)[1]
    if current_host != requested_host:
        return unobserved(f"The page shown is on {current_host or 'another site'}, not {requested_host}")
    # A consent page, a canonical address or a sign-in wall all look the same from here: say only that it differs.
    return unobserved(f"{requested_host} showed a different page than the one asked for")


@register(
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_FILL_FORM,
    method="conditional_field_value_postcondition",
)
async def browser_fields_hold_values(result: ActionResult) -> Outcome:
    """Each field the action set holds the requested value, read in the tab the action addressed."""
    from pilot.system.browser import browser_action_already_satisfied

    action = result.action
    observed = await browser_action_already_satisfied(action.action_type.value, action.parameters)
    if observed is None:
        return unobserved("The page could not be read")
    if observed[0]:
        return verified("Browser field state matches requested value")
    return mismatch("Browser field state differs from requested value")
