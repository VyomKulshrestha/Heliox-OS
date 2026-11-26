"""Post-conditions for browser navigation and form fields, read from the live page afresh.

These register without ``precheck``: ``Verifier.check_already_satisfied`` has its own, more conservative
browser path (it never skips an action for a tab other than the default one).
"""

from __future__ import annotations

from pilot.actions import ActionResult, ActionType
from pilot.agents.postconditions import Outcome, mismatch, register, unobserved, verified


@register(ActionType.BROWSER_NAVIGATE, method="conditional_exact_url_postcondition")
async def browser_at_requested_url(result: ActionResult) -> Outcome:
    """The page's URL is the requested one."""
    return await _browser_state(result)


@register(
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_SELECT,
    ActionType.BROWSER_FILL_FORM,
    method="conditional_field_value_postcondition",
)
async def browser_fields_hold_values(result: ActionResult) -> Outcome:
    """Each field the action set holds the requested value."""
    return await _browser_state(result)


async def _browser_state(result: ActionResult) -> Outcome:
    from pilot.system.browser import browser_action_already_satisfied

    action = result.action
    params = action.parameters
    if getattr(params, "tab_index", -1) != -1:
        return unobserved("Browser action tab is not bound to the executor backend")
    observed = await browser_action_already_satisfied(action.action_type.value, params)
    if observed is None:
        return unobserved("Browser postcondition is unavailable or has side effects")
    if action.action_type == ActionType.BROWSER_NAVIGATE:
        if observed[0]:
            return verified("Browser URL matches requested navigation")
        return unobserved("Browser URL differs or redirected; exact outcome is unverified")
    if observed[0]:
        return verified("Browser field state matches requested value")
    return mismatch("Browser field state differs from requested value")
