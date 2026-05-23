"""PlanDiffer — compares failed plan against verification feedback.

Instead of re-planning the entire ActionPlan on retry, PlanDiffer
identifies which actions succeeded and which failed, then returns
a minimal retry plan containing only the failed actions.

This reduces LLM cost and latency by ~50% on retries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pilot.actions import Action, ActionPlan, ActionResult, VerificationResult

logger = logging.getLogger("pilot.agents.plan_differ")

# If more than this fraction of actions failed, fall back to full re-plan
FULL_REPLAN_THRESHOLD = 0.5


class PlanDiffer:
    """Compares a failed plan against verification results.

    Returns a minimal retry plan with only the failed actions,
    preserving successful action outputs for context.
    """

    @staticmethod
    def diff(
        original_plan: ActionPlan,
        results: list[ActionResult],
        verification: VerificationResult,
    ) -> tuple[ActionPlan, list[ActionResult]]:
        """Compare failed plan against verification and return minimal retry plan.

        Returns:
            - retry_plan: ActionPlan with only failed actions
            - successful_results: list of ActionResult from successful actions

        Falls back to full re-plan if >50% of actions failed.
        """
        total = len(original_plan.actions)
        if total == 0:
            return original_plan, []

        failed_indices = set(verification.failed_actions)

        # Also include any actions that returned errors but aren't in failed_actions
        for i, result in enumerate(results):
            if not result.success:
                failed_indices.add(i)

        failure_rate = len(failed_indices) / total

        # Fall back to full re-plan if too many actions failed
        if failure_rate > FULL_REPLAN_THRESHOLD:
            logger.info(
                "PlanDiffer: %.0f%% of actions failed — falling back to full re-plan",
                failure_rate * 100,
            )
            return original_plan, []

        # Separate successful and failed actions
        successful_results = [results[i] for i in range(len(results)) if i not in failed_indices]
        failed_actions = [original_plan.actions[i] for i in sorted(failed_indices) if i < total]

        if not failed_actions:
            # Nothing to retry
            return original_plan, successful_results

        logger.info(
            "PlanDiffer: %d/%d actions failed — partial re-plan for indices %s",
            len(failed_actions),
            total,
            sorted(failed_indices),
        )

        retry_plan = ActionPlan(
            actions=failed_actions,
            explanation=f"Retrying {len(failed_actions)} failed action(s) from original plan",
            raw_input=original_plan.raw_input,
        )

        return retry_plan, successful_results

    @staticmethod
    def merge_results(
        successful_results: list[ActionResult],
        retry_results: list[ActionResult],
        original_plan: ActionPlan,
        verification: VerificationResult,
    ) -> list[ActionResult]:
        """Merge successful results with retry results in original order."""
        failed_indices = set(verification.failed_actions)
        for i, result in enumerate(successful_results + retry_results):
            if not result.success:
                failed_indices.add(i)

        merged: list[ActionResult | None] = [None] * len(original_plan.actions)

        # Place successful results back at their original positions
        success_iter = iter(successful_results)
        retry_iter = iter(retry_results)

        for i in range(len(original_plan.actions)):
            if i not in failed_indices:
                try:
                    merged[i] = next(success_iter)
                except StopIteration:
                    pass
            else:
                try:
                    merged[i] = next(retry_iter)
                except StopIteration:
                    pass

        return [r for r in merged if r is not None]
