from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    BrowserParams,
    SystemInfoParams,
    VerificationResult,
)
from pilot.response_contract import exact_labeled_finding_count, partial_failure_message, success_message


def _action() -> Action:
    return Action(
        action_type=ActionType.SYSTEM_INFO,
        target="system",
        parameters=SystemInfoParams(),
    )


def test_success_message_reports_verified_execution_not_just_plan_intent():
    plan = ActionPlan(actions=[_action()], explanation="Inspect system information.")
    result = ActionResult(action=_action(), success=True, output="Windows")
    verification = VerificationResult(passed=True, details=["Action 0: VERIFIED"])

    assert success_message(plan, [result], verification, dry_run=False) == "Windows"


def test_reported_browser_effect_is_not_called_independently_verified():
    action = Action(
        action_type=ActionType.BROWSER_CLICK_TEXT,
        target="Submit",
        parameters=BrowserParams(text="Submit"),
    )
    plan = ActionPlan(actions=[action], explanation="Submit the form.")
    result = ActionResult(action=action, success=True, output="Clicked Submit")
    reported = VerificationResult(passed=True, details=["Action 0: REPORTED — no independent postcondition"])

    message = success_message(plan, [result], reported, dry_run=False)
    assert message.startswith("Clicked Submit")
    assert "1 action effect was reported successful but not independently verified" in message

    observed = VerificationResult(passed=True, independently_observed_actions=[0])
    assert success_message(plan, [result], observed, dry_run=False) == "Clicked Submit"


def test_mixed_execution_results_avoid_verified_heading_for_unobserved_effect():
    read = _action()
    click = Action(action_type=ActionType.BROWSER_CLICK_TEXT, target="Submit",
                   parameters=BrowserParams(text="Submit"))
    plan = ActionPlan(actions=[read, click], explanation="Inspect and submit.")
    results = [ActionResult(action=read, success=True, output="Windows"),
               ActionResult(action=click, success=True, output="Clicked Submit")]

    message = success_message(plan, results, VerificationResult(passed=True), dry_run=False)

    assert message.startswith("Execution results:\n")
    assert "not independently verified" in message


def test_unobserved_exact_finding_does_not_use_verified_filler():
    action = Action(action_type=ActionType.BROWSER_CLICK_TEXT, target="Submit",
                    parameters=BrowserParams(text="Submit"))
    plan = ActionPlan(actions=[action], raw_input="Return exactly one finding.")
    result = ActionResult(action=action, success=True, output="")

    message = success_message(plan, [result], VerificationResult(passed=True), dry_run=False)

    assert message.startswith("1. Submit: Reported.")
    assert "not independently verified" in message


def test_dry_run_message_explicitly_says_no_changes_were_made():
    plan = ActionPlan(actions=[_action()], explanation="Change a setting.")
    verification = VerificationResult(passed=True)

    assert success_message(plan, [], verification, dry_run=True) == (
        "Dry run completed; no changes were made. Planned: Change a setting."
    )


def test_partial_failure_message_surfaces_real_error_instead_of_plan_intent():
    action = _action()
    result = ActionResult(action=action, success=False, error="Process exited with code 125")
    verification = VerificationResult(
        passed=False,
        details=["Action 0 (system_info): FAILED — Process exited with code 125"],
        failed_actions=[0],
    )

    message = partial_failure_message([result], verification)

    assert "1 of 1 action failed" in message
    assert "Process exited with code 125" in message


def test_exact_labeled_findings_preserve_requested_result_shape():
    actions = [
        Action(
            action_type=ActionType.SYSTEM_INFO,
            target=label,
            parameters=SystemInfoParams(),
        )
        for label in ("Operating System", "CPU", "Memory", "Disk")
    ]
    plan = ActionPlan(
        actions=actions,
        explanation="Inspect four metrics.",
        raw_input="Return exactly four labeled findings.",
    )
    results = [ActionResult(action=action, success=True, output=f"{action.target} details") for action in actions]
    verification = VerificationResult(passed=True)

    message = success_message(plan, results, verification, dry_run=False)

    assert exact_labeled_finding_count(plan) == 4
    assert message.splitlines() == [
        "1. Operating System: Operating System details",
        "2. CPU: CPU details",
        "3. Memory: Memory details",
        "4. Disk: Disk details",
    ]


def test_exact_labeled_findings_select_matching_section_from_each_composite_result():
    actions = [
        Action(
            action_type=ActionType.SYSTEM_INFO,
            target=label,
            parameters=SystemInfoParams(),
        )
        for label in ("Operating System", "CPU", "Memory", "Disk")
    ]
    plan = ActionPlan(
        actions=actions,
        explanation="Inspect four metrics.",
        raw_input="Return exactly four labeled findings.",
    )
    composite = (
        "=== Operating System ===\nrelease: 11\n"
        "=== CPU ===\nAverage usage: 16%\n"
        "=== Memory ===\nUsed: 84%\n"
        "=== Disk ===\nC: 90%\n"
        "=== Network ===\nConnected\n"
    )
    results = [ActionResult(action=action, success=True, output=composite) for action in actions]

    message = success_message(plan, results, VerificationResult(passed=True), dry_run=False)

    assert message.splitlines() == [
        "1. Operating System: release: 11",
        "2. CPU: Average usage: 16%",
        "3. Memory: Used: 84%",
        "4. Disk: C: 90%",
    ]


def test_exact_labeled_findings_split_one_composite_system_result():
    action = _action()
    plan = ActionPlan(
        actions=[action],
        explanation="Inspect four metrics.",
        raw_input="Return exactly four labeled findings.",
    )
    result = ActionResult(
        action=action,
        success=True,
        output=(
            "=== Operating System ===\nrelease: 11\n"
            "=== CPU ===\nAverage usage: 16%\n"
            "=== Memory ===\nUsed: 84%\n"
            "=== Disk ===\nC: 90%\n"
        ),
    )

    message = success_message(
        plan,
        [result],
        VerificationResult(passed=True),
        dry_run=False,
    )

    assert message.splitlines() == [
        "1. Operating System: release: 11",
        "2. CPU: Average usage: 16%",
        "3. Memory: Used: 84%",
        "4. Disk: C: 90%",
    ]
