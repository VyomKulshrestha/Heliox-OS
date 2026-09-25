"""Tests for evidence-gated River adaptation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, BrowserParams, FileParams
from pilot.agents.verifier import Verifier
from pilot.config import PilotConfig
from pilot.intelligence.experience import ExperienceEventType, ExperienceLedger
from pilot.intelligence.online_learning import MINIMUM_PROMOTION_LABELS, VerifiedOnlineLearner
from pilot.server import PilotServer


@pytest.fixture
def occurred_at() -> str:
    return datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc).isoformat()


@pytest.mark.asyncio
async def test_online_learning_reset_rpc_has_explicit_result_contract(tmp_path):
    server = PilotServer(PilotConfig())
    unavailable = await server._handle_online_learning_reset({}, ws=None)
    assert unavailable["status"] == "error"

    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    await learner.initialize(ledger)
    server._online_learning = learner

    reset = await server._handle_online_learning_reset({}, ws=None)

    assert reset["status"] == "ok"
    assert reset["enabled"] is True
    assert reset["suggestions"]["labels"] == 0
    await ledger.close()


@pytest.mark.asyncio
async def test_suggestion_model_promotes_only_after_repeated_explicit_feedback(
    tmp_path,
    occurred_at,
):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    await learner.initialize(ledger)
    ledger.subscribe(learner.consume)

    initial = learner.score_suggestion(
        pattern_id="terminal_error",
        app_name="powershell",
        priority="high",
        occurred_at=occurred_at,
    )
    assert initial.state == "candidate"
    assert initial.probability == 0.5

    for index in range(MINIMUM_PROMOTION_LABELS):
        suggestion_id = f"suggestion-{index}"
        await ledger.append(
            ExperienceEventType.SUGGESTION_SHOWN,
            idempotency_key=f"{suggestion_id}:shown",
            occurred_at=occurred_at,
            payload={
                "suggestion_id": suggestion_id,
                "pattern_id": "terminal_error",
                "context_app": "powershell",
                "priority": "high",
            },
        )
        await ledger.append(
            ExperienceEventType.SUGGESTION_FEEDBACK,
            idempotency_key=f"{suggestion_id}:feedback",
            occurred_at=occurred_at,
            payload={
                "suggestion_id": suggestion_id,
                "pattern_id": "terminal_error",
                "context_app": "powershell",
                "priority": "high",
                "decision": "accepted",
            },
        )
    await ledger.drain_subscribers()

    learned = learner.score_suggestion(
        pattern_id="terminal_error",
        app_name="powershell",
        priority="high",
        occurred_at=occurred_at,
    )
    assert learned.labels == MINIMUM_PROMOTION_LABELS
    assert learned.state == "promoted"
    assert learned.probability > 0.58
    assert learned.to_dict()["authority"] == "ranking_only"
    await ledger.close()


@pytest.mark.asyncio
async def test_ignored_suggestions_are_negative_labels(tmp_path, occurred_at):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)

    for index in range(MINIMUM_PROMOTION_LABELS):
        await ledger.append(
            ExperienceEventType.SUGGESTION_FEEDBACK,
            idempotency_key=f"ignored-{index}",
            occurred_at=occurred_at,
            payload={
                "suggestion_id": f"ignored-{index}",
                "pattern_id": "browser_research",
                "context_app": "chrome",
                "priority": "low",
                "decision": "ignored",
            },
        )
    await ledger.drain_subscribers()

    learned = learner.score_suggestion(
        pattern_id="browser_research",
        app_name="chrome",
        priority="low",
        occurred_at=occurred_at,
    )
    assert learned.state == "promoted"
    assert learned.probability < 0.42
    assert learner.status()["suggestions"]["negative"] == MINIMUM_PROMOTION_LABELS
    await ledger.close()


@pytest.mark.asyncio
async def test_transition_learning_requires_independent_postcondition(tmp_path, occurred_at):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)

    for index in range(MINIMUM_PROMOTION_LABELS):
        action_id = f"action-{index}"
        plan_id = f"plan-{index}"
        await ledger.append(
            ExperienceEventType.CANDIDATE_ACTION,
            idempotency_key=f"{action_id}:candidate",
            action_id=action_id,
            plan_id=plan_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": 0, "action": {"action_type": "file_write"}},
        )
        await ledger.append(
            ExperienceEventType.ACTION_STARTED,
            idempotency_key=f"{action_id}:started",
            action_id=action_id,
            plan_id=plan_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": 0, "action_type": "file_write"},
        )
        await ledger.append(
            ExperienceEventType.ACTION_COMPLETED,
            idempotency_key=f"{action_id}:complete",
            action_id=action_id,
            plan_id=plan_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={
                "index": 0,
                "success": True,
                "executed": True,
                "callback_observed": True,
                "output_excerpt": "loaded",
            },
        )
    await ledger.drain_subscribers()
    assert learner.status()["transitions"]["labels"] == 0

    for index in range(MINIMUM_PROMOTION_LABELS):
        await ledger.append(
            ExperienceEventType.OUTCOME_VERIFIED,
            idempotency_key=f"plan-{index}:verified",
            plan_id=f"plan-{index}",
            source="verifier",
            occurred_at=occurred_at,
            payload={
                "result_count": 1,
                "verification": {
                    "passed": True,
                    "failed_actions": [],
                    "independently_observed_actions": [0] if index else [],
                },
            },
        )
    await ledger.drain_subscribers()

    assert learner.status()["transitions"]["labels"] == MINIMUM_PROMOTION_LABELS - 1
    assert learner.status()["workflow_patterns"] == 1
    assert (
        learner.score_transition(
            action_type="file_write",
            source="interactive",
            occurred_at=occurred_at,
        ).state
        == "candidate"
    )
    await ledger.close()


@pytest.mark.asyncio
async def test_transition_learning_uses_failed_observed_postcondition(tmp_path, occurred_at):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)
    await ledger.append(
        ExperienceEventType.ACTION_STARTED,
        idempotency_key="started",
        plan_id="plan-1",
        action_id="action-1",
        source="interactive",
        occurred_at=occurred_at,
        payload={"index": 0, "action_type": "file_write"},
    )
    await ledger.append(
        ExperienceEventType.ACTION_COMPLETED,
        idempotency_key="completed",
        plan_id="plan-1",
        action_id="action-1",
        source="interactive",
        occurred_at=occurred_at,
        payload={"index": 0, "success": True, "executed": True, "callback_observed": True},
    )
    await ledger.append(
        ExperienceEventType.OUTCOME_VERIFIED,
        idempotency_key="verified",
        plan_id="plan-1",
        source="verifier",
        occurred_at=occurred_at,
        payload={
            "result_count": 1,
            "verification": {
                "passed": False,
                "failed_actions": [0],
                "independently_observed_actions": [0],
            },
        },
    )
    await ledger.drain_subscribers()

    assert learner.status()["transitions"]["labels"] == 1
    assert learner.status()["transitions"]["negative"] == 1
    await ledger.close()


@pytest.mark.asyncio
async def test_real_verifier_result_trains_only_checked_file_transition(tmp_path, occurred_at):
    from unittest.mock import MagicMock

    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)
    verifier = Verifier(MagicMock())

    for index, actual_content in enumerate(("expected", "unexpected")):
        path = tmp_path / f"output-{index}.txt"
        path.write_text(actual_content, encoding="utf-8")
        action = Action(
            action_type=ActionType.FILE_WRITE,
            target=str(path),
            parameters=FileParams(path=str(path), content="expected"),
        )
        plan_id = f"plan-{index}"
        action_id = f"action-{index}"
        for event_type, suffix, payload in (
            (ExperienceEventType.ACTION_STARTED, "started", {"index": 0, "action_type": "file_write"}),
            (
                ExperienceEventType.ACTION_COMPLETED,
                "completed",
                {"index": 0, "success": True, "executed": True, "callback_observed": True},
            ),
        ):
            await ledger.append(
                event_type,
                idempotency_key=f"{action_id}:{suffix}",
                action_id=action_id,
                plan_id=plan_id,
                source="interactive",
                occurred_at=occurred_at,
                payload=payload,
            )
        verification = await verifier.verify(
            ActionPlan(actions=[action]), [ActionResult(action=action, success=True)]
        )
        await ledger.append(
            ExperienceEventType.OUTCOME_VERIFIED,
            idempotency_key=f"{plan_id}:verified",
            plan_id=plan_id,
            source="verifier",
            occurred_at=occurred_at,
            payload={"result_count": 1, "verification": verification},
        )
    await ledger.drain_subscribers()

    assert learner.status()["transitions"]["labels"] == 2
    assert learner.status()["transitions"]["positive"] == 1
    assert learner.status()["transitions"]["negative"] == 1
    rebuilt = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    await rebuilt.initialize(ledger)
    assert rebuilt.status()["transitions"] == learner.status()["transitions"]
    await ledger.close()


@pytest.mark.asyncio
async def test_directory_copy_existence_does_not_train_world_transition(tmp_path, occurred_at):
    from unittest.mock import MagicMock

    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    action = Action(action_type=ActionType.FILE_COPY, target=str(source),
                    parameters=FileParams(path=str(source), destination=str(destination), recursive=True))
    for event_type, suffix, payload in (
        (ExperienceEventType.ACTION_STARTED, "started", {"index": 0, "action_type": "file_copy"}),
        (ExperienceEventType.ACTION_COMPLETED, "completed",
         {"index": 0, "success": True, "executed": True, "callback_observed": True}),
    ):
        await ledger.append(event_type, idempotency_key=f"copy:{suffix}", action_id="copy-action",
                            plan_id="copy-plan", source="interactive", occurred_at=occurred_at, payload=payload)
    verification = await Verifier(MagicMock()).verify(
        ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])
    await ledger.append(ExperienceEventType.OUTCOME_VERIFIED, idempotency_key="copy:verified",
                        plan_id="copy-plan", source="verifier", occurred_at=occurred_at,
                        payload={"result_count": 1, "verification": verification})
    await ledger.drain_subscribers()

    assert verification.passed is True
    assert verification.independently_observed_actions == []
    assert learner.status()["transitions"]["labels"] == 0
    await ledger.close()


@pytest.mark.asyncio
async def test_browser_world_transition_requires_matching_observed_url(tmp_path, occurred_at, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from pilot.system import browser

    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)
    action = Action(action_type=ActionType.BROWSER_NAVIGATE, target="https://example.com/account",
                    parameters=BrowserParams(url="https://example.com/account"))

    for index, matched in enumerate((True, False)):
        monkeypatch.setattr(browser, "browser_action_already_satisfied",
                            AsyncMock(return_value=(matched, "current page")))
        plan_id = f"browser-plan-{index}"
        action_id = f"browser-action-{index}"
        await ledger.append(ExperienceEventType.ACTION_STARTED, idempotency_key=f"{action_id}:started",
                            plan_id=plan_id, action_id=action_id, source="interactive", occurred_at=occurred_at,
                            payload={"index": 0, "action_type": "browser_navigate"})
        await ledger.append(ExperienceEventType.ACTION_COMPLETED, idempotency_key=f"{action_id}:completed",
                            plan_id=plan_id, action_id=action_id, source="interactive", occurred_at=occurred_at,
                            payload={"index": 0, "success": True, "executed": True, "callback_observed": True})
        verification = await Verifier(MagicMock()).verify(
            ActionPlan(actions=[action]), [ActionResult(action=action, success=True)])
        await ledger.append(ExperienceEventType.OUTCOME_VERIFIED, idempotency_key=f"{plan_id}:verified",
                            plan_id=plan_id, source="verifier", occurred_at=occurred_at,
                            payload={"result_count": 1, "verification": verification})
    await ledger.drain_subscribers()

    assert learner.status()["transitions"]["labels"] == 1
    assert learner.status()["transitions"]["positive"] == 1
    await ledger.close()


@pytest.mark.asyncio
async def test_transition_promotion_requires_support_for_action_and_source(
    tmp_path, occurred_at, monkeypatch
):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)

    for index in range(MINIMUM_PROMOTION_LABELS):
        action_id = f"action-{index}"
        plan_id = f"plan-{index}"
        await ledger.append(
            ExperienceEventType.ACTION_STARTED,
            idempotency_key=f"{action_id}:started",
            plan_id=plan_id,
            action_id=action_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": 0, "action_type": "file_write"},
        )
        await ledger.append(
            ExperienceEventType.ACTION_COMPLETED,
            idempotency_key=f"{action_id}:completed",
            plan_id=plan_id,
            action_id=action_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": 0, "success": True, "executed": True, "callback_observed": True},
        )
        await ledger.append(
            ExperienceEventType.OUTCOME_VERIFIED,
            idempotency_key=f"{plan_id}:verified",
            plan_id=plan_id,
            source="verifier",
            occurred_at=occurred_at,
            payload={
                "result_count": 1,
                "verification": {
                    "passed": False,
                    "failed_actions": [0],
                    "independently_observed_actions": [0],
                },
            },
        )
    await ledger.drain_subscribers()

    supported = learner.score_transition(action_type="file_write", source="interactive", occurred_at=occurred_at)
    other_action = learner.score_transition(action_type="browser_navigate", source="interactive", occurred_at=occurred_at)
    other_source = learner.score_transition(action_type="file_write", source="mcp", occurred_at=occurred_at)
    assert supported.labels == MINIMUM_PROMOTION_LABELS
    assert supported.state == "promoted"
    assert supported.probability < 0.42
    assert other_action.labels == other_source.labels == 0
    assert other_action.state == other_source.state == "candidate"
    rebuilt = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    await rebuilt.initialize(ledger)
    assert rebuilt.score_transition(
        action_type="file_write", source="interactive", occurred_at=occurred_at
    ).state == "promoted"

    from pilot.security.risk_gate import RiskGate
    from pilot.security.risk_observation import OsSnapshot

    monkeypatch.setattr(
        "pilot.security.risk_gate.capture_os_snapshot",
        lambda: OsSnapshot(
            proc_count=100,
            disk_usage_fraction=0.5,
            memory_usage_fraction=0.5,
            disk_path="/",
        ),
    )
    gate = RiskGate(weights_path="/nonexistent.npz")
    gate.set_online_learner(rebuilt)
    plan = ActionPlan(actions=[Action(
        action_type=ActionType.FILE_WRITE,
        target="safe.txt",
        parameters=FileParams(path="safe.txt", content="expected"),
    )])
    risk, reasons = gate.evaluate_plan(plan, PilotConfig(), source="interactive")
    assert risk > 0
    assert "independently verified" in reasons[0]
    assert "verified_online_transition" in gate.last_evaluation["prediction_sources"]
    assert gate.status(enabled=True)["verified_online_transition"]["labels"] == MINIMUM_PROMOTION_LABELS
    await ledger.close()


@pytest.mark.asyncio
async def test_repeated_actions_in_one_plan_do_not_promote_transition(tmp_path, occurred_at):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)

    for index in range(MINIMUM_PROMOTION_LABELS):
        action_id = f"action-{index}"
        await ledger.append(
            ExperienceEventType.ACTION_STARTED,
            idempotency_key=f"{action_id}:started",
            plan_id="one-plan",
            action_id=action_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": index, "action_type": "file_write"},
        )
        await ledger.append(
            ExperienceEventType.ACTION_COMPLETED,
            idempotency_key=f"{action_id}:completed",
            plan_id="one-plan",
            action_id=action_id,
            source="interactive",
            occurred_at=occurred_at,
            payload={"index": index, "success": True, "executed": True, "callback_observed": True},
        )
    await ledger.append(
        ExperienceEventType.OUTCOME_VERIFIED,
        idempotency_key="one-plan:verified",
        plan_id="one-plan",
        source="verifier",
        occurred_at=occurred_at,
        payload={
            "result_count": MINIMUM_PROMOTION_LABELS,
            "verification": {
                "passed": False,
                "failed_actions": list(range(MINIMUM_PROMOTION_LABELS)),
                "independently_observed_actions": list(range(MINIMUM_PROMOTION_LABELS)),
            },
        },
    )
    await ledger.drain_subscribers()

    assert learner.status()["transitions"]["labels"] == MINIMUM_PROMOTION_LABELS
    score = learner.score_transition(action_type="file_write", source="interactive", occurred_at=occurred_at)
    assert score.labels == 1
    assert score.state == "candidate"
    await ledger.close()


@pytest.mark.asyncio
async def test_rebuild_and_reset_checkpoint_are_deterministic(tmp_path, occurred_at):
    state_path = tmp_path / "learning-state.json"
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    for index in range(3):
        await ledger.append(
            ExperienceEventType.SUGGESTION_FEEDBACK,
            idempotency_key=f"feedback-{index}",
            occurred_at=occurred_at,
            payload={
                "suggestion_id": f"suggestion-{index}",
                "pattern_id": "terminal_error",
                "decision": "accepted",
            },
        )

    learner = VerifiedOnlineLearner(state_path)
    await learner.initialize(ledger)
    assert learner.status()["suggestions"]["labels"] == 3
    reset = await learner.reset()
    assert reset["suggestions"]["labels"] == 0

    rebuilt = VerifiedOnlineLearner(state_path)
    await rebuilt.initialize(ledger)
    assert rebuilt.status()["suggestions"]["labels"] == 0
    assert rebuilt.status()["reset_before_sequence"] == 3
    await ledger.close()


@pytest.mark.asyncio
async def test_observations_only_build_coarse_decayed_routines(tmp_path, occurred_at):
    learner = VerifiedOnlineLearner(tmp_path / "learning-state.json")
    ledger = ExperienceLedger(tmp_path / "experience.db")
    await ledger.initialize()
    ledger.subscribe(learner.consume)
    await ledger.append(
        ExperienceEventType.OBSERVATION,
        idempotency_key="screen-observation",
        occurred_at=occurred_at,
        payload={
            "active_app": "Visual Studio Code",
            "window_title": "secret project title is not a feature",
            "raw_media_excluded": True,
        },
    )
    await ledger.drain_subscribers()

    status = learner.status()
    assert status["suggestions"]["labels"] == 0
    assert status["transitions"]["labels"] == 0
    assert status["routine_patterns"][0]["pattern"].startswith("app:visual_studio_code:hour:")
    assert "window_title" not in str(status)
    assert "project title" not in str(status)
    assert status["privacy"]["secret_browsing"] is False
    await ledger.close()
