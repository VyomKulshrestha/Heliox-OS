import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionType, BrowserParams, PowerParams
from pilot.agents.destructive_critic import (
    CriticVerdict,
    DestructiveCriticAgent,
    constrain_verdict_to_plan_authority,
)


@pytest.mark.asyncio
async def test_tier4_plan_gets_blocked():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "BLOCK",
            "risk_score": 0.95,
            "issues": ["Dangerous destructive action"],
            "safe_actions": [],
            "flagged_actions": ["delete_root"],
            "recommendation": "Do not execute",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "ROOT_CRITICAL"
    fake_plan.explanation = "Delete system files"

    verdict = await critic.review(
        "Delete everything",
        fake_plan,
    )

    assert verdict.verdict == "BLOCK"
    assert verdict.is_blocked is True


@pytest.mark.asyncio
async def test_tier3_plan_gets_warned():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "WARN",
            "risk_score": 0.60,
            "issues": ["Potential risk detected"],
            "safe_actions": [],
            "flagged_actions": ["delete_files"],
            "recommendation": "Proceed carefully",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "DESTRUCTIVE"
    fake_plan.explanation = "Remove user files"

    verdict = await critic.review(
        "Delete selected files",
        fake_plan,
    )

    assert verdict.verdict == "WARN"
    assert verdict.has_warnings is True


@pytest.mark.asyncio
async def test_safe_plan_approved():
    model_router = AsyncMock()

    model_router.generate.return_value = json.dumps(
        {
            "verdict": "APPROVE",
            "risk_score": 0.10,
            "issues": [],
            "safe_actions": ["read_file"],
            "flagged_actions": [],
            "recommendation": "Safe to continue",
        }
    )

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "SAFE"
    fake_plan.explanation = "Read a file"

    verdict = await critic.review(
        "Open a text file",
        fake_plan,
    )

    assert verdict.verdict == "APPROVE"
    assert verdict.is_blocked is False
    _, kwargs = model_router.generate.await_args
    assert kwargs["system"]
    assert kwargs["json_mode"] is True
    assert "system_prompt" not in kwargs


@pytest.mark.asyncio
async def test_critic_error_falls_back_to_warn():
    model_router = AsyncMock()

    model_router.generate.side_effect = Exception("LLM unavailable")

    critic = DestructiveCriticAgent(model_router)

    fake_plan = MagicMock()
    fake_plan.actions = []
    fake_plan.max_tier.name = "ROOT_CRITICAL"
    fake_plan.explanation = "Delete files"

    verdict = await critic.review(
        "Delete files",
        fake_plan,
    )

    assert verdict.verdict == "WARN"
    assert verdict.has_warnings is True


def test_low_authority_plan_cannot_be_hard_blocked_for_invented_privileges():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BROWSER_NAVIGATE,
                target="https://example.com",
                parameters=BrowserParams(url="https://example.com"),
            )
        ],
        explanation="Open a public webpage.",
    )
    verdict = CriticVerdict(
        verdict="BLOCK",
        risk_score=0.95,
        issues=["Browser navigation allegedly requires root."],
        recommendation="Do not proceed.",
    )

    constrained = constrain_verdict_to_plan_authority(plan, verdict)

    assert constrained.verdict == "WARN"
    assert constrained.risk_score == 0.74
    assert "normal approval" in constrained.recommendation


def test_root_critical_plan_retains_hard_block():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.POWER_SHUTDOWN,
                target="system",
                parameters=PowerParams(),
                requires_root=True,
            )
        ],
        explanation="Shut down the system.",
    )
    verdict = CriticVerdict(
        verdict="BLOCK",
        risk_score=1.0,
        recommendation="Unsafe power operation.",
    )

    assert constrain_verdict_to_plan_authority(plan, verdict) is verdict


def _delete_plan() -> ActionPlan:
    def action(kind: str, path: str, **extra) -> Action:
        return Action.model_validate(
            {"action_type": kind, "target": path, "parameters": {"path": path, **extra}, "destructive": True}
        )

    return ActionPlan(
        actions=[
            action("file_copy", "C:/launch/post.md", destination="C:/release/post.md"),
            action("file_delete", "C:/launch/drafts", recursive=True),
            Action.model_validate(
                {
                    "action_type": "shell_script",
                    "target": "C:/release.zip",
                    "parameters": {"script": "Compress-Archive -Path C:/release/* -DestinationPath C:/release.zip"},
                }
            ),
        ],
        explanation="Copy the final post, delete the drafts, zip the release.",
    )


_FINDINGS_RESPONSE = json.dumps(
    {
        "verdict": "WARN",
        "risk_score": 0.42,
        "findings": [
            {
                "step": 2,
                "severity": "high",
                "text": "Step 2 deletes launch/drafts, which holds draft-final-FINAL.md. "
                "It can be undone from Heliox for 7 days.",
            },
            {"step": 3, "severity": "low", "text": "Step 3 uses a script for a simple zip."},
            {"step": 9, "severity": "urgent", "text": "Names a step the plan does not have."},
            {"step": None, "severity": "medium", "text": ""},
        ],
        "safe_actions": ["file_copy"],
        "flagged_actions": ["file_delete"],
        "recommendation": "Check draft-final-FINAL.md before approving.",
    }
)


@pytest.mark.asyncio
async def test_prompt_carries_undo_facts_per_action():
    from pilot.config import UndoConfig

    model_router = AsyncMock()
    model_router.generate.return_value = _FINDINGS_RESPONSE
    critic = DestructiveCriticAgent(model_router, undo_config=lambda: UndoConfig(retention_days=7, max_task_mb=512))

    verdict = await critic.review("Tidy the launch folder", _delete_plan())

    prompt = model_router.generate.await_args.args[0]
    lines = prompt.splitlines()
    copy_at = next(i for i, line in enumerate(lines) if line.strip().startswith("1. file_copy"))
    delete_at = next(i for i, line in enumerate(lines) if line.strip().startswith("2. file_delete"))
    script_at = next(i for i, line in enumerate(lines) if line.strip().startswith("3. shell_script"))
    undoable = "undo: Heliox can undo this for 7 days"
    assert undoable in "\n".join(lines[copy_at:delete_at])
    assert undoable in "\n".join(lines[delete_at:script_at])
    assert "512 MB" not in "\n".join(lines[: lines.index("Plan explanation: " + _delete_plan().explanation)])
    assert "undo: Heliox cannot undo this; treat it as permanent" in "\n".join(lines[script_at:])
    system = model_router.generate.await_args.kwargs["system"]
    assert "never call that action permanent" in system and "still warn" in system

    assert verdict.verdict == "WARN" and verdict.ran is True
    assert verdict.findings == [
        {"step": 2, "severity": "high", "text": verdict.findings[0]["text"]},
        {"step": 3, "severity": "low", "text": "Step 3 uses a script for a simple zip."},
        {"step": None, "severity": None, "text": "Names a step the plan does not have."},
    ]
    assert verdict.issues == [finding["text"] for finding in verdict.findings]
    payload = verdict.to_dict()
    assert payload["findings"] == verdict.findings and payload["ran"] is True and payload["downgraded"] is False


@pytest.mark.asyncio
async def test_undo_switched_off_is_reported_as_permanent():
    from pilot.config import UndoConfig

    model_router = AsyncMock()
    model_router.generate.return_value = _FINDINGS_RESPONSE
    critic = DestructiveCriticAgent(model_router, undo_config=lambda: UndoConfig(enabled=False))
    await critic.review("Tidy the launch folder", _delete_plan())
    prompt = model_router.generate.await_args.args[0]
    assert prompt.count("undo is turned off in Settings, so Heliox cannot undo this; treat it as permanent") == 2
    assert "can undo this for" not in prompt


@pytest.mark.asyncio
async def test_without_undo_config_the_prompt_makes_no_undo_claim():
    model_router = AsyncMock()
    model_router.generate.return_value = _FINDINGS_RESPONSE
    await DestructiveCriticAgent(model_router).review("Tidy the launch folder", _delete_plan())
    assert "undo:" not in model_router.generate.await_args.args[0]


@pytest.mark.asyncio
async def test_plain_issues_still_parse_with_steps_the_critic_wrote():
    model_router = AsyncMock()
    model_router.generate.return_value = json.dumps(
        {
            "verdict": "WARN",
            "risk_score": 0.5,
            "issues": ["Step 2 recursively deletes launch/drafts.", "The zip may be incomplete.", "Step 7 is odd."],
            "recommendation": "Check the drafts.",
        }
    )
    verdict = await DestructiveCriticAgent(model_router).review("Tidy", _delete_plan())
    assert [(f["step"], f["severity"]) for f in verdict.findings] == [(2, None), (None, None), (None, None)]
    assert verdict.issues[0] == "Step 2 recursively deletes launch/drafts."


@pytest.mark.asyncio
async def test_failed_review_says_it_did_not_run():
    model_router = AsyncMock()
    model_router.generate.side_effect = TimeoutError("provider timed out")
    verdict = await DestructiveCriticAgent(model_router).review("Tidy", _delete_plan())
    assert verdict.verdict == "WARN" and verdict.ran is False and verdict.findings == []


def test_downgraded_block_keeps_the_critics_findings_only():
    plan = ActionPlan(
        actions=[
            Action(
                action_type=ActionType.BROWSER_NAVIGATE,
                target="https://example.com",
                parameters=BrowserParams(url="https://example.com"),
            )
        ],
        explanation="Open a public webpage.",
    )
    finding = {"step": 1, "severity": "medium", "text": "Opens a page."}
    verdict = CriticVerdict(verdict="BLOCK", risk_score=0.9, issues=["Opens a page."], findings=[finding])
    constrained = constrain_verdict_to_plan_authority(plan, verdict)
    assert constrained.downgraded is True and constrained.findings == [finding]
    assert len(constrained.issues) == 2


def _sized_plan(folder, destination) -> ActionPlan:
    """The launch-folder job: copy a post, delete drafts, overwrite nothing else."""
    return ActionPlan(
        actions=[
            Action.model_validate(
                {
                    "action_type": "file_copy",
                    "target": str(folder / "post-final.md"),
                    "parameters": {"path": str(folder / "post-final.md"), "destination": str(destination)},
                }
            ),
            Action.model_validate(
                {
                    "action_type": "file_delete",
                    "target": str(folder / "drafts"),
                    "parameters": {"path": str(folder / "drafts"), "recursive": True},
                    "destructive": True,
                }
            ),
        ],
        explanation="Copy the final post and delete the drafts.",
    )


def _launch_folder(tmp_path, draft_bytes: int):
    folder = tmp_path / "launch"
    (folder / "drafts").mkdir(parents=True)
    (folder / "post-final.md").write_bytes(b"x" * 225)
    (folder / "drafts" / "draft-final-FINAL.md").write_bytes(b"d" * draft_bytes)
    destination = tmp_path / "release"
    destination.mkdir()
    return folder, destination


async def _prompt(plan: ActionPlan, undo, **kwargs) -> str:
    model_router = AsyncMock()
    model_router.generate.return_value = _FINDINGS_RESPONSE
    await DestructiveCriticAgent(model_router, undo_config=lambda: undo, **kwargs).review("Tidy", plan)
    return model_router.generate.await_args.args[0]


@pytest.mark.asyncio
async def test_small_task_gets_a_plain_undo_fact(tmp_path):
    from pilot.config import UndoConfig

    folder, destination = _launch_folder(tmp_path, draft_bytes=636 * 1024)
    prompt = await _prompt(_sized_plan(folder, destination), UndoConfig())
    assert prompt.count("undo: Heliox can undo this for 7 days") == 2
    assert "may not be undoable" not in prompt
    assert "about 636 KB, well under its 512 MB limit" in prompt


@pytest.mark.asyncio
async def test_large_task_may_not_be_undoable(tmp_path):
    from pilot.config import UndoConfig

    folder, destination = _launch_folder(tmp_path, draft_bytes=700 * 1024)
    prompt = await _prompt(_sized_plan(folder, destination), UndoConfig(max_task_mb=1))
    assert prompt.count("undo: may not be undoable: this task may change more than 1 MB") == 2
    assert "can undo this for" not in prompt
    assert "close to or over its 1 MB limit" in prompt


@pytest.mark.asyncio
async def test_unmeasured_task_may_not_be_undoable(tmp_path):
    from pilot.config import UndoConfig
    from pilot.security.risk_observation import TreeSize

    folder, destination = _launch_folder(tmp_path, draft_bytes=10)
    seen: list[str] = []

    def stopped_early(path: str, *, deadline: float) -> TreeSize:
        seen.append(path)
        return TreeSize(bytes=4096, entries=10, complete=False, stopped_early=True)

    prompt = await _prompt(_sized_plan(folder, destination), UndoConfig(), measure_size=stopped_early)
    # The copy lands on a new file, so only the deleted folder is measured.
    assert seen == [str(folder / "drafts")]
    assert prompt.count("undo: may not be undoable: this task may change more than 512 MB") == 2
    assert "could not finish measuring what undo must keep (at least 4.0 KB so far)" in prompt


@pytest.mark.asyncio
async def test_prompt_tells_the_critic_to_state_undo_once():
    model_router = AsyncMock()
    model_router.generate.return_value = _FINDINGS_RESPONSE
    await DestructiveCriticAgent(model_router).review("Tidy", _delete_plan())
    system = model_router.generate.await_args.kwargs["system"]
    assert "at most once, in the recommendation, never in each" in system
    assert "Never ask the user to check sizes" in system
