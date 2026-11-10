"""A multi-part file job ("copy these, delete those, zip the result") is planned, checked and undone well.

Each test pins one general rule found by rehearsing such a job against the real daemon.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, FileParams
from pilot.agents import request_folders
from pilot.agents.autonomous import AutonomousJob
from pilot.agents.autonomous_review import AutonomousReviewer
from pilot.agents.base_agent import AgentRole
from pilot.agents.orchestrator import AgentOrchestrator
from pilot.agents.planner import SYSTEM_PROMPT
from pilot.agents.verifier import Verifier
from pilot.config import PilotConfig
from pilot.intelligence.experience import experience_scope, get_experience_context
from pilot.models.router import ModelRouter
from pilot.security import risk_gate
from pilot.security.gateway import InvocationSource


def _copy_plan(source: str, destination: str) -> ActionPlan:
    return ActionPlan(
        actions=[
            Action(
                action_type=ActionType.FILE_COPY,
                target=source,
                parameters=FileParams(path=source, destination=destination, recursive=True),
            )
        ],
        explanation="copy",
    )


# --- planner guidance -------------------------------------------------------------------------


def test_planner_prefers_typed_undoable_file_actions_and_keeps_every_part():
    prompt = SYSTEM_PROMPT.format(os="Windows", path_style="Windows", home="C:\\Users\\me")
    assert 'For a folder add "recursive": true.' in prompt
    assert "Heliox can undo file_write, file_delete, file_move and file_copy" in prompt
    assert "never with Remove-Item" in prompt
    # PowerShell cmdlets do not run through cmd.exe, which is what shell_command uses on Windows.
    assert '"interpreter": "powershell"' in prompt
    assert "Plan every part the person asked for" in prompt
    assert "a backup is a file_copy with the real source" in prompt


# --- folders named in the request ---------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(request_folders, "_home", lambda: tmp_path)
    launch = tmp_path / "Desktop" / "launch"
    (launch / "drafts").mkdir(parents=True)
    (launch / "drafts" / "v1.md").write_text("draft one")
    cache = launch / "build" / "cache"
    cache.mkdir(parents=True)
    for index in range(20):
        (cache / f"chunk-{index}.bin").write_bytes(b"x" * 10)
    (launch / "post-final.md").write_text("final post")
    return tmp_path


def test_a_named_folder_is_listed_with_real_names_and_big_folders_summarised(home):
    text = request_folders.describe_request_folders("Get my launch folder on the Desktop ready to ship.")

    assert text.startswith("[FOLDERS NAMED IN THE REQUEST]")
    assert str(home / "Desktop" / "launch") in text
    assert "post-final.md (10 B)" in text
    assert "drafts\\ (1 file, 9 B)" in text
    assert "cache\\ (20 files, 200 B)" in text
    assert "chunk-3.bin" not in text  # 20 files in one folder collapse into the summary line


def test_only_explicit_existing_folders_inside_home_are_listed(home):
    assert request_folders.describe_request_folders("tidy the reports folder on my Desktop") == ""
    assert request_folders.describe_request_folders("back up my C drive to C:\\Backup") == ""
    assert request_folders.candidate_folders("list C:\\ please") == []
    assert request_folders.candidate_folders(f"zip {home}") == []  # the home folder itself is too broad
    named = request_folders.candidate_folders(f"zip {home / 'Desktop' / 'launch'} now")
    assert named == [home / "Desktop" / "launch"]


def test_heliox_private_folders_are_never_listed(home, monkeypatch):
    private = home / "Desktop" / "launch"
    monkeypatch.setattr("pilot.security.validator.inside_heliox_private_dirs", lambda path: Path(path) == private)
    assert request_folders.describe_request_folders("ship my launch folder on the Desktop") == ""


# --- world model wording --------------------------------------------------------------------------


def test_the_measured_reason_is_kept_next_to_a_stronger_generic_one():
    prediction = {
        "risk_evidence": [
            {
                "source": "permission_tier",
                "score": 1.0,
                "reason": "Action requests root privileges",
                "deterministic": True,
            },
            {
                "source": "size_estimate",
                "score": 0.8,
                "reason": "copying C: (about 692 GB) needs more than the 260 GB free on C:",
                "deterministic": True,
            },
            {"source": "learned", "score": 0.9, "reason": "learned guess", "deterministic": False},
            {"source": "rule", "score": 0.2, "reason": "weak rule", "deterministic": True},
        ]
    }
    reasons = risk_gate._with_supporting_reasons(["Action requests root privileges"], prediction)
    assert reasons == [
        "Action requests root privileges",
        "copying C: (about 692 GB) needs more than the 260 GB free on C:",
    ]


# --- autonomous review ----------------------------------------------------------------------------


async def test_autonomous_review_runs_the_world_model_off_the_event_loop(monkeypatch):
    seen = {}

    def fake_assess(plan, config, *, source=""):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        from pilot.agents.destructive_critic import PlanRiskAssessment

        return PlanRiskAssessment(heuristic_score=0.0, world_model_score=0.0, combined_score=0.0)

    monkeypatch.setattr("pilot.agents.destructive_critic.assess_plan_risk", fake_assess)
    reviewer = AutonomousReviewer(lambda: PilotConfig())
    job = AutonomousJob(job_id="j", goal="copy the folder")

    await reviewer.review_plan(job, _copy_plan("C:\\a", "C:\\b"), InvocationSource.AUTONOMOUS, "p")

    assert seen["on_loop"] is False  # a worker thread gets the full folder-scan budget


# --- verification of folder copies ---------------------------------------------------------------


async def test_a_folder_copy_is_verified_file_by_file(tmp_path):
    source = tmp_path / "screens"
    source.mkdir()
    (source / "a.png").write_bytes(b"1234")
    (source / "sub").mkdir()
    (source / "sub" / "b.png").write_bytes(b"12")
    destination = tmp_path / "release" / "screens"
    destination.mkdir(parents=True)
    (destination / "a.png").write_bytes(b"1234")
    verifier = Verifier(MagicMock())
    params = FileParams(path=str(source), destination=str(destination), recursive=True)

    passed, detail, independent = await verifier._verify_file_copy(params)
    assert (passed, independent) == (False, True)
    assert "missing 1 file" in detail and "sub/b.png" in detail

    (destination / "sub").mkdir()
    (destination / "sub" / "b.png").write_bytes(b"1")
    passed, detail, _ = await verifier._verify_file_copy(params)
    assert passed is False and "size differs" in detail

    (destination / "sub" / "b.png").write_bytes(b"12")
    passed, detail, independent = await verifier._verify_file_copy(params)
    assert (passed, independent) == (True, True)
    assert "all 2 files" in detail


# --- provider circuit --------------------------------------------------------------------------------


async def test_a_cancelled_cloud_call_does_not_mark_the_provider_unavailable():
    config = PilotConfig()
    config.model.provider = "cloud"
    config.model.cloud_provider = "claude"
    router = ModelRouter(config, MagicMock())
    started = asyncio.Event()

    async def slow_generate(*args, **kwargs):
        started.set()
        await asyncio.sleep(30)
        return "late"

    router._cloud = MagicMock()
    router._cloud.generate = slow_generate
    router._cache = MagicMock()
    router._cache.get = AsyncMock(return_value=None)
    router._ollama = MagicMock()
    router._ollama.is_available = AsyncMock(return_value=False)
    router._try_llamacpp = lambda: False

    task = asyncio.create_task(router._generate_with_cache("plan this", system="", json_mode=True, temperature=0.1))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()  # an advisory review's own time limit, or the person pressing Stop
    with pytest.raises(asyncio.CancelledError):
        await task

    router._raise_if_cloud_circuit_open()  # the safety critic and the next plan may still call the provider


# --- undo and receipts through specialist agents ---------------------------------------------------


async def test_specialist_agents_keep_the_callers_task_id_for_undo_and_receipts():
    orchestrator = AgentOrchestrator(model_router=MagicMock())
    plan = _copy_plan("C:\\a", "C:\\b")
    seen = {}

    async def handle_task(user_input, sub_plan, context=None, scope_override=None):
        current = get_experience_context()
        seen["task_id"], seen["plan_id"] = current.task_id, current.plan_id
        return [ActionResult(action=sub_plan.actions[0], success=True, output="ok")]

    agent = MagicMock()
    agent.role = AgentRole.SYSTEM
    agent.get_capabilities = MagicMock(return_value=[])
    agent.attach_orchestrator = MagicMock()
    agent.handle_task = handle_task
    orchestrator._action_registry[ActionType.FILE_COPY] = AgentRole.SYSTEM
    orchestrator._agents[AgentRole.SYSTEM] = agent
    try:
        with experience_scope(task_id="task-123"):
            await orchestrator.execute_plan("copy", plan, plan_id="plan-9")
    finally:
        await orchestrator.stop()

    assert seen == {"task_id": "task-123", "plan_id": "plan-9"}


# --- decomposition and autonomous rounds ------------------------------------------------------------


async def test_the_decomposer_is_told_not_to_invent_steps_that_ask_the_person():
    from pilot.agents.decomposer import TaskDecomposer

    router = MagicMock()
    router.generate = AsyncMock(return_value='{"is_complex": false, "subtasks": []}')

    await TaskDecomposer(router).decompose("copy the post, delete the drafts and zip the release")

    prompt = router.generate.await_args.args[0]
    assert "Do not add steps that ask the" in prompt
    assert "approval dialog before any step that changes or deletes something" in prompt
    assert "{os}" not in prompt


async def test_autonomous_rounds_see_the_folders_the_goal_names(home):
    from pilot.agents.autonomous import AutonomousExecutor

    text = await AutonomousExecutor._request_folder_context("zip my launch folder on the Desktop")
    assert "post-final.md" in text
    assert await AutonomousExecutor._request_folder_context("check the weather") == ""
