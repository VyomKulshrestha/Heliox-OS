import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import pilot.network.collab_executor as collab_module
from pilot.actions import (
    Action,
    ActionPlan,
    ActionResult,
    ActionType,
    ApiRequestParams,
    FileParams,
    ShellCommandParams,
    SystemInfoParams,
)
from pilot.network.collab_executor import CollabExecutor, _PendingDelegation, is_delegable_action
from pilot.network.mesh import HelioxMesh
from pilot.security.gateway import InvocationSource


def _system_action() -> Action:
    return Action(action_type=ActionType.SYSTEM_INFO, parameters=SystemInfoParams())


def _shell_action() -> Action:
    return Action(
        action_type=ActionType.SHELL_COMMAND,
        target="whoami",
        parameters=ShellCommandParams(command="whoami"),
    )


def test_only_portable_read_work_can_be_delegated():
    assert is_delegable_action(_system_action()) is True
    assert (
        is_delegable_action(
            Action(
                action_type=ActionType.API_REQUEST,
                parameters=ApiRequestParams(method="GET", url="https://example.com?token=secret"),
            )
        )
        is False
    )
    assert (
        is_delegable_action(
            Action(
                action_type=ActionType.FILE_WRITE,
                parameters=FileParams(path="notes.txt", content="remote write"),
            )
        )
        is False
    )


@pytest.mark.asyncio
async def test_task_result_is_bound_to_selected_peer_and_exact_actions():
    action = _system_action()
    future = asyncio.get_running_loop().create_future()
    collab = CollabExecutor(mesh=MagicMock(), local_executor=MagicMock())
    collab._pending["task"] = _PendingDelegation("expected-peer", (action,), future)
    payload = {
        "task_id": "task",
        "results": [ActionResult(action=action, success=True, output="ok").model_dump(mode="json")],
    }

    await collab.handle_task_result("different-peer", payload)
    assert future.done() is False

    substituted = _shell_action()
    payload["results"] = [ActionResult(action=substituted, success=True, output="ok").model_dump(mode="json")]
    await collab.handle_task_result("expected-peer", payload)

    with pytest.raises(ValueError, match="different action batch"):
        await future


@pytest.mark.asyncio
async def test_disconnected_peer_falls_back_without_waiting():
    action = _system_action()
    expected = [ActionResult(action=action, success=True, output="local")]
    mesh = SimpleNamespace(send_to=AsyncMock(return_value=False))
    local = SimpleNamespace(execute=AsyncMock(return_value=expected))
    collab = CollabExecutor(mesh=mesh, local_executor=local)

    results = await collab._delegate_to_peer("peer", [action], ActionPlan(actions=[action]))

    assert results == expected
    local.execute.assert_awaited_once()
    assert collab._pending == {}


@pytest.mark.asyncio
async def test_remote_callbacks_are_emitted_exactly_once():
    action = _system_action()
    result = ActionResult(action=action, success=True, output="remote")
    local = SimpleNamespace(execute=AsyncMock())
    on_start = AsyncMock()
    on_complete = AsyncMock()
    collab = None
    delegated_payload = None

    async def send_to(peer_id, message_type, payload):
        nonlocal delegated_payload
        assert peer_id == "peer"
        if message_type == "task_delegate":
            delegated_payload = payload
            asyncio.create_task(
                collab.handle_task_result(
                    "peer",
                    {"task_id": payload["task_id"], "results": [result.model_dump(mode="json")]},
                )
            )
        return True

    collab = CollabExecutor(mesh=SimpleNamespace(send_to=send_to), local_executor=local)
    results = await collab._delegate_to_peer(
        "peer",
        [action],
        ActionPlan(actions=[action], raw_input="private user prompt"),
        {"on_action_start": on_start, "on_action_complete": on_complete},
    )

    assert results == [result]
    assert results[0].action is action
    on_start.assert_awaited_once_with(action)
    on_complete.assert_awaited_once_with(result)
    assert on_complete.await_args.args[0].action is action
    local.execute.assert_not_awaited()
    assert "raw_input" not in delegated_payload


@pytest.mark.asyncio
async def test_timeout_retry_does_not_duplicate_start_callback(monkeypatch):
    monkeypatch.setattr(collab_module, "_REMOTE_TIMEOUT", 0.01)
    action = _system_action()
    result = ActionResult(action=action, success=True, output="local")
    on_start = AsyncMock()
    on_complete = AsyncMock()

    async def execute(_plan, **options):
        assert "on_action_start" not in options
        await options["on_action_complete"](result)
        return [result]

    mesh = SimpleNamespace(send_to=AsyncMock(return_value=True))
    local = SimpleNamespace(execute=AsyncMock(side_effect=execute))
    collab = CollabExecutor(mesh=mesh, local_executor=local)

    results = await collab._delegate_to_peer(
        "peer",
        [action],
        ActionPlan(actions=[action]),
        {"on_action_start": on_start, "on_action_complete": on_complete},
    )

    assert results == [result]
    on_start.assert_awaited_once_with(action)
    on_complete.assert_awaited_once_with(result)
    assert mesh.send_to.await_args_list[-1].args[1] == "task_cancel"


@pytest.mark.asyncio
async def test_receiver_blocks_over_authority_delegated_action():
    executor = SimpleNamespace(execute=AsyncMock())
    config = SimpleNamespace(port=8786, collab_exec_enabled=True, skill_sync_enabled=False)
    mesh = HelioxMesh(config, executor, MagicMock(), b"s" * 32)
    mesh.send_to = AsyncMock(return_value=True)
    action = _shell_action()
    task_id = "5fc7d0df-31dc-43f1-87f0-0938cc4c0a45"

    await mesh._handle_delegated_task(
        "peer",
        {"task_id": task_id, "actions": [action.model_dump(mode="json")], "raw_input": "run whoami"},
    )

    executor.execute.assert_not_awaited()
    response = mesh.send_to.await_args.args[2]
    result = ActionResult.model_validate(response["results"][0])
    assert result.success is False
    assert "portable system telemetry" in (result.error or "")


@pytest.mark.asyncio
async def test_receiver_attributes_allowed_action_to_network_agent():
    action = _system_action()
    expected = [ActionResult(action=action, success=True, output="ok")]
    executor = SimpleNamespace(execute=AsyncMock(return_value=expected))
    config = SimpleNamespace(port=8786, collab_exec_enabled=True, skill_sync_enabled=False)
    mesh = HelioxMesh(config, executor, MagicMock(), b"s" * 32)
    mesh.send_to = AsyncMock(return_value=True)
    task_id = "e02096e9-3e55-42df-a5d0-d86e29a44ef9"

    await mesh._handle_delegated_task(
        "peer",
        {"task_id": task_id, "actions": [action.model_dump(mode="json")], "raw_input": "system info"},
    )

    kwargs = executor.execute.await_args.kwargs
    assert kwargs["invocation_source"] is InvocationSource.NETWORK_AGENT
    assert kwargs["user_confirmed"] is False
    assert kwargs["plan_id"] == f"peer-{task_id}"
