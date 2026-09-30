"""Collaborative task execution across LAN peers.

Distributes parallelizable action batches to available peers so that
independent sub-tasks run concurrently on multiple machines.

How it works
------------
1. ``Executor._analyze_dependencies()`` already splits an ``ActionPlan``
   into batches of independent actions.
2. ``CollabExecutor.distribute()`` takes those batches and assigns each
   batch to either the local executor or a remote peer, based on peer
   load and capability.
3. Remote batches are sent as ``task_delegate`` messages.  The receiving
   peer executes them and returns ``task_result`` messages.
4. Results from all peers are merged and returned in original batch order.

Fallback
--------
If no peers are available, or if ``collab_exec_enabled`` is False, all
batches are executed locally — identical to the existing behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pilot.actions import Action, ActionPlan, ActionResult, ActionType, PermissionTier

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.network.mesh import HelioxMesh

logger = logging.getLogger("pilot.network.collab_executor")

# Maximum seconds to wait for a peer to return results
_REMOTE_TIMEOUT = 60

_REMOTE_SAFE_ACTION_TYPES = frozenset(
    {
        ActionType.SYSTEM_INFO,
        ActionType.SYSTEM_HEALTH_REVIEW,
        ActionType.CPU_USAGE,
        ActionType.MEMORY_USAGE,
        ActionType.DISK_USAGE,
        ActionType.NETWORK_INFO,
        ActionType.BATTERY_INFO,
        ActionType.PROCESS_LIST,
        ActionType.PROCESS_INFO,
    }
)


def is_delegable_action(action: Action) -> bool:
    """Return whether an action is safe and meaningful on another peer."""
    if action.permission_tier >= PermissionTier.SYSTEM_MODIFY or action.is_irreversible:
        return False
    if action.action_type not in _REMOTE_SAFE_ACTION_TYPES:
        return False
    return True


@dataclass(frozen=True)
class _PendingDelegation:
    peer_id: str
    actions: tuple[Action, ...]
    future: asyncio.Future[list[ActionResult]]


class CollabExecutor:
    """Distributes independent action batches across available LAN peers.

    Parameters
    ----------
    mesh:
        The ``HelioxMesh`` instance for peer communication.
    local_executor:
        The local ``Executor`` used for batches that stay on this machine.
    enabled:
        Master switch — if False all batches run locally.
    """

    def __init__(
        self,
        mesh: HelioxMesh,
        local_executor: Executor,
        enabled: bool = True,
    ) -> None:
        self._mesh = mesh
        self._local = local_executor
        self._enabled = enabled
        # Maps task IDs to peer-bound, action-bound pending records.
        self._pending: dict[str, _PendingDelegation] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def distribute(
        self,
        plan: ActionPlan,
        batches: list[list[Action]],
        execution_options: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Execute batches, distributing to peers where possible.

        Parameters
        ----------
        plan:
            The original ``ActionPlan`` (used for context/metadata).
        batches:
            Pre-computed independent action batches from
            ``Executor._analyze_dependencies()``.

        Returns
        -------
        list[ActionResult]
            All results in the same order as the input batches.
        """
        options = dict(execution_options or {})
        if not self._enabled or not self._mesh.peer_ids:
            # No peers — run everything locally
            return await self._run_all_local(plan, batches, options)

        available_peers = self._get_available_peers()
        all_results: list[ActionResult] = []

        for i, batch in enumerate(batches):
            if not batch:
                continue
            cancel_event = options.get("cancel_event")
            if cancel_event is not None and cancel_event.is_set():
                for remaining_batch in batches[i:]:
                    all_results.extend(
                        ActionResult(action=action, success=False, error="Skipped due to cancel request")
                        for action in remaining_batch
                    )
                break

            # Assign to a peer if one is available and the batch is safe to delegate
            peer_id = self._pick_peer(available_peers, batch)
            batch_options = self._batch_options(options, i)
            if peer_id:
                results = await self._delegate_to_peer(peer_id, batch, plan, batch_options)
            else:
                sub_plan = ActionPlan(
                    actions=batch,
                    explanation=f"Collab batch {i + 1}/{len(batches)}",
                    raw_input=plan.raw_input,
                )
                results = await self._local.execute(sub_plan, **batch_options)

            all_results.extend(results)

            # Stop distributing if a batch failed
            if any(not r.success for r in results):
                logger.warning("CollabExecutor: batch %d failed — running remaining batches locally", i + 1)
                for fallback_index, remaining_batch in enumerate(batches[i + 1 :], start=i + 1):
                    sub_plan = ActionPlan(
                        actions=remaining_batch,
                        explanation=f"Collab batch (fallback) {fallback_index + 1}/{len(batches)}",
                        raw_input=plan.raw_input,
                    )
                    all_results.extend(
                        await self._local.execute(sub_plan, **self._batch_options(options, fallback_index))
                    )
                break

        return all_results

    async def handle_task_result(self, peer_id: str, payload: dict[str, Any]) -> None:
        """Called by HelioxMesh when a ``task_result`` message arrives.

        Parameters
        ----------
        peer_id:
            The peer that completed the task.
        payload:
            Dict with ``task_id`` and ``results`` (list of serialised ActionResult).
        """
        task_id = payload.get("task_id", "")
        pending = self._pending.get(task_id)
        if pending is None:
            logger.warning("CollabExecutor: received result for unknown task_id %s", task_id)
            return
        if pending.peer_id != peer_id:
            logger.warning(
                "CollabExecutor: rejected task %s result from unexpected peer %s (expected %s)",
                task_id,
                peer_id,
                pending.peer_id,
            )
            return

        raw_results = payload.get("results", [])
        results = _deserialize_results(raw_results)
        expected_actions = [action.model_dump(mode="json") for action in pending.actions]
        returned_actions = [result.action.model_dump(mode="json") for result in results]
        if len(results) != len(pending.actions) or returned_actions != expected_actions:
            if not pending.future.done():
                pending.future.set_exception(
                    ValueError(f"Peer {peer_id} returned results for a different action batch")
                )
            return
        if not pending.future.done():
            # Peer payloads reconstruct equivalent Action objects. Restore the
            # coordinator's original action instances so parent-plan callbacks
            # can retain stable action indices and experience identities.
            pending.future.set_result(
                [result.model_copy(update={"action": action}) for action, result in zip(pending.actions, results)]
            )
        logger.info(
            "CollabExecutor: received %d result(s) from peer %s for task %s",
            len(results),
            peer_id,
            task_id,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def has_available_peers(self) -> bool:
        return self._enabled and bool(self._get_available_peers())

    def should_distribute(self, plan: ActionPlan, batches: list[list[Action]]) -> bool:
        if not self.has_available_peers or any(action.use_previous_output for action in plan.actions):
            return False
        available = self._get_available_peers()
        return any(self._pick_peer(available, batch) is not None for batch in batches)

    @staticmethod
    def _batch_options(options: dict[str, Any], batch_index: int) -> dict[str, Any]:
        batch_options = dict(options)
        batch_options["allow_collaboration"] = False
        batch_options["action_index_offset"] = 0
        plan_id = batch_options.get("plan_id")
        if plan_id:
            batch_options["plan_id"] = f"{plan_id}-mesh-{batch_index + 1}"
        return batch_options

    async def _run_all_local(
        self,
        plan: ActionPlan,
        batches: list[list[Action]],
        execution_options: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        results: list[ActionResult] = []
        options = dict(execution_options or {})
        for index, batch in enumerate(batches):
            if not batch:
                continue
            sub_plan = ActionPlan(
                actions=batch,
                explanation=plan.explanation,
                raw_input=plan.raw_input,
            )
            results.extend(await self._local.execute(sub_plan, **self._batch_options(options, index)))
        return results

    async def _delegate_to_peer(
        self,
        peer_id: str,
        batch: list[Action],
        plan: ActionPlan,
        execution_options: dict[str, Any] | None = None,
    ) -> list[ActionResult]:
        """Send a batch to a peer and wait for results."""
        task_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[ActionResult]] = loop.create_future()
        self._pending[task_id] = _PendingDelegation(peer_id=peer_id, actions=tuple(batch), future=future)

        payload = {
            "task_id": task_id,
            "actions": [_serialize_action(a) for a in batch],
        }
        sent = await self._mesh.send_to(peer_id, "task_delegate", payload)
        if not sent:
            self._pending.pop(task_id, None)
            sub_plan = ActionPlan(actions=batch, explanation=plan.explanation, raw_input=plan.raw_input)
            return await self._local.execute(sub_plan, **dict(execution_options or {}))
        logger.info(
            "CollabExecutor: delegated %d action(s) to peer %s (task %s)",
            len(batch),
            peer_id,
            task_id,
        )
        options = dict(execution_options or {})
        callback = options.get("on_action_start")
        if callback:
            for action in batch:
                await callback(action)

        try:
            results = await asyncio.wait_for(future, timeout=_REMOTE_TIMEOUT)
            # Integrity check: if the peer returned fewer results than actions
            # in the batch (e.g. due to silent deserialisation failures), fall
            # back to local execution to avoid silent task truncation.
            if len(results) != len(batch):
                logger.warning(
                    "CollabExecutor: peer %s returned %d result(s) for %d action(s) "
                    "(task %s) — falling back to local execution",
                    peer_id,
                    len(results),
                    len(batch),
                    task_id,
                )
                raise ValueError(f"Incomplete results from peer {peer_id}: expected {len(batch)}, got {len(results)}")
            callback = options.get("on_action_complete")
            if callback:
                for result in results:
                    await callback(result)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(self._cancel_remote_task(peer_id, task_id))
            finally:
                raise
        except (asyncio.TimeoutError, ValueError) as exc:
            logger.warning(
                "CollabExecutor: peer %s failed for task %s (%s) — running locally",
                peer_id,
                task_id,
                exc,
            )
            self._pending.pop(task_id, None)
            await self._cancel_remote_task(peer_id, task_id)
            sub_plan = ActionPlan(
                actions=batch,
                explanation=plan.explanation,
                raw_input=plan.raw_input,
            )
            fallback_options = dict(options)
            fallback_options.pop("on_action_start", None)
            results = await self._local.execute(sub_plan, **fallback_options)
        finally:
            self._pending.pop(task_id, None)

        return results

    async def _cancel_remote_task(self, peer_id: str, task_id: str) -> None:
        try:
            await self._mesh.send_to(peer_id, "task_cancel", {"task_id": task_id})
        except Exception:
            logger.warning("CollabExecutor: could not cancel remote task %s on peer %s", task_id, peer_id)

    def _get_available_peers(self) -> list[str]:
        """Return peer IDs that are connected and can execute tasks."""
        available = []
        for pid in self._mesh.peer_ids:
            conn = self._mesh.get_connection(pid)
            if conn and conn.connected:
                caps = conn.peer_capabilities
                if caps is None or caps.can_execute:
                    available.append(pid)
        return available

    def _pick_peer(self, available: list[str], batch: list[Action]) -> str | None:
        """Pick the least-loaded peer for a batch, or None to run locally.

        Only delegates READ_ONLY and USER_WRITE tier actions — never
        delegates SYSTEM_MODIFY, DESTRUCTIVE, or ROOT_CRITICAL actions.
        """
        if any(not is_delegable_action(action) for action in batch):
            return None

        if not available:
            return None

        # Priority criteria:
        # 1. Has NVIDIA GPU (has_gpu=True)
        # 2. Most available VRAM
        # 3. Lowest CPU load
        def peer_priority(pid: str) -> tuple[bool, int, float]:
            conn = self._mesh.get_connection(pid)
            if not conn or not conn.peer_capabilities:
                return (False, 0, 1.0)
            caps = conn.peer_capabilities
            return (caps.has_gpu, caps.vram_free, -caps.cpu_load)

        best = max(available, key=peer_priority)
        return best


# ── Serialisation helpers ─────────────────────────────────────────────────────


def _serialize_action(action: Action) -> dict[str, Any]:
    """Convert an Action to a JSON-serialisable dict."""
    return action.model_dump(mode="json")


def _deserialize_results(raw: list[dict[str, Any]]) -> list[ActionResult]:
    """Reconstruct ActionResult objects from peer response payload."""
    results = []
    for item in raw:
        try:
            results.append(ActionResult.model_validate(item))
        except Exception as exc:
            logger.warning("CollabExecutor: failed to deserialise result: %s", exc)
    return results
