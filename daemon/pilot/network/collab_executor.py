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
from typing import TYPE_CHECKING, Any

from pilot.actions import Action, ActionPlan, ActionResult

if TYPE_CHECKING:
    from pilot.agents.executor import Executor
    from pilot.network.mesh import HelioxMesh

logger = logging.getLogger("pilot.network.collab_executor")

# Maximum seconds to wait for a peer to return results
_REMOTE_TIMEOUT = 60


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
        # Maps task_id → asyncio.Future[list[ActionResult]]
        self._pending: dict[str, asyncio.Future[list[ActionResult]]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    async def distribute(
        self,
        plan: ActionPlan,
        batches: list[list[Action]],
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
        if not self._enabled or not self._mesh.peer_ids:
            # No peers — run everything locally
            return await self._run_all_local(plan, batches)

        available_peers = self._get_available_peers()
        all_results: list[ActionResult] = []

        for i, batch in enumerate(batches):
            if not batch:
                continue

            # Assign to a peer if one is available and the batch is safe to delegate
            peer_id = self._pick_peer(available_peers, batch)
            if peer_id:
                results = await self._delegate_to_peer(peer_id, batch, plan)
            else:
                sub_plan = ActionPlan(
                    actions=batch,
                    explanation=f"Collab batch {i + 1}/{len(batches)}",
                    raw_input=plan.raw_input,
                )
                results = await self._local.execute(sub_plan)

            all_results.extend(results)

            # Stop distributing if a batch failed
            if any(not r.success for r in results):
                logger.warning("CollabExecutor: batch %d failed — running remaining batches locally", i + 1)
                for remaining_batch in batches[i + 1 :]:
                    sub_plan = ActionPlan(
                        actions=remaining_batch,
                        explanation=f"Collab batch (fallback) {i + 2}/{len(batches)}",
                        raw_input=plan.raw_input,
                    )
                    all_results.extend(await self._local.execute(sub_plan))
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
        future = self._pending.get(task_id)
        if future is None:
            logger.warning("CollabExecutor: received result for unknown task_id %s", task_id)
            return

        raw_results = payload.get("results", [])
        results = _deserialize_results(raw_results)
        future.set_result(results)
        logger.info(
            "CollabExecutor: received %d result(s) from peer %s for task %s",
            len(results),
            peer_id,
            task_id,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_all_local(self, plan: ActionPlan, batches: list[list[Action]]) -> list[ActionResult]:
        results: list[ActionResult] = []
        for batch in batches:
            if not batch:
                continue
            sub_plan = ActionPlan(
                actions=batch,
                explanation=plan.explanation,
                raw_input=plan.raw_input,
            )
            results.extend(await self._local.execute(sub_plan))
        return results

    async def _delegate_to_peer(
        self,
        peer_id: str,
        batch: list[Action],
        plan: ActionPlan,
    ) -> list[ActionResult]:
        """Send a batch to a peer and wait for results."""
        task_id = str(uuid.uuid4())[:8]
        loop = asyncio.get_event_loop()
        future: asyncio.Future[list[ActionResult]] = loop.create_future()
        self._pending[task_id] = future

        payload = {
            "task_id": task_id,
            "actions": [_serialize_action(a) for a in batch],
            "raw_input": plan.raw_input,
        }
        await self._mesh.send_to(peer_id, "task_delegate", payload)
        logger.info(
            "CollabExecutor: delegated %d action(s) to peer %s (task %s)",
            len(batch),
            peer_id,
            task_id,
        )

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
        except (asyncio.TimeoutError, ValueError) as exc:
            logger.warning(
                "CollabExecutor: peer %s failed for task %s (%s) — running locally",
                peer_id,
                task_id,
                exc,
            )
            self._pending.pop(task_id, None)
            sub_plan = ActionPlan(
                actions=batch,
                explanation=plan.explanation,
                raw_input=plan.raw_input,
            )
            results = await self._local.execute(sub_plan)
        finally:
            self._pending.pop(task_id, None)

        return results

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
        from pilot.actions import PermissionTier

        for action in batch:
            if action.permission_tier >= PermissionTier.SYSTEM_MODIFY:
                return None  # keep sensitive actions local

        if not available:
            return None

        # Pick peer with lowest reported CPU load
        best = min(
            available,
            key=lambda pid: (
                self._mesh.get_connection(pid).peer_capabilities.cpu_load
                if self._mesh.get_connection(pid) and self._mesh.get_connection(pid).peer_capabilities
                else 1.0
            ),
        )
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
