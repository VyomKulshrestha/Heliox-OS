"""Deletions this run has already made and verified, so a repeat within the run is not a failure.

"Delete X" where nothing is called X fails: the person may have misnamed it, and the real one would
go on running. The one exception is a repeat of a delete that this same run (the same task id of
the experience context) already carried out and verified, for example when a later step failed and
the whole plan was planned again. The delete verifiers note each verified deletion here, and their
pre-check (which the executor runs before the action) treats an absent target as already done only
when it is noted for the current task. Without a task id nothing is noted and the delete fails, which
is the safer answer.

Kept in memory, bounded, and never written to disk.
"""

from __future__ import annotations

from collections import OrderedDict

from pilot.actions import ActionType

MAX_NOTED = 512
_noted: OrderedDict[tuple[str, str, str], None] = OrderedDict()


def _task_id() -> str:
    from pilot.intelligence.experience import get_experience_context

    return str(get_experience_context().task_id or "")


def note_verified_deletion(action_type: ActionType, target: str) -> None:
    task = _task_id()
    if not task or not target:
        return
    key = (task, action_type.value, target)
    _noted[key] = None
    _noted.move_to_end(key)
    while len(_noted) > MAX_NOTED:
        _noted.popitem(last=False)


def deleted_earlier_in_this_run(action_type: ActionType, target: str) -> bool:
    task = _task_id()
    return bool(task and target) and (task, action_type.value, target) in _noted


def clear() -> None:
    _noted.clear()
