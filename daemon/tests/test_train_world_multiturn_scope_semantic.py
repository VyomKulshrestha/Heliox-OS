"""Seed tasks stay together in the offline multi-turn training split."""

import json

from benchmarks.jev_scope_suite import ScopeCase
from scripts.train_world_multiturn_scope_semantic import split_agentic


def test_split_agentic_keeps_repeated_seed_task_together(tmp_path):
    tasks = [f"Task {index // 2}" for index in range(100)]
    rows = [{"metadata": {"request": {"conversation": {"seed_task_request":
             {"task_tool_request": {"task": task}}}}}} for task in tasks]
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    cases = tuple(ScopeCase(str(index), "dialogue", "tool", index % 2 == 0,
                            "agentic_validation") for index in range(len(rows)))
    train, calibration, seen = split_agentic(cases, path)
    assert seen == set(tasks)
    assert len(train) + len(calibration) == len(rows)
    train_tasks = {tasks[int(case.id)] for case in train}
    calibration_tasks = {tasks[int(case.id)] for case in calibration}
    assert not train_tasks & calibration_tasks
