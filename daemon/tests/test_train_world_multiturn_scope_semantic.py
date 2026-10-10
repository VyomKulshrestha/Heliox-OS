"""Seed tasks stay together in the offline multi-turn training split."""

import json

import numpy as np

from benchmarks.refscope_suite import ScopeCase
from scripts.train_world_multiturn_scope_semantic import select_head, split_agentic


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


def test_three_domain_head_calibrates_against_each_negative_group():
    domains = ("agentic", "tbac", "scope")
    train = np.asarray([[-2.0], [-1.0], [1.0], [2.0]] * len(domains))
    labels = {name: np.asarray([False, False, True, True]) for name in domains}
    calibration = {name: np.asarray([[-1.5], [-0.5], [0.5], [1.5]])
                   for name in domains}
    head, threshold, trials = select_head(train, labels, calibration, labels, domains)
    assert len(trials) == 4
    assert all(np.all(head.predict_proba(calibration[name])[:2, 1] < threshold)
               for name in domains)
