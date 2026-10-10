"""Collect observed, goal-conditioned MiniWoB first-action rewards for research.

Pinned MIT task assets are served only in memory. Every candidate is clicked
from a new, independently checked reset; incomplete actions are not labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.collect_miniwob_branches import (
    SOURCE_COMMIT,
    SOURCE_REPO,
    VIEWPORT,
    PinnedAssets,
    _reset,
    parse_tasks,
    visual_feature,
)

_CANDIDATES_JS = """viewport => {
  const found = [];
  const centers = new Set();
  for (const element of document.querySelectorAll('#area *')) {
    const tag = element.tagName.toLowerCase();
    const type = (element.getAttribute('type') || '').toLowerCase();
    const role = (element.getAttribute('role') || '').toLowerCase();
    const style = getComputedStyle(element);
    if (element.closest('[aria-disabled="true"]')) continue;
    const native = tag === 'button' || tag === 'select' || tag === 'a'
      || (tag === 'input' && (type === 'checkbox' || type === 'radio'));
    const painted = element.childElementCount === 0
      && style.backgroundColor !== 'rgba(0, 0, 0, 0)'
      && style.backgroundColor !== 'transparent';
    if (!(native || painted || role === 'button' || role === 'menuitem' || role === 'tab'
          || (style.cursor === 'pointer' && role !== 'menu' && tag !== 'ul'))) continue;
    const rect = element.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) continue;
    const x = Math.round((rect.left + rect.right) / 2);
    const y = Math.round((rect.top + rect.bottom) / 2);
    if (x < 0 || x >= viewport.width || y < 0 || y >= viewport.height) continue;
    const center = `${x}:${y}`;
    if (centers.has(center)) continue;
    centers.add(center);
    const icon = element.matches('svg,[class*="icon"]')
      ? element : element.querySelector('svg,[class*="icon"]');
    const label = element.getAttribute('aria-label')
      || element.closest('label')?.textContent
      || element.innerText || element.getAttribute('value')
      || element.getAttribute('data-color') || '';
    found.push({
      x, y, label: label.trim().replace(/\\s+/g, ' ').slice(0, 120),
      tag, type, role,
      width: Math.round(rect.width), height: Math.round(rect.height),
      background: style.backgroundColor, fill: style.fill,
      declared_color: element.getAttribute('data-color')
        || element.getAttribute('fill') || element.style.backgroundColor || '',
      icon_hint: (icon?.getAttribute('aria-label') || icon?.getAttribute('title')
        || icon?.getAttribute('class') || '').slice(0, 120),
      has_popup: element.getAttribute('aria-haspopup')
        || element.querySelector('[aria-haspopup]')?.getAttribute('aria-haspopup') || '',
      expanded: element.getAttribute('aria-expanded') || '',
      selected: element.getAttribute('aria-selected') || '',
    });
  }
  return found;
}"""

_GOAL_JS = """() => {
  const root = document.querySelector('#query');
  if (!root) return '';
  function describe(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent;
    if (node.nodeType !== Node.ELEMENT_NODE) return '';
    const children = [...node.childNodes].map(describe).join('');
    if (children.trim()) return children;
    const visible = node.getAttribute('aria-label') || node.getAttribute('title')
      || node.getAttribute('alt') || node.getAttribute('data-color')
      || node.getAttribute('fill') || node.style.backgroundColor
      || (node.matches('svg,[class*="icon"]') ? node.getAttribute('class') : '');
    return visible ? `[${node.tagName.toLowerCase()} ${visible}]` : '';
  }
  return [...root.childNodes].map(describe).join('').replace(/\\s+/g, ' ').trim();
}"""


def candidate_actions(page) -> list[dict]:
    return page.evaluate(_CANDIDATES_JS, VIEWPORT)


def goal_description(page) -> str:
    return page.evaluate(_GOAL_JS)


def collect_group(browser, assets: PinnedAssets, task: str, seed: str, max_candidates: int) -> tuple[list[dict], int]:
    reference = None
    rows = []
    attempted = 0
    for index in range(max_candidates):
        context, page, _, before_png, before_dom = _reset(browser, assets, task, seed)
        try:
            candidates = candidate_actions(page)
            state = visual_feature(before_png)
            goal = goal_description(page)
            identity = (before_dom, goal, candidates, state.tobytes())
            if reference is None:
                reference = identity
            elif reference != identity:
                raise ValueError("Independent reward branches did not reset to one exact measured state")
            if index >= len(candidates):
                break
            candidate = candidates[index]
            page.mouse.click(candidate["x"], candidate["y"])
            outcome = page.evaluate("({done: WOB_DONE_GLOBAL, reward: WOB_RAW_REWARD_GLOBAL})")
            attempted += 1
            if outcome["done"] and outcome["reward"] in (-1, 1):
                rows.append(
                    {
                        "goal": goal,
                        "candidate": candidate,
                        "reward": int(outcome["reward"]),
                        "prestate_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                    }
                )
        finally:
            context.close()
    if not ({-1, 1} <= {row["reward"] for row in rows}):
        return [], attempted
    return rows, attempted


def require_disjoint_prestates(rows: list[dict]) -> None:
    states = {
        split: {row["prestate_sha256"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "eval")
    }
    if any(
        states[left] & states[right]
        for left, right in (("train", "validation"), ("train", "eval"), ("validation", "eval"))
    ):
        raise ValueError("Identical modeled pre-action state leaked across task splits")


def main() -> None:
    # Optional dependency (pip install -e ".[browser]"); imported here so the helpers stay importable.
    from playwright.sync_api import sync_playwright

    root = Path(__file__).resolve().parents[1] / "benchmarks/.bench-state"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tasks", required=True)
    parser.add_argument("--validation-tasks", required=True)
    parser.add_argument("--eval-tasks", required=True)
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--eval-samples-per-task", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--rows", type=Path, default=root / "miniwob_reward_actions_v1.json")
    parser.add_argument("--report", type=Path, default=root / "miniwob_reward_actions_v1_report.json")
    args = parser.parse_args()
    try:
        tasks = {split: parse_tasks(getattr(args, f"{split}_tasks")) for split in ("train", "validation", "eval")}
    except ValueError as exc:
        parser.error(str(exc))
    if (
        len(set(task for names in tasks.values() for task in names)) != sum(map(len, tasks.values()))
        or not 2 <= args.samples_per_task <= 100
        or not 2 <= args.max_candidates <= 16
        or (args.eval_samples_per_task is not None and not 2 <= args.eval_samples_per_task <= 100)
        or not 0 <= args.seed_start <= 1000000
        or args.rows.exists()
        or args.report.exists()
    ):
        parser.error("Use task-disjoint splits, bounded settings, and fresh output paths")

    assets = PinnedAssets(SOURCE_COMMIT)
    rows = []
    attempts = Counter()
    skipped = Counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for split, names in tasks.items():
                for task in names:
                    sample_count = (
                        args.eval_samples_per_task
                        if split == "eval" and args.eval_samples_per_task is not None
                        else args.samples_per_task
                    )
                    for index in range(args.seed_start, args.seed_start + sample_count):
                        seed = f"{SOURCE_COMMIT}:{task}:{index}"
                        group_rows, count = collect_group(browser, assets, task, seed, args.max_candidates)
                        attempts[task] += count
                        if not group_rows:
                            skipped[task] += 1
                            continue
                        group = hashlib.sha256(seed.encode()).hexdigest()
                        rows.extend({**row, "group": group, "task": task, "split": split} for row in group_rows)
                    print(
                        json.dumps(
                            {
                                "task": task,
                                "accepted_groups": len({row["group"] for row in rows if row["task"] == task}),
                            }
                        ),
                        flush=True,
                    )
        finally:
            browser.close()
    if any(not any(row["split"] == split for row in rows) for split in tasks):
        raise ValueError("Each split needs positive and negative observed terminal actions")
    require_disjoint_prestates(rows)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n"
    args.rows.write_text(payload, encoding="utf-8")
    report = {
        "purpose": "research_only_goal_conditioned_first_action_rewards",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_license": "MIT",
        "representation": "Visible goal text plus inline visual hints; clickable or painted control "
        "text, role, type, geometry, CSS color, icon, and accessibility state. "
        "No screenshots stored.",
        "asset_sha256": assets.hashes(),
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "tasks": tasks,
        "split_basis": "task_family",
        "samples_per_task": args.samples_per_task,
        "eval_samples_per_task": args.eval_samples_per_task,
        "seed_start": args.seed_start,
        "max_candidates": args.max_candidates,
        "cross_split_identical_prestates": 0,
        "rows": dict(Counter(row["split"] for row in rows)),
        "groups": {split: len({row["group"] for row in rows if row["split"] == split}) for split in tasks},
        "rewards": {
            split: dict(Counter(str(row["reward"]) for row in rows if row["split"] == split)) for split in tasks
        },
        "candidate_attempts": dict(attempts),
        "skipped_noncontrastive_groups": dict(skipped),
        "limitations": "Synthetic first actions selected for both terminal reward classes; "
        "not unbiased task success, multi-step competence, real desktop transfer, "
        "safety, or Jev parity. Model input and runtime promotion require separate tests.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
