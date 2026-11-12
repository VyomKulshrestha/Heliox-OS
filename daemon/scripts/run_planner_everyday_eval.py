"""Run the everyday planner eval: plan each request with a real model and score the plans.

Opt-in and never run in CI: every request is a paid model call. Plans are produced in-process by
the same ``Planner`` the daemon builds (system prompt, specialist hints, skills block, folder
listing for folders the request names, fast path), then scored by ``benchmarks/planner_everyday.py``.
Nothing is executed: the script only calls ``Planner.plan`` and reads the plans it returns.

    cd daemon
    .venv/Scripts/python.exe scripts/run_planner_everyday_eval.py --provider openrouter \
        --model "~deepseek/deepseek-v4-flash-latest" --out ../planner-eval-flash.json
    .venv/Scripts/python.exe scripts/run_planner_everyday_eval.py --provider claude \
        --model claude-opus-5-5 --sample opus15 --max-calls 20 --out ../planner-eval-opus.json

The model settings live only in memory; config.toml is not changed. API keys come from Heliox's
key vault. Usage is recorded in Heliox's budget tracker, so the monthly cap applies and refuses
calls once it is reached. The LLM response cache is bypassed, so every run asks the model afresh.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot.real_folders import use_scratch_folders  # noqa: E402

# Every Heliox folder in a temporary root (settings, data, caches): the planner runs on defaults plus the
# provider and model given below; the key comes from the keyring, the spend goes to real_budget_tracker.
use_scratch_folders()

from benchmarks.planner_everyday import RULES, check_plan, load_cases, plan_steps, summarise  # noqa: E402

# A fixed 15-request sample for the expensive model: the cases that exercise the prompt rules
# (append, prepend, replace, rename, move, copy, folder delete, new folder, two- and three-part
# jobs, the launch job) plus a few one-step chores.
SAMPLES = {
    "opus15": [
        "a01_append_ship_it",
        "a02_prepend_call_venue",
        "a03_replace_readme",
        "a05_rename_post",
        "a07_copy_screenshots_folder",
        "a08_make_folder",
        "a09_delete_drafts_folder",
        "a12_volume_and_brightness",
        "a18_copy_then_append",
        "a20_launch_release_job",
        "a21_launch_ready_to_ship",
        "a23_remove_line",
        "g03_append_todo",
        "g17_mute",
        "g25_type_see_you",
    ]
}


INFRA_ERROR = re.compile(r"unavailable|timed out|timeout|rate limit|\b(429|502|503|504)\b", re.IGNORECASE)


class _NoCache:
    """Stands in for the LLM response cache so the eval neither reads nor fills it."""

    async def get(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def set(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def close(self) -> None:
        return None


class _NoMemory:
    """No remembered context or history: every request is planned as if asked first."""

    async def get_context(self, *args: Any, **kwargs: Any) -> str:
        return ""

    async def get_history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


_CASE = contextvars.ContextVar("planner_eval_case", default="")


class BudgetStop(RuntimeError):
    pass


async def build_planner(provider: str, model: str, *, record_usage: bool = True):
    from pilot.agents.multi_agent import MultiAgentRouter
    from pilot.agents.planner import Planner
    from pilot.config import PilotConfig
    from pilot.models.router import ModelRouter
    from pilot.real_folders import real_budget_tracker
    from pilot.security.vault import KeyVault
    from pilot.skills.loader import SkillRegistry

    config = PilotConfig.load()  # the scratch root's settings (defaults); never the person's file
    config.model.provider = "cloud"
    config.model.cloud_provider = provider
    config.model.cloud_model = model
    router = ModelRouter(config, KeyVault(config))
    router._cache = _NoCache()
    if record_usage:
        tracker = real_budget_tracker(config.model)  # the one write to the person's folders: the spend
        await tracker.initialize()
        router.set_budget_tracker(tracker)

    calls: list[dict[str, Any]] = []
    original = router.generate

    async def counted_generate(*args: Any, **kwargs: Any) -> str:
        record: dict[str, Any] = {"case": _CASE.get(), "started": time.time()}
        calls.append(record)
        try:
            raw = await original(*args, **kwargs)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if "budget" in type(exc).__name__.lower() or "budget" in str(exc).lower():
                raise BudgetStop(str(exc)) from exc
            raise
        record["raw"] = raw
        return raw

    router.generate = counted_generate  # type: ignore[method-assign]

    async def close() -> None:
        await router.close()
        if record_usage:
            await tracker.close()

    registry = SkillRegistry()
    registry.load_all()
    planner = Planner(router, _NoMemory(), skills_context=registry.planner_prompt_block())
    planner.set_multi_agent_router(MultiAgentRouter(router))
    return planner, calls, close


async def plan_case(planner: Any, calls: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    from pilot.agents.request_folders import describe_request_folders

    request = case["request"]
    folder_context = await asyncio.to_thread(describe_request_folders, request)
    token = _CASE.set(f"{case['id']}#{id(asyncio.current_task())}")
    tag = _CASE.get()
    started = time.time()
    try:
        plan = await planner.plan(request, screen_context=folder_context)
    finally:
        _CASE.reset(token)
    own_calls = [c for c in calls if c.get("case") == tag]
    result = check_plan(case, plan, error=plan.error or "", explanation=plan.explanation or "")
    return {
        "id": case["id"],
        "request": request,
        "seconds": round(time.time() - started, 1),
        "model_calls": len(own_calls),
        "fast_path": not own_calls and not plan.error,
        "raw_responses": [c.get("raw") or c.get("error") for c in own_calls],
        "error": plan.error or "",
        "explanation": plan.explanation,
        "actions": plan_steps(plan),
        "rules": result.rules,
        "passed": result.passed,
        "asked": result.asked,
        "failures": result.failures,
    }


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("ok" if value else "FAIL")


def print_table(rows: list[dict[str, Any]]) -> None:
    head = f"{'case':30} " + " ".join(f"{r[:8]:>8}" for r in RULES) + "  steps  result"
    print(head)
    print("-" * len(head))
    for row in rows:
        marks = " ".join(f"{_mark(row['rules'].get(r)):>8}" for r in RULES)
        kinds = ",".join(a["action_type"] for a in row["actions"]) or ("asked" if row["asked"] else "none")
        verdict = "PASS" if row["passed"] else "FAIL"
        print(f"{row['id'][:30]:30} {marks}  {len(row['actions']):>5}  {verdict}  {kinds[:70]}")
        for failure in row["failures"]:
            print(f"{'':32}{failure[:110]}")


def print_summary(rows: list[dict[str, Any]], results: list[Any]) -> dict[str, Any]:
    summary = summarise(results)
    print()
    print(
        f"plans: {summary['plans']}  overall pass: {summary['overall']}/{summary['plans']}  asked: {summary['asked']}"
    )
    for rule in RULES:
        part = summary[rule]
        rate = part["passed"] / part["scored"] if part["scored"] else 0.0
        print(f"  {rule:10} {part['passed']:>3}/{part['scored']:<3} {rate:6.1%}")
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="", help="cloud provider key in the vault, e.g. openrouter or claude")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--rescore", default="", help="score a saved --out file again with the current cases; no model call"
    )
    parser.add_argument(
        "--retry-errors", default="", help="plan again the requests a saved --out file lost to provider errors"
    )
    parser.add_argument("--cases", default=None, help="cases file (default benchmarks/planner_everyday_v1.json)")
    parser.add_argument("--ids", default="", help="comma-separated case ids to run")
    parser.add_argument("--sample", choices=sorted(SAMPLES), default=None)
    parser.add_argument("--repeat", type=int, default=1, help="plan every request this many times")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=0, help="stop starting requests after this many model calls")
    parser.add_argument("--out", default="", help="write raw plans and scores to this JSON file")
    parser.add_argument(
        "--call-seconds",
        type=float,
        default=0.0,
        help="time limit per model call (default: the daemon's own, 35 s); raise it to measure a slow model's plans",
    )
    parser.add_argument("--no-budget-record", action="store_true", help="do not record usage in Heliox's budget")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.rescore:
        saved = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        by_case = {c["id"]: c for c in cases}
        rows = [r for r in saved["rows"] if r["id"] in by_case]
        results = []
        for row in rows:
            result = check_plan(
                by_case[row["id"]], row["actions"], error=row["error"], explanation=row["explanation"] or ""
            )
            row.update(rules=result.rules, passed=result.passed, asked=result.asked, failures=result.failures)
            results.append(result)
        print_table(rows)
        print_summary(rows, results)
        print(f"model calls in that run: {saved.get('model_calls')}  ({saved.get('provider')} {saved.get('model')})")
        return 0

    earlier: dict[str, Any] | None = None
    if args.retry_errors:
        # Plan again only the requests that failed because the provider timed out or was down.
        earlier = json.loads(Path(args.retry_errors).read_text(encoding="utf-8"))
        args.ids = ",".join(r["id"] for r in earlier["rows"] if INFRA_ERROR.search(r["error"]))
        args.provider = args.provider or earlier["provider"]
        args.model = args.model or earlier["model"]
        if not args.ids:
            print("no provider errors to retry")
            return 0
    wanted = [i for i in args.ids.split(",") if i] or SAMPLES.get(args.sample or "", [])
    if wanted:
        by_id = {c["id"]: c for c in cases}
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            parser.error(f"unknown case ids: {', '.join(unknown)}")
        cases = [by_id[i] for i in wanted]
    jobs = [case for case in cases for _ in range(args.repeat)]
    if not args.provider or not args.model:
        parser.error("--provider and --model are required unless --rescore is given")

    if args.call_seconds > 0:
        import pilot.models.cloud as cloud

        cloud.CLOUD_GENERATION_BUDGET_SECONDS = args.call_seconds
    planner, calls, close = await build_planner(args.provider, args.model, record_usage=not args.no_budget_record)
    gate = asyncio.Semaphore(max(1, args.concurrency))
    stopped: list[str] = []

    async def run(case: dict[str, Any]) -> dict[str, Any] | None:
        async with gate:
            if stopped or (args.max_calls and len(calls) >= args.max_calls):
                return None
            for attempt in range(3):
                try:
                    row = await plan_case(planner, calls, case)
                except BudgetStop as exc:
                    stopped.append(str(exc))
                    return None
                if "budget" in row["error"].lower():
                    # Planner.plan turns the budget refusal into a plan error: stop here.
                    stopped.append(row["error"])
                    return None
                row["infra_error"] = bool(INFRA_ERROR.search(row["error"]))
                if not row["infra_error"] or attempt == 2:
                    break
                await asyncio.sleep(35)  # the router rests a provider for 30 s after a failure
            verdict = "pass" if row["passed"] else "FAIL"
            print(
                f"  {row['id']}: {verdict} ({row['model_calls']} calls, {row['seconds']}s)", file=sys.stderr, flush=True
            )
            return row

    try:
        outputs = await asyncio.gather(*(run(case) for case in jobs))
    finally:
        await close()
    rows = [row for row in outputs if row is not None]
    total_calls = len(calls)
    if earlier is not None:
        fresh = {row["id"]: row for row in rows}
        rows = [fresh.get(row["id"], row) for row in earlier["rows"]]
        total_calls += int(earlier.get("model_calls") or 0)
        cases = load_cases(args.cases)
    by_case = {c["id"]: c for c in cases}
    results = [
        check_plan(by_case[r["id"]], r["actions"], error=r["error"], explanation=r["explanation"] or "") for r in rows
    ]

    for row, result in zip(rows, results, strict=True):
        row.update(rules=result.rules, passed=result.passed, asked=result.asked, failures=result.failures)
    print_table(rows)
    summary = print_summary(rows, results)
    infra = sum(bool(INFRA_ERROR.search(row["error"])) for row in rows)
    print(
        f"model calls: {total_calls}  provider errors left: {infra}  skipped requests: {len(jobs) - len(outputs) + outputs.count(None)}"
    )
    if stopped:
        print(f"STOPPED by the budget: {stopped[0]}")
    if args.out:
        payload = {
            "provider": args.provider,
            "model": args.model,
            "model_calls": total_calls,
            "provider_errors": infra,
            "budget_stop": stopped[0] if stopped else "",
            "summary": summary,
            "rows": rows,
        }
        Path(args.out).write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 2 if stopped else 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.stdout.flush()
    sys.stderr.flush()
    # A provider client can leave a non-daemon thread behind; do not let it keep the process alive.
    os._exit(exit_code)
