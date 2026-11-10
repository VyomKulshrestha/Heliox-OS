"""How long a Gnomon plan review takes for plans of 1, 3 and 6 actions, with the model warm on a free GPU.

    python scripts/check_gnomon_review_timing.py --artifact-dir <dir> --report <training report> [--repeats 5]
        [--json out.json]

Each plan is reviewed ``--repeats`` times through the daemon's ``GnomonReviewer`` (undo on, no time budget), for a
one-part request (scope, harm), a multi-part request (harm only: scope is not reviewed) and a plan that read tool
output (adds the untrusted-instruction question). The interactive limit (``GNOMON_REQUEST_TIMEOUT`` in
pilot/server.py) is set from these numbers; see docs/GNOMON_TRAINING.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pilot.actions import Action, ActionPlan  # noqa: E402
from pilot.intelligence.gnomon_reviewer import GnomonReviewer  # noqa: E402

HOME = "C:\\Users\\sam"
PHOTOS = [f"IMG_{4100 + n}.jpg" for n in range(6)]
ONE_PART = "Move the trip photos from Downloads into Pictures\\Trip"
MULTI_PART = "Move the trip photos from Downloads into Pictures\\Trip, delete the duplicates and zip the folder."
CONTEXT = "Contents of README.txt: Trip photos, October. Keep the originals."


def _plan(size: int) -> ActionPlan:
    actions = [
        Action.model_validate(
            {
                "action_type": "file_move",
                "target": f"{HOME}\\Downloads\\{name}",
                "parameters": {"path": f"{HOME}\\Downloads\\{name}", "destination": f"{HOME}\\Pictures\\Trip\\{name}"},
            }
        )
        for name in PHOTOS[:size]
    ]
    return ActionPlan(actions=actions, explanation="", raw_input="")


def _gpu() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception as error:  # nvidia-smi missing
        return f"unknown ({type(error).__name__})"


async def measure(engine: Any, repeats: int) -> dict[str, Any]:
    reviewer = GnomonReviewer(lambda: engine, call_timeout=60.0)
    started = time.perf_counter()
    reviewer.warm()
    while reviewer.status()["state"] == "loading":
        await asyncio.sleep(0.5)
    load_seconds = round(time.perf_counter() - started, 1)
    if reviewer.status()["state"] != "ready":
        raise RuntimeError(f"Gnomon did not load: {reviewer.status()['last_error']}")
    rows = []
    try:
        for label, request, context in (
            ("one-part request", ONE_PART, ""),
            ("multi-part request", MULTI_PART, ""),
            ("one-part request with tool output", ONE_PART, CONTEXT),
        ):
            for size in (1, 3, 6):
                plan = _plan(size)
                seconds = []
                for _ in range(repeats):
                    tick = time.perf_counter()
                    verdict = await reviewer.review_plan(request, plan, context, undo_enabled=True)
                    seconds.append(time.perf_counter() - tick)
                    if verdict is None:
                        raise RuntimeError("no verdict while ready")
                rows.append(
                    {
                        "case": label,
                        "actions": size,
                        "questions": len(verdict.scores),
                        "median_seconds": round(statistics.median(seconds), 2),
                        "max_seconds": round(max(seconds), 2),
                    }
                )
                print(rows[-1])
    finally:
        await reviewer.close()
    return {"load_and_warm_up_seconds": load_seconds, "gpu_before": "", "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    from pilot.intelligence.typed_decision import LocalTypedDecisionModel

    gpu = _gpu()
    engine = LocalTypedDecisionModel(Path(args.artifact_dir).resolve(), Path(args.report).resolve())
    result = asyncio.run(measure(engine, args.repeats))
    result["gpu_before"] = gpu
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
