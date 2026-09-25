"""Audit shipped learned-risk world-model evidence and inference performance.

Run from ``daemon``:
    python -m benchmarks.world_model_suite --iterations 200 --json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

from benchmarks.react_latency import summarize
from pilot.actions import Action, ActionType, EmptyParams
from pilot.security.risk_model import LEARNABLE_ACTION_TYPE_ORDER, RiskTransitionModel
from pilot.security.risk_observation import OsSnapshot


def _improvement(error: float, baseline: float) -> float:
    return round((1.0 - error / baseline) * 100.0, 4) if baseline > 0 else 0.0


def benchmark(iterations: int = 200) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be at least one")

    model = RiskTransitionModel()
    if not model.is_loaded or not model.is_calibrated:
        raise RuntimeError("shipped calibrated risk-model weights did not load")
    if model.validation_mae is None or model.baseline_mae is None:
        raise RuntimeError("shipped weights do not contain validation evidence")

    snapshot = OsSnapshot(
        proc_count=300,
        disk_usage_fraction=0.868,
        memory_usage_fraction=0.836,
        disk_path="C:\\" if platform.system() == "Windows" else "/",
    )
    actions = {
        action_type.value: Action(
            action_type=action_type,
            target="benchmark",
            parameters=EmptyParams(),
        )
        for action_type in LEARNABLE_ACTION_TYPE_ORDER
    }
    predictions = {name: model.predict(snapshot, action) for name, action in actions.items()}
    direction_checks = {
        "file_write_increases_disk": predictions["file_write"].disk_usage_after > snapshot.disk_usage_fraction,
        "file_delete_decreases_disk": predictions["file_delete"].disk_usage_after < snapshot.disk_usage_fraction,
        "service_start_increases_processes": predictions["service_start"].proc_count_delta_normalized > 0,
        "service_stop_decreases_processes": predictions["service_stop"].proc_count_delta_normalized < 0,
        "process_kill_decreases_processes": predictions["process_kill"].proc_count_delta_normalized < 0,
    }

    timings_ms: list[float] = []
    for _ in range(iterations):
        for action in actions.values():
            started = time.perf_counter_ns()
            model.predict(snapshot, action)
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    validation_mae = model.validation_mae
    baseline_mae = model.baseline_mae
    return {
        "schema_version": "1.0.0",
        "benchmark": "shipped_learned_risk_world_model",
        "scope": "Shipped calibrated weights, stored temporal holdout evidence, direction invariants, and local inference latency",
        "environment": {
            "operating_system": platform.system(),
            "python": platform.python_version(),
        },
        "model_version": model.model_version,
        "training_samples": model.training_samples,
        "validation_samples": model.validation_samples,
        "learned_action_types": len(actions),
        "validation": {
            "disk_delta_mae": validation_mae[0],
            "disk_zero_baseline_mae": baseline_mae[0],
            "disk_improvement_percent": _improvement(validation_mae[0], baseline_mae[0]),
            "process_delta_mae": validation_mae[1],
            "process_zero_baseline_mae": baseline_mae[1],
            "process_improvement_percent": _improvement(validation_mae[1], baseline_mae[1]),
        },
        "direction_checks": direction_checks,
        "direction_checks_passed": sum(direction_checks.values()),
        "inference": summarize(timings_ms),
        "limitations": [
            "Validation metrics are metadata saved during a stratified temporal holdout before the production refit.",
            "The benchmark verifies shipped metadata and inference behavior; it does not recreate training.",
            "Only 12 learnable action types use these weights; deterministic rules remain authoritative for safety and broader coverage.",
            "The model predicts coarse disk/process effects, not the full physical world or user intent.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = benchmark(args.iterations)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(f"model_version: {report['model_version']}")
    print(f"direction_checks: {report['direction_checks_passed']}/{len(report['direction_checks'])}")
    print(f"inference_median_ms: {report['inference']['median_ms']}")


if __name__ == "__main__":
    main()
