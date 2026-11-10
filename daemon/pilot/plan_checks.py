"""Plain summary of the world model's assessment for the plan card's "checked before running" row.

Everything here is read from the real evaluation that ``assess_plan_risk`` returned; nothing is
estimated. A value the evaluation did not produce is ``None`` so the UI can leave it out.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _fraction(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _system(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    state = prediction.get("predicted_state")
    system = state.get("system") if isinstance(state, Mapping) else None
    return system if isinstance(system, Mapping) else {}


def _observed_state(first: Mapping[str, Any]) -> dict[str, Any] | None:
    """The machine state the first step started from. The world model copies the observed
    disk and process values into its prediction and only changes the process count through a
    process effect, whose ``before`` is the observed count."""
    if first.get("input_state_source") != "observed":
        return None
    system = _system(first)
    processes = _count(system.get("process_count"))
    for effect in first.get("expected_effects") or []:
        if isinstance(effect, Mapping) and effect.get("domain") == "process":
            processes = _count(effect.get("before"))
            break
    return {"disk_usage_fraction": _fraction(system.get("disk_usage_fraction")), "process_count": processes}


def _predicted_state(last: Mapping[str, Any]) -> dict[str, Any]:
    """What the world model predicted for the machine after the plan's last step."""
    system = _system(last)
    return {
        "disk_usage_fraction": _fraction(system.get("disk_usage_fraction")),
        "process_count": _count(system.get("process_count")),
    }


def _measured_sizes(predictions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from pilot.security.risk_gate import _format_bytes  # same wording as the gate's own reasons

    sizes: list[dict[str, Any]] = []
    for prediction in predictions:
        for record in prediction.get("size_estimates") or []:
            if not isinstance(record, Mapping) or _count(record.get("bytes")) is None:
                continue
            exact = bool(record.get("size_is_exact"))
            sizes.append(
                {
                    "path_label": str(record.get("path_label") or ""),
                    "bytes": record["bytes"],
                    "size_text": f"{'about' if exact else 'at least'} {_format_bytes(record['bytes'])}",
                    "size_is_exact": exact,
                    "volume": str(record.get("volume") or ""),
                    "usage_before": _fraction(record.get("usage_before")),
                    "usage_after": _fraction(record.get("usage_after")),
                }
            )
    return sizes


def world_model_summary(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Summary of ``PlanRiskAssessment.to_dict()`` for one plan (the gate was enabled)."""
    predictions = [item for item in assessment.get("predictions") or [] if isinstance(item, Mapping)]
    return {
        "enabled": True,
        "unavailable": assessment.get("model_version") == "evaluation-error",
        "raised": bool(assessment.get("requires_confirmation")),
        "score": _fraction(assessment.get("world_model_score")) or 0.0,
        "reasons": [str(reason) for reason in assessment.get("reasons") or []],
        "sources": [str(source) for source in assessment.get("prediction_sources") or []],
        "observed": _observed_state(predictions[0]) if predictions else None,
        "predicted": _predicted_state(predictions[-1]) if predictions else None,
        "measured_sizes": _measured_sizes(predictions),
    }
