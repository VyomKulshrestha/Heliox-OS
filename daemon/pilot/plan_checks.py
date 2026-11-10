"""Plain summary of the world model's assessment for the plan card's "checked before running" row.

Everything here is read from the real evaluation that ``assess_plan_risk`` returned; nothing is
estimated. A value the evaluation did not produce is ``None`` so the UI can leave it out.

Only figures the model has grounds for are passed on:

- Disk: the measured sizes (``size_estimates``). The per-step ``predicted_state`` disk value is
  never passed on, because the gate carries the rule table's fixed per-action guess into it
  (``risk_model._DISK_DELTA_RULES``, for example +2% of the drive for any unmeasured file write)
  and takes the larger of that and the learned model, without saying which one the value came
  from. For the same reason a measured size keeps its volume's usage before and after only when no
  earlier step could have added such a guess to the gate's disk ledger (an action type the rule
  table or the learned model covers).
- Processes: the observed count plus the starts and stops the world model's process lifecycle
  contract predicts. The per-step ``predicted_state`` count is not used, because the gate carries
  the rule table's fixed +1 per shell command into the next step's input. A step whose process
  effect "requires observation" (a shell command, a script) makes the count unknown.
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


def _predicted_process_count(predictions: list[Mapping[str, Any]], observed: int | None) -> int | None:
    """The observed count moved by every start and stop the process contract predicts; ``None``
    when no step starts or stops a process, or when any step's effect is not a known count."""
    if observed is None:
        return None
    delta = 0
    any_effect = False
    for prediction in predictions:
        for effect in prediction.get("expected_effects") or []:
            if not isinstance(effect, Mapping) or effect.get("domain") != "process":
                continue
            before, after = _count(effect.get("before")), _count(effect.get("after"))
            if before is None or after is None:
                return None
            delta += after - before
            any_effect = True
    return observed + delta if any_effect else None


def _predicted_state(predictions: list[Mapping[str, Any]], observed: Mapping[str, Any] | None) -> dict[str, Any]:
    """What the world model has grounds to predict for the machine after the plan (see module doc)."""
    return {
        "disk_usage_fraction": None,
        "process_count": _predicted_process_count(predictions, observed.get("process_count") if observed else None),
    }


def _measured_sizes(predictions: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from pilot.security.risk_gate import _format_bytes  # same wording as the gate's own reasons
    from pilot.security.risk_model import LEARNABLE_ACTION_TYPES

    guessed_types = {action_type.value for action_type in LEARNABLE_ACTION_TYPES}
    sizes: list[dict[str, Any]] = []
    usage_grounded = True
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
                    "usage_before": _fraction(record.get("usage_before")) if usage_grounded else None,
                    "usage_after": _fraction(record.get("usage_after")) if usage_grounded else None,
                }
            )
        if str(prediction.get("action_type") or "") in guessed_types:
            usage_grounded = False
    return sizes


def world_model_summary(assessment: Mapping[str, Any]) -> dict[str, Any]:
    """Summary of ``PlanRiskAssessment.to_dict()`` for one plan (the gate was enabled)."""
    predictions = [item for item in assessment.get("predictions") or [] if isinstance(item, Mapping)]
    observed = _observed_state(predictions[0]) if predictions else None
    return {
        "enabled": True,
        "unavailable": assessment.get("model_version") == "evaluation-error",
        "raised": bool(assessment.get("requires_confirmation")),
        "score": _fraction(assessment.get("world_model_score")) or 0.0,
        "reasons": [str(reason) for reason in assessment.get("reasons") or []],
        "sources": [str(source) for source in assessment.get("prediction_sources") or []],
        "observed": observed,
        "predicted": _predicted_state(predictions, observed) if predictions else None,
        "measured_sizes": _measured_sizes(predictions),
    }
