"""Heliox co-adaptation SDK core: a world model of person, interface and assistant, plus a constrained policy.

Standard library and numpy only. Every recommendation is presentation-only (``authority = "presentation_only"``,
``execution_authority = False``): the core adapts how targets are shown and what a selection needs from the person,
and never approves, clicks or runs anything.

The public API below is re-exported lazily (PEP 562), so ``import pilot.coadaptation.core`` loads no submodule and a
name is imported from its module on first use. The evaluation modules (``ope``, ``evaluation``, ``simulator`` and
``audit``) are not re-exported here; import them directly.
"""

from __future__ import annotations

import importlib
from typing import Any

SDK_VERSION = "0.1.0"

_EXPORTS: dict[str, tuple[str, ...]] = {
    "vocabulary": (
        "AssistanceStrategy",
        "ASSISTANCE_OFF",
        "NO_OFFER",
        "DecisionLevel",
        "VariantRole",
        "Variant",
        "VARIANTS",
        "CAUTION_ORDER",
        "l1_arms",
        "L2_OPTIONS",
        "L3_OPTIONS",
        "acting",
        "family_of",
        "applicability_reason",
        "STRATEGY_VERSION",
    ),
    "state": (
        "Modality",
        "TaskKind",
        "Phase",
        "AmbiguityBin",
        "Rect",
        "LayoutSummary",
        "DeviceContext",
        "ClockContext",
        "BlockContext",
        "ActivationContext",
        "OfferContext",
        "build_l1_features",
        "build_l2_features",
        "build_burden_features",
        "FEATURE_SCHEMA",
        "L1_DIM",
        "L2_DIM",
        "BURDEN_DIM",
        "ALLOWED_CONTEXT_KEYS",
        "DENIED_FEATURE_SUBSTRINGS",
        "day_index_for",
    ),
    "special": (
        "norm_pdf",
        "norm_cdf",
        "norm_ppf",
        "log_norm_cdf",
        "mills_ratio",
        "truncnorm_moments",
        "binom_sf",
    ),
    "priors": (
        "PRIOR_TABLE_V1",
        "fitts_time_ms",
        "keyboard_menu_time_ms",
        "keyboard_grid_time_ms",
        "keyboard_presses",
        "menu_sdp_time_s",
        "practice_multiplier",
        "endpoint_sd_px",
        "p_miss",
        "ambiguity_band",
        "l1_physics",
        "l2_physics",
        "burden_physics",
        "PRIOR_VERSION",
    ),
    "target_posterior": (
        "compute_posterior",
        "PosteriorResult",
        "keyboard_endpoint",
        "rendered_rects",
        "ambiguity_bin",
        "cell_id",
    ),
    "filters": (
        "GaussianFilter",
        "ProbitFilter",
        "OrdinalProbitFilter",
        "GammaPoisson",
        "SpreadInvGamma",
        "SPREAD_WINDOW_C",
        "SPREAD_TRUNCATION_FACTOR",
        "in_spread_window",
    ),
    "dynamics": (
        "BernoulliCusum",
        "GaussianCusum",
        "DriftDetector",
        "DRIFT_THRESHOLDS",
        "calibrate_drift_thresholds",
        "in_control_run_lengths",
        "DayClock",
        "Q_SLOW",
        "Q_FAST",
    ),
    "outcome_model": (
        "CoAdaptationWorldModel",
        "ModelVariant",
        "PopulationPrior",
        "InteractionPrediction",
        "OutcomeEstimate",
        "EvidenceCounts",
        "PracticeCounters",
        "CalibrationMonitor",
        "MODEL_VERSION",
    ),
    "preferences": (
        "PreferenceKind",
        "PreferenceScope",
        "Preference",
        "PreferenceSet",
        "PreferenceConflict",
        "preference_from_dict",
        "preference_to_dict",
    ),
    "constraints": (
        "ExclusionReason",
        "ConstraintState",
        "apply_hard_constraints",
        "Fallbacks",
        "FreezeConflict",
        "fallbacks",
        "fixed_l2_rule",
        "cap_applies",
        "preference_violations",
    ),
    "utility": (
        "UtilityWeights",
        "l1_utility",
        "l2_utility",
        "l3_utility",
        "realized_utility",
        "block_burden_term",
        "pointing_throughput",
    ),
    "policy": (
        "PolicyMode",
        "Decision",
        "TableCell",
        "AmbiguityTable",
        "ConstrainedThompsonPolicy",
        "select_from_cell",
        "decision_seed",
        "draw_stream",
        "POLICY_VERSION",
    ),
    "explain": ("Contrast", "Explanation", "build_explanation"),
    "records": (
        "SessionRow",
        "TrialRecord",
        "ActivationRecord",
        "ActivationGaze",
        "SpreadUsed",
        "L2Applied",
        "MovementSummary",
        "OfferResponse",
        "RevertRecord",
        "VetoRecord",
        "HandSummary",
        "GazeSummary",
        "BurdenReport",
        "TrialLabels",
        "EpisodeLabels",
        "StoredTrial",
        "DecisionRecord",
        "RECORD_SCHEMA_VERSION",
        "RECORD_SCHEMAS",
        "derive_labels",
    ),
    "study_plan": (
        "BlockPlan",
        "SessionPlan",
        "build_session_plan",
        "plan_seed",
        "rank_fixed_regimes",
        "STUDY_PROTOCOL_ID",
        "PLAN_RULES_VERSION",
        "PROTOCOL_HYPERPARAMETERS",
        "LAYOUT_GRID",
        "study_protocol_sha256",
    ),
    "engine": (
        "EngineSettings",
        "CoAdaptationEngine",
        "TrialObservation",
        "SessionSummary",
        "EngineStatus",
        "EngineEvent",
        "PolicyProbabilities",
        "NoAllowedRegime",
    ),
    "consent": (
        "CoAdaptationPurpose",
        "ConsentRecord",
        "ConsentPeriod",
        "consent_in_force",
        "ConsentError",
        "PARTICIPANT_KEY_PATTERN",
        "new_participant_key",
    ),
    "evidence": (
        "EvidenceKind",
        "SignalProvenance",
        "EvidenceFrame",
        "EvidenceSource",
        "PointerEvidence",
        "KeyboardEvidence",
        "GazeRegionEvidence",
        "NeuralEvidenceSource",
        "apply_evidence",
        "gate_ambiguity",
    ),
}

_MODULE_OF: dict[str, str] = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = ["SDK_VERSION", *_MODULE_OF]


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
