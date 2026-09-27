from __future__ import annotations

from pilot.actions import (
    Action,
    ActionType,
    BrowserParams,
    DownloadParams,
    EmptyParams,
    FileParams,
    MouseParams,
    ProcessParams,
    ServiceParams,
    ShellCommandParams,
)
from pilot.intelligence.world_model import (
    EffectDomain,
    HistoricalFailureRisk,
    HybridWorldModel,
    StructuredTransitionPredictor,
    UiJepaPredictor,
    WorldPrediction,
    WorldState,
)


def _state() -> WorldState:
    return WorldState(
        observed_at="2026-07-30T00:00:00+00:00",
        system={"process_count": 100, "disk_usage_fraction": 0.5},
        ui={"active_window": "Heliox"},
        browser={"url": "https://example.com", "dom_state": "observed"},
    )


def test_browser_navigation_predicts_url_and_invalidates_dom() -> None:
    action = Action(
        action_type=ActionType.BROWSER_NAVIGATE,
        target="https://example.org",
        parameters=BrowserParams(url="https://example.org"),
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)

    assert prediction.predicted_state.browser == {
        "url": "https://example.org",
        "dom_state": "requires_observation",
    }
    assert [effect.domain for effect in prediction.expected_effects[:2]] == [
        EffectDomain.BROWSER,
        EffectDomain.UI,
    ]
    assert prediction.uncertainty == 0.2


def test_browser_click_predicts_possible_dom_change_not_fake_success() -> None:
    action = Action(
        action_type=ActionType.BROWSER_CLICK_TEXT,
        target="Learn more",
        parameters=BrowserParams(text="Learn more"),
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)

    effect = prediction.expected_effects[0]
    assert effect.operation == "mutate_document"
    assert effect.after == "possibly_changed"
    assert effect.requires_observation is True
    assert prediction.uncertainty == 0.45


def test_file_delete_predicts_absence_and_irreversible_evidence() -> None:
    action = Action(
        action_type=ActionType.FILE_DELETE,
        target="C:/tmp/report.txt",
        parameters=FileParams(path="C:/tmp/report.txt"),
        reversible=False,
        dangerous_flags=["delete"],
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)

    file_effect = next(effect for effect in prediction.expected_effects if effect.domain == EffectDomain.FILESYSTEM)
    assert file_effect.after == "absent"
    assert file_effect.reversible is False
    assert {item.source for item in prediction.risk_evidence} == {
        "reversibility",
        "validator",
    }


def test_file_copy_and_move_predict_distinct_source_destination_effects() -> None:
    copy = Action(
        action_type=ActionType.FILE_COPY,
        target="source.txt",
        parameters=FileParams(path="source.txt", destination="copy.txt"),
    )
    move = Action(
        action_type=ActionType.FILE_MOVE,
        target="source.txt",
        parameters=FileParams(path="source.txt", destination="moved.txt"),
    )

    copy_effects = [
        effect
        for effect in StructuredTransitionPredictor().predict(_state(), copy).expected_effects
        if effect.domain == EffectDomain.FILESYSTEM
    ]
    move_effects = [
        effect
        for effect in StructuredTransitionPredictor().predict(_state(), move).expected_effects
        if effect.domain == EffectDomain.FILESYSTEM
    ]

    assert [(effect.target, effect.after) for effect in copy_effects] == [("copy.txt", "present_or_updated")]
    assert [(effect.target, effect.after) for effect in move_effects] == [
        ("source.txt", "absent"),
        ("moved.txt", "present_or_updated"),
    ]


def test_download_predicts_output_file_not_source_url() -> None:
    action = Action(
        action_type=ActionType.DOWNLOAD_FILE,
        target="https://example.invalid/file",
        parameters=DownloadParams(url="https://example.invalid/file", output_path="downloaded.bin"),
    )

    effects = [
        effect
        for effect in StructuredTransitionPredictor().predict(_state(), action).expected_effects
        if effect.domain == EffectDomain.FILESYSTEM
    ]

    assert [(effect.target, effect.after) for effect in effects] == [("downloaded.bin", "present_or_updated")]


def test_process_and_service_effects_are_both_exposed() -> None:
    action = Action(
        action_type=ActionType.SERVICE_START,
        target="spooler",
        parameters=ServiceParams(name="spooler"),
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)

    assert prediction.predicted_state.system["process_count"] == 101
    assert {effect.domain for effect in prediction.expected_effects} >= {
        EffectDomain.PROCESS,
        EffectDomain.SERVICE,
        EffectDomain.APPROVAL,
    }


def test_one_shot_command_does_not_claim_known_persistent_process_count() -> None:
    action = Action(
        action_type=ActionType.SHELL_COMMAND, target="", parameters=ShellCommandParams(command="echo ready")
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)
    process_effects = [effect for effect in prediction.expected_effects if effect.domain == EffectDomain.PROCESS]

    assert prediction.predicted_state.system["process_count"] == "requires_observation"
    assert [(effect.operation, effect.before, effect.after) for effect in process_effects] == [
        ("run_transient", 100, "requires_observation")
    ]


def test_unmodeled_read_action_is_honest_about_observation() -> None:
    action = Action(
        action_type=ActionType.PROCESS_INFO,
        target="42",
        parameters=ProcessParams(pid=42),
    )

    prediction = StructuredTransitionPredictor().predict(_state(), action)

    assert prediction.expected_effects[0].domain == EffectDomain.UNKNOWN
    assert prediction.expected_effects[0].after == "requires_observation"


def test_sensitive_shell_command_is_not_copied_into_prediction() -> None:
    action = Action(
        action_type=ActionType.SHELL_COMMAND,
        target="",
        parameters=ShellCommandParams(command="echo TOP_SECRET"),
    )

    payload = StructuredTransitionPredictor().predict(_state(), action).to_dict()

    assert "TOP_SECRET" not in str(payload)
    assert payload["expected_effects"][0]["target"] == "shell_command"


def test_contract_serializes_enum_domains_and_approval_state() -> None:
    action = Action(
        action_type=ActionType.POWER_SHUTDOWN,
        target="system",
        parameters=EmptyParams(),
    )

    payload = StructuredTransitionPredictor().predict(_state(), action).to_dict()

    assert payload["action_type"] == "power_shutdown"
    assert payload["sources"] == ["structured_rules"]
    assert payload["expected_effects"][-1]["domain"] == "approval"
    assert payload["predicted_state"]["system"]["approval_state"] == "required"


def _write_jepa_weights(
    path,
    *,
    validated: bool = False,
    action_types: list[str] | None = None,
    training_samples: int = 128,
    validation_error: float = 0.2,
    geometry: bool = False,
) -> None:
    import numpy as np

    action_count = len(tuple(ActionType))
    payload = {
        "state_weight": np.eye(3, dtype=np.float32),
        "action_weight": np.full((action_count, 3), 0.2, dtype=np.float32),
        "bias": np.zeros(3, dtype=np.float32),
        "action_types": np.array(
            action_types or [action_type.value for action_type in ActionType],
            dtype=np.str_,
        ),
        "model_version": np.array("ui-jepa-test-v1"),
        "training_samples": np.array(training_samples),
        "validation_error": np.array(validation_error),
        "validated_for_gating": np.array(validated),
    }
    if geometry:
        payload["action_geometry_version"] = np.array(UiJepaPredictor.ACTION_GEOMETRY_VERSION)
        payload["action_geometry_weight"] = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    np.savez(path, **payload)


def test_ui_jepa_is_honestly_unavailable_without_weights(tmp_path) -> None:
    predictor = UiJepaPredictor(tmp_path / "missing.npz")

    assert predictor.predict_optional(_state(), _action_for_jepa()) is None
    assert predictor.status()["mode"] == "shadow"
    assert predictor.status()["weights_loaded"] is False


def _action_for_jepa() -> Action:
    return Action(
        action_type=ActionType.BROWSER_CLICK_TEXT,
        target="Learn more",
        parameters=BrowserParams(text="Learn more"),
    )


def test_ui_jepa_predicts_latent_transition_without_pixels(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights)
    state = _state()
    state.ui["latent_embedding"] = [0.1, 0.2, 0.3]

    predictor = UiJepaPredictor(weights)
    prediction = predictor.predict_optional(
        state,
        _action_for_jepa(),
    )

    assert prediction is not None
    assert prediction.sources == ("ui_jepa",)
    assert prediction.risk_evidence == ()
    assert prediction.expected_effects[0].operation == "predict_latent_transition"
    assert "latent_embedding" not in prediction.to_dict()["expected_effects"][0]["after"]
    assert "latent_embedding" not in prediction.to_dict()["predicted_state"]["ui"]
    assert prediction.predicted_state.ui["latent_source"] == "predicted"
    assert prediction.predicted_state.ui["latent_embedding"] != state.ui["latent_embedding"]
    next_prediction = predictor.predict_optional(prediction.predicted_state, _action_for_jepa())
    assert next_prediction is not None
    assert next_prediction.predicted_state.ui["latent_embedding"] != prediction.predicted_state.ui["latent_embedding"]
    assert state.ui["latent_embedding"] == [0.1, 0.2, 0.3]


def test_structured_predictor_does_not_reuse_observed_latent_as_a_next_frame() -> None:
    state = _state()
    state.ui["latent_embedding"] = [0.1, 0.2, 0.3]
    state.ui["latent_source"] = "observed"
    state.ui["latent_transition"] = {"delta_norm": 1.0}

    predicted = StructuredTransitionPredictor().predict(state, _action_for_jepa()).predicted_state

    assert "latent_embedding" not in predicted.ui
    assert "latent_source" not in predicted.ui
    assert "latent_transition" not in predicted.ui
    assert state.ui["latent_embedding"] == [0.1, 0.2, 0.3]


def test_unvalidated_ui_jepa_cannot_add_gating_evidence(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights, validated=False)
    state = _state()
    state.ui["latent_embedding"] = [20.0, -20.0, 20.0]

    prediction = UiJepaPredictor(weights).predict_optional(
        state,
        _action_for_jepa(),
    )

    assert prediction is not None
    assert prediction.risk_evidence == ()


def test_ui_jepa_requires_exact_action_vocabulary(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    reordered = [action_type.value for action_type in ActionType]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    _write_jepa_weights(weights, action_types=reordered)

    predictor = UiJepaPredictor(weights)

    assert predictor.is_loaded is False
    assert predictor.status()["action_vocabulary_bound"] is False
    assert "action vocabulary" in predictor.status()["load_error"]


def test_ui_jepa_gating_requires_quality_metadata(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(
        weights,
        validated=True,
        training_samples=20,
        validation_error=0.8,
        geometry=True,
    )

    predictor = UiJepaPredictor(weights)

    assert predictor.is_loaded is True
    assert predictor.validated_for_gating is False
    assert predictor.status()["mode"] == "shadow"
    assert len(predictor.status()["gating_rejection_reasons"]) == 2


def test_ui_jepa_gating_accepts_bound_validated_artifact(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights, validated=True, geometry=True)

    predictor = UiJepaPredictor(weights)

    assert predictor.validated_for_gating is True
    assert predictor.status()["gating_rejection_reasons"] == []
    assert predictor.status()["gating_action_types"] == ["mouse_click", "mouse_double_click", "mouse_right_click"]


def test_legacy_action_type_only_artifact_cannot_gate_clicks(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights, validated=True)
    predictor = UiJepaPredictor(weights)

    assert predictor.is_loaded is True
    assert predictor.validated_for_gating is False
    assert "cannot distinguish click locations" in predictor.status()["gating_rejection_reasons"][0]


def test_geometry_bound_predictor_distinguishes_click_targets(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights, validated=True, geometry=True)
    predictor = UiJepaPredictor(weights)
    state = _state()
    state.ui.update(latent_embedding=[0.1, 0.2, 0.3], screen_bounds=(0, 0, 100, 100))
    left = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=25, y=25))
    right = Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=75, y=75))

    first = predictor.predict_optional(state, left)
    second = predictor.predict_optional(state, right)
    assert first is not None and second is not None
    assert first.predicted_state.ui["latent_embedding"] != second.predicted_state.ui["latent_embedding"]
    assert predictor.predict_optional(_state(), left) is None
    assert (
        predictor.predict_optional(
            state, Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=150, y=25))
        )
        is None
    )
    assert (
        predictor.predict_optional(state, Action(action_type=ActionType.MOUSE_CLICK, parameters=MouseParams(x=0, y=0)))
        is None
    )
    browser = predictor.predict_optional(state, _action_for_jepa())
    assert browser is not None and browser.risk_evidence == ()


def test_geometry_artifact_requires_exact_feature_contract(tmp_path) -> None:
    import numpy as np

    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights, geometry=True)
    with np.load(weights, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["action_geometry_version"] = np.array("unknown-feature-v1")
    np.savez(weights, **payload)

    predictor = UiJepaPredictor(weights)
    assert predictor.is_loaded is False
    assert "geometry version" in predictor.status()["load_error"]


def test_hybrid_model_fuses_visual_and_structured_predictions(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights)
    state = _state()
    state.ui["latent_embedding"] = [0.1, 0.2, 0.3]
    model = HybridWorldModel(visual=UiJepaPredictor(weights))

    prediction = model.predict(state, _action_for_jepa())

    assert prediction.sources == ("structured_rules", "ui_jepa")
    assert {effect.operation for effect in prediction.expected_effects} >= {
        "mutate_document",
        "predict_latent_transition",
    }
    assert "latent_transition" in prediction.predicted_state.ui


def test_visual_fusion_preserves_structured_window_transition(tmp_path) -> None:
    weights = tmp_path / "ui_jepa.npz"
    _write_jepa_weights(weights)
    state = _state()
    state.ui["latent_embedding"] = [0.1, 0.2, 0.3]
    action = Action(
        action_type=ActionType.WINDOW_FOCUS,
        target="Editor",
        parameters=EmptyParams(),
    )

    prediction = HybridWorldModel(visual=UiJepaPredictor(weights)).predict(state, action)

    assert prediction.predicted_state.ui["active_window"] == "Editor"
    assert prediction.predicted_state.ui["latent_source"] == "predicted"


def test_historical_failure_risk_waits_for_repeated_real_outcomes() -> None:
    history = HistoricalFailureRisk()

    history.record(ActionType.BROWSER_CLICK_TEXT, False)
    history.record(ActionType.BROWSER_CLICK_TEXT, False)
    assert history.score(ActionType.BROWSER_CLICK_TEXT) == (0.0, "")

    history.record(ActionType.BROWSER_CLICK_TEXT, False)
    score, reason = history.score(ActionType.BROWSER_CLICK_TEXT)
    assert score >= 0.3
    assert "3 of 3 observed" in reason
