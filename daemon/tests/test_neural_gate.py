from __future__ import annotations

import uuid

import pytest

from pilot.neural.gate import (
    NeuralGateError,
    NeuralIntentGate,
    NeuralIntentGateConfig,
    NeuralIntentSigner,
)
from pilot.neural.protocol import (
    ArtifactFlag,
    NeuralEvidenceKind,
    NeuralIntentClass,
    NeuralIntentV1,
    NeuralParadigm,
    NeuralScope,
    NeuralStreamDescriptorV1,
    NeuralTransport,
    SignalQuality,
)

NOW = 10_000_000_000
CALIBRATION = "b" * 64
SUBJECT = "local-subject"


def _descriptor(session_id: uuid.UUID) -> NeuralStreamDescriptorV1:
    return NeuralStreamDescriptorV1(
        session_id=session_id,
        source_id="synthetic-1",
        board_kind="brainflow-synthetic",
        transport=NeuralTransport.SYNTHETIC,
        evidence_kind=NeuralEvidenceKind.SYNTHETIC,
        sample_rate_hz=250,
        channel_count=3,
        channel_names=("O1", "Oz", "O2"),
        reference="linked-mastoids",
        sequence_start=1,
        started_monotonic_ns=NOW - 5_000_000_000,
    )


def _signed_intent(
    signer: NeuralIntentSigner,
    *,
    session_id: uuid.UUID,
    revision: int,
    sequence: int = 1,
    intent_class: NeuralIntentClass = NeuralIntentClass.SELECT,
    scope: NeuralScope = NeuralScope.NAVIGATE,
    command_id: str | None = None,
    quality: SignalQuality = SignalQuality.GOOD,
    artifacts: tuple[ArtifactFlag, ...] = (),
    intent_id: uuid.UUID | None = None,
) -> NeuralIntentV1:
    intent = NeuralIntentV1(
        session_id=session_id,
        intent_id=intent_id or uuid.uuid4(),
        sequence=sequence,
        window_start_ns=NOW - 300_000_000,
        window_end_ns=NOW - 100_000_000,
        expires_at_ns=NOW + 2_000_000_000,
        paradigm=NeuralParadigm.SYNTHETIC,
        intent_class=intent_class,
        command_id=command_id,
        posterior_permille=900,
        margin_permille=300,
        signal_quality=quality,
        artifact_flags=artifacts,
        dwell_windows=4,
        decoder_version="a" * 64,
        calibration_id=CALIBRATION,
        subject_key=SUBJECT,
        requested_scope=scope,
        state_revision=revision,
        signature="0" * 64,
    )
    return intent.model_copy(update={"signature": signer.sign(intent)})


async def _armed_gate(
    scope: NeuralScope = NeuralScope.NAVIGATE,
) -> tuple[NeuralIntentGate, NeuralIntentSigner, uuid.UUID, int]:
    signer = NeuralIntentSigner(b"n" * 32)
    session_id = uuid.uuid4()
    gate = NeuralIntentGate(
        signer=signer,
        safe_goals={"open-calendar": "Open the calendar and show today's events"},
    )
    await gate.connect(_descriptor(session_id))
    await gate.begin_calibration(session_id)
    await gate.finish_calibration(session_id, calibration_id=CALIBRATION, subject_key=SUBJECT)
    status = await gate.arm(session_id, scope=scope, non_neural_authorized=True)
    return gate, signer, session_id, int(status["state_revision"])


@pytest.mark.asyncio
async def test_signed_preview_and_tier_one_commit_respect_cancellation_window() -> None:
    gate, signer, session_id, revision = await _armed_gate()
    intent = _signed_intent(signer, session_id=session_id, revision=revision)
    preview = await gate.preview(intent, now_ns=NOW)
    assert preview is not None
    assert preview.canonical_goal == "neural_ui.select"
    with pytest.raises(NeuralGateError, match="cancellation window"):
        await gate.commit(
            preview.preview_id,
            expected_revision=preview.state_revision,
            effect_tier=1,
            now_ns=NOW + 100_000_000,
        )
    commit = await gate.commit(
        preview.preview_id,
        expected_revision=preview.state_revision,
        effect_tier=1,
        now_ns=preview.eligible_at_ns,
    )
    assert commit.intent_id == intent.intent_id

    status = await gate.status()
    next_intent = _signed_intent(
        signer,
        session_id=session_id,
        revision=int(status["state_revision"]),
        sequence=2,
    )
    next_preview = await gate.preview(next_intent, now_ns=NOW + 1_900_000_000)
    assert next_preview is not None


@pytest.mark.asyncio
async def test_status_restores_armed_state_after_cooldown() -> None:
    clock = [NOW]
    signer = NeuralIntentSigner(b"n" * 32)
    session_id = uuid.uuid4()
    gate = NeuralIntentGate(
        signer=signer,
        config=NeuralIntentGateConfig(cancellation_window_ns=1, cooldown_ns=100),
        monotonic_ns=lambda: clock[0],
    )
    await gate.connect(_descriptor(session_id))
    await gate.begin_calibration(session_id)
    await gate.finish_calibration(session_id, calibration_id=CALIBRATION, subject_key=SUBJECT)
    armed = await gate.arm(session_id, scope=NeuralScope.NAVIGATE, non_neural_authorized=True)
    intent = _signed_intent(signer, session_id=session_id, revision=int(armed["state_revision"]))
    preview = await gate.preview(intent, now_ns=NOW)
    assert preview is not None
    await gate.commit(
        preview.preview_id,
        expected_revision=preview.state_revision,
        effect_tier=0,
        now_ns=NOW + 1,
    )
    assert (await gate.status())["state"] == "cooldown"
    clock[0] = NOW + 101
    assert (await gate.status())["state"] == "armed_safe_ui"


@pytest.mark.asyncio
async def test_replay_signature_and_stale_revision_fail_closed() -> None:
    gate, signer, session_id, revision = await _armed_gate()
    intent = _signed_intent(signer, session_id=session_id, revision=revision)
    await gate.preview(intent, now_ns=NOW)
    with pytest.raises(NeuralGateError, match="stale controller state|replayed"):
        await gate.preview(intent, now_ns=NOW)
    tampered = intent.model_copy(update={"intent_id": uuid.uuid4(), "sequence": 2, "posterior_permille": 999})
    with pytest.raises(NeuralGateError, match="signature"):
        await gate.preview(tampered, now_ns=NOW)


@pytest.mark.asyncio
async def test_old_evidence_cannot_be_revived_with_a_future_expiry() -> None:
    gate, signer, session_id, revision = await _armed_gate()
    intent = _signed_intent(signer, session_id=session_id, revision=revision)
    old = intent.model_copy(
        update={
            "window_start_ns": NOW - 9_000_000_000,
            "window_end_ns": NOW - 8_000_000_000,
            "expires_at_ns": NOW + 2_000_000_000,
            "signature": "0" * 64,
        }
    )
    old = old.model_copy(update={"signature": signer.sign(old)})
    with pytest.raises(NeuralGateError, match="stale"):
        await gate.preview(old, now_ns=NOW)


@pytest.mark.asyncio
async def test_artifact_abstention_consumes_sequence_and_cannot_be_replayed() -> None:
    gate, signer, session_id, revision = await _armed_gate()
    intent = _signed_intent(
        signer,
        session_id=session_id,
        revision=revision,
        artifacts=(ArtifactFlag.MUSCLE,),
    )
    with pytest.raises(NeuralGateError, match="artifact"):
        await gate.preview(intent, now_ns=NOW)
    with pytest.raises(NeuralGateError, match="replayed"):
        await gate.preview(intent, now_ns=NOW)


@pytest.mark.asyncio
async def test_cancel_has_priority_and_disarms_controller() -> None:
    gate, signer, session_id, revision = await _armed_gate()
    cancel = _signed_intent(
        signer,
        session_id=session_id,
        revision=revision,
        intent_class=NeuralIntentClass.CANCEL,
    )
    assert await gate.preview(cancel, now_ns=NOW) is None
    status = await gate.status()
    assert status["state"] == "observe_only"
    assert status["armed_scope"] == "observe"


@pytest.mark.asyncio
async def test_desktop_goals_require_allowlist_and_never_commit_tier_two() -> None:
    gate, signer, session_id, revision = await _armed_gate(NeuralScope.SAFE_DESKTOP)
    intent = _signed_intent(
        signer,
        session_id=session_id,
        revision=revision,
        intent_class=NeuralIntentClass.SAFE_GOAL,
        scope=NeuralScope.SAFE_DESKTOP,
        command_id="open-calendar",
    )
    preview = await gate.preview(intent, now_ns=NOW)
    assert preview is not None
    with pytest.raises(NeuralGateError, match="Tier 0/1"):
        await gate.commit(
            preview.preview_id,
            expected_revision=preview.state_revision,
            effect_tier=2,
            now_ns=preview.eligible_at_ns,
        )


@pytest.mark.asyncio
async def test_physical_scope_and_neural_arming_are_impossible() -> None:
    signer = NeuralIntentSigner(b"n" * 32)
    session_id = uuid.uuid4()
    gate = NeuralIntentGate(signer=signer)
    await gate.connect(_descriptor(session_id))
    await gate.begin_calibration(session_id)
    await gate.finish_calibration(session_id, calibration_id=CALIBRATION, subject_key=SUBJECT)
    with pytest.raises(NeuralGateError, match="non-neural"):
        await gate.arm(session_id, scope=NeuralScope.NAVIGATE, non_neural_authorized=False)
    with pytest.raises(NeuralGateError, match="physical"):
        await gate.arm(session_id, scope=NeuralScope.PHYSICAL_GOAL, non_neural_authorized=True)


async def _clocked_gate(clock: list[int]) -> tuple[NeuralIntentGate, NeuralIntentSigner, uuid.UUID, int]:
    signer = NeuralIntentSigner(b"n" * 32)
    session_id = uuid.uuid4()
    gate = NeuralIntentGate(signer=signer, monotonic_ns=lambda: clock[0])
    await gate.connect(_descriptor(session_id))
    await gate.begin_calibration(session_id)
    await gate.finish_calibration(session_id, calibration_id=CALIBRATION, subject_key=SUBJECT)
    armed = await gate.arm(session_id, scope=NeuralScope.NAVIGATE, non_neural_authorized=True)
    return gate, signer, session_id, int(armed["state_revision"])


@pytest.mark.asyncio
async def test_a_preview_nobody_committed_lapses_and_the_session_is_armed_again() -> None:
    clock = [NOW]
    gate, signer, session_id, revision = await _clocked_gate(clock)
    preview = await gate.preview(_signed_intent(signer, session_id=session_id, revision=revision), now_ns=NOW)
    assert preview is not None
    assert (await gate.status())["state"] == "previewed"

    clock[0] = preview.expires_at_ns + 1
    status = await gate.status()
    assert status["state"] == "armed_safe_ui" and status["pending_preview"] is None
    with pytest.raises(NeuralGateError, match="missing or no longer current"):  # nothing was committed
        await gate.commit(preview.preview_id, expected_revision=preview.state_revision, effect_tier=1, now_ns=clock[0])
    # The bridge's next intent, decoded against the new revision, is previewed again.
    again = _signed_intent(signer, session_id=session_id, revision=int(status["state_revision"]), sequence=2)
    assert await gate.preview(again, now_ns=NOW + 100_000_000) is not None


@pytest.mark.asyncio
async def test_a_commit_that_came_too_late_does_not_leave_neural_control_stuck() -> None:
    clock = [NOW]
    gate, signer, session_id, revision = await _clocked_gate(clock)
    preview = await gate.preview(_signed_intent(signer, session_id=session_id, revision=revision), now_ns=NOW)
    assert preview is not None
    with pytest.raises(NeuralGateError, match="expired"):
        await gate.commit(
            preview.preview_id,
            expected_revision=preview.state_revision,
            effect_tier=1,
            now_ns=preview.expires_at_ns + 1,
        )
    assert (await gate.status())["state"] == "armed_safe_ui"


@pytest.mark.asyncio
async def test_a_live_preview_is_kept_until_it_lapses() -> None:
    clock = [NOW]
    gate, signer, session_id, revision = await _clocked_gate(clock)
    preview = await gate.preview(_signed_intent(signer, session_id=session_id, revision=revision), now_ns=NOW)
    assert preview is not None
    clock[0] = preview.expires_at_ns
    status = await gate.status()
    assert status["state"] == "previewed" and status["pending_preview"] == str(preview.preview_id)
