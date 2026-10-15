"""The export round trip between the service and the evaluation code (spec section 22.7).

One participant runs two study sessions through a real ``CoAdaptationService`` on temporary paths with an
``InMemoryKeyProvider`` and a fake clock. The test then decrypts the export with the analysis script, reads every entry
with its ``from_dict`` and checks it against the store, the verification rules, the agency audit and a rebuild.

Decision ids come from the UI as in the study, and the service draws a fresh seed salt per session, so the regimes of
randomized blocks change from run to run. The driver gives each special case to the first block whose regime fits it:
Phase A-lite runs every allowed arm once per round, so each case finds its block in every run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidTag

from pilot.coadaptation.core.audit import agency_audit
from pilot.coadaptation.core.consent import ConsentPeriod
from pilot.coadaptation.core.constraints import fixed_l2_rule
from pilot.coadaptation.core.engine import EngineEvent
from pilot.coadaptation.core.evaluation import rebuild_engine
from pilot.coadaptation.core.ope import BundleRows, replay_rows
from pilot.coadaptation.core.policy import select_from_cell
from pilot.coadaptation.core.preferences import Preference, preference_from_dict, preference_to_dict
from pilot.coadaptation.core.records import BurdenReport, DecisionRecord, SessionRow, StoredTrial
from pilot.coadaptation.core.study_plan import rank_fixed_regimes
from pilot.coadaptation.core.target_posterior import (
    GazeInput,
    ambiguity_bin,
    cell_id,
    compute_posterior,
    literal_hit,
    rendered_rects,
)
from pilot.coadaptation.core.vocabulary import VARIANTS
from pilot.coadaptation.schemas import FallbacksV1
from scripts.evaluate_coadaptation_study import (
    decrypt_bundle,
    read_human_bundle,
    recompute_posteriors,
    verify_logs,
)
from tests.test_coadaptation_service import (
    ACT_GUARDED,
    RESULT_OF,
    SURFACE,
    Harness,
    Live,
    begin,
    block_context,
    build_trial,
    candidate_rects,
    core_rects,
    decide_block,
    grant_params,
    make_harness,
    pref_params,
    run_block,
    same_state,
)

TAU_1 = 0.25  # the protocol value the first session runs with
TAU_2 = 0.15  # the lower value saved between the sessions
CAP = 1  # the participant's max_interruptions_per_block in session 2
GAZE_ACCURACY = 0.8
FIRST_PURPOSES = ("selection_study", "research_export", "gaze_prior")
SECOND_PURPOSES = (*FIRST_PURPOSES, "adaptive_assistance")
INTERRUPTING = frozenset(
    {"confirm_selection:second_click", "ask_clarification:2", "ask_clarification:3"},
)
ALARMS = frozenset({"drift", "calibration_suspended"})
# Three small candidates 2 px apart: an endpoint near the edge of the first one is a hit above any tau.
TIGHT = [{"cx": 300.0 + 30.0 * i, "cy": 200.0, "w": 28.0, "h": 28.0} for i in range(3)]


# ── driver ───────────────────────────────────────────────────────────────────


def _rect(rect: Any) -> dict[str, float]:
    return {"cx": rect.cx, "cy": rect.cy, "w": rect.w, "h": rect.h}


def _centre(candidate: dict[str, float]) -> tuple[int, int]:
    return int(candidate["cx"] + 0.5), int(candidate["cy"] + 0.5)


def _gap(task: str, candidates: list[dict[str, float]]) -> tuple[int, int]:
    """A press between the first two candidates, on no candidate."""
    if task == "menu":
        return int(candidates[0]["cx"]), int((candidates[0]["cy"] + candidates[1]["cy"]) / 2)
    return int((candidates[0]["cx"] + candidates[1]["cx"]) / 2), int(candidates[0]["cy"])


def _interruptions(activations: list[dict]) -> int:
    """The cap count k a trial adds (section 9.3): L1 confirm and hold steps once per trial, every other step once."""
    l1_steps = [a for a in activations if a["result"] in ("confirming", "held") and a["l2"] is None]
    prompts = ("confirming", "held", "asked")
    others = [a for a in activations if a["result"] in prompts and a not in l1_steps]
    return int(bool(l1_steps)) + len(others)


def _commit(activation: dict, regime: str) -> str:
    """The regime row for an activation that is not ambiguous: commits, confirms or holds its hit."""
    assert activation["hit_index"] is not None
    result, via = RESULT_OF[regime]
    activation.update(result=result, outcome="committed", committed_index=activation["hit_index"])
    return via


def _respond(activation: dict, variant: str) -> None:
    """An L2 response; the person cancels every step it opens."""
    if variant == "act" and activation["hit_index"] is not None:
        activation.update(
            result="committed",
            outcome="committed",
            committed_index=activation["hit_index"],
        )
    elif variant in INTERRUPTING:
        result = "confirming" if variant.startswith("confirm") else "asked"
        activation.update(result=result, outcome="cancelled", committed_index=None)
    else:
        activation.update(result="none", outcome="none", committed_index=None)


@dataclass
class Driver:
    """A study UI for one session: the fallbacks it holds, the cap count per block and the ids worth checking."""

    h: Harness
    live: Live
    fallbacks: dict[str, Any]
    marks: dict[str, Any]
    k: dict[int, int] = field(default_factory=dict)

    @property
    def tau(self) -> float:
        return float(self.live.tau)

    def task_of(self, block_index: int) -> str:
        return next(block["task"] for block in self.live.blocks if block["index"] == block_index)

    def hold(self, reply: dict) -> None:
        if reply.get("fallbacks"):
            self.fallbacks = reply["fallbacks"]

    async def decide(self, block_index: int) -> tuple[dict, dict | None]:
        self.h.clock.advance(seconds=20)
        return await decide_block(self.live, block_index)

    async def record(self, trial: dict, *, verification: str = "ok") -> dict:
        reply = await self.h.svc.record_trial({"session_id": self.live.session_id, "trial": trial})
        assert reply["status"] == "recorded", reply
        assert reply["verification"] == verification, (reply, trial)
        self.hold(reply)
        block = trial["block_index"]
        self.k[block] = self.k.get(block, 0) + _interruptions(trial["activations"])
        return reply

    async def burden(self, block_index: int, regime: str) -> None:
        report: dict[str, Any] = {
            "block_index": block_index,
            "scope": "block",
            "effort": 3,
            "control": 5,
        }
        if regime != "assistance_off":
            report["intrusiveness"] = 2
        reply = await self.h.svc.report_burden(
            {"session_id": self.live.session_id, "report": report},
        )
        assert reply["status"] == "recorded", reply

    async def set_rule(self, kind: str, value: Any, *, modality: str | None = None) -> dict:
        reply = await self.h.svc.set_preference(
            pref_params(kind, value, participant_key=self.live.participant_key, modality=modality)
        )
        assert reply["status"] == "ok", reply
        self.hold(reply)
        return reply

    async def retract(self, pref_id: str) -> dict:
        reply = await self.h.svc.retract_preference({"pref_id": pref_id})
        assert reply["status"] == "ok", reply
        self.hold(reply)
        return reply

    def press(
        self,
        candidates: list[dict],
        spread: dict,
        *,
        index: int,
        regime: str,
        endpoint: tuple[int, int],
        expanded: int | None = None,
        gaze: tuple[str, float] | None = None,
    ) -> tuple[dict, Any]:
        """One pointer activation with the posterior fields the daemon recomputes; the result is set later."""
        scale = float(VARIANTS[regime].param("scale") or 1.0)
        rendered = rendered_rects(core_rects(candidates), expanded, scale)
        press = literal_hit(rendered, endpoint[0], endpoint[1])
        press_index = -1 if press is None else press
        gaze_input = None
        if gaze is not None:
            gaze_input = GazeInput(region=gaze[0], conf=gaze[1], acc=GAZE_ACCURACY)
        posterior = compute_posterior(
            rendered,
            endpoint,
            modality="mouse",
            r=spread["r"],
            floor=spread["floor"],
            surface=SURFACE,
            gaze=gaze_input,
            press_index=press_index,
        )
        hit = posterior.hit_index
        activation = {
            "index": index,
            "t_ms": 300.0 + 150.0 * index,
            "modality": "mouse",
            "x": endpoint[0],
            "y": endpoint[1],
            "hit_index": hit,
            "press_index": press_index,
            "a_gate": posterior.ambiguity_gate,
            "n_plausible": posterior.n_plausible,
            "expanded_index": expanded,
            "hit_rect": None if hit is None else _rect(rendered[hit]),
            "gaze": None if gaze is None else {"region": gaze[0], "conf": gaze[1]},
            "l2": None,
            "irreversible_rule": False,
            "result": "none",
            "outcome": "none",
            "committed_index": None,
        }
        return activation, posterior

    def cell_of(self, activation: dict) -> str:
        bin_id = ambiguity_bin(activation["a_gate"], self.tau) or "b3"
        return cell_id(bin_id, activation["n_plausible"], activation["hit_index"] is not None)

    def routed(self, activation: dict, regime: str) -> bool:
        ambiguous = activation["a_gate"] > self.tau or activation["hit_index"] is None
        return regime in ACT_GUARDED and ambiguous

    def l2(
        self,
        activation: dict,
        *,
        decision_id: str,
        source: str,
        k: int,
        draw_index: int = 0,
        table: dict | None = None,
        trial_index: int = 0,
    ) -> str:
        """Route an ambiguous activation: the cap first (outside a pin_act collapse), then the table draw or the fixed
        rule of the fallbacks held (``fallback`` or ``fixed_rule``). Returns the applied variant."""
        cell = self.cell_of(activation)
        cells = [] if table is None else table["cells"]
        entry = next((c for c in cells if c["cell_id"] == cell), None)
        act_mass = 0.0 if entry is None else entry["propensities"].get("act", 0.0)
        pinned = source == "table" and act_mass == 1.0
        cap = self.fallbacks["interruption_cap"]
        if not pinned and cap is not None and k >= cap:
            variant, used, reason = "wait_for_evidence", "cap", None
        elif source == "table":
            assert table is not None and entry is not None, cell
            row = trial_index - int(table["start_trial_index"])
            variant = select_from_cell(
                entry["order"],
                entry["propensities"],
                float(table["draws"][row][draw_index]),
            )
            used, reason = "table", None
        else:
            variant = fixed_l2_rule(FallbacksV1.parse(self.fallbacks).to_core(), cell)
            used, reason = source, ("late" if source == "fallback" else None)
        activation["l2"] = {
            "decision_id": decision_id,
            "cell_id": cell,
            "draw_index": draw_index,
            "variant": variant,
            "source": used,
            "fallback_reason": reason,
        }
        _respond(activation, variant)
        return variant

    def trial(
        self,
        block_index: int,
        rec: dict,
        trial_index: int,
        *,
        activations: list[dict],
        candidates: list[dict],
        intended: int,
        table_ids: list[str],
        commit_via: str,
        corrections: int = 0,
        issued_at: str | None = None,
        **extra: Any,
    ) -> dict:
        indices = [a["committed_index"] for a in activations if a["committed_index"] is not None]
        committed = indices[-1] if indices else None
        trial = build_trial(
            self.live,
            block_index,
            self.task_of(block_index),
            rec["decision_id"],
            rec["variant"],
            rec["spread"],
            trial_index,
            intended=intended,
            table_ids=table_ids,
            activations=activations,
            candidates=candidates,
            issued_at=issued_at or self.fallbacks["issued_at"],
            committed_index=committed,
            commit_via=commit_via if committed is not None else "none",
            **extra,
        )
        trial["corrections"] = corrections
        return trial

    async def plain_trial(
        self,
        block_index: int,
        rec: dict,
        table: dict | None,
        trial_index: int,
        *,
        regime: str | None = None,
        spread: dict | None = None,
        enter: bool = False,
        verification: str = "ok",
        **extra: Any,
    ) -> dict:
        """A trial whose single activation hits the intended candidate's centre (or presses Enter on it)."""
        task = self.task_of(block_index)
        regime = regime or rec["variant"]
        spread = spread or rec["spread"]
        candidates = candidate_rects(task)
        intended = trial_index % len(candidates)
        if enter:
            trial = build_trial(
                self.live,
                block_index,
                task,
                rec["decision_id"],
                regime,
                spread,
                trial_index,
                intended=intended,
                table_ids=[table["decision_id"]] if table else [],
                keyboard=True,
                issued_at=self.fallbacks["issued_at"],
                **extra,
            )
        else:
            activation, posterior = self.press(
                candidates, spread, index=0, regime=regime, endpoint=_centre(candidates[intended])
            )
            assert posterior.hit_index == intended and not self.routed(activation, regime)
            via = _commit(activation, regime)
            trial = build_trial(
                self.live,
                block_index,
                task,
                rec["decision_id"],
                regime,
                spread,
                trial_index,
                intended=intended,
                table_ids=[table["decision_id"]] if table else [],
                activations=[activation],
                candidates=candidates,
                issued_at=self.fallbacks["issued_at"],
                commit_via=via,
                **extra,
            )
        await self.record(trial, verification=verification)
        return trial

    def ambiguous_hit(self, spread: dict, regime: str) -> tuple[int, int]:
        """An endpoint on the first TIGHT candidate whose A_gate lies above tau."""
        for dx in range(15):
            activation, posterior = self.press(
                TIGHT,
                spread,
                index=0,
                regime=regime,
                endpoint=(300 + dx, 200),
            )
            ambiguous = posterior.ambiguity_gate > self.tau and posterior.n_plausible >= 2
            if posterior.hit_index == 0 and ambiguous:
                return 300 + dx, 200
        pytest.fail("no endpoint on the first candidate lies above tau")


# ── the special blocks of session 2 ──────────────────────────────────────────


async def ordinary_block(
    d: Driver,
    block_index: int,
    rec: dict,
    table: dict | None,
    *,
    trials: int = 2,
) -> None:
    for trial_index in range(trials):
        menu_trial = d.task_of(block_index) == "menu" and trial_index == 1
        enter = menu_trial and "enter_trial" not in d.marks
        trial = await d.plain_trial(block_index, rec, table, trial_index, enter=enter)
        if enter:
            d.marks["enter_trial"] = trial["trial_id"]
    await d.burden(block_index, rec["variant"])


async def enlarge_block(d: Driver, block_index: int, rec: dict, table: dict) -> None:
    """A frozen expansion with gaze, then a veto of an L2 response, the table requested after it and a cap response."""
    task = d.task_of(block_index)
    regime, spread = rec["variant"], rec["spread"]
    candidates = candidate_rects(task)
    d.marks["enlarge_table"] = table["decision_id"]

    # Trial 0: the intended candidate expanded, a gaze sample that keeps the hit unambiguous.
    best: tuple[float, tuple[str, float]] | None = None
    for region in ("center", "left", "right", "up", "down"):
        _activation, posterior = d.press(
            candidates,
            spread,
            index=0,
            regime=regime,
            endpoint=_centre(candidates[0]),
            expanded=0,
            gaze=(region, 0.6),
        )
        if posterior.hit_index == 0 and (best is None or posterior.ambiguity_gate < best[0]):
            best = (posterior.ambiguity_gate, (region, 0.6))
    assert best is not None and best[0] <= d.tau
    activation, _posterior = d.press(
        candidates,
        spread,
        index=0,
        regime=regime,
        endpoint=_centre(candidates[0]),
        expanded=0,
        gaze=best[1],
    )
    via = _commit(activation, regime)
    trial = d.trial(
        block_index,
        rec,
        0,
        activations=[activation],
        candidates=candidates,
        intended=0,
        table_ids=[table["decision_id"]],
        commit_via=via,
        gaze_summary={"agreed_with_intended": True},
    )
    await d.record(trial)
    d.marks["gaze_trial"] = trial["trial_id"]

    # The first row whose draws give an interrupting response on an ambiguous hit.
    endpoint = d.ambiguous_hit(spread, regime)
    probe, _posterior = d.press(TIGHT, spread, index=0, regime=regime, endpoint=endpoint)
    cell = next(c for c in table["cells"] if c["cell_id"] == d.cell_of(probe))
    plan: tuple[int, int, str] | None = None
    for row in range(1, 7):
        for draw in range(4):
            variant = select_from_cell(
                cell["order"],
                cell["propensities"],
                float(table["draws"][row][draw]),
            )
            if variant in INTERRUPTING:
                plan = (row, draw, variant)
                break
        if plan is not None:
            break
    assert plan is not None, "no interrupting draw in the table"
    veto_row, veto_draw, vetoed = plan
    for trial_index in range(1, veto_row):
        await d.plain_trial(block_index, rec, table, trial_index)

    # The veto trial: draws until the interrupting response, "Not this time", then the plain regime.
    assert d.k.get(block_index, 0) == 0
    activations = []
    for draw in range(veto_draw + 1):
        activation, _posterior = d.press(
            TIGHT,
            spread,
            index=draw,
            regime=regime,
            endpoint=endpoint,
        )
        applied = d.l2(
            activation,
            decision_id=table["decision_id"],
            source="table",
            k=0,
            draw_index=draw,
            table=table,
            trial_index=veto_row,
        )
        assert applied == (vetoed if draw == veto_draw else "wait_for_evidence")
        activations.append(activation)
    plain = d.fallbacks["plain"]
    assert plain == "assistance_off"
    after, _posterior = d.press(
        TIGHT,
        spread,
        index=veto_draw + 1,
        regime=plain,
        endpoint=_centre(TIGHT[0]),
    )
    _commit(after, plain)
    activations.append(after)
    trial = d.trial(
        block_index,
        rec,
        veto_row,
        activations=activations,
        candidates=TIGHT,
        intended=0,
        table_ids=[table["decision_id"]],
        commit_via="direct",
        corrections=1,
        veto={"decision_id": table["decision_id"], "level": "ambiguity_response"},
        revert={
            "reason": "veto",
            "from_activation": veto_draw + 1,
            "regime": plain,
            "fallbacks_issued_at": d.fallbacks["issued_at"],
        },
    )
    await d.record(trial)
    d.marks["veto_trial"] = trial["trial_id"]
    d.marks["vetoed_variant"] = vetoed

    # The table requested after the veto: rows from the next trial on, without the vetoed response.
    start = veto_row + 1
    reply = await d.h.svc.recommend(
        {
            "session_id": d.live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "ambiguity_table",
            "context": {
                **block_context(block_index, task),
                "block_decision_id": rec["decision_id"],
                "start_trial_index": start,
                "trials_in_block": 8 - start,
            },
        }
    )
    assert reply["status"] == "ok", reply
    refreshed = reply["recommendation"]
    assert refreshed["start_trial_index"] == start > 0 and len(refreshed["draws"]) == 8 - start
    assert all(c["propensities"].get(vetoed, 0.0) == 0.0 for c in refreshed["cells"])
    d.marks["refreshed_table"] = refreshed["decision_id"]

    # The next trial: a press on a gap at the cap (k = 1), then the intended candidate.
    assert d.k[block_index] == CAP
    gap, _posterior = d.press(
        candidates,
        spread,
        index=0,
        regime=regime,
        endpoint=_gap(task, candidates),
    )
    assert d.routed(gap, regime)
    applied = d.l2(
        gap,
        decision_id=refreshed["decision_id"],
        source="table",
        k=d.k[block_index],
        table=refreshed,
    )
    assert applied == "wait_for_evidence" and gap["l2"]["source"] == "cap"
    hit, posterior = d.press(
        candidates,
        spread,
        index=1,
        regime=regime,
        endpoint=_centre(candidates[1]),
    )
    assert posterior.hit_index == 1 and not d.routed(hit, regime)
    via = _commit(hit, regime)
    trial = d.trial(
        block_index,
        rec,
        start,
        activations=[gap, hit],
        candidates=candidates,
        intended=1,
        table_ids=[refreshed["decision_id"]],
        commit_via=via,
    )
    await d.record(trial)
    d.marks["cap_trial"] = trial["trial_id"]
    await d.burden(block_index, regime)


async def l2_fallback_block(d: Driver, block_index: int, rec: dict, table: dict) -> None:
    """The table arrives late: a press on a gap takes the fixed rule of the fallbacks held; then a late offer."""
    task = d.task_of(block_index)
    regime, spread = rec["variant"], rec["spread"]
    candidates = candidate_rects(task)
    gap, posterior = d.press(
        candidates,
        spread,
        index=0,
        regime=regime,
        endpoint=_gap(task, candidates),
    )
    assert posterior.hit_index is None and d.routed(gap, regime)
    applied = d.l2(
        gap,
        decision_id=table["decision_id"],
        source="fallback",
        k=d.k.get(block_index, 0),
    )
    assert gap["l2"]["source"] == "fallback"
    hit, posterior = d.press(
        candidates,
        spread,
        index=1,
        regime=regime,
        endpoint=_centre(candidates[1]),
    )
    assert posterior.hit_index == 1 and not d.routed(hit, regime)
    via = _commit(hit, regime)
    trial = d.trial(
        block_index,
        rec,
        0,
        activations=[gap, hit],
        candidates=candidates,
        intended=1,
        table_ids=[table["decision_id"]],
        commit_via=via,
        corrections=int(applied in INTERRUPTING),
    )
    await d.record(trial)
    d.marks.update(
        l2_fallback_trial=trial["trial_id"],
        l2_fallback_table=table["decision_id"],
        l2_fallback_block=rec["decision_id"],
        l2_fallback_propensities=dict(rec["propensities"]),
        gap_trial=trial["trial_id"],
    )

    offer = await d.h.svc.recommend(
        {
            "session_id": d.live.session_id,
            "decision_id": str(uuid.uuid4()),
            "level": "offer",
            "context": {
                "block_index": block_index,
                "trial_index": 1,
                "task": task,
                "struggle_kind": "corrections",
                "keyboard_available": True,
            },
        }
    )
    assert offer["status"] == "ok", offer
    offer_id = offer["recommendation"]["decision_id"]
    await d.plain_trial(
        block_index,
        rec,
        table,
        1,
        offers=[
            {
                "decision_id": offer_id,
                "variant": "no_offer",
                "response": "ignored",
                "fallback_reason": "late",
            }
        ],
    )
    d.marks["late_offer"] = offer_id
    await d.burden(block_index, regime)


async def pin_act_block(d: Driver, block_index: int, rec: dict, table: dict) -> None:
    """Under pin_act the table collapses: an ambiguous hit commits what it hits."""
    assert table["mode"] == "pinned", table["mode"]
    regime, spread = rec["variant"], rec["spread"]
    endpoint = d.ambiguous_hit(spread, regime)
    activation, _posterior = d.press(TIGHT, spread, index=0, regime=regime, endpoint=endpoint)
    assert d.routed(activation, regime)
    applied = d.l2(
        activation,
        decision_id=table["decision_id"],
        source="table",
        k=d.k.get(block_index, 0),
        table=table,
    )
    assert applied == "act" and activation["committed_index"] == 0
    trial = d.trial(
        block_index,
        rec,
        0,
        activations=[activation],
        candidates=TIGHT,
        intended=0,
        table_ids=[table["decision_id"]],
        commit_via="direct",
    )
    await d.record(trial)
    d.marks["collapse_trial"] = trial["trial_id"]
    d.marks["collapse_table"] = table["decision_id"]
    await d.burden(block_index, regime)


async def l1_fallback_block(d: Driver, block_index: int, rec: dict) -> None:
    """The regime arrives late: the block runs fallbacks.l1 with the session_begin spread."""
    regime = d.fallbacks["l1"]
    for trial_index in range(2):
        await d.plain_trial(
            block_index,
            rec,
            None,
            trial_index,
            regime=regime,
            spread=d.live.begin_spread(),
            verification="fallback",
            l1_fallback_reason="late",
        )
    d.marks["l1_fallback_decision"] = rec["decision_id"]
    await d.burden(block_index, regime)


async def pause_trial(d: Driver, block_index: int, rec: dict, table: dict | None) -> None:
    """Pause adapting in the middle of a trial: the freeze names the running regime and the trial reverts to it."""
    task = d.task_of(block_index)
    regime, spread = rec["variant"], rec["spread"]
    candidates = candidate_rects(task)
    first, _posterior = d.press(
        candidates,
        spread,
        index=0,
        regime=regime,
        endpoint=_gap(task, candidates),
    )
    applied = None
    if d.routed(first, regime):
        assert table is not None
        applied = d.l2(
            first,
            decision_id=table["decision_id"],
            source="table",
            k=d.k.get(block_index, 0),
            table=table,
        )
    cue_issued_at = d.fallbacks["issued_at"]
    assert regime in d.fallbacks["freezable"], d.fallbacks
    d.h.clock.advance(seconds=1)
    reply = await d.set_rule("freeze_adaptation", {"regime": regime}, modality="mouse")
    assert reply["refresh_required"] is True and reply["fallbacks"]["refresh"] == regime
    hit, posterior = d.press(
        candidates,
        spread,
        index=1,
        regime=regime,
        endpoint=_centre(candidates[1]),
    )
    assert posterior.hit_index == 1 and not d.routed(hit, regime)
    via = _commit(hit, regime)
    trial = d.trial(
        block_index,
        rec,
        0,
        activations=[first, hit],
        candidates=candidates,
        intended=1,
        table_ids=[table["decision_id"]] if table else [],
        commit_via=via,
        corrections=int(applied in INTERRUPTING),
        issued_at=cue_issued_at,
        revert={
            "reason": "refresh",
            "from_activation": 1,
            "regime": reply["fallbacks"]["refresh"],
            "fallbacks_issued_at": reply["fallbacks"]["issued_at"],
        },
    )
    await d.record(trial)
    d.marks.update(
        pause_trial=trial["trial_id"],
        pause_regime=regime,
        freeze=reply["preference"]["pref_id"],
    )


# ── the two sessions ─────────────────────────────────────────────────────────


@dataclass
class Run:
    participant_key: str
    key: bytes
    export_path: Path
    bundle: dict[str, Any]
    rows: BundleRows
    store_rows: Any
    store_rules: list[Preference]
    store_consents: list[ConsentPeriod]
    snapshot: bytes
    marks: dict[str, Any]


async def _session_two(d: Driver) -> None:
    blocks = d.live.blocks
    done: set[str] = set()
    pin: dict | None = None
    for block in [b for b in blocks if b["phase"] == "A_lite"]:
        index = block["index"]
        if {"enlarge", "l2_fallback"} <= done and "collapse" not in done and pin is None:
            pin = await d.set_rule("pin_act", True)
            d.marks["pin_act"] = pin["preference"]["pref_id"]
            d.marks["pin_set_at"] = d.h.clock.now
        rec, table = await d.decide(index)
        d.marks.setdefault("modes", set()).add(rec["mode"])
        variant = rec["variant"]
        if variant in ACT_GUARDED:
            assert table is not None and table["ambiguity_threshold"] == TAU_2
            if "enlarge" not in done and variant.startswith("enlarge_target"):
                await enlarge_block(d, index, rec, table)
                done.add("enlarge")
            elif "l2_fallback" not in done:
                await l2_fallback_block(d, index, rec, table)
                done.add("l2_fallback")
            elif "collapse" not in done and pin is not None:
                await pin_act_block(d, index, rec, table)
                done.add("collapse")
                d.h.clock.advance(minutes=1)
                await d.retract(d.marks["pin_act"])
                d.marks["pin_retracted_at"] = d.h.clock.now
            else:
                await ordinary_block(d, index, rec, table)
        elif "l1_fallback" not in done:
            await l1_fallback_block(d, index, rec)
            done.add("l1_fallback")
        else:
            await ordinary_block(d, index, rec, table)
    assert done == {"enlarge", "l2_fallback", "collapse", "l1_fallback"}, done

    needed = {"adaptive", "probe", "comparator"}
    for block in [b for b in blocks if b["phase"] in ("B", "probe", "comparator")]:
        index = block["index"]
        rec, table = await d.decide(index)
        d.marks["modes"].add(rec["mode"])
        if needed:
            await ordinary_block(d, index, rec, table)
            needed.discard(rec["mode"])
        elif rec["mode"] == "adaptive":
            await pause_trial(d, index, rec, table)
            break
    assert not needed and "pause_trial" in d.marks, (needed, d.marks.get("modes"))


async def _drive(tmp_path: Path) -> Run:
    h = make_harness(tmp_path, adaptive_enabled=True, gaze_prior_enabled=True)
    await h.svc.start()
    try:
        marks: dict[str, Any] = {}
        device_rule = await h.svc.set_preference(
            pref_params("forbid_variant", "ask_clarification:3"),
        )
        assert device_rule["status"] == "ok", device_rule
        marks["device_rule"] = device_rule["preference"]["pref_id"]
        h.clock.advance(minutes=1)
        grant = await h.svc.grant_consent(grant_params(FIRST_PURPOSES))
        assert grant["status"] == "ok", grant
        pk = grant["participant_key"]

        # Session 1: Phase A, randomized, each block 8 trials so the comparator ranking has arms to rank.
        first = await begin(h, pk, gaze_check=GAZE_ACCURACY)
        assert first.tau == TAU_1 and first.reply["gaze_prior"] is True
        main = [b["index"] for b in first.blocks if b["phase"] == "A"]
        for index in main[:4]:
            h.clock.advance(seconds=20)
            await run_block(first, index, trials=8)
        ended = await h.svc.end_session({"session_id": first.session_id, "reason": "completed"})
        assert ended["status"] == "ok"

        # Between the sessions: the purpose change, the lower tau and an interruption cap.
        h.clock.advance(days=1)
        update = await h.svc.grant_consent(grant_params(SECOND_PURPOSES, participant_key=pk))
        assert update["status"] == "ok", update
        previous = dataclasses.replace(h.settings)
        h.settings.ambiguity_threshold = TAU_2
        await h.svc.on_config_changed(previous)
        cap = await h.svc.set_preference(
            pref_params("max_interruptions_per_block", CAP, participant_key=pk),
        )
        assert cap["status"] == "ok", cap
        h.clock.advance(minutes=5)

        second = await begin(h, pk, gaze_check=GAZE_ACCURACY)
        assert second.reply["session_number"] == 2 and second.tau == TAU_2
        assert second.fallbacks["interruption_cap"] == CAP
        driver = Driver(h=h, live=second, fallbacks=second.fallbacks, marks=marks)
        await _session_two(driver)
        ended = await h.svc.end_session({"session_id": second.session_id, "reason": "completed"})
        assert ended["status"] == "ok"
        marks["session_ids"] = (first.session_id, second.session_id)

        exported = await h.svc.export({"participant_key": pk})
        assert exported["status"] == "ok", exported
        key = h.keys.get(pk)
        bundle = decrypt_bundle(Path(exported["path"]), key)
        store = h.svc.store
        snapshot = await store.load_snapshot(pk)
        assert snapshot is not None
        return Run(
            participant_key=pk,
            key=key,
            export_path=Path(exported["path"]),
            bundle=bundle,
            rows=read_human_bundle(bundle),
            store_rows=await store.results_rows(pk, day_index=None),
            store_rules=list(await store.rules(pk)),
            store_consents=list(await store.consent_history(pk)),
            snapshot=snapshot[0],
            marks=marks,
        )
    finally:
        await h.svc.close()


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return asyncio.run(_drive(tmp_path_factory.mktemp("coadapt_roundtrip")))


def _stored(run: Run, trial_id: str) -> StoredTrial:
    return next(item for item in run.rows.trials if item.trial.trial_id == trial_id)


def _decision(run: Run, decision_id: str) -> DecisionRecord:
    return next(record for record in run.rows.decisions if record.decision_id == decision_id)


# ── assertions ───────────────────────────────────────────────────────────────


def test_session_two_ran_every_block_kind(run: Run) -> None:
    first, second = run.rows.sessions
    assert {b.phase for b in first.plan.blocks} >= {"A"}
    assert {"randomized", "adaptive", "probe", "comparator"} <= run.marks["modes"]
    tables = [d for d in run.rows.decisions if d.payload.get("level") == "ambiguity_table"]
    assert any(d.payload["mode"] == "pinned" for d in tables)
    variants = {d.decision_id: str(d.payload.get("variant", "")) for d in run.rows.decisions}
    enlarge = {decision_id for decision_id, v in variants.items() if v.startswith("enlarge_target")}
    assert _decision(run, run.marks["enlarge_table"]).payload["block_decision_id"] in enlarge
    for name in (
        "gaze_trial",
        "veto_trial",
        "cap_trial",
        "gap_trial",
        "collapse_trial",
        "enter_trial",
        "pause_trial",
    ):
        assert _stored(run, run.marks[name]).verification == "ok", name
    assert _stored(run, run.marks["gaze_trial"]).trial.activations[0].gaze is not None
    assert second.gaze_check_accuracy == GAZE_ACCURACY


def test_every_entry_reads_with_from_dict_and_equals_the_store(run: Run) -> None:
    bundle, store = run.bundle, run.store_rows
    for name, cls, stored in (
        ("sessions", SessionRow, store.sessions),
        ("decisions", DecisionRecord, store.decisions),
        ("trials", StoredTrial, store.trials),
        ("burden", BurdenReport, store.burden),
    ):
        assert [cls.from_dict(entry).to_dict() for entry in bundle[name]] == bundle[name], name
        assert bundle[name] == [row.to_dict() for row in stored], name
    events = bundle["events"]
    assert [EngineEvent.from_dict(entry).to_dict() for entry in events] == events
    preferences = bundle["preferences"]
    assert [preference_to_dict(preference_from_dict(entry)) for entry in preferences] == preferences
    assert preferences == [preference_to_dict(pref) for pref in run.store_rules]
    consents = bundle["consents"]
    assert [ConsentPeriod.from_dict(entry).to_dict() for entry in consents] == consents
    assert bundle["consents"] == [period.to_dict() for period in run.store_consents]
    assert bundle["learning_reset_at"] is None


def test_consents_hold_both_periods(run: Run) -> None:
    first, second = run.rows.consents
    assert first.replaced_at is not None and second.replaced_at is None
    assert second.record.expires_at == first.record.expires_at
    assert "adaptive_assistance" not in {str(p) for p in first.record.purposes}
    assert "adaptive_assistance" in {str(p) for p in second.record.purposes}
    assert first.replaced_at <= second.record.granted_at


def test_preferences_hold_the_device_rule_and_the_pin_act_history(run: Run) -> None:
    rules = {pref.pref_id: pref for pref in run.rows.preferences}
    device = rules[run.marks["device_rule"]]
    assert device.scope.participant_key is None and device.retracted_at is None
    assert str(device.kind) == "forbid_variant" and device.value == "ask_clarification:3"
    assert device.created_at < run.rows.consents[0].record.granted_at
    pin = rules[run.marks["pin_act"]]
    assert str(pin.kind) == "pin_act" and pin.scope.participant_key == run.participant_key
    assert pin.created_at == run.marks["pin_set_at"]
    assert pin.retracted_at == run.marks["pin_retracted_at"]
    assert set(rules) == {pref.pref_id for pref in run.store_rules}


def test_events_are_alarms_only(run: Run) -> None:
    assert all(str(event.kind) in ALARMS for event in run.rows.events)


def test_each_session_and_its_tables_carry_the_tau_it_ran_with(run: Run) -> None:
    taus = {}
    for session, tau in zip(run.rows.sessions, (TAU_1, TAU_2), strict=True):
        assert session.settings["ambiguity_threshold"] == tau
        taus[session.session_id] = tau
    tables = [d for d in run.rows.decisions if d.payload.get("level") == "ambiguity_table"]
    assert {d.session_id for d in tables} == set(taus)
    for table in tables:
        assert table.payload["ambiguity_threshold"] == taus[table.session_id]
    refreshed = _decision(run, run.marks["refreshed_table"])
    assert refreshed.payload["start_trial_index"] > 0
    assert _decision(run, run.marks["enlarge_table"]).superseded_by == refreshed.decision_id


def test_pause_adapting_records_a_refresh_revert_to_the_running_regime(run: Run) -> None:
    trial = _stored(run, run.marks["pause_trial"]).trial
    assert trial.revert is not None and trial.revert.reason == "refresh"
    assert trial.revert.regime == trial.applied_regime == run.marks["pause_regime"]
    assert trial.revert.fallbacks_issued_at > trial.fallbacks_issued_at
    freeze = next(pref for pref in run.rows.preferences if pref.pref_id == run.marks["freeze"])
    assert freeze.value == {"regime": run.marks["pause_regime"]}
    assert freeze.scope.modality == "mouse"


def test_seeds_draws_and_applied_responses_re_derive(run: Run) -> None:
    report = verify_logs(run.rows)
    assert report["passed"], report["mismatches"]
    assert report["decisions"] == len(run.rows.decisions)
    activations = [a for item in run.rows.trials for a in item.trial.activations]
    sources = [a.l2.source for a in activations if a.l2 is not None]
    assert {"table", "cap", "fallback"} <= set(sources)
    veto = _stored(run, run.marks["veto_trial"]).trial
    assert veto.veto is not None and veto.veto.decision_id == run.marks["enlarge_table"]
    assert str(veto.veto.level) == "ambiguity_response"
    assert veto.activations[-2].l2.variant == run.marks["vetoed_variant"]


def test_ok_trials_recompute_their_posteriors(run: Run) -> None:
    report = recompute_posteriors(run.rows)
    assert report["passed"], report["mismatches"]
    ok = sum(len(item.trial.activations) for item in run.rows.trials if item.verification == "ok")
    assert report["activations"] == ok
    enter = _stored(run, run.marks["enter_trial"]).trial
    assert enter.task == "menu" and [a.modality for a in enter.activations] == ["keyboard"]
    gaze_trial = _stored(run, run.marks["gaze_trial"]).trial
    gaze = gaze_trial.activations[0]
    assert gaze.expanded_index == 0 and gaze.gaze is not None
    assert gaze.hit_rect is not None and gaze.hit_rect.w > gaze_trial.candidates[0].w


def test_comparator_ranking_comes_from_the_exported_session_one_rows(run: Run) -> None:
    first, second = run.rows.sessions
    trials = [item for item in run.rows.trials if item.trial.session_id == first.session_id]
    burden = [report for report in run.rows.burden if report.session_id == first.session_id]
    decisions = [record for record in run.rows.decisions if record.session_id == first.session_id]
    expected = rank_fixed_regimes(trials, burden, decisions, modality="mouse")
    assert expected
    assert second.comparator_ranking == tuple(expected)


def test_agency_audit_over_the_bundle_passes(run: Run) -> None:
    rows = run.rows
    report = agency_audit(
        rows.decisions,
        rows.trials,
        rows.sessions,
        rows.burden,
        rows.preferences,
        rows.consents,
        events=rows.events,
        learning_reset_at=rows.learning_reset_at,
    )
    assert report.passed, report.findings
    for invariant in ("A1", "A2", "A5", "A6"):
        assert report.counts[invariant] == 0, (invariant, report.findings)


def test_only_the_levels_that_fell_back_are_relabelled(run: Run) -> None:
    baseline = {row.decision_id for row in run.rows.decisions if row.behavior == "baseline"}
    assert baseline == {
        run.marks["l1_fallback_decision"],
        run.marks["l2_fallback_table"],
        run.marks["late_offer"],
    }
    l1 = _decision(run, run.marks["l1_fallback_decision"])
    assert l1.fallback_reason == "late" and l1.applied is not None
    table = _decision(run, run.marks["l2_fallback_table"])
    assert table.fallback_reason == "late" and table.applied is None
    offer = _decision(run, run.marks["late_offer"])
    assert offer.applied == "no_offer"
    assert _stored(run, run.marks["l2_fallback_trial"]).verification == "ok"
    block = _decision(run, run.marks["l2_fallback_block"])
    assert block.behavior == "policy" and block.fallback_reason is None
    assert block.payload["propensities"] == run.marks["l2_fallback_propensities"]


def test_a_rebuild_from_the_exported_rows_equals_the_service_snapshot(run: Run) -> None:
    engine = rebuild_engine(
        replay_rows(run.rows, run.participant_key),
        preferences=run.rows.preferences,
    )
    assert same_state(engine.snapshot(), run.snapshot)


def test_the_bundle_opens_only_with_the_participant_key(run: Run) -> None:
    header = json.loads(run.export_path.read_bytes().splitlines()[0])
    assert header["participant_key"] == run.participant_key and header["cipher"] == "AES-256-GCM"
    assert decrypt_bundle(run.export_path, run.key) == run.bundle
    with pytest.raises(InvalidTag):
        decrypt_bundle(run.export_path, bytes(32))
