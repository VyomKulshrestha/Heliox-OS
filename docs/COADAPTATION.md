# Assistance Co-Adaptation

Heliox can run an opt-in selection study inside its own window. The study measures how one person selects
targets and menu items, tries different kinds of help (larger targets, a confirm step, a clarification prompt),
and estimates which help suits that person on that day. The code lives in `daemon/pilot/coadaptation`. The pure
model and policy (`pilot.coadaptation.core`) is also the core of a small SDK for assisted selection in menus and
pickers that a host renders.

**Claim boundary.** Simulated users only until the pre-registered study reports. Every published result so far
comes from simulated users (`evidence_kind = "simulated"`), and the quick benchmark covers the tuning families
in-sample. No result, document or screen in Heliox says that this feature improves selection for people, and
none may say so before the human study of `daemon/benchmarks/coadaptation_protocol_v1.json` reports.

The feature is off by default (`co_adaptation.enabled = false`), stores nothing without a consent record, and
never changes approvals, permissions or execution.

## Authority contract

Co-adaptation changes how Heliox-rendered targets are presented and what a selection needs from the person. It
never approves, clicks or runs anything. Every recommendation carries `authority = "presentation_only"` and
`execution_authority = false` as constant fields, and no recommendation carries a plan id, an action index, an
`ActionType`, a coordinate command or a callable. Strategies are not action types: no strategy or variant id equals
an `ActionType` value, and `pilot/actions.py`, the verifier registry and `capabilities.json` are unchanged.

Precedence, top to bottom. A lower layer never overrides a higher one.

1. **Authority layer** (unchanged): permissions, the gateway, the risk gate and the approval dialog.
2. **Rule C9, irreversible candidates.** An activation whose literal hit is marked `reversible = false` never
   commits on that activation alone; it gets at least a `confirm_selection:second_click` step. No preference
   (including "Always select exactly what I click") and no learned choice removes it. The commit reducer applies
   it.
3. **Explicit preferences**: origin `explicit_user`, retractable, never written by learning. They live in
   `co_adaptation_preferences.json`; while that file cannot be read, `co_adaptation_session_begin` is refused with
   `preferences_unreadable` until the file is fixed or removed and the daemon restarts.
4. **Deterministic agency rules**: C2 (no learned acting on ambiguity), C3 (only the person's own input commits),
   C4 (approvals untouched), C5 (veto cooldown), C6 (control floor), C7 (calibration suspension), C8 (offer
   cooldown).
5. **Probabilistic safety filter**: a heuristic screen on model predictions (see "What the safeguards are and are
   not").
6. **Learned policy**: Thompson sampling with an exploration floor. At the ambiguity level it chooses only among
   responses that commit nothing.

Commit rules decide when the person's own input is accepted as a selection. They never decide what runs. In
Heliox a committed study target does nothing at all; an SDK host runs only what its own authority layer allows
for a selection the person made, and `presentation_only` means exactly this.

The code enforces the contract by construction and by tests: `pilot/coadaptation/**` imports no authority,
execution, memory, cognitive, multimodal, neural, intelligence, workflow, server or config module (AST tests and a
subprocess import test); the core imports only the standard library and numpy; co-adaptation RPCs leave pending
confirmations and the experience ledger unchanged; no co-adaptation method is added to the neural sidecar or MCP
method sets.

## Vocabulary

A **variant** is a strategy with fixed parameters. A **regime** is the variant that governs a whole block of 8
trials (level L1). An ambiguous activation gets a response (level L2), and a struggle signal can trigger an offer
(level L3).

| Variant | Acting | Interrupts | Caution | Levels |
|---|---|---|---|---|
| `assistance_off` | yes | no | 0 | L1 |
| `act` | yes | no | 1 | L1, L2 |
| `enlarge_target:1.5`, `enlarge_target:2.0` | yes | no | 2, 3 | L1 |
| `confirm_selection:second_click` | no | yes | 4 | L1, L2 |
| `confirm_selection:hold_400`, `confirm_selection:hold_700` | no | yes | 5, 6 | L1 |
| `wait_for_evidence` | no | no | 7 | L2 |
| `ask_clarification:2`, `ask_clarification:3` | no | yes | 8, 9 | L2 |
| `no_offer`, `offer_modality:keyboard`, `offer_modality:pointer` | no | no | none | L3 |

The L1 arms are `assistance_off`, `act`, both enlarge regimes, a fast confirm (`second_click`, or `hold_400` for
the hand pointer) and `hold_700`. Keyboard sessions have no enlarge arms.

**The plain interface.** `assistance_off` is the plain interface: the person's literal click commits, as in every
ordinary UI, and the system adds no interpretation. It runs only in scheduled baseline, probe and Phase A blocks
the person consented to, in the person's own Pause adapting, and as a fallback that the person's rules and C7
allow. No learned, exploratory or fallback choice adds an acting response to an ambiguous activation beyond it:
under `act` and `enlarge_target:*` an ambiguous activation goes to an L2 table of responses that commit nothing,
and only the person's own `pin_act` rule commits its literal hit.

Nothing commits on a timeout, on silence, on gaze, on voice or on an unanswered prompt. Holding a press for
`hold_ms` is a sustained explicit input; losing the input aborts the hold without a commit.

## State

The model never reads stress, attention, cognitive load, mood, screen content, window titles, transcripts or
clipboard data. Decision-time features use layout geometry only (the mean index of difficulty over candidates,
`id_bar`), never the cued target; the realized target enters the physics prior on the update side.

- **Person state** (owned by `CoAdaptationWorldModel`): slow residuals per head and outcome, today's offsets, an
  endpoint spread per modality, idle activation rates, practice counters, menu-item familiarity with a 7-day
  half-life, drift statistics.
- **Interface state** (sent by the surface): layouts, surface size and DPR, current and previous regime.
- **Assistant state** (owned by `CoAdaptationEngine`): modes, versions, settings, calibration suspension, stale
  marks, the permuted round.
- **Task state**: the trial record, summarised after each trial.

The heads use 18 (L1 trial), 12 (L2 episode) and 10 (burden) features in the fixed order of `state.py`.

## Target posterior

For candidates with rendered rects `(cx, cy, w, h)` and an integer endpoint `e`:

```
s_x = r * sqrt((w / 4.133)^2 + floor^2)      s_y = r * sqrt((h / 4.133)^2 + floor^2)
P(i | e) = pi_i N(e; c_i, s) / (pi_0 / (W_surface * H_surface) + sum_j pi_j N(e; c_j, s)),  pi_0 = 0.02
A_hit = 1 - P(hit | e) with a literal hit, 1 without one;   A_gate = max(A_hit with gaze, A_hit without gaze)
```

The floor is 1 px (mouse), 1.5 (touchpad, pen), 3 (touch) and 6 (hand pointer). A literal hit needs the press and
the release on the same rendered candidate; a press on no candidate is recorded as `press_index = -1`. Gaze can
only raise `A_gate`: it reweights the prior of candidates in the gazed region by `min(2, 1 + accuracy * conf)`
and never chooses the expanded candidate. `n_plausible` counts candidates above 0.15, clipped to 1..3, so every
activation falls in one of 12 cells `b{1,2,3}:p{1,2,3}:h{0,1}`.

A keyboard activation (Enter or Space on the focused candidate) records the focused candidate's rounded centre
`floor(c + 0.5)`, `press_index = hit_index` = the focused candidate, `A_hit = A_gate = 0` and `n_plausible = 1`,
and uses no spread.

## Outcome model, priors and updates

Each head and outcome keeps a Gaussian state `theta = [beta; gamma] ~ N(m, P)` over its features, and the linear
predictor is `eta = h + f' theta` with a fixed physics prior `h`.

| Head | Unit | Outcomes | Observation |
|---|---|---|---|
| L1 trial | one active trial | complete, ua, correction, veto | probit |
| L1 trial | same | log_time | Gaussian, sigma 0.35, right-censored at the 10 s or 15 s limit |
| L2 episode | ambiguous activation to trial end | the same five | the same |
| Burden | one block | effort, intrusiveness, control (1-7) | ordinal probit, cutpoints -1.5, -0.9, -0.3, 0.3, 0.9, 1.5 |
| Idle | one 30 s epoch | commits | Gamma-Poisson per modality and acting class |
| Offer | one offer | accepted, dismissed | Beta(1, 4) each |

The physics priors come from Fitts' law (`T = 100 + b_m * ID` ms; `b` 230 mouse, 330 touchpad, 250 pen and touch,
600 hand pointer), a reaction time of 250 ms, the search-decision-pointing menu model with expertise
`1 - exp(-n_item_eff / 5)`, a practice multiplier `1 + 0.3 (n + 1)^-0.4`, keyboard key-press times and a
two-candidate ambiguity band. `PRIOR_TABLE_V1` lists every constant with its spread and reference.

Updates: Kalman for completed times, assumed-density filtering for censored times, the probit update of Graepel et
al. (2010), an ordinal-probit moment update, conjugate Gamma-Poisson counts and an inverse-gamma spread. Exactly
the trials verified `ok` update the filters, the spread, the idle rates, the drift detectors and the C7 buffer;
`fallback` trials advance only the practice counters; interrupted trials, skipped ratings, ignored offers and
mismatched trials change nothing. Every update uses the timestamp stored with its row, and learning uses no random
numbers.

Predictions never change model state: diffusion is applied to a copy, so any number of predictions, explanations
or Thompson draws leaves the snapshot bytes unchanged. `E[intrusiveness | assistance_off] = 1` by definition,
because "Did Heliox's help get in your way?" has no meaning without help.

## Dynamics

- **Day boundary**: a session whose stored `day_index` differs from the last one inflates each slow variance by
  `Q_slow * min(days, 30)` (0.03^2 per day for time, 0.08^2 for probit and ordinal), resets today's offsets,
  multiplies idle statistics above the prior by 0.7, clears the calibration suspension and the drift flags and
  restarts every drift statistic. The core never reads a clock or a time zone: `day_index_for(local_dt,
  boundary_hour)` uses the offset already on `local_dt`, and the service calls it once per session.
- **Within a day**: today's variances diffuse by 0.01^2 (time) or 0.02^2 (probit, ordinal) per minute, capped at
  the prior.
- **Device change** (modality, DPR or surface size class) resets today's offsets only.
- **Drift**: ten detectors (five outcomes of the L1 and of the L2 head) run risk-adjusted Bernoulli CUSUMs (odds
  ratios 2 and 1/2) and Gaussian mean-shift CUSUMs of +-0.5 predictive SD, with censored times scored by
  `ln Phi(delta - c) - ln Phi(-c)`. The thresholds `{"binary": 7.65, "log_time": 8.55}` come from simulated
  in-control run lengths. An alarm widens the slow block of that filter to at most its prior variance, keeps every
  correlation, flags the day and marks open decisions stale.
- **Learning across sessions off**: the model, spread and practice counters restart from the prior at every
  session. The analysis variant `SESSION_ONLY` restarts the filters, spread, idle rates and drift statistics but
  keeps the practice counters and menu familiarity.

## Policy and constraints

Every decision follows the same steps: candidates and applicability; hard constraints C1-C8 before any model
output is read; the safety filter in adaptive mode only; a fallback when nothing survives; the choice by mode with
the full propensity vector logged (zeros included); random numbers only from the decision's seed.

- **C1 explicit preferences**: forbids remove variants or families; overlapping `min_caution` rules combine to the
  strictest and overlapping caps to the smallest; a pin or freeze collapses L1 to its regime; `pin_act` collapses
  each L2 cell to `act` (with a hit) or `wait_for_evidence` (without one) and removes the confirm regimes at L1; a
  cap below the block's 8 trials removes the confirm regimes unless a minimum caution leaves nothing else.
- **C2**: `act` gets propensity 0 in every L2 table outside a `pin_act` collapse, and the plain interface is never
  in the adaptive L1 set.
- **C5**: a vetoed variant is excluded at the next decision of its level; a veto of a decision made in pinned
  mode adds no exclusion, and `wait_for_evidence` has no veto.
- **C6**: a regime rated with mean control below 3.0 and an 80% upper bound below 3.5 is excluded for the rest of
  the session; the bound is the closed form `E[y | eta = mu + 1.2816 s]`.
- **C7**: acting regimes leave adaptive decisions and every fallback outside a freeze or pin (see the safeguards).
- **C8**: no offers for 10 trials after a dismissed offer.

Modes: `randomized` (Phase A and A-lite, permuted rounds, propensity `1/|remaining arms|`), `adaptive` (Phase B,
`pi = (1 - epsilon) p_TS + epsilon/|S|` with epsilon 0.1 and 256 Thompson draws), `probe` (`assistance_off`),
`comparator` (the participant's best fixed assisted regime from session 1), `pinned` (a freeze or pin) and
`baseline` (the practice block). Probe and pinned blocks use the fixed L2 rule and make no offers.

L2 responses are pre-drawn: each block's table holds uniforms `u[k][j]` for its trials and four draws per trial,
and the surface applies `select_from_cell(order, propensities, u)`. After four draws a trial gets
`wait_for_evidence`.

**Fallbacks** (`constraints.fallbacks`) come from the preferences and the C7 state only, never from a model
output: `l1` is the frozen or pinned regime while one is in force for the modality, else the most cautious regime
the rules allow (and, during a suspension, not acting); `plain` is `assistance_off` only when the rules allow it
and C7 has not suspended acting regimes, else that most cautious regime; `refresh` and `practice` keep a frozen or
pinned regime and otherwise equal `plain`; `l2` ends with `wait_for_evidence`; `freezable`, `freeze_conflict`,
`interruption_cap` and `issued_at` complete the object. When `l1` is null the block does not start.

**Utility.** The policy optimises the person's priorities (fewer mistakes `m`, faster `f`, fewer interruptions
`i`): `w_UA = 2 + 2m`, `w_K = 0.5 + 0.5m`, `w_T = 0.3 + 0.3f`, `w_B = 0.15 + 0.35i`, `w_V = 0.5 + 0.5i`. The
study scores what happened with the realized utility of observed outcomes only, under each block's logged
priorities and without the switching penalty.

## Consent and data sheet

Consent is typed, purpose-bound, expiring and revocable. The purposes are `selection_study` (required),
`adaptive_assistance`, `cross_session_learning`, `gaze_prior`, `hand_pointer` and `research_export`; there is no
approval purpose because nothing about approval dialogs is stored.

- **Participants.** Keys are daemon-generated (`p-` and 8 hex characters from `secrets.token_hex(4)`); the UI can
  never choose one. An optional display label is stored only encrypted with that participant's key. Anyone who
  uses the device can open Settings and view or erase every participant; on a shared lab machine the operator
  controls access to the device.
- **Storage.** `co_adaptation.db` under `DATA_DIR`, separate from the experience ledger. Payload columns hold
  AES-256-GCM ciphertext with a per-participant key in the OS keyring. Every write checks the participant's
  active consent and the purpose of each field first.
- **Fields stored per trial**: the trial record (layout, cue-time candidate rects, activations with integer
  endpoints, `press_index`, the frozen `expanded_index` and the hit's rendered rect, the gaze region and
  confidence used for each posterior, the immediate `result` and final `outcome`, L2 responses with their source),
  plus what verification and replay need: the spread pair `{r, floor}`, the session's gaze-check accuracy (in the
  session row), the verification result, the derived labels, the episode contexts, the session's `day_index` and
  the regime's predicted upper bound on unintended activations. Movement is stored as summaries; raw pointer paths,
  video, landmarks, gaze samples, typed text and screen content are never stored.
- **Study gaze never reaches the ledger**: while a study session is active the UI sends no `gaze_event` RPC.
- **Retention and erasure.** The consent duration (30, 90, 180 or 365 days) sets retention. Expiry and revocation
  erase every row, every export file and the keyring key, and delete the rules scoped to that participant;
  device-wide rules stay until the person removes them. Erase-today deletes the day's sessions with their rows and
  every export written since the erased day's first session started.
- **When expiry erases.** An expired consent is refused at once for sessions, results and export. The erasure runs
  at the next service start, `co_adaptation_status`, `co_adaptation_session_begin` or call on that participant's
  open session, and otherwise from a prune every 24 hours (`PRUNE_INTERVAL_S`). If Heliox keeps running and nobody
  opens Settings or the study, expired rows, exports and the key can remain for up to 24 hours after the period
  ends. Revocation erases at once.
- **The feature turns off with the last consent.** When an expiry erasure leaves no active consent (at the next
  service start, status call or prune), the daemon turns `co_adaptation.enabled` off, as a revocation does, and
  broadcasts `config_updated`. A revocation that fails part-way also turns it off: the consent is gone even when
  the erasure is not finished. A grant made after a consent expired but before
  the erasure ran starts from off, so `enable: false` keeps the feature off.
- **What crypto-shredding does not cover.** `secure_delete` overwrites deleted rows, and destroying the key makes
  the encrypted columns of leftover pages (backups, WAL copies) unreadable, but plaintext columns of such pages
  (ids, timestamps, enum values) stay readable without the key. An export copied elsewhere by the person is
  outside Heliox's reach.
- **Erasure that cannot finish at once.** While the keyring cannot be reached (the service then starts
  degraded), revocation and the expiry prune still delete every row and export file, but the key stays in the
  keyring; the revoke reply carries `key_destroyed: false` and `pending: true`. An export file another program
  holds open is counted in `export_files_left` the same way. What is left goes on the pending-erasure list
  `co_adaptation_pending_erasure.json` next to the database (pseudonymous participant keys and export file
  names only), which the service retries at every start, status call and prune and on a repeated revocation;
  `co_adaptation_status` reports `erasure_pending: true` while the list is not empty. Until the key is deleted it still decrypts the encrypted columns of backups and leftover pages and any copied
  export, so crypto-shredding takes effect only once the keyring can be reached again.
- **Consent updates** keep the earlier record with `replaced_at`, never extend the original expiry (the new expiry
  is `min(old expiry, grant + retention)`), and delete what a removed purpose stored: export files for
  `research_export`, gaze fields for `gaze_prior`, hand summaries for `hand_pointer`.
- **Rule history.** Each participant's store keeps every rule that applied to them, with `created_at` and
  `retracted_at`, and the export carries it.
- **Browser storage.** Co-adaptation code keeps nothing in `localStorage` or other browser storage.

**Export bundle** (`research_export`; encrypted, local only):

```json
{"bundle_schema": 1, "record_schema": "coadapt-records-v1",
 "consents": [ConsentPeriod], "sessions": [SessionRow], "decisions": [DecisionRecord],
 "trials": [StoredTrial], "burden": [BurdenReport], "events": [EngineEvent],
 "preferences": [Preference], "learning_reset_at": "ISO-8601 or null",
 "versions": {"strategy", "feature_schema", "model", "prior", "policy", "sdk"}}
```

Every entry is a record's `to_dict()`; floats are written unrounded and read back bit-identical with
`from_dict()`.

## Evidence status

| `evidence_kind` | Status |
|---|---|
| `simulated` | Contract, policy-property, agency-audit and simulated-user tests. The registered quick benchmark is "tuning families, in-sample". |
| `live_human`, `recorded_human` | No human study has reported. |

With 16 to 32 Phase A trials per arm per person, per-person differences between regimes rest mostly on the prior:
separating 4% from 8% slips for one person needs about 550 trials per arm. The claims the study can test are
group-level: H8 (person updates improve prediction across the group) and H9 (carrying the learned state across
sessions improves early-session prediction; both models keep the practice and menu-familiarity counts, so H9 tests
the carry-over of learned state, not exposure). The published simulated benchmark covers the tuning families
in-sample unless a holdout result is cited.

Known failure modes: verification needs exact agreement between the surface and the daemon, so a rounding change,
a rect drawn differently or a spread not echoed turns trials into mismatches that are stored but never learned;
blocks end early after rule changes and refusals; a person whose rules and C7 leave no allowed regime cannot start
a block; a host that marks every candidate reversible defeats C9.

## SDK

**Scope.** Assisted selection in menus and pickers that the host renders.

**Public API** (`pilot.coadaptation.core`, version `SDK_VERSION = "0.1.0"`): `CoAdaptationEngine` and
`EngineSettings`; the records of `records.py` with `to_dict()`/`from_dict()`; `ConsentRecord`, `ConsentPeriod` and
`consent_in_force`; `PreferenceSet` and the preference types; `fallbacks`; `select_from_cell`; and the conformance
check `agency_audit` in `pilot.coadaptation.core.audit`.

**Dependency rules.** The standard library and numpy only.

**Host contract.**

1. Obtain a consent record before storing anything, and check its purposes.
2. Build contexts without ground truth: no cued target, no labels at decision time.
3. Mark every candidate with a deterministic `reversible: bool` that no learned component sets, and keep rule C9
   above the learned layer, so an irreversible candidate always needs at least a second explicit input.
4. Apply presentation only: the engine's output changes how candidates are drawn and what a selection needs.
5. Report outcomes with the reasons for missing data (interrupted, skipped, ignored).
6. Persist rows through the host's own store with the records' `to_dict()`.
7. Run the audit, A12 included, in CI.

Commit rules decide when the person's own input is accepted as a selection and never decide what runs: the host's
own authority layer decides that.

**Intended use.** Nonclinical. The SDK does not diagnose, monitor or treat, and does not infer health, cognitive
or emotional state.

**Prohibited uses.** Clinical claims; permission or approval decisions; acting on a person's behalf; covert
collection; letting any learned, exploratory or fallback choice add an acting response to an ambiguous activation
beyond the plain interface; committing an irreversible selection on one input. The plain interface is
`assistance_off`, in which the person's literal click commits as in every ordinary UI and the system adds no
interpretation; a host may run it only in scheduled baseline, probe or randomized blocks the person consented to,
in the person's own pause, and as a fallback that the person's rules and C7 allow.

## Replay and verification

- **Predictions never mutate state**, and learning uses no random numbers.
- **Replay learns only from verified rows**: `ok` rows update everything, `fallback` rows advance the practice
  counters, every other verification value changes nothing, live and in replay alike.
- **Events are stored and re-applied.** Drift and calibration-suspension alarms fire only live; each becomes an
  event row written with the trial that fired it, and a rebuild re-applies it right after that trial. The detectors
  are fed during replay but never fire, so a rebuild restores a suspension or a drift inflation exactly.
- **Day indices are stored**: each session stores its `day_index`, and day boundaries come only from it.
- **Settings are fixed per session.** A session runs and replays with exactly the settings stored in its row
  (`EngineSettings(**row.settings)`, including tau); a config change applies from the next session.
- **Study sessions follow the protocol.** `exploration_floor` and `switching_penalty` always take the protocol's
  values (0.1 each), and `ambiguity_threshold`, `unintended_activation_tolerance`, `noninferiority_margin` and
  `upper_bound_credibility` take the stricter of config and protocol, so a looser config value is saved but has no
  effect. `protocol_deviations` in the session row names every setting whose config value differs from the
  protocol, also when the effective value is unchanged.
- **Endpoints and presses.** Pointer endpoints are integers before any posterior. A press on no candidate is
  `press_index = -1`, which means no literal hit. A keyboard activation carries the focused candidate's rounded
  centre, `A_hit = 0` and no spread, and never updates the spread.
- **Fallbacks are recorded per level.** An L1 fallback makes the trial `fallback`; an L2 fallback (a late or stale
  table) and an L3 fallback (a late offer) relabel only their own decision and leave the trial `ok`. A stale-table
  fallback names the table requested after a veto, which had not arrived yet; when the same trial then applies that
  table's draws, the table stays a policy decision, and off-policy evaluation keeps the applied draws of a table
  relabelled only for a stale table.
- **Result versus outcome.** An activation's `result` is its immediate response, set once; its `outcome` is what
  finally happened (committed, cancelled, aborted, open at the end). Labels read the outcome; the cap recount reads
  the result.
- **The interruption cap** is applied per activation by the host's commit reducer and verified by the daemon,
  which recounts each block from every activation's `result`, `l2` and `irreversible_rule`: a trial's L1 confirm and
  hold steps count at most once, however often the person cancels or releases and tries again; every L2 confirm,
  C9 confirm and clarification prompt counts once each. A `cap` response below the cap, or any other L2 response
  outside a `pin_act` collapse at or above it, is a draw mismatch. `coadaptation_cap_recount.json` is shared with
  the UI reducer's tests.
- **A rule counts from the moment the host held it.** Each trial echoes the `fallbacks_issued_at` of the fallbacks
  it held at cue, and a revert records the issue time of the fallbacks whose regime it applied. Audit A1 judges a
  decision's regime at the decision's time and a fallback, fixed-rule, cap or revert choice at that issue time.
- **A refresh keeps a frozen or pinned regime** (`fallbacks.refresh`); `plain` serves only "Not this time".
- **One refresh per change.** Each change carries a `refresh_seq`; the host reverts or interrupts a trial at most
  once per value. That deduplication covers only the trial-level revert: a refresh that arrives with no trial in
  progress, or a stale block found later, ends the block before the next trial.
- **Rule conflicts are rejected**, freezes and pins included (also two pins or two freezes naming different
  regimes), with the conflicting rule named and an offer to retract it; no rule is ever rewritten. Overlapping
  minimum cautions and caps combine to the strictest. `fallbacks.freeze_conflict` names the rule that blocks a
  pause.
- **The spread window.** The spread update uses only first activations whose offsets lie within +-3 predicted SDs
  per axis at the prior-mean spread, and divides each included `z^2` by `1 - 2c phi(c) / (2 Phi(c) - 1) = 0.97334`
  at c = 3. This is unbiased for a mouse, but with the hand pointer the window is wider than the candidate pitch,
  so selection errors that land on a neighbour raise the estimate by a few percent (about 3% to 5% at the prior
  spread, up to about 16% for a steadier person with adjacent errors).
- Replaying a participant's stored rows reproduces the live engine's snapshot bytes exactly.

## What the safeguards are and are not

- **The safety filter is a heuristic screen with no error-rate guarantee.** It removes a variant when its upper
  credible bound on unintended activations exceeds `max(0.05, baseline mean + 0.01)`. It compares an upper bound
  with the baseline's mean, ignores the baseline's own uncertainty and the correlation between the two
  predictions, and admits a variant whose bound is up to 5% absolute whenever the baseline predicts less than 4%,
  which is looser than the +2 percentage-point margin of hypothesis H6.
- **C7 is a tripwire for gross miscalibration, not a calibration guarantee.** With 50 to 100 buffered trials, the
  exact binomial test detects a true unintended-activation rate of twice the mean predicted upper bound with
  probability of about 14-21% when that bound is 0.02, 38-55% at 0.05 and 56-87% at 0.10. A suspension removes the
  acting regimes from adaptive decisions and from every fallback outside a freeze or pin until the next day
  boundary or learning reset.
- **The drift detectors have the false-alarm rate they were calibrated for and little power for rare outcomes.**
  Each detector raises a false alarm within a 200-observation day with probability at most 0.005 (about 5% of days
  across the ten detectors by the union bound). A fourfold increase in the odds of an unintended activation from
  observation 100 of a 200-observation day is caught that day with probability about 0.96 at a base rate of 0.10,
  0.54 at 0.05 and 0.03 at 0.02; a shift of 0.5 SD in log time, about 0.90.
- **The utility weights let speed offset slips.** Halving trial time outweighs a rise in unintended activations of
  up to 5.2 percentage points (m = 1, f = 0) to 20.8 points (m = 0, f = 1), 10.4 points at the default sliders.
- **The hard limits are C2, C9 and the person's rules.** None of the probabilistic layers is an authority control.
