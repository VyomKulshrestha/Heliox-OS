/**
 * TypeScript mirrors of the co-adaptation RPC payloads (spec section 14) and the constants the study UI shares
 * with the daemon's vocabulary (section 4). Pure: no Svelte, Tauri, store or DOM access.
 *
 * Field names are the daemon's snake_case names, so a payload built here is sent unchanged.
 */

export type Modality = "mouse" | "touchpad" | "pen" | "touch" | "keyboard" | "hand_pointer";
export type TaskKind = "pointing" | "menu" | "idle";
export type Phase = "practice" | "A" | "A_lite" | "B" | "probe" | "comparator";
export type AmbiguityBin = "b1" | "b2" | "b3";
export type GazeRegion = "center" | "left" | "right" | "up" | "down";
export type PolicyMode = "randomized" | "adaptive" | "probe" | "comparator" | "pinned" | "baseline";

export type L1Variant =
  | "assistance_off"
  | "act"
  | "enlarge_target:1.5"
  | "enlarge_target:2.0"
  | "confirm_selection:second_click"
  | "confirm_selection:hold_400"
  | "confirm_selection:hold_700";
export type L2Variant =
  "act" | "confirm_selection:second_click" | "wait_for_evidence" | "ask_clarification:2" | "ask_clarification:3";
export type L3Variant = "no_offer" | "offer_modality:keyboard" | "offer_modality:pointer";
export type VariantId = L1Variant | L2Variant | L3Variant;

export type CoAdaptationPurpose =
  | "selection_study"
  | "adaptive_assistance"
  | "cross_session_learning"
  | "gaze_prior"
  | "hand_pointer"
  | "research_export";

/** Purposes in enum order (section 10.2); the consent text lists them in this order. */
export const PURPOSES: readonly CoAdaptationPurpose[] = [
  "selection_study",
  "adaptive_assistance",
  "cross_session_learning",
  "gaze_prior",
  "hand_pointer",
  "research_export",
];

export interface AuthorityFields {
  authority: "presentation_only";
  execution_authority: false;
}

// ---- Geometry and device -------------------------------------------------------------------------------------------

export interface RectV1 {
  cx: number;
  cy: number;
  w: number;
  h: number;
}

export interface LayoutSummaryV1 {
  task: TaskKind;
  n_candidates: number;
  w_px: number;
  h_px: number;
  gap_px: number;
  id_bar: number;
  menu_n: number;
  origin_x: number;
  origin_y: number;
  /**
   * Pointing layouts only: mean keyboard presses over the candidates, |dcolumn| + |drow| + 1 from the start candidate
   * (nearest the origin, the lower index on ties), for the decision-time keyboard prior (spec 7.3). Absent otherwise.
   */
  presses_mean?: number;
}

export interface DeviceV1 {
  modality: Modality;
  pointer_types: Array<"mouse" | "pen" | "touch">;
  dpr: number;
  surface_w: number;
  surface_h: number;
  gaze: boolean;
  hand_pointer: boolean;
}

export interface SpreadV1 {
  r: number;
  floor: number;
}

export interface GazeCheckV1 {
  accuracy: number;
}

export interface ActivationGazeV1 {
  region: GazeRegion;
  conf: number;
}

// ---- Trials --------------------------------------------------------------------------------------------------------

export type L2Source = "table" | "fixed_rule" | "fallback" | "cap";
export type FallbackReason = "late" | "rpc_error";
export type L2FallbackReason = "late" | "rpc_error" | "stale_table";
export type ActivationResult = "committed" | "confirming" | "held" | "asked" | "none";
export type ActivationOutcome = "committed" | "cancelled" | "aborted" | "open_at_end" | "none";
export type CommitVia = "direct" | "confirm" | "hold" | "ask" | "none";
export type InterruptionReason =
  | "modal"
  | "hidden"
  | "blur"
  | "resize"
  | "palm_exit"
  | "paused"
  | "daemon"
  | "refused"
  | "refresh"
  | "no_allowed_regime";

export interface L2AppliedV1 {
  decision_id: string;
  cell_id: string;
  draw_index: number;
  variant: L2Variant;
  source: L2Source;
  fallback_reason: L2FallbackReason | null;
}

export interface ActivationV1 {
  index: number;
  t_ms: number;
  modality: Modality;
  x: number;
  y: number;
  hit_index: number | null;
  press_index: number;
  a_gate: number;
  n_plausible: number;
  expanded_index: number | null;
  hit_rect: RectV1 | null;
  gaze: ActivationGazeV1 | null;
  l2: L2AppliedV1 | null;
  irreversible_rule: boolean;
  result: ActivationResult;
  outcome: ActivationOutcome;
  committed_index: number | null;
}

export interface RevertV1 {
  reason: "veto" | "refresh";
  from_activation: number;
  regime: L1Variant;
  fallbacks_issued_at: string;
}

export interface MovementSummaryV1 {
  rt_ms: number;
  mt_ms: number;
  path_ratio: number;
  submovements: number;
  peak_speed_px_s: number;
  reentries: number;
  endpoint_dx_norm: number;
  endpoint_dy_norm: number;
}

export interface OfferResponseV1 {
  decision_id: string;
  variant: L3Variant;
  response: "accepted" | "dismissed" | "ignored";
  fallback_reason: FallbackReason | null;
}

export interface VetoV1 {
  decision_id: string;
  level: "block_regime" | "ambiguity_response";
}

export interface HandSummaryV1 {
  quality_mean: number;
  quality_min: number;
  tracking_losses: number;
}

export interface GazeSummaryV1 {
  agreed_with_intended: boolean | null;
}

export interface TrialV1 {
  trial_id: string;
  block_index: number;
  trial_index: number;
  task: TaskKind;
  regime: L1Variant;
  regime_decision_id: string;
  table_decision_ids: string[];
  applied_regime: L1Variant;
  l1_fallback_reason: FallbackReason | null;
  fallbacks_issued_at: string;
  spread: SpreadV1 | null;
  layout: LayoutSummaryV1;
  candidates: RectV1[];
  surface_w: number;
  surface_h: number;
  dpr: number;
  reversible: boolean[];
  cue_at: string;
  intended_index: number | null;
  modality: Modality;
  activations: ActivationV1[];
  revert: RevertV1 | null;
  committed_index: number | null;
  commit_t_ms: number | null;
  commit_via: CommitVia;
  corrections: number;
  time_ms: number;
  limit_ms: 10000 | 15000 | 30000;
  duration_ms: number;
  movement: MovementSummaryV1 | null;
  veto: VetoV1 | null;
  offers: OfferResponseV1[];
  hand: HandSummaryV1 | null;
  gaze: GazeSummaryV1 | null;
  interrupted: boolean;
  interruption_reason: InterruptionReason | null;
  explanation_views: number;
}

export interface TlxV1 {
  mental: number | null;
  physical: number | null;
  temporal: number | null;
  performance: number | null;
  effort: number | null;
  frustration: number | null;
}

export interface BurdenV1 {
  block_index: number | null;
  scope: "block" | "session";
  effort: number | null;
  intrusiveness: number | null;
  control: number | null;
  tlx: TlxV1 | null;
  predictability: number | null;
  preferred_regime: L1Variant | null;
  skipped: string[];
}

// ---- Decisions -----------------------------------------------------------------------------------------------------

export interface FreezeConflictV1 {
  message: string;
  conflicting: string[];
}

export interface FallbacksV1 {
  l1: L1Variant | null;
  plain: L1Variant | null;
  refresh: L1Variant | null;
  practice: L1Variant | null;
  freezable: L1Variant[];
  freeze_conflict: FreezeConflictV1 | null;
  l2: L2Variant[];
  interruption_cap: number | null;
  issued_at: string;
}

export interface IntervalV1 {
  mean: number;
  lo90: number;
  hi90: number;
}

export interface BurdenPredictionV1 {
  effort: number;
  intrusiveness: number;
  control: number;
}

export interface PredictionV1 {
  variant: string;
  p_complete: IntervalV1;
  p_unintended: IntervalV1;
  ucb_unintended: number;
  p_correction: IntervalV1;
  p_veto: IntervalV1;
  time_ms: IntervalV1;
  burden: BurdenPredictionV1;
  evidence: { trials_person: number; trials_today: number; source: "literature_prior" | "your_history" | "today" };
  authority: "presentation_only";
}

export interface ContrastV1 {
  metric: "slips" | "time" | "corrections" | "effort";
  variant: string;
  baseline: string;
  variant_mean: number;
  baseline_mean: number;
  delta_ci90: [number, number];
}

export interface ExplanationV1 {
  summary_key: string;
  values: Record<string, string | number>;
  contrasts: ContrastV1[];
  excluded: Array<[string, string]>;
  evidence_source: "literature_prior" | "your_history" | "today";
  trials_person: number;
  trials_today: number;
}

export interface PrioritiesV1 {
  fewer_mistakes: number;
  faster: number;
  fewer_interruptions: number;
}

export interface RecommendationV1 extends AuthorityFields {
  decision_id: string;
  level: "block_regime" | "offer";
  mode: PolicyMode;
  variant: string;
  params: Record<string, string | number>;
  eligible: string[];
  excluded: Record<string, string>;
  propensities: Record<string, number>;
  p_thompson: Record<string, number> | null;
  exploration: number;
  fallback: boolean;
  protocol_deviation: boolean;
  round_index: number | null;
  /** L1 only: the pair every posterior of the block uses (6.1); null in a keyboard session. */
  spread?: SpreadV1 | null;
  priorities?: PrioritiesV1;
  predictions: Record<string, PredictionV1>;
  explanation: ExplanationV1 | null;
  seed: string;
  versions: Record<string, string>;
}

export interface TableCellV1 {
  cell_id: string;
  bin: AmbiguityBin;
  n_plausible: number;
  has_hit: boolean;
  order: L2Variant[];
  propensities: Record<string, number>;
  eligible: string[];
  excluded: Record<string, string>;
}

export interface AmbiguityBinV1 {
  id: AmbiguityBin;
  lo: number;
  hi: number;
}

export interface AmbiguityTableV1 extends AuthorityFields {
  decision_id: string;
  level: "ambiguity_table";
  block_decision_id: string;
  mode: PolicyMode;
  ambiguity_threshold: number;
  bins: AmbiguityBinV1[];
  cells: TableCellV1[];
  start_trial_index: number;
  draws: number[][];
  max_draws_per_trial: number;
  after_max: "wait_for_evidence";
  explanation: ExplanationV1 | null;
  seed: string;
  versions: Record<string, string>;
}

// ---- Plan and session ----------------------------------------------------------------------------------------------

export interface BlockPlanV1 {
  index: number;
  phase: Phase;
  task: TaskKind;
  trials: number;
  idle_after: boolean;
  burden_after: boolean;
}

export interface SessionPlanV1 {
  session_number: number;
  end_questionnaire: boolean;
  blocks: BlockPlanV1[];
}

export interface ThresholdsV1 {
  ambiguity: number;
  bins: Record<AmbiguityBin, [number, number]>;
}

export interface SessionBeginParams {
  participant_key: string;
  context: "study";
  consent_text_sha256: string;
  device: DeviceV1;
  protocol_id: string;
  gaze_check: GazeCheckV1 | null;
}

export interface SessionBeginResultV1 {
  status: "ok";
  session_id: string;
  participant_key: string;
  day_index: number;
  session_number: number;
  plan: SessionPlanV1;
  layout_seed: number;
  spread: Partial<Record<Modality, number>>;
  floors: Partial<Record<Modality, number>>;
  thresholds: ThresholdsV1;
  gaze_prior: boolean;
  hand_pointer: boolean;
  burden_prompt_every_blocks: number;
  evidence_kind: string;
  fallbacks: FallbacksV1;
  refresh_seq: number;
}

export interface SessionSummaryV1 {
  trials: number;
  completion: number;
  ua_rate: number;
  idle_ua_per_min: number | null;
  median_time_ms: number | null;
  burden_mean: Record<string, number | null>;
  overrides: number;
  regime_entropy_bits: number;
  switch_rate: number;
  allowed_now: string[];
  excluded_now: Record<string, string>;
  predictions: Record<string, PredictionV1>;
}

// ---- Preferences, config and consent -------------------------------------------------------------------------------

export type PreferenceKind =
  | "forbid_variant"
  | "pin_regime"
  | "pin_act"
  | "min_caution"
  | "max_interruptions_per_block"
  | "priorities"
  | "freeze_adaptation";

export interface PreferenceScopeV1 {
  context: "study" | "all";
  modality: Modality | null;
  participant_key: string | null;
}

export interface PreferenceV1 {
  pref_id: string;
  kind: PreferenceKind;
  scope: PreferenceScopeV1;
  value: unknown;
  origin: "explicit_user";
  created_at: string;
  retracted_at: string | null;
}

export interface PreferenceConflictItemV1 {
  pref_id: string;
  summary: string;
}

/** The section 15 config fields. */
export interface CoAdaptationConfigV1 {
  enabled: boolean;
  adaptive_enabled: boolean;
  learn_across_sessions: boolean;
  gaze_prior_enabled: boolean;
  hand_pointer_enabled: boolean;
  ambiguity_threshold: number;
  unintended_activation_tolerance: number;
  noninferiority_margin: number;
  upper_bound_credibility: number;
  exploration_floor: number;
  switching_penalty: number;
  burden_prompt_every_blocks: number;
  day_boundary_hour: number;
}

/** Numeric config ranges of section 15 (inclusive). The person may only make the ambiguity threshold stricter. */
export const CONFIG_RANGES: Readonly<
  Record<
    | "ambiguity_threshold"
    | "unintended_activation_tolerance"
    | "noninferiority_margin"
    | "upper_bound_credibility"
    | "exploration_floor"
    | "switching_penalty"
    | "burden_prompt_every_blocks"
    | "day_boundary_hour",
    { min: number; max: number; integer: boolean }
  >
> = {
  ambiguity_threshold: { min: 0.05, max: 0.25, integer: false },
  unintended_activation_tolerance: { min: 0, max: 0.2, integer: false },
  noninferiority_margin: { min: 0, max: 0.05, integer: false },
  upper_bound_credibility: { min: 0.8, max: 0.99, integer: false },
  exploration_floor: { min: 0, max: 0.3, integer: false },
  switching_penalty: { min: 0, max: 1, integer: false },
  burden_prompt_every_blocks: { min: 1, max: 8, integer: true },
  day_boundary_hour: { min: 0, max: 23, integer: true },
};

export const DEFAULT_CONFIG: Readonly<CoAdaptationConfigV1> = {
  enabled: false,
  adaptive_enabled: false,
  learn_across_sessions: false,
  gaze_prior_enabled: false,
  hand_pointer_enabled: false,
  ambiguity_threshold: 0.25,
  unintended_activation_tolerance: 0.05,
  noninferiority_margin: 0.01,
  upper_bound_credibility: 0.95,
  exploration_floor: 0.1,
  switching_penalty: 0.1,
  burden_prompt_every_blocks: 1,
  day_boundary_hour: 4,
};

export interface ConsentSummaryV1 {
  consent_id: string;
  purposes: CoAdaptationPurpose[];
  granted_at: string;
  expires_at: string;
  retention_days: number;
  consent_text_version: string;
  consent_text_sha256: string;
  protocol_id: string | null;
  active: boolean;
}

/** The `co_adaptation_consent_grant` params of section 14.2. `label` is sent only for a new participant. */
export interface GrantInput {
  participant_key: string | null;
  label?: string | null;
  purposes: CoAdaptationPurpose[];
  retention_days: number;
  consent_text_version: string;
  consent_text_sha256: string;
  protocol_id: string | null;
  protocol_sha256: string | null;
  authorized: true;
  enable?: boolean;
}

export interface ParticipantModelV1 {
  trials_seen: number;
  sessions_seen: number;
  last_day_index: number | null;
  drift_flags: string[];
  calibration_suspended: boolean;
  learn_across_sessions_effective: boolean;
}

export interface ParticipantStatusV1 {
  participant_key: string;
  label: string | null;
  consent: ConsentSummaryV1;
  model?: ParticipantModelV1;
  fallbacks?: Partial<Record<Modality, FallbacksV1>>;
  sessions: number;
  last_session_at: string | null;
}

export interface StoreSummaryV1 {
  exists: boolean;
  bytes: number;
  participants: number;
  sessions: number;
  decisions: number;
  trials: number;
  burden_reports: number;
  oldest_recorded_at: string | null;
  pruned_sessions: number;
}

export interface ProposalV1 {
  participant_key: string | null;
  kind: "forbid_variant";
  variant: string;
  reason: "three_vetoes";
}

export interface CoAdaptationStatusV1 {
  status: "ok";
  available: boolean;
  config: CoAdaptationConfigV1;
  message?: string;
  keyring_available?: boolean;
  /** True while a revoke, expiry or erase left an export file, a keyring key or a rule to delete later. */
  erasure_pending?: boolean;
  store?: StoreSummaryV1;
  participants?: ParticipantStatusV1[];
  preferences?: PreferenceV1[];
  proposals?: ProposalV1[];
  protocol?: { protocol_id: string; sha256: string };
  versions?: Record<string, string>;
  authority: "presentation_only";
  execution_authority: false;
}

// ---- Results -------------------------------------------------------------------------------------------------------

export interface RegimeResultV1 {
  variant: string;
  trials: number;
  completion: number | null;
  ua_per_100: number | null;
  idle_ua_per_min: number | null;
  corrections_per_trial: number | null;
  median_time_ms: number | null;
  p95_time_ms: number | null;
  throughput_bits_s: number | null;
  burden: { effort: number | null; intrusiveness: number | null; control: number | null };
  verification: Record<string, number>;
}

export interface CalibrationBinV1 {
  p_mean: number;
  observed: number;
  n: number;
}

export interface CalibrationReportV1 {
  brier?: number | null;
  ece?: number | null;
  log_score?: number | null;
  coverage90?: number | null;
  n: number;
  bins?: CalibrationBinV1[];
}

export interface ResultsV1 {
  status: "ok";
  evidence_kind: string;
  claim: string;
  per_regime: RegimeResultV1[];
  calibration: Record<string, CalibrationReportV1>;
  probe_trend: Array<{ session_number: number; throughput_bits_s: number | null; ua_per_100: number | null }>;
  drift_flags: string[];
  allowed_now: { eligible: string[]; excluded: Record<string, string> };
  predictions_now: Record<string, PredictionV1>;
}

// ---- Vocabulary constants (section 4) ------------------------------------------------------------------------------

/** L1 arms per modality (section 4.4). */
export function L1_ARMS(modality: Modality): L1Variant[] {
  if (modality === "keyboard") {
    return ["assistance_off", "act", "confirm_selection:second_click", "confirm_selection:hold_700"];
  }
  const fastConfirm: L1Variant =
    modality === "hand_pointer" ? "confirm_selection:hold_400" : "confirm_selection:second_click";
  return [
    "assistance_off",
    "act",
    "enlarge_target:1.5",
    "enlarge_target:2.0",
    fastConfirm,
    "confirm_selection:hold_700",
  ];
}

export const L2_OPTIONS: readonly L2Variant[] = [
  "act",
  "confirm_selection:second_click",
  "wait_for_evidence",
  "ask_clarification:2",
  "ask_clarification:3",
];

export const L3_OPTIONS: readonly L3Variant[] = ["no_offer", "offer_modality:keyboard", "offer_modality:pointer"];

/** Variants with caution >= 0 in CAUTION_ORDER (section 4.2), least cautious first. */
export const CAUTION_ORDER: readonly string[] = [
  "assistance_off",
  "act",
  "enlarge_target:1.5",
  "enlarge_target:2.0",
  "confirm_selection:second_click",
  "confirm_selection:hold_400",
  "confirm_selection:hold_700",
  "wait_for_evidence",
  "ask_clarification:2",
  "ask_clarification:3",
];

/** i18n key of each variant's plain-language name (coadaptation.en.json). */
export const VARIANT_LABEL_KEYS: Readonly<Record<VariantId, string>> = {
  assistance_off: "coadaptation.variant.assistance_off",
  act: "coadaptation.variant.act",
  "enlarge_target:1.5": "coadaptation.variant.enlarge_target_1_5",
  "enlarge_target:2.0": "coadaptation.variant.enlarge_target_2_0",
  "confirm_selection:second_click": "coadaptation.variant.confirm_second_click",
  "confirm_selection:hold_400": "coadaptation.variant.confirm_hold_400",
  "confirm_selection:hold_700": "coadaptation.variant.confirm_hold_700",
  wait_for_evidence: "coadaptation.variant.wait_for_evidence",
  "ask_clarification:2": "coadaptation.variant.ask_clarification_2",
  "ask_clarification:3": "coadaptation.variant.ask_clarification_3",
  no_offer: "coadaptation.variant.no_offer",
  "offer_modality:keyboard": "coadaptation.variant.offer_keyboard",
  "offer_modality:pointer": "coadaptation.variant.offer_pointer",
};

/** Variant families that a `forbid_variant` rule may name (section 9.6). */
export const VARIANT_FAMILIES = ["confirm_selection", "ask_clarification", "enlarge_target", "offer_modality"] as const;

export function isL1Variant(value: unknown): value is L1Variant {
  return (
    typeof value === "string" &&
    (value === "assistance_off" ||
      value === "act" ||
      value === "enlarge_target:1.5" ||
      value === "enlarge_target:2.0" ||
      value === "confirm_selection:second_click" ||
      value === "confirm_selection:hold_400" ||
      value === "confirm_selection:hold_700")
  );
}

export function isL2Variant(value: unknown): value is L2Variant {
  return typeof value === "string" && (L2_OPTIONS as readonly string[]).includes(value);
}

/** True only for an object that declares itself presentation-only with no execution authority (section 3). */
export function isPresentationOnly(rec: unknown): boolean {
  if (rec === null || typeof rec !== "object") return false;
  const record = rec as Record<string, unknown>;
  return record.authority === "presentation_only" && record.execution_authority === false;
}
