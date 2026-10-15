import { call } from "../api/daemon";
import type {
  AmbiguityTableV1,
  BurdenV1,
  CoAdaptationConfigV1,
  CoAdaptationStatusV1,
  ConsentSummaryV1,
  FallbacksV1,
  GrantInput,
  LayoutSummaryV1,
  PreferenceConflictItemV1,
  PreferenceKind,
  PreferenceScopeV1,
  PreferenceV1,
  RecommendationV1,
  ResultsV1,
  SessionBeginParams,
  SessionBeginResultV1,
  SessionSummaryV1,
  TaskKind,
  TrialV1,
} from "./types";

/**
 * One typed function per co-adaptation RPC (spec section 14). Each takes the RPC's params object unchanged and
 * calls the daemon with the literal method name, so the RPC surface wiring test can find every call site.
 *
 * Nothing here concerns approvals: no co-adaptation RPC addresses an approval request.
 */

export interface RpcReply {
  status: string;
  reason?: string;
  message?: string;
}

/** Refused or ignored replies: `reason` names why (section 14.3). */
export interface RpcNotOk {
  status: "refused" | "ignored" | "error";
  reason?: string;
  message?: string;
}

export interface RefreshFields {
  refresh_required: boolean;
  refresh_seq: number | null;
  fallbacks: FallbacksV1 | null;
}

export type ConfigUpdateReply = { status: "ok"; config: CoAdaptationConfigV1; restart_required: boolean } | RpcNotOk;

export type ConsentGrantReply =
  | {
      status: "ok";
      participant_key: string;
      consent: ConsentSummaryV1;
      config: CoAdaptationConfigV1;
      removed: { purposes: string[]; export_files: number; trials_rewritten: number; sessions_rewritten: number };
    }
  | RpcNotOk;

export type ConsentRevokeReply =
  | {
      status: "ok";
      erased: {
        rows: number;
        /** True when no keyring key exists for the participant any more; false while its deletion is pending. */
        key_destroyed: boolean;
        export_files: number;
        rules_deleted: number;
        /** Export files that could not be deleted yet (on the daemon's pending-erasure list). */
        export_files_left?: number;
        /** True while a key, an export file or a rule still awaits deletion; the daemon retries it. */
        pending?: boolean;
      };
      config: CoAdaptationConfigV1;
    }
  | RpcNotOk;

export interface PreferenceSetParams {
  kind: PreferenceKind;
  scope: PreferenceScopeV1;
  value: unknown;
}

export type PreferenceSetReply =
  | ({ status: "ok"; preference: PreferenceV1; notes: string[] } & RefreshFields)
  | { status: "error"; reason: "preference_conflict"; message: string; conflicting: PreferenceConflictItemV1[] }
  | RpcNotOk;

export type PreferenceRetractReply = ({ status: "ok" } & RefreshFields) | RpcNotOk;

export type LearningResetReply = { status: "ok"; model: Record<string, unknown>; session_ended: boolean } | RpcNotOk;

export type EraseReply =
  { status: "ok"; erased: { rows: number; export_files: number }; session_ended: boolean } | RpcNotOk;

export type ResultsReply = ResultsV1 | RpcNotOk;

export type ExportReply = { status: "ok"; path: string; sha256: string; bytes: number } | RpcNotOk;

export type SessionBeginReply = SessionBeginResultV1 | RpcNotOk;

export type SessionResumeReply =
  { status: "ok"; refresh_required: boolean; refresh_seq: number; fallbacks: FallbacksV1 } | RpcNotOk;

export interface BlockRegimeContext {
  block_index: number;
  layouts: LayoutSummaryV1[];
  previous_regime: string | null;
  session_minute: number;
  trials_today: number;
  hand_quality_recent: number | null;
}

export interface AmbiguityTableContext extends BlockRegimeContext {
  block_decision_id: string;
  start_trial_index: number;
  trials_in_block: number;
}

export interface OfferContext {
  block_index: number;
  trial_index: number;
  task: TaskKind;
  struggle_kind: "corrections" | "tracking_loss";
  keyboard_available: boolean;
}

export type RecommendParams =
  | { session_id: string; decision_id: string; level: "block_regime"; context: BlockRegimeContext }
  | { session_id: string; decision_id: string; level: "ambiguity_table"; context: AmbiguityTableContext }
  | { session_id: string; decision_id: string; level: "offer"; context: OfferContext };

export type RecommendReply =
  | { status: "ok"; logged: true; recommendation: RecommendationV1 | AmbiguityTableV1 }
  | {
      status: "ignored";
      reason: string;
      logged: false;
      recommendation: Partial<RecommendationV1> | Partial<AmbiguityTableV1>;
    }
  | RpcNotOk;

export type RecordTrialReply =
  | ({
      status: "recorded";
      verification: "ok" | "draw_mismatch" | "posterior_mismatch" | "interrupted" | "fallback";
      labels: { complete: boolean; ua: boolean; correction: boolean; veto: boolean; time_ms: number } | null;
      preference_violation: boolean;
    } & RefreshFields)
  | RpcNotOk;

export type ReportBurdenReply = { status: "recorded" } | RpcNotOk;

export type SessionEndReason = "completed" | "stopped" | "abandoned" | "interrupted";

export type SessionEndReply = { status: "ok"; summary: SessionSummaryV1 } | RpcNotOk;

export function coadaptStatus(params: Record<string, never> = {}): Promise<CoAdaptationStatusV1> {
  return call<CoAdaptationStatusV1>("co_adaptation_status", params);
}

export function coadaptConfigUpdate(params: Partial<CoAdaptationConfigV1>): Promise<ConfigUpdateReply> {
  return call<ConfigUpdateReply>("co_adaptation_config_update", { ...params });
}

export function coadaptConsentGrant(params: GrantInput): Promise<ConsentGrantReply> {
  return call<ConsentGrantReply>("co_adaptation_consent_grant", { ...params });
}

export function coadaptConsentRevoke(params: { participant_key: string }): Promise<ConsentRevokeReply> {
  return call<ConsentRevokeReply>("co_adaptation_consent_revoke", { ...params });
}

export function coadaptPreferenceSet(params: PreferenceSetParams): Promise<PreferenceSetReply> {
  return call<PreferenceSetReply>("co_adaptation_preference_set", { ...params });
}

export function coadaptPreferenceRetract(params: { pref_id: string }): Promise<PreferenceRetractReply> {
  return call<PreferenceRetractReply>("co_adaptation_preference_retract", { ...params });
}

export function coadaptLearningReset(params: { participant_key: string }): Promise<LearningResetReply> {
  return call<LearningResetReply>("co_adaptation_learning_reset", { ...params });
}

export function coadaptErase(params: { participant_key: string; scope: "today" }): Promise<EraseReply> {
  return call<EraseReply>("co_adaptation_erase", { ...params });
}

export function coadaptResults(params: { participant_key: string; scope: "today" | "all" }): Promise<ResultsReply> {
  return call<ResultsReply>("co_adaptation_results", { ...params });
}

export function coadaptExport(params: { participant_key: string }): Promise<ExportReply> {
  return call<ExportReply>("co_adaptation_export", { ...params });
}

export function coadaptSessionBegin(params: SessionBeginParams): Promise<SessionBeginReply> {
  return call<SessionBeginReply>("co_adaptation_session_begin", { ...params });
}

export function coadaptSessionResume(params: { session_id: string }): Promise<SessionResumeReply> {
  return call<SessionResumeReply>("co_adaptation_session_resume", { ...params });
}

export function coadaptRecommend(params: RecommendParams): Promise<RecommendReply> {
  return call<RecommendReply>("co_adaptation_recommend", { ...params });
}

export function coadaptRecordTrial(params: { session_id: string; trial: TrialV1 }): Promise<RecordTrialReply> {
  return call<RecordTrialReply>("co_adaptation_record_trial", { ...params });
}

export function coadaptReportBurden(params: { session_id: string; report: BurdenV1 }): Promise<ReportBurdenReply> {
  return call<ReportBurdenReply>("co_adaptation_report_burden", { ...params });
}

export function coadaptSessionEnd(params: { session_id: string; reason: SessionEndReason }): Promise<SessionEndReply> {
  return call<SessionEndReply>("co_adaptation_session_end", { ...params });
}
