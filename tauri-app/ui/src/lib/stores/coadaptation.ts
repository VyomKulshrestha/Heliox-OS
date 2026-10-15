import { derived, get, writable, type Readable } from "svelte/store";
import { onConfigChange } from "../api/configSync";
import { offNotification, onConnectionState, onNotification } from "../api/daemon";
import {
  coadaptConfigUpdate,
  coadaptConsentGrant,
  coadaptConsentRevoke,
  coadaptErase,
  coadaptExport,
  coadaptLearningReset,
  coadaptPreferenceRetract,
  coadaptPreferenceSet,
  coadaptRecommend,
  coadaptRecordTrial,
  coadaptReportBurden,
  coadaptResults,
  coadaptSessionBegin,
  coadaptSessionEnd,
  coadaptSessionResume,
  coadaptStatus,
  type AmbiguityTableContext,
  type BlockRegimeContext,
  type OfferContext,
  type RefreshFields,
  type SessionEndReason,
} from "../coadaptation/api";
import type { TrialEvent } from "../coadaptation/commitRules";
import { consentTextSha256, englishConsentText } from "../coadaptation/consentText";
import { coadaptText } from "../coadaptation/messages";
import type {
  AmbiguityTableV1,
  BurdenV1,
  CoAdaptationConfigV1,
  CoAdaptationPurpose,
  CoAdaptationStatusV1,
  DeviceV1,
  FallbacksV1,
  GazeCheckV1,
  GrantInput,
  Modality,
  PreferenceConflictItemV1,
  PreferenceKind,
  PreferenceScopeV1,
  PreferenceV1,
  RecommendationV1,
  ResultsV1,
  SessionBeginResultV1,
  SessionPlanV1,
  SessionSummaryV1,
  SpreadV1,
  TrialV1,
} from "../coadaptation/types";
import { requestStudyPointer, studyPointer } from "./studyPointer";

/**
 * Co-adaptation state for Settings and the Selection Study (spec section 16.2). The store loads the daemon's status,
 * forwards settings, consent and rule changes, and keeps the running study's session state: the newest fallbacks,
 * the refresh sequence of section 9.7 and the in-memory queue of recorded trials.
 *
 * Recording functions send nothing unless co-adaptation is available and on, and the participant has an active
 * consent with the needed purpose. Nothing here is kept in browser storage, and nothing here concerns approvals:
 * co-adaptation changes presentation only and never grants or widens authority.
 */

export const STUDY_PROTOCOL_ID = "coadapt-study-v1";
export const MAX_PENDING_TRIALS = 20;
/** The 2000 ms fallback of section 9.7 for block decisions and tables, 150 ms for offers (section 9.4). */
export const DECISION_TIMEOUT_MS = 2000;
export const OFFER_TIMEOUT_MS = 150;
const OLDER_DAEMON = "coadaptation.older_daemon";

export type StudyPhase =
  | "setup"
  | "consent"
  | "gaze_check"
  | "practice"
  | "running"
  | "burden"
  | "idle_epoch"
  | "end_questionnaire"
  | "summary"
  | "paused"
  | "error";

export interface CoAdaptationStudyState {
  active: boolean;
  phase: StudyPhase;
  participantKey: string;
  sessionId: string;
  plan: SessionPlanV1 | null;
  blockIndex: number;
  trialIndex: number;
  regime: RecommendationV1 | null;
  table: AmbiguityTableV1 | null;
  /**
   * Pair for this block's posteriors (6.1): the applied L1 recommendation's, else the last applied one's, else
   * session_begin's; echoed in every TrialV1.spread; null for a whole keyboard session.
   */
  spread: SpreadV1 | null;
  /** In memory from the gaze check until session_begin sends it, then for the session's posteriors (12.2). */
  gazeCheckAccuracy: number | null;
  /** Latest from session_begin, status or a refresh_required reply; null means no block starts. */
  fallbacks: FallbacksV1 | null;
  /** The block's count k for the cap rule (9.3); 0 at block start and at an idle epoch. */
  interruptionsUsed: number;
  /** From session_begin's refresh_seq; raised to each applied refresh signal's value (9.7). */
  lastRefreshSeq: number;
  /** lastRefreshSeq when the current block's L1 decision was requested (9.7). */
  blockRefreshSeq: number;
  /** True from an applied refresh signal until fresh fallbacks are held (9.7). */
  inputPaused: boolean;
  /** Set by an applied refresh signal: the current block ends after the current trial, or at once without one. */
  endBlockAfterTrial: boolean;
  /** Retry queue (max 20), in memory only; sent one at a time in trial order (12.2). */
  pendingTrials: TrialV1[];
  message: string;
}

export interface CoAdaptationState {
  /** False on an older daemon ("Method not found") or when the service is unavailable. */
  available: boolean;
  loading: boolean;
  error: string;
  status: CoAdaptationStatusV1 | null;
  study: CoAdaptationStudyState;
}

/** A rule rejected because it conflicts with an active rule (section 9.6). */
export class PreferenceConflictError extends Error {
  readonly reason = "preference_conflict";
  readonly conflicting: PreferenceConflictItemV1[];

  constructor(message: string, conflicting: PreferenceConflictItemV1[]) {
    super(message);
    this.name = "PreferenceConflictError";
    this.conflicting = conflicting;
  }
}

export type BeginSessionOutcome =
  { ok: true; reply: SessionBeginResultV1 } | { ok: false; status: string; reason: string; message: string };

export type RecommendOutcome<T> =
  | { status: "ok"; decisionId: string; recommendation: T }
  | { status: "late" | "rpc_error"; decisionId: string }
  | { status: "refused" | "ignored"; decisionId: string; reason: string; message: string };

export interface SetPreferenceResult {
  preference: PreferenceV1 | null;
  notes: string[];
}

/**
 * What a withdrawal could not delete yet (the daemon retries at every start, status and prune). `keyPending`: the
 * participant's keyring key still exists; `filesPending`: an export file or a rule is still to be deleted.
 */
export interface RevokeResult {
  keyPending: boolean;
  filesPending: boolean;
}

const ENDED_REASONS = new Set(["revoked", "expired", "reset", "erased"]);
/**
 * Ignored reasons a daemon reply on an open session gives once the daemon no longer holds that session (it closed or
 * dropped it, or the consent or the feature is gone). Every later call is ignored the same way, so the study ends.
 */
const SESSION_GONE_REASONS = new Set([
  "session_closed",
  "unknown_session",
  "no_consent",
  "consent_expired",
  "purpose_missing",
  "disabled",
]);

/** The `study.ended.*` message key for a session-level ignored reason. */
function endedReasonOf(reason: string | undefined): string {
  if (reason === "consent_expired") return "expired";
  if (reason === "disabled") return "disabled";
  return "closed";
}
const REFRESH_REASONS = new Set(["preferences", "drift", "calibration_suspended"]);
const REFUSAL_KEYS = new Set([
  "recording_active",
  "task_running",
  "neural_armed",
  "surface_too_small",
  "consent_text_changed",
  "protocol_changed",
  "no_allowed_regime",
  "disabled",
  "no_consent",
  "consent_expired",
  "purpose_missing",
  "session_active",
  "hand_pointer_not_allowed",
  "keyring_unavailable",
  "preferences_unreadable",
]);
const PAUSE_KEYS = new Set(["modal", "hidden", "blur", "resize", "palm_exit", "daemon", "tab", "user"]);

/** The message shown for a start refusal or a pause reason (sections 12.2 and 16.6). */
export function refusalMessage(reason: string): string {
  if (REFUSAL_KEYS.has(reason)) return coadaptText(`coadaptation.refusal.${reason}`);
  if (PAUSE_KEYS.has(reason)) return coadaptText(`study.pause.${reason}`);
  if (reason === "paused") return coadaptText("study.pause.user");
  return coadaptText("coadaptation.refusal.unknown", { reason });
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function isMethodNotFound(cause: unknown): boolean {
  return /method not found/i.test(errorMessage(cause));
}

function newDecisionId(): string {
  const crypto = globalThis.crypto;
  if (crypto && typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function issuedAtMs(fallbacks: FallbacksV1 | null): number {
  if (!fallbacks) return -Infinity;
  const value = Date.parse(fallbacks.issued_at);
  return Number.isNaN(value) ? -Infinity : value;
}

function initialStudy(): CoAdaptationStudyState {
  return {
    active: false,
    phase: "setup",
    participantKey: "",
    sessionId: "",
    plan: null,
    blockIndex: 0,
    trialIndex: 0,
    regime: null,
    table: null,
    spread: null,
    gazeCheckAccuracy: null,
    fallbacks: null,
    interruptionsUsed: 0,
    lastRefreshSeq: 0,
    blockRefreshSeq: 0,
    inputPaused: false,
    endBlockAfterTrial: false,
    pendingTrials: [],
    message: "",
  };
}

function normaliseStatus(status: CoAdaptationStatusV1): CoAdaptationStatusV1 {
  return {
    ...status,
    participants: status.participants ?? [],
    preferences: status.preferences ?? [],
    proposals: status.proposals ?? [],
  };
}

function createCoadaptation() {
  const store = writable<CoAdaptationState>({
    available: true,
    loading: false,
    error: "",
    status: null,
    study: initialStudy(),
  });

  // Session-scoped values that are not part of the published state.
  let sessionModality: Modality | null = null;
  let resumePhase: StudyPhase | null = null;
  let trialDispatch: ((event: TrialEvent) => void) | null = null;
  let sending = false;
  let queueBlocked = false;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let retryDelayMs = 0;
  let flushWaiters: Array<() => void> = [];
  let refreshChain: Promise<void> = Promise.resolve();
  /** Raised by every pause the store applies, so a resume reply never overrides a pause that came while it waited. */
  let pauseCount = 0;
  let loadSeq = 0;
  /** Raised by every openStudy and exitStudy: a session_begin reply for an earlier study is not adopted. */
  let studyGeneration = 0;

  const study = () => get(store).study;
  const patchStudy = (patch: Partial<CoAdaptationStudyState>) =>
    store.update((state) => ({ ...state, study: { ...state.study, ...patch } }));
  const sessionOpen = () => {
    const current = study();
    return current.active && current.sessionId !== "";
  };

  function participantStatus(participantKey: string) {
    return get(store).status?.participants?.find((p) => p.participant_key === participantKey) ?? null;
  }

  /** Whether recording for this participant and purpose may send anything (section 16.2). */
  function mayRecord(participantKey: string, purpose: CoAdaptationPurpose): boolean {
    const state = get(store);
    if (!state.available || !state.status?.available || !state.status.config?.enabled) return false;
    const participant = participantStatus(participantKey);
    if (!participant?.consent?.active) return false;
    const expires = Date.parse(participant.consent.expires_at);
    if (!Number.isNaN(expires) && expires <= Date.now()) return false;
    return participant.consent.purposes.includes(purpose);
  }

  /** The running participant's fallbacks for the session's modality in a status payload (the cached one by default). */
  function statusFallbacks(status: CoAdaptationStatusV1 | null = get(store).status): FallbacksV1 | null {
    const current = study();
    if (!current.participantKey || !sessionModality || !status) return null;
    const participant = status.participants?.find((p) => p.participant_key === current.participantKey);
    return participant?.fallbacks?.[sessionModality] ?? null;
  }

  /** Keeps the newest fallbacks for the session's modality (the daemon rejects an issue time that goes back). */
  function adoptFallbacks(fallbacks: FallbacksV1 | null): void {
    if (!fallbacks) return;
    const held = study().fallbacks;
    if (!held || issuedAtMs(fallbacks) >= issuedAtMs(held)) patchStudy({ fallbacks });
  }

  /**
   * Fetches `co_adaptation_status` and publishes it unless a newer fetch started meanwhile; returns what it fetched
   * either way, so a refresh always reads fallbacks issued after the change it applies.
   */
  async function fetchStatus(): Promise<CoAdaptationStatusV1 | null> {
    const seq = ++loadSeq;
    store.update((state) => ({ ...state, loading: true }));
    try {
      const status = normaliseStatus(await coadaptStatus());
      if (seq !== loadSeq) return status;
      store.update((state) => ({
        ...state,
        available: status.available !== false,
        loading: false,
        error: status.available === false ? status.message || coadaptText("coadaptation.unavailable") : "",
        status,
      }));
      if (sessionOpen()) {
        if (!status.config?.enabled) {
          endedByDaemon("disabled");
        } else {
          adoptFallbacks(statusFallbacks(status));
        }
      }
      return status;
    } catch (cause) {
      if (seq !== loadSeq) return null;
      const older = isMethodNotFound(cause);
      store.update((state) => ({
        ...state,
        available: older ? false : state.available,
        loading: false,
        error: older
          ? coadaptText(OLDER_DAEMON)
          : coadaptText("coadaptation.load_failed", { message: errorMessage(cause) }),
      }));
      return null;
    }
  }

  async function load(): Promise<void> {
    await fetchStatus();
  }

  function wrapRpcError(cause: unknown): Error {
    return new Error(isMethodNotFound(cause) ? coadaptText(OLDER_DAEMON) : errorMessage(cause));
  }

  async function updateConfig(changed: Partial<CoAdaptationConfigV1>): Promise<void> {
    if (Object.keys(changed).length === 0) return;
    let reply;
    try {
      reply = await coadaptConfigUpdate(changed);
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || coadaptText("coadaptation.unavailable"));
    await load();
  }

  async function grantConsent(input: GrantInput): Promise<string> {
    let reply;
    try {
      reply = await coadaptConsentGrant(input);
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") {
      throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    }
    await load();
    return reply.participant_key;
  }

  async function revokeConsent(participantKey: string): Promise<RevokeResult> {
    let reply;
    try {
      reply = await coadaptConsentRevoke({ participant_key: participantKey });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    if (sessionOpen() && study().participantKey === participantKey) endedByDaemon("revoked");
    await load();
    // key_destroyed false means a key still exists; pending covers files and rules too (older daemons omit it).
    const keyPending = reply.erased?.key_destroyed === false;
    return { keyPending, filesPending: !keyPending && reply.erased?.pending === true };
  }

  async function setPreference(
    kind: PreferenceKind,
    scope: PreferenceScopeV1,
    value: unknown,
  ): Promise<SetPreferenceResult> {
    let reply;
    try {
      reply = await coadaptPreferenceSet({ kind, scope, value });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") {
      if ("reason" in reply && reply.reason === "preference_conflict") {
        const conflicting = "conflicting" in reply && Array.isArray(reply.conflicting) ? reply.conflicting : [];
        throw new PreferenceConflictError(reply.message ?? "", conflicting);
      }
      throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    }
    if (sessionOpen() && scope.participant_key !== null && scope.participant_key === study().participantKey) {
      handleReplyRefresh(reply);
    }
    await load();
    return { preference: reply.preference ?? null, notes: reply.notes ?? [] };
  }

  async function retractPreference(prefId: string): Promise<void> {
    const rule = get(store).status?.preferences?.find((pref) => pref.pref_id === prefId) ?? null;
    let reply;
    try {
      reply = await coadaptPreferenceRetract({ pref_id: prefId });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    if (sessionOpen() && rule?.scope.participant_key && rule.scope.participant_key === study().participantKey) {
      handleReplyRefresh(reply);
    }
    await load();
  }

  async function resetLearning(participantKey: string): Promise<void> {
    let reply;
    try {
      reply = await coadaptLearningReset({ participant_key: participantKey });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    if (reply.session_ended && sessionOpen() && study().participantKey === participantKey) endedByDaemon("reset");
    await load();
  }

  async function eraseToday(participantKey: string): Promise<void> {
    let reply;
    try {
      reply = await coadaptErase({ participant_key: participantKey, scope: "today" });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    if (reply.session_ended && sessionOpen() && study().participantKey === participantKey) endedByDaemon("erased");
    await load();
  }

  async function results(participantKey: string, scope: "today" | "all"): Promise<ResultsV1> {
    let reply;
    try {
      reply = await coadaptResults({ participant_key: participantKey, scope });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    return reply as ResultsV1;
  }

  async function exportData(participantKey: string): Promise<{ path: string; sha256: string }> {
    if (!mayRecord(participantKey, "research_export")) {
      throw new Error(coadaptText("coadaptation.data.export_needs_consent"));
    }
    let reply;
    try {
      reply = await coadaptExport({ participant_key: participantKey });
    } catch (cause) {
      throw wrapRpcError(cause);
    }
    if (reply.status !== "ok") throw new Error(reply.message || refusalMessage(reply.reason ?? "unknown"));
    return { path: reply.path, sha256: reply.sha256 };
  }

  // ---- Study lifecycle -------------------------------------------------------------------------------------------

  function openStudy(): void {
    if (study().active) return;
    studyGeneration += 1;
    resetSessionLocals();
    patchStudy({ ...initialStudy(), active: true, phase: "setup" });
  }

  function resetSessionLocals(): void {
    sessionModality = null;
    resumePhase = null;
    trialDispatch = null;
    queueBlocked = false;
    retryDelayMs = 0;
    if (retryTimer !== null) clearTimeout(retryTimer);
    retryTimer = null;
    resolveFlushWaiters();
  }

  function releaseStudyPointer(): void {
    if (get(studyPointer).requested) requestStudyPointer(false);
  }

  /**
   * Starts a session (section 12.2 step 3a) after the recording checks. The reply's refresh_seq becomes the last
   * applied value, its fallbacks the held fallbacks and its spread pair the posterior parameters until the first
   * applied L1 recommendation (none for a keyboard session).
   *
   * When the study is closed (Cancel) or reopened while this waits, nothing is adopted: no request is sent if it had
   * not left yet, and a session the daemon opened meanwhile is ended at once as abandoned.
   */
  async function beginSession(input: {
    participantKey: string;
    device: DeviceV1;
    gazeCheck: GazeCheckV1 | null;
  }): Promise<BeginSessionOutcome> {
    const generation = studyGeneration;
    const closed = () => generation !== studyGeneration || !study().active;
    const closedOutcome = (): BeginSessionOutcome => ({
      ok: false,
      status: "ignored",
      reason: "study_closed",
      message: coadaptText("study.ended.closed"),
    });
    const refuse = (status: string, reason: string): BeginSessionOutcome => {
      const message = refusalMessage(reason);
      patchStudy({ message });
      return { ok: false, status, reason, message };
    };
    if (!get(store).status) await load();
    if (closed()) return closedOutcome();
    const state = get(store);
    if (!state.available || !state.status?.available) return refuse("ignored", "disabled");
    if (!state.status.config?.enabled) return refuse("ignored", "disabled");
    if (!participantStatus(input.participantKey)?.consent?.active) return refuse("ignored", "no_consent");
    if (!mayRecord(input.participantKey, "selection_study")) return refuse("ignored", "purpose_missing");
    if (input.device.modality === "hand_pointer" && !mayRecord(input.participantKey, "hand_pointer")) {
      return refuse("ignored", "hand_pointer_not_allowed");
    }
    const gazeCheck = input.gazeCheck && mayRecord(input.participantKey, "gaze_prior") ? input.gazeCheck : null;

    let reply;
    try {
      const consentSha = await consentTextSha256(englishConsentText());
      if (closed()) return closedOutcome();
      reply = await coadaptSessionBegin({
        participant_key: input.participantKey,
        context: "study",
        consent_text_sha256: consentSha,
        device: input.device,
        protocol_id: STUDY_PROTOCOL_ID,
        gaze_check: gazeCheck,
      });
    } catch (cause) {
      if (closed()) return closedOutcome();
      const message = isMethodNotFound(cause) ? coadaptText(OLDER_DAEMON) : errorMessage(cause);
      patchStudy({ message });
      return { ok: false, status: "error", reason: "rpc_error", message };
    }
    if (closed()) {
      if (reply.status === "ok") await abandonOrphanSession((reply as SessionBeginResultV1).session_id);
      return closedOutcome();
    }
    if (reply.status !== "ok") {
      const reason = reply.reason ?? "unknown";
      const message = reply.status === "error" && reply.message ? reply.message : refusalMessage(reason);
      patchStudy({ message });
      return { ok: false, status: reply.status, reason, message };
    }

    const begun = reply as SessionBeginResultV1;
    const modality = input.device.modality;
    resetSessionLocals();
    sessionModality = modality;
    const r = begun.spread?.[modality];
    const floor = begun.floors?.[modality];
    patchStudy({
      active: true,
      phase: "practice",
      participantKey: begun.participant_key || input.participantKey,
      sessionId: begun.session_id,
      plan: begun.plan,
      blockIndex: 0,
      trialIndex: 0,
      regime: null,
      table: null,
      spread: modality === "keyboard" || r === undefined || floor === undefined ? null : { r, floor },
      gazeCheckAccuracy: begun.gaze_prior ? (gazeCheck?.accuracy ?? null) : null,
      fallbacks: begun.fallbacks ?? null,
      interruptionsUsed: 0,
      lastRefreshSeq: begun.refresh_seq ?? 0,
      blockRefreshSeq: begun.refresh_seq ?? 0,
      inputPaused: false,
      endBlockAfterTrial: false,
      pendingTrials: [],
      message: "",
    });
    return { ok: true, reply: begun };
  }

  /** Ends a session whose begin reply arrived after its study closed; the study state is left as it is. */
  async function abandonOrphanSession(sessionId: string): Promise<void> {
    if (!sessionId) return;
    try {
      await coadaptSessionEnd({ session_id: sessionId, reason: "abandoned" });
    } catch {
      /* the daemon closes a leftover session when it restarts */
    }
  }

  async function recommendLevel<T>(
    level: "block_regime" | "ambiguity_table" | "offer",
    context: BlockRegimeContext | AmbiguityTableContext | OfferContext,
    options: { timeoutMs?: number; decisionId?: string } = {},
  ): Promise<RecommendOutcome<T>> {
    const decisionId = options.decisionId ?? newDecisionId();
    const current = study();
    if (!sessionOpen() || !mayRecord(current.participantKey, "selection_study")) {
      return { status: "ignored", decisionId, reason: "no_consent", message: refusalMessage("no_consent") };
    }
    if (level === "block_regime") patchStudy({ blockRefreshSeq: current.lastRefreshSeq });
    const timeoutMs = options.timeoutMs ?? (level === "offer" ? OFFER_TIMEOUT_MS : DECISION_TIMEOUT_MS);
    const request = coadaptRecommend({
      session_id: current.sessionId,
      decision_id: decisionId,
      level,
      context,
    } as Parameters<typeof coadaptRecommend>[0]);
    let timer: ReturnType<typeof setTimeout> | null = null;
    const late = new Promise<"late">((resolve) => {
      timer = setTimeout(() => resolve("late"), timeoutMs);
    });
    let reply;
    try {
      reply = await Promise.race([request, late]);
    } catch {
      return { status: "rpc_error", decisionId };
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
    if (reply === "late") {
      request.catch(() => undefined);
      return { status: "late", decisionId };
    }
    if (reply.status === "ok" && "recommendation" in reply) {
      const recommendation = reply.recommendation as unknown as T;
      if (level === "block_regime") {
        const rec = reply.recommendation as RecommendationV1;
        patchStudy({ regime: rec, table: null, ...(rec.spread ? { spread: rec.spread } : {}) });
      } else if (level === "ambiguity_table") {
        patchStudy({ table: reply.recommendation as AmbiguityTableV1 });
      }
      return { status: "ok", decisionId, recommendation };
    }
    if (reply.status === "refused") {
      const reason = reply.reason ?? "unknown";
      pauseStudy(reason);
      return { status: "refused", decisionId, reason, message: refusalMessage(reason) };
    }
    if (reply.status === "ignored") {
      const reason = reply.reason ?? "unknown";
      endIfSessionGone(reply, current.sessionId);
      return { status: "ignored", decisionId, reason, message: refusalMessage(reason) };
    }
    return { status: "rpc_error", decisionId };
  }

  /** Requests the block's L1 decision; sets blockRefreshSeq to lastRefreshSeq first (9.7). */
  function recommendBlock(context: BlockRegimeContext, options?: { timeoutMs?: number; decisionId?: string }) {
    return recommendLevel<RecommendationV1>("block_regime", context, options);
  }

  function recommendTable(context: AmbiguityTableContext, options?: { timeoutMs?: number; decisionId?: string }) {
    return recommendLevel<AmbiguityTableV1>("ambiguity_table", context, options);
  }

  function recommendOffer(context: OfferContext, options?: { timeoutMs?: number; decisionId?: string }) {
    return recommendLevel<RecommendationV1>("offer", context, options);
  }

  // ---- Trial queue (section 12.2 step 4b) ------------------------------------------------------------------------

  function resolveFlushWaiters(): void {
    const waiters = flushWaiters;
    flushWaiters = [];
    for (const resolve of waiters) resolve();
  }

  /** Queues a trial for `co_adaptation_record_trial`; false when nothing may be recorded or the queue is full. */
  function recordTrial(trial: TrialV1): boolean {
    const current = study();
    if (!sessionOpen() || !mayRecord(current.participantKey, "selection_study")) return false;
    if (current.pendingTrials.length >= MAX_PENDING_TRIALS) return false;
    patchStudy({ pendingTrials: [...current.pendingTrials, trial] });
    void pump();
    return true;
  }

  function dropHead(trialId: string): void {
    const pending = study().pendingTrials;
    if (pending[0]?.trial_id === trialId) patchStudy({ pendingTrials: pending.slice(1) });
  }

  function scheduleRetry(): void {
    if (retryTimer !== null) return;
    retryDelayMs = Math.min(10000, retryDelayMs > 0 ? retryDelayMs * 2 : 1000);
    retryTimer = setTimeout(() => {
      retryTimer = null;
      void pump();
    }, retryDelayMs);
  }

  async function pump(): Promise<void> {
    if (sending || queueBlocked) return;
    const current = study();
    const head = current.pendingTrials[0];
    if (!head || !current.sessionId) {
      resolveFlushWaiters();
      return;
    }
    sending = true;
    let reply;
    try {
      reply = await coadaptRecordTrial({ session_id: current.sessionId, trial: head });
    } catch {
      sending = false;
      scheduleRetry();
      return;
    }
    sending = false;
    if (study().sessionId !== current.sessionId) return;
    if (reply.status === "recorded") {
      retryDelayMs = 0;
      dropHead(head.trial_id);
      handleReplyRefresh(reply);
      void pump();
      return;
    }
    if (reply.status === "refused") {
      // Not stored: the trial stays at the head, marked interrupted, and is sent again after a successful resume.
      const pending = study().pendingTrials;
      if (pending[0]?.trial_id === head.trial_id) {
        patchStudy({
          pendingTrials: [{ ...pending[0], interrupted: true, interruption_reason: "refused" }, ...pending.slice(1)],
        });
      }
      queueBlocked = true;
      pauseStudy(reply.reason ?? "unknown");
      resolveFlushWaiters();
      return;
    }
    if (endIfSessionGone(reply, current.sessionId)) return;
    // Rejected (for example a validation error): the daemon will not store this trial, so it leaves the queue.
    dropHead(head.trial_id);
    void pump();
  }

  /**
   * A daemon reply ignored for a session-level reason means the daemon dropped the session (a storage failure, a
   * restart, a lost consent): the study ends with a message instead of sending, and losing, every later trial.
   */
  function endIfSessionGone(reply: { status: string; reason?: string }, sessionId: string): boolean {
    if (reply.status !== "ignored" || !SESSION_GONE_REASONS.has(reply.reason ?? "")) return false;
    if (sessionId && study().sessionId === sessionId) endedByDaemon(endedReasonOf(reply.reason));
    return true;
  }

  /** Resolves once every queued trial was sent, or the queue waits for a resume (a veto goes before a new table). */
  function flushTrials(timeoutMs = 5000): Promise<void> {
    if (queueBlocked || study().pendingTrials.length === 0) return Promise.resolve();
    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        flushWaiters = flushWaiters.filter((waiter) => waiter !== done);
        resolve();
      }, timeoutMs);
      const done = () => {
        clearTimeout(timer);
        resolve();
      };
      flushWaiters.push(done);
      void pump();
    });
  }

  async function reportBurden(report: BurdenV1): Promise<boolean> {
    const current = study();
    if (!sessionOpen() || !mayRecord(current.participantKey, "selection_study")) return false;
    let reply;
    try {
      reply = await coadaptReportBurden({ session_id: current.sessionId, report });
    } catch {
      return false;
    }
    endIfSessionGone(reply, current.sessionId);
    return reply.status === "recorded";
  }

  /** Ends the open session (queued trials are sent first unless they wait for a resume). */
  async function endSession(reason: SessionEndReason): Promise<SessionSummaryV1 | null> {
    const current = study();
    if (!current.sessionId) return null;
    if (!queueBlocked) await flushTrials();
    const sessionId = study().sessionId;
    if (!sessionId) return null;
    let summary: SessionSummaryV1 | null = null;
    try {
      const reply = await coadaptSessionEnd({ session_id: sessionId, reason });
      if (reply.status === "ok") summary = reply.summary;
    } catch {
      summary = null;
    }
    resetSessionLocals();
    releaseStudyPointer();
    patchStudy({ sessionId: "", pendingTrials: [], inputPaused: false, endBlockAfterTrial: false });
    return summary;
  }

  async function exitStudy(): Promise<void> {
    studyGeneration += 1;
    const current = study();
    if (current.sessionId && current.phase !== "summary") await endSession("stopped");
    resetSessionLocals();
    releaseStudyPointer();
    patchStudy(initialStudy());
  }

  /** The daemon already ended or dropped the session (revoked, expired, reset, erased, disabled, closed). */
  function endedByDaemon(reason: string): void {
    resetSessionLocals();
    releaseStudyPointer();
    patchStudy({
      sessionId: "",
      phase: "error",
      pendingTrials: [],
      inputPaused: false,
      endBlockAfterTrial: false,
      message: coadaptText(`study.ended.${reason}`),
    });
  }

  function pauseStudy(reason: string): void {
    const current = study();
    if (!current.active) return;
    const message = refusalMessage(reason);
    if (!current.sessionId || ["setup", "consent", "gaze_check", "summary", "error"].includes(current.phase)) {
      patchStudy({ message });
      return;
    }
    if (current.phase !== "paused") resumePhase = current.phase;
    pauseCount += 1;
    releaseStudyPointer();
    patchStudy({ phase: "paused", message });
  }

  /**
   * `co_adaptation_session_resume` first; continues only on "ok" (12.2). A pause the store applied while the request
   * was in flight (a recording, a lost connection, a refused call) stands: the study stays paused and the call returns
   * false, so only a later resume continues it.
   */
  async function resumeStudy(): Promise<boolean> {
    const current = study();
    if (!current.sessionId) return false;
    const pausesBefore = pauseCount;
    let reply;
    try {
      reply = await coadaptSessionResume({ session_id: current.sessionId });
    } catch {
      patchStudy({ message: refusalMessage("daemon") });
      return false;
    }
    if (study().sessionId !== current.sessionId) return false;
    if (reply.status !== "ok") {
      if (reply.status === "ignored") {
        endedByDaemon(endedReasonOf(reply.reason));
      } else {
        patchStudy({ message: reply.message || refusalMessage(reply.reason ?? "unknown") });
      }
      return false;
    }
    const before = study().lastRefreshSeq;
    handleReplyRefresh(reply);
    if (pauseCount !== pausesBefore) return false;
    const applied = study().lastRefreshSeq > before;
    const phase = resumePhase ?? (current.phase === "paused" ? "running" : current.phase);
    resumePhase = null;
    queueBlocked = false;
    patchStudy({ phase, message: applied || study().endBlockAfterTrial ? study().message : "" });
    void pump();
    return true;
  }

  // ---- Refresh signals (section 9.7) -----------------------------------------------------------------------------

  /**
   * A refresh signal with a new refresh_seq: input pauses at once and the current trial receives refreshPending;
   * once fallbacks issued after the change are held (the reply's, or a status reload's for a broadcast), the trial
   * receives refresh with the fresh `refresh` regime and cap. Without a trial in progress the block ends at once.
   */
  function applyRefresh(seq: number, fallbacks: FallbacksV1 | null): void {
    const dispatch = trialDispatch;
    patchStudy({
      lastRefreshSeq: seq,
      inputPaused: true,
      endBlockAfterTrial: true,
      message: coadaptText(dispatch ? "study.refresh.updating" : "study.refresh.new_block"),
    });
    dispatch?.({ type: "refreshPending" });
    refreshChain = refreshChain.then(async () => {
      let fresh = fallbacks;
      if (!fresh) fresh = statusFallbacks(await fetchStatus());
      if (!sessionOpen()) return;
      if (!fresh) {
        pauseStudy("daemon");
        return;
      }
      // The newest fallbacks held are issued after the change; the fallbacks a trial echoes never go back in time.
      adoptFallbacks(fresh);
      const held = study().fallbacks ?? fresh;
      patchStudy({ inputPaused: false });
      if (dispatch && trialDispatch === dispatch) {
        dispatch({
          type: "refresh",
          regime: held.refresh,
          interruptionCap: held.interruption_cap,
          fallbacksIssuedAt: held.issued_at,
        });
        patchStudy({
          message: coadaptText(held.refresh === null ? "study.refresh.nothing_left" : "study.refresh.ends_after"),
        });
      } else {
        patchStudy({ message: coadaptText("study.refresh.new_block") });
      }
    });
  }

  /** A reply with refresh fields from preference_set, preference_retract, record_trial or session_resume. */
  function handleReplyRefresh(reply: Partial<RefreshFields>): void {
    if (!sessionOpen()) return;
    const fallbacks = reply.fallbacks ?? null;
    if (!reply.refresh_required) {
      adoptFallbacks(fallbacks);
      return;
    }
    const seq = reply.refresh_seq;
    const current = study();
    if (typeof seq === "number" && seq > current.lastRefreshSeq) {
      applyRefresh(seq, fallbacks);
      return;
    }
    // A duplicate never reverts or interrupts a trial again, but a stale block still ends before the next trial.
    adoptFallbacks(fallbacks);
    if (current.blockRefreshSeq < current.lastRefreshSeq) patchStudy({ endBlockAfterTrial: true });
  }

  function handleUpdated(params: Record<string, unknown>): void {
    const reason = String(params.reason ?? "");
    const participantKey = typeof params.participant_key === "string" ? params.participant_key : null;
    const seq = typeof params.refresh_seq === "number" ? params.refresh_seq : null;
    const running = sessionOpen() && participantKey !== null && participantKey === study().participantKey;
    if (running && ENDED_REASONS.has(reason)) {
      endedByDaemon(reason);
      void load();
      return;
    }
    if (running && REFRESH_REASONS.has(reason) && seq !== null && seq > study().lastRefreshSeq) {
      applyRefresh(seq, null);
      return;
    }
    void load();
  }

  const notificationHandler = (method: string, params: unknown) => {
    const payload = (params ?? {}) as Record<string, unknown>;
    if (method === "co_adaptation_updated") {
      handleUpdated(payload);
    } else if (method === "recorded_workflow_status_changed" && payload.active === true) {
      pauseStudy("recording_active");
    }
  };
  onNotification(notificationHandler);

  const connectionHandler = (connected: boolean) => {
    if (!connected) pauseStudy("daemon");
  };
  try {
    onConnectionState(connectionHandler);
  } catch {
    /* no connection events in this environment */
  }

  const stopConfigSync = onConfigChange(
    { methods: ["co_adaptation_config_update", "co_adaptation_consent_grant", "co_adaptation_consent_revoke"] },
    () => void load(),
  );

  const hot = (import.meta as ImportMeta & { hot?: { dispose(callback: () => void): void } }).hot;
  if (hot) {
    hot.dispose(() => {
      offNotification(notificationHandler);
      stopConfigSync();
    });
  }

  /**
   * Registers the running trial's reducer dispatch for refreshPending and refresh. Only a cued trial attaches (not a
   * block intro, burden prompt or idle epoch); the returned function detaches it when the trial ends.
   */
  function attachTrial(dispatch: (event: TrialEvent) => void): () => void {
    trialDispatch = dispatch;
    return () => {
      if (trialDispatch === dispatch) trialDispatch = null;
    };
  }

  /** The study ended the block for a refresh and will request fresh decisions; ending twice changes nothing. */
  function blockEnded(): void {
    patchStudy({ endBlockAfterTrial: false });
  }

  /** Updates study fields the study host owns (phase, block and trial indices, the cap count, the gaze accuracy). */
  function setStudy(patch: Partial<CoAdaptationStudyState>): void {
    patchStudy(patch);
  }

  return {
    subscribe: store.subscribe,
    load,
    updateConfig,
    grantConsent,
    revokeConsent,
    setPreference,
    retractPreference,
    resetLearning,
    eraseToday,
    results,
    exportData,
    openStudy,
    exitStudy,
    pauseStudy,
    resumeStudy,
    // Study host API (SelectionStudy).
    beginSession,
    recommendBlock,
    recommendTable,
    recommendOffer,
    recordTrial,
    flushTrials,
    reportBurden,
    endSession,
    attachTrial,
    blockEnded,
    setStudy,
    mayRecord,
    statusFallbacks,
  };
}

export const coadaptation = createCoadaptation();

/** True from openStudy() until exitStudy(): the whole study session, paused or not. */
export const studyActive: Readable<boolean> = derived(coadaptation, (state) => state.study.active);
