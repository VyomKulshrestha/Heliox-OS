import { writable, get } from "svelte/store";
import { _ } from "svelte-i18n";
import { call, connect, isConnected, onConnectionState, onNotification, requireOkResult } from "../api/daemon";
import { classifyExecuteResponse, normalizeActionResult, repairLegacyPlanFallback } from "../utils/executeResponse";
import {
  LEGACY_CHAT_HISTORY_KEY,
  createChatSession,
  deriveChatTitle,
  loadChatSessions,
  saveChatSessions,
  summarizeChatSessions,
  type ChatSessionRecord,
  type ChatSessionSummary,
  type DurableTaskReference,
} from "../utils/chatSessions";
import { settings } from "./settings";
import { companion } from "./companion";
import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";

import { applyAgentRunStatus, applyToolCall, type AgentRun } from "../utils/agentRuns";

export type { AgentRun, ToolCall } from "../utils/agentRuns";
export type MessageType = "user" | "system" | "error" | "plan" | "result" | "assistant" | "git_conflict" | "agent_run";

// 1. Definition interfaces for structuring session data models
export interface PlanAction {
  action_type: string;
  target: string;
  requires_root: boolean;
  destructive: boolean;
  parameters: Record<string, unknown>;
  dry_run?: boolean;
  irreversible?: boolean;
  /** Whether Heliox can undo this action afterwards; decided by the daemon from the action type. */
  undo?: UndoSupport;
  index?: number;
}

export interface UndoSupport {
  kind: "undoable" | "no_change" | "not_undoable";
  reason: string;
}

/** Offered after a task that changed files or the clipboard in ways Heliox can reverse. */
export interface UndoOffer {
  taskKey: string;
  undoable: number;
  notUndoable: number;
}

/** A routine proposed from a finished task: it may use only the actions that task used. */
export interface RoutineOffer {
  receipt_id?: string;
  name: string;
  goal: string;
  allowed_actions: string[];
  daily_at?: string;
  weekdays?: number[];
  every_minutes?: number;
  label?: string;
}

export interface UndoPreviewItem {
  seq: number;
  type: string;
  target: string;
  state: "ready" | "conflict" | "unavailable";
  description: string;
}

export interface UndoPreview {
  task_key: string;
  undone: boolean;
  ready: number;
  items: UndoPreviewItem[];
  not_undoable: { type: string; target: string; reason: string }[];
  preview_token: string;
}

export interface ActionResultData {
  action_type: string;
  target: string;
  success: boolean;
  output: string;
  error: string | null;
}

export interface VerificationData {
  passed: boolean;
  details: string[];
}

export interface Plan {
  plan_id: string;
  explanation: string;
  actions: PlanAction[];
  dry_run?: boolean;
}

export interface GitConflictBlock {
  path: string;
  original_hunk: string;
  conflict_hunk: string;
  proposed_resolution_code: string;
  full_block: string;
}

export interface GitConflictPayload {
  status: string;
  conflicts: GitConflictBlock[];
}

export interface Message {
  type: MessageType;
  text: string;
  timestamp: number;
  plan?: Plan;
  actionResults?: ActionResultData[];
  verification?: VerificationData;
  gitConflict?: GitConflictPayload;
  agentRun?: AgentRun;
  /** What the finished task left behind: its signed receipt and whether it can still be undone. */
  task?: TaskFooter;
}

export interface TaskFooter {
  receiptId?: string;
  undo?: { taskKey: string; available: boolean; undoable: number; notUndoable: number; undone?: boolean };
  receiptCheck?: string;
}

function taskFooterFrom(result: Record<string, unknown>): TaskFooter | undefined {
  const footer: TaskFooter = {};
  if (typeof result.receipt_id === "string" && result.receipt_id) footer.receiptId = result.receipt_id;
  const undo = result.undo && typeof result.undo === "object" ? (result.undo as Record<string, unknown>) : null;
  if (undo && typeof undo.task_key === "string") {
    footer.undo = {
      taskKey: undo.task_key,
      available: undo.available === true,
      undoable: Number(undo.undoable ?? 0),
      notUndoable: Number(undo.not_undoable ?? 0),
    };
  }
  return footer.receiptId || footer.undo ? footer : undefined;
}

export interface LiveActionState {
  index: number;
  action: PlanAction;
  status: "pending" | "running" | "success" | "error";
  output?: string;
  error?: string;
}

export interface BudgetInfo {
  exceeded: boolean;
  errorType: string; // "ActionBudgetExceededError" | "TaskBudgetExceededError" | "BudgetExceededError" | "CircuitBreakerOpenError"
  message: string;
  taskId: string;
  failureCount?: number; // populated for circuit-breaker events
  timestamp: number;
}

export interface RollbackAvailable {
  planId: string;
  snapshotId: string;
}

export interface ProactiveSuggestion {
  suggestionId: string;
  title: string;
  description: string;
  triggerReason: string;
  priority: string;
  learnedRelevance: number;
}

export interface InteractionState {
  interactionId: string;
  source: string;
  phase: string;
  message: string;
  active: boolean;
  elapsedMs: number;
  sequence: number;
}

export interface QueuedTask {
  taskId: string;
  resumeToken: string;
  sessionId: string;
  input: string;
  status: "queued" | "running";
}

/** The authority channel that answered an approval; recorded by the daemon as `input_channel`. Pointer versus
 * keyboard is never distinguished. */
export type ConfirmInputChannel = "dialog" | "gesture" | "voice_deny" | "internal" | "unknown";

/** What the visible ConfirmDialog currently shows as checked, for the plan it was rendered for. */
export interface ConfirmSelection {
  planId: string;
  indices: number[];
}

/** Shown when a channel other than the dialog's own buttons tries to approve with nothing checked. */
export const NOTHING_CHECKED_MESSAGE = "Nothing is checked. Check the actions you want to approve, or deny.";

/**
 * What became of one `session.confirm` call. A caller may say "accepted" or "denied" only for `recorded` with the
 * matching `confirmed` value: anything else means the daemon did not take this answer.
 */
export type ConfirmOutcome =
  /** Nothing was sent: another answer is in flight, the request has no plan ID, or a channel without checkboxes
   * tried to approve with nothing checked. */
  | { kind: "not_sent"; reason: "submitting" | "missing_plan" | "nothing_checked" }
  /** The daemon replied ok. `confirmed` is the decision it says it recorded, or null when the reply did not say. */
  | { kind: "recorded"; confirmed: boolean | null }
  /** The daemon refused the answer or the call failed. `errorShown` is true when the chat already shows
   * "Approval failed: …" for it (the plan was still pending when the reply came). */
  | { kind: "rejected"; message: string; errorShown: boolean };

/** Fallback text for the interject replies to typed or spoken text during an approval (daemon F6). */
function interjectApprovalMessage(status: string): string | null {
  switch (status) {
    case "approval_requires_dialog":
      return "Approve in the visible confirmation dialog. Speech and typed text can deny but never approve.";
    case "denied":
      return "Okay, I won't run that.";
    case "not_denied":
      return "That approval was already answered, so nothing was denied. Say or press Stop to cancel the task.";
    default:
      return null;
  }
}

/** The translated message when i18n is ready, else the English text (tests, early start-up). */
function localizedMessage(key: string, english: string): string {
  try {
    const text = get(_)(key, { default: english });
    return typeof text === "string" && text ? text : english;
  } catch {
    return english;
  }
}

interface SessionState {
  activeSessionId: string;
  daemonConnected: boolean;
  loading: boolean;
  messages: Message[];
  currentPlan: Plan | null;
  confirmRequired: boolean;
  confirmPlanId: string;
  confirmActions: PlanAction[];
  confirmReason: string;
  confirmRiskAssessment: Record<string, unknown> | null;
  /** Gnomon's review of a plan an autonomous job wants to run (advisory; can only ask for approval). */
  confirmGnomon: Record<string, unknown> | null;
  /** True when the pending approval belongs to background work (a job, scheduled run or workflow). */
  confirmBackground: boolean;
  confirmSubmitting: boolean;
  confirmError: string;
  /** Reported by ConfirmDialog on mount and on every toggle; null until it reports for the current approval. */
  confirmSelection: ConfirmSelection | null;
  /** Set by App: Command tab active, the confirmation is the active modal and the document is visible. */
  confirmVisible: boolean;
  /** performance.now() when the dialog last became visible (or showed a new approval while visible); null while
   * hidden. Gesture approval needs at least a second of it. */
  confirmVisibleSince: number | null;
  phase: string;
  interaction: InteractionState | null;
  liveActions: LiveActionState[];
  totalTokens: number;
  estimatedCost: number;
  streamingText: string;
  budget: BudgetInfo | null;
  rollback: RollbackAvailable | null;
  rollbackPending: boolean;
  undoOffer: UndoOffer | null;
  /** The preview shown in the undo dialog; the dialog is open while this is set. */
  undoPreview: UndoPreview | null;
  undoSubmitting: boolean;
  undoError: string;
  /** Further tasks to preview after the current one ("undo the last two tasks"), newest first. */
  undoQueue: string[];
  routineOffer: RoutineOffer | null;
  routineSaving: boolean;
  proactiveSuggestion: ProactiveSuggestion | null;
  proactiveSuggestionPending: boolean;
  terminalStatus: string;
  durableTask?: DurableTaskReference;
  queuedTasks: QueuedTask[];
}

export interface Attachment {
  name: string;
  type: string;
  content: string;
}

function normalizeMessages(messages: Message[]): Message[] {
  let previousPlanExplanation = "";
  return messages.map((message) => {
    if (message.type === "plan") {
      previousPlanExplanation = String(message.plan?.explanation ?? message.text).trim();
    }
    const repaired = repairLegacyPlanFallback(message, previousPlanExplanation);
    return {
      ...repaired,
      actionResults: repaired.actionResults?.map((result) => normalizeActionResult(result)),
    };
  });
}

const browserStorage = typeof localStorage === "undefined" ? null : localStorage;
const loadedChatCollection = loadChatSessions(browserStorage);
let chatSessions: ChatSessionRecord[] = loadedChatCollection.sessions.map((chat) => ({
  ...chat,
  messages: normalizeMessages(chat.messages),
}));
const loadedActiveChat =
  chatSessions.find((chat) => chat.id === loadedChatCollection.activeSessionId) ?? chatSessions[0];
const completedQueuedTaskIds = new Set<string>();

const initialState: SessionState = {
  activeSessionId: loadedActiveChat.id,
  daemonConnected: false,
  loading: false,
  messages: loadedActiveChat.messages,
  currentPlan: null,
  confirmRequired: false,
  confirmPlanId: "",
  confirmActions: [],
  confirmReason: "",
  confirmRiskAssessment: null,
  confirmGnomon: null,
  confirmBackground: false,
  confirmSubmitting: false,
  confirmError: "",
  confirmSelection: null,
  confirmVisible: false,
  confirmVisibleSince: null,
  phase: "",
  interaction: null,
  liveActions: [],
  totalTokens: loadedActiveChat.totalTokens,
  estimatedCost: loadedActiveChat.estimatedCost,
  streamingText: "",
  budget: null,
  rollback: null,
  rollbackPending: false,
  undoOffer: null,
  undoPreview: null,
  undoSubmitting: false,
  undoError: "",
  undoQueue: [],
  routineOffer: null,
  routineSaving: false,
  proactiveSuggestion: null,
  proactiveSuggestionPending: false,
  terminalStatus: "",
  durableTask: loadedActiveChat.durableTask,
  queuedTasks: [],
};

const MODEL_RATES: Record<string, number> = {
  "gemini-1.5-pro": 0.000003,
  "gpt-4o": 0.000005,
  "claude-sonnet": 0.000004,
  openrouter: 0.000005,
};

const NOTIFY_MIN_DURATION_MS = 15000;
let _lastNotifyPlanId = "";
let _lastNotifyTime = 0;

function isTauriRuntime(): boolean {
  // Tauri v2: __TAURI_INTERNALS__ is always injected into the webview.
  // __TAURI__ only exists if withGlobalTauri:true is set — do NOT use it.
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function notifyDesktop(title: string, body: string) {
  if (!isTauriRuntime()) return;
  try {
    let granted = await isPermissionGranted();
    if (!granted) granted = (await requestPermission()) === "granted";
    sendNotification({ title, body });
  } catch (err) {
    console.error("[Heliox] notification error:", err);
  }
}

function notifyDailyBrief(text: string) {
  return notifyDesktop("Heliox daily brief", text);
}

async function notifyTaskComplete(payload: Record<string, unknown>) {
  if (!isTauriRuntime()) return;

  // Deduplicate: multiple WS connections can broadcast the same completion event.
  const planId = String(payload.plan_id ?? "");
  const now = Date.now();
  if (planId && planId === _lastNotifyPlanId && now - _lastNotifyTime < 2000) return;
  _lastNotifyPlanId = planId;
  _lastNotifyTime = now;

  const durationMs = Number(payload.duration_ms ?? 0);
  if (durationMs > 0 && durationMs < NOTIFY_MIN_DURATION_MS) return;

  const status = String(payload.status ?? "completed");
  const summary = String(payload.summary ?? "");
  const dryRun = Boolean(payload.dry_run);

  let title = "Heliox OS task complete";
  if (status === "error") title = "Heliox OS task failed";
  else if (status === "partial_failure") title = "Heliox OS task completed with issues";
  else if (status === "blocked_by_critic") title = "Heliox OS task blocked by safety review";
  else if (status === "cancelled") title = "Heliox OS task cancelled";
  else if (dryRun) title = "Heliox OS dry run complete";

  const body = summary || "The task has finished.";

  try {
    let granted = await isPermissionGranted();
    if (!granted) {
      const permission = await requestPermission();
      granted = permission === "granted";
    }
    // On Windows, desktop apps are typically always allowed — try even if denied.
    sendNotification({ title, body });
  } catch (err) {
    console.error("[Heliox] notification error:", err);
  }
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// 2. Custom store creation managing real-time core states
/** Approval fields to keep when the interactive task resets: a background approval stays visible. */
function backgroundApproval(s: SessionState): Partial<SessionState> {
  if (!s.confirmBackground || !s.confirmRequired) return { confirmBackground: false };
  return {
    confirmRequired: true,
    confirmPlanId: s.confirmPlanId,
    confirmActions: s.confirmActions,
    confirmReason: s.confirmReason,
    confirmRiskAssessment: s.confirmRiskAssessment,
    confirmGnomon: s.confirmGnomon,
    confirmSubmitting: s.confirmSubmitting,
    confirmError: s.confirmError,
    confirmBackground: true,
  };
}

/** A new or resumed approval starts with no reported selection, and a dialog that is already visible counts as
 * newly shown, so a gesture can never approve it on the strength of the previous approval's selection or time. */
function freshApprovalPresentation(s: SessionState): Partial<SessionState> {
  return {
    confirmSelection: null,
    confirmVisibleSince: s.confirmVisible ? performance.now() : null,
  };
}

function createSession() {
  const { subscribe, update, set } = writable<SessionState>(initialState);
  let lastPersistedMessages: Message[] | null = null;
  let lastPersistedTokens = Number.NaN;
  let lastPersistedCost = Number.NaN;
  let lastPersistedDurableTask: DurableTaskReference | undefined;
  let resumeInFlight = false;
  const pendingCompanionFollowUps = new Map<string, string>();

  function flushPendingCompanionFollowUp() {
    update((s) => {
      const text = pendingCompanionFollowUps.get(s.activeSessionId);
      if (!text) return s;
      pendingCompanionFollowUps.delete(s.activeSessionId);
      if (s.messages.at(-1)?.type === "assistant" && s.messages.at(-1)?.text === text) return s;
      return {
        ...s,
        messages: [...s.messages, { type: "assistant" as MessageType, text, timestamp: Date.now() }],
      };
    });
  }

  function persistActiveSession(state: SessionState, touch = true) {
    const index = chatSessions.findIndex((chat) => chat.id === state.activeSessionId);
    if (index === -1) return;
    const previous = chatSessions[index];
    const changed =
      previous.messages !== state.messages ||
      previous.totalTokens !== state.totalTokens ||
      previous.estimatedCost !== state.estimatedCost ||
      previous.durableTask?.taskId !== state.durableTask?.taskId ||
      previous.durableTask?.resumeToken !== state.durableTask?.resumeToken;
    chatSessions[index] = {
      ...previous,
      title: deriveChatTitle(state.messages),
      updatedAt: touch && changed ? Date.now() : previous.updatedAt,
      messages: state.messages,
      totalTokens: state.totalTokens,
      estimatedCost: state.estimatedCost,
      durableTask: state.durableTask,
    };
    saveChatSessions(browserStorage, chatSessions, state.activeSessionId);
    if (browserStorage) {
      if (state.messages.length > 0) {
        browserStorage.setItem(LEGACY_CHAT_HISTORY_KEY, JSON.stringify(state.messages));
      } else {
        browserStorage.removeItem(LEGACY_CHAT_HISTORY_KEY);
      }
    }
  }

  function stateForChat(current: SessionState, chat: ChatSessionRecord): SessionState {
    return {
      ...initialState,
      activeSessionId: chat.id,
      daemonConnected: current.daemonConnected,
      messages: chat.messages,
      totalTokens: chat.totalTokens,
      estimatedCost: chat.estimatedCost,
      durableTask: chat.durableTask,
      queuedTasks: current.queuedTasks,
      liveActions: [],
      confirmActions: [],
    };
  }

  // Background daemon message channel routing
  onNotification((method, params) => {
    const p = params as Record<string, unknown>;

    switch (method) {
      case "interaction_state": {
        const interaction = {
          interactionId: String(p.interaction_id ?? ""),
          source: String(p.source ?? "system"),
          phase: String(p.phase ?? "idle"),
          message: String(p.message ?? ""),
          active: Boolean(p.active),
          elapsedMs: Number(p.elapsed_ms ?? 0),
          sequence: Number(p.sequence ?? 0),
        };
        update((s) => ({
          ...s,
          interaction,
          phase: interaction.message || interaction.phase.replaceAll("_", " "),
        }));
        break;
      }

      case "task_registered":
        update((s) => {
          if (String(p.session_id ?? "") !== s.activeSessionId) return s;
          return {
            ...s,
            durableTask: {
              taskId: String(p.task_id ?? ""),
              resumeToken: String(p.resume_token ?? ""),
            },
          };
        });
        break;

      case "queued_task_added": {
        const taskId = String(p.task_id ?? "");
        const sessionId = String(p.session_id ?? "");
        update((s) => ({
          ...s,
          queuedTasks: s.queuedTasks.some((task) => task.taskId === taskId)
            ? s.queuedTasks
            : [
                ...s.queuedTasks,
                {
                  taskId,
                  resumeToken: "",
                  sessionId,
                  input: String(p.input ?? ""),
                  status: "queued" as const,
                },
              ],
        }));
        break;
      }

      case "queued_task_started": {
        const taskId = String(p.task_id ?? "");
        const sessionId = String(p.session_id ?? "");
        update((s) => ({
          ...s,
          loading: sessionId === s.activeSessionId ? true : s.loading,
          phase: sessionId === s.activeSessionId ? "running queued task" : s.phase,
          queuedTasks: s.queuedTasks.some((task) => task.taskId === taskId)
            ? s.queuedTasks.map((task) => (task.taskId === taskId ? { ...task, status: "running" as const } : task))
            : [
                ...s.queuedTasks,
                {
                  taskId,
                  resumeToken: "",
                  sessionId,
                  input: String(p.input ?? ""),
                  status: "running" as const,
                },
              ],
        }));
        break;
      }

      case "queued_task_result": {
        const taskId = String(p.task_id ?? "");
        completedQueuedTaskIds.add(taskId);
        if (completedQueuedTaskIds.size > 100) {
          completedQueuedTaskIds.clear();
          completedQueuedTaskIds.add(taskId);
        }
        const sessionId = String(p.session_id ?? "");
        const status = String(p.status ?? "error");
        const result = p.result && typeof p.result === "object" ? (p.result as Record<string, unknown>) : null;
        const terminal = result ? classifyExecuteResponse(result) : null;
        const message = terminal?.text || String(p.message ?? "Task finished.");
        const actionResults: ActionResultData[] = Array.isArray(result?.results)
          ? result.results.map((item) => {
              const raw = item as Record<string, unknown>;
              const action = (raw.action ?? {}) as Record<string, unknown>;
              return normalizeActionResult({
                action_type: String(action.action_type ?? "unknown"),
                target: String(action.target ?? ""),
                success: Boolean(raw.success),
                output: String(raw.output ?? ""),
                error: raw.error ? String(raw.error) : null,
              });
            })
          : [];
        const rawVerification = result?.verification as Record<string, unknown> | undefined;
        const verification: VerificationData | undefined = rawVerification
          ? { passed: Boolean(rawVerification.passed), details: (rawVerification.details ?? []) as string[] }
          : undefined;
        const terminalMessage: Message = {
          type: (terminal?.messageType ?? (status === "success" ? "result" : "system")) as MessageType,
          text: message,
          timestamp: Date.now(),
          actionResults,
          verification,
        };
        update((s) => ({
          ...s,
          loading: sessionId === s.activeSessionId ? false : s.loading,
          phase: sessionId === s.activeSessionId ? "" : s.phase,
          durableTask: s.durableTask?.taskId === taskId ? undefined : s.durableTask,
          queuedTasks: s.queuedTasks.filter((task) => task.taskId !== taskId),
          messages: sessionId === s.activeSessionId ? [...s.messages, terminalMessage] : s.messages,
        }));
        if (sessionId !== get({ subscribe }).activeSessionId) {
          const index = chatSessions.findIndex((chat) => chat.id === sessionId);
          if (index !== -1) {
            const chat = chatSessions[index];
            chatSessions[index] = {
              ...chat,
              messages: [...chat.messages, terminalMessage],
              updatedAt: Date.now(),
            };
            saveChatSessions(browserStorage, chatSessions, get({ subscribe }).activeSessionId);
          }
        }
        break;
      }

      case "status":
        update((s) => ({ ...s, phase: String(p.phase ?? "") }));
        break;

      case "voice_command": {
        const command = String(p.command ?? "").trim();
        update((s) => ({
          ...s,
          loading: true,
          phase: "planning voice command",
          messages:
            command && !(s.messages.at(-1)?.type === "user" && s.messages.at(-1)?.text === command)
              ? [...s.messages, { type: "user" as MessageType, text: command, timestamp: Date.now() }]
              : s.messages,
        }));
        break;
      }

      case "voice_result": {
        const status = String(p.status ?? "error");
        if (status === "approval_required") {
          const message = String(p.message ?? "Approve this request in the visible Heliox confirmation dialog.").trim();
          update((s) => ({
            ...s,
            phase: s.confirmRequired ? "awaiting visible approval" : s.phase,
            messages:
              message && !(s.messages.at(-1)?.type === "system" && s.messages.at(-1)?.text === message)
                ? [...s.messages, { type: "system" as MessageType, text: message, timestamp: Date.now() }]
                : s.messages,
          }));
          break;
        }
        // A started voice workflow is not a finished task: show the note, keep loading and approvals as they are.
        if (status === "answered" || status === "revising" || status === "queued" || status === "submitted") {
          const message = String(p.message ?? p.result ?? "").trim();
          if (message) {
            update((s) => ({
              ...s,
              messages: [
                ...s.messages,
                {
                  type: status === "answered" ? ("result" as MessageType) : ("system" as MessageType),
                  text: message,
                  timestamp: Date.now(),
                },
              ],
            }));
          }
          break;
        }
        applyTaskResult({
          status: status === "partial" ? "partial_failure" : status,
          message: String(p.result ?? p.message ?? p.error ?? "The voice command finished without a visible result."),
          companion_follow_up: p.companion_follow_up,
        });
        break;
      }

      case "companion_follow_up": {
        const sessionId = String(p.session_id ?? "");
        const message = String(p.message ?? "").trim();
        const suggestions = Array.isArray(p.suggestions)
          ? p.suggestions
              .map((item) => String(item).trim())
              .filter(Boolean)
              .slice(0, 3)
          : [];
        if (!message || suggestions.length === 0) break;
        update((s) => {
          if (sessionId && sessionId !== s.activeSessionId) return s;
          const text = `${message}\n\nPossible next steps:\n${suggestions.map((idea) => `- ${idea}`).join("\n")}`;
          if (s.loading) {
            pendingCompanionFollowUps.set(sessionId || s.activeSessionId, text);
            return s;
          }
          if (s.messages.at(-1)?.type === "assistant" && s.messages.at(-1)?.text === text) return s;
          return {
            ...s,
            messages: [...s.messages, { type: "assistant" as MessageType, text, timestamp: Date.now() }],
          };
        });
        break;
      }

      case "plan_preview": {
        const plan: Plan = {
          plan_id: String(p.plan_id ?? ""),
          explanation: String(p.explanation ?? ""),
          actions: (p.actions ?? []) as PlanAction[],
          dry_run: Boolean(p.dry_run),
        };
        const newLiveActions: LiveActionState[] = plan.actions.map((a, i) => ({
          index: i,
          action: a,
          status: "pending",
        }));
        update((s) => ({
          ...s,
          currentPlan: plan,
          liveActions: newLiveActions,
          messages: [
            ...s.messages,
            {
              type: "plan" as MessageType,
              text: plan.explanation,
              timestamp: Date.now(),
              plan,
            },
          ],
        }));
        break;
      }

      case "action_start": {
        update((s) => {
          const nextIdx = s.liveActions.findIndex((a) => a.status === "pending");
          if (nextIdx !== -1) {
            const live = [...s.liveActions];
            live[nextIdx] = { ...live[nextIdx], status: "running" };
            return { ...s, liveActions: live };
          }
          return s;
        });
        break;
      }

      case "action_complete": {
        const resultObj = p.result as Record<string, unknown>;
        const success = Boolean(resultObj.success);
        const snapshotId = resultObj.snapshot_id ? String(resultObj.snapshot_id) : "";
        update((s) => {
          const runningIdx = s.liveActions.findIndex((a) => a.status === "running");
          let next = s;
          if (runningIdx !== -1) {
            const live = [...s.liveActions];
            live[runningIdx] = {
              ...live[runningIdx],
              status: success ? "success" : "error",
              output: String(resultObj.output || ""),
              error: String(resultObj.error || ""),
            };
            next = { ...next, liveActions: live };
          }
          if (snapshotId && next.currentPlan?.plan_id) {
            next = { ...next, rollback: { planId: next.currentPlan.plan_id, snapshotId } };
          }
          return next;
        });
        break;
      }

      case "rollback_complete":
        update((s) => ({
          ...s,
          rollback: null,
          rollbackPending: false,
          messages: [
            ...s.messages,
            {
              type: "system" as MessageType,
              text: String(p.message ?? "Rollback complete."),
              timestamp: Date.now(),
            },
          ],
        }));
        break;

      case "confirm_required":
        update((s) => ({
          ...s,
          confirmRequired: true,
          confirmPlanId: String(p.plan_id ?? ""),
          confirmActions: (p.actions ?? []) as PlanAction[],
          confirmReason: String(p.reason ?? ""),
          confirmRiskAssessment:
            p.risk_assessment && typeof p.risk_assessment === "object"
              ? (p.risk_assessment as Record<string, unknown>)
              : null,
          confirmGnomon: p.gnomon && typeof p.gnomon === "object" ? (p.gnomon as Record<string, unknown>) : null,
          confirmBackground: Boolean(p.source),
          confirmSubmitting: false,
          confirmError: "",
          ...freshApprovalPresentation(s),
        }));
        companion.speak({
          channel: "approval_risk",
          text: String(p.reason ?? "").trim() || "I need your approval before I continue with this action.",
          taskId: String(p.task_id ?? ""),
          dedupeKey: `approval:${String(p.plan_id ?? "")}`,
        });
        break;

      case "token_stream":
        // Fallback for generic tokens
        update((s) => ({
          ...s,
          streamingText: s.streamingText + String(p.token ?? ""),
        }));
        break;

      case "task_complete": {
        void notifyTaskComplete(p);
        const undo = p.undo && typeof p.undo === "object" ? (p.undo as Record<string, unknown>) : null;
        if (undo && undo.available === true && typeof undo.task_key === "string") {
          const offer: UndoOffer = {
            taskKey: undo.task_key,
            undoable: Number(undo.undoable ?? 0),
            notUndoable: Number(undo.not_undoable ?? 0),
          };
          update((s) => ({ ...s, undoOffer: offer }));
        }
        break;
      }

      case "undo_offer": {
        const keys = Array.isArray(p.task_keys)
          ? p.task_keys.filter((key): key is string => typeof key === "string")
          : typeof p.task_key === "string"
            ? [p.task_key]
            : [];
        if (keys.length) {
          update((s) => ({ ...s, undoQueue: keys.slice(1) }));
          void openUndoPreview(keys[0]);
        }
        break;
      }

      case "routine_offer":
        if (typeof p.goal === "string" && Array.isArray(p.allowed_actions)) {
          update((s) => ({ ...s, routineOffer: p as unknown as RoutineOffer }));
        }
        break;

      case "reminder": {
        const text = String(p.text ?? "").trim();
        if (text) {
          addSystemMessage(
            p.late
              ? `Reminder (late): ${text}`
              : p.held
                ? `Reminder (held while you focused): ${text}`
                : `Reminder: ${text}`,
          );
          void notifyDesktop("Heliox reminder", text);
        }
        break;
      }

      case "nudge": {
        const text = String(p.text ?? "").trim();
        if (text) {
          addSystemMessage(p.held ? `Nudge (held while you focused): ${text}` : `Nudge: ${text}`);
          void notifyDesktop("Heliox nudge", text);
        }
        break;
      }

      case "focus_distraction": {
        const text = String(p.text ?? "").trim();
        if (text) addSystemMessage(text);
        break;
      }

      case "watch_fired": {
        const text = String(p.text ?? "").trim();
        if (text) {
          addSystemMessage(text);
          void notifyDesktop("Heliox", text);
        }
        break;
      }

      case "focus_ended": {
        const text = String(p.text ?? "").trim();
        if (text) {
          addSystemMessage(text);
          void notifyDesktop("Heliox focus session", text);
        }
        break;
      }

      case "request_expanded":
        if (typeof p.note === "string" && p.note) addSystemMessage(p.note);
        break;

      case "daily_brief": {
        const text = String(p.text ?? "").trim();
        if (text) {
          addSystemMessage(`Daily brief: ${text}`);
          void notifyDailyBrief(text);
        }
        break;
      }

      case "undo_complete":
        update((s) => ({
          ...s,
          undoOffer: s.undoOffer?.taskKey === p.task_key ? null : s.undoOffer,
          undoPreview: s.undoPreview?.task_key === p.task_key ? null : s.undoPreview,
          messages: s.messages.map((message): Message => {
            const undo = message.task?.undo;
            if (!message.task || !undo || undo.taskKey !== p.task_key) return message;
            return { ...message, task: { ...message.task, undo: { ...undo, available: false, undone: true } } };
          }),
        }));
        break;

      case "budget_update":
      case "token_usage":
        update((s) => {
          const tok = Number(p.tokens ?? p.total_tokens ?? 0);
          const cost = Number(p.cost_usd ?? p.estimated_cost ?? tok * 0.000002);
          return {
            ...s,
            totalTokens: s.totalTokens + tok,
            estimatedCost: s.estimatedCost + cost,
          };
        });
        break;

      case "budget_exceeded":
        companion.speak({
          channel: "task_failure",
          text: `I stopped this task because its budget was exceeded. ${String(p.error ?? "")}`.trim(),
          taskId: String(p.task_id ?? ""),
          dedupeKey: `budget:${String(p.task_id ?? "")}`,
        });
        update((s) => ({
          ...s,
          budget: {
            exceeded: true,
            errorType: String(p.error_type ?? "BudgetExceededError"),
            message: String(p.error ?? "Budget exceeded"),
            taskId: String(p.task_id ?? ""),
            timestamp: Date.now(),
          },
          loading: false,
          phase: "",
          messages: [
            ...s.messages,
            {
              type: "error" as MessageType,
              text: `Budget halt: ${String(p.error ?? "limit reached")}`,
              timestamp: Date.now(),
            },
          ],
        }));
        break;

      case "circuit_breaker_tripped":
        companion.speak({
          channel: "task_failure",
          text: "I stopped this task after repeated provider failures.",
          taskId: String(p.task_id ?? ""),
          dedupeKey: `circuit-breaker:${String(p.task_id ?? "")}`,
        });
        update((s) => ({
          ...s,
          budget: {
            exceeded: true,
            errorType: "CircuitBreakerOpenError",
            message: String(p.error ?? "Circuit breaker tripped"),
            taskId: String(p.task_id ?? ""),
            failureCount: Number(p.failure_count ?? 0),
            timestamp: Date.now(),
          },
          loading: false,
          phase: "",
          messages: [
            ...s.messages,
            {
              type: "error" as MessageType,
              text: `Circuit breaker tripped after ${p.failure_count ?? "several"} consecutive failures. ${String(p.error ?? "")}`,
              timestamp: Date.now(),
            },
          ],
        }));
        break;

      case "companion_interjection":
        update((s) => ({
          ...s,
          phase: String(p.mode ?? "") === "stop" ? "stopping" : "applying your correction",
        }));
        break;

      case "companion_revision_started":
        update((s) => ({
          ...s,
          phase: `revising plan (revision ${Number(p.revision ?? 1)})`,
          currentPlan: null,
          liveActions: [],
        }));
        break;

      case "companion_revision_rejected":
        update((s) => ({
          ...s,
          messages: [
            ...s.messages,
            {
              type: "error" as MessageType,
              text: String(p.message ?? "The live correction could not be applied."),
              timestamp: Date.now(),
            },
          ],
        }));
        break;

      case "companion_plan_intervention": {
        const decision = String(p.decision ?? "WARN").toUpperCase();
        const reason = String(p.reason ?? "The companion found a problem with the proposed plan.");
        update((s) => ({
          ...s,
          phase:
            decision === "REVISE"
              ? "companion revising plan"
              : decision === "STOP"
                ? "companion stopped plan"
                : s.phase,
          currentPlan: decision === "REVISE" ? null : s.currentPlan,
          liveActions: decision === "REVISE" ? [] : s.liveActions,
          messages: [
            ...s.messages,
            {
              type: decision === "STOP" ? ("error" as MessageType) : ("system" as MessageType),
              text: `Companion ${decision.toLowerCase()}: ${reason}`,
              timestamp: Date.now(),
            },
          ],
        }));
        break;
      }

      case "proactive_suggestion":
        companion.speak({
          channel: "proactive_suggestion",
          text: `${String(p.title ?? "I have a suggestion")}. ${String(p.description ?? "")}`.trim(),
          dedupeKey: `suggestion:${String(p.suggestion_id ?? "")}`,
        });
        update((s) => ({
          ...s,
          proactiveSuggestion: {
            suggestionId: String(p.suggestion_id ?? ""),
            title: String(p.title ?? "Heliox has a suggestion"),
            description: String(p.description ?? ""),
            triggerReason: String(p.trigger_reason ?? ""),
            priority: String(p.priority ?? "low"),
            learnedRelevance: Number(p.learned_relevance ?? 0.5),
          },
          proactiveSuggestionPending: false,
        }));
        break;

      case "agent_tool_call":
        update((s) => ({ ...s, messages: applyToolCall(s.messages, p) }));
        break;

      case "autonomous_complete":
      case "autonomous_cancelled":
        update((s) => ({
          ...s,
          messages: applyAgentRunStatus(s.messages, String(p.job_id ?? ""), String(p.status ?? "")),
        }));
        break;

      case "daemon_speech":
        // The daemon already spoke this out loud via its own OS-level TTS
        // (pilot.system.voice.speak(), e.g. the cognitive-stress-gate phrase
        // or AutonomousExecutor's end-of-job announcement) -- this is
        // display-only, so it must NOT also call speakText() here, or the
        // phrase would be spoken twice (daemon audio + frontend speechSynthesis).
        addSystemMessage(String(p.text ?? ""));
        break;
    }
  });

  onConnectionState((connected) => {
    update((s) => ({ ...s, daemonConnected: connected }));
    if (connected) void resumeDurableTask();
  });

  async function init() {
    await connect();
    update((s) => ({ ...s, daemonConnected: isConnected() }));

    setInterval(() => {
      update((s) => ({ ...s, daemonConnected: isConnected() }));
    }, 500);
  }

  async function resumeDurableTask() {
    const current = get({ subscribe });
    if (!current.durableTask || !isConnected() || resumeInFlight) return;
    resumeInFlight = true;
    update((s) => ({
      ...s,
      loading: true,
      phase: "recovering interrupted task",
      confirmError: "",
    }));
    try {
      const result = (await call("resume_task", {
        task_id: current.durableTask.taskId,
        resume_token: current.durableTask.resumeToken,
      })) as Record<string, unknown>;
      if (result.status === "awaiting_approval") {
        const approval = (result.approval ?? {}) as Record<string, unknown>;
        update((s) => ({
          ...s,
          loading: true,
          phase: "awaiting approval",
          confirmRequired: true,
          confirmPlanId: String(result.plan_id ?? ""),
          confirmActions: (approval.actions ?? []) as PlanAction[],
          confirmReason: String(approval.reason ?? ""),
          confirmRiskAssessment:
            approval.risk_assessment && typeof approval.risk_assessment === "object"
              ? (approval.risk_assessment as Record<string, unknown>)
              : null,
          confirmSubmitting: false,
          confirmError: "",
          ...freshApprovalPresentation(s),
        }));
      } else if (result.status === "error" && !result.task_id) {
        throw new Error(String(result.message ?? "The interrupted task could not be resumed."));
      } else {
        applyTaskResult(result);
      }
    } catch (err) {
      update((s) => ({
        ...s,
        loading: false,
        phase: "resume needs attention",
        confirmSubmitting: false,
        messages: [
          ...s.messages,
          {
            type: "error" as MessageType,
            text: `Safe resume paused: ${String(err instanceof Error ? err.message : err)}`,
            timestamp: Date.now(),
          },
        ],
      }));
    } finally {
      resumeInFlight = false;
    }
  }

  function applyTaskResult(result: Record<string, unknown>) {
    const rawResults = (result.results ?? []) as Array<Record<string, unknown>>;
    const actionResults: ActionResultData[] = rawResults.map((r) => {
      const action = (r.action ?? {}) as Record<string, unknown>;
      return normalizeActionResult({
        action_type: String(action.action_type ?? "unknown"),
        target: String(action.target ?? ""),
        success: Boolean(r.success),
        output: String(r.output ?? ""),
        error: r.error ? String(r.error) : null,
      });
    });

    const rawVerification = result.verification as Record<string, unknown> | undefined;
    const verification: VerificationData | undefined = rawVerification
      ? {
          passed: Boolean(rawVerification.passed),
          details: (rawVerification.details ?? []) as string[],
        }
      : undefined;

    const terminal = classifyExecuteResponse(result);
    const estimatedTokens = estimateTokens(terminal.text);
    let estimatedCost = 0;
    if (terminal.messageType === "result") {
      const settingsState = get(settings);
      const cloudProvider = settingsState?.model?.cloud_provider?.toLowerCase() || "";
      const model = settingsState?.model?.cloud_model || settingsState?.model?.cloud_provider || "ollama";
      const normalizedModel = model.toLowerCase();
      let rate = 0;
      if (cloudProvider === "openrouter") {
        rate = MODEL_RATES.openrouter;
      } else if (normalizedModel.includes("gemini")) {
        rate = MODEL_RATES["gemini-1.5-pro"];
      } else if (normalizedModel.includes("gpt-4o")) {
        rate = MODEL_RATES["gpt-4o"];
      } else if (normalizedModel.includes("claude")) {
        rate = MODEL_RATES["claude-sonnet"];
      }
      estimatedCost = Number((estimatedTokens * rate).toFixed(6));
    }

    update((s) => ({
      ...s,
      loading: false,
      phase: "",
      currentPlan: null,
      confirmRequired: false,
      confirmPlanId: "",
      confirmActions: [],
      confirmReason: "",
      confirmRiskAssessment: null,
      confirmGnomon: null,
      confirmSubmitting: false,
      confirmError: "",
      ...backgroundApproval(s),
      streamingText: "",
      terminalStatus: terminal.status,
      durableTask: undefined,
      totalTokens: s.totalTokens + (terminal.messageType === "result" ? estimatedTokens : 0),
      estimatedCost: s.estimatedCost + estimatedCost,
      messages: [
        ...s.messages,
        {
          type: terminal.messageType as MessageType,
          text: terminal.text,
          timestamp: Date.now(),
          actionResults,
          verification,
          task: taskFooterFrom(result),
        },
      ],
    }));
    if (terminal.messageType === "result") {
      flushPendingCompanionFollowUp();
    } else {
      pendingCompanionFollowUps.delete(get({ subscribe }).activeSessionId);
    }
  }

  async function sendCommand(
    input: string,
    attachments: Attachment[] = [],
    inTaskMode: "correct" | "ask" | "next" = "correct",
  ) {
    const current = get({ subscribe });
    if (current.loading) {
      const previousPhase = current.phase;
      const attachmentContext = attachments
        .map((attachment) => `[Attached File: ${attachment.name}]\n${attachment.content}`)
        .join("\n\n");
      const inTaskInput = [input.trim(), attachmentContext].filter(Boolean).join("\n\n");
      if (!inTaskInput) return;

      if (inTaskMode === "next") {
        try {
          const queued = (await call("queue_execute", {
            input: inTaskInput,
            session_id: current.activeSessionId,
          })) as { status: string; task_id?: string; resume_token?: string; message?: string };
          if (queued.status !== "queued" || !queued.task_id || !queued.resume_token) {
            throw new Error(queued.message || "The daemon did not queue the task.");
          }
          const queuedTaskId = queued.task_id;
          const queuedResumeToken = queued.resume_token;
          update((s) => ({
            ...s,
            queuedTasks: completedQueuedTaskIds.has(queuedTaskId)
              ? s.queuedTasks
              : s.queuedTasks.some((task) => task.taskId === queuedTaskId)
                ? s.queuedTasks.map((task) =>
                    task.taskId === queuedTaskId ? { ...task, resumeToken: queuedResumeToken } : task,
                  )
                : [
                    ...s.queuedTasks,
                    {
                      taskId: queuedTaskId,
                      resumeToken: queuedResumeToken,
                      sessionId: current.activeSessionId,
                      input,
                      status: "queued" as const,
                    },
                  ],
            messages: [
              ...s.messages,
              { type: "user" as MessageType, text: input, timestamp: Date.now() },
              { type: "system" as MessageType, text: "Queued after the current task.", timestamp: Date.now() },
            ],
          }));
        } catch (err) {
          addErrorMessage(`Could not queue task: ${String(err instanceof Error ? err.message : err)}`);
        }
        return;
      }

      update((s) => ({
        ...s,
        phase: inTaskMode === "ask" ? "answering without interrupting" : "sending live correction",
        messages: [...s.messages, { type: "user", text: input || "Attached in-task message", timestamp: Date.now() }],
      }));

      try {
        const result = (await call("interject", { input: inTaskInput, mode: inTaskMode })) as {
          status: string;
          message?: string;
        };
        const approvalNote = interjectApprovalMessage(result.status);
        if (result.status === "answered") {
          update((s) => ({
            ...s,
            phase: previousPhase,
            messages: [
              ...s.messages,
              { type: "result", text: result.message || "No answer available.", timestamp: Date.now() },
            ],
          }));
        } else if (result.status === "revising") {
          update((s) => ({ ...s, phase: "stopping current step and revising" }));
        } else if (result.status === "aborted") {
          update((s) => ({ ...s, phase: "stopping" }));
        } else if (result.status === "no_active_execution") {
          update((s) => ({ ...s, phase: previousPhase }));
          addSystemMessage("That task had just finished. Send the message again to start a new task.");
        } else if (approvalNote !== null) {
          // Typed or spoken text during an approval: an approval word is refused (only the dialog approves), and
          // a denial word is reported as denied only when the daemon says the denial took effect. Nothing replans.
          update((s) => ({ ...s, phase: previousPhase }));
          addSystemMessage(result.message || approvalNote);
        } else {
          update((s) => ({ ...s, phase: previousPhase }));
          addErrorMessage(
            `Live correction failed: ${result.message ?? `unexpected daemon status ${result.status || "missing"}`}`,
          );
        }
      } catch (err) {
        update((s) => ({ ...s, phase: previousPhase }));
        addErrorMessage(`Live correction failed: ${String(err instanceof Error ? err.message : err)}`);
      }
      return;
    }

    if (input.startsWith("/git-resolve ") || input.startsWith("git-resolve ")) {
      const filepath = input.replace(/^(\/)?git-resolve\s+/, "").trim();
      update((s) => ({
        ...s,
        loading: true,
        phase: "detecting conflicts",
        messages: [...s.messages, { type: "user", text: input, timestamp: Date.now() }],
      }));
      try {
        const res = (await call("resolve_git_conflict", { filepath })) as {
          status?: string;
          message?: string;
          conflicts?: GitConflictBlock[];
        };
        if (res.status !== "success") {
          throw new Error(res.message || "Git conflict detection failed.");
        }
        const conflicts = res.conflicts ?? [];
        if (conflicts.length > 0) {
          update((s) => ({
            ...s,
            loading: false,
            phase: "",
            messages: [
              ...s.messages,
              {
                type: "git_conflict",
                text: `Found ${conflicts.length} git conflicts in ${filepath}`,
                timestamp: Date.now(),
                gitConflict: { status: "success", conflicts },
              },
            ],
          }));
        } else {
          update((s) => ({
            ...s,
            loading: false,
            phase: "",
            messages: [
              ...s.messages,
              {
                type: "system",
                text: res.message || `No git conflict markers found in ${filepath}`,
                timestamp: Date.now(),
              },
            ],
          }));
        }
      } catch (err) {
        update((s) => ({
          ...s,
          loading: false,
          phase: "",
          messages: [
            ...s.messages,
            {
              type: "error",
              text: String(err instanceof Error ? err.message : err),
              timestamp: Date.now(),
            },
          ],
        }));
      }
      return;
    }

    update((s) => ({
      ...s,
      loading: true,
      phase: "",
      currentPlan: null,
      liveActions: [],
      confirmRequired: false,
      confirmPlanId: "",
      confirmActions: [],
      confirmReason: "",
      confirmRiskAssessment: null,
      confirmGnomon: null,
      confirmSubmitting: false,
      confirmError: "",
      ...backgroundApproval(s),
      streamingText: "",
      terminalStatus: "",
      messages: [...s.messages, { type: "user", text: input, timestamp: Date.now() }],
    }));

    try {
      const result = (await call("execute", {
        input,
        attachments,
        session_id: current.activeSessionId,
      })) as Record<string, unknown>;
      const queuedRunning = get({ subscribe }).queuedTasks.some(
        (task) => task.sessionId === current.activeSessionId && task.status === "running",
      );
      if (queuedRunning) {
        const terminal = classifyExecuteResponse(result);
        update((s) => ({
          ...s,
          messages: [
            ...s.messages,
            {
              type: terminal.messageType as MessageType,
              text: terminal.text,
              timestamp: Date.now(),
            },
          ],
        }));
      } else {
        applyTaskResult(result);
      }
    } catch (err) {
      const state = get({ subscribe });
      if (
        state.durableTask &&
        String(err instanceof Error ? err.message : err).includes("connection was interrupted")
      ) {
        update((s) => ({
          ...s,
          loading: true,
          phase: "connection interrupted — resuming safely",
          confirmSubmitting: false,
        }));
        return;
      }
      update((s) => ({
        ...s,
        loading: false,
        phase: "",
        confirmRequired: false,
        confirmPlanId: "",
        confirmActions: [],
        confirmReason: "",
        confirmRiskAssessment: null,
        confirmGnomon: null,
        confirmSubmitting: false,
        confirmError: "",
        ...backgroundApproval(s),
        streamingText: "",
        terminalStatus: "error",
        messages: [
          ...s.messages,
          {
            type: "error",
            text: String(err instanceof Error ? err.message : err),
            timestamp: Date.now(),
          },
        ],
      }));
    }
  }

  /** ConfirmDialog reports what it shows as checked, so another channel can approve exactly that and no more. */
  function setConfirmSelection(planId: string, indices: number[]) {
    const current = get({ subscribe }).confirmSelection;
    if (
      current?.planId === planId &&
      current.indices.length === indices.length &&
      current.indices.every((index, i) => index === indices[i])
    ) {
      return;
    }
    update((s) => ({ ...s, confirmSelection: { planId, indices: [...indices] } }));
  }

  /** App reports whether the person can see the dialog; `confirmVisibleSince` restarts each time it appears. */
  function setConfirmVisible(visible: boolean) {
    if (get({ subscribe }).confirmVisible === visible) return;
    update((s) => ({
      ...s,
      confirmVisible: visible,
      confirmVisibleSince: visible ? performance.now() : null,
    }));
  }

  async function confirm(
    accepted: boolean,
    approvedIndices?: number[],
    opts: { inputChannel?: ConfirmInputChannel } = {},
  ): Promise<ConfirmOutcome> {
    const confirmation = get({ subscribe });
    if (confirmation.confirmSubmitting) return { kind: "not_sent", reason: "submitting" };
    const planId = confirmation.confirmPlanId;

    if (!planId) {
      update((s) => ({
        ...s,
        confirmError: "This approval request is missing its plan ID. Wait for the plan to refresh and try again.",
      }));
      return { kind: "not_sent", reason: "missing_plan" };
    }

    let indices = approvedIndices;
    if (accepted && indices === undefined && opts.inputChannel !== undefined) {
      // A channel without its own checkboxes (a gesture) approves only what the dialog reported as checked for
      // this plan. Sending no indices would approve every action that needs approval, unchecked ones included.
      const selection = confirmation.confirmSelection;
      if (!selection || selection.planId !== planId || selection.indices.length === 0) {
        update((s) => ({ ...s, confirmError: localizedMessage("confirm.nothing_checked", NOTHING_CHECKED_MESSAGE) }));
        return { kind: "not_sent", reason: "nothing_checked" };
      }
      indices = [...selection.indices];
    }

    const params: Record<string, unknown> = { plan_id: planId, confirmed: accepted };
    if (indices !== undefined) {
      params.approved_indices = indices;
    }
    if (opts.inputChannel !== undefined) {
      params.input_channel = opts.inputChannel;
    }

    update((s) => ({ ...s, confirmSubmitting: true, confirmError: "" }));

    try {
      const result = (await call("confirm", params)) as Record<string, unknown>;
      if (result.status !== "ok") {
        throw new Error(String(result.message ?? "The daemon did not accept this decision."));
      }

      update((s) =>
        s.confirmPlanId === planId
          ? {
              ...s,
              confirmRequired: false,
              confirmPlanId: "",
              confirmActions: [],
              confirmReason: "",
              confirmRiskAssessment: null,
              confirmGnomon: null,
              confirmSubmitting: false,
              confirmError: "",
            }
          : s,
      );
      if (Boolean(result.resume_required)) {
        // Not awaited: the outcome describes the answer, and the resumed task may run for a long time. The resume
        // reports its own failures in the chat.
        void resumeDurableTask();
      }
      return { kind: "recorded", confirmed: typeof result.confirmed === "boolean" ? result.confirmed : null };
    } catch (err) {
      const message = String(err instanceof Error ? err.message : err);
      const errorShown = get({ subscribe }).confirmPlanId === planId;
      update((s) =>
        s.confirmPlanId === planId
          ? {
              ...s,
              confirmSubmitting: false,
              confirmError: `Decision was not accepted: ${message}`,
              messages: [
                ...s.messages,
                {
                  type: "error" as MessageType,
                  text: `Approval failed: ${message}`,
                  timestamp: Date.now(),
                },
              ],
            }
          : s,
      );
      return { kind: "rejected", message, errorShown };
    }
  }

  function requestRollback() {
    update((s) => ({ ...s, rollbackPending: true }));
  }

  function cancelRollback() {
    update((s) => ({ ...s, rollbackPending: false }));
  }

  async function confirmRollback() {
    let planId = "";
    const unsub = subscribe((s) => {
      planId = s.rollback?.planId ?? "";
    });
    unsub();
    if (!planId) {
      update((s) => ({ ...s, rollbackPending: false }));
      return;
    }

    try {
      const res = (await call("rollback_plan", { plan_id: planId })) as { status: string; message?: string };
      if (res.status !== "ok") {
        update((s) => ({ ...s, rollbackPending: false }));
        addSystemMessage(`Undo failed: ${res.message ?? "unknown error"}`);
      }
      // On success, the "rollback_complete" notification clears rollback/rollbackPending state.
    } catch (err) {
      update((s) => ({ ...s, rollbackPending: false }));
      addSystemMessage(`Undo failed: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function openUndoPreview(taskKey?: string) {
    const key = taskKey ?? get({ subscribe }).undoOffer?.taskKey ?? "";
    if (!key) return;
    try {
      const res = (await call("undo_preview", { task_key: key })) as {
        status: string;
        message?: string;
        preview?: UndoPreview;
      };
      if (res.status !== "ok" || !res.preview) {
        addSystemMessage(`Undo is not available: ${res.message ?? "unknown error"}`);
        return;
      }
      update((s) => ({ ...s, undoPreview: res.preview ?? null, undoSubmitting: false, undoError: "" }));
    } catch (err) {
      addSystemMessage(`Undo is not available: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function suggestRoutine(receiptId: string) {
    try {
      const res = (await call("routine_suggest", { receipt_id: receiptId })) as {
        status: string;
        message?: string;
        routine?: RoutineOffer;
      };
      if (res.status !== "ok" || !res.routine) {
        addSystemMessage(res.message ?? "A routine cannot be made from this task.");
        return;
      }
      update((s) => ({ ...s, routineOffer: res.routine ?? null }));
    } catch (err) {
      addSystemMessage(String(err instanceof Error ? err.message : err));
    }
  }

  function setRoutineTime(dailyAt: string) {
    update((s) => (s.routineOffer ? { ...s, routineOffer: { ...s.routineOffer, daily_at: dailyAt } } : s));
  }

  function dismissRoutine() {
    update((s) => ({ ...s, routineOffer: null, routineSaving: false }));
  }

  async function createRoutine() {
    const offer = get({ subscribe }).routineOffer;
    if (!offer || get({ subscribe }).routineSaving) return;
    update((s) => ({ ...s, routineSaving: true }));
    const timing = offer.every_minutes
      ? { every_minutes: offer.every_minutes }
      : { daily_at: offer.daily_at ?? "09:00", weekdays: offer.weekdays ?? [] };
    try {
      const res = (await call("scheduled_run_create", {
        name: offer.name,
        goal: offer.goal,
        allowed_actions: offer.allowed_actions,
        ...timing,
      })) as { status: string; message?: string };
      if (res.status !== "ok") {
        update((s) => ({ ...s, routineSaving: false }));
        addSystemMessage(`Routine not created: ${res.message ?? "unknown error"}`);
        return;
      }
      update((s) => ({ ...s, routineOffer: null, routineSaving: false }));
      addSystemMessage(
        `Routine created: "${offer.name}". It may use only ${offer.allowed_actions.join(", ")}. Manage it in Settings → Scheduled runs.`,
      );
    } catch (err) {
      update((s) => ({ ...s, routineSaving: false }));
      addSystemMessage(`Routine not created: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function verifyReceipt(receiptId: string) {
    let note: string;
    try {
      const res = (await call("receipt_verify", { receipt_id: receiptId })) as {
        status: string;
        message?: string;
        result?: { valid: boolean; this_computer?: boolean; reasons: string[] };
      };
      if (res.status !== "ok" || !res.result) note = res.message ?? "The receipt could not be checked.";
      else if (!res.result.valid) note = `Receipt not valid: ${res.result.reasons.join(" ")}`;
      else note = res.result.this_computer ? "Signed by this computer, unchanged" : "Valid, signed by another key";
    } catch (err) {
      note = String(err instanceof Error ? err.message : err);
    }
    update((s) => ({
      ...s,
      messages: s.messages.map((message) =>
        message.task?.receiptId === receiptId ? { ...message, task: { ...message.task, receiptCheck: note } } : message,
      ),
    }));
  }

  function cancelUndo() {
    update((s) => ({ ...s, undoPreview: null, undoSubmitting: false, undoError: "", undoQueue: [] }));
  }

  function openNextQueuedUndo() {
    const next = get({ subscribe }).undoQueue[0];
    if (!next) return;
    update((s) => ({ ...s, undoQueue: s.undoQueue.slice(1) }));
    void openUndoPreview(next);
  }

  function dismissUndoOffer() {
    update((s) => ({ ...s, undoOffer: null }));
  }

  async function confirmUndo() {
    const preview = get({ subscribe }).undoPreview;
    if (!preview || get({ subscribe }).undoSubmitting) return;
    update((s) => ({ ...s, undoSubmitting: true, undoError: "" }));
    try {
      const res = (await call("undo_task", {
        task_key: preview.task_key,
        preview_token: preview.preview_token,
        confirmed: true,
      })) as {
        status: string;
        message?: string;
        preview?: UndoPreview;
        undone?: number;
        skipped?: number;
        failed?: number;
      };
      if (res.status === "changed" && res.preview) {
        update((s) => ({
          ...s,
          undoPreview: res.preview ?? null,
          undoSubmitting: false,
          undoError: res.message ?? "",
        }));
        return;
      }
      if (res.status !== "ok" && res.status !== "partial") {
        update((s) => ({ ...s, undoSubmitting: false, undoError: res.message ?? "Undo failed." }));
        return;
      }
      update((s) => ({
        ...s,
        undoPreview: null,
        undoSubmitting: false,
        undoError: "",
        undoOffer: s.undoOffer?.taskKey === preview.task_key ? null : s.undoOffer,
      }));
      const undone = res.undone ?? 0;
      const skipped = res.skipped ?? 0;
      const failed = res.failed ?? 0;
      const parts = [`Undid ${undone} change${undone === 1 ? "" : "s"}.`];
      if (skipped) parts.push(`Left ${skipped} alone because ${skipped === 1 ? "it" : "they"} changed since.`);
      if (failed) parts.push(`${failed} could not be undone.`);
      addSystemMessage(parts.join(" "));
      openNextQueuedUndo();
    } catch (err) {
      update((s) => ({
        ...s,
        undoSubmitting: false,
        undoError: String(err instanceof Error ? err.message : err),
      }));
    }
  }

  async function exportChat(format: "json" | "csv" | "markdown") {
    let msgs: Message[] = [];
    const unsub = subscribe((s) => {
      msgs = s.messages;
    });
    unsub();

    try {
      const res = (await call("export_session_chat", {
        format,
        messages: msgs,
      })) as { status: string; path?: string; message?: string };

      if (res.status === "ok") {
        addSystemMessage(`Chat exported (${format.toUpperCase()}) to: ${res.path}`);
      } else {
        addSystemMessage(`Export failed: ${res.message ?? "unknown error"}`);
      }
    } catch (err) {
      addSystemMessage(`Export failed: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function abort() {
    try {
      const res = (await call("abort")) as { status?: string; message?: string };
      if (res.status === "no_active_execution") {
        addSystemMessage("Nothing to stop.");
      } else if (res.status !== "aborted") {
        addErrorMessage(`Stop failed: ${res.message ?? `unexpected daemon status ${res.status || "missing"}`}`);
      }
      // On "aborted", the in-flight execute/resume_plan RPC resolves on its
      // own with status "cancelled" (handled above), which clears loading --
      // no state update needed here.
    } catch (err) {
      addErrorMessage(`Stop failed: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function cancelQueuedTask(taskId: string) {
    const task = get({ subscribe }).queuedTasks.find((item) => item.taskId === taskId);
    if (!task || !task.resumeToken || task.status !== "queued") return;
    try {
      const result = (await call("cancel_queued_execute", {
        task_id: task.taskId,
        resume_token: task.resumeToken,
      })) as { status?: string; message?: string };
      if (result.status !== "cancelled") {
        addErrorMessage(`Could not cancel queued task: ${result.message || "The task may have started."}`);
      }
    } catch (err) {
      addErrorMessage(`Could not cancel queued task: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  function addErrorMessage(text: string) {
    update((s) => ({
      ...s,
      messages: [...s.messages, { type: "error" as MessageType, text, timestamp: Date.now() }],
    }));
  }

  function addSystemMessage(text: string) {
    update((s) => ({
      ...s,
      messages: [...s.messages, { type: "system" as MessageType, text, timestamp: Date.now() }],
    }));
  }

  function clearMessages() {
    update((s) => ({ ...s, messages: [] }));
  }

  function listChatSessions(): ChatSessionSummary[] {
    persistActiveSession(get({ subscribe }), false);
    return summarizeChatSessions(chatSessions);
  }

  function newChat(): boolean {
    const current = get({ subscribe });
    if (current.loading) return false;
    persistActiveSession(current, false);
    const chat = createChatSession();
    chatSessions = [...chatSessions, chat];
    saveChatSessions(browserStorage, chatSessions, chat.id);
    set(stateForChat(current, chat));
    return true;
  }

  function switchChatSession(sessionId: string): boolean {
    const current = get({ subscribe });
    if (current.loading || sessionId === current.activeSessionId) return !current.loading;
    const target = chatSessions.find((chat) => chat.id === sessionId);
    if (!target) return false;
    persistActiveSession(current, false);
    saveChatSessions(browserStorage, chatSessions, target.id);
    set(stateForChat(current, target));
    if (target.durableTask && isConnected()) void resumeDurableTask();
    return true;
  }

  function resetUsage() {
    update((s) => ({
      ...s,
      totalTokens: 0,
      estimatedCost: 0,
    }));
  }

  init();

  subscribe((s) => {
    if (
      s.messages === lastPersistedMessages &&
      s.totalTokens === lastPersistedTokens &&
      s.estimatedCost === lastPersistedCost &&
      s.durableTask?.taskId === lastPersistedDurableTask?.taskId &&
      s.durableTask?.resumeToken === lastPersistedDurableTask?.resumeToken
    ) {
      return;
    }
    lastPersistedMessages = s.messages;
    lastPersistedTokens = s.totalTokens;
    lastPersistedCost = s.estimatedCost;
    lastPersistedDurableTask = s.durableTask;
    try {
      persistActiveSession(s);
    } catch (error) {
      console.warn("[Heliox] could not persist chat session", error);
    }
  });

  function acknowledgeBudgetEvent() {
    update((s) => ({ ...s, budget: null }));
  }

  async function acceptProactiveSuggestion() {
    const current = get({ subscribe });
    const suggestionId = current.proactiveSuggestion?.suggestionId;
    if (!suggestionId || current.proactiveSuggestionPending) return;

    update((s) => ({ ...s, proactiveSuggestionPending: true }));
    try {
      const result = (await call("proactive_accept", {
        suggestion_id: suggestionId,
      })) as { status?: string; message?: string; error?: string };
      if (result.status === "executing") {
        update((s) => ({
          ...s,
          proactiveSuggestion: null,
          proactiveSuggestionPending: false,
        }));
        addSystemMessage("Suggestion accepted. Heliox started it through the guarded task pipeline.");
      } else {
        update((s) => ({ ...s, proactiveSuggestionPending: false }));
        addSystemMessage(result.message ?? result.error ?? "The suggestion could not be started.");
      }
    } catch (err) {
      update((s) => ({ ...s, proactiveSuggestionPending: false }));
      addSystemMessage(`Suggestion failed: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  async function dismissProactiveSuggestion() {
    const current = get({ subscribe });
    const suggestionId = current.proactiveSuggestion?.suggestionId;
    if (!suggestionId || current.proactiveSuggestionPending) return;

    update((s) => ({ ...s, proactiveSuggestionPending: true }));
    try {
      requireOkResult(
        (await call("proactive_dismiss", { suggestion_id: suggestionId })) as {
          status?: string;
          message?: string;
          error?: string;
        },
        "The daemon did not dismiss the suggestion.",
      );
      update((s) => ({
        ...s,
        proactiveSuggestion: null,
        proactiveSuggestionPending: false,
      }));
    } catch (err) {
      update((s) => ({ ...s, proactiveSuggestionPending: false }));
      addSystemMessage(`Could not dismiss suggestion: ${String(err instanceof Error ? err.message : err)}`);
    }
  }

  return {
    subscribe,
    sendCommand,
    confirm,
    setConfirmSelection,
    setConfirmVisible,
    requestRollback,
    cancelRollback,
    confirmRollback,
    openUndoPreview,
    verifyReceipt,
    suggestRoutine,
    setRoutineTime,
    dismissRoutine,
    createRoutine,
    cancelUndo,
    dismissUndoOffer,
    confirmUndo,
    exportChat,
    abort,
    cancelQueuedTask,
    resumeDurableTask,
    addSystemMessage,
    clearMessages,
    listChatSessions,
    newChat,
    switchChatSession,
    resetUsage,
    acknowledgeBudgetEvent,
    acceptProactiveSuggestion,
    dismissProactiveSuggestion,
  };
}

export const session = createSession();
