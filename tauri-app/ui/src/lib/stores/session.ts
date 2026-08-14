import { writable, get } from "svelte/store";
import { call, connect, isConnected, onConnectionState, onNotification } from "../api/daemon";
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

export type MessageType = "user" | "system" | "error" | "plan" | "result" | "assistant" | "git_conflict";

// 1. Definition interfaces for structuring session data models
export interface PlanAction {
  action_type: string;
  target: string;
  requires_root: boolean;
  destructive: boolean;
  parameters: Record<string, unknown>;
  dry_run?: boolean;
  irreversible?: boolean;
  index?: number;
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
  confirmSubmitting: boolean;
  confirmError: string;
  phase: string;
  interaction: InteractionState | null;
  liveActions: LiveActionState[];
  totalTokens: number;
  estimatedCost: number;
  streamingText: string;
  budget: BudgetInfo | null;
  rollback: RollbackAvailable | null;
  rollbackPending: boolean;
  proactiveSuggestion: ProactiveSuggestion | null;
  proactiveSuggestionPending: boolean;
  terminalStatus: string;
  durableTask?: DurableTaskReference;
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
  confirmSubmitting: false,
  confirmError: "",
  phase: "",
  interaction: null,
  liveActions: [],
  totalTokens: loadedActiveChat.totalTokens,
  estimatedCost: loadedActiveChat.estimatedCost,
  streamingText: "",
  budget: null,
  rollback: null,
  rollbackPending: false,
  proactiveSuggestion: null,
  proactiveSuggestionPending: false,
  terminalStatus: "",
  durableTask: loadedActiveChat.durableTask,
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
function createSession() {
  const { subscribe, update, set } = writable<SessionState>(initialState);
  let lastPersistedMessages: Message[] | null = null;
  let lastPersistedTokens = Number.NaN;
  let lastPersistedCost = Number.NaN;
  let lastPersistedDurableTask: DurableTaskReference | undefined;
  let resumeInFlight = false;

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
        applyTaskResult({
          status: status === "partial" ? "partial_failure" : status,
          message: String(p.result ?? p.message ?? "The voice command finished without a visible result."),
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
          confirmSubmitting: false,
          confirmError: "",
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

      case "task_complete":
        void notifyTaskComplete(p);
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

    clearInterval(window.__interval); window.__interval = setInterval(() => {
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
      confirmSubmitting: false,
      confirmError: "",
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
        },
      ],
    }));
  }

  async function sendCommand(input: string, attachments: Attachment[] = []) {
    const current = get({ subscribe });
    if (current.loading) {
      const attachmentContext = attachments
        .map((attachment) => `[Attached File: ${attachment.name}]\n${attachment.content}`)
        .join("\n\n");
      const correction = [input.trim(), attachmentContext].filter(Boolean).join("\n\n");
      if (!correction) return;

      update((s) => ({
        ...s,
        phase: "sending live correction",
        messages: [...s.messages, { type: "user", text: input || "Attached live correction", timestamp: Date.now() }],
      }));

      try {
        const result = (await call("interject", { input: correction })) as {
          status: string;
          message?: string;
        };
        if (result.status === "revising") {
          update((s) => ({ ...s, phase: "stopping current step and revising" }));
        } else if (result.status === "aborted") {
          update((s) => ({ ...s, phase: "stopping" }));
        } else if (result.status !== "no_active_execution") {
          addSystemMessage(result.message ?? "The live correction was not accepted.");
        } else {
          addSystemMessage("That task had just finished. Send the message again to start a new task.");
        }
      } catch (err) {
        addSystemMessage(`Live correction failed: ${String(err instanceof Error ? err.message : err)}`);
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
        const res = (await call("resolve_git_conflict", { filepath })) as any;
        if (res.status === "success" && res.conflicts && res.conflicts.length > 0) {
          update((s) => ({
            ...s,
            loading: false,
            phase: "",
            messages: [
              ...s.messages,
              {
                type: "git_conflict",
                text: `Found ${res.conflicts.length} git conflicts in ${filepath}`,
                timestamp: Date.now(),
                gitConflict: res,
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
      confirmSubmitting: false,
      confirmError: "",
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
      applyTaskResult(result);
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
        confirmSubmitting: false,
        confirmError: "",
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

  async function confirm(accepted: boolean, approvedIndices?: number[]) {
    let planId = "";
    const unsub = subscribe((s) => {
      planId = s.confirmPlanId;
    });
    unsub();

    if (!planId) {
      update((s) => ({
        ...s,
        confirmError: "This approval request is missing its plan ID. Wait for the plan to refresh and try again.",
      }));
      return;
    }

    const params: Record<string, unknown> = { plan_id: planId, confirmed: accepted };
    if (approvedIndices !== undefined) {
      params.approved_indices = approvedIndices;
    }

    update((s) => ({ ...s, confirmSubmitting: true, confirmError: "" }));

    try {
      const result = (await call("confirm", params)) as Record<string, unknown>;
      if (result.status !== "ok") {
        throw new Error(String(result.message ?? "The daemon did not accept this decision."));
      }

      update((s) => ({
        ...s,
        confirmRequired: false,
        confirmPlanId: "",
        confirmActions: [],
        confirmReason: "",
        confirmRiskAssessment: null,
        confirmSubmitting: false,
        confirmError: "",
      }));
      if (Boolean(result.resume_required)) {
        await resumeDurableTask();
      }
    } catch (err) {
      const message = String(err instanceof Error ? err.message : err);
      update((s) => ({
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
      }));
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
      const res = (await call("abort")) as { status: string };
      if (res.status === "no_active_execution") {
        addSystemMessage("Nothing to stop.");
      }
      // On "aborted", the in-flight execute/resume_plan RPC resolves on its
      // own with status "cancelled" (handled above), which clears loading --
      // no state update needed here.
    } catch (err) {
      addSystemMessage(`Stop failed: ${String(err instanceof Error ? err.message : err)}`);
    }
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
      await call("proactive_dismiss", { suggestion_id: suggestionId });
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
    requestRollback,
    cancelRollback,
    confirmRollback,
    exportChat,
    abort,
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
