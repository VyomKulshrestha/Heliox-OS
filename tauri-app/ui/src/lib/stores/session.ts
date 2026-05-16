import { writable, get } from "svelte/store";
import { call, connect, isConnected, onNotification } from "../api/daemon";
import { settings } from "./settings";

export type MessageType = "user" | "system" | "error" | "plan" | "result";

export interface PlanAction {
  action_type: string;
  target: string;
  requires_root: boolean;
  destructive: boolean;
  parameters: Record<string, unknown>;
  dry_run?: boolean;
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

export interface Message {
  type: MessageType;
  text: string;
  timestamp: number;
  plan?: Plan;
  actionResults?: ActionResultData[];
  verification?: VerificationData;
}

export interface LiveActionState {
  index: number;
  action: PlanAction;
  status: "pending" | "running" | "success" | "error";
  output?: string;
  error?: string;
}

interface SessionState {
  daemonConnected: boolean;
  loading: boolean;
  messages: Message[];
  currentPlan: Plan | null;
  confirmRequired: boolean;
  confirmPlanId: string;
  confirmActions: PlanAction[];
  phase: string;
  liveActions: LiveActionState[];
  totalTokens: number;
  estimatedCost: number;
  streamingText: string;
}

const initialState: SessionState = {
  daemonConnected: false,
  loading: false,
  messages: [],
  currentPlan: null,
  confirmRequired: false,
  confirmPlanId: "",
  confirmActions: [],
  phase: "",
  liveActions: [],
  totalTokens: 0,
  estimatedCost: 0,
  streamingText: "",
};

const MODEL_RATES: Record<string, number> = {
  "gemini-1.5-pro": 0.000003,
  "gpt-4o": 0.000005,
  "claude-sonnet": 0.000004,
};

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function createSession() {
  const { subscribe, update, set } = writable<SessionState>(initialState);

  onNotification((method, params) => {
    const p = params as Record<string, unknown>;

    switch (method) {
      case "status":
        update((s) => ({ ...s, phase: String(p.phase ?? "") }));
        break;

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
          status: "pending"
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
        update(s => {
          const nextIdx = s.liveActions.findIndex(a => a.status === "pending");
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
        update(s => {
          const runningIdx = s.liveActions.findIndex(a => a.status === "running");
          if (runningIdx !== -1) {
            const live = [...s.liveActions];
            live[runningIdx] = {
              ...live[runningIdx],
              status: success ? "success" : "error",
              output: String(resultObj.output || ""),
              error: String(resultObj.error || "")
            };
            return { ...s, liveActions: live };
          }
          return s;
        });
        break;
      }

      case "confirm_required":
        update((s) => ({
          ...s,
          confirmRequired: true,
          confirmPlanId: String(p.plan_id ?? ""),
          confirmActions: (p.actions ?? []) as PlanAction[],
        }));
        break;

      case "token_stream":
        update((s) => ({
          ...s,
          streamingText: s.streamingText + String(p.token ?? ""),
        }));
        break;
    }
  });

  async function init() {
    const connected = await connect();
    update((s) => ({ ...s, daemonConnected: connected }));

    setInterval(async () => {
      update((s) => ({ ...s, daemonConnected: isConnected() }));
      if (!isConnected()) await connect();
    }, 5000);
  }

  async function sendCommand(input: string) {
    update((s) => ({
      ...s,
      loading: true,
      phase: "",
      currentPlan: null,
      liveActions: [],
      confirmRequired: false,
      confirmPlanId: "",
      streamingText: "",
      messages: [
        ...s.messages,
        { type: "user", text: input, timestamp: Date.now() },
      ],
    }));

    try {
      const result = (await call("execute", { input })) as Record<string, unknown>;
      const isDryRun = Boolean(result.dry_run);

      const rawResults = (result.results ?? []) as Array<Record<string, unknown>>;
      const actionResults: ActionResultData[] = rawResults.map((r) => {
        const action = (r.action ?? {}) as Record<string, unknown>;
        return {
          action_type: String(action.action_type ?? "unknown"),
          target: String(action.target ?? ""),
          success: Boolean(r.success),
          output: String(r.output ?? ""),
          error: r.error ? String(r.error) : null,
        };
      });

      const rawVerification = result.verification as Record<string, unknown> | undefined;
      const verification: VerificationData | undefined = rawVerification
        ? {
          passed: Boolean(rawVerification.passed),
          details: ((rawVerification.details ?? []) as string[]),
        }
        : undefined;

      if (result.status === "error") {
        const streamingContent = get(session).streamingText;
        update((s) => ({
          ...s,
          loading: false,
          phase: "",
          currentPlan: null,
          streamingText: "",
          messages: [
            ...s.messages,
            {
              type: "error" as MessageType,
              text: streamingContent || String(result.message ?? result.explanation ?? "Unknown error"),
              timestamp: Date.now(),
            },
          ],
        }));
      } else if (result.status === "cancelled") {
        update((s) => ({
          ...s,
          loading: false,
          phase: "",
          currentPlan: null,
          confirmRequired: false,
          streamingText: "",
          messages: [
            ...s.messages,
            {
              type: "system" as MessageType,
              text: String(result.message ?? "Action cancelled by user."),
              timestamp: Date.now(),
            },
          ],
        }));
      } else {
        const responseText = isDryRun
          ? String(result.explanation || "(dry run) No changes were made.")
          : String(result.explanation ?? "");

        const estimatedTokens = estimateTokens(responseText);

        const settingsState = get(settings);
        const model =
          settingsState?.model?.cloud_model ||
          settingsState?.model?.cloud_provider ||
          "ollama";

        const normalizedModel = model.toLowerCase();

        let rate = 0;

        if (normalizedModel.includes("gemini")) {
          rate = MODEL_RATES["gemini-1.5-pro"];
        } else if (normalizedModel.includes("gpt-4o")) {
          rate = MODEL_RATES["gpt-4o"];
        } else if (normalizedModel.includes("claude")) {
          rate = MODEL_RATES["claude-sonnet"];
        }
        const estimatedCost = Number((estimatedTokens * rate).toFixed(6));

        const finalText = get(session).streamingText || responseText;

        update((s) => ({
          ...s,
          loading: false,
          phase: "",
          currentPlan: null,
          confirmRequired: false,
          streamingText: "",

          totalTokens: s.totalTokens + estimatedTokens,
          estimatedCost: s.estimatedCost + estimatedCost,

          messages: [
            ...s.messages,
            {
              type: "result" as MessageType,
              text: finalText,
              timestamp: Date.now(),
              actionResults,
              verification,
            },
          ],
        }));
      }
    } catch (err) {
      update((s) => ({
        ...s,
        loading: false,
        phase: "",
        streamingText: "",
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

  function confirm(accepted: boolean) {
    let planId = "";
    const unsub = subscribe((s) => { planId = s.confirmPlanId; });
    unsub();

    update((s) => ({
      ...s,
      confirmRequired: false,
      confirmPlanId: "",
      confirmActions: [],
    }));

    if (!accepted) {
      update((s) => ({
        ...s,
        messages: [
          ...s.messages,
          { type: "system" as MessageType, text: "Action denied.", timestamp: Date.now() },
        ],
      }));
    }

    call("confirm", { plan_id: planId, confirmed: accepted }).catch(() => { });
  }

  function addSystemMessage(text: string) {
    update((s) => ({
      ...s,
      messages: [
        ...s.messages,
        { type: "system" as MessageType, text, timestamp: Date.now() },
      ],
    }));
  }

  function clearMessages() {
    update((s) => ({ ...s, messages: [] }));
  }

  function resetUsage() {
    update((s) => ({
      ...s,
      totalTokens: 0,
      estimatedCost: 0,
    }));
  }

  init();

  return {
    subscribe,
    sendCommand,
    confirm,
    addSystemMessage,
    clearMessages,
    resetUsage,
  };
}

export const session = createSession();
