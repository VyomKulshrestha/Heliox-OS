import { writable } from "svelte/store";
import { call, offConnectionState, offNotification, onConnectionState, onNotification } from "../api/daemon";

/**
 * Human takeover of the desktop (SECURITY.md, "Human takeover of the desktop").
 *
 * - "human_control" carries who holds the desktop. While the person holds it,
 *   Heliox refuses the desktop input it starts on its own and autonomous jobs pause.
 * - Autonomous job notifications track whether Heliox is working on its own,
 *   so "Take control" is offered only when there is something to take back.
 */

export interface HumanControlState {
  holder: "agent" | "human";
  reason: string;
  activeJobs: string[];
  busy: boolean;
  error: string;
}

const DEFAULT_STATE: HumanControlState = {
  holder: "agent",
  reason: "",
  activeJobs: [],
  busy: false,
  error: "",
};

const JOB_STARTED = new Set(["autonomous_started"]);
const JOB_ENDED = new Set(["autonomous_complete", "autonomous_cancelled"]);

function jobId(params: Record<string, unknown>): string {
  const job = (params.job ?? params) as Record<string, unknown>;
  return String(job.job_id ?? "");
}

function createHumanControl() {
  const store = writable<HumanControlState>({ ...DEFAULT_STATE });

  function applyStatus(params: Record<string, unknown>) {
    const holder = params.holder === "human" ? "human" : "agent";
    store.update((state) => ({ ...state, holder, reason: holder === "human" ? String(params.reason ?? "") : "" }));
  }

  const notificationHandler = (method: string, params: unknown) => {
    const p = (params ?? {}) as Record<string, unknown>;
    if (method === "human_control") {
      applyStatus(p);
      return;
    }
    const id = jobId(p);
    if (!id) return;
    if (JOB_STARTED.has(method)) {
      store.update((state) =>
        state.activeJobs.includes(id) ? state : { ...state, activeJobs: [...state.activeJobs, id] },
      );
    } else if (JOB_ENDED.has(method)) {
      store.update((state) => ({ ...state, activeJobs: state.activeJobs.filter((job) => job !== id) }));
    }
  };

  // Re-read the holder after every (re)connect, e.g. after a reload or daemon restart.
  const connectionHandler = (connected: boolean) => {
    if (!connected) return;
    void call("human_control_status")
      .then((result) => applyStatus((result ?? {}) as Record<string, unknown>))
      .catch(() => undefined);
  };

  onNotification(notificationHandler);
  onConnectionState(connectionHandler);
  const hot = (
    import.meta as ImportMeta & {
      hot?: { dispose(callback: () => void): void };
    }
  ).hot;
  if (hot) {
    hot.dispose(() => {
      offNotification(notificationHandler);
      offConnectionState(connectionHandler);
    });
  }

  async function change(method: "take_control" | "release_control", params: Record<string, unknown> = {}) {
    store.update((state) => ({ ...state, busy: true, error: "" }));
    try {
      const result = (await call(method, params)) as Record<string, unknown>;
      if (result?.status === "error") {
        throw new Error(String(result.message ?? "request failed"));
      }
      applyStatus(result ?? {});
      store.update((state) => ({ ...state, busy: false }));
    } catch (error) {
      store.update((state) => ({
        ...state,
        busy: false,
        error: error instanceof Error ? error.message : String(error),
      }));
    }
  }

  return {
    subscribe: store.subscribe,
    take: () => change("take_control", { reason: "taken from the Heliox app" }),
    release: () => change("release_control"),
  };
}

export const humanControl = createHumanControl();
