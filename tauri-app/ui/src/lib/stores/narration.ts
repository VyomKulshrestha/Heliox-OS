import { writable, get } from "svelte/store";
import { onNotification, call } from "../api/daemon";
import { speakText } from "../utils/tts";

/**
 * Live Execution Narrator store — consumes the two notification types the
 * backend's ExecutionNarrator broadcasts:
 *
 * - "execution_narration": ambient, non-blocking (start/complete of an
 *   action) — spoken immediately via speakText(), no UI state kept.
 * - "execution_interrupt": pre-emptive pause awaiting a response — sets
 *   `active` (which InterruptDialog.svelte renders off) *and* speaks the
 *   reason, satisfying "always pair voice with a visual modal" from a
 *   single event rather than two separate code paths.
 *
 * Consumes notification payloads directly (not a re-fetch-on-notify
 * pattern) since latency matters for a live interruption.
 */

export interface InterruptState {
  active: boolean;
  planId: string;
  reason: string;
  kind: string;
  timeoutSeconds: number;
}

const DEFAULT_STATE: InterruptState = {
  active: false,
  planId: "",
  reason: "",
  kind: "",
  timeoutSeconds: 120,
};

function createNarration() {
  const store = writable<InterruptState>({ ...DEFAULT_STATE });

  onNotification((method, params) => {
    const p = (params ?? {}) as Record<string, unknown>;

    if (method === "execution_narration") {
      const text = String(p.text ?? "");
      if (text) speakText(text);
      return;
    }

    if (method === "execution_interrupt") {
      const reason = String(p.reason ?? "");
      store.set({
        active: true,
        planId: String(p.plan_id ?? ""),
        reason,
        kind: String(p.kind ?? ""),
        timeoutSeconds: Number(p.timeout_seconds ?? 120),
      });
      if (reason) speakText(reason);
      return;
    }

    if (method === "execution_interrupt_timeout" || method === "execution_interrupt_denied") {
      const planId = String(p.plan_id ?? "");
      if (get(store).planId === planId) {
        store.set({ ...DEFAULT_STATE });
      }
    }
  });

  async function respond(confirmed: boolean) {
    const current = get(store);
    if (!current.planId) return;
    store.set({ ...DEFAULT_STATE });
    try {
      await call("confirm", { plan_id: current.planId, confirmed });
    } catch {
      /* best-effort -- the backend times out its own wait either way */
    }
  }

  return {
    subscribe: store.subscribe,
    respond,
  };
}

export const narration = createNarration();
