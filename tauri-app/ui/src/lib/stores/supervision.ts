import { writable } from "svelte/store";
import { onNotification } from "../api/daemon";
import { speakText } from "../utils/tts";

/**
 * User Manual Supervision store — consumes the two notification types
 * UserSupervisionEngine broadcasts:
 *
 * - "supervision_risk_warning": the OCR snippet or keystroke buffer matched
 *   a known destructive-action pattern. Payload only ever carries the
 *   pattern NAME, never the matched content.
 * - "supervision_cognitive_checkin": a sustained stress/cognitive-load
 *   threshold crossing.
 *
 * Unlike narration.ts's InterruptState, there is nothing to block or
 * resolve server-side here -- Heliox cannot intercept the user's own
 * OS-level input, it only observed a copy via the hook. So dismiss() is
 * purely local UI state; no RPC call at all.
 */

export interface SupervisionAlertState {
  active: boolean;
  kind: "risk" | "cognitive" | "";
  message: string;
  pattern: string;
}

const DEFAULT_STATE: SupervisionAlertState = {
  active: false,
  kind: "",
  message: "",
  pattern: "",
};

function createSupervision() {
  const store = writable<SupervisionAlertState>({ ...DEFAULT_STATE });

  onNotification((method, params) => {
    const p = (params ?? {}) as Record<string, unknown>;

    if (method === "supervision_risk_warning") {
      const message = String(p.message ?? "");
      store.set({
        active: true,
        kind: "risk",
        message,
        pattern: String(p.pattern ?? ""),
      });
      if (message) speakText(message);
      return;
    }

    if (method === "supervision_cognitive_checkin") {
      const message = String(p.message ?? "");
      store.set({
        active: true,
        kind: "cognitive",
        message,
        pattern: "",
      });
      if (message) speakText(message);
    }
  });

  function dismiss() {
    store.set({ ...DEFAULT_STATE });
  }

  return {
    subscribe: store.subscribe,
    dismiss,
  };
}

export const supervision = createSupervision();
