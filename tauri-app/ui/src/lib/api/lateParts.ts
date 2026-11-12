import { offNotification, onNotification } from "./daemon";

/**
 * Parts of the daemon that start after it is ready (neural input, agent mesh, strategy evolution, the evolution
 * harness, proactive learning and the others in pilot.system.late_start). Asked before they have started, the daemon
 * answers { status: "starting", message: "<what it is waiting for>" }: not an error. A panel shows that sentence as a
 * waiting state and asks again when the daemon announces the part is ready (the late_parts notification), or after
 * LATE_PART_RETRY_MS if no notice comes.
 */
export const LATE_PART_RETRY_MS = 5_000;

export type Starting = { status: "starting"; message: string };

/** The part's sentence when a reply says it is still starting; "" otherwise. */
export function startingMessage(result: unknown): string {
  const value = result as { status?: unknown; message?: unknown } | null;
  if (!value || typeof value !== "object" || value.status !== "starting") return "";
  return String(value.message ?? "").trim() || "Still starting; ask again in a moment.";
}

/** Call ``again`` once: when the daemon reports a late part's state, or after ``fallbackMs``. Returns a cancel. */
export function askAgainWhenReady(again: () => void, fallbackMs = LATE_PART_RETRY_MS): () => void {
  let done = false;
  const finish = () => {
    if (done) return;
    done = true;
    clearTimeout(timer);
    offNotification(handler);
    again();
  };
  const handler = (method: string) => {
    if (method === "late_parts") finish();
  };
  const timer = setTimeout(finish, fallbackMs);
  onNotification(handler);
  return () => {
    done = true;
    clearTimeout(timer);
    offNotification(handler);
  };
}

/** One pending "ask again" at a time, for a panel or store; cancel it when the panel goes away. */
export function lateRetry(): { schedule(again: () => void): void; cancel(): void } {
  let cancel: (() => void) | null = null;
  return {
    schedule(again) {
      cancel?.();
      cancel = askAgainWhenReady(() => {
        cancel = null;
        again();
      });
    },
    cancel() {
      cancel?.();
      cancel = null;
    },
  };
}
