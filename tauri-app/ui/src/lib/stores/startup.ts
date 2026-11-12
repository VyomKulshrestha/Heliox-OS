import { readable, type Readable } from "svelte/store";
import { call, offNotification, onNotification } from "../api/daemon";
import { session } from "./session";

/** What the voice bar says while the daemon is still starting: the microphone only opens once Heliox is ready. */
export const LISTENING_STARTS_LINE = "Listening starts when Heliox is ready.";

type Startup = { status?: string; state?: string };

/** How the voice bar asks: how long a reply may take, how long it waits before asking again, and how many times. */
export const STARTUP_ASK = { replyMs: 5_000, retryMs: 2_000, attempts: 3 };

type Asked<T> = { kind: "ok"; value: T } | { kind: "refused" } | { kind: "none" };

/** One request with the three replies, and no reply after STARTUP_ASK.replyMs counts as none (a live connection
 * that never answers would otherwise leave the request pending for the client's five-minute limit). */
async function ask<T>(method: string): Promise<Asked<T>> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<Asked<T>>((resolve) => {
    timer = setTimeout(() => resolve({ kind: "none" }), STARTUP_ASK.replyMs);
  });
  const answered = call<T>(method).then(
    (value): Asked<T> => ({ kind: "ok", value }),
    (error): Asked<T> => (isNoReply(error) ? { kind: "none" } : { kind: "refused" }),
  );
  try {
    return await Promise.race([answered, late]);
  } finally {
    clearTimeout(timer);
  }
}

/** No reply (the connection dropped or the request timed out): Heliox may still be starting, so nothing changes. */
function isNoReply(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error);
  return /timeout|interrupted|did not start|cannot connect/i.test(message);
}

/**
 * Whether Heliox is still starting, for the voice bar. True until the daemon says it is ready: while the window is
 * not yet connected, and while connected with start-up still running (``startup_status`` notifications and, on each
 * connect, one ``startup_status`` request). The three replies to that request: ``ok`` with its state decides; a
 * refusal (any other answer, including an older daemon without the method) means it answers, so not starting; no
 * reply within STARTUP_ASK.replyMs is followed by a ping: a daemon that answers that is up; one that answers nothing is
 * asked again after STARTUP_ASK.retryMs, up to STARTUP_ASK.attempts times, and until then listening waits.
 */
export const daemonStarting: Readable<boolean> = readable(true, (set) => {
  set(true); // each start asks again: a value left from an earlier subscription may be out of date
  let connected = false;
  let generation = 0; // a reconnect or the end of the watch abandons any retry still waiting
  let notified = 0; // startup_status notifications seen: a reply asked for before the latest one is out of date

  const apply = (status: Startup) => {
    if (status.state) set(status.state !== "ready" && status.state !== "failed");
  };
  const handler = (method: string, params: unknown) => {
    if (method !== "startup_status") return;
    notified += 1;
    apply((params ?? {}) as Startup);
  };
  onNotification(handler);

  const unsubscribe = session.subscribe((s) => {
    const now = Boolean(s.daemonConnected);
    if (now === connected) return;
    connected = now;
    if (!now) {
      generation += 1;
      set(true); // disconnected: listening waits for Heliox again
      return;
    }
    void settle(++generation, 1);
  });

  /** Ask startup_status; on no reply, check whether the daemon answers anything (ping), and ask again a few times, so
   * a lost or slow reply never strands the microphone. */
  async function settle(mine: number, attempt: number): Promise<void> {
    const seen = notified;
    const reply = await ask<Startup>("startup_status");
    if (mine !== generation || notified !== seen) return; // reconnected, or the daemon has said something newer
    if (reply.kind === "ok") {
      if (reply.value?.status === "ok" && reply.value.state) apply(reply.value);
      else set(false); // a refusal in words, or an answer without a state: it is up and answering
      return;
    }
    if (reply.kind === "refused") {
      set(false); // e.g. "Method not found" from an older daemon: it is up
      return;
    }
    const pong = await ask<{ pong?: boolean }>("ping");
    if (mine !== generation) return;
    if (pong.kind !== "none") {
      set(false); // it answers other requests, just not this one: treat it as up
      return;
    }
    if (attempt < STARTUP_ASK.attempts) {
      setTimeout(() => void settle(mine, attempt + 1), STARTUP_ASK.retryMs);
    }
  }

  return () => {
    generation += 1;
    offNotification(handler);
    unsubscribe();
  };
});
