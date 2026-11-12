import { readable, type Readable } from "svelte/store";
import { call, offNotification, onNotification } from "../api/daemon";
import { session } from "./session";

/** What the voice bar says while the daemon is still starting: the microphone only opens once Heliox is ready. */
export const LISTENING_STARTS_LINE = "Listening starts when Heliox is ready.";

type Startup = { status?: string; state?: string };

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
 * reply leaves the state as it was.
 */
export const daemonStarting: Readable<boolean> = readable(true, (set) => {
  set(true); // each start asks again: a value left from an earlier subscription may be out of date
  let connected = false;

  const apply = (status: Startup) => {
    if (status.state) set(status.state !== "ready" && status.state !== "failed");
  };
  const handler = (method: string, params: unknown) => {
    if (method === "startup_status") apply((params ?? {}) as Startup);
  };
  onNotification(handler);

  const unsubscribe = session.subscribe((s) => {
    const now = Boolean(s.daemonConnected);
    if (now === connected) return;
    connected = now;
    if (!now) {
      set(true); // disconnected: listening waits for Heliox again
      return;
    }
    call<Startup>("startup_status")
      .then((reply) => {
        if (reply?.status === "ok" && reply.state) apply(reply);
        else set(false); // a refusal or an answer without a state: it is up and answering
      })
      .catch((error) => {
        if (!isNoReply(error)) set(false); // e.g. "Method not found" from an older daemon: it is up
      });
  });

  return () => {
    offNotification(handler);
    unsubscribe();
  };
});
