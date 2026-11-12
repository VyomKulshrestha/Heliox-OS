import { listen } from "@tauri-apps/api/event";
import { invoke as tauriInvoke } from "@tauri-apps/api/core";
import { invoke } from "./invoke";

/**
 * WebSocket client for communicating with the Heliox OS Python daemon.
 * Uses JSON-RPC 2.0 protocol over a local WebSocket connection.
 */

const DAEMON_URL = "ws://127.0.0.1:8785";

type JsonRpcResponse = {
  jsonrpc: "2.0";
  result?: unknown;
  error?: { code: number; message: string };
  id: string | number | null;
};

type NotificationHandler = (method: string, params: unknown) => void;
type ConnectionHandler = (connected: boolean) => void;

export type DaemonStatusResult = {
  status?: string;
  message?: string;
  error?: string;
};

/** Reject application-level RPC results outside the operation's success contract. */
export function requireResultStatus<T extends DaemonStatusResult>(
  result: T,
  expectedStatus: string | readonly string[],
  fallbackMessage: string,
): T {
  const expected = typeof expectedStatus === "string" ? [expectedStatus] : expectedStatus;
  if (!expected.includes(result?.status ?? "")) {
    throw new Error(result?.message || result?.error || fallbackMessage);
  }
  return result;
}

/** Reject application-level RPC failures that still arrived successfully. */
export function requireOkResult<T extends DaemonStatusResult>(result: T, fallbackMessage: string): T {
  return requireResultStatus(result, "ok", fallbackMessage);
}

let ws: WebSocket | null = null;
let messageId = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
const notificationHandlers: NotificationHandler[] = [];
const connectionHandlers: ConnectionHandler[] = [];
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectEnabled = true;

export function onNotification(handler: NotificationHandler) {
  notificationHandlers.push(handler);
}

export function offNotification(handler: NotificationHandler) {
  const idx = notificationHandlers.indexOf(handler);
  if (idx !== -1) notificationHandlers.splice(idx, 1);
}

export function onConnectionState(handler: ConnectionHandler) {
  connectionHandlers.push(handler);
}

export function offConnectionState(handler: ConnectionHandler) {
  const idx = connectionHandlers.indexOf(handler);
  if (idx !== -1) connectionHandlers.splice(idx, 1);
}

function notifyConnectionState(connected: boolean) {
  for (const handler of connectionHandlers) handler(connected);
}

export function isConnected(): boolean {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

let connectPromise: Promise<boolean> | null = null;

export async function connect(): Promise<boolean> {
  if (isConnected()) return true;
  if (connectPromise) return connectPromise;
  reconnectEnabled = true;

  connectPromise = new Promise(async (resolve) => {
    const finish = (result: boolean) => {
      connectPromise = null;
      resolve(result);
    };

    // Fetch the auth token from the Rust backend before opening the socket.
    // get_auth_token() reads the file the Python daemon writes on startup.
    // Fallback: VITE_DAEMON_TOKEN env var for browser-only dev mode (no Tauri).
    let authToken = "";
    try {
      authToken = (await tauriInvoke<string>("get_auth_token")) ?? "";
    } catch {
      // Running in browser dev mode without Tauri — try fetching from Vite dev server middleware first
      try {
        const res = await fetch("/api/auth_token");
        if (res.ok) {
          authToken = (await res.text()).trim();
        }
      } catch {
        // ignore
      }
      if (!authToken) {
        authToken = (import.meta as any).env?.VITE_DAEMON_TOKEN ?? "";
      }
    }

    try {
      const socket = new WebSocket(DAEMON_URL);
      ws = socket;

      socket.onopen = async () => {
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }

        // Send the auth handshake as the very first message.
        // The daemon rejects any connection whose first message is not auth.
        const authId = ++messageId;
        const authRequest = {
          jsonrpc: "2.0",
          method: "auth",
          params: { token: authToken },
          id: authId,
        };

        // Register a one-shot pending resolver for the auth response
        pending.set(authId, {
          resolve: (_v) => {
            notifyConnectionState(true);
            finish(true);
          },
          reject: (_e) => {
            if (ws === socket) ws = null;
            finish(false);
          },
        });

        socket.send(JSON.stringify(authRequest));

        // Timeout auth handshake after 5 seconds to prevent hanging
        setTimeout(() => {
          if (pending.has(authId)) {
            pending.delete(authId);
            if (ws === socket) {
              socket.close();
              ws = null;
            }
            finish(false);
          }
        }, 5000);
      };

      socket.onmessage = (event) => {
        try {
          const data: JsonRpcResponse = JSON.parse(event.data);

          if (data.id != null && pending.has(Number(data.id))) {
            const { resolve, reject } = pending.get(Number(data.id))!;
            pending.delete(Number(data.id));
            if (data.error) {
              reject(new Error(data.error.message));
            } else {
              resolve(data.result);
            }
          } else if (!data.id && "method" in data) {
            const notification = data as unknown as { method: string; params: unknown };
            for (const handler of notificationHandlers) {
              handler(notification.method, notification.params);
            }
          }
        } catch {
          // ignore parse errors
        }
      };

      socket.onclose = () => {
        // A delayed close/error from an obsolete socket must never clear a
        // newer authenticated connection or reject that connection's calls.
        if (ws !== socket) return;
        ws = null;
        for (const [id, waiter] of pending) {
          pending.delete(id);
          waiter.reject(new Error("Daemon connection was interrupted"));
        }
        notifyConnectionState(false);
        scheduleReconnect();
        finish(false);
      };

      socket.onerror = () => {
        if (ws !== socket) return;
        ws = null;
        notifyConnectionState(false);
        // A refused connection (the daemon not listening yet) fires error before close, and close then sees an
        // obsolete socket and returns: without this, nothing tried again and the window stayed "Connecting...".
        scheduleReconnect();
        finish(false);
      };
    } catch {
      finish(false);
    }
  });

  return connectPromise;
}

/** How long a request made before the daemon listens waits for it: a first start on a small machine takes minutes. */
export const START_WAIT_MS = 180_000;
export const START_WAIT_FAILED = "Heliox did not start within 3 minutes, so this request was not sent. Try again.";

/** Connect, or wait (bounded) until the reconnect loop connects: a request typed while Heliox is still starting is
 * kept and sent once the daemon listens, where it waits for start-up to finish and is answered then. */
export async function waitForConnection(limitMs: number = START_WAIT_MS): Promise<boolean> {
  if (await connect()) return true;
  return new Promise((resolve) => {
    let settled = false;
    const done = (ok: boolean) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      offConnectionState(handler);
      resolve(ok);
    };
    const handler = (connected: boolean) => {
      if (connected) done(true);
    };
    onConnectionState(handler);
    const timer = setTimeout(() => done(isConnected()), limitMs);
    scheduleReconnect();
  });
}

export async function call<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (!isConnected()) {
    const connected = await waitForConnection();
    if (!connected) throw new Error(START_WAIT_FAILED);
  }

  const id = ++messageId;
  const request = {
    jsonrpc: "2.0",
    method,
    params,
    id,
  };

  return new Promise((resolve, reject) => {
    pending.set(id, {
      resolve: resolve as (v: unknown) => void,
      reject,
    });

    ws!.send(JSON.stringify(request));

    setTimeout(() => {
      if (pending.has(id)) {
        pending.delete(id);
        reject(new Error("Request timeout"));
      }
    }, 300_000); // 5 minute timeout for complex agentic workflows
  });
}

export function disconnect() {
  reconnectEnabled = false;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  notifyConnectionState(false);
}

function scheduleReconnect() {
  if (!reconnectEnabled || reconnectTimer) return;
  reconnectTimer = setTimeout(async () => {
    reconnectTimer = null;
    await connect();
  }, 3000);
}

/**
 * Triggers the Tauri native streaming command bridge
 */
export async function sendPromptToStream(method: string, params: any) {
  // Directly invoke the updated Rust backend pipeline
  return await invoke("send_to_daemon", { method, params });
}

/**
 * Listens to Tauri background stream channels
 */
export async function listenToLLMStream(onChunk: (data: any) => void, onComplete: () => void) {
  // Listen for real-time tokens
  const unlistenChunk = await listen("llm-chunk", (event) => {
    onChunk(event.payload);
  });

  // Listen for stream end
  const unlistenComplete = await listen("llm-complete", () => {
    onComplete();
    // Memory cleanup
    unlistenChunk();
    unlistenComplete();
  });
}
