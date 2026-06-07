import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';

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

let ws: WebSocket | null = null;
let messageId = 0;
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
const notificationHandlers: NotificationHandler[] = [];
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

export function onNotification(handler: NotificationHandler) {
  notificationHandlers.push(handler);
}

export function offNotification(handler: NotificationHandler) {
  const idx = notificationHandlers.indexOf(handler);
  if (idx !== -1) notificationHandlers.splice(idx, 1);
}

export function isConnected(): boolean {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

export async function connect(): Promise<boolean> {
  if (isConnected()) return true;

  // Fetch the auth token from the Rust backend before opening the socket.
  // get_auth_token() reads the file the Python daemon writes on startup.
  // Fallback: VITE_DAEMON_TOKEN env var for browser-only dev mode (no Tauri).
  let authToken = "";
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    authToken = (await invoke<string>("get_auth_token")) ?? "";
  } catch {
    // Running in browser dev mode without Tauri — read from env variable.
    // Set VITE_DAEMON_TOKEN in your .env.local to match the daemon token.
    authToken = (import.meta as any).env?.VITE_DAEMON_TOKEN ?? "";
  }

  return new Promise((resolve) => {
    try {
      ws = new WebSocket(DAEMON_URL);

      ws.onopen = async () => {
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
          resolve: (_v) => resolve(true),
          reject: (_e) => {
            ws = null;
            resolve(false);
          },
        });

        ws!.send(JSON.stringify(authRequest));
      };

      ws.onmessage = (event) => {
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

      ws.onclose = () => {
        ws = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        ws = null;
        resolve(false);
      };
    } catch {
      resolve(false);
    }
  });
}

export async function call<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
  if (!isConnected()) {
    const connected = await connect();
    if (!connected) throw new Error("Cannot connect to Heliox OS daemon");
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
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
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
  return await invoke('send_to_daemon', { method, params });
}

/**
 * Listens to Tauri background stream channels
 */
export async function listenToLLMStream(onChunk: (data: any) => void, onComplete: () => void) {
    // Listen for real-time tokens
    const unlistenChunk = await listen('llm-chunk', (event) => {
        onChunk(event.payload);
    });

    // Listen for stream end
    const unlistenComplete = await listen('llm-complete', () => {
        onComplete();
        // Memory cleanup
        unlistenChunk();
        unlistenComplete();
    });
}