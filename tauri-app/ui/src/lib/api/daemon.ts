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
let notificationHandler: NotificationHandler | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

export function onNotification(handler: NotificationHandler) {
  notificationHandler = handler;
}

export function isConnected(): boolean {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

export async function connect(): Promise<boolean> {
  if (isConnected()) return true;

  return new Promise((resolve) => {
    try {
      ws = new WebSocket(DAEMON_URL);

      ws.onopen = () => {
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        resolve(true);
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
            notificationHandler?.(notification.method, notification.params);
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
