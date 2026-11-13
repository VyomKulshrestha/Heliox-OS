/**
 * That the page met errors, never what they said. Kept in memory only and installed first in main.ts so it sees
 * the page load; the real-launch check reads it through WebDriver (daemon/tests/platform_real/app_launch_check.py).
 * Nothing is sent anywhere or written to disk.
 *
 * An error's message can quote what a person typed or what a page showed (a JSON parse error quotes its input), so it
 * is not kept. Each entry is the kind, the error's class name and the place in the app's own code (file name and
 * line), or for a policy violation the directive and the blocked address's scheme and host. At most LIMIT entries of
 * at most MAX_TEXT characters each.
 */
export type RecordedError = { kind: "error" | "rejection" | "policy"; message: string };

export const LIMIT = 50;
export const MAX_TEXT = 120;

function line(text: string): string {
  return text.replace(/\s+/g, " ").trim().slice(0, MAX_TEXT);
}

function className(value: unknown): string {
  if (value instanceof Error) return value.name || "Error";
  return value === null ? "null" : typeof value;
}

function place(filename: string | undefined, lineno: number | undefined): string {
  if (!filename) return "unknown place";
  const file = filename.split(/[?#]/)[0].split("/").pop() || "page";
  return `${file}:${lineno ?? 0}`;
}

function origin(address: string): string {
  if (!address || !address.includes(":")) return address || "inline";
  try {
    const url = new URL(address);
    return url.host ? `${url.protocol}//${url.host}` : url.protocol;
  } catch {
    return address.split(":")[0]; // "inline", "eval", "wasm-eval"
  }
}

export function installErrorRecorder(target: Window = window): RecordedError[] {
  const recorded: RecordedError[] = [];
  const keep = (kind: RecordedError["kind"], message: string) => {
    if (recorded.length >= LIMIT) recorded.shift();
    recorded.push({ kind, message: line(message) });
  };
  target.addEventListener("error", (event) => {
    const error = event as ErrorEvent;
    keep("error", `${className(error.error)} at ${place(error.filename, error.lineno)}`);
  });
  target.addEventListener("unhandledrejection", (event) => {
    keep("rejection", className((event as PromiseRejectionEvent).reason));
  });
  target.document.addEventListener("securitypolicyviolation", (event) => {
    const violation = event as SecurityPolicyViolationEvent;
    keep("policy", `${violation.effectiveDirective} blocked ${origin(violation.blockedURI)}`);
  });
  Object.defineProperty(target, "__helioxErrors", { value: recorded, configurable: false, writable: false });
  return recorded;
}
