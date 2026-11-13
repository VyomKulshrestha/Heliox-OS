import { offNotification, onNotification } from "./daemon";

/**
 * Run `reload` when the daemon announces a saved change that touches this panel.
 *
 * The daemon sends `config_updated` with `{ method, section }` after every successful
 * settings write, whoever made it: another window, a voice command, a reset, or a hand
 * edit of config.toml picked up on the next write (`method: "reload"`). `methods` lists
 * the dedicated RPCs that own the panel's settings; `sections` lists `update_config`
 * sections. Returns a function that stops listening.
 */
export function onConfigChange(
  match: { methods?: readonly string[]; sections?: readonly string[] },
  reload: () => void,
): () => void {
  const handler = (method: string, params: unknown) => {
    if (method !== "config_updated") return;
    const change = (params ?? {}) as { method?: string; section?: string };
    const changedMethod = change.method ?? "";
    if (
      changedMethod === "reload" ||
      changedMethod === "reset_config" ||
      match.methods?.includes(changedMethod) ||
      (changedMethod === "update_config" && match.sections?.includes(change.section ?? ""))
    ) {
      reload();
    }
  };
  try {
    onNotification(handler);
  } catch {
    return () => {};
  }
  return () => {
    try {
      offNotification(handler);
    } catch {
      /* ignore */
    }
  };
}

/** What a panel says when the person did not approve a setting change: it is left as it was. */
export const NOT_CHANGED = "Not changed.";

/**
 * A setting change that takes a protection away waits for the person's approval in the daemon's own approval
 * dialog. When they deny it, leave it unanswered or approve only part of it, the daemon changes nothing and replies
 * `{ status: "cancelled" }`. That is not an error: the panel puts the control back to the saved value.
 */
export function isCancelled(result: unknown): boolean {
  return Boolean(result) && typeof result === "object" && (result as { status?: unknown }).status === "cancelled";
}

/** The fields of `next` whose values differ from `saved` (compared as JSON). */
export function changedFields<T extends Record<string, unknown>>(saved: T, next: T): Partial<T> {
  const changed: Partial<T> = {};
  for (const key of Object.keys(next) as Array<keyof T>) {
    if (JSON.stringify(saved[key]) !== JSON.stringify(next[key])) changed[key] = next[key];
  }
  return changed;
}
