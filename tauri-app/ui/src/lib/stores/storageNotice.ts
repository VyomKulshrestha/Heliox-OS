import { get, writable, type Readable } from "svelte/store";

/**
 * A full disk or a read-only data folder, in the daemon's own words, for the one strip every screen shows
 * (StorageNotice.svelte). Whatever panel asked, the caller still gets its rejection as before; this is only the place
 * the person is told. The daemon says it three ways, and each sets the same notice, so a burst of failures shows it
 * once:
 * - an RPC error with code STORAGE_ERROR_CODE, or data.storage_problem (every method but execute and resume_task);
 * - an execute or resume_task result with storage_problem: true;
 * - the storage_problem notification, which also says when the problem is over (kind "").
 * The notice goes when the daemon says so, or when a method that failed this way later succeeds.
 */
export const STORAGE_ERROR_CODE = -32010;

export type StorageNotice = { message: string; kind: string };

const notice = writable<StorageNotice | null>(null);
const failedMethods = new Set<string>();

export const storageNotice: Readable<StorageNotice | null> = { subscribe: notice.subscribe };

/** The daemon's sentence about a full or read-only store; an empty sentence is ignored. */
export function reportStorageProblem(message: string, kind = "", method = ""): void {
  const text = message.trim();
  if (!text) return;
  if (method) failedMethods.add(method);
  const current = get(notice);
  // The same sentence again changes nothing: subscribers are told once for a burst (svelte stores re-notify on
  // every set of an object, even an equal one).
  if (current && current.message === text && current.kind === kind) return;
  notice.set({ message: text, kind });
}

/** The problem is over: the daemon said so, or a request that had failed this way has now succeeded. */
export function clearStorageProblem(): void {
  failedMethods.clear();
  if (get(notice) !== null) notice.set(null);
}

/** A request succeeded: if its method had failed for lack of room or a read-only folder, the notice goes. */
export function storageRequestSucceeded(method: string): void {
  if (failedMethods.has(method)) clearStorageProblem();
}

type RpcError = { code?: number; message?: string; data?: { storage_problem?: boolean; kind?: string } };

/** Whether an RPC error is the daemon's storage answer; if so it is reported. */
export function noteStorageError(method: string, error: RpcError | undefined): void {
  if (!error) return;
  if (error.code === STORAGE_ERROR_CODE || error.data?.storage_problem) {
    reportStorageProblem(String(error.message ?? ""), String(error.data?.kind ?? ""), method);
  }
}

/** Whether a result is the daemon's storage answer (execute, resume_task); if so it is reported, else it counts as
 * a success of its method. */
export function noteStorageResult(method: string, result: unknown): void {
  const value = result as { storage_problem?: boolean; message?: unknown } | null;
  if (value && typeof value === "object" && value.storage_problem) {
    reportStorageProblem(String(value.message ?? ""), "", method);
  } else {
    storageRequestSucceeded(method);
  }
}

/** The daemon's storage_problem notification: a kind and sentence, or kind "" when it can save again. */
export function noteStorageNotification(params: unknown): void {
  const value = (params ?? {}) as { kind?: unknown; message?: unknown };
  const kind = String(value.kind ?? "");
  if (!kind) clearStorageProblem();
  else reportStorageProblem(String(value.message ?? ""), kind);
}
