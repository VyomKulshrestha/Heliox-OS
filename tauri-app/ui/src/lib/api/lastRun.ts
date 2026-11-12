/**
 * Heliox's last run, when it did not end cleanly: the daemon puts it in its startup_status answer as `last_run`
 * (pilot.system.run_record). The window says it once per unclean end, as one quiet line. A computer restart has no
 * notice and is never said, and voice says nothing about it.
 */

export type LastRun = {
  ended?: string;
  notice?: string | null;
  id?: string;
  /** When the run that did not end cleanly started, in seconds since the epoch. */
  started_at?: number | null;
  unclean_today?: number;
};

const SHOWN_KEY = "heliox.lastRunShown";

type Store = Pick<Storage, "getItem" | "setItem">;

function shownStore(): Store | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null; // a private window or blocked storage: remember for this page only
  }
}

let shownThisPage = "";

/** The daemon writes the start as "(started 22:10)" in its own process's zone and a fixed 24-hour form; the window
 * says it in the person's own time zone and format instead, from the time itself. */
function inLocalTime(notice: string, startedAt: unknown, locales?: string | string[]): string {
  if (typeof startedAt !== "number" || !Number.isFinite(startedAt)) return notice;
  const local = new Date(startedAt * 1000).toLocaleTimeString(locales, { timeStyle: "short" });
  return notice.replace(/\(started [^)]*\)/, `(started ${local})`);
}

/** The line to show for this answer, once: null when there is nothing to say or it was already said. */
export function lastRunLine(
  reply: unknown,
  store: Store | null = shownStore(),
  locales?: string | string[],
): string | null {
  const lastRun = (reply as { last_run?: LastRun } | null)?.last_run;
  const notice = typeof lastRun?.notice === "string" ? lastRun.notice.trim() : "";
  const id = String(lastRun?.id ?? "");
  if (!notice || !id) return null;
  let shown = shownThisPage;
  try {
    shown = store?.getItem(SHOWN_KEY) ?? shownThisPage;
  } catch {
    // unreadable storage: the page's own memory decides
  }
  if (shown === id) return null;
  shownThisPage = id;
  try {
    store?.setItem(SHOWN_KEY, id);
  } catch {
    // unwritable storage: said once for this page
  }
  return inLocalTime(notice, lastRun?.started_at, locales);
}

/** For tests: forget what this page has shown. */
export function resetLastRunForTests(): void {
  shownThisPage = "";
}
