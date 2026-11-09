/**
 * Heliox Live in the UI: the companion's presence and the correction cards it raises.
 *
 * The daemon decides when to speak (pilot.live); this store shows what it said as cards that float over any tab,
 * raises an OS notification when Heliox is not the window in front (most corrections happen in other apps),
 * and sends the person's response back: helpful, not now, don't mention this again, or "fix it".
 */
import { writable } from "svelte/store";
import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import { call, onNotification } from "../api/daemon";
import { isTauriRuntime } from "../utils/runtime";
import { session } from "./session";

export interface LiveCard {
  id: string;
  kind: string;
  title: string;
  message: string;
  detail: string;
  evidence: string;
  fix: string;
  fix_kind: "replace" | "command" | "ask" | "";
  urgency: "urgent" | "normal" | "low";
  spoken: boolean;
  source: string;
  app: string;
}

export interface LiveState {
  enabled: boolean;
  paused: boolean;
  watching: string;
  userState: string;
  cards: LiveCard[];
  /** "why?" answers fetched for a card, by card id. */
  explanations: Record<string, string>;
  /** Brief confirmation shown on a card after an action, by card id. */
  notes: Record<string, string>;
}

const INITIAL: LiveState = {
  enabled: false,
  paused: false,
  watching: "",
  userState: "off",
  cards: [],
  explanations: {},
  notes: {},
};

const MAX_CARDS = 3;

async function notifyIfAway(card: LiveCard) {
  if (!isTauriRuntime() || card.urgency === "low") return;
  if (typeof document !== "undefined" && document.hasFocus()) return;
  try {
    let granted = await isPermissionGranted();
    if (!granted) granted = (await requestPermission()) === "granted";
    if (granted) sendNotification({ title: card.title, body: card.message });
  } catch {
    /* notifications are a courtesy; the card is still there */
  }
}

function createLiveStore() {
  const { subscribe, update, set } = writable<LiveState>({ ...INITIAL });

  function drop(id: string) {
    update((s) => ({ ...s, cards: s.cards.filter((card) => card.id !== id) }));
  }

  onNotification((method, params) => {
    const p = (params ?? {}) as Record<string, unknown>;
    if (method === "live_interjection") {
      const card = p as unknown as LiveCard;
      update((s) => ({ ...s, cards: [...s.cards.filter((c) => c.id !== card.id), card].slice(-MAX_CARDS) }));
      void notifyIfAway(card);
    } else if (method === "live_interjection_withdrawn") {
      const id = String(p.id ?? "");
      if (p.reason === "fixed") {
        // A beat of acknowledgement before the card goes: they fixed it.
        update((s) => ({ ...s, notes: { ...s.notes, [id]: "Fixed" } }));
        setTimeout(() => drop(id), 1200);
      } else {
        drop(id);
      }
    } else if (method === "live_state") {
      update((s) => ({
        ...s,
        enabled: Boolean(p.enabled),
        paused: Boolean(p.paused),
        watching: String(p.watching ?? ""),
        userState: String(p.user_state ?? "off"),
        cards: p.enabled ? s.cards : [],
      }));
    }
  });

  async function load() {
    try {
      const status = await call<Record<string, unknown>>("live_status");
      if (status.status !== "ok") return;
      update((s) => ({
        ...s,
        enabled: Boolean(status.enabled),
        paused: Boolean(status.paused),
        watching: String(status.watching ?? ""),
        userState: String(status.user_state ?? "off"),
        cards: (status.visible as LiveCard[] | undefined) ?? s.cards,
      }));
    } catch {
      /* daemon not connected yet */
    }
  }

  async function setPaused(paused: boolean) {
    update((s) => ({ ...s, paused }));
    try {
      await call("live_set", { paused });
    } catch {
      update((s) => ({ ...s, paused: !paused }));
    }
  }

  async function feedback(card: LiveCard, verdict: "helpful" | "dismissed" | "used_fix" | "mute") {
    drop(card.id);
    try {
      await call("live_feedback", { id: card.id, verdict });
    } catch {
      /* the card is gone either way */
    }
  }

  async function explain(card: LiveCard) {
    try {
      const result = await call<{ status: string; text: string }>("live_explain", { id: card.id });
      const text = result.status === "ok" ? result.text : card.detail || card.message;
      update((s) => ({ ...s, explanations: { ...s.explanations, [card.id]: text } }));
    } catch {
      update((s) => ({ ...s, explanations: { ...s.explanations, [card.id]: card.detail || card.message } }));
    }
  }

  async function copyFix(card: LiveCard) {
    try {
      await writeText(card.fix);
      update((s) => ({ ...s, notes: { ...s.notes, [card.id]: "Copied" } }));
      setTimeout(() => void feedback(card, "used_fix"), 900);
    } catch {
      update((s) => ({ ...s, notes: { ...s.notes, [card.id]: "Could not copy" } }));
    }
  }

  async function askHeliox(card: LiveCard) {
    // The background model does the work, through the same planner and approvals as a typed request.
    await feedback(card, "used_fix");
    await session.sendCommand(card.fix);
  }

  return { subscribe, load, setPaused, feedback, explain, copyFix, askHeliox, reset: () => set({ ...INITIAL }) };
}

export const live = createLiveStore();
