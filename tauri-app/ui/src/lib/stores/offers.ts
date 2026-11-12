/**
 * Offers in the app: "You usually open Outlook about now. Want me to open it?"
 *
 * The daemon computes offers on this computer (pilot.personal.offers) and sends each one as an `offer`
 * notification; this store shows it as a card next to Heliox Live's cards and sends the person's answer back:
 * yes (the request goes down the normal road, with every approval), not today, don't offer this again, or why.
 * An offer answered by voice or in chat arrives as `offer_withdrawn` and its card goes.
 */
import { writable } from "svelte/store";
import { call, onNotification } from "../api/daemon";

export interface OfferCard {
  id: string;
  key: string;
  kind: string;
  text: string;
  request: string;
}

export type OfferAnswer = "yes" | "not_today" | "silence" | "why";

export interface SilencedOffer {
  key: string;
  label: string;
}

export interface OffersStatus {
  status: string;
  mode: "auto" | "on" | "off";
  active: boolean;
  waiting_for_profile: boolean;
  silenced: SilencedOffer[];
  /** Whether the getting-to-know offer may be made while the profile is empty. */
  intro?: boolean;
  per_hour: number;
  per_day: number;
  reply?: string;
  message?: string;
}

export interface OffersState {
  cards: OfferCard[];
  /** The answer to "why?" for a card, by card id. */
  whys: Record<string, string>;
}

const MAX_CARDS = 2;

function createOffersStore() {
  const { subscribe, update, set } = writable<OffersState>({ cards: [], whys: {} });

  function drop(id: string) {
    update((s) => ({ ...s, cards: s.cards.filter((card) => card.id !== id) }));
  }

  onNotification((method, params) => {
    const p = (params ?? {}) as Record<string, unknown>;
    if (method === "offer") {
      const card: OfferCard = {
        id: String(p.id ?? ""),
        key: String(p.key ?? ""),
        kind: String(p.kind ?? ""),
        text: String(p.text ?? ""),
        request: String(p.request ?? ""),
      };
      if (!card.id || !card.text) return;
      update((s) => ({ ...s, cards: [...s.cards.filter((c) => c.id !== card.id), card].slice(-MAX_CARDS) }));
    } else if (method === "offer_withdrawn") {
      drop(String(p.id ?? ""));
    }
  });

  async function answer(card: OfferCard, how: OfferAnswer): Promise<void> {
    if (how !== "why") drop(card.id);
    try {
      const result = await call<OffersStatus>("offers", { action: "answer", id: card.id, answer: how });
      if (how === "why") {
        const text = result.status === "ok" && result.reply ? result.reply : "That offer is no longer available.";
        update((s) => ({ ...s, whys: { ...s.whys, [card.id]: text } }));
      }
    } catch {
      if (how === "why") update((s) => ({ ...s, whys: { ...s.whys, [card.id]: "Heliox is not connected." } }));
    }
  }

  return { subscribe, answer, reset: () => set({ cards: [], whys: {} }) };
}

export const offers = createOffersStore();

/** The Settings side: the mode, and the offers the person silenced. */
export async function offersStatus(): Promise<OffersStatus> {
  return call<OffersStatus>("offers", { action: "status" });
}

export async function setOffersMode(mode: OffersStatus["mode"]): Promise<OffersStatus> {
  return call<OffersStatus>("offers", { action: "set", mode });
}

export async function setIntroOffer(on: boolean): Promise<OffersStatus> {
  return call<OffersStatus>("offers", { action: "intro", on });
}

export async function unsilenceOffer(key: string): Promise<OffersStatus> {
  return call<OffersStatus>("offers", { action: "unsilence", key });
}
