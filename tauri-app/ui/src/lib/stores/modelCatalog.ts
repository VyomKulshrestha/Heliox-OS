// The cloud models Settings offers, read from the daemon (model_catalog, pilot.models.catalog): the panel never keeps
// a list of its own, so it cannot offer a model the provider has retired.
import { writable } from "svelte/store";
import { call } from "../api/daemon";

export type Withdrawn = {
  state: "deprecated" | "retired" | "unlisted" | "closed";
  successor: string;
  date: string;
  says: string;
};
export type ProviderModels = {
  planner_default: string;
  talk_default: string;
  // "Default chosen from <provider>'s published list on <date>; not yet checked ..." until a live run has answered.
  planner_note?: string;
  talk_note?: string;
  offered: string[];
  withdrawn: Record<string, Withdrawn>;
};
export type ModelCatalog = { read_on: string; providers: Record<string, ProviderModels> };

export const modelCatalog = writable<ModelCatalog | null>(null);

let loading: Promise<void> | null = null;

export function loadModelCatalog(): Promise<void> {
  loading ??= (async () => {
    try {
      const found = (await call("model_catalog")) as (ModelCatalog & { status?: string }) | undefined;
      if (found?.providers) modelCatalog.set({ read_on: found.read_on, providers: found.providers });
    } catch {
      loading = null; // tried again the next time the panel opens
    }
  })();
  return loading;
}

const SNAPSHOT = /^-(?:\d{8}|\d{4}-\d{2}-\d{2}|latest)$/;

/** The withdrawal of a saved model id (a dated snapshot or "-latest" alias counts as its model), as the daemon does. */
export function withdrawnFor(catalog: ModelCatalog | null, provider: string, model: string): Withdrawn | null {
  if (!catalog || !model) return null;
  let key = model.trim().toLowerCase();
  let models = catalog.providers[provider];
  if (provider === "openrouter") {
    // An OpenRouter slug is its author's model: "anthropic/claude-3.7-sonnet" is claude-3-7-sonnet.
    const slash = key.indexOf("/");
    const inner = { openai: "openai", anthropic: "claude", google: "gemini", meta: "meta" }[key.slice(0, slash)];
    if (slash < 0 || !inner) return null;
    key = inner === "claude" ? key.slice(slash + 1).replaceAll(".", "-") : key.slice(slash + 1);
    models = catalog.providers[inner];
  }
  if (!models) return null;
  for (const [known, found] of Object.entries(models.withdrawn)) {
    if (key === known || (key.startsWith(known) && SNAPSHOT.test(key.slice(known.length)))) return found;
  }
  return null;
}

const MAKERS: Record<string, string> = {
  claude: "Anthropic",
  openai: "OpenAI",
  gemini: "Google",
  openrouter: "its provider",
  meta: "Meta",
};

/** The line shown under a saved model that is no longer current: never replaced, only named, with a suggestion. */
export function withdrawnLine(provider: string, model: string, found: Withdrawn): string {
  const who = MAKERS[provider] ?? provider;
  const on = found.date ? ` on ${found.date}` : "";
  if (found.state === "unlisted") return `${model} is no longer on ${who}'s list of models.`;
  if (found.state === "closed") return `${model} is no longer offered to new ${who} accounts.`;
  if (found.state === "deprecated") return `${model} is deprecated by ${who} and stops working${on}.`;
  return `${model} was retired by ${who}${on}.`;
}

export function resetModelCatalogForTests() {
  loading = null;
  modelCatalog.set(null);
}
