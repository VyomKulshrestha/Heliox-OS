import { get } from "svelte/store";
import { _, addMessages } from "svelte-i18n";
import en from "../../locales/coadaptation.en.json";
import hi from "../../locales/coadaptation.hi.json";

/**
 * Registers the co-adaptation strings (`coadaptation.*` and `study.*`) with svelte-i18n once. Every co-adaptation
 * component imports this module; `src/i18n.ts` initialises the locale. The Hindi file holds reviewed strings only:
 * consent keys are absent there and fall back to English.
 */

let registered = false;

export function registerCoadaptationMessages(): void {
  if (registered) return;
  registered = true;
  addMessages("en", en);
  addMessages("hi", hi);
}

registerCoadaptationMessages();

type Values = Record<string, string | number>;

function lookup(dictionary: unknown, key: string): string | null {
  let node: unknown = dictionary;
  for (const part of key.split(".")) {
    if (node === null || typeof node !== "object") return null;
    node = (node as Record<string, unknown>)[part];
  }
  return typeof node === "string" ? node : null;
}

function interpolate(template: string, values: Values | undefined): string {
  if (!values) return template;
  return template.replace(/\{(\w+)\}/g, (match, name: string) => (name in values ? String(values[name]) : match));
}

/** The English string for a key, with `{name}` values filled in; the key itself when it is missing. */
export function englishText(key: string, values?: Values): string {
  const template = lookup(en, key);
  return template === null ? key : interpolate(template, values);
}

/**
 * The string for a key in the active locale, for code outside components (the store). Falls back to English when
 * no locale is set yet or the key has no translation.
 */
export function coadaptText(key: string, values?: Values): string {
  try {
    const formatted = get(_)(key, values ? { values } : undefined);
    if (typeof formatted === "string" && formatted !== key) return formatted;
  } catch {
    /* no locale initialised yet */
  }
  return englishText(key, values);
}
