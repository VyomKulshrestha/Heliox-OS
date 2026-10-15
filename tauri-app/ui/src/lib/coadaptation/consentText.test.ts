import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import en from "../../locales/coadaptation.en.json";
import hi from "../../locales/coadaptation.hi.json";
import {
  CONSENT_KEYS,
  CONSENT_TEXT_VERSION,
  canonicalConsentText,
  consentTextSha256,
  englishConsentText,
} from "./consentText";
import { englishText } from "./messages";

function lookup(dictionary: unknown, key: string): unknown {
  return key
    .split(".")
    .reduce<unknown>((node, part) => (node as Record<string, unknown> | undefined)?.[part], dictionary);
}

describe("canonical consent text", () => {
  it("joins the section 10.4 keys in order, purposes as label — detail in enum order", () => {
    const text = englishConsentText();
    const lines = text.split("\n");
    expect(lines).toHaveLength(16);
    expect(lines[0]).toBe("Assistance study: how Heliox helps you select things");
    expect(lines[5]).toBe(englishText("coadaptation.consent.purposes_note"));
    expect(lines[6]).toBe("Run the selection study — Store study trials, Heliox's choices and your ratings.");
    expect(lines.slice(6, 12).map((line) => line.split(" — ")[0])).toEqual([
      "Run the selection study",
      "Let Heliox adapt during later sessions",
      "Remember across sessions",
      "Use coarse gaze direction",
      "Use the hand pointer",
      "Allow encrypted export",
    ]);
    expect(lines[15]).toBe(englishText("coadaptation.consent.authority"));
    expect(CONSENT_TEXT_VERSION).toBe("coadapt-consent-v1");
  });

  it("uses the exact English of section 10.4", () => {
    expect(englishText("coadaptation.consent.intro")).toBe(
      "Heliox can run a short selection study inside its own window. It measures how you select targets and menu items, tries different kinds of help, and estimates which help suits you. After a few sessions that estimate still rests mostly on general data about how people select, not on yours. You can stop at any time.",
    );
    expect(englishText("coadaptation.consent.retention")).toBe(
      "Your data is erased automatically after the period you choose (default 90 days), or immediately when you withdraw.",
    );
  });

  it("hashes to a stable lowercase SHA-256 of the UTF-8 text", async () => {
    const text = englishConsentText();
    const hash = await consentTextSha256(text);
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
    expect(hash).toBe(createHash("sha256").update(text, "utf8").digest("hex"));
    expect(await consentTextSha256(englishConsentText())).toBe(hash);
  });

  it("changes the hash when any key changes", async () => {
    const original = await consentTextSha256(englishConsentText());
    for (const key of CONSENT_KEYS) {
      const changed = canonicalConsentText((k) => (k === key ? `${englishText(k)}.` : englishText(k)));
      expect(await consentTextSha256(changed), key).not.toBe(original);
    }
  });

  it("has no approval purpose and makes no claim that Heliox learns which help works", () => {
    for (const key of CONSENT_KEYS) expect(key).not.toMatch(/approv/);
    const purposes = Object.keys(
      (en as { coadaptation: { consent: { purpose: object } } }).coadaptation.consent.purpose,
    );
    expect(purposes).toEqual([
      "selection_study",
      "adaptive_assistance",
      "cross_session_learning",
      "gaze_prior",
      "hand_pointer",
      "research_export",
    ]);
    const text = englishConsentText();
    expect(text).not.toMatch(/learns what works|learns which help works|found the best/i);
    expect(text).not.toMatch(/stress|attention|cognitive load|health status/i);
  });

  it("keeps the consent keys out of the Hindi file but carries the English-only notice", () => {
    for (const key of CONSENT_KEYS) expect(lookup(hi, key)).toBeUndefined();
    expect(typeof lookup(hi, "coadaptation.consent.english_only_notice")).toBe("string");
  });
});
