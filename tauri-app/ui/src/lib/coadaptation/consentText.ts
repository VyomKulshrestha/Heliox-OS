import { englishText } from "./messages";
import { PURPOSES } from "./types";

/**
 * The canonical consent text and its SHA-256 (spec section 10.4). The daemon stores the hash with the consent and
 * refuses `session_begin` when the UI's current hash differs, so the person reviews any changed text again.
 */

export const CONSENT_TEXT_VERSION = "coadapt-consent-v1";

/** Keys of the canonical text, in order; each purpose contributes one line "label — detail". */
export const CONSENT_KEYS: readonly string[] = [
  "coadaptation.consent.title",
  "coadaptation.consent.intro",
  "coadaptation.consent.measured",
  "coadaptation.consent.not_measured",
  "coadaptation.consent.where",
  "coadaptation.consent.purposes_note",
  ...PURPOSES.flatMap((purpose) => [
    `coadaptation.consent.purpose.${purpose}.label`,
    `coadaptation.consent.purpose.${purpose}.detail`,
  ]),
  "coadaptation.consent.retention",
  "coadaptation.consent.nonclinical",
  "coadaptation.consent.rights",
  "coadaptation.consent.authority",
];

/** The keys of section 10.4 joined with "\n" in order; purposes as "label — detail" in enum order. */
export function canonicalConsentText(t: (key: string) => string): string {
  const lines = [
    t("coadaptation.consent.title"),
    t("coadaptation.consent.intro"),
    t("coadaptation.consent.measured"),
    t("coadaptation.consent.not_measured"),
    t("coadaptation.consent.where"),
    t("coadaptation.consent.purposes_note"),
    ...PURPOSES.map(
      (purpose) =>
        `${t(`coadaptation.consent.purpose.${purpose}.label`)} — ${t(`coadaptation.consent.purpose.${purpose}.detail`)}`,
    ),
    t("coadaptation.consent.retention"),
    t("coadaptation.consent.nonclinical"),
    t("coadaptation.consent.rights"),
    t("coadaptation.consent.authority"),
  ];
  return lines.join("\n");
}

/** The canonical text built from the English strings (the consent keys have no translation). */
export function englishConsentText(): string {
  return canonicalConsentText((key) => englishText(key));
}

/** Lowercase hex SHA-256 of the UTF-8 text, computed with WebCrypto. */
export async function consentTextSha256(text: string): Promise<string> {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
