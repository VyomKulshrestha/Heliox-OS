<script lang="ts">
  /**
   * Consent sheet (spec 10.3, 10.4). Every purpose starts unticked, selection_study included, and Grant stays disabled
   * until it is ticked. The text shown is exactly the canonical English text whose SHA-256 is sent with the grant; the
   * Hindi UI shows it under a reviewed one-line notice. Closing the sheet grants nothing.
   *
   * With `participantKey` the sheet updates that participant's consent: the current purposes start ticked, it says
   * what unticking deletes and that the erase date never moves later, and granted items stay ticked while a study
   * session is open.
   */
  import { _, locale } from "svelte-i18n";
  import { englishText } from "../../coadaptation/messages";
  import { coadaptation } from "../../stores/coadaptation";
  import { CONSENT_TEXT_VERSION, consentTextSha256, englishConsentText } from "../../coadaptation/consentText";
  import { PURPOSES, type CoAdaptationPurpose } from "../../coadaptation/types";
  import type { Translate } from "./AssistChip.svelte";

  interface Props {
    participantKey?: string;
    onclose: () => void;
    ongranted: (participantKey: string) => void;
    /** Turn the feature on with the grant (the grant RPC's `enable`). */
    enable?: boolean;
    /** Overrides the store's view of whether a study session is open for this participant. */
    sessionOpen?: boolean;
  }

  let { participantKey, onclose, ongranted, enable = true, sessionOpen }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);
  /** The consent text is always the English text that is hashed (section 10.4). */
  const en = (key: string): string => englishText(`coadaptation.consent.${key}`);

  const RETENTION_DAYS = [30, 90, 180, 365];
  const DEFAULT_RETENTION = 90;

  const participant = $derived(
    participantKey
      ? (($coadaptation.status?.participants ?? []).find((entry) => entry.participant_key === participantKey) ?? null)
      : null,
  );
  const current = $derived(participant?.consent ?? null);
  const updating = $derived(Boolean(participantKey && current?.active));
  const grantedBefore = $derived(new Set<CoAdaptationPurpose>(updating ? (current?.purposes ?? []) : []));
  const studyOpen = $derived(
    sessionOpen ??
      ($coadaptation.study.active &&
        (!$coadaptation.study.participantKey || $coadaptation.study.participantKey === participantKey)),
  );

  let checked = $state<Record<CoAdaptationPurpose, boolean>>({
    selection_study: false,
    adaptive_assistance: false,
    cross_session_learning: false,
    gaze_prior: false,
    hand_pointer: false,
    research_export: false,
  });
  let retention = $state(DEFAULT_RETENTION);
  let label = $state("");
  let busy = $state(false);
  let error = $state("");
  let initializedFor = "";

  // In update mode the current purposes and retention start selected, once per participant.
  $effect(() => {
    const key = participantKey ?? "";
    if (!updating || !current || initializedFor === key) return;
    initializedFor = key;
    for (const purpose of PURPOSES) checked[purpose] = current.purposes.includes(purpose);
    retention = RETENTION_DAYS.includes(current.retention_days) ? current.retention_days : DEFAULT_RETENTION;
  });

  const removed = $derived(PURPOSES.filter((purpose) => grantedBefore.has(purpose) && !checked[purpose]));
  const canGrant = $derived(checked.selection_study && !busy && !(studyOpen && removed.length > 0));
  const showHindiNotice = $derived(($locale ?? "").toLowerCase().startsWith("hi"));
  const expiresOn = $derived(current?.expires_at ? new Date(current.expires_at).toLocaleDateString() : "");

  /** While a session is open a granted purpose cannot be unticked (the daemon refuses with session_active). */
  function lockedWhileOpen(purpose: CoAdaptationPurpose): boolean {
    return updating && studyOpen && grantedBefore.has(purpose) && checked[purpose];
  }

  async function grant() {
    if (!canGrant) return;
    busy = true;
    error = "";
    try {
      const sha256 = await consentTextSha256(englishConsentText());
      const protocol = $coadaptation.status?.protocol ?? null;
      const grantedKey = await coadaptation.grantConsent({
        participant_key: updating && participantKey ? participantKey : null,
        // The optional label of a new participant travels only in this call and is stored encrypted.
        label: updating ? null : label.trim().slice(0, 32) || null,
        purposes: PURPOSES.filter((purpose) => checked[purpose]),
        retention_days: retention,
        consent_text_version: CONSENT_TEXT_VERSION,
        consent_text_sha256: sha256,
        protocol_id: protocol?.protocol_id ?? null,
        protocol_sha256: protocol?.sha256 ?? null,
        authorized: true,
        enable,
      });
      ongranted(grantedKey);
    } catch (cause) {
      error = t("coadaptation.consent.grant_failed", {
        values: { message: cause instanceof Error ? cause.message : String(cause) },
      });
    } finally {
      busy = false;
    }
  }
</script>

<div class="consent-sheet" role="dialog" aria-modal="true" aria-labelledby="coadapt-consent-title">
  {#if showHindiNotice}
    <p class="language-notice" lang="hi">{t("coadaptation.consent.english_only_notice")}</p>
  {/if}

  <div class="consent-text" lang="en">
    <h3 id="coadapt-consent-title">{en("title")}</h3>
    <p>{en("intro")}</p>
    <p>{en("measured")}</p>
    <p>{en("not_measured")}</p>
    <p>{en("where")}</p>
    <p class="purposes-note">{en("purposes_note")}</p>

    <div class="purposes">
      {#each PURPOSES as purpose (purpose)}
        <label class="purpose">
          <input
            type="checkbox"
            name={purpose}
            bind:checked={checked[purpose]}
            disabled={busy || lockedWhileOpen(purpose)}
          />
          <span>
            <strong>{en(`purpose.${purpose}.label`)}</strong> — {en(`purpose.${purpose}.detail`)}
            {#if purpose === "selection_study"}
              <em class="required" lang={$locale ?? "en"}>{t("coadaptation.consent.required")}</em>
            {/if}
          </span>
        </label>
      {/each}
    </div>

    <p>{en("retention")}</p>
    <p>{en("nonclinical")}</p>
    <p>{en("rights")}</p>
    <p class="authority">{en("authority")}</p>
  </div>

  <div class="consent-controls">
    <label class="field">
      <span>{t("coadaptation.consent.retention_label")}</span>
      <select bind:value={retention} disabled={busy}>
        {#each RETENTION_DAYS as days (days)}
          <option value={days}>
            {t(
              days === DEFAULT_RETENTION
                ? "coadaptation.consent.retention_default"
                : "coadaptation.consent.retention_days",
              {
                values: { days },
              },
            )}
          </option>
        {/each}
      </select>
    </label>

    {#if !updating}
      <label class="field">
        <span>{t("coadaptation.consent.label_label")}</span>
        <input type="text" maxlength="32" bind:value={label} disabled={busy} autocomplete="off" />
        <small>{t("coadaptation.consent.label_hint")}</small>
      </label>
    {/if}

    {#if updating}
      <div class="update-note" role="note">
        <p>{t("coadaptation.consent.update_deletion_note", { values: { expires_at: expiresOn } })}</p>
        {#if removed.length}
          <p class="deletes-title">{t("coadaptation.consent.update_deletes_title")}</p>
          <ul class="deletions">
            {#each removed.filter((purpose) => purpose !== "selection_study") as purpose (purpose)}
              <li>
                <strong>{en(`purpose.${purpose}.label`)}</strong>: {t(`coadaptation.consent.deletes.${purpose}`)}
              </li>
            {/each}
          </ul>
        {/if}
        {#if studyOpen}
          <p class="session-note">{t("coadaptation.consent.session_open_untick")}</p>
        {/if}
      </div>
    {/if}

    {#if error}
      <p class="error" role="alert">{error}</p>
    {/if}

    <div class="actions">
      <button class="secondary" onclick={() => onclose()} disabled={busy}>{t("coadaptation.consent.close")}</button>
      <button class="primary" onclick={grant} disabled={!canGrant}>
        {busy
          ? t("coadaptation.consent.granting")
          : updating
            ? t("coadaptation.consent.update_title")
            : t("coadaptation.consent.grant")}
      </button>
    </div>
  </div>
</div>

<style>
  .consent-sheet {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-height: 80vh;
    padding: 14px 16px;
    overflow-y: auto;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }

  .language-notice {
    margin: 0;
    padding: 8px 10px;
    font-size: 12px;
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-sm);
  }

  .consent-text {
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 12px;
    line-height: 1.5;
  }

  h3 {
    margin: 0;
    font-size: 14px;
  }

  p {
    margin: 0;
  }

  .purposes-note,
  .authority {
    font-weight: 600;
  }

  .purposes {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .purpose {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    cursor: pointer;
  }

  .purpose input {
    margin-top: 3px;
    accent-color: var(--accent);
  }

  .required {
    margin-left: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .consent-controls {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .field small {
    font-size: 11px;
    color: var(--text-muted);
  }

  select,
  input[type="text"] {
    max-width: 320px;
    padding: 5px 8px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .update-note {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 8px 10px;
    font-size: 12px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .deletes-title {
    font-weight: 600;
  }

  .deletions {
    margin: 0;
    padding-left: 18px;
  }

  .session-note,
  .error {
    color: var(--danger);
  }

  .error {
    font-size: 12px;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  button {
    padding: 6px 14px;
    font-size: 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .secondary {
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
  }

  .primary {
    font-weight: 600;
    color: white;
    background: var(--accent);
    border: 1px solid var(--accent);
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
</style>
