<script lang="ts">
  /**
   * Settings section "Assistance & co-adaptation" (spec 16.5). Off by default; nothing is stored before a consent
   * record exists, and every option that uses data needs its own consent purpose. The feature changes how study
   * targets are shown and what a selection needs; it never approves, clicks or runs anything, so this panel has no
   * approval control. Destructive actions take two clicks, and withdrawing also needs the typed word "erase".
   */
  import { onMount, untrack } from "svelte";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { coadaptation } from "../../stores/coadaptation";
  import { changedFields } from "../../api/configSync";
  import { CONFIG_RANGES, type CoAdaptationConfigV1, type CoAdaptationPurpose } from "../../coadaptation/types";
  import type { Translate } from "./AssistChip.svelte";
  import CoAdaptationConsent from "./CoAdaptationConsent.svelte";
  import AssistanceRulesEditor from "./AssistanceRulesEditor.svelte";
  import CoAdaptationResults from "./CoAdaptationResults.svelte";

  const t: Translate = (key, options) => $_(key, options);

  type OptionField = "adaptive_enabled" | "learn_across_sessions" | "gaze_prior_enabled" | "hand_pointer_enabled";
  type AdvancedField =
    | "ambiguity_threshold"
    | "unintended_activation_tolerance"
    | "upper_bound_credibility"
    | "exploration_floor"
    | "switching_penalty"
    | "burden_prompt_every_blocks"
    | "day_boundary_hour";

  const OPTIONS: Array<{ field: OptionField; purpose: CoAdaptationPurpose; key: string }> = [
    { field: "adaptive_enabled", purpose: "adaptive_assistance", key: "coadaptation.options.adaptive" },
    { field: "learn_across_sessions", purpose: "cross_session_learning", key: "coadaptation.options.cross_session" },
    { field: "gaze_prior_enabled", purpose: "gaze_prior", key: "coadaptation.options.gaze" },
    { field: "hand_pointer_enabled", purpose: "hand_pointer", key: "coadaptation.options.hand_pointer" },
  ];

  /** The Advanced settings of 16.5 with the ranges of section 15 (the threshold can only be lowered from 0.25). */
  const ADVANCED: Array<{ field: AdvancedField; step: number }> = [
    { field: "ambiguity_threshold", step: 0.01 },
    { field: "unintended_activation_tolerance", step: 0.01 },
    { field: "upper_bound_credibility", step: 0.01 },
    { field: "exploration_floor", step: 0.01 },
    { field: "switching_penalty", step: 0.05 },
    { field: "burden_prompt_every_blocks", step: 1 },
    { field: "day_boundary_hour", step: 1 },
  ];

  let selectedKey = $state<string | null>(null);
  let consentSheet = $state<{ participantKey?: string; enable: boolean } | null>(null);
  let busy = $state(false);
  let error = $state("");
  let notice = $state("");
  /** What the last withdrawal could not delete yet (the keyring key, or export files and rules). */
  let erasureWarning = $state("");
  let withdrawing = $state<string | null>(null);
  let withdrawWord = $state("");
  let confirmingReset = $state(false);
  let confirmingErase = $state(false);
  let advanced = $state<Record<AdvancedField, number> | null>(null);
  let savedAdvanced = $state<Record<AdvancedField, number> | null>(null);

  const status = $derived($coadaptation.status);
  const config = $derived(status?.config ?? null);
  const participants = $derived(status?.participants ?? []);
  const activeParticipants = $derived(participants.filter((entry) => entry.consent?.active));
  const hasActiveConsent = $derived(activeParticipants.length > 0);
  const selected = $derived(
    activeParticipants.find((entry) => entry.participant_key === selectedKey) ?? activeParticipants[0] ?? null,
  );
  const selectedPurposes = $derived(new Set<CoAdaptationPurpose>(selected?.consent.purposes ?? []));
  const keyringUnavailable = $derived(status?.keyring_available === false);
  /**
   * What still awaits deletion. The status is the fresher source (its prune retries the pending list), so the last
   * withdrawal's detail shows only while the status does not report everything deleted.
   */
  const erasureMessage = $derived.by(() => {
    if (erasureWarning && status?.erasure_pending !== false) return erasureWarning;
    return status?.erasure_pending === true ? t("coadaptation.data.erasure_pending") : "";
  });
  const withdrawWordExpected = $derived(t("coadaptation.participants.withdraw_word"));
  const advancedDirty = $derived(
    advanced !== null && savedAdvanced !== null && Object.keys(changedFields(savedAdvanced, advanced)).length > 0,
  );
  const studyBlockedReason = $derived.by(() => {
    if (keyringUnavailable) return t("coadaptation.keyring_unavailable");
    if (!config?.enabled) return t("coadaptation.run.needs_on");
    if (!selected) return t("coadaptation.run.needs_consent");
    if ($coadaptation.study.active) return t("coadaptation.run.running");
    return "";
  });

  onMount(() => {
    void coadaptation.load();
  });

  // Keep the Advanced inputs in step with the daemon unless the person is editing them.
  $effect(() => {
    if (!config) return;
    const values = advancedValues(config);
    untrack(() => {
      if (savedAdvanced === null || !advancedDirty) {
        advanced = { ...values };
        savedAdvanced = { ...values };
      }
    });
  });

  function advancedValues(source: CoAdaptationConfigV1): Record<AdvancedField, number> {
    const values = {} as Record<AdvancedField, number>;
    for (const entry of ADVANCED) values[entry.field] = Number(source[entry.field]);
    return values;
  }

  function purposeLabel(purpose: string): string {
    return t(`coadaptation.consent.purpose.${purpose}.label`);
  }

  function formatDate(value: string | null | undefined): string {
    if (!value) return "–";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString();
  }

  function messageOf(cause: unknown): string {
    return cause instanceof Error ? cause.message : String(cause);
  }

  async function run(action: () => Promise<unknown>, success = ""): Promise<boolean> {
    busy = true;
    error = "";
    notice = "";
    erasureWarning = "";
    try {
      await action();
      notice = success;
      return true;
    } catch (cause) {
      error = messageOf(cause);
      return false;
    } finally {
      busy = false;
    }
  }

  async function toggleEnabled() {
    if (!config || keyringUnavailable) return;
    if (config.enabled) {
      await run(() => coadaptation.updateConfig({ enabled: false }));
    } else if (!hasActiveConsent) {
      // Turning on without an active consent opens the consent sheet; the grant turns the feature on.
      consentSheet = { enable: true };
    } else {
      await run(() => coadaptation.updateConfig({ enabled: true }));
    }
  }

  async function toggleOption(field: OptionField) {
    if (!config) return;
    const changed: Partial<CoAdaptationConfigV1> = { [field]: !config[field] };
    await run(() => coadaptation.updateConfig(changed));
  }

  function setAdvanced(field: AdvancedField, event: Event) {
    if (!advanced) return;
    const input = event.currentTarget as HTMLInputElement;
    const range = CONFIG_RANGES[field];
    const raw = Number(input.value);
    if (!Number.isFinite(raw)) {
      input.value = String(advanced[field]);
      return;
    }
    const clamped = Math.max(range.min, Math.min(range.max, range.integer ? Math.round(raw) : raw));
    advanced[field] = clamped;
    input.value = String(clamped);
  }

  async function saveAdvanced() {
    if (!advanced || !savedAdvanced) return;
    const changed = changedFields(savedAdvanced, advanced);
    if (!Object.keys(changed).length) {
      notice = t("coadaptation.options.nothing_changed");
      return;
    }
    const next = { ...advanced };
    busy = true;
    error = "";
    notice = "";
    try {
      // Only the changed fields are sent (16.5); an open session keeps its settings (13.3).
      await coadaptation.updateConfig(changed as Partial<CoAdaptationConfigV1>);
      savedAdvanced = next;
      notice = t("coadaptation.options.saved");
    } catch (cause) {
      error = t("coadaptation.options.save_failed", { values: { message: messageOf(cause) } });
    } finally {
      busy = false;
    }
  }

  function openConsent(participantKey?: string) {
    consentSheet = { participantKey, enable: Boolean(config?.enabled) };
  }

  async function granted(participantKey: string) {
    consentSheet = null;
    selectedKey = participantKey;
    await coadaptation.load();
  }

  async function confirmWithdraw() {
    const key = withdrawing;
    if (!key || withdrawWord.trim().toLowerCase() !== withdrawWordExpected.toLowerCase()) return;
    const done = await run(async () => {
      const left = await coadaptation.revokeConsent(key);
      // The daemon retries what it could not delete; say so instead of implying everything is gone.
      if (left.keyPending) erasureWarning = t("coadaptation.participants.withdraw_key_pending");
      else if (left.filesPending) erasureWarning = t("coadaptation.participants.withdraw_files_pending");
    });
    if (done) {
      withdrawing = null;
      withdrawWord = "";
      if (selectedKey === key) selectedKey = null;
    }
  }

  async function exportData() {
    if (!selected) return;
    const key = selected.participant_key;
    busy = true;
    error = "";
    notice = "";
    try {
      const result = await coadaptation.exportData(key);
      notice = t("coadaptation.data.export_done", { values: { path: result.path } });
    } catch (cause) {
      error = messageOf(cause);
    } finally {
      busy = false;
    }
  }

  async function resetLearning() {
    if (!selected) return;
    if (!confirmingReset) {
      confirmingReset = true;
      return;
    }
    const key = selected.participant_key;
    await run(() => coadaptation.resetLearning(key), t("coadaptation.data.reset_done"));
    confirmingReset = false;
  }

  async function eraseToday() {
    if (!selected) return;
    if (!confirmingErase) {
      confirmingErase = true;
      return;
    }
    const key = selected.participant_key;
    await run(() => coadaptation.eraseToday(key));
    confirmingErase = false;
  }
</script>

<div class="coadapt-panel">
  <div class="panel-header">
    <div>
      <h3>{t("coadaptation.title")}</h3>
      <span class="subtitle">{t("coadaptation.subtitle")}</span>
    </div>
    <div class="header-actions">
      {#if config && !keyringUnavailable}
        <span class="status-text">
          {t("coadaptation.status.label")}: {t(config.enabled ? "coadaptation.status.on" : "coadaptation.status.off")}
        </span>
        <button
          class="toggle"
          class:active={config.enabled}
          role="switch"
          aria-checked={config.enabled}
          aria-label={t(config.enabled ? "coadaptation.status.turn_off" : "coadaptation.status.turn_on")}
          onclick={toggleEnabled}
          disabled={busy}
        >
          <span class="toggle-knob"></span>
        </button>
      {/if}
      <button class="btn-secondary" onclick={() => coadaptation.load()} disabled={$coadaptation.loading}>
        {t("coadaptation.refresh")}
      </button>
    </div>
  </div>

  <p class="authority-note">{t("coadaptation.authority_note")}</p>

  {#if !$coadaptation.available}
    <div class="unavailable" role="alert">
      <span>{$coadaptation.error || t("coadaptation.unavailable")}</span>
      <button class="btn-secondary" onclick={() => coadaptation.load()}>{t("coadaptation.retry")}</button>
    </div>
  {:else if !status}
    {#if $coadaptation.error}
      <div class="unavailable" role="alert">
        <span>{$coadaptation.error}</span>
        <button class="btn-secondary" onclick={() => coadaptation.load()}>{t("coadaptation.retry")}</button>
      </div>
    {:else}
      <p class="empty">{t("coadaptation.loading")}</p>
    {/if}
  {:else}
    {#if keyringUnavailable}
      <p class="panel-error" role="alert">{t("coadaptation.keyring_unavailable")}</p>
    {/if}
    {#if error}
      <p class="panel-error" role="alert">{error}</p>
    {/if}
    {#if erasureMessage}
      <p class="panel-warning" role="alert">{erasureMessage}</p>
    {/if}
    {#if notice}
      <p class="panel-notice" role="status">{notice}</p>
    {/if}

    <section class="block">
      <div class="block-header">
        <h4>{t("coadaptation.participants.title")}</h4>
        <button class="btn-secondary" onclick={() => openConsent()} disabled={busy || keyringUnavailable}>
          {t("coadaptation.participants.new")}
        </button>
      </div>
      {#if participants.length}
        <ul class="participants">
          {#each participants as entry (entry.participant_key)}
            <li class:selected={selected?.participant_key === entry.participant_key}>
              <button
                class="participant-name"
                onclick={() => (selectedKey = entry.participant_key)}
                title={t("coadaptation.participants.select")}
                disabled={!entry.consent?.active}
              >
                {entry.label || entry.participant_key}
              </button>
              <div class="participant-meta">
                {#if entry.consent?.active}
                  <span>
                    {t("coadaptation.participants.purposes")}:
                    {entry.consent.purposes.map((purpose) => purposeLabel(purpose)).join(", ")}
                  </span>
                  <span>
                    {t("coadaptation.participants.expires", { values: { date: formatDate(entry.consent.expires_at) } })}
                    · {t("coadaptation.participants.protocol", {
                      values: { protocol: entry.consent.protocol_id ?? "–" },
                    })}
                    · {t("coadaptation.participants.sessions", { values: { count: entry.sessions } })}
                  </span>
                {:else}
                  <span>{t("coadaptation.participants.inactive")}</span>
                {/if}
              </div>
              <div class="participant-actions">
                {#if entry.consent?.active}
                  <button class="btn-secondary" onclick={() => openConsent(entry.participant_key)} disabled={busy}>
                    {t("coadaptation.participants.update_consent")}
                  </button>
                {/if}
                <button
                  class="btn-danger"
                  onclick={() => {
                    withdrawing = entry.participant_key;
                    withdrawWord = "";
                  }}
                  disabled={busy}
                >
                  {t("coadaptation.participants.withdraw")}
                </button>
              </div>
            </li>
          {/each}
        </ul>
      {:else}
        <p class="empty inline">{t("coadaptation.participants.none")}</p>
      {/if}
      <p class="small-note">{t("coadaptation.participants.device_note")}</p>
    </section>

    {#if withdrawing}
      <div class="withdraw-dialog" role="alertdialog" aria-labelledby="coadapt-withdraw-text">
        <p id="coadapt-withdraw-text">{t("coadaptation.participants.withdraw_confirm")}</p>
        <label class="field">
          <span>{t("coadaptation.participants.withdraw_type")}</span>
          <input type="text" bind:value={withdrawWord} autocomplete="off" />
        </label>
        <div class="row-actions">
          <button class="btn-secondary" onclick={() => (withdrawing = null)} disabled={busy}>
            {t("coadaptation.participants.cancel")}
          </button>
          <button
            class="btn-danger"
            onclick={confirmWithdraw}
            disabled={busy || withdrawWord.trim().toLowerCase() !== withdrawWordExpected.toLowerCase()}
          >
            {t("coadaptation.participants.withdraw_button")}
          </button>
        </div>
      </div>
    {/if}

    {#if consentSheet}
      <CoAdaptationConsent
        participantKey={consentSheet.participantKey}
        enable={consentSheet.enable}
        onclose={() => (consentSheet = null)}
        ongranted={granted}
      />
    {/if}

    <section class="block">
      <h4>{t("coadaptation.options.title")}</h4>
      {#each OPTIONS as option (option.field)}
        {@const allowed = selectedPurposes.has(option.purpose)}
        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-label">{t(option.key)}</span>
            {#if !allowed}
              <span class="setting-desc">
                {t("coadaptation.options.needs_consent", { values: { purpose: purposeLabel(option.purpose) } })}
              </span>
            {/if}
          </div>
          {#if !allowed && selected}
            <button class="btn-secondary" onclick={() => openConsent(selected?.participant_key)} disabled={busy}>
              {t("coadaptation.options.update_consent")}
            </button>
          {/if}
          <button
            class="toggle toggle-sm"
            class:active={config?.[option.field]}
            role="switch"
            aria-checked={Boolean(config?.[option.field])}
            aria-label={t(option.key)}
            onclick={() => toggleOption(option.field)}
            disabled={busy || !allowed}
          >
            <span class="toggle-knob"></span>
          </button>
        </div>
      {/each}

      <details class="advanced">
        <summary>{t("coadaptation.options.advanced")}</summary>
        {#if advanced}
          <div class="advanced-grid">
            {#each ADVANCED as entry (entry.field)}
              <label class="field">
                <span>{t(`coadaptation.options.${entry.field}`)}</span>
                <input
                  type="number"
                  name={entry.field}
                  min={CONFIG_RANGES[entry.field].min}
                  max={CONFIG_RANGES[entry.field].max}
                  step={entry.step}
                  value={advanced[entry.field]}
                  onchange={(event) => setAdvanced(entry.field, event)}
                />
                <small>
                  {entry.field === "ambiguity_threshold"
                    ? t("coadaptation.options.ambiguity_threshold_hint")
                    : t("coadaptation.options.range", {
                        values: { min: CONFIG_RANGES[entry.field].min, max: CONFIG_RANGES[entry.field].max },
                      })}
                </small>
              </label>
            {/each}
          </div>
          <p class="small-note">{t("coadaptation.options.advanced_note")}</p>
          <div class="row-actions">
            <button class="btn-save" onclick={saveAdvanced} disabled={busy || !advancedDirty}>
              {t("coadaptation.options.save")}
            </button>
          </div>
        {/if}
      </details>
    </section>

    <section class="block">
      <AssistanceRulesEditor participantKey={selected?.participant_key ?? null} />
    </section>

    <section class="block">
      <h4>{t("coadaptation.run.title")}</h4>
      <div class="row-actions start">
        <button
          class="btn-save"
          onclick={() => coadaptation.openStudy()}
          disabled={busy || Boolean(studyBlockedReason)}
        >
          {t("coadaptation.run.button")}
        </button>
        {#if studyBlockedReason}
          <span class="small-note">{studyBlockedReason}</span>
        {/if}
      </div>
    </section>

    {#if selected}
      <section class="block">
        <CoAdaptationResults participantKey={selected.participant_key} />
      </section>
    {/if}

    <section class="block">
      <h4>{t("coadaptation.data.title")}</h4>
      {#if status.store?.exists}
        <p class="small-note">
          {t("coadaptation.data.counts", {
            values: {
              sessions: status.store.sessions,
              trials: status.store.trials,
              burden: status.store.burden_reports,
            },
          })}
        </p>
      {:else}
        <p class="small-note">{t("coadaptation.data.empty")}</p>
      {/if}
      {#if selected}
        <div class="row-actions start">
          {#if selectedPurposes.has("research_export")}
            <button class="btn-secondary" onclick={exportData} disabled={busy}>{t("coadaptation.data.export")}</button>
          {/if}
          <button class="btn-danger" onclick={resetLearning} disabled={busy}>
            {confirmingReset ? t("coadaptation.data.click_again") : t("coadaptation.data.reset_learning")}
          </button>
          <button class="btn-danger" onclick={eraseToday} disabled={busy}>
            {confirmingErase ? t("coadaptation.data.click_again") : t("coadaptation.data.erase_today")}
          </button>
        </div>
        {#if confirmingReset}
          <p class="small-note warn" role="status">{t("coadaptation.data.reset_learning_confirm")}</p>
        {/if}
        {#if confirmingErase}
          <p class="small-note warn" role="status">{t("coadaptation.data.erase_today_confirm")}</p>
        {/if}
      {/if}
    </section>
  {/if}
</div>

<style>
  .coadapt-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .block-header,
  .header-actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .header-actions {
    flex-shrink: 0;
    gap: 8px;
  }

  .panel-header {
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  .subtitle,
  .small-note,
  .setting-desc,
  .status-text,
  .field small {
    font-size: 11px;
    color: var(--text-muted);
  }

  .small-note {
    margin: 0;
  }

  .small-note.warn {
    color: var(--warning);
  }

  .authority-note {
    margin: 10px 14px 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-sm);
  }

  .block {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .participants {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .participants li {
    display: grid;
    grid-template-columns: minmax(90px, auto) minmax(0, 1fr) auto;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .participants li + li {
    margin-top: 6px;
  }

  .participants li.selected {
    border-color: var(--accent);
  }

  .participant-name {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    text-align: left;
    background: none;
    border: none;
    cursor: pointer;
  }

  .participant-meta {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 2px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .participant-actions,
  .row-actions {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: flex-end;
    gap: 6px;
  }

  .row-actions.start {
    justify-content: flex-start;
  }

  .withdraw-dialog {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 8px 14px;
    padding: 10px 12px;
    font-size: 12px;
    border: 1px solid var(--danger);
    border-radius: var(--radius-sm);
  }

  .withdraw-dialog p {
    margin: 0;
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 6px 0;
  }

  .setting-info {
    display: flex;
    min-width: 0;
    flex: 1;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 12px;
    font-weight: 500;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .field input {
    max-width: 220px;
    padding: 4px 8px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .advanced summary {
    font-size: 12px;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .advanced-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin: 8px 0;
  }

  .toggle {
    position: relative;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
    cursor: pointer;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 11px;
    transition: all 0.2s;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
  }

  .toggle:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .toggle-sm {
    transform: scale(0.8);
  }

  .btn-secondary,
  .btn-danger {
    padding: 4px 10px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .btn-danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
  }

  .btn-save {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    white-space: nowrap;
    background: var(--accent);
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .empty,
  .unavailable {
    padding: 16px 14px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .empty.inline {
    margin: 0;
    padding: 4px 0;
  }

  .unavailable {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    color: var(--danger);
  }

  .panel-error,
  .panel-warning,
  .panel-notice {
    margin: 10px 14px 0;
    padding: 8px 10px;
    font-size: 11px;
    border-radius: var(--radius-sm);
  }

  .panel-error {
    color: var(--danger);
    background: var(--danger-bg);
    border: 1px solid color-mix(in srgb, var(--danger) 45%, var(--border));
  }

  .panel-warning {
    color: var(--warning);
    background: var(--bg-tertiary);
    border: 1px solid color-mix(in srgb, var(--warning) 45%, var(--border));
  }

  .panel-notice {
    color: var(--text-secondary);
    background: var(--bg-tertiary);
  }

  @media (max-width: 760px) {
    .participants li {
      grid-template-columns: 1fr;
    }

    .advanced-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
