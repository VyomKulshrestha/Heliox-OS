<script lang="ts">
  /**
   * Session summary (spec 12.2 step 7): what happened in this session, what the model now expects for this
   * participant, and which kinds of help are allowed and why. Every number is this person's data on this device.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import type { SessionSummaryV1 } from "../../coadaptation/types";
  import { reasonLabel, variantLabel, type Translate } from "./AssistChip.svelte";

  interface Props {
    summary: SessionSummaryV1 | null;
    onclose: () => void;
    /** "Never" for a kind of help listed as allowed (a study rule for this participant). */
    onnever?: (variant: string) => void | Promise<void>;
    /** "Change my rules": leaves the study for Settings. */
    onrules?: () => void;
    /** Erase today's study data for this participant. */
    onerase?: () => void | Promise<void>;
  }

  let { summary, onclose, onnever, onrules, onerase }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  let confirmingErase = $state(false);
  let busy = $state(false);

  const trials = $derived(summary?.trials ?? 0);
  const completed = $derived(Math.round((summary?.completion ?? 0) * trials));
  const slipsPer100 = $derived(((summary?.ua_rate ?? 0) * 100).toFixed(1));
  const medianMs = $derived(summary?.median_time_ms ?? null);
  const burden = $derived(Object.entries(summary?.burden_mean ?? {}).filter(([, value]) => value !== null));
  const allowed = $derived(summary?.allowed_now ?? []);
  const excluded = $derived(Object.entries(summary?.excluded_now ?? {}));
  const predictions = $derived(Object.entries(summary?.predictions ?? {}));

  function pct(value: number | null | undefined): string {
    return value == null || !Number.isFinite(value) ? "–" : `${Math.round(value * 100)}`;
  }

  function seconds(ms: number | null | undefined): string {
    return ms == null || !Number.isFinite(ms) ? "–" : `${(ms / 1000).toFixed(1)} s`;
  }

  async function never(variant: string) {
    busy = true;
    try {
      await onnever?.(variant);
    } finally {
      busy = false;
    }
  }

  async function erase() {
    if (!confirmingErase) {
      confirmingErase = true;
      return;
    }
    busy = true;
    try {
      await onerase?.();
    } finally {
      busy = false;
      confirmingErase = false;
    }
  }
</script>

<section class="summary" aria-labelledby="summary-title">
  <h3 id="summary-title">{t("study.summary.title")}</h3>
  <p class="claim">{t("study.summary.claim")}</p>

  {#if summary}
    <dl class="figures">
      <div>
        <dt>{t("study.summary.completed")}</dt>
        <dd>{completed} / {trials}</dd>
      </div>
      <div>
        <dt>{t("study.summary.slips")}</dt>
        <dd>{slipsPer100} / 100</dd>
      </div>
      <div>
        <dt>{t("study.summary.median_time")}</dt>
        <dd>{seconds(medianMs)}</dd>
      </div>
    </dl>

    {#if burden.length}
      <h4>{t("study.summary.burden")}</h4>
      <ul>
        {#each burden as [item, value] (item)}
          <li>
            <span>{t(`coadaptation.results.${item}`)}</span>
            <span>{(value ?? 0).toFixed(1)} / 7</span>
          </li>
        {/each}
      </ul>
    {/if}

    {#if predictions.length}
      <h4>{t("study.summary.expects")}</h4>
      <ul>
        {#each predictions as [variant, prediction] (variant)}
          <li>
            <span class="name">{variantLabel(t, variant)}</span>
            <span>
              {t("study.summary.slips")}
              {pct(prediction.p_unintended?.mean)}% ({pct(prediction.p_unintended?.lo90)}–{pct(
                prediction.p_unintended?.hi90,
              )}%) · {seconds(prediction.time_ms?.mean)}
            </span>
          </li>
        {/each}
      </ul>
    {/if}

    <h4>{t("study.summary.allowed")}</h4>
    <ul>
      {#each allowed as variant (variant)}
        <li>
          <span>{variantLabel(t, variant)}</span>
          {#if onnever && variant !== "assistance_off"}
            <button onclick={() => never(variant)} disabled={busy} aria-label={t("study.summary.never")}>
              {t("study.chip.never")}
            </button>
          {/if}
        </li>
      {/each}
    </ul>

    {#if excluded.length}
      <ul class="excluded">
        {#each excluded as [variant, reason] (variant)}
          <li>
            <span>{variantLabel(t, variant)}</span>
            <span class="reason">{t("study.summary.not_used", { values: { reason: reasonLabel(t, reason) } })}</span>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}

  {#if confirmingErase}
    <p class="warn" role="status">{t("coadaptation.data.erase_today_confirm")}</p>
  {/if}

  <div class="actions">
    {#if onerase}
      <button class="danger" onclick={erase} disabled={busy}>
        {confirmingErase ? t("coadaptation.data.click_again") : t("study.summary.erase_today")}
      </button>
    {/if}
    {#if onrules}
      <button onclick={() => onrules?.()} disabled={busy}>{t("study.summary.rules")}</button>
    {/if}
    <button class="primary" onclick={() => onclose()}>{t("study.summary.close")}</button>
  </div>
</section>

<style>
  .summary {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 600px;
    margin: 0 auto;
    padding: 16px;
    color: var(--text-primary);
  }

  h3 {
    margin: 0;
    font-size: 15px;
  }

  h4 {
    margin: 6px 0 0;
    font-size: 13px;
  }

  .claim {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .warn {
    margin: 0;
    font-size: 12px;
    color: var(--warning);
  }

  .figures {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 0;
  }

  .figures div {
    padding: 8px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  dt {
    font-size: 11px;
    color: var(--text-muted);
  }

  dd {
    margin: 2px 0 0;
    font-size: 14px;
    font-weight: 600;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 5px 8px;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
  }

  .name {
    font-weight: 600;
  }

  .reason {
    color: var(--text-muted);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
  }

  button {
    padding: 5px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button.primary {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button.danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  @media (max-width: 640px) {
    .figures {
      grid-template-columns: 1fr;
    }
  }
</style>
