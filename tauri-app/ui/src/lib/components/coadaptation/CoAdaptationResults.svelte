<script lang="ts">
  /**
   * Results view (spec 16.7): one participant's own study data, today or over all retained sessions. Simulated data
   * never appears here, and every view says it is this person's data on this device, not a general result.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { coadaptation } from "../../stores/coadaptation";
  import type { ResultsV1 } from "../../coadaptation/types";
  import { reasonLabel, variantLabel, type Translate } from "./AssistChip.svelte";

  interface Props {
    participantKey: string;
  }

  let { participantKey }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  let scope = $state<"today" | "all">("today");
  let results = $state<ResultsV1 | null>(null);
  let loading = $state(false);
  let error = $state("");
  let requestId = 0;

  $effect(() => {
    void load(participantKey, scope);
  });

  async function load(key: string, which: "today" | "all") {
    const id = ++requestId;
    loading = true;
    error = "";
    try {
      const next = await coadaptation.results(key, which);
      if (id === requestId) results = next;
    } catch (cause) {
      if (id === requestId) {
        results = null;
        error = t("coadaptation.results.failed", {
          values: { message: cause instanceof Error ? cause.message : String(cause) },
        });
      }
    } finally {
      if (id === requestId) loading = false;
    }
  }

  function fixed(value: number | null | undefined, digits = 1): string {
    return value == null || !Number.isFinite(value) ? "–" : value.toFixed(digits);
  }

  function percent(value: number | null | undefined): string {
    return value == null || !Number.isFinite(value) ? "–" : `${Math.round(value * 100)}%`;
  }

  function seconds(ms: number | null | undefined): string {
    return ms == null || !Number.isFinite(ms) ? "–" : `${(ms / 1000).toFixed(2)} s`;
  }

  const ua = $derived(results?.calibration?.ua ?? null);
  const calibrationBins = $derived((ua?.bins ?? []).slice(0, 10));
  const excludedNow = $derived(Object.entries(results?.allowed_now?.excluded ?? {}));
</script>

<div class="results">
  <div class="results-header">
    <h4>{t("coadaptation.results.title")}</h4>
    <div class="scope" role="group" aria-label={t("coadaptation.results.title")}>
      <button class:active={scope === "today"} aria-pressed={scope === "today"} onclick={() => (scope = "today")}>
        {t("coadaptation.results.today")}
      </button>
      <button class:active={scope === "all"} aria-pressed={scope === "all"} onclick={() => (scope = "all")}>
        {t("coadaptation.results.all")}
      </button>
    </div>
  </div>
  <p class="claim">{t("coadaptation.results.claim")}</p>

  {#if loading && !results}
    <p class="empty">{t("coadaptation.results.loading")}</p>
  {:else if error}
    <p class="error" role="alert">{error}</p>
  {:else if results}
    {#if results.per_regime.length}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{t("coadaptation.results.regime")}</th>
              <th>{t("coadaptation.results.trials")}</th>
              <th>{t("coadaptation.results.completion")}</th>
              <th>{t("coadaptation.results.ua_per_100")}</th>
              <th>{t("coadaptation.results.idle_ua_per_min")}</th>
              <th>{t("coadaptation.results.corrections_per_trial")}</th>
              <th>{t("coadaptation.results.median_time")}</th>
              <th>{t("coadaptation.results.p95_time")}</th>
              <th>{t("coadaptation.results.effort")}</th>
              <th>{t("coadaptation.results.intrusiveness")}</th>
              <th>{t("coadaptation.results.control")}</th>
            </tr>
          </thead>
          <tbody>
            {#each results.per_regime as row (row.variant)}
              <tr>
                <td>{variantLabel(t, row.variant)}</td>
                <td>{row.trials}</td>
                <td>{percent(row.completion)}</td>
                <td>{fixed(row.ua_per_100)}</td>
                <td>{fixed(row.idle_ua_per_min, 2)}</td>
                <td>{fixed(row.corrections_per_trial, 2)}</td>
                <td>{seconds(row.median_time_ms)}</td>
                <td>{seconds(row.p95_time_ms)}</td>
                <td>{fixed(row.burden?.effort)}</td>
                <td>{fixed(row.burden?.intrusiveness)}</td>
                <td>{fixed(row.burden?.control)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <p class="empty">{t("coadaptation.results.none")}</p>
    {/if}

    <div class="panels">
      <section>
        <h5>{t("coadaptation.results.calibration_title")}</h5>
        {#if ua}
          <p class="metric">
            {t("coadaptation.results.brier")}
            {fixed(ua.brier, 3)} · {t("coadaptation.results.ece")}
            {fixed(ua.ece, 3)} · n = {ua.n}
          </p>
          {#if calibrationBins.length}
            <ul class="bins">
              {#each calibrationBins as bin, index (index)}
                <li>
                  <span>{t("coadaptation.results.predicted")} {percent(bin.p_mean)}</span>
                  <span>{t("coadaptation.results.observed")} {percent(bin.observed)}</span>
                  <span class="muted">n = {bin.n}</span>
                </li>
              {/each}
            </ul>
          {/if}
        {:else}
          <p class="empty">{t("coadaptation.results.none")}</p>
        {/if}
      </section>

      <section>
        <h5>{t("coadaptation.results.probe_title")}</h5>
        {#if results.probe_trend?.length}
          <ul class="bins">
            {#each results.probe_trend as point (point.session_number)}
              <li>
                <span>{t("coadaptation.results.session", { values: { n: point.session_number } })}</span>
                <span>{t("coadaptation.results.throughput")} {fixed(point.throughput_bits_s, 2)}</span>
                <span class="muted">{t("coadaptation.results.ua_per_100")} {fixed(point.ua_per_100)}</span>
              </li>
            {/each}
          </ul>
        {:else}
          <p class="empty">{t("coadaptation.results.none")}</p>
        {/if}
        <h5>{t("coadaptation.results.drift_title")}</h5>
        {#if results.drift_flags?.length}
          <ul class="bins">
            {#each results.drift_flags as flag (flag)}
              <li>{flag}</li>
            {/each}
          </ul>
        {:else}
          <p class="empty">{t("coadaptation.results.no_drift")}</p>
        {/if}
      </section>
    </div>

    <section class="allowed">
      <h5>{t("coadaptation.results.allowed_title")}</h5>
      <ul class="bins">
        {#each results.allowed_now?.eligible ?? [] as variant (variant)}
          <li>{variantLabel(t, variant)}</li>
        {/each}
        {#each excludedNow as [variant, reason] (variant)}
          <li>
            <span>{variantLabel(t, variant)}</span>
            <span class="muted">
              {t("coadaptation.results.not_used", { values: { reason: reasonLabel(t, reason) } })}
            </span>
          </li>
        {/each}
      </ul>
    </section>
  {/if}
</div>

<style>
  .results {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 14px;
  }

  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  h4,
  h5 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  h5 {
    font-size: 12px;
  }

  .scope {
    display: flex;
    gap: 4px;
  }

  .scope button {
    padding: 3px 10px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .scope button.active {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  .claim,
  .empty,
  .muted {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .error {
    margin: 0;
    font-size: 11px;
    color: var(--danger);
  }

  .table-wrap {
    overflow-x: auto;
  }

  table {
    width: 100%;
    font-size: 11px;
    border-collapse: collapse;
  }

  th,
  td {
    padding: 4px 6px;
    text-align: left;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }

  th {
    font-weight: 600;
    color: var(--text-muted);
  }

  .panels {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric {
    margin: 0;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .bins {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .bins li {
    display: flex;
    justify-content: space-between;
    gap: 6px;
    padding: 3px 6px;
    font-size: 11px;
    border-bottom: 1px solid var(--border);
  }

  @media (max-width: 760px) {
    .panels {
      grid-template-columns: 1fr;
    }
  }
</style>
