<script module lang="ts">
  export type TlxScale = "mental" | "physical" | "temporal" | "performance" | "effort" | "frustration";

  /** The session-scope answers; the study adds scope "session" and block_index null to make a BurdenV1. */
  export interface EndAnswers {
    tlx: Record<TlxScale, number | null> | null;
    predictability: number | null;
    preferred_regime: string | null;
    skipped: string[];
  }
</script>

<script lang="ts">
  /**
   * End-of-session questionnaire (spec 12.2 step 6): raw NASA-TLX (six 0-100 scales, Hart 2006), predictability (1-7)
   * and a preference between the two regimes used most, plus a reminder of the cross-session choice. Every item can
   * be skipped, and an item left unanswered is sent as skipped.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { variantLabel, type Translate } from "./AssistChip.svelte";

  interface Props {
    /** The regimes used most in the session, most used first; the preference item compares the first two. */
    regimesUsed: string[];
    rememberAcrossSessions?: boolean;
    onsubmit: (report: EndAnswers) => void;
  }

  let { regimesUsed, rememberAcrossSessions = false, onsubmit }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const SCALES: TlxScale[] = ["mental", "physical", "temporal", "performance", "effort", "frustration"];
  const PREDICTABILITY = [1, 2, 3, 4, 5, 6, 7];

  let tlx = $state<Record<TlxScale, number | null>>({
    mental: null,
    physical: null,
    temporal: null,
    performance: null,
    effort: null,
    frustration: null,
  });
  let predictability = $state<number | null>(null);
  /** A regime id, "none" for "No preference", or null while unanswered. */
  let preferred = $state<string | null>(null);

  const pair = $derived(regimesUsed.slice(0, 2));
  const askPreference = $derived(pair.length === 2);

  function setScale(scale: TlxScale, event: Event) {
    tlx[scale] = Number((event.currentTarget as HTMLInputElement).value);
  }

  function submit() {
    const skipped: string[] = [];
    for (const scale of SCALES) if (tlx[scale] === null) skipped.push(`tlx.${scale}`);
    if (predictability === null) skipped.push("predictability");
    if (askPreference && preferred === null) skipped.push("preferred_regime");
    const answered = SCALES.some((scale) => tlx[scale] !== null);
    onsubmit({
      tlx: answered ? { ...tlx } : null,
      predictability,
      preferred_regime: askPreference && preferred !== null && preferred !== "none" ? preferred : null,
      skipped,
    });
  }
</script>

<section class="end" aria-labelledby="end-title">
  <h3 id="end-title">{t("study.end.title")}</h3>
  <p class="note">{t("study.end.tlx_intro")}</p>

  {#each SCALES as scale (scale)}
    <div class="item" class:answered={tlx[scale] !== null}>
      <label for={`tlx-${scale}`}>{t(`study.end.tlx.${scale}`)}</label>
      <div class="slider-row">
        <input
          id={`tlx-${scale}`}
          type="range"
          min="0"
          max="100"
          step="5"
          value={tlx[scale] ?? 50}
          aria-valuetext={tlx[scale] === null ? t("study.burden.skipped") : String(tlx[scale])}
          oninput={(event) => setScale(scale, event)}
          onchange={(event) => setScale(scale, event)}
        />
        <output for={`tlx-${scale}`}>{tlx[scale] ?? "–"}</output>
        <button class="skip" onclick={() => (tlx[scale] = null)}>{t("study.end.skip")}</button>
      </div>
    </div>
  {/each}

  <fieldset class="item">
    <legend>{t("study.end.predictability")}</legend>
    <div class="scale" role="radiogroup" aria-label={t("study.end.predictability")}>
      {#each PREDICTABILITY as value (value)}
        <button
          role="radio"
          aria-checked={predictability === value}
          class:selected={predictability === value}
          onclick={() => (predictability = value)}
        >
          {value}
        </button>
      {/each}
    </div>
    <div class="anchors">
      <span>1 · {t("study.end.predictability_low")}</span>
      <span>7 · {t("study.end.predictability_high")}</span>
    </div>
    <button class="skip" onclick={() => (predictability = null)}>{t("study.end.skip")}</button>
  </fieldset>

  {#if askPreference}
    <fieldset class="item">
      <legend>{t("study.end.preference")}</legend>
      <div class="pair" role="radiogroup" aria-label={t("study.end.preference")}>
        {#each pair as regime (regime)}
          <button
            role="radio"
            aria-checked={preferred === regime}
            class:selected={preferred === regime}
            onclick={() => (preferred = regime)}
          >
            {variantLabel(t, regime)}
          </button>
        {/each}
        <button
          role="radio"
          aria-checked={preferred === "none"}
          class:selected={preferred === "none"}
          onclick={() => (preferred = "none")}
        >
          {t("study.end.no_preference")}
        </button>
      </div>
      <button class="skip" onclick={() => (preferred = null)}>{t("study.end.skip")}</button>
    </fieldset>
  {/if}

  <p class="reminder">{t(rememberAcrossSessions ? "study.end.cross_session_on" : "study.end.cross_session_off")}</p>

  <div class="actions">
    <button class="continue" onclick={submit}>{t("study.end.finish")}</button>
  </div>
</section>

<style>
  .end {
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

  .note,
  .reminder {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 0;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  label,
  legend {
    font-size: 13px;
  }

  .slider-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .slider-row input {
    flex: 1;
    accent-color: var(--accent);
  }

  .item:not(.answered) .slider-row input {
    opacity: 0.55;
  }

  output {
    min-width: 32px;
    font-size: 12px;
    color: var(--text-secondary);
    text-align: right;
  }

  .anchors {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
  }

  .scale {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 4px;
  }

  .pair {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .scale button,
  .pair button {
    min-height: 36px;
    padding: 4px 10px;
    font-size: 13px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .scale button.selected,
  .pair button.selected {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  .skip {
    align-self: flex-end;
    padding: 3px 10px;
    font-size: 11px;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
  }

  .continue {
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border: none;
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button:focus-visible,
  input:focus-visible {
    outline: 2px solid var(--accent-hover);
    outline-offset: 2px;
  }
</style>
