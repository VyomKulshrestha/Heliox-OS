<script module lang="ts">
  export type BurdenItem = "effort" | "intrusiveness" | "control";

  /** The answers of one prompt; the study adds block_index and scope to make a BurdenV1. */
  export interface BurdenAnswers {
    effort: number | null;
    intrusiveness: number | null;
    control: number | null;
    skipped: BurdenItem[];
  }
</script>

<script lang="ts">
  /**
   * Burden prompt after a block (spec 12.8): three 7-point items about the block just finished, each with Skip. B2
   * (intrusiveness) is hidden after an assistance_off block, where its value is 1 by definition; it is then left null
   * and not counted as skipped. An item left unanswered is sent as skipped, never as a value. A custom single-item
   * scale, not a validated clinical instrument.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import type { Translate } from "./AssistChip.svelte";

  interface Props {
    showIntrusiveness: boolean;
    onsubmit: (report: BurdenAnswers) => void;
  }

  let { showIntrusiveness, onsubmit }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const SCALE = [1, 2, 3, 4, 5, 6, 7];
  const ITEMS: BurdenItem[] = ["effort", "intrusiveness", "control"];

  let answers = $state<Record<BurdenItem, number | null>>({ effort: null, intrusiveness: null, control: null });
  let skipped = $state<Record<BurdenItem, boolean>>({ effort: false, intrusiveness: false, control: false });

  const items = $derived(ITEMS.filter((item) => item !== "intrusiveness" || showIntrusiveness));

  function pick(item: BurdenItem, value: number) {
    answers[item] = value;
    skipped[item] = false;
  }

  function skip(item: BurdenItem) {
    answers[item] = null;
    skipped[item] = true;
  }

  function onScaleKey(event: KeyboardEvent, item: BurdenItem) {
    const current = answers[item];
    if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      event.preventDefault();
      pick(item, current === null ? 1 : Math.min(7, current + 1));
    } else if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      event.preventDefault();
      pick(item, current === null ? 1 : Math.max(1, current - 1));
    }
  }

  function submit() {
    const report: BurdenAnswers = { effort: null, intrusiveness: null, control: null, skipped: [] };
    for (const item of items) {
      if (answers[item] === null) report.skipped.push(item);
      else report[item] = answers[item];
    }
    onsubmit(report);
  }
</script>

<section class="burden" aria-labelledby="burden-title">
  <h3 id="burden-title">{t("study.burden.title")}</h3>
  <p class="scale-note">{t("study.burden.scale_note")}</p>
  {#each items as item (item)}
    <fieldset class="item" class:skipped={skipped[item]}>
      <legend>{t(`study.burden.${item}`)}</legend>
      <div class="scale" role="radiogroup" aria-label={t(`study.burden.${item}`)}>
        {#each SCALE as value (value)}
          <button
            role="radio"
            aria-checked={answers[item] === value}
            class:selected={answers[item] === value}
            tabindex={answers[item] === value || (answers[item] === null && value === 1) ? 0 : -1}
            onclick={() => pick(item, value)}
            onkeydown={(event) => onScaleKey(event, item)}
          >
            {value}
          </button>
        {/each}
      </div>
      <div class="anchors">
        <span>1 · {t(`study.burden.${item}_low`)}</span>
        <span>7 · {t(`study.burden.${item}_high`)}</span>
      </div>
      <div class="item-footer">
        {#if skipped[item]}
          <span class="skipped-label">{t("study.burden.skipped")}</span>
        {/if}
        <button class="skip" aria-pressed={skipped[item]} onclick={() => skip(item)}>{t("study.burden.skip")}</button>
      </div>
    </fieldset>
  {/each}
  <div class="actions">
    <button class="continue" onclick={submit}>{t("study.burden.continue")}</button>
  </div>
</section>

<style>
  .burden {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 560px;
    margin: 0 auto;
    padding: 16px;
    color: var(--text-primary);
  }

  h3 {
    margin: 0;
    font-size: 15px;
  }

  .scale-note {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .item {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 0;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .item.skipped {
    opacity: 0.7;
  }

  legend {
    padding: 0 4px;
    font-size: 13px;
  }

  .scale {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 4px;
  }

  .scale button {
    min-height: 40px;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .scale button.selected {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  .scale button:focus-visible,
  .skip:focus-visible,
  .continue:focus-visible {
    outline: 2px solid var(--accent-hover);
    outline-offset: 2px;
  }

  .anchors {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
  }

  .item-footer {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 8px;
  }

  .skipped-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .skip {
    padding: 3px 10px;
    font-size: 11px;
    color: var(--text-secondary);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .skip[aria-pressed="true"] {
    border-color: var(--accent);
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
</style>
