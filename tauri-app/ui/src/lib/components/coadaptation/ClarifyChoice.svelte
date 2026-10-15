<script lang="ts">
  /**
   * ask_clarification (spec 4.3): the top-k candidates by the with-gaze posterior as buttons at twice their size, plus
   * "None of these". Only an explicit choice commits; "None of these" commits nothing and counts as a correction.
   * With candidate centres given, each button is drawn over its candidate (surface coordinates); otherwise the
   * buttons form a list. Where enlarged buttons overlap, the higher-ranked candidate is drawn on top.
   *
   * An explicit choice is a fresh input that starts at least CONFIRM_MIN_MS after the prompt appeared (the confirm
   * step's rule, spec 4.3): a pointer press that begins after that delay, or an Enter/Space keydown that is not a
   * repeat. The prompt opens on the release of an ambiguous activation, so without this the second click of a
   * double-click, a tremor or a held key could choose a candidate the person never looked at.
   */
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { CONFIRM_MIN_MS } from "../../coadaptation/commitRules";
  import type { Translate } from "./AssistChip.svelte";

  interface ClarifyCandidate {
    index: number;
    label: string;
    /** Rendered candidate centre and size in surface CSS px; the button is drawn at twice this size. */
    cx?: number;
    cy?: number;
    w?: number;
    h?: number;
  }

  interface Props {
    candidates: ClarifyCandidate[];
    onchoose: (index: number | null) => void;
  }

  let { candidates, onchoose }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  let buttons = $state<Array<HTMLButtonElement | null>>([]);

  const positioned = $derived(
    candidates.length > 0 && candidates.every((c) => c.cx !== undefined && c.cy !== undefined),
  );
  const noneTop = $derived(positioned ? Math.max(...candidates.map((c) => (c.cy ?? 0) + (c.h ?? 18))) + 12 : 0);
  const noneLeft = $derived(
    positioned ? candidates.reduce((sum, c) => sum + (c.cx ?? 0), 0) / Math.max(1, candidates.length) : 0,
  );

  let shownAt = performance.now();
  /** performance.now() of a press that started late enough to choose, until its click arrives. */
  let pressStartedAt: number | null = null;
  let chosen = false;

  onMount(() => {
    shownAt = performance.now();
    buttons[0]?.focus({ preventScroll: true });
  });

  function settled(): boolean {
    return performance.now() - shownAt >= CONFIRM_MIN_MS;
  }

  function choose(index: number | null) {
    if (chosen) return;
    chosen = true;
    onchoose(index);
  }

  function onPress(event: PointerEvent) {
    event.stopPropagation();
    // Only a press that starts once the prompt has been visible for CONFIRM_MIN_MS can choose.
    pressStartedAt = settled() ? performance.now() : null;
  }

  function onClick(event: MouseEvent, index: number | null) {
    event.stopPropagation();
    if (event.detail === 0) {
      // No pointer: an assistive technology's activation (keys are handled in onKey and never click). It has no press
      // to time, so it is accepted once the prompt has been visible for CONFIRM_MIN_MS.
      if (settled()) choose(index);
      return;
    }
    const startedAt = pressStartedAt;
    pressStartedAt = null;
    if (startedAt === null) return;
    choose(index);
  }

  function onKey(event: KeyboardEvent, index: number | null) {
    if (event.key !== "Enter" && event.key !== " ") return;
    // The keys never become a click: a key choice is judged here, on its keydown, and only once.
    event.preventDefault();
    event.stopPropagation();
    if (event.type !== "keydown" || event.repeat || !settled()) return;
    choose(index);
  }

  function buttonStyle(candidate: ClarifyCandidate, position: number): string {
    const width = Math.max(32, 2 * (candidate.w ?? 32));
    const height = Math.max(32, 2 * (candidate.h ?? 18));
    if (!positioned) return `min-width: ${width}px; min-height: ${height}px;`;
    // Candidates come in posterior order: an earlier (more likely) one is drawn above a later one it overlaps.
    const z = candidates.length - position;
    return `left: ${(candidate.cx ?? 0) - width / 2}px; top: ${(candidate.cy ?? 0) - height / 2}px; width: ${width}px; height: ${height}px; z-index: ${z};`;
  }
</script>

<div class="clarify" class:positioned role="dialog" aria-label={t("study.chip.which_one")}>
  <p class="title">{t("study.chip.which_one")}</p>
  <div class="choices">
    {#each candidates as candidate, position (candidate.index)}
      <button
        bind:this={buttons[position]}
        class="choice"
        style={buttonStyle(candidate, position)}
        aria-label={candidate.label}
        onpointerdown={onPress}
        onclick={(event) => onClick(event, candidate.index)}
        onkeydown={(event) => onKey(event, candidate.index)}
        onkeyup={(event) => onKey(event, candidate.index)}
      >
        {#if !positioned || 2 * (candidate.w ?? 0) >= 56}{candidate.label}{/if}
      </button>
    {/each}
  </div>
  <button
    class="none"
    style={positioned ? `left: ${noneLeft}px; top: ${noneTop}px; transform: translateX(-50%);` : ""}
    onpointerdown={onPress}
    onclick={(event) => onClick(event, null)}
    onkeydown={(event) => onKey(event, null)}
    onkeyup={(event) => onKey(event, null)}
  >
    {t("study.chip.none_of_these")}
  </button>
</div>

<style>
  .clarify {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  }

  .clarify.positioned {
    position: absolute;
    inset: 0;
    padding: 0;
    background: color-mix(in srgb, var(--bg-primary) 55%, transparent);
    border: none;
    border-radius: 0;
    box-shadow: none;
  }

  .title {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  .positioned .title {
    position: absolute;
    top: 8px;
    left: 50%;
    padding: 4px 10px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    transform: translateX(-50%);
  }

  .choices {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
  }

  .positioned .choices {
    display: contents;
  }

  button {
    padding: 4px 12px;
    font-size: 13px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .positioned button {
    position: absolute;
    padding: 0;
  }

  .choice {
    font-weight: 600;
    border: 2px solid var(--accent);
  }

  .positioned .none {
    padding: 4px 12px;
  }

  button:hover,
  button:focus-visible {
    border-color: var(--accent-hover);
    outline: 2px solid var(--accent-hover);
    outline-offset: 1px;
  }

  .none {
    color: var(--text-secondary);
  }
</style>
