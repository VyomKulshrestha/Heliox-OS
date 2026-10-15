<script module lang="ts">
  /** A confirm input reported to the reducer as `confirmInput{kind, sinceChipMs}`. */
  export interface ConfirmInput {
    kind: "click" | "key";
    sinceChipMs: number;
  }
</script>

<script lang="ts">
  /**
   * The confirm step of confirm_selection (spec 4.3, 12.5). A confirming click needs a fresh press that starts at
   * least 250 ms after the chip appeared, so a double-click, a key repeat or a tremor cannot confirm. There is no
   * timer: the step stays open until the person confirms or cancels, or the trial ends. In hold regimes the chip only
   * shows the progress of the hold the surface is tracking; releasing early cancels.
   */
  import { onDestroy, onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import type { Translate } from "./AssistChip.svelte";

  interface Props {
    label: string;
    method: "second_click" | "hold";
    holdMs?: number;
    /** performance.now() when the hold started (hold method only). */
    holdStartedAt?: number | null;
    onconfirm: (input: ConfirmInput) => void;
    oncancel: () => void;
  }

  const MIN_CONFIRM_DELAY_MS = 250;
  const RADIUS = 14;
  const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

  let { label, method, holdMs = 0, holdStartedAt = null, onconfirm, oncancel }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  let shownAt = performance.now();
  let pressStartedAt: number | null = null;
  let progress = $state(0);
  let frame = 0;
  let confirmButton = $state<HTMLButtonElement | null>(null);

  onMount(() => {
    shownAt = performance.now();
    if (method === "second_click") confirmButton?.focus({ preventScroll: true });
    if (method === "hold") tick();
  });

  onDestroy(() => {
    if (frame && typeof cancelAnimationFrame === "function") cancelAnimationFrame(frame);
  });

  function tick() {
    progress =
      holdStartedAt !== null && holdMs > 0 ? Math.max(0, Math.min(1, (performance.now() - holdStartedAt) / holdMs)) : 0;
    if (typeof requestAnimationFrame === "function") frame = requestAnimationFrame(tick);
  }

  function onPress(event: PointerEvent) {
    event.stopPropagation();
    const now = performance.now();
    // Only a press that starts after the chip has been visible for 250 ms can confirm.
    pressStartedAt = now - shownAt >= MIN_CONFIRM_DELAY_MS ? now : null;
  }

  function onClick(event: MouseEvent) {
    event.stopPropagation();
    // detail 0: a click synthesised from the keyboard, which onKey handles.
    if (event.detail === 0) return;
    const startedAt = pressStartedAt;
    pressStartedAt = null;
    if (startedAt === null) return;
    onconfirm({ kind: "click", sinceChipMs: Math.round(startedAt - shownAt) });
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      oncancel();
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    if (event.repeat) return;
    const since = performance.now() - shownAt;
    if (since < MIN_CONFIRM_DELAY_MS) return;
    onconfirm({ kind: "key", sinceChipMs: Math.round(since) });
  }

  function cancel(event: MouseEvent) {
    event.stopPropagation();
    oncancel();
  }
</script>

{#if method === "hold"}
  <div class="confirm-chip hold" role="status">
    <svg
      class="ring"
      width="34"
      height="34"
      viewBox="0 0 34 34"
      role="progressbar"
      aria-label={t("study.chip.hold")}
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={Math.round(progress * 100)}
    >
      <circle class="track" cx="17" cy="17" r={RADIUS} />
      <circle
        class="fill"
        cx="17"
        cy="17"
        r={RADIUS}
        stroke-dasharray={CIRCUMFERENCE}
        stroke-dashoffset={CIRCUMFERENCE * (1 - progress)}
      />
    </svg>
    <span>{t("study.chip.hold_progress", { values: { label } })}</span>
  </div>
{:else}
  <div class="confirm-chip" role="group" aria-label={t("study.chip.select_prompt", { values: { label } })}>
    <span class="prompt">{t("study.chip.select_prompt", { values: { label } })}</span>
    <button class="confirm" bind:this={confirmButton} onpointerdown={onPress} onclick={onClick} onkeydown={onKey}>
      {t("study.chip.select")}
    </button>
    <button class="cancel" onclick={cancel} onkeydown={onKey}>{t("study.chip.not_this")}</button>
  </div>
{/if}

<style>
  .confirm-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .prompt {
    font-weight: 600;
  }

  button {
    min-width: 64px;
    min-height: 32px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button.confirm {
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button:focus-visible {
    outline: 2px solid var(--accent-hover);
    outline-offset: 2px;
  }

  .ring {
    flex-shrink: 0;
    transform: rotate(-90deg);
  }

  .track,
  .fill {
    fill: none;
    stroke-width: 4;
  }

  .track {
    stroke: var(--border);
  }

  .fill {
    stroke: var(--accent);
    stroke-linecap: round;
  }
</style>
