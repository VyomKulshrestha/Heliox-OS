<script lang="ts">
  /**
   * Gaze check (spec 12.2 step 3): the person looks at five marked regions for 2.5 s each. Accuracy is the share of
   * stabilized gazeRuntime regions that match the marked region over the last 1.5 s of each. It is read locally from
   * the gazeRuntime store; nothing is sent to the daemon here (the study sends no gaze_event). The result stays in
   * memory until session_begin sends it.
   */
  import { onDestroy } from "svelte";
  import { get } from "svelte/store";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { gazeRuntime } from "../../stores/gazeRuntime";
  import type { GazeRegion } from "../../coadaptation/types";
  import type { Translate } from "./AssistChip.svelte";

  interface Props {
    /** Called with the accuracy in [0, 1], or null when the person skips the check. */
    ondone: (accuracy: number | null) => void;
  }

  let { ondone }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const REGIONS: GazeRegion[] = ["center", "left", "right", "up", "down"];
  const REGION_MS = 2500;
  const SAMPLE_WINDOW_MS = 1500;
  const SAMPLE_EVERY_MS = 100;

  let running = $state(false);
  let regionIndex = $state(0);
  let timers: Array<ReturnType<typeof setTimeout>> = [];
  let sampler: ReturnType<typeof setInterval> | null = null;
  let matches = 0;
  let samples = 0;

  const target = $derived(REGIONS[regionIndex] ?? "center");

  onDestroy(stop);

  function stop() {
    for (const timer of timers) clearTimeout(timer);
    timers = [];
    if (sampler !== null) clearInterval(sampler);
    sampler = null;
  }

  function start() {
    stop();
    running = true;
    matches = 0;
    samples = 0;
    showRegion(0);
  }

  function showRegion(index: number) {
    regionIndex = index;
    const region = REGIONS[index];
    timers.push(
      setTimeout(() => {
        sampler = setInterval(() => sample(region), SAMPLE_EVERY_MS);
      }, REGION_MS - SAMPLE_WINDOW_MS),
    );
    timers.push(
      setTimeout(() => {
        if (sampler !== null) clearInterval(sampler);
        sampler = null;
        if (index + 1 < REGIONS.length) showRegion(index + 1);
        else finish();
      }, REGION_MS),
    );
  }

  function sample(region: GazeRegion) {
    const gaze = get(gazeRuntime);
    // Only stabilized readings count; a missing reading is not evidence either way.
    if (gaze.phase !== "active" || gaze.region === null) return;
    samples += 1;
    if (gaze.region === region) matches += 1;
  }

  function finish() {
    stop();
    running = false;
    ondone(samples > 0 ? matches / samples : 0);
  }

  function skip() {
    stop();
    running = false;
    ondone(null);
  }
</script>

<div class="gaze-check">
  {#if running}
    <div class="field" aria-live="polite">
      <span class={`marker ${target}`} aria-hidden="true"></span>
      <p class="instruction">{t("study.gaze_check.instruction")}</p>
    </div>
    <p class="progress">{t(`study.gaze_check.region.${target}`)} · {regionIndex + 1} / {REGIONS.length}</p>
  {:else}
    <h3>{t("study.gaze_check.title")}</h3>
    <p>{t("study.gaze_check.instruction")}</p>
    <div class="actions">
      <button class="secondary" onclick={skip}>{t("study.gaze_check.skip")}</button>
      <button class="primary" onclick={start}>{t("study.setup.start")}</button>
    </div>
  {/if}
</div>

<style>
  .gaze-check {
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 100%;
    min-height: 320px;
    padding: 16px;
    color: var(--text-primary);
  }

  h3 {
    margin: 0;
    font-size: 15px;
  }

  p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
  }

  .field {
    position: relative;
    flex: 1;
    min-height: 280px;
    border: 1px dashed var(--border);
    border-radius: var(--radius-md);
  }

  .marker {
    position: absolute;
    width: 22px;
    height: 22px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 0 6px var(--accent-muted);
    transform: translate(-50%, -50%);
  }

  .marker.center {
    top: 50%;
    left: 50%;
  }

  .marker.left {
    top: 50%;
    left: 12%;
  }

  .marker.right {
    top: 50%;
    left: 88%;
  }

  .marker.up {
    top: 12%;
    left: 50%;
  }

  .marker.down {
    top: 88%;
    left: 50%;
  }

  .instruction {
    position: absolute;
    bottom: 8px;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 11px;
  }

  .progress {
    font-size: 11px;
    color: var(--text-muted);
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
</style>
