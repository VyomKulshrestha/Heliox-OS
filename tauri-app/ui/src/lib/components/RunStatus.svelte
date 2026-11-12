<script lang="ts">
  /**
   * What Heliox is doing right now, shown under the conversation while a task runs: the current phase in plain
   * words, where the run is on the understand / plan / act / check track, the step count once a plan exists,
   * the elapsed time after a few seconds, and any streamed answer text. Mounted only while the session is loading.
   */
  import { onDestroy } from "svelte";
  import { _ } from "svelte-i18n";
  import { session } from "../stores/session";
  import { describePhase, formatElapsed, RUN_STAGES } from "../utils/phase";
  import { renderStreamingMarkdown } from "../utils/markdown";
  import mark from "../../assets/heliox-mark.webp";

  const SHOW_ELAPSED_AFTER_S = 3;

  let view = $derived(describePhase($session.phase));
  let steps = $derived($session.liveActions);
  let finished = $derived(steps.filter((step) => step.status === "success" || step.status === "error").length);
  let running = $derived(steps.find((step) => step.status === "running"));
  let stepText = $derived(
    steps.length === 0
      ? ""
      : running
        ? $_("run.step_of", { values: { current: running.index + 1, total: steps.length } })
        : $_("run.steps_done", { values: { done: finished, total: steps.length } }),
  );
  /** How far through the Act stage the run is, once there is a plan to count against. */
  let actProgress = $derived(steps.length ? finished / steps.length : 0);

  const startedAt = performance.now();
  let elapsed = $state(0);
  const timer = setInterval(() => {
    elapsed = (performance.now() - startedAt) / 1000;
  }, 1000);
  onDestroy(() => clearInterval(timer));
</script>

<div class="run-status" class:waiting={view.waitingOnYou} data-testid="run-status" aria-busy="true">
  <span class="avatar" aria-hidden="true">
    <img src={mark} alt="" />
  </span>
  <div class="body">
    <div class="head">
      <p class="label" class:shimmer={!view.waitingOnYou} role="status" aria-live="polite">{view.label}</p>
      {#if stepText}
        <span class="meta">{stepText}</span>
      {/if}
      {#if elapsed >= SHOW_ELAPSED_AFTER_S}
        <span class="meta tabular elapsed" aria-hidden="true">{formatElapsed(elapsed)}</span>
      {/if}
    </div>

    <ol class="track" aria-label={$_("run.progress")}>
      {#each RUN_STAGES as stage, i (stage)}
        <li
          class:done={i < view.stage}
          class:current={i === view.stage}
          aria-current={i === view.stage ? "step" : undefined}
          style:--fill={i === 2 && view.stage === 2 && steps.length ? `${Math.round(actProgress * 100)}%` : undefined}
        >
          <span class="bar"></span>
          <span class="stage-name">{$_(`run.stage_${stage.toLowerCase()}`, { default: stage })}</span>
        </li>
      {/each}
    </ol>

    {#if $session.streamingText}
      <!-- The reply as it arrives, rendered as it will look when complete (open code blocks and emphasis closed). -->
      <div class="stream">
        {@html renderStreamingMarkdown($session.streamingText)}<span class="caret" aria-hidden="true"></span>
      </div>
    {/if}
  </div>
</div>

<style>
  .run-status {
    display: flex;
    gap: 12px;
    padding: 6px 4px 10px;
    /* A brief delay keeps instant local answers from flashing a status they never needed. */
    animation: appear var(--dur-slow) var(--ease-out) 220ms both;
  }

  .avatar {
    position: relative;
    flex: none;
    width: 28px;
    height: 28px;
  }

  .avatar img {
    width: 100%;
    height: 100%;
    border-radius: 50%;
    border: 1px solid var(--accent-line);
    object-fit: cover;
  }

  /* A single arc travelling around the mark: the run is alive. */
  .avatar::after {
    content: "";
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    border: 2px solid transparent;
    border-top-color: var(--accent);
    animation: orbit 1.1s linear infinite;
  }

  .waiting .avatar::after {
    border-top-color: var(--warning);
    border-right-color: var(--warning);
    animation-duration: 2.4s;
  }

  .body {
    display: grid;
    gap: 8px;
    min-width: 0;
    flex: 1;
    padding-top: 3px;
  }

  .head {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
  }

  .label {
    font-family: var(--font-display);
    font-size: var(--text-md);
    font-weight: 500;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .waiting .label {
    color: var(--warning);
  }

  .label.shimmer {
    background: linear-gradient(
      90deg,
      var(--text-secondary) 0%,
      var(--text-secondary) 38%,
      var(--text-primary) 50%,
      var(--text-secondary) 62%,
      var(--text-secondary) 100%
    );
    background-size: 260% 100%;
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    animation: shimmer 2.4s linear infinite;
  }

  .meta {
    flex: none;
    font-size: var(--text-xs);
    color: var(--text-muted);
  }

  .track {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
    max-width: 420px;
    list-style: none;
  }

  .track li {
    display: grid;
    gap: 5px;
    min-width: 0;
  }

  .bar {
    position: relative;
    height: 3px;
    overflow: hidden;
    border-radius: var(--radius-pill);
    background: var(--border);
  }

  .done .bar {
    background: var(--accent);
  }

  .current .bar::before {
    content: "";
    position: absolute;
    inset: 0;
    width: var(--fill, 0%);
    background: var(--accent);
    transition: width var(--dur-slow) var(--ease-out);
  }

  .current .bar::after {
    content: "";
    position: absolute;
    inset: 0;
    width: 40%;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    animation: sweep 1.4s var(--ease-in-out) infinite;
  }

  .waiting .current .bar::after {
    background: linear-gradient(90deg, transparent, var(--warning), transparent);
  }

  .stage-name {
    font-size: var(--text-2xs);
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .current .stage-name {
    color: var(--text-primary);
    font-weight: 600;
  }

  .stream {
    max-width: 68ch;
    color: var(--text-primary);
    font-size: var(--text-base);
    line-height: 1.6;
    word-break: break-word;
  }

  /* Rendered markdown while it streams: blocks sit flush, and the caret follows the last line of text. */
  .stream :global(:is(p, pre, ul, ol, .hlx-code-wrapper)) {
    margin: 0;
  }

  .stream :global(* + :is(p, pre, ul, ol, .hlx-code-wrapper)) {
    margin-top: 0.5em;
  }

  .stream :global(:is(ul, ol)) {
    padding-left: 1.25em;
  }

  .stream :global(p:last-of-type) {
    display: inline;
  }

  .stream :global(code) {
    font-family: var(--font-mono);
  }

  .caret {
    display: inline-block;
    width: 7px;
    height: 1.05em;
    margin-left: 2px;
    vertical-align: text-bottom;
    border-radius: 1px;
    background: var(--accent);
    animation: blink 1s steps(2, start) infinite;
  }

  /* Without motion the sweep cannot mark the current stage, so it gets a steady tint instead. */
  @media (prefers-reduced-motion: reduce) {
    .current .bar {
      background: var(--accent-line);
    }

    .waiting .current .bar {
      background: color-mix(in srgb, var(--warning) 45%, transparent);
    }
  }

  @keyframes appear {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
    to {
      opacity: 1;
      transform: none;
    }
  }

  @keyframes orbit {
    to {
      transform: rotate(360deg);
    }
  }

  @keyframes shimmer {
    from {
      background-position: 100% 0;
    }
    to {
      background-position: -160% 0;
    }
  }

  @keyframes sweep {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(250%);
    }
  }

  @keyframes blink {
    to {
      visibility: hidden;
    }
  }
</style>
