<script lang="ts">
  /** The companion's presence in the header: there when it is watching, and the quickest way to pause it. */
  import { live } from "../stores/live";

  let label = $derived($live.paused ? "Paused" : $live.userState === "speaking" ? "Listening" : "Live");
  let hint = $derived(
    $live.paused
      ? "Live companion paused. Click to resume."
      : $live.blindNote
        ? `${$live.blindNote} Click to pause.`
        : $live.watching
          ? `Live companion is with you in ${$live.watching}. It speaks up at your pauses. Click to pause.`
          : "Live companion is on. Click to pause.",
  );
</script>

{#if $live.enabled}
  <button
    type="button"
    class="live-pill"
    class:paused={$live.paused}
    class:typing={$live.userState === "typing" && !$live.paused}
    aria-pressed={!$live.paused}
    aria-label={hint}
    title={hint}
    onclick={() => live.setPaused(!$live.paused)}
    data-testid="live-presence"
  >
    <span class="dot" aria-hidden="true"></span>
    <span class="label">{label}</span>
  </button>
{/if}

<style>
  .live-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 30px;
    padding: 0 11px 0 9px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--accent);
    background: var(--accent-muted);
    border: 1px solid var(--accent-line);
    border-radius: var(--radius-pill);
    -webkit-app-region: no-drag;
    transition:
      background var(--dur-fast) ease,
      color var(--dur-fast) ease;
  }

  .live-pill:hover {
    background: color-mix(in srgb, var(--accent) 20%, transparent);
  }

  .dot {
    position: relative;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
  }

  /* A slow breath while it watches; quicker while the person types, the moment it is paying most attention. */
  .dot::after {
    content: "";
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    border: 1px solid var(--accent);
    opacity: 0;
    animation: breathe 2.6s var(--ease-out) infinite;
  }

  .typing .dot::after {
    animation-duration: 1.2s;
  }

  .paused {
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-color: var(--border);
  }

  .paused .dot {
    background: var(--text-muted);
  }

  .paused .dot::after {
    animation: none;
  }

  @keyframes breathe {
    0% {
      opacity: 0.7;
      transform: scale(0.6);
    }
    100% {
      opacity: 0;
      transform: scale(1.5);
    }
  }

  @media (max-width: 880px) {
    .label {
      display: none;
    }

    .live-pill {
      width: 30px;
      padding: 0;
      justify-content: center;
    }
  }
</style>
