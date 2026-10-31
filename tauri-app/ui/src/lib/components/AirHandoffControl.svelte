<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { airHandoff } from "../stores/airHandoff";

  let refreshTimer: ReturnType<typeof setInterval> | null = null;
  let targetName = $derived(
    $airHandoff.paired_devices.find((device) => device.device_id === $airHandoff.selectedDeviceId)?.name || "phone",
  );

  onMount(() => {
    void airHandoff.refresh();
    refreshTimer = setInterval(() => void airHandoff.refresh(), 10_000);
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
  });

  async function toggleArm() {
    if ($airHandoff.gestureArmed && $airHandoff.draft) {
      try {
        await airHandoff.cancelDraft();
      } catch {
        // The shared store exposes the daemon error.
      }
      return;
    }
    airHandoff.setGestureArmed(!$airHandoff.gestureArmed);
  }

  async function cancelHandoff() {
    try {
      await airHandoff.cancelDraft();
    } catch {
      // The shared store exposes the daemon error.
    }
  }
</script>

{#if $airHandoff.enabled}
  <div class="handoff-control" class:armed={$airHandoff.gestureArmed} class:holding={Boolean($airHandoff.draft)}>
    <button
      class="handoff-main"
      type="button"
      disabled={!$airHandoff.running || !$airHandoff.selectedDeviceId || $airHandoff.busy}
      title={$airHandoff.selectedDeviceId
        ? `Arm one gesture handoff to ${targetName}`
        : "Pair and select a phone in Settings"}
      onclick={() => void toggleArm()}
    >
      <span class="icon">↗</span>
      <span class="copy">
        <strong>
          {$airHandoff.draft ? "Screen held" : $airHandoff.gestureArmed ? "Handoff armed" : "Air Handoff"}
        </strong>
        <small>
          {$airHandoff.draft
            ? "Push palm to send"
            : $airHandoff.gestureArmed
              ? "Make a fist to grab"
              : $airHandoff.selectedDeviceId
                ? targetName
                : "Pair a phone"}
        </small>
      </span>
    </button>
    {#if $airHandoff.gestureArmed}
      <button class="cancel" type="button" aria-label="Cancel Air Handoff" onclick={() => void cancelHandoff()}
        >×</button
      >
    {/if}
  </div>
{/if}

<style>
  .handoff-control {
    display: flex;
    align-items: stretch;
    min-width: 136px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: var(--bg-secondary);
    overflow: hidden;
  }
  .handoff-control.armed {
    border-color: color-mix(in srgb, var(--accent) 72%, var(--border));
    box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 15%, transparent);
  }
  .handoff-control.holding {
    border-color: var(--success);
  }
  .handoff-main {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    padding: 7px 10px;
    border: 0;
    background: transparent;
    color: var(--text-primary);
    cursor: pointer;
    text-align: left;
  }
  .handoff-main:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .icon {
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--accent) 16%, transparent);
    color: var(--accent);
    font-weight: 900;
  }
  .copy {
    display: grid;
    gap: 1px;
  }
  .copy strong {
    font-size: 10px;
    white-space: nowrap;
  }
  .copy small {
    max-width: 92px;
    overflow: hidden;
    color: var(--text-secondary);
    font: 9px var(--font-mono);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cancel {
    width: 28px;
    border: 0;
    border-left: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 17px;
  }
</style>
