<script lang="ts" module>
  /** autonomy_review_status's gpu.driver_record (daemon pilot/system/gpu_driver_record.status). */
  export type DriverRecord = {
    errors?: number | null;
    limit?: number;
    window_days?: number;
    speech_on_processor?: boolean;
    speech_despite_driver_errors?: boolean;
  };

  /** Shown when the driver's record moved speech recognition to the processor, or when the switch is on over a record
   * past the limit (so it can be turned off again). Never on a card with a clean record. */
  export function showRow(record: DriverRecord | null | undefined): boolean {
    if (!record || typeof record !== "object") return false;
    if (record.speech_on_processor === true) return true;
    const limit = typeof record.limit === "number" ? record.limit : 3;
    return record.speech_despite_driver_errors === true && typeof record.errors === "number" && record.errors >= limit;
  }

  export function countWords(record: DriverRecord): string {
    const days = record.window_days ?? 7;
    const errors = record.errors;
    if (typeof errors !== "number") return `This graphics card's driver reported errors in the last ${days} days.`;
    return `This graphics card's driver reported ${errors} ${errors === 1 ? "error" : "errors"} in the last ${days} days.`;
  }
</script>

<script lang="ts">
  // Settings, at the end of Voice & Speech: [gpu] speech_despite_driver_errors. Speech recognition runs inside the
  // daemon, so a driver that aborts during it ends Heliox; with 3 or more driver errors in 7 days it runs on the
  // processor instead (pilot.system.voice._driver_record_keeps_whisper_off). This switch uses the card anyway.
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  let record = $state<DriverRecord | null>(null);
  let note = $state("");
  let busy = $state(false);
  const on = $derived($settings.gpu?.speech_despite_driver_errors ?? record?.speech_despite_driver_errors === true);

  onMount(() => {
    void (async () => {
      try {
        const result = (await call("autonomy_review_status")) as { gpu?: { driver_record?: DriverRecord } } | null;
        const found = result?.gpu?.driver_record;
        record = found && typeof found === "object" ? found : null;
      } catch {
        record = null; // an older daemon, or none: nothing to show
      }
    })();
  });

  async function toggle() {
    busy = true;
    note = "";
    try {
      const next = !on;
      const result = await settings.updateSectionDetailed("gpu", { speech_despite_driver_errors: next });
      if (result.ok) {
        note = result.restartRequired ? "Saved. Restart Heliox to apply it." : "Saved.";
      } else if (result.cancelled) {
        note = "Not changed: it was not approved.";
      } else {
        note = result.message || "Not changed.";
      }
    } finally {
      busy = false;
    }
  }
</script>

{#if record && showRow(record)}
  <div class="setting-row" data-testid="speech-driver-errors">
    <div class="setting-info">
      <span class="setting-label">Speech recognition on the graphics card despite driver errors</span>
      <span class="setting-desc"
        >{countWords(record)} Speech recognition runs inside Heliox, so a driver error during it would stop Heliox; it uses
        the processor instead. Turn this on to use the card anyway. Heliox's voice keeps the card either way.</span
      >
      <span class="security-setting-status"
        >{on ? "On: speech recognition uses the graphics card" : "Off: speech recognition uses the processor"}</span
      >
      {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
    </div>
    <button
      type="button"
      class="toggle"
      class:active={on}
      disabled={busy}
      onclick={toggle}
      role="switch"
      aria-checked={on}
      aria-label="Speech recognition on the graphics card despite driver errors"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>
{/if}
