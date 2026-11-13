<script lang="ts">
  // Settings, Voice: the short sound Heliox makes while a spoken answer is on its way (conversation.acknowledge,
  // pilot.conversation.acknowledge). Off by default until the sounds have been heard in the local voice; turning it on
  // or off asks nothing, since it sends nothing anywhere.
  import { settings } from "../stores/settings";

  let note = $state("");
  let busy = $state(false);
  const on = $derived(($settings.conversation?.acknowledge ?? false) === true);

  async function choose(value: boolean) {
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("conversation", { acknowledge: value });
      note = result.ok ? "" : result.message || "Not changed.";
    } finally {
      busy = false;
    }
  }
</script>

<div class="setting-row" data-testid="acknowledge">
  <div class="setting-info">
    <span class="setting-label">Say "Mm" while thinking</span>
    <span class="setting-desc">
      Heliox can say a short "Mm" or "Let me see" while it thinks, so you are not left in silence. Off for now while the
      sounds are being checked.
    </span>
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <button
    type="button"
    class="toggle"
    class:active={on}
    disabled={busy}
    onclick={() => choose(!on)}
    aria-label="Say a short sound while thinking"
    title="Say a short sound while thinking"
  >
    <span class="toggle-knob"></span>
  </button>
</div>
