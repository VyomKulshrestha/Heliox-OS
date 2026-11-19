<script lang="ts">
  // Settings, Voice: the short sound Heliox makes while a spoken answer is on its way (conversation.acknowledge,
  // pilot.conversation.acknowledge). On unless turned off; it plays audio Heliox's own voice made ahead, so nothing is
  // said with the system voice. Turning it on or off asks nothing, since it sends nothing anywhere.
  import { settings } from "../stores/settings";

  let note = $state("");
  let busy = $state(false);
  const on = $derived(($settings.conversation?.acknowledge ?? true) === true);

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
      Heliox says a short "Mm" or "Hmm" while it thinks, so you are not left in silence. Only in Heliox's own voice,
      once it has made the sound; never with the system voice.
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
