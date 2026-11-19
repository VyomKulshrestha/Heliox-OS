<script lang="ts">
  // Settings, Voice: whether Heliox says so when it has a reason against a decision (conversation.disagree,
  // pilot.conversation.debate). Changing it asks nothing: it sends nothing anywhere and never lets Heliox refuse,
  // approve or hold up a stop.
  import { settings } from "../stores/settings";

  type Level = "off" | "when_it_matters" | "often";
  const LEVELS: Level[] = ["off", "when_it_matters", "often"];

  let note = $state("");
  let busy = $state(false);
  const level = $derived.by((): Level => {
    const value = $settings.conversation?.disagree;
    return LEVELS.includes(value as Level) ? (value as Level) : "when_it_matters";
  });

  async function choose(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    const value = select.value as Level;
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("conversation", { disagree: value });
      note = result.ok ? "" : result.message || "Not changed.";
      if (!result.ok) select.value = level; // the saved choice, not the one that was refused
    } finally {
      busy = false;
    }
  }
</script>

<div class="setting-row" data-testid="disagree">
  <div class="setting-info">
    <span class="setting-label">Push back on decisions</span>
    <span class="setting-desc">
      Heliox tells you when it sees a reason against what you decided, like a clash in your calendar or a step that
      cannot be undone. It gives way after a couple of replies, or when you say "do it anyway". It never refuses, and it
      never approves anything for you. On a model on this computer, Often works like When it matters.
    </span>
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <select
    class="input-md"
    value={level}
    onchange={choose}
    disabled={busy}
    aria-label="When Heliox pushes back on decisions"
    title="When Heliox pushes back on decisions"
  >
    <option value="off">Never</option>
    <option value="when_it_matters">When it matters</option>
    <option value="often">Often</option>
  </select>
</div>
