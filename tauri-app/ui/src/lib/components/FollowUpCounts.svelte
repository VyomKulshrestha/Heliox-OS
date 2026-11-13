<script lang="ts">
  // Settings, Voice: how often a follow-up came after talk Heliox left out (pilot.conversation.follow_up_counts).
  // Four numbers and the day counting began, kept on this computer only; they tell whether the second-voice rule
  // could be turned on without getting in the way of a person alone.
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type Counts = { since?: string; taken?: number; at_least_1?: number; at_least_2?: number; at_least_3?: number };

  let counts = $state<Counts | null>(null);
  let note = $state("");
  let busy = $state(false);

  function day(value: string | undefined): string {
    if (!value) return "";
    const parsed = new Date(`${value}T12:00:00`);
    return Number.isNaN(parsed.getTime())
      ? value
      : parsed.toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  }

  async function load() {
    try {
      const stats = (await call("voice_listener_stats")) as { follow_up_counts?: Counts } | undefined;
      counts = stats?.follow_up_counts ?? null;
      note = counts ? "" : "Heliox did not say how many follow-ups it took.";
    } catch {
      counts = null;
      note = "Heliox did not answer, so the counts could not be read.";
    }
  }

  async function reset() {
    busy = true;
    try {
      const result = (await call("voice_follow_up_counts_reset")) as
        { status?: string; message?: string; follow_up_counts?: Counts } | undefined;
      if (result?.status === "ok") {
        counts = result.follow_up_counts ?? { taken: 0, at_least_1: 0, at_least_2: 0, at_least_3: 0 };
        note = "Started again from today.";
      } else if (result) {
        note = result.message || "The counts were not reset.";
      } else {
        note = "Heliox did not answer, so the counts were not reset.";
      }
    } catch {
      note = "Heliox did not answer, so the counts were not reset.";
    } finally {
      busy = false;
    }
  }

  onMount(() => void load());
</script>

<div class="setting-row" data-testid="follow-up-counts">
  <div class="setting-info">
    <span class="setting-label">Follow-ups and talk Heliox left out</span>
    {#if counts}
      <span class="setting-desc">
        Follow-ups Heliox took: {counts.taken ?? 0}. After something it left out: {counts.at_least_1 ?? 0} (twice or more:
        {counts.at_least_2 ?? 0}, three times or more: {counts.at_least_3 ?? 0}).
      </span>
      {#if counts.since}<span class="setting-desc">Counting since {day(counts.since)}.</span>{/if}
    {/if}
    <span class="setting-desc">
      Only these numbers are kept, on this computer: no words, nothing about who spoke. They show whether asking for
      “Hey Heliox” after other talk in the room would get in your way.
    </span>
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <button type="button" class="btn-secondary" onclick={reset} disabled={busy}>Start again</button>
</div>
