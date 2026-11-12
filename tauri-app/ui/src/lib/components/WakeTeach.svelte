<script lang="ts">
  // Settings, Voice: "Teach Heliox how you say its name". The person says "Hey Heliox" alone a few times; the
  // spellings their recogniser writes at least twice, and that are not ordinary words, become theirs (at most five,
  // removable here). Local only: nothing is recorded to disk or sent anywhere; only the spellings are kept.
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";

  type Variant = { text: string; taught?: boolean };
  type TeachResult = {
    status?: string;
    message?: string;
    heard?: string[];
    added?: string[];
    refused?: Record<string, string>;
    already_heliox?: number;
  };

  let taught = $state<Variant[]>([]);
  let running = $state(false);
  let step = $state("");
  let outcome = $state("");

  async function refresh() {
    try {
      const result = (await call("list_wake_variants")) as { variants?: Variant[] } | undefined;
      taught = (result?.variants ?? []).filter((variant) => variant.taught);
    } catch {
      taught = [];
    }
  }

  function describe(result: TeachResult): string {
    const added = result.added ?? [];
    const refused = Object.values(result.refused ?? {});
    if (added.length) return `Heliox now also answers to: ${added.map((a) => `“${a}”`).join(", ")}.`;
    if ((result.already_heliox ?? 0) > 0 && !refused.length) return "Heliox already hears its name in your voice.";
    if (refused.length) return `Nothing added. ${refused[0]}`;
    return "Nothing was heard. Check the microphone and try again.";
  }

  async function teach() {
    running = true;
    outcome = "";
    step = "Say “Hey Heliox” on its own, then pause. (1 of 4)";
    try {
      const result = (await call("voice_wake_teach", { tries: 4 })) as TeachResult | undefined;
      if (!result || result.status === "error") {
        outcome = result?.message || "Heliox could not listen just now.";
      } else {
        outcome = describe(result);
        await refresh();
      }
    } catch {
      outcome = "Heliox did not answer. Try again in a moment.";
    } finally {
      running = false;
      step = "";
    }
  }

  async function remove(text: string) {
    try {
      const result = (await call("remove_wake_variant", { text })) as { status?: string } | undefined;
      if (result?.status === "ok") {
        taught = taught.filter((variant) => variant.text !== text);
        outcome = `Removed “${text}”.`;
      } else {
        outcome = `“${text}” was not removed.`;
      }
    } catch {
      outcome = `Heliox did not answer, so “${text}” was not removed.`;
    }
  }

  const handler = (method: string, params: unknown) => {
    if (method !== "voice_status") return;
    const data = params as { status?: string; try?: number; of?: number; heard?: string };
    if (data.status !== "wake_teach" || !running) return;
    const next = (data.try ?? 0) + 1;
    const heard = data.heard ? `Heard “${data.heard}”. ` : "Heard nothing. ";
    step = next <= (data.of ?? 4) ? `${heard}Again, please. (${next} of ${data.of ?? 4})` : `${heard}Checking…`;
  };

  onMount(() => {
    onNotification(handler);
    void refresh();
  });
  onDestroy(() => offNotification(handler));
</script>

<div class="setting-row" data-testid="wake-teach">
  <div class="setting-info">
    <span class="setting-label">Teach Heliox how you say its name</span>
    <span class="setting-desc">
      If Heliox often misses “Hey Heliox”, say it four times; the spellings your recogniser writes for it become yours.
      Ordinary words are never added. Only the spellings are kept, on this computer.
    </span>
    {#if step}<span class="setting-desc" aria-live="polite">{step}</span>{/if}
    {#if outcome}<span class="setting-desc" aria-live="polite">{outcome}</span>{/if}
    {#each taught as variant (variant.text)}
      <span class="setting-desc">
        “{variant.text}”
        <button type="button" class="btn-secondary" onclick={() => remove(variant.text)}>Remove</button>
      </span>
    {/each}
  </div>
  <button type="button" class="btn-secondary" onclick={teach} disabled={running}>
    {running ? "Listening…" : "Teach"}
  </button>
</div>
