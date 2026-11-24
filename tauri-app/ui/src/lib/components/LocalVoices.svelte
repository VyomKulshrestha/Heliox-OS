<script lang="ts">
  // Settings, at the end of Voice & Speech: the local voices' licences and sources (Pocket TTS's CC BY 4.0 asks for
  // its attribution wherever it is used), and the one suggestion for a person whose voice already speaks. A new
  // install starts with the voice the daemon's table picks (pilot/system/voice_default.py); an existing one is never
  // switched. On a computer where the table picks Pocket TTS, someone hearing Kokoro is told once what Pocket would
  // change (speech_models_list's engine_suggestion); nothing is switched or downloaded unless they press the button.
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";
  import { licenceName, localVoices } from "../firstRun";

  type Suggestion = { engine: string; voice: string; download_bytes: number; say: string };

  const voices = localVoices();
  let suggestion = $state<Suggestion | null>(null);
  let busy = $state(false);
  let note = $state("");
  const engine = $derived($settings.voice?.tts_engine ?? "");

  onMount(() => {
    void (async () => {
      try {
        const result = await call<{ engine_suggestion?: Suggestion | null }>("speech_models_list");
        suggestion = result?.engine_suggestion ?? null;
      } catch {
        suggestion = null; // an older daemon, or none: nothing to suggest
      }
    })();
  });

  async function useSuggested() {
    if (!suggestion) return;
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("voice", {
        tts_engine: suggestion.engine,
        tts_voice: suggestion.voice,
      });
      if (result.ok) {
        note = "Heliox now speaks with Pocket TTS.";
        suggestion = null;
      } else {
        note = result.message || "Not changed.";
      }
    } finally {
      busy = false;
    }
  }
</script>

{#if suggestion && engine === "kokoro_tts"}
  <div class="setting-row" data-testid="voice-suggestion">
    <div class="setting-info">
      <span class="setting-label">A quicker voice for this computer</span>
      <span class="setting-desc">{suggestion.say}</span>
    </div>
    <button type="button" class="btn-secondary" disabled={busy} onclick={useSuggested}>Use Pocket TTS</button>
  </div>
{/if}
{#if note}
  <div class="setting-row">
    <div class="setting-info">
      <span class="setting-desc" aria-live="polite">{note}</span>
    </div>
  </div>
{/if}
<div class="setting-row" data-testid="voice-licences">
  <div class="setting-info">
    <span class="setting-label">Voice models</span>
    {#each voices as voice (voice.engine)}
      <span class="setting-desc"
        >{voice.title} ({voice.model}): licence {licenceName(voice.licence)}, from {voice.source}.{voice.attribution
          ? ` ${voice.attribution}`
          : ""} Fetched the first time it speaks, about {Math.round(voice.download_bytes / 1_000_000)} MB.</span
      >
    {/each}
  </div>
</div>
