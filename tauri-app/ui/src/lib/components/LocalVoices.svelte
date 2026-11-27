<script lang="ts">
  // Settings, at the end of Voice & Speech: the local voices' licences and sources (Pocket TTS's CC BY 4.0 asks for
  // its attribution wherever it is used), and the one suggestion for a person whose voice already speaks. A new
  // install starts with the voice the daemon's table picks (pilot/system/voice_default.py); an existing one is never
  // switched. On a computer where the table picks Pocket TTS, someone hearing Kokoro is told once what Pocket would
  // change (speech_models_list's engine_suggestion); nothing is switched or downloaded unless they press the button.
  // A chosen voice that is not on the computer, or that made no sound (speech_models_list's voices_aside, and the
  // voice_status notices): the voice that worked keeps speaking, and a missing one is offered here with its download
  // (voice_download). Nothing is fetched until the button is pressed; the voice speaks only after its test sentence.
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";
  import { settings } from "../stores/settings";
  import { licenceName, localVoices, pocketVoiceCredit } from "../firstRun";

  type Suggestion = { engine: string; voice: string; download_bytes: number; say: string };
  type Aside = {
    engine: string;
    voice: string;
    reason: string;
    download_bytes: number;
    download_ok: boolean;
    download_reason: string;
    say: string;
    downloading?: boolean;
  };

  const voices = localVoices();
  let suggestion = $state<Suggestion | null>(null);
  let aside = $state<Aside[]>([]);
  let asideNote = $state("");
  let busy = $state(false);
  let note = $state("");
  const engine = $derived($settings.voice?.tts_engine ?? "");
  // The credit follows the voice speaking: Pocket's model and its selected voice each have their own line.
  const voiceCredit = $derived(engine === "pocket_tts" ? pocketVoiceCredit($settings.voice?.tts_voice ?? "") : "");

  async function refresh() {
    try {
      const result = await call<{ engine_suggestion?: Suggestion | null; voices_aside?: Aside[] }>(
        "speech_models_list",
      );
      suggestion = result?.engine_suggestion ?? null;
      aside = Array.isArray(result?.voices_aside) ? result.voices_aside : [];
    } catch {
      suggestion = null; // an older daemon, or none: nothing to suggest
      aside = [];
    }
  }

  const heard = (method: string, params: unknown) => {
    if (method !== "voice_status") return;
    const status = params as { status?: string; message?: string };
    if (status.status === "voice_put_aside" || status.status === "voice_downloaded") {
      if (status.status === "voice_downloaded") asideNote = status.message ?? "";
      void refresh();
    } else if (status.status === "voice_download_failed") {
      asideNote = status.message ?? "";
      void refresh();
    }
  };

  onMount(() => {
    onNotification(heard);
    void refresh();
  });
  onDestroy(() => offNotification(heard));

  function sizeWords(bytes: number): string {
    return bytes < 1_000_000 ? "about half a megabyte" : `about ${Math.round(bytes / 1_000_000)} MB`;
  }

  async function download(item: Aside) {
    busy = true;
    asideNote = "";
    try {
      const result = await call<{ status?: string; message?: string; downloading?: boolean }>("voice_download", {
        engine: item.engine,
        voice: item.voice,
        apply: "download",
      });
      if (result?.status === "error") asideNote = result.message || "Not downloaded.";
      else asideNote = `Downloading ${item.voice} (${sizeWords(item.download_bytes)}).`;
      await refresh();
    } catch {
      asideNote = "Heliox did not answer, so nothing was downloaded.";
    } finally {
      busy = false;
    }
  }

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

  // "Not now": hidden for good (voice.engine_suggestion_dismissed); the daemon stops suggesting it too.
  async function notNow() {
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("voice", { engine_suggestion_dismissed: true });
      if (result.ok) suggestion = null;
      else note = result.message || "Not changed.";
    } finally {
      busy = false;
    }
  }
</script>

{#each aside as item (`${item.engine}:${item.voice}`)}
  <div class="setting-row" data-testid="voice-aside">
    <div class="setting-info">
      <span class="setting-label">Voice {item.voice}</span>
      <span class="setting-desc">{item.say}</span>
      {#if item.reason === "missing" && !item.download_ok && item.download_reason}
        <span class="setting-desc">Cannot download now: {item.download_reason}</span>
      {/if}
    </div>
    {#if item.reason === "missing"}
      <button
        type="button"
        class="btn-secondary"
        disabled={busy || !item.download_ok || item.downloading}
        onclick={() => download(item)}
        >{item.downloading ? "Downloading…" : `Download ${item.voice} (${sizeWords(item.download_bytes)})`}</button
      >
    {/if}
  </div>
{/each}
{#if asideNote}
  <div class="setting-row">
    <div class="setting-info">
      <span class="setting-desc" aria-live="polite">{asideNote}</span>
    </div>
  </div>
{/if}
{#if suggestion && engine === "kokoro_tts" && !$settings.voice?.engine_suggestion_dismissed}
  <div class="setting-row" data-testid="voice-suggestion">
    <div class="setting-info">
      <span class="setting-label">A quicker voice for this computer</span>
      <span class="setting-desc">{suggestion.say}</span>
    </div>
    <div class="suggestion-actions">
      <button type="button" class="btn-secondary" disabled={busy} onclick={notNow}>Not now</button>
      <button type="button" class="btn-secondary" disabled={busy} onclick={useSuggested}>Use Pocket TTS</button>
    </div>
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
    {#if voiceCredit}<span class="setting-desc" data-testid="voice-credit">{voiceCredit}</span>{/if}
  </div>
</div>

<style>
  .suggestion-actions {
    display: flex;
    gap: 6px;
  }
</style>
