<script lang="ts">
  // Settings, Voice: the speech models Heliox can download. Each says what it is for, its size, licence and source;
  // a download starts only when the person presses its button, resumes after an interruption, and is checked
  // before it is used. The daemon does the downloading (speech_model_download) and reports speech_model_progress.
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";

  type Model = {
    id: string;
    title: string;
    purpose: string;
    languages: string[];
    licence: string;
    source: string;
    download_bytes: number;
    installed: boolean;
    partial_bytes: number;
  };
  // The voice for the person's voice language (pilot/system/voice_by_language.py): what they will hear, and
  // whether accepting changes anything.
  type VoicePlan = { language: string; engine: string; voice: string; change: boolean; say: string };
  type Progress = { id: string; state: string; received: number; total: number; message: string };
  type Recommended = {
    title: string;
    why: string;
    ids: string[];
    missing: string[];
    missing_bytes: number;
    partial_bytes?: number;
    where?: string;
  };

  let models = $state<Model[]>([]);
  let recommended = $state<Recommended | null>(null);
  let voicePlan = $state<VoicePlan | null>(null);
  let progress = $state<Record<string, Progress>>({});
  let loadError = $state("");

  function megabytes(bytes: number): string {
    return `${Math.max(1, Math.round(bytes / 1_048_576))} MB`;
  }

  async function refresh() {
    try {
      const result = await call<{ status: string; models?: Model[]; recommended?: Recommended; voice?: VoicePlan }>(
        "speech_models_list",
      );
      models = result.models ?? [];
      recommended = result.recommended ?? null;
      voicePlan = result.voice ?? null;
      loadError = "";
    } catch (error) {
      loadError = error instanceof Error ? error.message : "Heliox is not running.";
    }
  }

  async function download(model: Model) {
    progress = {
      ...progress,
      [model.id]: { id: model.id, state: "queued", received: 0, total: 0, message: "Starting…" },
    };
    let result: { status?: string; message?: string };
    try {
      result = (await call<{ status: string; message?: string }>("speech_model_download", { id: model.id })) ?? {};
    } catch {
      result = { status: "error", message: "Heliox did not answer, so the download did not start. Try again." };
    }
    if (result.status === "busy" || result.status === "error") {
      progress = {
        ...progress,
        [model.id]: { id: model.id, state: "failed", received: 0, total: 0, message: result.message ?? "" },
      };
    }
    if (result.status === "ready") await refresh();
  }

  // The recommended set for the voice language: one press asks for each missing model; the daemon fetches them
  // one after another.
  async function downloadRecommended() {
    for (const id of recommended?.missing ?? []) {
      const model = models.find((m) => m.id === id);
      if (model && !busy(id)) await download(model);
      if (progress[id]?.state === "failed") return; // said under that model; the next one is not asked for
    }
  }

  // Nothing is fetched here: a Kokoro voice is fetched by Kokoro the first time it speaks.
  async function useVoiceForLanguage() {
    const result = await call<VoicePlan & { status: string }>("voice_for_language_apply");
    if (result) voicePlan = result;
  }

  async function cancel(model: Model) {
    await call("speech_model_cancel", { id: model.id });
  }

  async function remove(model: Model) {
    await call("speech_model_remove", { id: model.id });
    await refresh();
  }

  const handler = (method: string, params: unknown) => {
    if (method !== "speech_model_progress") return;
    const update = params as Progress;
    progress = { ...progress, [update.id]: update };
    if (update.state === "ready" || update.state === "cancelled" || update.state === "failed") void refresh();
  };

  onMount(() => {
    onNotification(handler);
    void refresh();
  });
  onDestroy(() => offNotification(handler));

  function busy(id: string): boolean {
    const state = progress[id]?.state;
    return state === "queued" || state === "downloading" || state === "verifying";
  }
</script>

<div class="speech-models" id="voice-models">
  {#if loadError}
    <p class="setting-desc">{loadError}</p>
  {/if}
  {#if recommended && recommended.ids.length}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{recommended.title}</span>
        <span class="setting-desc">{recommended.why}</span>
        {#if recommended.where}<span class="setting-desc">{recommended.where}</span>{/if}
      </div>
      {#if recommended.missing.length}
        <button type="button" class="btn-secondary" onclick={downloadRecommended}>
          {#if recommended.partial_bytes}
            Continue · {megabytes(recommended.missing_bytes - recommended.partial_bytes)} left
          {:else}
            {recommended.missing.length > 1 ? "Download both" : "Download"} · {megabytes(recommended.missing_bytes)}
          {/if}
        </button>
      {:else}
        <span class="setting-desc">Downloaded</span>
      {/if}
    </div>
  {/if}
  {#if voicePlan && voicePlan.say}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Heliox's voice</span>
        <span class="setting-desc">{voicePlan.say}</span>
      </div>
      {#if voicePlan.change}
        <button type="button" class="btn-secondary" onclick={useVoiceForLanguage}>Use this voice</button>
      {/if}
    </div>
  {/if}
  {#each models as model (model.id)}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">
          {model.title}{#if recommended?.ids.includes(model.id)}<span class="recommended-tag">Recommended</span>{/if}
        </span>
        <span class="setting-desc">
          {model.purpose}
          {#if model.languages.includes("en") && !model.languages.includes("multilingual")}English only.{/if}
        </span>
        <span class="setting-desc">{megabytes(model.download_bytes)} · {model.licence} · {model.source}</span>
        {#if progress[model.id] && !(model.installed && progress[model.id].state === "ready")}
          <span class="security-setting-status" aria-live="polite">{progress[model.id].message}</span>
        {/if}
      </div>
      {#if model.installed}
        <button type="button" class="btn-secondary" onclick={() => remove(model)}>Remove</button>
      {:else if busy(model.id)}
        <button type="button" class="btn-secondary" onclick={() => cancel(model)}>Stop</button>
      {:else}
        <button type="button" class="btn-secondary" onclick={() => download(model)}>
          {model.partial_bytes > 0 ? "Continue download" : `Download ${megabytes(model.download_bytes)}`}
        </button>
      {/if}
    </div>
  {/each}
</div>

<style>
  .recommended-tag {
    margin-left: 8px;
    font-size: var(--text-xs);
    color: var(--accent);
  }
</style>
