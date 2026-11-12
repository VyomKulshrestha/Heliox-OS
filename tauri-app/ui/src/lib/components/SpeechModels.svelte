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
  type Progress = { id: string; state: string; received: number; total: number; message: string };

  let models = $state<Model[]>([]);
  let progress = $state<Record<string, Progress>>({});
  let loadError = $state("");

  function megabytes(bytes: number): string {
    return `${Math.max(1, Math.round(bytes / 1_048_576))} MB`;
  }

  async function refresh() {
    try {
      const result = await call<{ status: string; models?: Model[] }>("speech_models_list");
      models = result.models ?? [];
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
    const result = await call<{ status: string; message?: string }>("speech_model_download", { id: model.id });
    if (result.status === "busy" || result.status === "error") {
      progress = {
        ...progress,
        [model.id]: { id: model.id, state: "failed", received: 0, total: 0, message: result.message ?? "" },
      };
    }
    if (result.status === "ready") await refresh();
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
  {#each models as model (model.id)}
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{model.title}</span>
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
