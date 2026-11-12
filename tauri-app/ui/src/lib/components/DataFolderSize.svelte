<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  type Store = { name: string; bytes: number; note?: string };
  type Sizes = {
    status: string;
    message?: string;
    total_bytes?: number;
    stores?: Store[];
    warn_gb?: number;
    over?: boolean;
  };
  type Archived = { status: string; message?: string; sizes?: Sizes };

  let sizes = $state<Sizes | null>(null);
  let limit = $state(5);
  let olderThan = $state(daysAgo(90));
  let armed = $state(false);
  let busy = $state(false);
  let notice = $state("");
  let error = $state("");

  function daysAgo(days: number): string {
    const day = new Date(Date.now() - days * 86_400_000);
    return day.toISOString().slice(0, 10);
  }

  function readable(bytes: number): string {
    if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
    if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MB`;
    return `${Math.round(bytes / 1024)} KB`;
  }

  /** One request with its three possible replies: ok, a refusal in words, or no reply at all. */
  async function request<T extends { status: string; message?: string }>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T | null> {
    error = "";
    try {
      const result = await call<T>(method, params);
      if (result?.status !== "ok") {
        error = result?.message || "Heliox did not say why it refused.";
        return null;
      }
      return result;
    } catch (err) {
      error = err instanceof Error && err.message ? err.message : "Heliox did not answer. Try again in a moment.";
      return null;
    }
  }

  function show(found: Sizes) {
    sizes = found;
    if (typeof found.warn_gb === "number") limit = found.warn_gb;
  }

  async function load() {
    const found = await request<Sizes>("storage_sizes");
    if (found) show(found);
  }

  async function saveLimit() {
    const gb = Math.round(Number(limit));
    if (!Number.isFinite(gb) || gb < 1) {
      error = "Choose a whole number of gigabytes, 1 or more.";
      return;
    }
    error = "";
    const result = await settings.updateSectionDetailed("storage", { data_folder_warn_gb: gb });
    if (!result.ok) error = result.message;
    else {
      notice = `Heliox will say when its data folder passes ${gb} GB.`;
      await load();
    }
  }

  /** Archiving changes the audit log, so it takes two clicks; it never deletes anything. */
  async function archive() {
    if (!armed) {
      armed = true;
      setTimeout(() => (armed = false), 5000);
      return;
    }
    armed = false;
    busy = true;
    notice = "";
    const result = await request<Archived>("audit_archive", { older_than: olderThan });
    busy = false;
    if (result) {
      notice = result.message ?? "Archived.";
      if (result.sizes) show(result.sizes);
    }
  }

  onMount(load);
</script>

<div class="data-size" data-testid="data-folder-size">
  <h4>Size on disk</h4>
  <p>
    The audit log, task records, checkpoints and experience ledger are never deleted by Heliox on its own. Heliox tells
    you once when its data folder passes the size below.
  </p>

  {#if error}<div class="message" role="alert">{error}</div>{/if}
  {#if notice}<div class="note" role="status">{notice}</div>{/if}

  {#if sizes}
    <p class="total" class:over={sizes.over}>
      Data folder: <strong>{readable(sizes.total_bytes ?? 0)}</strong>{sizes.over ? `, over ${sizes.warn_gb} GB` : ""}
    </p>
    <ul class="rows">
      {#each sizes.stores ?? [] as store (store.name)}
        <li>
          <span>{store.name}</span><small>{readable(store.bytes)}</small>
          {#if store.note}<small class="store-note">{store.note}</small>{/if}
        </li>
      {/each}
    </ul>
  {/if}

  <div class="line">
    <label
      >Tell me when it passes <input type="number" min="1" step="1" bind:value={limit} aria-label="Size limit in GB" /> GB</label
    >
    <button type="button" onclick={saveLimit}>Save</button>
  </div>
  <div class="line">
    <label
      >Archive audit entries older than <input
        type="date"
        bind:value={olderThan}
        aria-label="Archive entries older than"
      /></label
    >
    <button type="button" onclick={archive} disabled={busy}>
      {armed ? "Click again to archive" : "Archive"}
    </button>
  </div>
  <small>The older entries move to a dated file beside the audit log; nothing is deleted.</small>
</div>

<style>
  .data-size {
    padding: 14px 20px 18px;
    border-top: 1px solid var(--border);
  }
  h4 {
    margin: 0 0 4px;
    font-size: 13px;
  }
  p,
  small {
    margin: 0 0 8px;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .total {
    color: var(--text-primary);
  }
  .total.over {
    color: var(--warning);
  }
  .rows {
    list-style: none;
    margin: 0 0 10px;
    padding: 0;
  }
  .rows li {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 12px;
    font-size: 12px;
    padding: 3px 0;
  }
  .line {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
    font-size: 12px;
  }
  .store-note {
    flex-basis: 100%;
    margin: 2px 0 0;
  }
  input[type="number"] {
    width: 5em;
  }
  .message,
  .note {
    margin: 6px 0;
    font-size: 12px;
  }
  .message {
    color: var(--danger);
  }
</style>
