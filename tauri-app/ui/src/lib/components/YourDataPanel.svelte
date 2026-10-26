<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  type Store = {
    key: string;
    label: string;
    about: string;
    size: string;
    items: number | null;
    memory_only?: boolean;
  };
  type Backup = { name: string; size: string; at: number };
  type Result = { status: string; message?: string; [key: string]: unknown };

  let stores = $state<Store[]>([]);
  let backups = $state<Backup[]>([]);
  let includeTimeline = $state(false);
  let armed = $state("");
  let notice = $state("");
  let error = $state("");
  let busy = $state(false);

  async function request(method: string, params: Record<string, unknown> = {}): Promise<Result | null> {
    error = "";
    try {
      const result = await call<Result>(method, params);
      if (result.status !== "ok") throw new Error(result.message || "The daemon refused the request.");
      return result;
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
      return null;
    }
  }

  async function load() {
    const inventory = await request("privacy_inventory");
    if (inventory) stores = (inventory.stores as Store[]) ?? [];
    const listed = await request("backup_list");
    if (listed) backups = (listed.backups as Backup[]) ?? [];
  }

  /** Two clicks for anything that deletes or replaces data: the first arms the button. */
  function confirmed(id: string): boolean {
    if (armed === id) {
      armed = "";
      return true;
    }
    armed = id;
    setTimeout(() => {
      if (armed === id) armed = "";
    }, 5000);
    return false;
  }

  async function clear(store: Store) {
    if (!confirmed(`clear:${store.key}`)) return;
    const result = await request("privacy_clear", { key: store.key, confirmed: true });
    if (result) {
      stores = (result.stores as Store[]) ?? stores;
      notice = `${store.label} cleared.`;
    }
  }

  async function setClipboardHistory(on: boolean) {
    const result = await settings.updateSectionDetailed("clipboard", { history: on });
    if (!result.ok) error = result.message;
    else {
      notice = on ? "Clipboard history is on." : "Clipboard history is off, and what you copied was forgotten.";
      await load();
    }
  }

  async function forgetLastHour() {
    const result = await request("timeline_forget", { minutes: 60 });
    if (result) {
      notice = `Removed ${result.removed} timeline entries from the last hour.`;
      await load();
    }
  }

  async function backup() {
    busy = true;
    const result = await request("backup_create", { include_timeline: includeTimeline });
    busy = false;
    if (result) {
      notice = `Backup saved to ${result.path}`;
      await load();
    }
  }

  async function restore(item: Backup) {
    if (!confirmed(`restore:${item.name}`)) return;
    busy = true;
    const result = await request("backup_restore", { name: item.name, confirmed: true });
    busy = false;
    if (result) {
      notice = `Restored: ${(result.restored as string[]).join(", ") || "nothing"}.`;
      await load();
    }
  }

  onMount(load);
</script>

<section class="data-card" aria-labelledby="your-data-heading" data-testid="your-data">
  <div class="section-title">
    <span class="eyebrow">Privacy</span>
    <h3 id="your-data-heading">Your data</h3>
    <p>
      Everything below stays on this computer. Clear any part of it, forget a stretch of your activity, or take it with
      you as a backup. Backups never include settings, tokens or keys.
    </p>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}
  {#if notice}<div class="note" role="status">{notice}</div>{/if}

  <ul class="rows">
    {#each stores as store (store.key)}
      <li>
        <span class="row-text">
          <strong>{store.label}</strong>
          <small
            >{store.about}
            {store.items ?? 0} item{store.items === 1 ? "" : "s"} · {store.memory_only
              ? "in memory only"
              : store.size}</small
          >
        </span>
        {#if store.key === "clipboard"}
          <label class="inline-toggle"
            ><input
              type="checkbox"
              checked={Boolean($settings.clipboard?.history)}
              onchange={(event) => setClipboardHistory((event.currentTarget as HTMLInputElement).checked)}
            /> Keep</label
          >
        {/if}
        {#if store.key === "timeline"}
          <button type="button" onclick={forgetLastHour}>Forget last hour</button>
        {/if}
        <button type="button" aria-label={`Clear ${store.label}`} onclick={() => clear(store)}>
          {armed === `clear:${store.key}` ? "Click again to clear" : "Clear"}
        </button>
      </li>
    {/each}
  </ul>

  <div class="backup">
    <label><input type="checkbox" bind:checked={includeTimeline} /> Include the activity timeline</label>
    <button type="button" onclick={backup} disabled={busy}>Back up to Downloads</button>
  </div>
  {#if backups.length}
    <ul class="rows">
      {#each backups as item (item.name)}
        <li>
          <span class="row-text"
            ><strong>{item.name}</strong> <small>{new Date(item.at * 1000).toLocaleString()} · {item.size}</small></span
          >
          <button type="button" aria-label={`Restore ${item.name}`} disabled={busy} onclick={() => restore(item)}>
            {armed === `restore:${item.name}` ? "Click again to restore" : "Restore"}
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .data-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title {
    padding: 16px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p,
  small {
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .eyebrow {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .message,
  .note {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    font-size: 11px;
  }
  .message {
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
  }
  .note {
    color: var(--text-secondary);
  }
  .rows {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .rows li {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-top: 1px solid var(--border);
  }
  .inline-toggle {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--text-secondary);
    font-size: 11px;
  }
  .row-text {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    font-size: 12px;
    overflow-wrap: anywhere;
  }
  .backup {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
    font-size: 12px;
  }
  .backup label {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  button {
    flex-shrink: 0;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 11px;
    cursor: pointer;
  }
  button:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>
