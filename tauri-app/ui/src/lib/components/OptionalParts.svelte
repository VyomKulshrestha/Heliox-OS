<script lang="ts">
  import { invoke } from "../api/invoke";
  import { isNativeTauriRuntime } from "../hotkey";
  import { catalog, formatMb, type GroupId } from "../firstRun";

  interface Props {
    /** For tests: render even outside the desktop app. */
    force?: boolean;
  }

  let { force = false }: Props = $props();
  const groups = catalog(typeof navigator !== "undefined" ? navigator.platform : "");
  let message = $state("");

  async function install(id: GroupId) {
    message = "";
    try {
      // Progress and any failure appear on the setup screen, which opens while it runs.
      await invoke("setup_run", { groups: [id] });
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
  }
</script>

{#if force || isNativeTauriRuntime()}
  <section class="settings-group">
    <h3>Optional parts</h3>
    <p class="intro">
      Add what you skipped at setup. Each is downloaded only when you press its button, and is used after Heliox
      restarts.
    </p>
    {#each groups as group (group.id)}
      <div class="setting-row">
        <div class="setting-info">
          <span class="setting-label">{group.label} <span class="size">{formatMb(group.downloadMb)}</span></span>
          <span class="setting-desc">{group.gives} From {group.source}.</span>
        </div>
        <button class="btn-save" onclick={() => install(group.id)}>Install or repair</button>
      </div>
    {/each}
    {#if message}<p class="error" role="alert">{message}</p>{/if}
  </section>
{/if}

<style>
  .settings-group {
    margin-bottom: 20px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
  }

  h3 {
    margin: 0;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    background: var(--bg-tertiary);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
  }

  .setting-row {
    display: flex;
    gap: 12px;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 13px;
    font-weight: 500;
  }

  .setting-desc,
  .size {
    color: var(--text-muted);
    font-size: var(--text-xs);
    font-weight: 400;
  }

  .btn-save {
    padding: 5px 14px;
    border-radius: var(--radius-sm);
    color: var(--on-accent);
    background: var(--accent);
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
  }

  .intro,
  .error {
    margin: 10px 14px;
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .error {
    color: var(--danger);
  }
</style>
