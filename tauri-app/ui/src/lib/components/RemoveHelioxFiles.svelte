<script lang="ts">
  import { invoke } from "../api/invoke";
  import { isNativeTauriRuntime } from "../hotkey";
  import { formatMb } from "../firstRun";

  type Location = { path: string; label: string; kind: "app" | "data"; bytes: number };
  type Plan = {
    home_known?: boolean;
    message?: string;
    locations?: Location[];
    manual_steps?: string[];
    credential_services?: string[];
  };
  type Credentials = { outcome?: string; removed?: string[]; remaining?: string[]; how?: string };
  type Report = {
    removed?: string[];
    failed?: string[];
    freed_bytes?: number;
    kept_data?: boolean;
    credentials?: Credentials;
    manual_steps?: string[];
  };

  interface Props {
    /** For tests: render even outside the desktop app. */
    force?: boolean;
    /** How long to wait for the desktop app before saying it did not answer (ms). */
    replyTimeoutMs?: number;
  }

  let { force = false, replyTimeoutMs = 120_000 }: Props = $props();
  let plan = $state<Plan | null>(null);
  let removeData = $state(false);
  let confirmed = $state(false);
  let busy = $state(false);
  let report = $state<Report | null>(null);
  let unknownResult = $state(false);
  let error = $state("");

  const locations = $derived(plan?.locations ?? []);
  const shown = $derived(locations.filter((l) => removeData || l.kind === "app"));
  const total = $derived(shown.reduce((sum, l) => sum + (l.bytes || 0), 0));
  const finished = $derived(report !== null || unknownResult);

  function withTimeout<T>(promise: Promise<T>): Promise<T> {
    return Promise.race([
      promise,
      new Promise<T>((_, reject) => setTimeout(() => reject(new Error("timeout")), Math.max(1, replyTimeoutMs))),
    ]);
  }

  async function look() {
    error = "";
    try {
      const answer = await withTimeout(invoke<Plan>("heliox_files_plan"));
      if (!answer || !Array.isArray(answer.locations)) {
        error = answer?.message || "The desktop app did not say what Heliox stores. Nothing was removed.";
        return;
      }
      if (answer.home_known === false) {
        error =
          answer.message || "Heliox cannot tell which folder is your home folder, so it lists and removes nothing.";
        return;
      }
      plan = answer;
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      error =
        text === "timeout"
          ? "The desktop app did not answer. Nothing was removed."
          : `Heliox could not list its files: ${text}`;
    }
  }

  async function removeFiles() {
    if (!confirmed) return;
    busy = true;
    error = "";
    try {
      const answer = await withTimeout(
        invoke<Report>("heliox_files_remove", { keepData: !removeData, confirm: "remove" }),
      );
      if (answer && Array.isArray(answer.removed)) report = answer;
      else unknownResult = true;
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      if (text === "timeout") unknownResult = true;
      else error = `Nothing was removed: ${text}`;
    } finally {
      busy = false;
      confirmed = false;
    }
  }

  async function quit() {
    try {
      await invoke("heliox_quit");
    } catch {
      /* the window may already be closing */
    }
  }
</script>

{#if force || isNativeTauriRuntime()}
  <section class="settings-group">
    <h3>Remove Heliox's files</h3>
    <p class="intro">
      Everything Heliox put on this computer, listed with its size before anything is deleted. Nothing outside Heliox's
      own folders is touched.
    </p>
    {#if !plan}
      <div class="row"><button class="btn" onclick={look}>Show what Heliox stores</button></div>
    {:else if !finished}
      <fieldset class="choice">
        <label
          ><input type="radio" name="scope" checked={!removeData} onchange={() => (removeData = false)} /> Remove the app's
          files, keep my data</label
        >
        <label
          ><input type="radio" name="scope" checked={removeData} onchange={() => (removeData = true)} /> Remove everything,
          including my settings, memory, timeline, receipts and the keys saved in this computer's credential store</label
        >
      </fieldset>
      <ul class="list" aria-label="What will be removed">
        {#each shown as location (location.path)}
          <li>
            <span>{location.label} <span class="size">{formatMb((location.bytes || 0) / 1_000_000)}</span></span>
            <span class="path">{location.path}</span>
          </li>
        {/each}
      </ul>
      <p class="intro">
        In all: {formatMb(total / 1_000_000)}. This cannot be undone.
        {removeData
          ? "Heliox's saved keys (API keys, SSH keys, paired devices) are removed from the credential store first."
          : "Your saved keys stay in the credential store."}
      </p>
      <label class="confirm"
        ><input type="checkbox" bind:checked={confirmed} /> I understand these files will be deleted</label
      >
      <div class="row">
        <button class="btn danger" onclick={removeFiles} disabled={!confirmed || busy}>
          {busy ? "Removing…" : removeData ? "Remove everything" : "Remove the app's files"}
        </button>
      </div>
      {#if (plan.manual_steps ?? []).length}
        <p class="intro">Heliox leaves these to you:</p>
        <ul class="list">
          {#each plan.manual_steps ?? [] as step (step)}<li>{step}</li>{/each}
        </ul>
      {/if}
    {/if}
    {#if error}<p class="intro error" role="alert">{error}</p>{/if}
  </section>
{/if}

{#if finished}
  <div class="done" role="dialog" aria-modal="true" aria-labelledby="removed-title">
    <div class="panel">
      <h2 id="removed-title">Heliox's files were removed</h2>
      {#if report}
        <p>
          Removed {report.removed?.length ?? 0}
          {(report.removed?.length ?? 0) === 1 ? "folder" : "folders"}, freeing {formatMb(
            (report.freed_bytes ?? 0) / 1_000_000,
          )}. Heliox's background service is stopped and will not start again.
        </p>
        {#if (report.failed ?? []).length}
          <p class="label">Not removed:</p>
          <ul class="list">
            {#each report.failed ?? [] as failure (failure)}<li class="error">{failure}</li>{/each}
          </ul>
        {/if}
        {#if report.credentials?.outcome === "kept"}
          <p>
            Your saved keys stay in this computer's credential store (services: {(
              report.credentials.remaining ?? []
            ).join(", ")}).
          </p>
        {:else if report.credentials}
          {#if (report.credentials.removed ?? []).length}
            <p>
              Removed {report.credentials.removed?.length} saved {report.credentials.removed?.length === 1
                ? "key"
                : "keys"} from the credential store.
            </p>
          {/if}
          {#if (report.credentials.remaining ?? []).length}
            <p class="label">Still in the credential store, to remove by hand:</p>
            <ul class="list">
              {#each report.credentials.remaining ?? [] as entry (entry)}<li class="error">{entry}</li>{/each}
            </ul>
            <p>{report.credentials.how}</p>
          {/if}
        {/if}
        {#if (report.manual_steps ?? []).length}
          <p class="label">Heliox leaves these to you:</p>
          <ul class="list">
            {#each report.manual_steps ?? [] as step (step)}<li>{step}</li>{/each}
          </ul>
        {/if}
      {:else}
        <p class="error">
          The desktop app did not say what it removed. Heliox's background service may be stopped. Quit Heliox and look
          in your home folder for .config/heliox-os, .local/share/heliox-os and .heliox.
        </p>
      {/if}
      <div class="row"><button class="btn danger" onclick={quit}>Quit Heliox</button></div>
    </div>
  </div>
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

  .intro {
    margin: 10px 14px;
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .choice {
    display: grid;
    gap: 6px;
    margin: 0 14px 10px;
    padding: 0;
    border: 0;
    font-size: var(--text-sm);
  }

  .list {
    display: grid;
    gap: 4px;
    margin: 0 14px 10px;
    padding: 0;
    list-style: none;
    font-size: var(--text-sm);
  }

  .list li {
    display: grid;
    gap: 2px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
  }

  .size,
  .path {
    color: var(--text-muted);
    font-size: var(--text-xs);
  }

  .path {
    overflow-wrap: anywhere;
  }

  .confirm {
    display: block;
    margin: 0 14px 10px;
    font-size: var(--text-sm);
  }

  .row {
    margin: 0 14px 12px;
  }

  .btn {
    padding: 5px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    background: var(--bg-tertiary);
    font-size: 12px;
    font-weight: 600;
  }

  .btn.danger {
    border-color: var(--danger);
    color: var(--danger);
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error {
    color: var(--danger);
  }

  .done {
    position: fixed;
    inset: 0;
    z-index: 1200;
    display: grid;
    place-items: center;
    padding: 16px;
    background: var(--bg-primary);
  }

  .panel {
    width: min(640px, 100%);
    max-height: calc(100vh - 32px);
    overflow-y: auto;
    padding: 22px 24px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-secondary);
  }

  .panel h2 {
    margin: 0 0 8px;
    font-size: var(--text-lg);
  }

  .panel p {
    margin: 0 0 10px;
    color: var(--text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .panel .list,
  .panel .row {
    margin-right: 0;
    margin-left: 0;
  }

  .label {
    color: var(--text-primary);
  }
</style>
