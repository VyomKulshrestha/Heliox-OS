<script lang="ts">
  import { onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";

  type Shortcut = { phrase: string; command: string };
  type Result = { status: string; message?: string; shortcuts?: Shortcut[] };

  let shortcuts = $state<Shortcut[]>([]);
  let phrase = $state("");
  let command = $state("");
  let error = $state("");
  let busy = $state(false);

  async function request(method: string, params: Record<string, unknown> = {}): Promise<Result | null> {
    error = "";
    try {
      const result = await call<Result>(method, params);
      if (result.status !== "ok") throw new Error(result.message || "The daemon refused the request.");
      if (result.shortcuts) shortcuts = result.shortcuts;
      return result;
    } catch (err) {
      error = err instanceof Error ? err.message : "The request failed.";
      return null;
    }
  }

  async function add(event: SubmitEvent) {
    event.preventDefault();
    busy = true;
    const result = await request("shortcut_add", { phrase, command });
    busy = false;
    if (result) {
      phrase = "";
      command = "";
    }
  }

  function onChange(method: string, params: unknown) {
    const next = (params as { shortcuts?: Shortcut[] } | undefined)?.shortcuts;
    if (method === "shortcuts_changed" && Array.isArray(next)) shortcuts = next;
  }

  onMount(() => {
    void request("shortcut_list");
    onNotification(onChange);
    return () => offNotification(onChange);
  });
</script>

<section class="shortcuts-card" aria-labelledby="shortcuts-heading" data-testid="voice-shortcuts">
  <div class="section-title">
    <span class="eyebrow">Voice</span>
    <h3 id="shortcuts-heading">Voice shortcuts</h3>
    <p>
      Say a short phrase to run a longer command. You can also say "when I say start work, open VS Code and Slack". The
      command still goes through the normal approvals.
    </p>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}

  <form class="form" onsubmit={add}>
    <input bind:value={phrase} maxlength="60" placeholder="start work" aria-label="Shortcut phrase" required />
    <input
      bind:value={command}
      maxlength="500"
      placeholder="open VS Code and Slack"
      aria-label="Shortcut command"
      required
    />
    <button type="submit" disabled={busy}>Add</button>
  </form>

  {#if shortcuts.length === 0}
    <p class="empty">No shortcuts yet.</p>
  {:else}
    <ul class="rows">
      {#each shortcuts as shortcut (shortcut.phrase)}
        <li>
          <span class="row-text"><strong>"{shortcut.phrase}"</strong> <small>{shortcut.command}</small></span>
          <button
            type="button"
            aria-label={`Remove ${shortcut.phrase}`}
            onclick={() => request("shortcut_remove", { phrase: shortcut.phrase })}>Remove</button
          >
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .shortcuts-card {
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
  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
    font-size: 11px;
  }
  .form {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .form input {
    flex: 1;
    min-width: 120px;
    padding: 6px 9px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 12px;
  }
  .empty {
    padding: 0 20px 14px;
  }
  .rows {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .rows li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 20px;
    border-top: 1px solid var(--border);
  }
  .row-text {
    min-width: 0;
    font-size: 12px;
    overflow-wrap: anywhere;
  }
  button {
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
