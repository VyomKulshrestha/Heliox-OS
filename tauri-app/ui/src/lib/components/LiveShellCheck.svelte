<script lang="ts" module>
  export type ShellState = {
    installed: boolean;
    script: string;
    startup_file: string;
    line: string;
    remove: string;
  };
  export type ShellStatus = {
    status: string;
    supported?: boolean;
    message?: string;
    listening?: boolean;
    problem?: string;
    shells?: Record<string, ShellState>;
    bash?: string;
  };

  /** What pressing Add will do, in one sentence, so nothing is added unseen. */
  export function addSentence(shell: string, state: ShellState): string {
    if (state.startup_file) return `Add writes ${state.script} and adds this one line to ${state.startup_file}:`;
    return `Add writes ${state.script}; ${shell} loads it by itself, and no start-up file is changed.`;
  }
</script>

<script lang="ts">
  /** Live's terminal check for zsh and fish (macOS and Linux): shown, added and removed only on the person's click. */
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  let status = $state<ShellStatus | null>(null);
  let shown = $state<string>("");
  let error = $state("");
  let busy = $state(false);

  async function run(params: Record<string, unknown>) {
    busy = true;
    try {
      const result = await call<ShellStatus>("live_shell", params);
      if (result?.status === "ok") {
        status = result;
        error = "";
      } else {
        error = result?.message || "That did not work.";
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Heliox is not connected.";
    } finally {
      busy = false;
    }
  }

  onMount(() => void run({ action: "status" }));
</script>

{#if status?.supported && status.shells}
  <div class="block" data-testid="live-shell-check">
    <span class="block-label">Terminal check (zsh and fish)</span>
    <p>
      Some terminals (kitty, Alacritty, WezTerm, foot, and on macOS anything but Terminal and iTerm2) show Heliox
      nothing, so a risky command typed there is only noticed after it ran. Adding this to your shell lets Heliox check
      the line before you press Enter, while Live is on. It goes through a file on this computer that only you can open,
      never a line that starts with a space, and never anything after something that looks like a key.
    </p>
    {#each Object.entries(status.shells) as [shell, state] (shell)}
      <div class="shell-row">
        <span class="shell-name">{shell}: {state.installed ? "added" : "not added"}</span>
        {#if state.installed}
          <button type="button" disabled={busy} onclick={() => void run({ action: "uninstall", shell })}>Remove</button>
        {:else if shown === shell}
          <button type="button" class="primary" disabled={busy} onclick={() => void run({ action: "install", shell })}
            >Add</button
          >
        {:else}
          <button type="button" disabled={busy} onclick={() => (shown = shell)}>Show what Add does</button>
        {/if}
      </div>
      {#if shown === shell && !state.installed}
        <div class="preview" data-testid={`live-shell-preview-${shell}`}>
          <p>{addSentence(shell, state)}</p>
          {#if state.line}<code>{state.line}</code>{/if}
          <p>To take it out yourself: {state.remove}</p>
        </div>
      {/if}
    {/each}
    {#if status.bash}<p class="note">bash: {status.bash}</p>{/if}
    {#if status.problem}<p class="error">The check cannot start: {status.problem}.</p>{/if}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  </div>
{/if}

<style>
  .block {
    display: grid;
    gap: 8px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }

  .block-label {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  p {
    max-width: 72ch;
    margin: 0;
    color: var(--text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .shell-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .shell-name {
    font-size: var(--text-sm);
    color: var(--text-primary);
  }

  .preview {
    display: grid;
    gap: 6px;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--bg-primary);
  }

  code {
    overflow-wrap: anywhere;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-primary);
  }

  .note {
    font-size: var(--text-xs);
  }

  .error {
    font-size: var(--text-xs);
    color: var(--danger);
  }

  button {
    flex: none;
    height: 28px;
    padding: 0 10px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  button:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .primary {
    color: var(--on-accent);
    background: var(--accent);
    border-color: transparent;
  }

  @media (max-width: 760px) {
    .block {
      padding-inline: 14px;
    }
  }
</style>
