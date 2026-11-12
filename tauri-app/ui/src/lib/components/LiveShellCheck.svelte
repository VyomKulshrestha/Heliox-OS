<script lang="ts" module>
  export type ShellState = {
    installed: boolean;
    script: string;
    startup_file: string;
    line: string;
    remove: string;
  };
  /** The dry run: would the hook reach Heliox from this shell on this computer. */
  export type ShellReady = {
    ok: boolean;
    problem?: string;
    install?: string;
    answered_ms?: number | null;
    silent_ms?: number | null;
    bound_ms?: number;
  };
  export type ShellStatus = {
    status: string;
    supported?: boolean;
    message?: string;
    listening?: boolean;
    problem?: string;
    shells?: Record<string, ShellState>;
    ready?: Record<string, ShellReady>;
    fish_bound_ms?: number;
    bash?: string;
  };

  /** fish's Enter asks Heliox first: how long it can wait, said plainly. */
  export function boundSentence(ready: ShellReady | undefined, boundMs = 1000): string {
    const bound = boundMs >= 1000 ? `${boundMs / 1000} second${boundMs === 1000 ? "" : "s"}` : `${boundMs} ms`;
    let text = `Enter asks Heliox before running each command, and waits at most ${bound}, only while Live is on. If Heliox is not running, Enter does not wait at all.`;
    if (ready?.answered_ms != null)
      text += ` In the test just now Heliox answered in ${ready.answered_ms} ms, counting the time to start fish.`;
    if (ready?.silent_ms != null)
      text += ` Against a Heliox that never answered, Enter gave up after ${ready.silent_ms} ms.`;
    return text;
  }

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
  // The dry run per shell, kept across status refreshes: Add is offered only after it passed.
  let ready = $state<Record<string, ShellReady>>({});

  async function run(params: Record<string, unknown>) {
    busy = true;
    try {
      const result = await call<ShellStatus>("live_shell", params);
      if (result?.ready && typeof result.ready === "object") ready = { ...ready, ...result.ready };
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

  function show(shell: string) {
    shown = shell;
    void run({ action: "test", shell });
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
        {:else if shown === shell && ready[shell]?.ok}
          <button type="button" class="primary" disabled={busy} onclick={() => void run({ action: "install", shell })}
            >Add</button
          >
        {:else if shown !== shell}
          <button type="button" disabled={busy} onclick={() => show(shell)}>Show what Add does</button>
        {/if}
      </div>
      {#if shown === shell && !state.installed}
        <div class="preview" data-testid={`live-shell-preview-${shell}`}>
          {#if !ready[shell]}
            <p>Checking that {shell} on this computer can reach Heliox…</p>
          {:else if !ready[shell].ok}
            <p class="error" data-testid={`live-shell-cannot-${shell}`}>
              Heliox cannot add this: {ready[shell].problem}{ready[shell].install ? ` ${ready[shell].install}` : ""}
              Nothing was changed.
            </p>
            <button type="button" disabled={busy} onclick={() => show(shell)}>Test again</button>
          {:else}
            <p>{addSentence(shell, state)}</p>
            {#if state.line}<code>{state.line}</code>{/if}
            {#if shell === "fish"}<p data-testid="live-shell-bound">
                {boundSentence(ready[shell], ready[shell].bound_ms || status.fish_bound_ms || 1000)}
              </p>{/if}
            <p>To take it out yourself: {state.remove}</p>
          {/if}
        </div>
      {:else if shell === "fish" && state.installed}
        <p class="note">{boundSentence(undefined, status.fish_bound_ms || 1000)}</p>
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

  .preview button {
    justify-self: start;
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
