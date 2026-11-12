<script lang="ts">
  import { onDestroy, onMount, untrack } from "svelte";
  import { invoke } from "../api/invoke";
  import { isNativeTauriRuntime } from "../hotkey";
  import {
    catalog,
    CORE_DOWNLOAD_MB,
    defaultChoice,
    failedGroups,
    formatMb,
    pythonSentence,
    screenFor,
    stepLabel,
    totalMb,
    type GroupId,
    type SetupState,
  } from "../firstRun";

  interface Props {
    /** For tests and previews: the state to show instead of asking the desktop app. */
    initial?: SetupState | null;
  }

  let { initial = null }: Props = $props();

  const platform = typeof navigator !== "undefined" ? navigator.platform : "";
  const groups = catalog(platform);
  let setupState = $state<SetupState | null>(untrack(() => initial));
  let chosen = $state<GroupId[]>(defaultChoice(groups, false));
  let startError = $state("");
  let dismissed = $state(false);
  let showDetails = $state(false);
  let unlisten: (() => void) | null = null;

  const screen = $derived(dismissed ? "hidden" : screenFor(setupState));

  onMount(async () => {
    if (initial || !isNativeTauriRuntime()) return;
    try {
      const { listen } = await import("@tauri-apps/api/event");
      unlisten = await listen<SetupState>("setup-progress", (event) => {
        setupState = event.payload;
      });
      setupState = await invoke<SetupState>("setup_status");
    } catch {
      setupState = null; // an older desktop build without first-run setup
    }
    try {
      const devices = await navigator.mediaDevices?.enumerateDevices?.();
      if (devices?.some((d) => d.kind === "audioinput")) chosen = defaultChoice(groups, true);
    } catch {
      /* no microphone information: voice stays unselected */
    }
  });

  onDestroy(() => unlisten?.());

  function toggle(id: GroupId, on: boolean) {
    chosen = on ? [...new Set([...chosen, id])] : chosen.filter((g) => g !== id);
  }

  async function start(selection: GroupId[]) {
    startError = "";
    try {
      await invoke("setup_run", { groups: selection });
    } catch (error) {
      startError = error instanceof Error ? error.message : String(error);
    }
  }
</script>

{#if screen !== "hidden"}
  <div class="first-run" role="dialog" aria-modal="true" aria-labelledby="first-run-title">
    <div class="panel">
      {#if screen === "choose"}
        <h2 id="first-run-title">Set up Heliox</h2>
        <p>
          Heliox installs its core first ({formatMb(CORE_DOWNLOAD_MB)} from PyPI). Choose anything else you want now; you
          can add the rest later in Settings.
        </p>
        <p class="python">{pythonSentence(setupState?.python)}</p>
        <ul class="groups">
          {#each groups as group (group.id)}
            <li>
              <label>
                <input
                  type="checkbox"
                  checked={chosen.includes(group.id)}
                  onchange={(e) => toggle(group.id, (e.currentTarget as HTMLInputElement).checked)}
                />
                <span class="group-text">
                  <span class="group-name">{group.label} <span class="size">{formatMb(group.downloadMb)}</span></span>
                  <span class="group-gives">{group.gives}</span>
                  <span class="group-source">From {group.source}.{group.note ? ` ${group.note}` : ""}</span>
                </span>
              </label>
            </li>
          {/each}
        </ul>
        <div class="actions">
          <span class="total">Download: {formatMb(totalMb(groups, chosen, setupState?.python))} in all</span>
          <button
            class="primary"
            onclick={() => start(chosen)}
            disabled={!setupState?.python?.kind || setupState.python.kind === "none"}
          >
            Set up Heliox
          </button>
        </div>
      {:else}
        <h2 id="first-run-title">
          {screen === "installing" ? "Setting up Heliox" : screen === "failed" ? "Setup stopped" : "Heliox is ready"}
        </h2>
        {#if screen === "done"}
          <p>
            Some optional parts could not be installed. Heliox works without them; you can retry now or later in
            Settings.
          </p>
        {/if}
        <ol class="steps">
          {#each setupState?.steps ?? [] as step (step.id)}
            <li class="step {step.status}">
              <span class="step-name">{stepLabel(step.id, groups)}</span>
              <span class="step-status">{step.status === "running" ? "working" : step.status}</span>
              {#if step.message}<span class="step-message">{step.message}</span>{/if}
              {#if step.status === "running" && step.detail}<span class="step-detail">{step.detail}</span>{/if}
              {#if step.hint}<span class="step-hint">{step.hint}</span>{/if}
            </li>
          {/each}
        </ol>
        {#if screen !== "installing"}
          <div class="actions">
            <button class="link" onclick={() => (showDetails = !showDetails)}>
              {showDetails ? "Hide details" : "Show details"}
            </button>
            {#if screen === "done"}
              <button onclick={() => (dismissed = true)}>Continue</button>
            {/if}
            <button
              class="primary"
              onclick={() => start(screen === "done" && setupState ? failedGroups(setupState) : chosen)}
            >
              Retry
            </button>
          </div>
          {#if showDetails}
            <pre class="log">{(setupState?.log_tail ?? []).join("\n")}</pre>
          {/if}
        {/if}
      {/if}
      {#if startError}<p class="error" role="alert">{startError}</p>{/if}
    </div>
  </div>
{/if}

<style>
  .first-run {
    position: fixed;
    inset: 0;
    z-index: 1100;
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
    box-shadow: var(--shadow);
  }

  h2 {
    margin: 0 0 8px;
    font-size: var(--text-lg);
  }

  p {
    margin: 0 0 10px;
    color: var(--text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .python {
    color: var(--text-primary);
  }

  .groups,
  .steps {
    display: grid;
    gap: 8px;
    margin: 12px 0;
    padding: 0;
    list-style: none;
  }

  .groups label {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md, 8px);
    cursor: pointer;
  }

  .groups input {
    margin-top: 3px;
    accent-color: var(--accent);
  }

  .group-text {
    display: grid;
    gap: 2px;
  }

  .group-name {
    color: var(--text-primary);
    font-size: var(--text-sm);
    font-weight: 600;
  }

  .size,
  .group-source,
  .step-detail {
    color: var(--text-muted);
    font-size: var(--text-xs, 12px);
    font-weight: 400;
  }

  .group-gives,
  .step-message {
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .step {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 2px 12px;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md, 8px);
  }

  .step-message,
  .step-detail,
  .step-hint {
    grid-column: 1 / -1;
  }

  .step-status {
    color: var(--text-muted);
    font-size: var(--text-xs, 12px);
  }

  .step.done .step-status {
    color: var(--success);
  }

  .step.failed .step-status,
  .step-hint,
  .error {
    color: var(--danger);
  }

  .step-hint {
    font-size: var(--text-sm);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: flex-end;
  }

  .total {
    margin-right: auto;
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  button {
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md, 8px);
    color: var(--text-primary);
    background: var(--bg-tertiary);
    font: inherit;
    cursor: pointer;
  }

  button.primary {
    border-color: var(--accent);
    color: var(--on-accent);
    background: var(--accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  button.link {
    margin-right: auto;
    padding: 0;
    border: 0;
    color: var(--accent);
    background: none;
    text-decoration: underline;
  }

  .log {
    max-height: 200px;
    overflow: auto;
    margin: 10px 0 0;
    padding: 10px;
    border-radius: var(--radius-md, 8px);
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    font-size: var(--text-xs, 12px);
    white-space: pre-wrap;
  }
</style>
