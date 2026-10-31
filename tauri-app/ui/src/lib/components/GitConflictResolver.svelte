<script lang="ts">
  import { invoke } from "../api/invoke";
  import { call, requireResultStatus, type DaemonStatusResult } from "../api/daemon";
  import type { GitConflictPayload } from "../stores/session";
  import { session } from "../stores/session";
  import { isTauriRuntime } from "../utils/runtime";

  interface Props {
    payload: GitConflictPayload;
  }

  let { payload }: Props = $props();

  // Selected option for each conflict block: 'proposed' | 'original' | 'conflict'
  let selections = $state<string[]>([]);
  $effect(() => {
    selections = payload.conflicts.map(() => "proposed");
  });

  let saveStatus = $state<"idle" | "saving" | "success" | "error">("idle");
  let errorMessage = $state<string>("");

  function getFilename(filepath: string): string {
    return filepath.split(/[/\\]/).pop() || filepath;
  }

  async function handleApply() {
    saveStatus = "saving";
    errorMessage = "";

    try {
      // Check if running inside the native Tauri container or standard web browser
      const isTauri = isTauriRuntime();

      for (let i = 0; i < payload.conflicts.length; i++) {
        const conflict = payload.conflicts[i];
        const selection = selections[i];

        let resolvedCode = "";
        if (selection === "proposed") {
          resolvedCode = conflict.proposed_resolution_code;
        } else if (selection === "original") {
          resolvedCode = conflict.original_hunk;
        } else if (selection === "conflict") {
          resolvedCode = conflict.conflict_hunk;
        }

        let result: DaemonStatusResult;
        if (isTauri) {
          // Apply via Tauri IPC bridge command
          result = await invoke<DaemonStatusResult>("apply_git_conflict_resolution", {
            path: conflict.path,
            fullBlock: conflict.full_block,
            resolvedCode: resolvedCode,
          });
        } else {
          // Fallback: Apply via WebSocket JSON-RPC bridge command when running in standard browser
          result = (await call("apply_git_resolution", {
            path: conflict.path,
            full_block: conflict.full_block,
            resolved_code: resolvedCode,
          })) as DaemonStatusResult;
        }
        requireResultStatus(result, "success", "The daemon did not apply the conflict resolution.");
      }

      saveStatus = "success";
      session.addSystemMessage(
        `Successfully applied conflict resolutions to: ${getFilename(payload.conflicts[0].path)}`,
      );
    } catch (err: any) {
      console.error(err);
      saveStatus = "error";
      errorMessage = err.message || String(err) || "Failed to apply resolutions";
    }
  }

  function handleReject() {
    saveStatus = "idle";
    session.addSystemMessage("Dismissed git conflict resolution UI.");
  }
</script>

<div class="git-resolver-card">
  <div class="resolver-header">
    <div class="header-title">
      <span class="git-icon">&#9473;&#9733;&#9473;</span>
      <span>Git Conflict Resolver</span>
    </div>
    <div class="header-subtitle">
      {getFilename(payload.conflicts[0]?.path || "Unknown File")}
    </div>
  </div>

  <div class="conflicts-container">
    {#each payload.conflicts as conflict, idx}
      <div class="conflict-block">
        <div class="block-header">
          <span class="block-badge">Block {idx + 1} of {payload.conflicts.length}</span>
          <span class="block-path">{getFilename(conflict.path)}</span>
        </div>

        <div class="diff-side-by-side">
          <div class="diff-pane pane-original">
            <div class="pane-header">Original Local (HEAD)</div>
            <pre class="code-container"><code>{conflict.original_hunk || "(empty)"}</code></pre>
          </div>
          <div class="diff-pane pane-conflict">
            <div class="pane-header">Incoming Changes (Other)</div>
            <pre class="code-container"><code>{conflict.conflict_hunk || "(empty)"}</code></pre>
          </div>
        </div>

        <div class="resolution-pane">
          <div class="resolution-header-bar">
            <div class="resolution-title-group">
              <span class="ai-glow-star">&#10024;</span>
              <span class="resolution-title">AI Proposed Resolution</span>
            </div>
            <span class="ai-badge">PROPOSED</span>
          </div>
          <pre class="code-container ai-code"><code>{conflict.proposed_resolution_code}</code></pre>
        </div>

        <div class="selection-bar">
          <span class="selection-label">Select resolution:</span>
          <div class="option-picker">
            <button
              class="opt-btn"
              class:active={selections[idx] === "original"}
              onclick={() => {
                selections[idx] = "original";
              }}
            >
              Keep Local (HEAD)
            </button>
            <button
              class="opt-btn"
              class:active={selections[idx] === "conflict"}
              onclick={() => {
                selections[idx] = "conflict";
              }}
            >
              Keep Incoming
            </button>
            <button
              class="opt-btn opt-ai"
              class:active={selections[idx] === "proposed"}
              onclick={() => {
                selections[idx] = "proposed";
              }}
            >
              Keep AI Proposed
            </button>
          </div>
        </div>
      </div>
    {/each}
  </div>

  {#if saveStatus === "error"}
    <div class="error-banner" role="alert">
      <strong>Error:</strong>
      {errorMessage}
    </div>
  {/if}

  {#if saveStatus === "success"}
    <div class="success-banner">Successfully resolved and saved to disk.</div>
  {/if}

  <div class="action-footer">
    <button class="btn-secondary" onclick={handleReject} disabled={saveStatus === "saving"}> Dismiss </button>
    <button class="btn-primary" onclick={handleApply} disabled={saveStatus === "saving" || saveStatus === "success"}>
      {#if saveStatus === "saving"}
        Applying Resolution...
      {:else}
        Apply Resolution
      {/if}
    </button>
  </div>
</div>

<style>
  .git-resolver-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg, 12px);
    margin: 16px 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    color: var(--text-primary);
    font-family: inherit;
  }

  .resolver-header {
    background: linear-gradient(90deg, var(--bg-hover) 0%, var(--bg-tertiary) 100%);
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 16px;
    color: var(--danger);
  }

  .git-icon {
    font-family: monospace;
    font-weight: bold;
    color: var(--danger);
  }

  .header-subtitle {
    font-size: 13px;
    color: var(--text-secondary);
    font-family: var(--font-mono, monospace);
    background: var(--bg-tertiary);
    padding: 4px 10px;
    border-radius: 6px;
  }

  .conflicts-container {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    max-height: 500px;
    overflow-y: auto;
  }

  .conflict-block {
    border: 1px solid var(--border);
    background: var(--bg-tertiary);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .block-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }

  .block-badge {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
  }

  .block-path {
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-mono, monospace);
  }

  .diff-side-by-side {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  @media (max-width: 768px) {
    .diff-side-by-side {
      grid-template-columns: 1fr;
    }
  }

  .diff-pane {
    background: var(--bg-tertiary);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .pane-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 6px 12px;
    letter-spacing: 0.5px;
  }

  .pane-original {
    border-left: 3px solid var(--danger);
  }
  .pane-original .pane-header {
    background: color-mix(in srgb, var(--danger) 10%, transparent);
    color: var(--danger);
  }
  .pane-original .code-container {
    background: color-mix(in srgb, var(--danger) 2%, transparent);
  }

  .pane-conflict {
    border-left: 3px solid var(--accent);
  }
  .pane-conflict .pane-header {
    background: color-mix(in srgb, var(--accent) 10%, transparent);
    color: var(--accent);
  }
  .pane-conflict .code-container {
    background: color-mix(in srgb, var(--accent) 2%, transparent);
  }

  .code-container {
    margin: 0;
    padding: 12px;
    font-family: var(--font-mono, monospace);
    font-size: 12px;
    line-height: 1.5;
    overflow-x: auto;
    max-height: 180px;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .resolution-pane {
    background: color-mix(in srgb, var(--success) 3%, transparent);
    border: 1px solid color-mix(in srgb, var(--success) 15%, transparent);
    border-radius: 8px;
    overflow: hidden;
  }

  .resolution-header-bar {
    background: color-mix(in srgb, var(--success) 8%, transparent);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 14px;
    border-bottom: 1px solid color-mix(in srgb, var(--success) 12%, transparent);
  }

  .resolution-title-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .ai-glow-star {
    color: var(--success);
    animation: pulse 2s infinite alternate;
  }

  .resolution-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--success);
  }

  .ai-badge {
    font-size: 10px;
    font-weight: 800;
    background: var(--success);
    color: var(--on-accent);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .ai-code {
    background: var(--bg-tertiary) !important;
  }

  .selection-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-tertiary);
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 4px;
    flex-wrap: wrap;
    gap: 8px;
  }

  .selection-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .option-picker {
    display: flex;
    gap: 6px;
  }

  .opt-btn {
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 600;
    border-radius: 4px;
    background: var(--bg-hover);
    border: 1px solid transparent;
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .opt-btn:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .opt-btn.active {
    background: var(--accent);
    color: var(--on-accent);
  }

  .opt-btn.active.opt-ai {
    background: var(--success);
    color: var(--on-accent);
  }

  .error-banner {
    background: color-mix(in srgb, var(--danger) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--danger) 20%, transparent);
    border-radius: 8px;
    padding: 12px;
    margin: 0 20px;
    font-size: 12px;
    color: var(--danger);
  }

  .success-banner {
    background: color-mix(in srgb, var(--success) 10%, transparent);
    border: 1px solid color-mix(in srgb, var(--success) 20%, transparent);
    border-radius: 8px;
    padding: 12px;
    margin: 0 20px;
    font-size: 12px;
    color: var(--success);
    text-align: center;
  }

  .action-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 16px 20px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
  }

  .btn-secondary {
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    border-radius: var(--radius-sm, 6px);
    background: var(--bg-hover);
    color: var(--text-secondary);
    transition: all 0.15s ease;
  }

  .btn-secondary:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-primary {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    border-radius: var(--radius-sm, 6px);
    background: var(--success);
    color: var(--on-accent);
    transition: all 0.15s ease;
  }

  .btn-primary:hover:not(:disabled) {
    background: var(--accent);
    transform: translateY(-1px);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @keyframes pulse {
    0% {
      opacity: 0.7;
    }
    100% {
      opacity: 1;
    }
  }
</style>
