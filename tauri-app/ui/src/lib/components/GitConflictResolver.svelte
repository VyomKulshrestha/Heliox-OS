<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import type { GitConflictPayload } from "../stores/session";
  import { session } from "../stores/session";

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
      let isTauri = false;
      try {
        isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
      } catch (e) {}

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

        if (isTauri) {
          // Apply via Tauri IPC bridge command
          await invoke("apply_git_conflict_resolution", {
            path: conflict.path,
            fullBlock: conflict.full_block,
            resolvedCode: resolvedCode
          });
        } else {
          // Fallback: Apply via WebSocket JSON-RPC bridge command when running in standard browser
          const { call } = await import("../api/daemon");
          await call("apply_git_resolution", {
            path: conflict.path,
            full_block: conflict.full_block,
            resolved_code: resolvedCode
          });
        }
      }

      saveStatus = "success";
      session.addSystemMessage(`Successfully applied conflict resolutions to: ${getFilename(payload.conflicts[0].path)}`);
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
              onclick={() => { selections[idx] = "original"; }}
            >
              Keep Local (HEAD)
            </button>
            <button
              class="opt-btn"
              class:active={selections[idx] === "conflict"}
              onclick={() => { selections[idx] = "conflict"; }}
            >
              Keep Incoming
            </button>
            <button
              class="opt-btn opt-ai"
              class:active={selections[idx] === "proposed"}
              onclick={() => { selections[idx] = "proposed"; }}
            >
              Keep AI Proposed
            </button>
          </div>
        </div>
      </div>
    {/each}
  </div>

  {#if saveStatus === "error"}
    <div class="error-banner">
      <strong>Error:</strong> {errorMessage}
    </div>
  {/if}

  {#if saveStatus === "success"}
    <div class="success-banner">
      Successfully resolved and saved to disk.
    </div>
  {/if}

  <div class="action-footer">
    <button class="btn-secondary" onclick={handleReject} disabled={saveStatus === "saving"}>
      Dismiss
    </button>
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
    background: var(--bg-secondary, #1e1e2e);
    border: 1px solid var(--border, #313244);
    border-radius: var(--radius-lg, 12px);
    margin: 16px 0;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    color: var(--text-primary, #cdd6f4);
    font-family: inherit;
  }

  .resolver-header {
    background: linear-gradient(90deg, #313244 0%, #1e1e2e 100%);
    padding: 16px 20px;
    border-bottom: 1px solid var(--border, #313244);
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
    color: #f38ba8;
  }

  .git-icon {
    font-family: monospace;
    font-weight: bold;
    color: #f38ba8;
  }

  .header-subtitle {
    font-size: 13px;
    color: var(--text-secondary, #a6adc8);
    font-family: var(--font-mono, monospace);
    background: rgba(0, 0, 0, 0.2);
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
    border: 1px solid rgba(255, 255, 255, 0.05);
    background: rgba(0, 0, 0, 0.15);
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
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding-bottom: 8px;
  }

  .block-badge {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent, #cba6f7);
    text-transform: uppercase;
  }

  .block-path {
    font-size: 12px;
    color: var(--text-secondary, #a6adc8);
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
    background: rgba(0, 0, 0, 0.25);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }

  .pane-header {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 6px 12px;
    letter-spacing: 0.5px;
  }

  .pane-original {
    border-left: 3px solid #f38ba8;
  }
  .pane-original .pane-header {
    background: rgba(243, 139, 168, 0.1);
    color: #f38ba8;
  }
  .pane-original .code-container {
    background: rgba(243, 139, 168, 0.02);
  }

  .pane-conflict {
    border-left: 3px solid #89b4fa;
  }
  .pane-conflict .pane-header {
    background: rgba(137, 180, 250, 0.1);
    color: #89b4fa;
  }
  .pane-conflict .code-container {
    background: rgba(137, 180, 250, 0.02);
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
    background: rgba(166, 227, 161, 0.03);
    border: 1px solid rgba(166, 227, 161, 0.15);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(166, 227, 161, 0.05);
  }

  .resolution-header-bar {
    background: rgba(166, 227, 161, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 14px;
    border-bottom: 1px solid rgba(166, 227, 161, 0.12);
  }

  .resolution-title-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .ai-glow-star {
    color: #a6e3a1;
    animation: pulse 2s infinite alternate;
  }

  .resolution-title {
    font-size: 12px;
    font-weight: 700;
    color: #a6e3a1;
  }

  .ai-badge {
    font-size: 10px;
    font-weight: 800;
    background: #a6e3a1;
    color: #11111b;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .ai-code {
    background: rgba(0, 0, 0, 0.3) !important;
  }

  .selection-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.02);
    padding: 8px 12px;
    border-radius: 6px;
    margin-top: 4px;
    flex-wrap: wrap;
    gap: 8px;
  }

  .selection-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary, #a6adc8);
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
    background: #313244;
    border: 1px solid transparent;
    color: var(--text-secondary, #a6adc8);
    transition: all 0.15s ease;
  }

  .opt-btn:hover {
    background: #45475a;
    color: #cdd6f4;
  }

  .opt-btn.active {
    background: var(--accent, #cba6f7);
    color: #11111b;
  }

  .opt-btn.active.opt-ai {
    background: #a6e3a1;
    color: #11111b;
    box-shadow: 0 0 10px rgba(166, 227, 161, 0.2);
  }

  .error-banner {
    background: rgba(243, 139, 168, 0.1);
    border: 1px solid rgba(243, 139, 168, 0.2);
    border-radius: 8px;
    padding: 12px;
    margin: 0 20px;
    font-size: 12px;
    color: #f38ba8;
  }

  .success-banner {
    background: rgba(166, 227, 161, 0.1);
    border: 1px solid rgba(166, 227, 161, 0.2);
    border-radius: 8px;
    padding: 12px;
    margin: 0 20px;
    font-size: 12px;
    color: #a6e3a1;
    text-align: center;
  }

  .action-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 16px 20px;
    background: #11111b;
    border-top: 1px solid var(--border, #313244);
  }

  .btn-secondary {
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    border-radius: var(--radius-sm, 6px);
    background: #313244;
    color: var(--text-secondary, #a6adc8);
    transition: all 0.15s ease;
  }

  .btn-secondary:hover {
    background: #45475a;
    color: #cdd6f4;
  }

  .btn-primary {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    border-radius: var(--radius-sm, 6px);
    background: #a6e3a1;
    color: #11111b;
    transition: all 0.15s ease;
  }

  .btn-primary:hover:not(:disabled) {
    background: #94e2d5;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(166, 227, 161, 0.3);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @keyframes pulse {
    0% {
      opacity: 0.7;
      text-shadow: 0 0 2px rgba(166, 227, 161, 0.5);
    }
    100% {
      opacity: 1;
      text-shadow: 0 0 10px rgba(166, 227, 161, 0.9);
    }
  }
</style>
