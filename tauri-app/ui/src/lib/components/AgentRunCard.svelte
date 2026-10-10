<script lang="ts">
  import type { AgentRun, ToolCall } from "../utils/agentRuns";

  let { run }: { run: AgentRun } = $props();

  let expanded = $state<Record<string, boolean>>({});

  const SOURCE_LABELS: Record<string, string> = {
    scheduled: "Scheduled run",
    trigger: "Trigger",
    recorded_workflow: "Recorded workflow",
    recorded_workflow_schedule: "Recorded workflow",
    neural: "Neural task",
    proactive: "Suggestion",
    voice: "Voice task",
  };

  const STATUS_LABELS: Record<string, string> = {
    running: "Working",
    queued: "Working",
    decomposing: "Working",
    success: "Done",
    partial: "Partly done",
    failed: "Failed",
    cancelled: "Cancelled",
    interrupted: "Interrupted",
  };

  function label(call: ToolCall): string {
    const name = call.actionType.replace(/_/g, " ");
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function seconds(ms: number): string {
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
  }

  function toggle(callId: string) {
    expanded = { ...expanded, [callId]: !expanded[callId] };
  }
</script>

<div class="message agent-run">
  <div class="run-header">
    <span class="msg-label">{SOURCE_LABELS[run.source] ?? "Background task"}</span>
    <span class="run-status {run.status}">{STATUS_LABELS[run.status] ?? run.status}</span>
  </div>
  <p class="run-goal">{run.goal}</p>
  <ol class="calls">
    {#each run.calls as call (call.callId)}
      <li class="call {call.status}">
        <button
          class="call-row"
          type="button"
          aria-expanded={Boolean(expanded[call.callId])}
          onclick={() => toggle(call.callId)}
        >
          <span class="dot" aria-hidden="true"></span>
          <span class="call-name">{label(call)}</span>
          {#if call.target}<code class="call-target">{call.target}</code>{/if}
          <span class="call-meta">
            {#if call.status === "running"}running{:else if call.status === "skipped"}not run{:else}{seconds(
                call.durationMs,
              )}{/if}
          </span>
        </button>
        {#if expanded[call.callId]}
          <div class="call-detail">
            {#if call.arguments && call.arguments !== "{}"}
              <span class="detail-label">Arguments</span>
              <pre>{call.arguments}</pre>
            {/if}
            {#if call.error}
              <span class="detail-label">Error</span>
              <pre class="error-text">{call.error}</pre>
            {:else if call.output}
              <span class="detail-label">Output</span>
              <pre>{call.output}</pre>
            {/if}
            <span class="detail-meta">Step {call.step + 1}, round {call.round} · {call.tier.replace(/_/g, " ")}</span>
          </div>
        {/if}
      </li>
    {/each}
  </ol>
</div>

<style>
  .agent-run {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 12px;
    border: 1px solid var(--accent-muted);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-md);
    background: var(--bg-tertiary);
  }
  .msg-label {
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .run-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .run-status {
    padding: 2px 8px;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-secondary);
    font: 600 10px/1.4 var(--font-mono);
  }
  .run-status.success {
    border-color: var(--success);
    color: var(--success);
  }
  .run-status.failed,
  .run-status.cancelled {
    border-color: var(--danger);
    color: var(--danger);
  }
  .run-status.partial,
  .run-status.interrupted {
    border-color: var(--warning);
    color: var(--warning);
  }
  .run-goal {
    margin: 0 0 2px;
    font-size: 12px;
    color: var(--text-secondary);
    white-space: pre-wrap;
  }
  .calls {
    display: grid;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .call {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
  }
  .call-row {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 6px 10px;
    border: 0;
    background: none;
    color: var(--text-primary);
    font-size: 12px;
    text-align: left;
    cursor: pointer;
  }
  .dot {
    flex-shrink: 0;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
  }
  .call.running .dot {
    background: var(--accent);
    animation: pulse 1s ease-in-out infinite;
  }
  .call.success .dot {
    background: var(--success);
  }
  .call.error .dot {
    background: var(--danger);
  }
  .call-name {
    flex-shrink: 0;
    font-weight: 600;
  }
  .call-target {
    min-width: 0;
    overflow: hidden;
    color: var(--text-secondary);
    font-size: 11px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .call-meta {
    flex-shrink: 0;
    margin-left: auto;
    color: var(--text-muted);
    font: 10px/1 var(--font-mono);
  }
  .call-detail {
    display: grid;
    gap: 4px;
    padding: 0 10px 8px 25px;
  }
  .detail-label,
  .detail-meta {
    color: var(--text-muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .detail-meta {
    text-transform: none;
    letter-spacing: 0;
  }
  pre {
    max-height: 180px;
    margin: 0;
    overflow: auto;
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    font: 11px/1.45 var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .error-text {
    color: var(--danger);
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .call.running .dot {
      animation: none;
    }
  }
</style>
