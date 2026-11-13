<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call, requireOkResult, type DaemonStatusResult } from "../api/daemon";
  import { lateRetry, startingMessage } from "../api/lateParts";

  interface LearnerStatus {
    enabled: boolean;
    model_version: string;
    backend: string;
    authority: string;
    event_cursor: number;
    suggestions: {
      labels: number;
      positive: number;
      negative: number;
      replay_samples: number;
      drift_events: number;
      promotion_threshold: number;
    };
    transitions: {
      labels: number;
      positive: number;
      negative: number;
      replay_samples: number;
      drift_events: number;
      promotion_threshold: number;
    };
    prediction_errors: number;
    corrections: number;
    explicit_rules: number;
    routine_patterns: Array<{ pattern: string; decayed_evidence: number }>;
    workflow_patterns: number;
    privacy: {
      raw_media_stored: boolean;
      secret_browsing: boolean;
      external_observation_requires_permission: boolean;
    };
  }

  let status = $state<LearnerStatus | null>(null);
  let loading = $state(true);
  let resetting = $state(false);
  let confirmingReset = $state(false);
  let error = $state("");
  /** The learner loads after Heliox is ready: until it has, the daemon's sentence saying so (api/lateParts.ts). */
  let waiting = $state("");
  const retry = lateRetry();

  onMount(loadStatus);
  onDestroy(retry.cancel);

  async function loadStatus() {
    loading = true;
    error = "";
    try {
      const reply = (await call("online_learning_status")) as LearnerStatus & DaemonStatusResult;
      waiting = startingMessage(reply);
      if (waiting) {
        retry.schedule(loadStatus);
        return;
      }
      const result = requireOkResult(reply, "Verified online learning is unavailable.");
      if (!result.enabled) {
        throw new Error("Verified online learning is unavailable.");
      }
      status = result;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      error = /method not found.*online_learning_status/i.test(message)
        ? "This app is connected to an older Heliox daemon. Restart the daemon, then retry."
        : `Could not load online-learning status: ${message}`;
    } finally {
      loading = false;
    }
  }

  async function resetLearning() {
    if (!confirmingReset) {
      confirmingReset = true;
      return;
    }
    resetting = true;
    error = "";
    try {
      status = requireOkResult(
        (await call("online_learning_reset")) as LearnerStatus & DaemonStatusResult,
        "The daemon did not reset learned adaptation.",
      );
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not reset learned adaptation.";
    } finally {
      resetting = false;
      confirmingReset = false;
    }
  }

  function evidenceState(labels: number, threshold: number): string {
    return labels >= threshold ? "eligible for promoted ranking" : `${threshold - labels} more labels required`;
  }

  function readableRoutine(pattern: string): string {
    const match = /^app:(.+):hour:(\d+)$/.exec(pattern);
    if (!match) return pattern;
    const app = match[1].replaceAll("_", " ");
    const startHour = Number(match[2]) * 3;
    return `${app} · ${String(startHour).padStart(2, "0")}:00–${String(startHour + 3).padStart(2, "0")}:00`;
  }
</script>

<div class="learning-panel">
  <div class="panel-header">
    <div>
      <h3>Verified Continuous Learning</h3>
      <span class="subtitle">On-device River adaptation from explicit feedback and verified outcomes</span>
    </div>
    <button class="btn-secondary" onclick={loadStatus} disabled={loading}>Refresh</button>
  </div>

  <p class="safety-note">
    Learning may rank or suppress suggestions, but it cannot approve or execute an action. Passive observations build
    coarse app/time routines only—no raw screen, camera, audio, or window-title content is retained. Heliox never
    browses merely to collect training data.
  </p>

  {#if loading || waiting}
    <div class="empty" role={waiting ? "status" : undefined}>{waiting || "Loading verified learning status..."}</div>
  {:else if error && !status}
    <div class="unavailable" role="alert">
      <strong>Learning status unavailable</strong>
      <span>{error}</span>
      <button class="btn-secondary" onclick={loadStatus}>Retry</button>
    </div>
  {:else if status}
    <div class="status-grid">
      <div class="status-card">
        <span class="status-label">Authority</span>
        <strong class="ready">Advisory ranking only</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Suggestion labels</span>
        <strong>{status.suggestions.labels}</strong>
        <small>{evidenceState(status.suggestions.labels, status.suggestions.promotion_threshold)}</small>
      </div>
      <div class="status-card">
        <span class="status-label">Verified transitions</span>
        <strong>{status.transitions.labels}</strong>
        <small>{status.transitions.positive} succeeded · {status.transitions.negative} failed</small>
      </div>
      <div class="status-card">
        <span class="status-label">Replay protection</span>
        <strong>{status.suggestions.replay_samples + status.transitions.replay_samples} samples</strong>
        <small>Bounded recent evidence</small>
      </div>
      <div class="status-card">
        <span class="status-label">Drift events</span>
        <strong>{status.suggestions.drift_events + status.transitions.drift_events}</strong>
        <small>Outdated behavior is decayed and replayed</small>
      </div>
      <div class="status-card">
        <span class="status-label">Corrections / surprise</span>
        <strong>{status.corrections} / {status.prediction_errors}</strong>
        <small>User corrections · prediction errors</small>
      </div>
    </div>

    <div class="evidence-row">
      <div>
        <span class="status-label">Learned routines</span>
        {#if status.routine_patterns.length}
          <ul>
            {#each status.routine_patterns.slice(0, 4) as routine}
              <li>
                <span>{readableRoutine(routine.pattern)}</span>
                <code>{routine.decayed_evidence.toFixed(2)}</code>
              </li>
            {/each}
          </ul>
        {:else}
          <p class="empty inline">No stable app/time routine has been observed yet.</p>
        {/if}
      </div>
      <div class="model-meta">
        <span>{status.backend}</span>
        <code>{status.model_version}</code>
        <span>Event cursor {status.event_cursor.toLocaleString()}</span>
      </div>
    </div>

    {#if error}
      <div class="error" role="alert">{error}</div>
    {/if}

    <div class="actions">
      <span>Reset forgets trained state but preserves the immutable audit ledger.</span>
      <button
        class="btn-danger"
        onclick={resetLearning}
        disabled={resetting || (status.suggestions.labels === 0 && status.transitions.labels === 0)}
      >
        {resetting ? "Resetting..." : confirmingReset ? "Confirm reset learned adaptation" : "Reset learned adaptation"}
      </button>
    </div>
  {/if}
</div>

<style>
  .learning-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .panel-header {
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .subtitle,
  .status-label,
  small {
    font-size: 11px;
    color: var(--text-muted);
  }

  .safety-note {
    margin: 10px 14px 0;
    padding: 10px 12px;
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-left: 2px solid var(--accent);
    border-radius: var(--radius-sm);
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    padding: 12px 14px;
  }

  .status-card {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 3px;
    padding: 9px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .status-card strong {
    font-size: 12px;
    color: var(--text-primary);
  }

  strong.ready {
    color: var(--success);
  }

  .evidence-row {
    display: grid;
    grid-template-columns: minmax(0, 2fr) minmax(180px, 1fr);
    gap: 12px;
    padding: 0 14px 12px;
  }

  ul {
    margin: 6px 0 0;
    padding: 0;
    list-style: none;
  }

  li,
  .model-meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 5px 8px;
    font-size: 11px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
  }

  .model-meta {
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-muted);
  }

  code {
    color: var(--accent-light);
  }

  .actions {
    padding: 10px 14px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
  }

  .empty,
  .unavailable {
    padding: 20px 14px;
    color: var(--text-muted);
    font-size: 12px;
  }

  .empty.inline {
    padding: 8px 0;
    margin: 0;
  }

  .unavailable {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    color: var(--danger);
  }

  .error {
    margin: 0 14px 10px;
    color: var(--danger);
    font-size: 11px;
  }

  .btn-secondary,
  .btn-danger {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 5px 10px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    cursor: pointer;
    /* Settings' shared button rules set a size; this panel keeps the browser's. */
    font-size: revert;
  }

  .btn-danger {
    color: var(--danger);
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  @media (max-width: 760px) {
    .status-grid,
    .evidence-row {
      grid-template-columns: 1fr;
    }

    .actions {
      align-items: flex-start;
      flex-direction: column;
    }
  }
</style>
