<script lang="ts">
  import { onMount } from "svelte";
  import { call, requireResultStatus } from "../api/daemon";
  import { NOT_CHANGED, isCancelled } from "../api/configSync";

  interface LastEvaluation {
    evaluated_at: string;
    action_count: number;
    risk_score: number;
    reasons: string[];
    worst_action_type: string | null;
    prediction_sources: string[];
    prediction_confidence: number | null;
    predictions: Array<{
      action_type: string;
      uncertainty: number;
      sources: string[];
      expected_effects: Array<{ domain: string; operation: string; target: string }>;
    }>;
  }

  interface PredictionContract {
    contract_version: string;
    structured_model_version: string;
    ui_jepa: {
      weights_loaded: boolean;
      model_version: string;
      training_samples: number;
      validation_error: number | null;
      validated_for_gating: boolean;
      latent_dimension: number;
      mode: "shadow" | "gating";
    };
  }

  interface RiskStatus {
    status?: string;
    message?: string;
    error?: string;
    enabled: boolean;
    weights_loaded: boolean;
    model_version: string;
    training_samples: number;
    validation_samples: number;
    calibrated: boolean;
    validation_mae: { disk_delta: number; process_delta: number } | null;
    embedding_size: number;
    learnable_action_types: string[];
    prediction_contract: PredictionContract;
    last_evaluation: LastEvaluation | null;
  }

  let enabled = $state(false);
  let runtimeEnabled = $state(false);
  let weightsLoaded = $state(false);
  let modelVersion = $state("unknown");
  let trainingSamples = $state(0);
  let validationSamples = $state(0);
  let calibrated = $state(false);
  let validationMae = $state<{ disk_delta: number; process_delta: number } | null>(null);
  let embeddingSize = $state(0);
  let actionTypes = $state<string[]>([]);
  let lastEvaluation = $state<LastEvaluation | null>(null);
  let predictionContract = $state<PredictionContract | null>(null);
  let loading = $state(true);
  let statusAvailable = $state(false);
  let saving = $state(false);
  let saved = $state(false);
  // Turning the check off waits for the person's approval in the daemon's dialog; without it nothing changed.
  let notChanged = $state(false);
  let error = $state("");

  onMount(loadStatus);

  function applyStatus(result: RiskStatus) {
    enabled = result.enabled ?? false;
    runtimeEnabled = result.enabled ?? false;
    weightsLoaded = result.weights_loaded ?? false;
    modelVersion = result.model_version ?? "unknown";
    trainingSamples = result.training_samples ?? 0;
    validationSamples = result.validation_samples ?? 0;
    calibrated = result.calibrated ?? false;
    validationMae = result.validation_mae ?? null;
    embeddingSize = result.embedding_size ?? 0;
    actionTypes = result.learnable_action_types ?? [];
    predictionContract = result.prediction_contract ?? null;
    lastEvaluation = result.last_evaluation ?? null;
    statusAvailable = true;
  }

  function statusError(cause: unknown): string {
    const message = cause instanceof Error ? cause.message : String(cause);
    if (/method not found.*risk_gate_status/i.test(message)) {
      return "This app is connected to an older Heliox daemon. Restart the daemon, then retry.";
    }
    return `Could not load world-model status: ${message}`;
  }

  async function loadStatus() {
    loading = true;
    error = "";
    try {
      const result = (await call("risk_gate_status")) as RiskStatus;
      requireResultStatus(result, "ok", "The daemon rejected the world-model status request.");
      applyStatus(result);
    } catch (cause) {
      statusAvailable = false;
      error = statusError(cause);
    } finally {
      loading = false;
    }
  }

  async function save() {
    saving = true;
    saved = false;
    notChanged = false;
    error = "";
    try {
      const result = (await call("risk_gate_config_update", { enabled })) as RiskStatus;
      if (isCancelled(result)) {
        enabled = runtimeEnabled;
        notChanged = true;
        return;
      }
      requireResultStatus(result, "ok", "The daemon rejected the world-model setting.");
      applyStatus(result);
      saved = true;
      setTimeout(() => (saved = false), 2500);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : "Could not save the setting";
    } finally {
      saving = false;
    }
  }

  function formatScore(score: number): string {
    return `${Math.round(score * 100)}%`;
  }

  function formatMae(value: number): string {
    return value < 0.0001 ? value.toExponential(2) : value.toFixed(5);
  }
</script>

<div class="world-model-panel">
  <div class="panel-header">
    <div>
      <h3>Hybrid World Model</h3>
      <span class="subtitle">Structured OS/UI transitions, learned risk, and optional UI-JEPA</span>
    </div>
    <button
      class="toggle"
      class:active={enabled}
      onclick={() => {
        enabled = !enabled;
        notChanged = false;
      }}
      aria-label="Toggle Learned Risk World Model"
      aria-pressed={enabled}
      title="Toggle Learned Risk World Model"
      disabled={loading || !statusAvailable}
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  <p class="safety-note">
    Learned predictions run beside deterministic safety rules. The riskier result wins, so the model can add caution but
    cannot remove a rule-based warning. It can run at the same time as the camera's 3D gesture model: the camera model
    improves hand recognition, while this model pauses risky OS actions. Neither disables the other.
  </p>

  {#if loading}
    <div class="empty">Loading model status...</div>
  {:else if !statusAvailable}
    <div class="unavailable" role="alert">
      <strong>World-model status unavailable</strong>
      <span>{error}</span>
      <button class="btn-secondary" onclick={loadStatus}>Retry</button>
    </div>
  {:else}
    <div class="status-grid">
      <div class="status-card">
        <span class="status-label">Runtime</span>
        <strong class:ready={runtimeEnabled}>{runtimeEnabled ? "Enabled" : "Disabled"}</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Weights</span>
        <strong class:ready={weightsLoaded}>{weightsLoaded ? "Loaded" : "Rule fallback only"}</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Training data</span>
        <strong>{trainingSamples.toLocaleString()} real samples</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Coverage</span>
        <strong>{actionTypes.length} action types / {embeddingSize} inputs</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Validation</span>
        <strong class:ready={validationSamples > 0}>{validationSamples.toLocaleString()} held-out samples</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Calibration</span>
        <strong class:ready={calibrated}>{calibrated ? "Active" : "Unavailable"}</strong>
      </div>
      <div class="status-card">
        <span class="status-label">Prediction contract</span>
        <strong class:ready={Boolean(predictionContract)}>
          {predictionContract?.contract_version ?? "Unavailable"}
        </strong>
      </div>
      <div class="status-card">
        <span class="status-label">UI-JEPA</span>
        <strong class:ready={predictionContract?.ui_jepa.weights_loaded ?? false}>
          {predictionContract?.ui_jepa.weights_loaded
            ? `${predictionContract.ui_jepa.mode} · ${predictionContract.ui_jepa.training_samples.toLocaleString()} samples`
            : "Optional · no weights staged"}
        </strong>
      </div>
    </div>

    <div class="model-meta">
      <span>Model</span>
      <code>{modelVersion}</code>
      {#if validationMae}
        <span>
          Held-out MAE: disk {formatMae(validationMae.disk_delta)} · process {formatMae(validationMae.process_delta)}
        </span>
      {/if}
    </div>

    <div class="evaluation">
      <div class="evaluation-header">
        <span class="status-label">Latest plan evaluation</span>
        <button class="btn-secondary" onclick={loadStatus}>Refresh</button>
      </div>
      {#if lastEvaluation}
        <div class="evaluation-summary">
          <strong>{formatScore(lastEvaluation.risk_score)} risk</strong>
          <span>{lastEvaluation.action_count} actions</span>
          <span>{lastEvaluation.worst_action_type ?? "no action"}</span>
          <span>{lastEvaluation.prediction_sources.join(" + ") || "rule"}</span>
          <span>{lastEvaluation.predictions?.length ?? 0} structured predictions</span>
          {#if lastEvaluation.prediction_confidence != null}
            <span>{formatScore(lastEvaluation.prediction_confidence)} model confidence</span>
          {/if}
        </div>
        {#if lastEvaluation.reasons.length}
          <ul>
            {#each lastEvaluation.reasons as reason}
              <li>{reason}</li>
            {/each}
          </ul>
        {:else}
          <p class="empty inline">No safety threshold fired.</p>
        {/if}
      {:else}
        <p class="empty inline">No plan has been evaluated in this daemon session.</p>
      {/if}
    </div>

    {#if error}
      <div class="error" role="alert">{error}</div>
    {:else if notChanged}
      <p class="not-changed" role="status">{NOT_CHANGED}</p>
    {/if}

    <div class="actions">
      <button class="btn-save" onclick={save} disabled={saving}>
        {saving ? "Saving..." : saved ? "✓ Saved" : "Save"}
      </button>
    </div>
  {/if}
</div>

<style>
  .world-model-panel {
    display: flex;
    flex-direction: column;
  }

  .panel-header,
  .evaluation-header,
  .actions {
    display: flex;
    align-items: center;
    justify-content: space-between;
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
  .status-label {
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
    padding: 9px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .status-card strong {
    overflow: hidden;
    font-size: 12px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .ready {
    color: var(--success);
  }

  .model-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    padding: 0 14px 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  code {
    color: var(--text-secondary);
  }

  .evaluation {
    margin: 0 14px;
    padding: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .evaluation-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .evaluation-summary strong {
    color: var(--text-primary);
  }

  ul {
    margin: 8px 0 0;
    padding-left: 18px;
    font-size: 11px;
    color: var(--warning);
  }

  .empty,
  .error,
  .unavailable {
    padding: 18px;
    font-size: 12px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty.inline {
    padding: 8px 0 0;
    margin: 0;
    text-align: left;
  }

  .not-changed {
    margin: 0;
    padding: 8px 14px 0;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .error {
    padding: 8px 14px 0;
    color: var(--danger);
  }

  .unavailable {
    display: flex;
    align-items: flex-start;
    flex-direction: column;
    gap: 7px;
    margin: 12px 14px;
    padding: 12px;
    color: var(--danger);
    text-align: left;
    background: color-mix(in srgb, var(--danger) 8%, transparent);
    border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
    border-radius: var(--radius-sm);
  }

  .unavailable span {
    color: var(--text-secondary);
  }

  .toggle:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }

  .actions {
    justify-content: flex-end;
    padding: 10px 14px 12px;
  }

  .toggle {
    position: relative;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
    cursor: pointer;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 11px;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
    color: var(--on-accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    transition: transform 0.2s;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .btn-save,
  .btn-secondary {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: var(--on-accent);
    background: var(--accent);
    border-radius: var(--radius-sm);
  }

  .btn-secondary {
    padding: 3px 8px;
    font-size: 10px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
  }

  @media (max-width: 900px) {
    .status-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
