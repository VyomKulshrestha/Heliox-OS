<script lang="ts">
  import { call, isConnected, requireResultStatus } from "../api/daemon";

  let attention = $state(0);
  let stress = $state(0);
  let load = $state(0);
  let modality = $state("VISUAL");
  let confidence = $state(0);
  let signalSources = $state(0);
  let connected = $state(false);
  let hasSample = false;

  let sampleWindowStarted = performance.now();
  let lastActivityAt = Date.now();
  let keystrokes = 0;
  let clicks = 0;
  let pointerMoves = 0;
  let lastPointerSampleAt = 0;

  function recordActivity(kind: "keyboard" | "click" | "pointer") {
    lastActivityAt = Date.now();
    if (kind === "keyboard") keystrokes += 1;
    if (kind === "click") clicks += 1;
    if (kind === "pointer") {
      const now = performance.now();
      if (now - lastPointerSampleAt >= 250) {
        pointerMoves += 1;
        lastPointerSampleAt = now;
      }
    }
  }

  function collectInputDynamics() {
    const now = performance.now();
    const elapsedMs = Math.max(1000, now - sampleWindowStarted);
    const rateMultiplier = 60_000 / elapsedMs;
    const sample = {
      keystroke_rate_per_min: keystrokes * rateMultiplier,
      click_rate_per_min: clicks * rateMultiplier,
      pointer_move_rate_per_min: pointerMoves * rateMultiplier,
      idle_seconds: Math.max(0, (Date.now() - lastActivityAt) / 1000),
    };
    sampleWindowStarted = now;
    keystrokes = 0;
    clicks = 0;
    pointerMoves = 0;
    return sample;
  }

  async function fetchCognitiveState() {
    if (!isConnected()) {
      connected = false;
      return;
    }
    try {
      const state = (await call("cognitive_state", {
        input_dynamics: collectInputDynamics(),
      })) as {
        status?: string;
        message?: string;
        error?: string;
        attention_score?: number;
        stress_level?: number;
        cognitive_load?: number;
        confidence?: number;
        signal_sources?: number;
        dominant_modality?: string;
      };
      requireResultStatus(state, "ok", "Cognitive state is unavailable.");
      if (
        Number.isFinite(state.attention_score) &&
        Number.isFinite(state.stress_level) &&
        Number.isFinite(state.cognitive_load)
      ) {
        const blend = hasSample ? 0.35 : 1;
        attention = attention * (1 - blend) + Number(state.attention_score ?? 0.5) * blend;
        stress = stress * (1 - blend) + Number(state.stress_level ?? 0.3) * blend;
        load = load * (1 - blend) + Number(state.cognitive_load ?? 0.4) * blend;
        confidence = Number(state.confidence ?? 0);
        signalSources = Number(state.signal_sources ?? 0);
        modality = (state.dominant_modality || "VISUAL").toUpperCase();
        hasSample = true;
        connected = true;
      } else {
        connected = false;
      }
    } catch {
      connected = false;
    }
  }

  $effect(() => {
    fetchCognitiveState();
    const interval = setInterval(fetchCognitiveState, 2000);

    const onMouseMove = () => recordActivity("pointer");
    const onKeyPress = () => recordActivity("keyboard");
    const onClick = () => recordActivity("click");

    window.addEventListener("pointermove", onMouseMove, { passive: true });
    window.addEventListener("keydown", onKeyPress);
    window.addEventListener("click", onClick);

    return () => {
      clearInterval(interval);
      window.removeEventListener("pointermove", onMouseMove);
      window.removeEventListener("keydown", onKeyPress);
      window.removeEventListener("click", onClick);
    };
  });

  function getAttentionColor(val: number) {
    if (val > 0.7) return "var(--success)";
    if (val > 0.4) return "var(--accent)";
    return "var(--text-muted)";
  }

  function getStressColor(val: number) {
    if (val > 0.7) return "var(--danger)";
    if (val > 0.4) return "var(--warning)";
    return "var(--success)";
  }

  function getLoadColor(val: number) {
    if (val > 0.7) return "var(--warning)";
    if (val > 0.4) return "var(--accent)";
    return "var(--text-secondary)";
  }
</script>

<div class="cognitive-hud" class:active={connected}>
  <div class="hud-header">
    <div class="title">
      <span class="cognitive-dot"></span>
      COGNITIVE STATE
    </div>
    <div class="estimate-meta">
      <div class="modality">{modality}</div>
      <div class="confidence" title="Confidence based on the amount and freshness of local behavioural signals">
        {Math.round(confidence * 100)}% CONF
      </div>
    </div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="metric-header">
        <span class="label">ATTENTION</span>
        <span class="value" style="color: {getAttentionColor(attention)}">{Math.round(attention * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div
          class="bar-fill"
          style="width: {attention * 100}%; background: {getAttentionColor(
            attention,
          )}; box-shadow: 0 0 10px {getAttentionColor(attention)}"
        ></div>
      </div>
    </div>

    <div class="metric">
      <div class="metric-header">
        <span class="label">STRESS</span>
        <span class="value" style="color: {getStressColor(stress)}">{Math.round(stress * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div
          class="bar-fill"
          style="width: {stress * 100}%; background: {getStressColor(stress)}; box-shadow: 0 0 10px {getStressColor(
            stress,
          )}"
        ></div>
      </div>
    </div>

    <div class="metric">
      <div class="metric-header">
        <span class="label">LOAD</span>
        <span class="value" style="color: {getLoadColor(load)}">{Math.round(load * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div
          class="bar-fill"
          style="width: {load * 100}%; background: {getLoadColor(load)}; box-shadow: 0 0 10px {getLoadColor(load)}"
        ></div>
      </div>
    </div>
  </div>

  <div class="estimate-note">
    {signalSources > 0 ? `${signalSources} live signal source${signalSources === 1 ? "" : "s"}` : "Collecting signals"}
    · behavioural estimate, not a medical measurement
  </div>

  {#if !connected}
    <div class="overlay">
      <span>CONNECTING...</span>
    </div>
  {/if}
</div>

<style>
  .cognitive-hud {
    position: relative;
    margin-top: 14px;
    padding: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 88%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.5s ease;
  }

  .cognitive-hud.active {
    border-color: color-mix(in srgb, var(--accent) 50%, transparent);
  }

  .hud-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: var(--text-primary);
  }

  .cognitive-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);

    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.5;
      transform: scale(0.8);
    }
  }

  .modality {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    background: color-mix(in srgb, var(--accent) 15%, transparent);
    color: var(--accent);
    border-radius: 4px;
    letter-spacing: 1px;
  }

  .estimate-meta {
    display: flex;
    gap: 5px;
    align-items: center;
  }

  .confidence {
    padding: 2px 6px;
    font-size: 8px;
    font-weight: 700;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    border-radius: 4px;
    letter-spacing: 0.7px;
  }

  .metrics {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1px;
    color: var(--text-secondary);
  }

  .value {
    font-size: 10px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .bar-bg {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition:
      width 1s cubic-bezier(0.4, 0, 0.2, 1),
      background 1s;
  }

  .estimate-note {
    margin-top: 9px;
    font-size: 8px;
    line-height: 1.35;
    color: var(--text-muted);
  }

  .overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: color-mix(in srgb, var(--bg-secondary) 94%, transparent);
    backdrop-filter: blur(2px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2;
  }

  .overlay span {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--accent);
    animation: flash 1.5s infinite;
  }

  @keyframes flash {
    0%,
    100% {
      opacity: 0.4;
    }
    50% {
      opacity: 1;
    }
  }
</style>
