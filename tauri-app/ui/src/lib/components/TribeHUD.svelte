<script lang="ts">
  import { call, isConnected } from "../api/daemon";

  let attention = $state(0);
  let stress = $state(0);
  let load = $state(0);
  let modality = $state("VISUAL");
  let connected = $state(false);
  
  // Track basic user activity to feed real stimuli to the neural engine
  let currentStimulus = "User is passively observing the dashboard";
  
  function updateStimulus(activity: string) {
    currentStimulus = activity;
  }

  async function fetchCognitiveState() {
    if (!isConnected()) {
      connected = false;
      return;
    }
    try {
      // Pass the stimulus so the backend uses the real TRIBE model instead of heuristics
      const state: any = await call("cognitive_state", { stimulus: currentStimulus });
      if (state && !state.error) {
        attention = state.attention_score || 0;
        stress = state.stress_level || 0;
        load = state.cognitive_load || 0;
        modality = (state.dominant_modality || "VISUAL").toUpperCase();
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
    
    const onMouseMove = () => updateStimulus("User is actively moving the mouse and exploring the interface");
    const onKeyPress = () => updateStimulus("User is actively typing on the keyboard");
    const onClick = () => updateStimulus("User is clicking and interacting with the system");
    
    window.addEventListener("mousemove", onMouseMove, { once: true });
    window.addEventListener("keypress", onKeyPress);
    window.addEventListener("click", onClick);
    
    // Reset to passive if idle for 5 seconds
    const idleInterval = setInterval(() => {
      updateStimulus("User is passively observing the dashboard");
      window.addEventListener("mousemove", onMouseMove, { once: true });
    }, 5000);

    return () => {
      clearInterval(interval);
      clearInterval(idleInterval);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("keypress", onKeyPress);
      window.removeEventListener("click", onClick);
    };
  });

  function getAttentionColor(val: number) {
    if (val > 0.7) return "#00ff88";
    if (val > 0.4) return "#00c8ff";
    return "#888888";
  }

  function getStressColor(val: number) {
    if (val > 0.7) return "#ff3c3c";
    if (val > 0.4) return "#ffb400";
    return "#00ff88";
  }

  function getLoadColor(val: number) {
    if (val > 0.7) return "#7c3aed";
    if (val > 0.4) return "#a78bfa";
    return "#00c8ff";
  }
</script>

<div class="tribe-hud" class:active={connected}>
  <div class="hud-header">
    <div class="title">
      <span class="tribe-dot"></span>
      TRIBE v2 COGNITIVE STATE
    </div>
    <div class="modality">{modality}</div>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="metric-header">
        <span class="label">ATTENTION</span>
        <span class="value" style="color: {getAttentionColor(attention)}">{Math.round(attention * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width: {attention * 100}%; background: {getAttentionColor(attention)}; box-shadow: 0 0 10px {getAttentionColor(attention)}"></div>
      </div>
    </div>

    <div class="metric">
      <div class="metric-header">
        <span class="label">STRESS</span>
        <span class="value" style="color: {getStressColor(stress)}">{Math.round(stress * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width: {stress * 100}%; background: {getStressColor(stress)}; box-shadow: 0 0 10px {getStressColor(stress)}"></div>
      </div>
    </div>

    <div class="metric">
      <div class="metric-header">
        <span class="label">LOAD</span>
        <span class="value" style="color: {getLoadColor(load)}">{Math.round(load * 100)}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width: {load * 100}%; background: {getLoadColor(load)}; box-shadow: 0 0 10px {getLoadColor(load)}"></div>
      </div>
    </div>
  </div>
  
  {#if !connected}
    <div class="overlay">
      <span>INITIALIZING NEURAL LINK...</span>
    </div>
  {/if}
</div>

<style>
  .tribe-hud {
    position: relative;
    margin-top: 14px;
    padding: 12px;
    background: rgba(10, 12, 24, 0.6);
    border: 1px solid rgba(124, 58, 237, 0.2);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.5s ease;
  }
  
  .tribe-hud.active {
    border-color: rgba(124, 58, 237, 0.5);
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.1), inset 0 0 10px rgba(124, 58, 237, 0.05);
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
    color: rgba(255, 255, 255, 0.9);
  }

  .tribe-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #7c3aed;
    box-shadow: 0 0 8px #7c3aed;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
  }

  .modality {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    background: rgba(124, 58, 237, 0.15);
    color: #a78bfa;
    border-radius: 4px;
    letter-spacing: 1px;
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
    color: rgba(200, 200, 220, 0.6);
  }

  .value {
    font-size: 10px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }

  .bar-bg {
    height: 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1), background 1s;
  }

  .overlay {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(10, 12, 24, 0.8);
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
    color: rgba(124, 58, 237, 0.8);
    animation: flash 1.5s infinite;
  }

  @keyframes flash {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
  }
</style>
