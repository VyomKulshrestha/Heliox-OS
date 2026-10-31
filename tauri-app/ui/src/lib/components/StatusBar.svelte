<script lang="ts">
  import { invoke } from "../api/invoke";
  import { isConnected } from "../api/daemon";
  import { onMount } from "svelte";
  type DashboardStatus = {
    connected: boolean;
    agents: number | null;
    cpu: string;
    memory: string;
    network_up: string;
    network_down: string;
  };
  let status: DashboardStatus = {
    connected: false,
    agents: null,
    cpu: "0%",
    memory: "0%",
    network_up: "0 KB/s",
    network_down: "0 KB/s",
  };
  async function loadStatus() {
    try {
      const res = await invoke("get_dashboard_status");
      if (res && typeof res === "object") {
        status = { ...res, connected: isConnected() };
      }
    } catch (err) {
      console.error("Dashboard status failed", err);
    }
  }
  onMount(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 2000);
    return () => clearInterval(interval);
  });
</script>

<div class="status-wrapper">
  <div class="status-bar">
    <div class="scroll-content">
      <div class="status-item">
        <span class="label"> JSON-RPC </span>
        <div class:connected={status.connected} class="badge">
          ●
          {status.connected ? "Connected" : "Disconnected"}
        </div>
      </div>
      <div class="status-item">
        <span class="label"> Agents </span>
        <div class="badge active">
          {status.agents == null ? "Unavailable" : `+${status.agents} Active`}
        </div>
      </div>
      <div class="status-item">
        <span class="label"> CPU Load </span>
        <div class="badge cpu">
          {status.cpu}
        </div>
      </div>
      <div class="status-item">
        <span class="label"> Memory </span>
        <div class="badge memory">
          ● {status.memory}
        </div>
      </div>
      <div class="status-item network">
        <span class="label"> Network </span>
        <div class="network-stats">
          ↑ {status.network_up}
          ↓ {status.network_down}
        </div>
      </div>
      <div class="dashboard-title">Heliox OS Dashboard • Real-time Agentic System Monitor</div>
    </div>
  </div>
</div>

<style>
  .status-wrapper {
    width: 100%;
    overflow: hidden;
    border-radius: 24px;
    margin-top: 26px;
  }
  .status-bar {
    width: 100%;
    padding: 20px 28px;
    border-radius: 24px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    color: var(--text-primary);
    overflow: hidden;
  }
  .scroll-content {
    display: flex;
    align-items: center;
    gap: 40px;
    width: max-content;
    animation: scrollStatus 14s linear infinite;
  }
  @keyframes scrollStatus {
    0% {
      transform: translateX(100%);
    }
    100% {
      transform: translateX(-100%);
    }
  }
  .status-item {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
  }
  .label {
    font-size: 15px;
    color: var(--text-secondary);
    white-space: nowrap;
  }
  .badge {
    padding: 8px 16px;
    border-radius: 999px;
    background: var(--bg-hover);
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
  }
  .badge.connected {
    background: rgba(0, 255, 170, 0.12);
    color: var(--success);
    border: 1px solid rgba(0, 255, 170, 0.35);
  }
  .badge.active {
    background: rgba(0, 255, 170, 0.12);
    color: var(--success);
  }
  .badge.cpu {
    background: rgba(0, 255, 170, 0.12);
    color: var(--success);
  }
  .badge.memory {
    background: rgba(139, 92, 246, 0.14);
    color: var(--accent);
  }
  .network {
    gap: 12px;
  }
  .network-stats {
    color: var(--text-primary);
    font-size: 15px;
    white-space: nowrap;
  }
  .dashboard-title {
    color: var(--text-secondary);
    font-size: 15px;
    white-space: nowrap;
    padding-right: 40px;
  }
</style>
