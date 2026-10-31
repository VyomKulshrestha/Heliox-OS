<script lang="ts">
  /**
   * AmbientHUD — Iron Man JARVIS-style holographic system monitor.
   * Shows real-time CPU, RAM, network stats with animated visualizations.
   */

  import { call, isConnected, requireOkResult, type DaemonStatusResult } from "../api/daemon";
  import { invoke } from "../api/invoke";
  import BudgetMeter from "./BudgetMeter.svelte";
  import CognitiveHUD from "./CognitiveHUD.svelte";

  // ── State ──
  let cpuPercent = $state(0);
  let ramPercent = $state(0);
  let ramUsedGB = $state("0");
  let ramTotalGB = $state("0");
  let diskPercent = $state(0);
  let diskUsedGB = $state("0");
  let diskTotalGB = $state("0");
  let networkUp = $state("0 KB/s");
  let networkDown = $state("0 KB/s");
  let hostname = $state("HELIOX");
  let uptime = $state("0h 0m");
  let currentTime = $state("");
  let cpuHistory: number[] = $state(new Array(30).fill(0));
  let isExpanded = $state(false);
  let telemetryAvailable = $state(false);

  // Update time every second
  let timeInterval: ReturnType<typeof setInterval>;

  function updateTime() {
    const now = new Date();
    currentTime = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function applyTelemetry(raw: unknown) {
    const info = requireOkResult(raw as DaemonStatusResult, "System telemetry is unavailable.") as Record<
      string,
      unknown
    >;
    const required = [
      "cpu_percent",
      "memory_percent",
      "memory_used",
      "memory_total",
      "disk_percent",
      "disk_used",
      "disk_total",
      "uptime_seconds",
    ] as const;
    const values = Object.fromEntries(required.map((key) => [key, Number(info[key])])) as Record<
      (typeof required)[number],
      number
    >;
    if (
      required.some((key) => !Number.isFinite(values[key]) || values[key] < 0) ||
      values.cpu_percent > 100 ||
      values.memory_percent > 100 ||
      values.disk_percent > 100 ||
      values.memory_total <= 0 ||
      values.disk_total <= 0 ||
      typeof info.hostname !== "string" ||
      !info.hostname.trim()
    ) {
      throw new Error("System telemetry is incomplete.");
    }

    cpuPercent = Math.round(values.cpu_percent);
    ramPercent = Math.round(values.memory_percent);
    ramUsedGB = (values.memory_used / 1024 ** 3).toFixed(1);
    ramTotalGB = (values.memory_total / 1024 ** 3).toFixed(1);
    diskPercent = Math.round(values.disk_percent);
    diskUsedGB = (values.disk_used / 1024 ** 3).toFixed(0);
    diskTotalGB = (values.disk_total / 1024 ** 3).toFixed(0);
    hostname = info.hostname.trim().toUpperCase();
    const days = Math.floor(values.uptime_seconds / 86400);
    const hrs = Math.floor((values.uptime_seconds % 86400) / 3600);
    const mins = Math.floor((values.uptime_seconds % 3600) / 60);
    uptime = days > 0 ? `${days}d ${hrs}h ${mins}m` : `${hrs}h ${mins}m`;
    cpuHistory = [...cpuHistory.slice(1), cpuPercent];
    telemetryAvailable = true;
  }

  // Fetch system stats from daemon or fallback
  async function fetchStats() {
    try {
      const upRes: any = await call("get_uptime");
      if (upRes && typeof upRes === "string") uptime = upRes;
    } catch (e) {
      try {
        const upRes: any = await invoke("get_uptime");
        if (upRes && typeof upRes === "string") uptime = upRes;
      } catch (e2) {}
    }

    if (!isConnected()) {
      try {
        applyTelemetry(await invoke("system_info"));
        return;
      } catch (e) {
        telemetryAvailable = false;
      }
      return;
    }

    try {
      applyTelemetry(await call("system_info"));
    } catch {
      try {
        applyTelemetry(await invoke("system_info"));
        return;
      } catch (e) {}
      telemetryAvailable = false;
    }
  }

  // Generate SVG path for CPU history graph
  function getCpuGraphPath(): string {
    const width = 200;
    const height = 40;
    const step = width / (cpuHistory.length - 1);

    let path = `M 0 ${height - (cpuHistory[0] / 100) * height}`;
    for (let i = 1; i < cpuHistory.length; i++) {
      const x = i * step;
      const y = height - (cpuHistory[i] / 100) * height;
      path += ` L ${x} ${y}`;
    }
    return path;
  }

  function getCpuAreaPath(): string {
    const width = 200;
    const height = 40;
    const step = width / (cpuHistory.length - 1);

    let path = `M 0 ${height}`;
    for (let i = 0; i < cpuHistory.length; i++) {
      const x = i * step;
      const y = height - (cpuHistory[i] / 100) * height;
      path += ` L ${x} ${y}`;
    }
    path += ` L ${width} ${height} Z`;
    return path;
  }

  function getStatusColor(percent: number): string {
    if (percent > 85) return "var(--danger)";
    if (percent > 65) return "var(--warning)";
    return "var(--success)";
  }

  $effect(() => {
    updateTime();
    timeInterval = setInterval(updateTime, 1000);
    fetchStats();
    const statsInterval = setInterval(fetchStats, 3000);

    return () => {
      clearInterval(timeInterval);
      clearInterval(statsInterval);
    };
  });
</script>

<div class="ambient-hud" class:expanded={isExpanded}>
  <button class="hud-toggle" onclick={() => (isExpanded = !isExpanded)} title="Toggle HUD">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="16" height="16">
      <path d="M12 2L2 7l10 5 10-5-10-5z" />
      <path d="M2 17l10 5 10-5" />
      <path d="M2 12l10 5 10-5" />
    </svg>
  </button>

  {#if isExpanded}
    <div class="hud-panel">
      <!-- Header -->
      <div class="hud-header">
        <div class="hud-title">
          <span class="heliox-dot"></span>
          HELIOX OS
        </div>
        <div class="hud-time">{currentTime}</div>
      </div>

      <div class="hud-divider"></div>

      <!-- CPU Usage -->
      <div class="stat-section">
        <div class="stat-header">
          <span class="stat-label">CPU</span>
          <span class="stat-value" style="color: {getStatusColor(cpuPercent)}"
            >{telemetryAvailable ? `${cpuPercent}%` : "Unavailable"}</span
          >
        </div>
        <div class="cpu-graph">
          <svg viewBox="0 0 200 40" preserveAspectRatio="none">
            <defs>
              <linearGradient id="cpuFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="rgba(0, 200, 255, 0.3)" />
                <stop offset="100%" stop-color="rgba(0, 200, 255, 0.02)" />
              </linearGradient>
            </defs>
            <path d={getCpuAreaPath()} fill="url(#cpuFill)" />
            <path d={getCpuGraphPath()} fill="none" stroke="rgba(0, 200, 255, 0.8)" stroke-width="1.5" />
          </svg>
        </div>
      </div>

      <!-- RAM Usage -->
      <div class="stat-section">
        <div class="stat-header">
          <span class="stat-label">MEMORY</span>
          <span class="stat-value">{telemetryAvailable ? `${ramUsedGB} / ${ramTotalGB} GB` : "Unavailable"}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {ramPercent}%; background: {getStatusColor(ramPercent)};"></div>
          <span class="progress-text">{telemetryAvailable ? `${ramPercent}%` : "--"}</span>
        </div>
      </div>

      <!-- Disk Usage -->
      <div class="stat-section">
        <div class="stat-header">
          <span class="stat-label">DISK</span>
          <span class="stat-value">{telemetryAvailable ? `${diskUsedGB} / ${diskTotalGB} GB` : "Unavailable"}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: {diskPercent}%; background: {getStatusColor(diskPercent)};"></div>
          <span class="progress-text">{telemetryAvailable ? `${diskPercent}%` : "--"}</span>
        </div>
      </div>

      <BudgetMeter />
      <CognitiveHUD />

      <!-- Uptime & Network -->
      <div class="hud-divider"></div>
      <div class="hud-footer">
        <div class="footer-item">
          <span class="footer-label">UPTIME</span>
          <span class="footer-value">{uptime}</span>
        </div>
        <div class="footer-item">
          <span class="footer-label">HOST</span>
          <span class="footer-value">{hostname}</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .ambient-hud {
    position: relative;
  }

  .hud-toggle {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
    background: color-mix(in srgb, var(--accent) 4%, transparent);
    color: var(--accent);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s;
  }

  .hud-toggle:hover {
    border-color: color-mix(in srgb, var(--accent) 50%, transparent);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 10%, transparent);
  }

  .ambient-hud.expanded .hud-toggle {
    border-color: color-mix(in srgb, var(--accent) 50%, transparent);
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
  }

  .hud-panel {
    position: fixed;
    top: 50px;
    right: 8px;
    width: 260px;
    background: var(--bg-primary);
    border: 1px solid color-mix(in srgb, var(--accent) 15%, transparent);
    border-radius: 12px;
    padding: 14px;
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    z-index: 500;
    animation: hudSlideIn 0.3s ease;
    font-family: "Inter", "JetBrains Mono", monospace;
  }

  @keyframes hudSlideIn {
    from {
      opacity: 0;
      transform: translateY(-8px) scale(0.97);
    }
    to {
      opacity: 1;
      transform: translateY(0) scale(1);
    }
  }

  .hud-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .hud-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--accent);
  }

  .heliox-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: color-mix(in srgb, var(--success) 90%, transparent);

    animation: dot-pulse 2s ease-in-out infinite;
  }

  @keyframes dot-pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.4;
    }
  }

  .hud-time {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
    letter-spacing: 1px;
  }

  .hud-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, color-mix(in srgb, var(--accent) 20%, transparent), transparent);
    margin: 10px 0;
  }

  .stat-section {
    margin-bottom: 10px;
  }

  .stat-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
  }

  .stat-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: var(--text-secondary);
  }

  .stat-value {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    font-variant-numeric: tabular-nums;
  }

  /* CPU Graph */
  .cpu-graph {
    height: 40px;
    border-radius: 6px;
    overflow: hidden;
    background: color-mix(in srgb, var(--accent) 3%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 8%, transparent);
  }

  .cpu-graph svg {
    width: 100%;
    height: 100%;
  }

  /* Progress Bars */
  .progress-bar {
    position: relative;
    height: 14px;
    border-radius: 7px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 7px;
    transition:
      width 0.8s ease,
      background 0.5s ease;
    box-shadow: 0 0 8px currentColor;
    opacity: 0.65;
  }

  .progress-text {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8px;
    font-weight: 700;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
  }

  /* Footer */
  .hud-footer {
    display: flex;
    justify-content: space-between;
  }

  .footer-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .footer-label {
    font-size: 8px;
    font-weight: 600;
    letter-spacing: 1.5px;
    color: var(--text-muted);
  }

  .footer-value {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }
</style>
