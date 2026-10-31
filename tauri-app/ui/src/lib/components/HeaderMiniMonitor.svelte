<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { invoke } from "../api/invoke";
  import { fade } from "svelte/transition";

  type Stats = {
    cpu: number;
    ram: number;
  };

  let cpu = $state(0);
  let ram = $state(0);
  let cpuHistory: number[] = $state(Array(20).fill(0));
  let ramHistory: number[] = $state(Array(20).fill(0));
  let intervalId: any;
  let mounted = $state(false);
  let statsAvailable = $state(false);

  async function loadStats() {
    try {
      const stats: Stats = await invoke("get_system_stats");
      if (!Number.isFinite(stats.cpu) || !Number.isFinite(stats.ram)) {
        statsAvailable = false;
        return;
      }
      cpu = stats.cpu;
      ram = stats.ram;
      statsAvailable = true;

      cpuHistory = [...cpuHistory.slice(1), cpu];
      ramHistory = [...ramHistory.slice(1), ram];
    } catch (e) {
      statsAvailable = false;
      console.error("Failed to load mini stats", e);
    }
  }

  onMount(() => {
    mounted = true;
    loadStats();
    intervalId = setInterval(loadStats, 2000);
  });

  onDestroy(() => {
    if (intervalId) clearInterval(intervalId);
  });

  function generatePath(data: number[], width: number, height: number) {
    const min = 0;
    const max = 100;
    const step = width / (data.length - 1);

    return data
      .map((val, i) => {
        const x = i * step;
        const y = height - ((val - min) / (max - min)) * height;
        return `${i === 0 ? "M" : "L"} ${x},${y}`;
      })
      .join(" ");
  }

  function getEndPos(data: number[], width: number, height: number) {
    const min = 0;
    const max = 100;
    const val = data[data.length - 1];
    return {
      x: width,
      y: height - ((val - min) / (max - min)) * height,
    };
  }

  let cpuEnd = $derived(getEndPos(cpuHistory, 45, 14));
  let ramEnd = $derived(getEndPos(ramHistory, 45, 14));
</script>

{#if mounted}
  <div class="mini-monitor" transition:fade>
    <div class="stat-group cpu-group">
      <div class="stat-info">
        <span class="label">CPU</span>
        <span class="value">{statsAvailable ? `${cpu.toFixed(0).padStart(2, "0")}%` : "--"}</span>
      </div>
      <div class="chart-wrapper">
        <svg width="45" height="14" viewBox="0 0 45 14" class="sparkline">
          <defs>
            <linearGradient id="cpu-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="currentColor" stop-opacity="0.15" />
              <stop offset="100%" stop-color="currentColor" />
            </linearGradient>
          </defs>
          <path
            d={generatePath(cpuHistory, 45, 14)}
            fill="none"
            stroke="url(#cpu-grad)"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="chart-path"
          />
          <circle cx={cpuEnd.x} cy={cpuEnd.y} r="2" fill="currentColor" />
        </svg>
      </div>
    </div>

    <div class="divider"></div>

    <div class="stat-group ram-group">
      <div class="stat-info">
        <span class="label">RAM</span>
        <span class="value">{statsAvailable ? `${ram.toFixed(0).padStart(2, "0")}%` : "--"}</span>
      </div>
      <div class="chart-wrapper">
        <svg width="45" height="14" viewBox="0 0 45 14" class="sparkline">
          <defs>
            <linearGradient id="ram-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stop-color="currentColor" stop-opacity="0.15" />
              <stop offset="100%" stop-color="currentColor" />
            </linearGradient>
          </defs>
          <path
            d={generatePath(ramHistory, 45, 14)}
            fill="none"
            stroke="url(#ram-grad)"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
            class="chart-path"
          />
          <circle cx={ramEnd.x} cy={ramEnd.y} r="2" fill="currentColor" />
        </svg>
      </div>
    </div>
  </div>
{/if}

<style>
  .mini-monitor {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 32px;
    padding: 0 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    -webkit-app-region: no-drag;
  }

  .stat-group {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: default;
  }

  .cpu-group {
    color: var(--accent);
  }

  .ram-group {
    color: var(--success);
  }

  .stat-info {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 30px;
  }

  .label {
    margin-bottom: -1px;
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--text-muted);
  }

  .value {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--text-primary);
  }

  .divider {
    width: 1px;
    height: 18px;
    background: var(--border);
  }

  .chart-wrapper {
    display: flex;
    align-items: center;
    padding-top: 2px;
  }

  .chart-path {
    transition: d 0.3s linear;
  }

  circle {
    transition: cy 0.3s linear;
  }
</style>
