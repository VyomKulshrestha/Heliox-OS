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

  async function loadStats() {
    try {
      const stats: Stats = await invoke("get_system_stats");
      cpu = stats.cpu;
      ram = stats.ram;
      
      cpuHistory = [...cpuHistory.slice(1), cpu];
      ramHistory = [...ramHistory.slice(1), ram];
    } catch (e) {
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
    
    return data.map((val, i) => {
      const x = i * step;
      const y = height - ((val - min) / (max - min)) * height;
      return `${i === 0 ? 'M' : 'L'} ${x},${y}`;
    }).join(' ');
  }

  function getEndPos(data: number[], width: number, height: number) {
    const min = 0;
    const max = 100;
    const val = data[data.length - 1];
    return {
      x: width,
      y: height - ((val - min) / (max - min)) * height
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
      <span class="value">{cpu.toFixed(0).padStart(2, '0')}%</span>
    </div>
    <div class="chart-wrapper">
      <svg width="45" height="14" viewBox="0 0 45 14" class="sparkline">
        <defs>
          <linearGradient id="cpu-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="rgba(0, 255, 136, 0.1)" />
            <stop offset="100%" stop-color="rgba(0, 255, 136, 1)" />
          </linearGradient>
          <filter id="glow-cpu" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        <path 
          d={generatePath(cpuHistory, 45, 14)} 
          fill="none" 
          stroke="url(#cpu-grad)" 
          stroke-width="1.5" 
          stroke-linecap="round"
          stroke-linejoin="round"
          filter="url(#glow-cpu)"
          class="chart-path"
        />
        <circle cx={cpuEnd.x} cy={cpuEnd.y} r="2" fill="#00ff88" filter="url(#glow-cpu)" />
      </svg>
    </div>
  </div>

  <div class="divider"></div>

  <div class="stat-group ram-group">
    <div class="stat-info">
      <span class="label">RAM</span>
      <span class="value">{ram.toFixed(0).padStart(2, '0')}%</span>
    </div>
    <div class="chart-wrapper">
      <svg width="45" height="14" viewBox="0 0 45 14" class="sparkline">
        <defs>
          <linearGradient id="ram-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="rgba(168, 85, 247, 0.1)" />
            <stop offset="100%" stop-color="rgba(168, 85, 247, 1)" />
          </linearGradient>
          <filter id="glow-ram" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="1.5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        <path 
          d={generatePath(ramHistory, 45, 14)} 
          fill="none" 
          stroke="url(#ram-grad)" 
          stroke-width="1.5" 
          stroke-linecap="round"
          stroke-linejoin="round"
          filter="url(#glow-ram)"
          class="chart-path"
        />
        <circle cx={ramEnd.x} cy={ramEnd.y} r="2" fill="#a855f7" filter="url(#glow-ram)" />
      </svg>
    </div>
  </div>
</div>
{/if}

<style>
  .mini-monitor {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 14px;
    background: rgba(15, 20, 35, 0.5);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    margin-right: 8px;
    -webkit-app-region: no-drag;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
  }

  .mini-monitor:hover {
    background: rgba(20, 25, 45, 0.7);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
  }

  .stat-group {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: default;
  }

  .stat-info {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-width: 32px;
  }

  .label {
    font-size: 8px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted, #8b949e);
    margin-bottom: -2px;
  }

  .value {
    font-size: 11px;
    font-weight: 700;
    font-family: 'SF Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    letter-spacing: -0.5px;
  }

  .cpu-group .value {
    color: #00ff88;
    text-shadow: 0 0 8px rgba(0, 255, 136, 0.4);
  }

  .ram-group .value {
    color: #a855f7;
    text-shadow: 0 0 8px rgba(168, 85, 247, 0.4);
  }

  .divider {
    width: 1px;
    height: 18px;
    background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0.1), rgba(255,255,255,0));
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
