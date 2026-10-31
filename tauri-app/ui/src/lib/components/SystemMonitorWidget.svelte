<script lang="ts">
  import { Pin } from "@lucide/svelte";
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler } from "chart.js";
  import { Line } from "svelte-chartjs";
  ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Filler);
  let pinned = false;
  function togglePin() {
    pinnedWidgets.update((items) => {
      const exists = items.some((item) => item.title === "System Monitor");
      if (exists) {
        return items.filter((item) => item.title !== "System Monitor");
      }
      return [
        ...items,
        {
          title: "System Monitor",
          color: "#00ffae",
        },
      ];
    });
  }
  $: pinned = $pinnedWidgets.some((item) => item.title === "System Monitor");
  type Stats = {
    cpu: number;
    ram: number;
    disk: number;
    network_up: number;
    network_down: number;
    cpu_name: string;
    total_ram: number;
    disk_size: number;
  };
  let cpu = 0;
  let ram = 0;
  let disk = 0;
  let networkUp = 0;
  let cpuName = "";
  let totalRam = 0;
  let diskSize = 0;
  let networkDown = 0;
  let cpuHistory = Array(6).fill(0);
  let ramHistory = Array(6).fill(0);
  let diskHistory = Array(6).fill(0);
  let networkHistory = Array(6).fill(0);
  let statsAvailable = false;
  async function loadStats() {
    try {
      const stats: Stats = await invoke("get_system_stats");
      if (
        !stats ||
        typeof stats !== "object" ||
        !Number.isFinite(stats.cpu) ||
        !Number.isFinite(stats.ram) ||
        !Number.isFinite(stats.disk)
      ) {
        statsAvailable = false;
        return;
      }
      statsAvailable = true;
      cpu = Number(stats.cpu.toFixed(1));
      ram = Number(stats.ram.toFixed(1));
      disk = Number(stats.disk.toFixed(1));
      networkUp = Number.isFinite(stats.network_up) ? Number(stats.network_up.toFixed(0)) : 0;
      networkDown = Number.isFinite(stats.network_down) ? Number(stats.network_down.toFixed(0)) : 0;
      cpuName = stats.cpu_name;
      totalRam = stats.total_ram;
      diskSize = stats.disk_size;
      cpuHistory = [...cpuHistory.slice(1), cpu];
      ramHistory = [...ramHistory.slice(1), ram];
      diskHistory = [...diskHistory.slice(1), disk];
      const networkValue = Math.min((networkUp + networkDown) / 2000, 100);
      networkHistory = [...networkHistory.slice(1), networkValue];
    } catch (error) {
      statsAvailable = false;
      console.error("Failed to fetch system stats:", error);
    }
  }
  onMount(() => {
    loadStats();
    const interval = setInterval(() => {
      loadStats();
    }, 1500);
    return () => {
      clearInterval(interval);
    };
  });
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: false,
      },
    },
    scales: {
      x: {
        display: false,
      },
      y: {
        display: false,
        min: 0,
        max: 100,
      },
    },
    elements: {
      point: {
        radius: 0,
      },
      line: {
        tension: 0.4,
      },
    },
  };
  function createChart(data: number[], color: string) {
    return {
      labels: ["1", "2", "3", "4", "5", "6"],
      datasets: [
        {
          data,
          borderColor: color,
          borderWidth: 2,
          fill: false,
        },
      ],
    };
  }
</script>

<div class="monitor-card">
  <!-- Header -->
  <div class="header">
    <h3>System monitor</h3>
    <div class="actions">
      <button
        class="pin-btn"
        class:pinned
        on:click={togglePin}
        aria-pressed={pinned}
        aria-label={pinned ? "Unpin widget" : "Pin widget"}
        title={pinned ? "Unpin" : "Pin"}><Pin size={15} /></button
      >
    </div>
  </div>
  <!-- Grid -->
  <div class="grid">
    <!-- CPU -->
    <div class="box">
      <span>CPU</span>
      <h2 class="green">{statsAvailable ? `${cpu}%` : "--"}</h2>
      <div class="chart">
        <Line data={createChart(cpuHistory, "#00ff88")} {options} />
      </div>
    </div>
    <!-- RAM -->
    <div class="box">
      <span>RAM</span>
      <h2 class="purple">{statsAvailable ? `${ram}%` : "--"}</h2>
      <div class="chart">
        <Line data={createChart(ramHistory, "#a855f7")} {options} />
      </div>
    </div>
    <!-- DISK -->
    <div class="box">
      <span>DISK</span>
      <h2 class="blue">{statsAvailable ? `${disk}%` : "--"}</h2>
      <div class="chart">
        <Line data={createChart(diskHistory, "#3b82f6")} {options} />
      </div>
    </div>
    <!-- NETWORK -->
    <div class="box">
      <span>NETWORK</span>
      <div class="network">
        <p>↑ {statsAvailable && networkUp ? `${networkUp} KB/s` : "Unavailable"}</p>
        <p>↓ {statsAvailable && networkDown ? `${networkDown} KB/s` : "Unavailable"}</p>
      </div>
      <div class="chart">
        <Line data={createChart(networkHistory, "#06b6d4")} {options} />
      </div>
    </div>
  </div>
  <!-- Footer -->
  <div class="footer">
    {statsAvailable ? `${cpuName} • ${totalRam}GB RAM • ${diskSize.toFixed(0)}GB disk` : "Telemetry unavailable"}
  </div>
</div>

<style>
  .monitor-card {
    width: 100%;
    max-width: 680px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    color: var(--text-primary);
    overflow: hidden;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  .pin-btn {
    width: 34px;
    height: 34px;
    border: none;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    cursor: pointer;
  }
  .pin-btn.pinned {
    background: rgba(255, 0, 0, 0.15);
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  h3 {
    margin: 0;
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 700;
    font-family: var(--font-display);
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  button {
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    cursor: pointer;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    width: 100%;
  }
  .box {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px;
    min-height: 160px;
    overflow: hidden;
  }
  span {
    color: var(--text-secondary);
    font-size: 13px;
  }
  h2 {
    margin-top: 10px;
    margin-bottom: 12px;
    font-size: 36px;
    font-weight: 700;
    font-family: var(--font-display);
  }
  .green {
    color: var(--success);
  }
  .purple {
    color: var(--accent);
  }
  .blue {
    color: var(--accent);
  }
  .network {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 10px;
    color: #67e8f9;
    font-size: 13px;
  }
  .network p {
    margin: 0;
  }
  .chart {
    width: 100%;
    height: 60px;
    margin-top: 10px;
  }
  .footer {
    margin-top: 18px;
    color: var(--text-muted);
    font-size: 12px;
  }
</style>
