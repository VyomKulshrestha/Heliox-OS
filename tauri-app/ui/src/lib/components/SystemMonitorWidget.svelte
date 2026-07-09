<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Filler
  } from "chart.js";
  import { Line } from "svelte-chartjs";
  ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Filler
  );
  let pinned = false;
  function togglePin() {
  pinnedWidgets.update((items) => {
    const exists = items.some(
      (item) => item.title === "System Monitor"
    );
    if (exists) {
      return items.filter(
        (item) => item.title !== "System Monitor"
      );
    }
    return [
      ...items,
      {
        title: "System Monitor",
        color: "#00ffae"
      }
    ];
  });
}
$: pinned =
  $pinnedWidgets.some(
    (item) => item.title === "System Monitor"
  );
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
  let cpuHistory = [20, 30, 25, 40, 35, 45];
  let ramHistory = [50, 55, 58, 60, 62, 65];
  let diskHistory = [30, 32, 35, 36, 38, 40];
  let networkHistory = [10, 12, 15, 18, 20, 22];
  async function loadStats() {
    try {
      const stats: Stats = await invoke("get_system_stats");
      if (!stats || typeof stats !== "object" || typeof stats?.cpu !== "number") return;
      console.log(stats);
      cpu = Number(stats.cpu.toFixed(1));
    ram = Number(stats.ram.toFixed(1));
    disk = Number(stats.disk.toFixed(1));
    networkUp = Number(
      stats.network_up.toFixed(0)
    );
    networkDown = Number(
      stats.network_down.toFixed(0)
    );
    cpuName = stats.cpu_name;
   totalRam = stats.total_ram;
   diskSize = stats.disk_size;
    cpuHistory = [
      ...cpuHistory.slice(1),
      cpu
    ];
    ramHistory = [
      ...ramHistory.slice(1),
      ram
    ];
    diskHistory = [
      ...diskHistory.slice(1),
      disk
    ];
    const networkValue =
  Math.min(
    (networkUp + networkDown) / 2000,
    100
  );
    networkHistory = [
      ...networkHistory.slice(1),
      networkValue,
      
    ];
  } catch (error) {
    console.error(
      "Failed to fetch system stats:",
      error
    );
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
        display: false
      },
      tooltip: {
        enabled: false
      }
    },
    scales: {
      x: {
        display: false
      },
      y: {
        display: false,
        min: 0,
        max: 100
      }
    },
    elements: {
      point: {
        radius: 0
      },
      line: {
        tension: 0.4
      }
    }
  };
  function createChart(data: number[], color: string) {
    return {
      labels: ["1", "2", "3", "4", "5", "6"],
      datasets: [
        {
          data,
          borderColor: color,
          borderWidth: 2,
          fill: false
        }
      ]
    };
  }
</script>
<div class="monitor-card">
  <!-- Header -->
  <div class="header">
    <h3>🟢 SYSTEM MONITOR</h3>
    <div class="actions">
      <button class="pin-btn" class:pinned={pinned} on:click={togglePin}>📌</button>
    </div>
  </div>
  <!-- Grid -->
  <div class="grid">
    <!-- CPU -->
    <div class="box">
      <span>CPU</span>
      <h2 class="green">{cpu}%</h2>
      <div class="chart">
        <Line
          data={createChart(cpuHistory, "#00ff88")}
          options={options}
        />
      </div>
    </div>
    <!-- RAM -->
    <div class="box">
      <span>RAM</span>
      <h2 class="purple">{ram}%</h2>
      <div class="chart">
        <Line
          data={createChart(ramHistory, "#a855f7")}
          options={options}
        />
      </div>
    </div>
    <!-- DISK -->
    <div class="box">
      <span>DISK</span>
      <h2 class="blue">{disk}%</h2>
      <div class="chart">
        <Line
          data={createChart(diskHistory, "#3b82f6")}
          options={options}
        />
      </div>
    </div>
    <!-- NETWORK -->
    <div class="box">
      <span>NETWORK</span>
      <div class="network">
        <p>↑ {networkUp} KB/s</p>
        <p>↓ {networkDown} KB/s</p>
      </div>
      <div class="chart">
        <Line
          data={createChart(networkHistory, "#06b6d4")}
          options={options}
        />
      </div>
    </div>
  </div>
  <!-- Footer -->
  <div class="footer">
    {cpuName} • {totalRam}GB RAM • {diskSize}GB SSD
  </div>
</div>
<style>
  .monitor-card {
    width: 100%;
    max-width: 680px;
    background: #0b1020;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 20px;
    color: white;
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
  background: rgba(255,255,255,0.05);
  color: white;
  cursor: pointer;
}
.pin-btn.pinned {
  background: rgba(255,0,0,0.15);
  color: #ff4d4d;
  border: 1px solid #ff4d4d;
  box-shadow: 0 0 12px rgba(255,0,0,0.4);
}
  h3 {
    margin: 0;
    color: #00ff88;
    font-size: 14px;
    font-weight: 700;
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
    background: rgba(255,255,255,0.05);
    color: white;
    cursor: pointer;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    width: 100%;
  }
  .box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 16px;
    min-height: 160px;
    overflow: hidden;
  }
  span {
    color: #9ca3af;
    font-size: 13px;
  }
  h2 {
    margin-top: 10px;
    margin-bottom: 12px;
    font-size: 36px;
    font-weight: 700;
  }
  .green {
    color: #00ff88;
  }
  .purple {
    color: #a855f7;
  }
  .blue {
    color: #3b82f6;
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
    color: #6b7280;
    font-size: 12px;
  }
</style>