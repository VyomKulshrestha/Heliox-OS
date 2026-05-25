<script lang="ts">
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
import { pinnedWidgets } from "../stores/pinnedWidgets";
  type Temps = {
    cpu: number;
    gpu: number;
    motherboard: number;
    ssd: number;
    vrm: number;
    battery: number;
    battery_percent: number;
    cpu_name: string;
    cpu_threads: number;
    fan_speed: number;
    power_draw: number;
    thermal_status: string;
  };
  let temps: Temps = {
    cpu: 0,
    gpu: 0,
    motherboard: 0,
    ssd: 0,
    vrm: 0,
    battery: 0,
    battery_percent: 0,
    cpu_name: "",
    cpu_threads: 0,
    fan_speed: 0,
    power_draw: 0,
    thermal_status: "Optimal",
  };
function togglePin() {
  pinnedWidgets.update((items) => {
    const exists = items.some(
      (item) => item.title === "System Temperatures"
    );
    if (exists) {
      return items.filter(
        (item) => item.title !== "System Temperatures"
      );
    }
    return [
      ...items,
      {
        title: "System Temperatures",
        color: "#ff4d8d"
      }
    ];
  });
}
$: pinned =
  $pinnedWidgets.some(
    (item) =>
      item.title === "System Temperatures"
  );
  function clamp(value: number) {
    return Math.max(0, Math.min(value, 100));
  }
  function getColor(temp: number) {
    if (temp >= 85) return "#ff3b30";
    if (temp >= 70) return "#ff9500";
    return "#22c55e";
  }
  async function loadTemps() {
    try {
      const data = await invoke("get_temperature_stats");
      temps = data as Temps;
    } catch (err) {
      console.error("Temperature fetch failed:", err);
    }
  }
  onMount(() => {
    loadTemps();
    const interval = setInterval(loadTemps, 1000);
    return () => clearInterval(interval);
  });
</script>
<div class="card">
  <!-- HEADER -->
  <div class="header">
    <h3>🌡 SYSTEM TEMPERATURES</h3>
    <div class="actions">
      <button class="pin-btn" class:pinned={pinned} on:click={togglePin}>📌</button>
    </div>
  </div>
  <!-- BODY -->
  <div class="body">
    <!-- LEFT -->
    <div class="left">
      {#each [
        ["CPU", temps.cpu],
        ["GPU", temps.gpu],
        ["Motherboard", temps.motherboard],
        ["SSD", temps.ssd],
        ["VRM", temps.vrm],
        ["Battery", temps.battery]
      ] as [label, value]}
        <div class="temp-item">
          <div class="top">
            <span>{label}</span>
            <p style={`color:${getColor(Number(value))}`}>
              {Number(value).toFixed(0)}°C
            </p>
          </div>
          <div class="bar">
            <div
              class="fill"
              style={`
                width:${clamp(Number(Number(value)))}%;
                background:${function getColor(temp: number) {
                              if (temp >= 85) return "#ff1f7a";
                              if (temp >= 70) return "#ff4da6";
                             return "#ff66c4";
                            }(Number(value))};
              `}>
            </div>
          </div>
        </div>
      {/each}
      <!-- EXTRA STATS -->
      <div class="extra">
        <div class="stat-box">
          <span>🔋 Battery</span>
  <strong>
      {temps.battery_percent}%
  </strong>
        </div>
        <div class="stat-box">
          <span>🧠 CPU</span>
          <strong>
                 {Math.round(temps.cpu || 0)}%
          </strong>
        </div>
      </div>
       <div class="stat-box">
  <span>⚙ Threads</span>
       <strong>
       {temps.cpu_threads}
      </strong>
    </div>
    </div>
    <!-- RIGHT -->
    <div class="right">
     <div class="circle"
     style={`
      --temp:${clamp(temps.cpu)}%;
      --circleColor:${getColor(temps.cpu)};
    `}
    >
      <div class="inner">
         <h1>{temps.cpu.toFixed(0)}°</h1>
         <span>C</span>
      </div>
   </div>
      <p class="label">CPU Temperature</p>
    </div>
 </div>
</div>
<style>
  .card {
    background: #0b1020;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 22px;
    padding: 20px;
    color: white;
    width: 100%;
    min-height: 420px;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }
  h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: #ff5c8a;
  }
  .pin-btn {
  width: 42px;
  height: 42px;
  border: none;
  border-radius: 14px;
  background: rgba(255,255,255,0.05);
  color: white;
  cursor: pointer;
  transition: 0.2s ease;
}
.pin-btn.pinned {
 background: rgba(255,0,0,0.15);
  color: #ff4d4d;
  border: 1px solid #ff4d4d;
  box-shadow: 0 0 12px rgba(255,0,0,0.4);
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
    transition: 0.3s;
  }
  button:hover {
    background: rgba(255,255,255,0.12);
  }
  .body {
    display: grid;
    grid-template-columns: 1fr 220px;
    gap: 28px;
    align-items: center;
  }
  .left {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }
  .temp-item {
    width: 100%;
  }
  .top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  .top span {
    color: #d1d5db;
    font-size: 14px;
  }
  .top p {
    margin: 0;
    font-size: 14px;
    font-weight: 700;
  }
  .bar {
    width: 100%;
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
  }
  .fill {
    height: 100%;
    border-radius: 999px;
    transition: all 0.5s ease;
    box-shadow: 0 0 14px rgba(255,255,255,0.2);
  }
  .extra {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 10px;
  }
  .stat-box {
    background: rgba(255,255,255,0.04);
    border-radius: 14px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .stat-box span {
    color: #9ca3af;
    font-size: 13px;
  }
  .stat-box strong {
    font-size: 16px;
  }
  .right {
    display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 18px;
  }
  .circle {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position:relative;
    margin-left: -35px;
    background:
      radial-gradient(
        circle at center,
        #0b1020 58%,
        transparent 59%
      ),
      conic-gradient(
        var(--circleColor) 0% var(--temp),
        rgba(255,255,255,0.08) var(--temp) 100%
      );
    box-shadow:
       0 0 45px rgba(255, 77, 166, 0.35),
    inset 0 0 20px rgba(255,255,255,0.03);
    padding: 0%;
  flex-shrink: 0;
    transition: all 0.5s ease;
  }
  .inner {
    display: flex;
    align-items: flex-start;
    color: white;
  }
  .inner h1 {
    margin: 0;
    font-size: 52px;
    line-height: 1;
  }
  .label {
    margin-top: 10px;
    color: #9ca3af;
    font-size: 14px;
    text-align: left;
  }
  @media (max-width: 900px) {
    .body {
      grid-template-columns: 1fr;
    }
    .right {
      margin-top: 20px;
    }
  }
</style>