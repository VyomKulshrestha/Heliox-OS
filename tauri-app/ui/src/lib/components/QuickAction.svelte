<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { onMount } from "svelte";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
    let uptime = "Loading...";
  let logCount = 0;
  let clearMessage = "";
  let previousLogCount = 0;
function togglePin() {
  pinnedWidgets.update((items) => {
    const exists = items.some(
      (item) =>
        item.title === "Quick Actions"
    );
    if (exists) {
      return items.filter(
        (item) =>
          item.title !== "Quick Actions"
      );
    }
    return [
      ...items,
      {
        title: "Quick Actions",
        color: "#ff4d8d"
      }
    ];
  });
}
$: pinned =
  $pinnedWidgets.some(
    (item) =>
      item.title === "Quick Actions"
  );
  async function loadUptime() {
  try {
    uptime = await invoke("get_uptime");
  } catch(err) {
    console.error(err);
  }
}
  async function loadLogCount() {
  try {
    logCount = await invoke("get_log_count");
  } catch(err) {
    console.error(err);
  }
}
  onMount(() => {
    loadUptime();
    loadLogCount();
    const interval = setInterval(loadUptime, 3000);
  return () => clearInterval(interval);
  });
  async function handleAction(title: string) {
    try {
      switch(title) {
        case "Open Terminal":
          await invoke("open_terminal");
          break;
 case "Clear Logs":
  previousLogCount = logCount;
  await invoke("clear_logs");
  await loadLogCount();
  clearMessage = `Cleared ${previousLogCount} Active Logs`;
  setTimeout(() => {
    clearMessage = "";
  }, 3000);
  break;
        case "Restart Agents":
          await invoke("restart_agents");
          alert("Agents Restarted");
          break;
        case "System Scan":
          const stats = await invoke("system_scan");
          console.log(stats);
          alert(JSON.stringify(stats, null, 2));
          break;
        case "Take Screenshot":
          await invoke("take_screenshot");
          alert("Screenshot Saved");
          break;
        case "Voice Command":
          alert("Voice command coming soon");
          break;
      }
    } catch(err) {
      console.error(err);
    }
  }
  const actions = [
    {
      icon: "⌲",
      title: "Open Terminal"
    },
    {
      icon: "🗑",
      title: "Clear Logs"
    },
    {
      icon: "⟳",
      title: "Restart Agents"
    },
    {
      icon: "🛡",
      title: "System Scan"
    },
    {
      icon: "🎙",
      title: "Voice Command"
    },
    {
      icon: "📷",
      title: "Take Screenshot"
    }
  ];
</script>
<div class="wrapper">
  <!-- QUICK ACTIONS -->
  <div class="card">
    <!-- HEADER -->
    <div class="header">
      <h3>⚡ QUICK ACTIONS</h3>
      <div class="actions">
        <button class="pin-btn" class:pinned={pinned} on:click={togglePin}>📌</button>
      </div>
    </div>
    <!-- GRID -->
    <div class="grid">
      {#each actions as action}
        <button class="action-btn" on:click={() => handleAction(action.title)}>
          <div class="icon">
            {action.icon}
          </div>
          <span>
            {action.title}
          </span>
        </button>
      {/each}
    </div>
  </div>
{#if clearMessage}
  <div class="log-status">
    {clearMessage}
  </div>
{/if}
  <!-- SYSTEM UPTIME -->
  <div class="uptime-card">
    <div>
      <p class="uptime-title">
        SYSTEM UPTIME
      </p>
      <h2>
       {uptime}
      </h2>
    </div>
    <!-- ECG LINE -->
    <div class="ecg">
      <svg
        viewBox="0 0 300 80"
        preserveAspectRatio="none"
      >
        <path
          d="
          M0 40
          L20 40
          L35 35
          L50 45
          L65 40
          L85 40
          L100 15
          L115 65
          L130 35
          L145 40
          L165 40
          L180 30
          L195 50
          L210 40
          L230 40
          L245 20
          L260 60
          L275 40
          L300 40
          "
        />
      </svg>
    </div>
  </div>
</div>
<style>
.wrapper {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
/* QUICK ACTIONS CARD */
.log-status {
  margin-top: 18px;
  padding: 10px 14px;
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.05);
  color: #d1d5db;
  font-size: 13px;
}
.card {
  background: #0b1020;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 22px;
  padding: 20px;
  color: white;
}
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
h3 {
  margin: 0;
  color: #3b82f6;
  font-size: 14px;
  font-weight: 700;
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
.actions button {
  width: 30px;
  height: 30px;
  border: none;
  border-radius: 10px;
  background: rgba(255,255,255,0.05);
  color: white;
  cursor: pointer;
}
.grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
.action-btn {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 18px;
  padding: 18px 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  cursor: pointer;
  transition: 0.2s ease;
  color: white;
}
.action-btn:hover {
  border-color: rgba(59,130,246,0.5);
  box-shadow: 0 0 18px rgba(59,130,246,0.18);
  transform: translateY(-2px);
}
.icon {
  font-size: 26px;
  color: #38bdf8;
}
.action-btn span {
  font-size: 13px;
  text-align: center;
  color: #d1d5db;
}
/* UPTIME CARD */
.uptime-card {
  background: #0b1020;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 22px;
  padding: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 20px;
}
.uptime-title {
  color: #9ca3af;
  font-size: 12px;
  margin-bottom: 8px;
}
.uptime-card h2 {
  margin: 0;
  color: white;
  font-size: 28px;
}
.ecg {
  width: 180px;
  height: 60px;
}
svg {
  width: 100%;
  height: 100%;
}
path {
  fill: none;
  stroke: #14f1c7;
  stroke-width: 3;
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: drop-shadow(
    0 0 8px rgba(20,241,199,0.7)
  );
}
</style>