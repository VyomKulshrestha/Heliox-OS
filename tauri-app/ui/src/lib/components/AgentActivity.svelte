<script lang="ts">
import WidgetCard from "./WidgetCard.svelte";
import { onMount }from "svelte";
import { invoke }from "@tauri-apps/api/core";
import { pinnedWidgets } from "../stores/pinnedWidgets";
function togglePin() {
  pinnedWidgets.update((items) => {
    const exists = items.some(
      (item) => item.title === "Agent Activity"
    );
    if (exists) {
      return items.filter(
        (item) => item.title !== "Agent Activity"
      );
    }
    return [
      ...items,
      {
        title: "Agent Activity",
        color: "#7c3aed"
      }
    ];
  });
}
$: pinned =
  $pinnedWidgets.some(
    (item) => item.title === "Agent Activity"
  );
type Agent = {
  name: string;
  status: string;
  message: string;
};
let agents: Agent[] = [];
async function loadAgents() {
  try {
    agents =
      await invoke(
        "get_agent_activity"
      );
  } catch (err) {
    console.error(
      "Agent activity failed",
      err
    );
  }
}
onMount(() => {
  loadAgents();
  const interval =
    setInterval(
      loadAgents,
      2000
    );
  return () =>
    clearInterval(interval);
});
</script>
<div class="card">
<div class="actions">
  <h3>⚡ AGENT ACTIVITY</h3>
     <button class="pin-btn" class:pinned={pinned} on:click={togglePin}> 📌 </button>
</div>
  {#each agents as agent}
    <div class="agent">
      <div class="left">
        <div class:green={agent.status === "Active"} class:yellow={agent.status === "Idle"} class="dot"></div>
          <div>
            <h4>{agent.name}</h4>
             <p>{agent.message}</p>
          </div>
       </div>
      <span class:active={agent.status === "Active"} class:idle={agent.status === "Idle"} class:warning={agent.status === "Warning"}>
           {agent.status}
      </span>
    </div>
  {/each}
</div>
<style>
.card {
  background: #0b1020;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 22px;
  padding: 20px;
  color: white;
}
h3 {
  color: #8b5cf6;
  font-size: 14px;
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
.agent {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 0;
  border-bottom: 1px solid rgba(255,255,255,0.05);
}
.left {
  display: flex;
  align-items: center;
  gap: 12px;
}
.dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
}
.green {
  background: #00ff88;
  box-shadow: 0 0 10px #00ff88;
}
.yellow {
  background: #facc15;
  box-shadow: 0 0 10px #facc15;
}
h4 {
  margin: 0;
  font-size: 14px;
}
p {
 margin: 2px 0 0;
  color: #9ca3af;
  font-size: 12px;
}
span {
  font-size: 12px;
  font-weight: 600;
}
.actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 18px;
    gap: 8px;
}
.active {
  color: #00ff88;
}
.idle {
 color: #facc15;
}
.warning {
  box-shadow:0 0 12px #ff4d4d;
  color: #f97316
}
</style>