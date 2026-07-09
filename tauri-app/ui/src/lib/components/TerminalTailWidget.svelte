<script lang="ts">
  import WidgetCard from "./WidgetCard.svelte";
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  let logs: string[] = [];
  async function loadLogs() {
    if (typeof localStorage !== "undefined" && localStorage.getItem("heliox_terminal_logs") === "cleared") {
      logs = ["[System] All terminal logs cleared cleanly."];
      return;
    }
    try {
      const res = await invoke("get_terminal_logs");
      if (Array.isArray(res)) logs = res;
    } catch (error) {
      console.error("Failed to fetch terminal logs:", error);
    }
  }
  onMount(() => {
    loadLogs();
    const interval =
    setInterval(
      loadLogs,
      2000
    );
  return () =>
    clearInterval(interval);
  });
</script>
<div class="terminal-card">
  <WidgetCard title="Terminal Tail" color="#8b5cf6">
    <div class="terminal-content">
      {#each logs as log}
        <p
          class:info={log.includes("[INFO]")}
          class:success={log.includes("[SUCCESS]")}
          class:warn={log.includes("[WARN]")}
        >
          {log}
        </p>
      {/each}
    </div>
  </WidgetCard>
</div>
<style>
  .terminal-card {
    background: #0b1020;
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 22px;
  padding: 20px;
  min-height: 450px;
  color: white;
  height: 100%;
  }
  .terminal-content {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-family: monospace;
    overflow-y: auto;
    max-height: 700px;
  }
  p {
    margin: 0;
    font-size: 14px;
    color: #d1d5db;
  }
  .info {
    color: #22d3ee;
  }
  .success {
    color: #00ff88;
  }
  .warn {
    color: #facc15;
  }
</style>