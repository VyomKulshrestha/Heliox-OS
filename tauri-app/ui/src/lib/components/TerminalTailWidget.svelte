<script lang="ts">
  import WidgetCard from "./WidgetCard.svelte";
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  let logs: string[] = [];
  let unavailable = false;
  async function loadLogs() {
    if (typeof localStorage !== "undefined" && localStorage.getItem("heliox_terminal_logs") === "cleared") {
      logs = ["[System] All terminal logs cleared cleanly."];
      return;
    }
    try {
      const res = await invoke("get_terminal_logs");
      if (Array.isArray(res)) {
        logs = res;
        unavailable = false;
      }
    } catch (error) {
      logs = [];
      unavailable = true;
      console.error("Failed to fetch terminal logs:", error);
    }
  }
  onMount(() => {
    loadLogs();
    const interval = setInterval(loadLogs, 2000);
    return () => clearInterval(interval);
  });
</script>

<div class="terminal-card">
  <WidgetCard title="Terminal Tail" color="#8b5cf6">
    <div class="terminal-content">
      {#if unavailable}
        <p class="warn">Terminal logs are unavailable in this runtime.</p>
      {:else if logs.length === 0}
        <p>No recorded terminal logs.</p>
      {/if}
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
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    min-height: 450px;
    color: var(--text-primary);
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
    color: var(--text-secondary);
  }
  .info {
    color: var(--accent);
  }
  .success {
    color: var(--success);
  }
  .warn {
    color: var(--warning);
  }
</style>
