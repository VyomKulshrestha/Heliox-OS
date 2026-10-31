<script lang="ts">
  import { Pin } from "@lucide/svelte";
  import WidgetCard from "./WidgetCard.svelte";
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  import { pollAfterSuccessfulProbe } from "../polling";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  function togglePin() {
    pinnedWidgets.update((items) => {
      const exists = items.some((item) => item.title === "Agent Activity");
      if (exists) {
        return items.filter((item) => item.title !== "Agent Activity");
      }
      return [
        ...items,
        {
          title: "Agent Activity",
          color: "#7c3aed",
        },
      ];
    });
  }
  $: pinned = $pinnedWidgets.some((item) => item.title === "Agent Activity");
  type Agent = {
    name: string;
    status: string;
    message: string;
  };
  let agents: Agent[] = [];
  let unavailable = false;
  async function loadAgents(): Promise<boolean> {
    try {
      const res = await invoke("get_agent_activity");
      if (Array.isArray(res)) {
        agents = res;
        unavailable = false;
        return true;
      }
    } catch (err) {
      agents = [];
      unavailable = true;
      console.error("Agent activity failed", err);
    }
    return false;
  }
  onMount(() => pollAfterSuccessfulProbe(loadAgents, 2000));
</script>

<div class="card">
  <div class="actions">
    <h3>Agent activity</h3>
    <button
      class="pin-btn"
      class:pinned
      on:click={togglePin}
      aria-pressed={pinned}
      aria-label={pinned ? "Unpin widget" : "Pin widget"}
      title={pinned ? "Unpin" : "Pin"}><Pin size={15} /></button
    >
  </div>
  {#if unavailable}
    <p class="unavailable">Agent activity requires a connected daemon.</p>
  {/if}
  {#each agents as agent}
    <div class="agent">
      <div class="left">
        <div class:green={agent.status === "Active"} class:yellow={agent.status === "Idle"} class="dot"></div>
        <div>
          <h4>{agent.name}</h4>
          <p>{agent.message}</p>
        </div>
      </div>
      <span
        class:active={agent.status === "Active"}
        class:idle={agent.status === "Idle"}
        class:warning={agent.status === "Warning"}
      >
        {agent.status}
      </span>
    </div>
  {/each}
</div>

<style>
  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    color: var(--text-primary);
  }
  h3 {
    color: var(--text-primary);
    font-size: 14px;
    margin-bottom: 20px;
    font-family: var(--font-display);
  }
  .unavailable {
    color: var(--warning);
    font-size: 13px;
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
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  .agent {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0;
    border-bottom: 1px solid var(--border);
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
    background: var(--success);
  }
  .yellow {
    background: var(--warning);
  }
  h4 {
    margin: 0;
    font-size: 14px;
  }
  p {
    margin: 2px 0 0;
    color: var(--text-secondary);
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
    color: var(--success);
  }
  .idle {
    color: var(--warning);
  }
  .warning {
    color: var(--warning);
  }
</style>
