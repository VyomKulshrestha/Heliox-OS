<script lang="ts">
  import { Pin } from "@lucide/svelte";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  export let title = "";
  export let color = "#8b5cf6";
  function togglePin() {
    pinnedWidgets.update((items) => {
      const exists = items.some((item) => item.title === title);
      if (exists) {
        return items.filter((item) => item.title !== title);
      }
      return [
        ...items,
        {
          title,
          color,
        },
      ];
    });
  }
  $: pinned = $pinnedWidgets.some((item) => item.title === title);
</script>

<div class="widget-card">
  <div class="widget-header">
    <h3>{title}</h3>
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
  <div class="widget-content">
    <slot />
  </div>
</div>

<style>
  .widget-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px;
    color: var(--text-primary);
    transition: border-color var(--dur) ease;
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
  .widget-card:hover {
    border-color: var(--border-strong);
  }
  .widget-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
  }
  h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
    font-family: var(--font-display);
  }
  .pin-btn {
    width: 34px;
    height: 34px;
    border: none;
    border-radius: 10px;
  }
</style>
