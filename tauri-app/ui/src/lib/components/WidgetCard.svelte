<script lang="ts">
   import { pinnedWidgets } from "../stores/pinnedWidgets";
  export let title = "";
  export let color = "#8b5cf6";
  function togglePin() {
    pinnedWidgets.update((items) => {
      const exists = items.some(
        (item) => item.title === title
      );
      if (exists) {
        return items.filter(
          (item) => item.title !== title
        );
      }
      return [
        ...items,
        {
          title,
          color
        }
      ];
    });
  }
  $: pinned =
    $pinnedWidgets.some(
      (item) => item.title === title
    );
</script>
<div class="widget-card">
  <div class="widget-header">
    <h3>{title}</h3>
     <div class="actions">
    <button
      class="pin-btn"
      class:pinned={pinned}
      on:click={togglePin}
    >
      📌
    </button>
  </div>
</div>
<div class="widget-content">
    <slot />
</div>
</div>
<style>
  .widget-card {
    background: rgba(20, 20, 35, 0.9);
    border: 1px solid rgba(120, 120, 255, 0.15);
    border-radius: 18px;
    padding: 18px;
    color: white;
    backdrop-filter: blur(10px);
    transition: 0.2s ease;
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
  .widget-card:hover {
    transform: translateY(-2px);
    border-color: rgba(140, 140, 255, 0.35);
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
    color: #a78bfa;
  }
  .pin-btn {
    width: 34px;
    height: 34px;
    border: none;
    border-radius: 10px;
  }
</style>