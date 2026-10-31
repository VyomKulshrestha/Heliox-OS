<script lang="ts">
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  let showWidgetMenu = false;
  const availableWidgets = [
    {
      title: "System Monitor",
      color: "#00ffae",
    },
    {
      title: "Terminal Tail",
      color: "#8b5cf6",
    },
    {
      title: "RSS Feed",
      color: "#ff7b00",
    },
    {
      title: "System Temperatures",
      color: "#ff4d8d",
    },
    {
      title: "Agent Activity",
      color: "#7c3aed",
    },
    {
      title: "Quick Actions",
      color: "#3b82f6",
    },
  ];
  function addWidget(widget: { title: string; color: string }) {
    pinnedWidgets.update((items) => {
      const exists = items.some((item) => item.title === widget.title);
      if (exists) return items;
      return [...items, widget];
    });
    showWidgetMenu = false;
  }
  function removeWidget(title: string) {
    pinnedWidgets.update((items) => {
      return items.filter((item) => item.title !== title);
    });
  }
</script>

<div class="pinned-widgets">
  <div class="pinned-header">
    <h2>Pinned widgets</h2>
  </div>
  <div class="pinned-row">
    {#each $pinnedWidgets as widget}
      <div
        class="pinned-card"
        style={`
					background:
					linear-gradient(
						135deg,
						${widget.color}22,
						#0b1020
					);
					border-color:
					${widget.color}55;
				`}
      >
        <button class="remove-btn" on:click={() => removeWidget(widget.title)}> ✕ </button>
        <div class="card-content">
          <div class="dot" style={`background:${widget.color}`}></div>
          <span>
            {widget.title}
          </span>
        </div>
      </div>
    {/each}
    <button class="add-card" on:click={() => (showWidgetMenu = !showWidgetMenu)}>
      <span class="plus">＋</span>
      <span>Add Widget</span>
    </button>
  </div>
</div>
{#if showWidgetMenu}
  <div class="widget-menu">
    {#each availableWidgets as widget}
      <button class="widget-option" on:click={() => addWidget(widget)}>
        {widget.title}
      </button>
    {/each}
  </div>
{/if}

<style>
  .widget-menu {
    position: absolute;
    top: 140px;
    right: 40px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 14px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    z-index: 50;
  }
  .widget-option {
    background: transparent;
    border: none;
    color: var(--text-primary);
    padding: 10px 14px;
    border-radius: 10px;
    text-align: left;
    cursor: pointer;
  }
  .widget-option:hover {
    background: var(--bg-hover);
  }
  .pinned-widgets {
    width: 100%;
    position: relative;
    padding: 20px;
    border-radius: 22px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    margin-top: 20px;
  }
  .pinned-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
  }
  .pinned-header h2 {
    font-size: 15px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px;
  }
  .pinned-row {
    display: flex;
    align-items: center;
    gap: 16px;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .pinned-row::-webkit-scrollbar {
    display: none;
  }
  .pinned-card {
    position: relative;
    min-width: 210px;
    height: 92px;
    border-radius: 18px;
    border: 1px solid;
    padding: 18px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    transition: 0.25s ease;
    cursor: grab;
  }
  .pinned-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 20px #00000055;
  }
  .card-content {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .dot {
    width: 12px;
    height: 12px;
    border-radius: 999px;
  }
  .card-content span {
    font-size: 17px;
    font-weight: 600;
    color: var(--text-primary);
  }
  .remove-btn {
    position: absolute;
    top: 12px;
    right: 12px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 14px;
  }
  .remove-btn:hover {
    color: var(--accent);
  }
  .add-card {
    min-width: 210px;
    height: 92px;
    border-radius: 18px;
    border: 1px dashed var(--border-strong);
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--text-secondary);
    cursor: pointer;
    flex-shrink: 0;
    transition: 0.25s ease;
  }
  .add-card:hover {
    border-color: var(--accent);
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  .plus {
    font-size: 22px;
  }
</style>
