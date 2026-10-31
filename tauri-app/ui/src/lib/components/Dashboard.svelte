<script lang="ts">
  import SystemMonitorWidget from "./SystemMonitorWidget.svelte";
  import TerminalTailWidget from "./TerminalTailWidget.svelte";
  import RSSFeedWidget from "./RSSFeedWidget.svelte";
  import AgentActivity from "./AgentActivity.svelte";
  import SystemTemperature from "./SystemTemperature.svelte";
  import QuickAction from "./QuickAction.svelte";
  import PinnedWidgets from "./PinnedWidgets.svelte";
  import StatusBar from "./StatusBar.svelte";
</script>

<div class="dashboard">
  <div class="dashboard-header">
    <h1>Dashboard</h1>
    <p>Monitor active agents and system activity in real time.</p>
  </div>
  <div class="dashboard-grid">
    <div class="left-column">
      <SystemMonitorWidget />
      <AgentActivity />
    </div>
    <!-- CENTER COLUMN -->
    <div class="center-column">
      <TerminalTailWidget />
      <SystemTemperature />
    </div>
    <!-- RIGHT COLUMN -->
    <div class="right-column">
      <RSSFeedWidget />
      <QuickAction />
    </div>
  </div>
  <PinnedWidgets />
  <StatusBar />
</div>

<style>
  .dashboard {
    flex: 1;
    min-height: 0;
    padding: 20px 20px 96px;
    color: var(--text-primary);
    background: var(--bg-primary);
    overflow-x: hidden;
    overflow-y: auto;
  }

  .dashboard-header {
    margin-bottom: 18px;
  }

  .dashboard-header h1 {
    font-size: var(--text-xl);
  }

  .dashboard-header p {
    margin-top: 2px;
    font-size: var(--text-base);
    color: var(--text-secondary);
  }

  .dashboard-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr 0.9fr;
    gap: 16px;
    align-items: start;
  }

  .dashboard-grid > div {
    min-width: 0;
  }

  .left-column,
  .center-column,
  .right-column {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Two columns in a mid-size window, one in the default 700 px window. */
  @media (max-width: 1080px) {
    .dashboard-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .right-column {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 760px) {
    .dashboard {
      padding: 16px 12px 96px;
    }

    .dashboard-grid,
    .right-column {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
