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
    <h1>Dashboard </h1>
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
  padding: 24px;
  padding-bottom: 120px;
  color: white;
  background: #050816;
  height: 100vh;
  overflow-y: scroll;
  overflow-x: hidden;
}
  /* WIDTH */
.dashboard::-webkit-scrollbar {
  width: 10px;
}
/* TRACK */
.dashboard::-webkit-scrollbar-track {
  background: rgba(255,255,255,0.02);
  border-radius: 999px;
}
/* HANDLE */
.dashboard::-webkit-scrollbar-thumb {
  background: linear-gradient(
    180deg,
    #00ff88,
    #06b6d4
  );
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
/* HOVER */
.dashboard::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(
    180deg,
    #34d399,
    #38bdf8
  );
}
/* FIREFOX */
.dashboard {
  scrollbar-width: thin;
  scrollbar-color: #06b6d4 rgba(255,255,255,0.02);
}
  .dashboard-grid > div {
  min-width: 0;
}
  .dashboard-header {
    margin-bottom: 24px;
 }
  .dashboard-header p {
    color: #9ca3af;
  }
  .dashboard-grid {
    display: grid;
  grid-template-columns: 1.2fr 1fr 0.9fr;
  gap: 20px;
  align-items: start;
  }
  .left-column,
  .center-column,
  .right-column {
        display: flex;
        flex-direction: column;
        gap: 20px;      
   }
  
</style>