<script lang="ts">
  import { session } from "../stores/session";

  // Track if we were ever connected, to distinguish
  // "reconnecting" (yellow) from "never connected" (red)
  let wasConnected = false;

  $: if ($session.daemonConnected) {
    wasConnected = true;
  }

  // Dot color: green = connected, yellow = reconnecting, red = disconnected
  $: dotColor = $session.daemonConnected
    ? "dot-green"
    : wasConnected
    ? "dot-yellow"
    : "dot-red";

  // Tooltip text shown on hover
  $: dotTitle = $session.daemonConnected
    ? "Daemon connected"
    : wasConnected
    ? "Reconnecting to daemon..."
    : "Daemon disconnected";
</script>

<div class="status-indicator" title={dotTitle}>
  <div class="status-dot {dotColor}"></div>
</div>

<style>
  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: default;
  }

  .status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    transition: background-color 0.4s ease;
  }

  .dot-green {
    background-color: #22c55e;
    box-shadow: 0 0 6px #22c55e88;
  }

  .dot-yellow {
    background-color: #facc15;
    box-shadow: 0 0 6px #facc1588;
  }

  .dot-red {
    background-color: #ef4444;
    box-shadow: 0 0 6px #ef444488;
  }
</style>