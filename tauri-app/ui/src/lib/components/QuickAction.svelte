<script lang="ts">
  import { Camera, Mic, Pin, RotateCw, ShieldCheck, SquareTerminal, Trash2 } from "@lucide/svelte";
  import { invoke } from "../api/invoke";
  import { call, requireOkResult, type DaemonStatusResult } from "../api/daemon";
  import { onMount } from "svelte";
  import { pinnedWidgets } from "../stores/pinnedWidgets";
  let uptime = "Loading...";
  let logCount = 0;
  let clearMessage = "";
  let previousLogCount = 0;
  function togglePin() {
    pinnedWidgets.update((items) => {
      const exists = items.some((item) => item.title === "Quick Actions");
      if (exists) {
        return items.filter((item) => item.title !== "Quick Actions");
      }
      return [
        ...items,
        {
          title: "Quick Actions",
          color: "#ff4d8d",
        },
      ];
    });
  }
  $: pinned = $pinnedWidgets.some((item) => item.title === "Quick Actions");
  async function loadUptime() {
    try {
      const res = await invoke("get_uptime");
      if (res != null) uptime = String(res);
    } catch (err) {
      console.error(err);
    }
  }
  async function loadLogCount() {
    try {
      const res = await invoke("get_log_count");
      if (typeof res === "number") logCount = res;
    } catch (err) {
      console.error(err);
    }
  }
  onMount(() => {
    loadUptime();
    loadLogCount();
    const interval = setInterval(loadUptime, 3000);
    return () => clearInterval(interval);
  });
  async function handleAction(title: string) {
    try {
      switch (title) {
        case "Open Terminal":
          await invoke("open_terminal");
          clearMessage = "System terminal window opened";
          setTimeout(() => {
            clearMessage = "";
          }, 3000);
          break;
        case "Clear Logs":
          previousLogCount = logCount;
          await invoke("clear_logs");
          if (typeof localStorage !== "undefined") {
            localStorage.removeItem("heliox_activity_logs");
            localStorage.removeItem("heliox_terminal_logs");
          }
          await loadLogCount();
          clearMessage = `All ${previousLogCount || "active"} Logs Cleared Cleanly`;
          setTimeout(() => {
            clearMessage = "";
          }, 3000);
          break;
        case "Restart Agents":
          const restartRes = await invoke("restart_agents");
          clearMessage = `Agent status: ${restartRes}`;
          break;
        case "System Scan":
          const stats: any = await invoke("system_scan");
          console.log(stats);
          const formatted =
            typeof stats === "object"
              ? Object.entries(stats)
                  .map(([k, v]) => `• ${k.replace(/_/g, " ")}: ${v}`)
                  .join("\n")
              : String(stats);
          clearMessage = `System health scan\n\n${formatted}`;
          break;
        case "Take Screenshot":
          // The daemon takes it, as it does for "take a screenshot" (the Screenshot portal on Wayland).
          const shot = requireOkResult(
            await call<DaemonStatusResult & { path?: string }>("take_screenshot"),
            "The screenshot could not be taken.",
          );
          clearMessage = `Screenshot saved to ${shot.path}`;
          break;
        case "Voice Command":
          let voiceRequestHandled = false;
          window.dispatchEvent(
            new CustomEvent("heliox:voice-command-request", {
              detail: {
                respond(message: string) {
                  voiceRequestHandled = true;
                  clearMessage = message;
                },
              },
            }),
          );
          setTimeout(() => {
            clearMessage = voiceRequestHandled ? "" : "Voice controls are unavailable on this screen";
          }, 4000);
          break;
      }
    } catch (err) {
      console.error(err);
      clearMessage = `That action failed: ${err}`;
    }
  }
  const actions = [
    {
      icon: SquareTerminal,
      title: "Open Terminal",
    },
    {
      icon: Trash2,
      title: "Clear Logs",
    },
    {
      icon: RotateCw,
      title: "Restart Agents",
    },
    {
      icon: ShieldCheck,
      title: "System Scan",
    },
    {
      icon: Mic,
      title: "Voice Command",
    },
    {
      icon: Camera,
      title: "Take Screenshot",
    },
  ];
</script>

<div class="wrapper">
  <!-- QUICK ACTIONS -->
  <div class="card">
    <!-- HEADER -->
    <div class="header">
      <h3>Quick actions</h3>
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
    <!-- GRID -->
    <div class="grid">
      {#each actions as action}
        <button class="action-btn" on:click={() => handleAction(action.title)}>
          <div class="icon" aria-hidden="true">
            <svelte:component this={action.icon} size={20} />
          </div>
          <span>
            {action.title}
          </span>
        </button>
      {/each}
    </div>
  </div>
  {#if clearMessage}
    <div class="log-status" role="status">
      {clearMessage}
    </div>
  {/if}
  <!-- SYSTEM UPTIME -->
  <div class="uptime-card">
    <div>
      <p class="uptime-title">SYSTEM UPTIME</p>
      <h2>
        {uptime}
      </h2>
    </div>
    <!-- ECG LINE -->
    <div class="ecg">
      <svg viewBox="0 0 300 80" preserveAspectRatio="none">
        <path
          d="
          M0 40
          L20 40
          L35 35
          L50 45
          L65 40
          L85 40
          L100 15
          L115 65
          L130 35
          L145 40
          L165 40
          L180 30
          L195 50
          L210 40
          L230 40
          L245 20
          L260 60
          L275 40
          L300 40
          "
        />
      </svg>
    </div>
  </div>
</div>

<style>
  .wrapper {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  /* QUICK ACTIONS CARD */
  .log-status {
    white-space: pre-line;
    margin-top: 18px;
    padding: 10px 14px;
    border-radius: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 13px;
  }
  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    color: var(--text-primary);
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  h3 {
    margin: 0;
    color: var(--text-primary);
    font-size: 14px;
    font-weight: 700;
    font-family: var(--font-display);
  }
  .pin-btn {
    width: 42px;
    height: 42px;
    border: none;
    border-radius: 14px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    cursor: pointer;
    transition: 0.2s ease;
  }
  .pin-btn.pinned {
    background: color-mix(in srgb, var(--danger) 15%, transparent);
    color: var(--danger);
    border: 1px solid var(--danger);
  }
  .actions {
    display: flex;
    gap: 8px;
  }
  .actions button {
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 10px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    cursor: pointer;
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
  }
  .action-btn {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    cursor: pointer;
    transition: 0.2s ease;
    color: var(--text-primary);
  }
  .action-btn:hover {
    border-color: color-mix(in srgb, var(--accent) 50%, transparent);
    transform: translateY(-2px);
  }
  .icon {
    font-size: 26px;
    color: var(--accent);
  }
  .action-btn span {
    font-size: 13px;
    text-align: center;
    color: var(--text-secondary);
  }
  /* UPTIME CARD */
  .uptime-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
  }
  .uptime-title {
    color: var(--text-secondary);
    font-size: 12px;
    margin-bottom: 8px;
  }
  .uptime-card h2 {
    margin: 0;
    color: var(--text-primary);
    font-size: 28px;
  }
  .ecg {
    width: 180px;
    height: 60px;
  }
  svg {
    width: 100%;
    height: 100%;
  }
  path {
    fill: none;
    stroke: var(--accent);
    stroke-width: 3;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
