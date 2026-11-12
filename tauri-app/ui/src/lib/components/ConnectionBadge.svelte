<script lang="ts">
  import { onDestroy } from "svelte";
  import { session } from "../stores/session";
  import { offNotification, onNotification } from "../api/daemon";
  import { _ } from "svelte-i18n";
  import { Wifi, Loader2 } from "@lucide/svelte";

  let wasConnected = $state(false);
  /** While the daemon is still starting it already answers: what it is doing ("opening Heliox's records"). */
  let starting = $state("");

  type Startup = { state?: string; stage?: string };
  function applyStartup(status: Startup) {
    starting = status.state && status.state !== "ready" && status.state !== "failed" ? status.stage || "starting" : "";
  }

  const handler = (method: string, params: unknown) => {
    if (method === "startup_status") applyStartup((params ?? {}) as Startup);
  };
  onNotification(handler);
  onDestroy(() => offNotification(handler));

  $effect(() => {
    if ($session.daemonConnected) {
      wasConnected = true; // the daemon sends the start-up stage right after authentication while it starts
    } else {
      starting = "";
    }
  });

  let stateClass = $derived(
    $session.daemonConnected ? (starting ? "starting" : "online") : wasConnected ? "reconnecting" : "connecting",
  );
</script>

<div class="connection-hub {stateClass}" data-testid="connection-badge">
  {#if $session.daemonConnected && starting}
    <Loader2 size={14} class="spin" />
    <span class="label">Starting: {starting}</span>
  {:else if $session.daemonConnected}
    <Wifi size={14} />
    <span class="label">{$_("app.online", { default: "Online" })}</span>
  {:else if wasConnected}
    <Loader2 size={14} class="spin" />
    <span class="label">Reconnecting...</span>
  {:else}
    <Loader2 size={14} class="spin" />
    <span class="label">{$_("app.connecting", { default: "Connecting..." })}</span>
  {/if}
</div>

<style>
  .connection-hub {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    cursor: default;
    user-select: none;
    -webkit-app-region: no-drag;
  }

  /* Online State - Sleek, success-themed glass pill */
  .connection-hub.online {
    background: color-mix(in srgb, var(--success) 15%, transparent); /* Emerald 500 with low opacity */
    color: var(--success); /* Emerald 400 */
    border-color: color-mix(in srgb, var(--success) 30%, transparent);
  }

  .connection-hub.online:hover {
    background: color-mix(in srgb, var(--success) 25%, transparent);
  }

  /* Connecting State - Shimmering, fluid glowing gradient */
  .connection-hub.connecting,
  .connection-hub.starting,
  .connection-hub.reconnecting {
    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--accent) 15%, transparent),
      color-mix(in srgb, var(--accent) 15%, transparent),
      color-mix(in srgb, var(--accent) 15%, transparent)
    );
    background-size: 200% 100%;
    color: var(--accent); /* Blue 400 */
    border-color: color-mix(in srgb, var(--accent) 30%, transparent);

    animation:
      gradient-pan 2s linear infinite,
      pulse-glow 2s ease-in-out infinite;
  }

  .connection-hub.reconnecting {
    color: var(--warning); /* Amber 400 */
    border-color: color-mix(in srgb, var(--warning) 30%, transparent);

    background: linear-gradient(
      90deg,
      color-mix(in srgb, var(--warning) 15%, transparent),
      color-mix(in srgb, var(--warning) 15%, transparent),
      color-mix(in srgb, var(--warning) 15%, transparent)
    );
    background-size: 200% 100%;
  }

  @keyframes gradient-pan {
    0% {
      background-position: 100% 0;
    }
    100% {
      background-position: -100% 0;
    }
  }

  @keyframes pulse-glow {
    0%,
    100% {
      opacity: 0.8;
    }
    50% {
      opacity: 1;
    }
  }

  :global(.spin) {
    animation: spin 1.2s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .label {
    text-transform: uppercase;
  }
</style>
