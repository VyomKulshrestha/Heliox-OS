<script lang="ts">
  import { session } from "../stores/session";
  import { _ } from 'svelte-i18n';
  import { Wifi, Loader2 } from 'lucide-svelte';

  let wasConnected = $state(false);

  $effect(() => {
    if ($session.daemonConnected) {
      wasConnected = true;
    }
  });

  let stateClass = $derived(
    $session.daemonConnected
      ? "online"
      : wasConnected
      ? "reconnecting"
      : "connecting"
  );
</script>

<div class="connection-hub {stateClass}">
  {#if $session.daemonConnected}
    <Wifi size={14} />
    <span class="label">{$_('app.online', { default: 'Online' })}</span>
  {:else if wasConnected}
    <Loader2 size={14} class="spin" />
    <span class="label">Reconnecting...</span>
  {:else}
    <Loader2 size={14} class="spin" />
    <span class="label">{$_('app.connecting', { default: 'Connecting...' })}</span>
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
    border: 1px solid rgba(255, 255, 255, 0.05);
    cursor: default;
    user-select: none;
    -webkit-app-region: no-drag;
  }

  /* Online State - Sleek, success-themed glass pill */
  .connection-hub.online {
    background: rgba(16, 185, 129, 0.15); /* Emerald 500 with low opacity */
    color: #34d399; /* Emerald 400 */
    border-color: rgba(52, 211, 153, 0.3);
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.2), inset 0 0 8px rgba(52, 211, 153, 0.1);
  }

  .connection-hub.online:hover {
    background: rgba(16, 185, 129, 0.25);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.3), inset 0 0 10px rgba(52, 211, 153, 0.2);
  }

  /* Connecting State - Shimmering, fluid glowing gradient */
  .connection-hub.connecting, .connection-hub.reconnecting {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15));
    background-size: 200% 100%;
    color: #60a5fa; /* Blue 400 */
    border-color: rgba(96, 165, 250, 0.3);
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.2);
    animation: gradient-pan 2s linear infinite, pulse-glow 2s ease-in-out infinite;
  }

  .connection-hub.reconnecting {
    color: #fbbf24; /* Amber 400 */
    border-color: rgba(251, 191, 36, 0.3);
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.15), rgba(245, 158, 11, 0.15));
    background-size: 200% 100%;
  }

  @keyframes gradient-pan {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
  }

  @keyframes pulse-glow {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
  }

  :global(.spin) {
    animation: spin 1.2s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .label {
    text-transform: uppercase;
  }
</style>
