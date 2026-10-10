<script lang="ts">
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type McpServer = {
    name: string;
    running: boolean;
    tools: number;
    error: string;
  };

  type McpClientStatus = {
    status: string;
    servers: McpServer[];
    tool_count: number;
    config_path?: string;
    message?: string;
  };

  let status = $state<McpClientStatus | null>(null);
  let busy = $state(false);
  let error = $state("");

  async function load(method: "mcp_client_status" | "mcp_client_refresh") {
    error = "";
    try {
      const next = await call<McpClientStatus>(method);
      if (next.status !== "ok") throw new Error(next.message || "Could not read MCP servers.");
      status = next;
    } catch (err) {
      error = err instanceof Error ? err.message : "Could not read MCP servers.";
    }
  }

  async function reload() {
    busy = true;
    try {
      await load("mcp_client_refresh");
    } finally {
      busy = false;
    }
  }

  onMount(() => {
    void load("mcp_client_status");
  });
</script>

<section class="mcp-card" aria-labelledby="mcp-servers-heading">
  <div class="section-title">
    <div>
      <span class="eyebrow">Tools</span>
      <h3 id="mcp-servers-heading">MCP servers</h3>
      <p>
        Heliox can call tools from MCP servers you list in <code>mcp_servers.json</code>. Read-only tools run straight
        away; every other tool asks first.
      </p>
    </div>
    <button class="secondary" type="button" disabled={busy} onclick={reload}>
      {busy ? "Reloading..." : "Reload servers"}
    </button>
  </div>

  {#if error}<div class="message error" role="alert">{error}</div>{/if}

  {#if status?.config_path}
    <div class="config-row">
      <span class="label">Config file</span>
      <code class="path">{status.config_path}</code>
    </div>
  {/if}

  {#if status && status.servers.length === 0}
    <p class="empty">No servers configured. Add one to the config file, then reload.</p>
  {:else if status}
    <ul class="servers">
      {#each status.servers as server (server.name)}
        <li>
          <div>
            <strong>{server.name}</strong>
            {#if server.error}<small class="server-error">{server.error}</small>{/if}
          </div>
          <span class="status-badge" class:online={server.running}>
            {server.running ? `${server.tools} tool${server.tools === 1 ? "" : "s"}` : "Not running"}
          </span>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .mcp-card {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
    overflow: hidden;
  }
  .section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 20px;
  }
  h3 {
    margin: 3px 0 5px;
    font-size: 16px;
  }
  p,
  small {
    display: block;
    margin: 0;
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.45;
  }
  .eyebrow,
  .label {
    color: var(--accent);
    font: 700 10px/1 var(--font-mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    font-size: 11px;
  }
  .error {
    color: #ff8b94;
    background: rgba(255, 70, 85, 0.08);
  }
  .config-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .path {
    overflow-wrap: anywhere;
    font-size: 11px;
  }
  .empty {
    padding: 12px 20px 16px;
    border-top: 1px solid var(--border);
  }
  .servers {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .servers li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 20px;
    border-top: 1px solid var(--border);
  }
  .servers strong {
    font-size: 12px;
  }
  .server-error {
    margin-top: 4px;
    color: #ff8b94;
  }
  .status-badge {
    flex-shrink: 0;
    padding: 5px 9px;
    border: 1px solid var(--border);
    border-radius: 999px;
    color: var(--text-secondary);
    font: 700 10px/1 var(--font-mono);
  }
  .status-badge.online {
    border-color: #18d6b0;
    color: #18d6b0;
  }
  button.secondary {
    flex-shrink: 0;
    padding: 8px 12px;
    border: 1px solid var(--border);
    border-radius: 7px;
    background: var(--bg-primary);
    color: var(--text-primary);
    cursor: pointer;
  }
  button.secondary:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>
