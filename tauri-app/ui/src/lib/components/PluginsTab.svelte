<script lang="ts">
  import { call } from "../api/daemon";

  interface Plugin {
    name: string;
    description: string;
    version: string;
    author: string;
    url: string;
    tools: { name: string; description: string }[];
    installed: boolean;
  }

  let plugins: Plugin[] = $state([]);
  let loading = $state(true);
  let error = $state("");
  let installing = $state<string | null>(null);
  let uninstalling = $state<string | null>(null);

  async function loadPlugins() {
    loading = true;
    error = "";
    try {
      const result = await call<{ plugins: Plugin[]; error?: string }>("plugin_market_list");
      if (result.error) {
        error = result.error;
      } else {
        plugins = result.plugins || [];
      }
    } catch (e) {
      error = "Failed to load plugins";
      console.error(e);
    } finally {
      loading = false;
    }
  }

  async function installPlugin(plugin: Plugin) {
    installing = plugin.name;
    try {
      await call("plugin_install", { plugin_name: plugin.name });
      plugin.installed = true;
    } catch (e) {
      console.error("Install failed:", e);
    } finally {
      installing = null;
    }
  }

  async function uninstallPlugin(plugin: Plugin) {
    uninstalling = plugin.name;
    try {
      await call("plugin_uninstall", { plugin_name: plugin.name });
      plugin.installed = false;
    } catch (e) {
      console.error("Uninstall failed:", e);
    } finally {
      uninstalling = null;
    }
  }

  loadPlugins();
</script>

<div class="plugins-tab">
  <div class="header">
    <h2>Plugin Marketplace</h2>
    <button class="refresh-btn" onclick={loadPlugins} disabled={loading}>
      {loading ? "Loading..." : "Refresh"}
    </button>
  </div>

  {#if error}
    <div class="error-message">{error}</div>
  {/if}

  {#if loading && plugins.length === 0}
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Loading plugins...</p>
    </div>
  {:else if plugins.length === 0}
    <div class="empty-state">
      <p>No plugins available in the marketplace.</p>
    </div>
  {:else}
    <div class="plugin-grid">
      {#each plugins as plugin}
        <div class="plugin-card" class:installed={plugin.installed}>
          <div class="plugin-header">
            <div class="plugin-info">
              <h3>{plugin.name}</h3>
              <span class="version">v{plugin.version}</span>
              <span class="author">by {plugin.author}</span>
            </div>
            <span class="status-badge" class:installed={plugin.installed}>
              {plugin.installed ? "Installed" : "Available"}
            </span>
          </div>

          <p class="description">{plugin.description}</p>

          <div class="tools-section">
            <span class="tools-label">{plugin.tools.length} tools</span>
            <div class="tools-list">
              {#each plugin.tools.slice(0, 3) as tool}
                <span class="tool-tag">{tool.name}</span>
              {/each}
              {#if plugin.tools.length > 3}
                <span class="tool-tag more">+{plugin.tools.length - 3} more</span>
              {/if}
            </div>
          </div>

          <div class="plugin-actions">
            {#if plugin.installed}
              <button
                class="btn-uninstall"
                onclick={() => uninstallPlugin(plugin)}
                disabled={uninstalling === plugin.name}
              >
                {uninstalling === plugin.name ? "Removing..." : "Uninstall"}
              </button>
            {:else}
              <button
                class="btn-install"
                onclick={() => installPlugin(plugin)}
                disabled={installing === plugin.name}
              >
                {installing === plugin.name ? "Installing..." : "Install"}
              </button>
            {/if}
            <a href={plugin.url} target="_blank" rel="noopener" class="btn-link">
              View on GitHub
            </a>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .plugins-tab {
    height: 100%;
    overflow-y: auto;
    padding: 16px;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
  }

  .refresh-btn {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .error-message {
    padding: 10px 14px;
    background: var(--danger-bg);
    color: var(--danger);
    border: 1px solid var(--danger);
    border-radius: var(--radius-md);
    margin-bottom: 16px;
    font-size: 13px;
  }

  .loading-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: var(--text-muted);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .plugin-grid {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .plugin-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    transition: all 0.2s;
  }

  .plugin-card.installed {
    border-color: var(--success);
    background: rgba(74, 222, 128, 0.05);
  }

  .plugin-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .plugin-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    text-transform: capitalize;
  }

  .version {
    font-size: 11px;
    color: var(--text-muted);
  }

  .author {
    font-size: 11px;
    color: var(--text-muted);
  }

  .status-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    background: var(--accent-muted);
    color: var(--accent);
  }

  .status-badge.installed {
    background: rgba(74, 222, 128, 0.15);
    color: var(--success);
  }

  .description {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0 0 10px 0;
    line-height: 1.4;
  }

  .tools-section {
    margin-bottom: 12px;
  }

  .tools-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: block;
    margin-bottom: 6px;
  }

  .tools-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .tool-tag {
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 2px 8px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border-radius: 10px;
  }

  .tool-tag.more {
    color: var(--text-muted);
  }

  .plugin-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .btn-install, .btn-uninstall {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-install {
    background: var(--accent);
    color: white;
  }

  .btn-install:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .btn-uninstall {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }

  .btn-uninstall:hover:not(:disabled) {
    background: var(--danger-bg);
    color: var(--danger);
    border-color: var(--danger);
  }

  .btn-install:disabled, .btn-uninstall:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .btn-link {
    font-size: 11px;
    color: var(--text-muted);
    text-decoration: none;
    padding: 4px 8px;
  }

  .btn-link:hover {
    color: var(--accent);
  }
</style>
