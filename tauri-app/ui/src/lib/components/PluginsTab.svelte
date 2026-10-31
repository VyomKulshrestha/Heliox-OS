<script lang="ts">
  import { call } from "../api/daemon";

  interface PluginTool {
    name: string;
    description: string;
    inputs?: string[];
    outputs?: string[];
  }

  interface Plugin {
    name: string;
    description: string;
    version: string;
    author: string;
    url: string;
    tools: PluginTool[];
    installed: boolean;
    local_only?: boolean;
    source?: "github" | "local";
    capabilities: {
      filesystem: { read: string[]; write: string[] };
      network_domains: string[];
      processes: string[];
      credentials: string[];
      clipboard: { read: boolean; write: boolean };
      media: { camera: boolean; microphone: boolean };
      data_retention: { mode: "none" | "session" | "persistent"; max_days: number };
      destructive_actions: boolean;
    };
    capability_risks?: string[];
  }

  interface MarketplaceResult {
    plugins: Plugin[];
    source?: "github" | "bundled";
    registry_url?: string;
    submission_url?: string;
    warning?: string;
    error?: string;
  }

  let plugins: Plugin[] = $state([]);
  let loading = $state(true);
  let error = $state("");
  let notice = $state("");
  let installing = $state<string | null>(null);
  let uninstalling = $state<string | null>(null);
  let marketplaceSource = $state<"github" | "bundled" | "">("");
  let marketplaceWarning = $state("");
  let submissionUrl = $state("https://github.com/VyomKulshrestha/Heliox-OS/blob/main/docs/PLUGIN_MARKETPLACE.md");
  let showPublishingGuide = $state(false);

  // Tool execution
  let runningTool = $state<string | null>(null);
  let toolResult = $state<Record<string, any>>({});
  let toolArgs = $state<Record<string, Record<string, string>>>({});

  // Create plugin
  let showCreate = $state(false);
  let createName = $state("");
  let createDesc = $state("");
  let createAuthor = $state("");
  let createCode = $state("");
  let createToolName = $state("");
  let createToolDesc = $state("");
  let createToolInputs = $state("");
  let createTools: { name: string; description: string; inputs: string[] }[] = $state([]);
  let creating = $state(false);

  // Active view per plugin
  let expandedPlugin = $state<string | null>(null);

  async function loadPlugins() {
    loading = true;
    error = "";
    try {
      const result = await call<MarketplaceResult>("plugin_market_list");
      if (result.error) {
        error = result.error;
      } else {
        plugins = result.plugins || [];
        marketplaceSource = result.source || "";
        marketplaceWarning = result.warning || "";
        submissionUrl = result.submission_url || submissionUrl;
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
      const res = await call<{ success?: boolean; error?: string }>("plugin_install", { plugin_name: plugin.name });
      if (res.success) {
        notice = `${plugin.name} was verified and installed from the approved catalog.`;
        await loadPlugins();
      } else {
        error = res.error || "Install failed";
      }
    } catch (e) {
      console.error("Install failed:", e);
      error = "Install failed";
    } finally {
      installing = null;
    }
  }

  async function uninstallPlugin(plugin: Plugin) {
    uninstalling = plugin.name;
    try {
      const res = await call<{ success?: boolean; error?: string }>("plugin_uninstall", { plugin_name: plugin.name });
      if (res.success) {
        notice = `${plugin.name} was uninstalled.`;
        // Clear results for this plugin
        delete toolResult[plugin.name];
        await loadPlugins();
      }
    } catch (e) {
      console.error("Uninstall failed:", e);
    } finally {
      uninstalling = null;
    }
  }

  async function runTool(pluginName: string, toolName: string) {
    const key = `${pluginName}:${toolName}`;
    runningTool = key;
    try {
      const args = toolArgs[key] || {};
      const res = await call<{ result?: any; error?: string }>("plugin_run_tool", {
        tool_name: toolName,
        args,
      });
      toolResult[key] = res.result || res;
      toolResult = { ...toolResult };
    } catch (e: any) {
      toolResult[key] = { error: e.message || "Execution failed" };
      toolResult = { ...toolResult };
    } finally {
      runningTool = null;
    }
  }

  function setToolArg(key: string, argName: string, value: string) {
    if (!toolArgs[key]) toolArgs[key] = {};
    toolArgs[key][argName] = value;
    toolArgs = { ...toolArgs };
  }

  function addCreateTool() {
    if (!createToolName.trim()) return;
    createTools = [
      ...createTools,
      {
        name: createToolName.trim().replace(/\s+/g, "_").toLowerCase(),
        description: createToolDesc.trim() || createToolName.trim(),
        inputs: createToolInputs.trim() ? createToolInputs.split(",").map((s) => s.trim()) : [],
      },
    ];
    createToolName = "";
    createToolDesc = "";
    createToolInputs = "";
  }

  function removeCreateTool(idx: number) {
    createTools = createTools.filter((_, i) => i !== idx);
  }

  async function createPlugin() {
    if (!createName.trim()) return;
    if (createTools.length === 0) {
      error = "Add at least one tool before creating the plugin.";
      return;
    }
    if (!createCode.trim()) {
      error = "Write a real handle_tool implementation before creating the plugin.";
      return;
    }
    creating = true;
    try {
      const res = await call<{ success?: boolean; error?: string; local_only?: boolean }>("plugin_create", {
        name: createName,
        description: createDesc || `Custom plugin: ${createName}`,
        author: createAuthor || "User",
        tools: createTools,
        code: createCode,
      });
      if (res.success) {
        notice = `${createName} was created locally. Submit it by GitHub pull request to make it available to everyone.`;
        showCreate = false;
        createName = "";
        createDesc = "";
        createAuthor = "";
        createCode = "";
        createTools = [];
        await loadPlugins();
      } else {
        error = res.error || "Create failed";
      }
    } catch (e: any) {
      error = e.message || "Create failed";
    } finally {
      creating = false;
    }
  }

  function toggleExpand(name: string) {
    expandedPlugin = expandedPlugin === name ? null : name;
  }

  function formatResult(obj: any): string {
    if (typeof obj === "string") return obj;
    try {
      return JSON.stringify(obj, null, 2);
    } catch {
      return String(obj);
    }
  }

  function capabilityDetails(plugin: Plugin): { label: string; value: string }[] {
    const capabilities = plugin.capabilities;
    if (!capabilities) return [];
    const details: { label: string; value: string }[] = [];
    if (capabilities.network_domains.length) {
      details.push({ label: "Network", value: capabilities.network_domains.join(", ") });
    }
    if (capabilities.credentials.length) {
      details.push({ label: "Credentials", value: capabilities.credentials.join(", ") });
    }
    if (capabilities.filesystem.read.length) {
      details.push({ label: "Read paths", value: capabilities.filesystem.read.join(", ") });
    }
    if (capabilities.filesystem.write.length) {
      details.push({ label: "Write paths", value: capabilities.filesystem.write.join(", ") });
    }
    if (capabilities.processes.length) {
      details.push({ label: "Processes", value: capabilities.processes.join(", ") });
    }
    const deviceAccess = [
      capabilities.clipboard.read || capabilities.clipboard.write ? "clipboard" : "",
      capabilities.media.camera ? "camera" : "",
      capabilities.media.microphone ? "microphone" : "",
    ].filter(Boolean);
    if (deviceAccess.length) {
      details.push({ label: "Device access", value: deviceAccess.join(", ") });
    }
    details.push({
      label: "Retention",
      value:
        capabilities.data_retention.mode === "persistent"
          ? `${capabilities.data_retention.max_days} days`
          : capabilities.data_retention.mode,
    });
    return details;
  }

  loadPlugins();
</script>

<div class="plugins-tab">
  <div class="header">
    <div class="header-title">
      <h2>Plugin marketplace</h2>
      {#if marketplaceSource}
        <span class="catalog-source" class:offline={marketplaceSource === "bundled"}>
          {marketplaceSource === "github" ? "GitHub catalog" : "Bundled offline catalog"}
        </span>
      {/if}
    </div>
    <div class="header-actions">
      <button class="btn-guide" onclick={() => (showPublishingGuide = !showPublishingGuide)}>
        {showPublishingGuide ? "Hide Guide" : "How to Publish"}
      </button>
      <button class="btn-create" onclick={() => (showCreate = !showCreate)}>
        {showCreate ? "✕ Cancel" : "＋ Create Local Plugin"}
      </button>
      <button class="refresh-btn" onclick={loadPlugins} disabled={loading}>
        {loading ? "⟳ Loading..." : "⟳ Refresh"}
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-message">{error} <button class="dismiss-btn" onclick={() => (error = "")}>✕</button></div>
  {/if}

  {#if notice}
    <div class="notice-message">
      {notice}
      <button class="dismiss-btn notice-dismiss" onclick={() => (notice = "")}>✕</button>
    </div>
  {/if}

  {#if marketplaceWarning}
    <div class="catalog-warning">
      GitHub could not be reached, so Heliox is showing its verified bundled catalog. Refresh when you are online.
    </div>
  {/if}

  {#if showPublishingGuide}
    <div class="publishing-guide">
      <div>
        <h3>Publish a plugin for everyone</h3>
        <p>
          Local plugins stay on this device. Public marketplace plugins are reviewed through the Heliox GitHub
          repository.
        </p>
      </div>
      <ol>
        <li>Create and test the plugin locally.</li>
        <li>Fork <code>VyomKulshrestha/Heliox-OS</code> on GitHub.</li>
        <li>Add <code>plugins/&lt;plugin-name&gt;/manifest.json</code> and <code>plugin.py</code>.</li>
        <li>Run <code>python scripts/validate_marketplace.py --write</code>.</li>
        <li>Open a pull request. CI validates paths, hashes, manifests, and code policy.</li>
        <li>After review and merge to <code>main</code>, users see it when they press Refresh.</li>
      </ol>
      <p class="release-note">A desktop app release is not required for each approved plugin.</p>
      <a href={submissionUrl} target="_blank" rel="noopener" class="guide-link">
        Open the full submission guide on GitHub →
      </a>
    </div>
  {/if}

  <!-- Create Plugin Panel -->
  {#if showCreate}
    <div class="create-panel">
      <h3>Create Local Plugin</h3>
      <p class="local-explainer">
        This installs only on this device. Use “How to Publish” when you want to submit it to the public marketplace.
        New local plugins start with filesystem, network, process, credential, clipboard, camera, and microphone access
        denied.
      </p>
      <div class="form-grid">
        <div class="form-group">
          <label for="cp-name">Plugin Name</label>
          <input id="cp-name" type="text" bind:value={createName} placeholder="my-awesome-plugin" />
        </div>
        <div class="form-group">
          <label for="cp-desc">Description</label>
          <input id="cp-desc" type="text" bind:value={createDesc} placeholder="What does your plugin do?" />
        </div>
        <div class="form-group">
          <label for="cp-author">Author</label>
          <input id="cp-author" type="text" bind:value={createAuthor} placeholder="Your name" />
        </div>
      </div>

      <div class="tools-builder">
        <h4>Tools</h4>
        {#each createTools as tool, idx}
          <div class="tool-row">
            <span class="tool-name-tag">{tool.name}</span>
            <span class="tool-desc-text">{tool.description}</span>
            {#if tool.inputs.length > 0}
              <span class="tool-inputs-text">({tool.inputs.join(", ")})</span>
            {/if}
            <button class="tool-remove" onclick={() => removeCreateTool(idx)}>✕</button>
          </div>
        {/each}
        <div class="add-tool-row">
          <input type="text" bind:value={createToolName} placeholder="tool_name" class="tool-input-sm" />
          <input type="text" bind:value={createToolDesc} placeholder="Description" class="tool-input-md" />
          <input type="text" bind:value={createToolInputs} placeholder="arg1, arg2" class="tool-input-sm" />
          <button class="btn-add-tool" onclick={addCreateTool}>Add</button>
        </div>
      </div>

      <div class="form-group">
        <label for="cp-code">Plugin Code (Python)</label>
        <textarea
          id="cp-code"
          bind:value={createCode}
          rows="8"
          placeholder={'def handle_tool(tool_name, params):\n    return {"status": "success", "result": "Hello!"}'}
        ></textarea>
      </div>

      <button
        class="btn-submit"
        onclick={createPlugin}
        disabled={creating || !createName.trim() || createTools.length === 0 || !createCode.trim()}
      >
        {creating ? "Creating..." : "Create & Install Plugin"}
      </button>
    </div>
  {/if}

  <!-- Plugin List -->
  {#if loading && plugins.length === 0}
    <div class="loading-state">
      <div class="spinner"></div>
      <p>Loading plugins...</p>
    </div>
  {:else if plugins.length === 0}
    <div class="empty-state">
      <p>No plugins available. Click "Create Local Plugin" to build your own!</p>
    </div>
  {:else}
    <div class="plugin-grid">
      {#each plugins as plugin}
        <div class="plugin-card" class:installed={plugin.installed}>
          <div class="plugin-header">
            <div class="plugin-info">
              <h3>{plugin.name}</h3>
              {#if plugin.local_only}
                <span class="local-badge">Local only</span>
              {/if}
              <span class="version">v{plugin.version}</span>
              <span class="author">by {plugin.author}</span>
            </div>
            <span class="status-badge" class:installed={plugin.installed}>
              {plugin.installed ? "✓ Installed" : "Available"}
            </span>
          </div>

          <p class="description">{plugin.description}</p>

          <div class="capability-panel" class:destructive={plugin.capabilities?.destructive_actions}>
            <div class="capability-heading">
              <span>Declared access</span>
              <span class="isolation-badge">
                {plugin.capabilities?.destructive_actions ? "Approval required" : "Broker isolated"}
              </span>
            </div>
            <div class="capability-list">
              {#each capabilityDetails(plugin) as detail}
                <span><strong>{detail.label}:</strong> {detail.value}</span>
              {/each}
            </div>
            {#if plugin.capabilities?.destructive_actions}
              <p class="approval-note">
                Direct execution is disabled. Ask Heliox in chat so the guarded approval flow can review this action.
              </p>
            {/if}
          </div>

          <div class="tools-section">
            <span class="tools-label">{plugin.tools.length} tools</span>
            <div class="tools-list">
              {#each plugin.tools as tool}
                <button
                  class="tool-tag"
                  class:clickable={plugin.installed && !plugin.capabilities?.destructive_actions}
                  onclick={() => {
                    if (plugin.installed && !plugin.capabilities?.destructive_actions) {
                      toggleExpand(`${plugin.name}:${tool.name}`);
                    }
                  }}
                  title={plugin.capabilities?.destructive_actions
                    ? "Run this tool through chat approval"
                    : plugin.installed
                      ? `Run ${tool.name}`
                      : tool.description}
                >
                  {tool.name}
                  {#if plugin.installed}
                    <span class="run-icon">▶</span>
                  {/if}
                </button>
              {/each}
            </div>
          </div>

          <!-- Expanded tool runner -->
          {#each plugin.tools as tool}
            {#if expandedPlugin === `${plugin.name}:${tool.name}` && plugin.installed}
              <div class="tool-runner">
                <div class="tool-runner-header">
                  <span class="tool-runner-name">{tool.name}</span>
                  <span class="tool-runner-desc">{tool.description}</span>
                </div>
                {#if tool.inputs && tool.inputs.length > 0}
                  <div class="tool-inputs-form">
                    {#each tool.inputs as input}
                      <div class="input-row">
                        <label for="ti-{tool.name}-{input}">{input}</label>
                        <input
                          id="ti-{tool.name}-{input}"
                          type="text"
                          placeholder={`Enter ${input}`}
                          oninput={(e: Event) =>
                            setToolArg(`${plugin.name}:${tool.name}`, input, (e.target as HTMLInputElement).value)}
                        />
                      </div>
                    {/each}
                  </div>
                {/if}
                <button
                  class="btn-run"
                  onclick={() => runTool(plugin.name, tool.name)}
                  disabled={runningTool === `${plugin.name}:${tool.name}`}
                >
                  {runningTool === `${plugin.name}:${tool.name}` ? "⟳ Running..." : "▶ Execute"}
                </button>
                {#if toolResult[`${plugin.name}:${tool.name}`]}
                  <div class="tool-output" class:error-output={toolResult[`${plugin.name}:${tool.name}`]?.error}>
                    <pre>{formatResult(toolResult[`${plugin.name}:${tool.name}`])}</pre>
                  </div>
                {/if}
              </div>
            {/if}
          {/each}

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
              <button class="btn-install" onclick={() => installPlugin(plugin)} disabled={installing === plugin.name}>
                {installing === plugin.name ? "Installing..." : "Install"}
              </button>
            {/if}
            {#if plugin.url}
              <a href={plugin.url} target="_blank" rel="noopener" class="btn-link">GitHub ↗</a>
            {/if}
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

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .catalog-source,
  .local-badge {
    width: fit-content;
    padding: 3px 8px;
    border: 1px solid rgba(74, 222, 128, 0.35);
    border-radius: 999px;
    color: var(--success, #4ade80);
    background: rgba(74, 222, 128, 0.1);
    font-size: 10px;
    font-weight: 600;
  }

  .catalog-source.offline {
    color: #f59e0b;
    border-color: rgba(245, 158, 11, 0.35);
    background: rgba(245, 158, 11, 0.1);
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
  }

  .refresh-btn,
  .btn-create,
  .btn-guide {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
    cursor: pointer;
  }

  .refresh-btn {
    color: var(--on-accent);
    background: var(--accent);
  }

  .btn-create,
  .btn-guide {
    color: var(--accent);
    background: var(--accent-muted, rgba(99, 102, 241, 0.12));
    border: 1px solid var(--accent);
  }

  .btn-create:hover,
  .btn-guide:hover {
    background: var(--accent);
    color: var(--on-accent);
  }

  .refresh-btn:hover:not(:disabled) {
    background: var(--accent-hover);
    color: var(--on-accent);
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .error-message {
    padding: 10px 14px;
    background: var(--danger-bg, rgba(239, 68, 68, 0.1));
    color: var(--danger, #ef4444);
    border: 1px solid var(--danger, #ef4444);
    border-radius: var(--radius-md);
    margin-bottom: 16px;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .dismiss-btn {
    background: none;
    color: var(--danger, #ef4444);
    font-size: 14px;
    cursor: pointer;
    padding: 0 4px;
  }

  .notice-message,
  .catalog-warning {
    padding: 10px 14px;
    border-radius: var(--radius-md);
    margin-bottom: 12px;
    font-size: 12px;
  }

  .notice-message {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(74, 222, 128, 0.4);
    color: var(--success, #4ade80);
    background: rgba(74, 222, 128, 0.08);
  }

  .notice-dismiss {
    color: var(--success, #4ade80);
  }

  .catalog-warning {
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.08);
  }

  .publishing-guide {
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
    font-size: 12px;
    line-height: 1.55;
  }

  .publishing-guide h3 {
    color: var(--accent);
  }

  .publishing-guide p {
    margin: 5px 0 10px;
    color: var(--text-secondary);
  }

  .publishing-guide ol {
    margin: 8px 0 12px 20px;
    color: var(--text-secondary);
  }

  .publishing-guide li {
    margin: 4px 0;
  }

  .publishing-guide code {
    padding: 1px 5px;
    border-radius: 4px;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    font-family: var(--font-mono, monospace);
  }

  .publishing-guide .release-note {
    color: var(--success, #4ade80);
    font-weight: 600;
  }

  .guide-link {
    color: var(--accent);
    font-weight: 600;
  }

  .loading-state,
  .empty-state {
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
    to {
      transform: rotate(360deg);
    }
  }

  /* ── Create Plugin Panel ── */
  .create-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 16px;
    animation: slideDown 0.2s ease-out;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .create-panel h3 {
    font-size: 13px;
    font-weight: 600;
    margin: 0 0 12px 0;
    color: var(--accent);
  }

  .local-explainer {
    margin: -4px 0 12px;
    color: var(--text-muted);
    font-size: 11px;
    line-height: 1.4;
  }

  .create-panel h4 {
    font-size: 12px;
    font-weight: 600;
    margin: 12px 0 8px 0;
    color: var(--text-secondary);
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 10px;
    margin-bottom: 8px;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .form-group label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }

  .form-group input,
  .form-group textarea {
    padding: 7px 10px;
    font-size: 12px;
    font-family: var(--font-mono, monospace);
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    outline: none;
    transition: border-color 0.15s;
  }

  .form-group input:focus,
  .form-group textarea:focus {
    border-color: var(--accent);
  }

  .form-group textarea {
    resize: vertical;
    min-height: 80px;
  }

  .tools-builder {
    margin-bottom: 10px;
  }

  .tool-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 8px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    margin-bottom: 4px;
    font-size: 12px;
  }

  .tool-name-tag {
    font-family: var(--font-mono, monospace);
    font-weight: 600;
    color: var(--accent);
  }

  .tool-desc-text {
    color: var(--text-secondary);
    flex: 1;
  }

  .tool-inputs-text {
    color: var(--text-muted);
    font-size: 11px;
  }

  .tool-remove {
    background: none;
    color: var(--danger, #ef4444);
    font-size: 12px;
    cursor: pointer;
    padding: 0 4px;
  }

  .add-tool-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .tool-input-sm {
    width: 120px;
    padding: 5px 8px;
    font-size: 11px;
    font-family: var(--font-mono, monospace);
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    outline: none;
  }

  .tool-input-md {
    flex: 1;
    padding: 5px 8px;
    font-size: 11px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    outline: none;
  }

  .btn-add-tool {
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    background: var(--accent-muted, rgba(99, 102, 241, 0.12));
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .btn-submit {
    width: 100%;
    padding: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--on-accent);
    background: var(--accent);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.15s;
    margin-top: 8px;
  }

  .btn-submit:hover:not(:disabled) {
    background: var(--accent-hover);
    color: var(--on-accent);
  }

  .btn-submit:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* ── Plugin Cards ── */
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
    border-color: var(--success, #4ade80);
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
    background: var(--accent-muted, rgba(99, 102, 241, 0.12));
    color: var(--accent);
  }

  .status-badge.installed {
    background: rgba(74, 222, 128, 0.15);
    color: var(--success, #4ade80);
  }

  .description {
    font-size: 12px;
    color: var(--text-secondary);
    margin: 0 0 10px 0;
    line-height: 1.4;
  }

  .capability-panel {
    margin: 0 0 12px;
    padding: 10px;
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: var(--radius-sm);
    background: rgba(56, 189, 248, 0.05);
  }

  .capability-panel.destructive {
    border-color: rgba(245, 158, 11, 0.4);
    background: rgba(245, 158, 11, 0.06);
  }

  .capability-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .isolation-badge {
    padding: 2px 7px;
    border-radius: 999px;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    font-size: 9px;
  }

  .destructive .isolation-badge {
    color: #f59e0b;
    background: rgba(245, 158, 11, 0.12);
  }

  .capability-list {
    display: flex;
    flex-wrap: wrap;
    gap: 5px 12px;
    color: var(--text-muted);
    font-size: 10px;
  }

  .capability-list strong {
    color: var(--text-secondary);
  }

  .approval-note {
    margin: 7px 0 0;
    color: #f59e0b;
    font-size: 10px;
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
    font-family: var(--font-mono, monospace);
    padding: 3px 10px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border-radius: 10px;
    border: 1px solid transparent;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .tool-tag.clickable {
    cursor: pointer;
    border-color: var(--border);
  }

  .tool-tag.clickable:hover {
    background: var(--accent-muted, rgba(99, 102, 241, 0.12));
    border-color: var(--accent);
    color: var(--accent);
  }

  .run-icon {
    font-size: 8px;
    opacity: 0.6;
  }

  /* ── Tool Runner ── */
  .tool-runner {
    background: var(--bg-tertiary);
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    padding: 12px;
    margin-bottom: 10px;
    animation: slideDown 0.15s ease-out;
  }

  .tool-runner-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .tool-runner-name {
    font-size: 13px;
    font-weight: 600;
    font-family: var(--font-mono, monospace);
    color: var(--accent);
  }

  .tool-runner-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .tool-inputs-form {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-bottom: 8px;
  }

  .input-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .input-row label {
    font-size: 11px;
    font-weight: 600;
    font-family: var(--font-mono, monospace);
    color: var(--text-muted);
    min-width: 70px;
  }

  .input-row input {
    flex: 1;
    padding: 5px 8px;
    font-size: 12px;
    font-family: var(--font-mono, monospace);
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    outline: none;
  }

  .input-row input:focus {
    border-color: var(--accent);
  }

  .btn-run {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-run:hover:not(:disabled) {
    filter: brightness(1.1);
    transform: translateY(-1px);
  }

  .btn-run:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }

  .tool-output {
    margin-top: 8px;
    padding: 10px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: var(--radius-sm);
    max-height: 200px;
    overflow-y: auto;
  }

  .tool-output.error-output {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.25);
  }

  .tool-output pre {
    font-size: 11px;
    font-family: var(--font-mono, monospace);
    color: var(--text-primary);
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .error-output pre {
    color: var(--danger, #ef4444);
  }

  /* ── Action Buttons ── */
  .plugin-actions {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .btn-install,
  .btn-uninstall {
    padding: 6px 16px;
    font-size: 12px;
    font-weight: 600;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
    cursor: pointer;
  }

  .btn-install {
    background: var(--accent);
    color: var(--on-accent);
  }

  .btn-install:hover:not(:disabled) {
    background: var(--accent-hover);
    color: var(--on-accent);
  }

  .btn-uninstall {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }

  .btn-uninstall:hover:not(:disabled) {
    background: var(--danger-bg, rgba(239, 68, 68, 0.1));
    color: var(--danger, #ef4444);
    border-color: var(--danger, #ef4444);
  }

  .btn-install:disabled,
  .btn-uninstall:disabled {
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
