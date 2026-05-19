<script lang="ts">
  import { settings } from "../stores/settings";
  import { session } from "../stores/session";
  import { call } from "../api/daemon";
  import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
  import ResetConfirmModal from "./ResetConfirmModal.svelte";

  let apiKeyInput = $state("");
  let apiKeySaved = $state(false);
  let apiKeySaving = $state(false);
  let showResetModal = $state(false);
  let resetting = $state(false);

  function toggleRoot() {
    settings.updateSection("security", { root_enabled: !$settings.security.root_enabled });
  }

  function toggleDryRun() {
    settings.updateSection("security", { dry_run: !$settings.security.dry_run });
  }

  function setMode(mode: string) {
    settings.updateSection("model", { mode });
  }

  function setProvider(provider: string) {
    settings.updateSection("model", { provider });
  }

  function setCloudProvider(cloud_provider: string) {
    settings.updateSection("model", { cloud_provider, provider: "cloud" });
  }

  function updateCloudModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { cloud_model: val });
  }

  function updateGpuLimit(e: Event) {
    const val = parseInt((e.target as HTMLInputElement).value) || 0;
    settings.updateSection("model", { gpu_memory_limit_mb: val });
  }

  function updateRetention(e: Event) {
    const val = parseInt((e.target as HTMLInputElement).value) || 10;
    settings.updateSection("security", { snapshot_retention_count: val });
  }

  function updateScreenVisionInterval(e: Event) {
    const rawValue = Number((e.target as HTMLInputElement).value);
    if (!Number.isFinite(rawValue)) return;
    const capture_interval_seconds = Math.min(60, Math.max(0.5, rawValue));
    settings.updateSection("screen_vision", { capture_interval_seconds });
  }

  function updateOllamaModel(e: Event) {
    const val = (e.target as HTMLInputElement).value;
    settings.updateSection("model", { ollama_model: val });
  }

  async function saveApiKey() {
    if (!apiKeyInput.trim()) return;
    apiKeySaving = true;
    try {
      const provider = $settings.model.cloud_provider || "gemini";
      await call("store_api_key", { provider, api_key: apiKeyInput.trim() });
      apiKeySaved = true;
      apiKeyInput = "";
      setTimeout(() => {
        apiKeySaved = false;
      }, 3000);
    } catch (err) {
      console.error("Failed to save API key:", err);
    } finally {
      apiKeySaving = false;
    }
  }

  // Toggle between dark and light mode targeting root settings configuration state
  function toggleTheme() {
    const currentTheme = $settings.theme || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    
    // Update the central store root section directly
    // The store's internal side-effects will automatically manage document classes and localStorage synchronization
    settings.updateSection("", { theme: nextTheme });
  }
  async function testNotification() {
    try {
      let granted = await isPermissionGranted();
      if (!granted) {
        const permission = await requestPermission();
        granted = permission === 'granted';
      }
      if (granted) {
        sendNotification({
          title: 'Heliox OS',
          body: 'Test notification — desktop notifications are working! 🚀',
        });
      } else {
        console.warn('Notification permission was denied');
      }
    } catch (err) {
      console.error('Notification test failed:', err);
    }
  }

  async function handleReset() {
    resetting = true;
    try {
      await settings.resetToDefaults();
      showResetModal = false;
      sendNotification({
        title: "Settings Reset",
        body: "All configurations have been restored to factory defaults."
      });
    } catch (err) {
      console.error("Failed to reset:", err);
      sendNotification({
        title: "Reset Failed",
        body: "There was an error while resetting configurations."
      });
    } finally {
      resetting = false;
    }
  }
</script>

<div class="settings-panel">
  <h2>Settings</h2>

  <section class="settings-group">
    <h3>Appearance</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Light Mode</span>
        <span class="setting-desc">Switch between dark and light themes</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.theme === "light"}
        onclick={toggleTheme}
        aria-label="Toggle Light Mode"
        title="Toggle Light Mode"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>Security</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Root Access</span>
        <span class="setting-desc">Allow actions that require superuser privileges</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.root_enabled}
        onclick={toggleRoot}
        aria-label="Toggle Root Access"
        title="Toggle Root Access"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Auto-Snapshot</span>
        <span class="setting-desc">Create system snapshot before destructive actions</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.snapshot_on_destructive}
        onclick={() =>
          settings.updateSection("security", {
            snapshot_on_destructive: !$settings.security.snapshot_on_destructive,
          })}
        aria-label="Toggle Auto Snapshot"
        title="Toggle Auto Snapshot"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Dry Run Mode</span>
        <span class="setting-desc">Plan and log actions without changing the OS, files, or processes</span>
      </div>
      <button
        class="toggle"
        class:active={$settings.security.dry_run}
        onclick={toggleDryRun}
        aria-label="Toggle Dry Run Mode"
        title="Toggle Dry Run Mode"
      >
        <span class="toggle-knob"></span>
      </button>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Snapshot Retention</span>
        <span class="setting-desc">Number of snapshots to keep</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.security.snapshot_retention_count}
        onchange={updateRetention}
        min="1"
        max="100"
      />
    </div>
  </section>

  <section class="settings-group">
    <h3>Usage</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Total Tokens</span>
        <span class="setting-desc">Estimated session token usage</span>
      </div>
      <span>{$session.totalTokens}</span>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Estimated Cost</span>
        <span class="setting-desc">Approximate API usage cost</span>
      </div>

      <span>
        {$settings.model.provider === "ollama" ? "Free (local)" : `$${$session.estimatedCost.toFixed(4)}`}
      </span>
    </div>

    <div class="setting-row">
      <button class="btn-save" onclick={() => session.resetUsage()}> Reset Session Usage </button>
    </div>
  </section>

  <section class="settings-group">
    <h3>Screen Vision</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Capture Interval</span>
        <span class="setting-desc">Seconds between screen awareness captures</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.screen_vision?.capture_interval_seconds ?? 3}
        onchange={updateScreenVisionInterval}
        min="0.5"
        max="60"
        step="0.5"
      />
    </div>
  </section>

  <section class="settings-group">
    <h3>Model</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Provider</span>
        <span class="setting-desc">Primary model backend</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.provider === "ollama"} onclick={() => setProvider("ollama")}
          >Ollama</button
        >
        <button class:active={$settings.model.provider === "cloud"} onclick={() => setProvider("cloud")}>Cloud</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Reasoning Mode</span>
        <span class="setting-desc">Trade speed for accuracy</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.mode === "lightweight"} onclick={() => setMode("lightweight")}
          >Light</button
        >
        <button class:active={$settings.model.mode === "full"} onclick={() => setMode("full")}>Full</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Ollama Model</span>
        <span class="setting-desc">Model tag to use with Ollama</span>
      </div>
      <input
        type="text"
        class="input-md"
        value={$settings.model.ollama_model}
        onchange={updateOllamaModel}
        placeholder="llama3.1:8b"
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">GPU Memory Limit</span>
        <span class="setting-desc">Max VRAM in MB (0 = unlimited)</span>
      </div>
      <input
        type="number"
        class="input-sm"
        value={$settings.model.gpu_memory_limit_mb}
        onchange={updateGpuLimit}
        min="0"
        step="512"
      />
    </div>
  </section>

  <section class="settings-group">
    <h3>Cloud API (Fast)</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Cloud Provider</span>
        <span class="setting-desc">Select your cloud LLM provider</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.cloud_provider === "gemini"} onclick={() => setCloudProvider("gemini")}
          >Gemini</button
        >
        <button class:active={$settings.model.cloud_provider === "openai"} onclick={() => setCloudProvider("openai")}
          >OpenAI</button
        >
        <button class:active={$settings.model.cloud_provider === "claude"} onclick={() => setCloudProvider("claude")}
          >Claude</button
        >
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Cloud Model</span>
        <span class="setting-desc">Override model (blank = default)</span>
      </div>
      <input
        type="text"
        class="input-md"
        value={$settings.model.cloud_model}
        onchange={updateCloudModel}
        placeholder="gemini-2.0-flash"
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">API Key</span>
        <span class="setting-desc">Stored securely in system keyring</span>
      </div>
      <div class="api-key-row">
        <input type="password" class="input-md" bind:value={apiKeyInput} placeholder="Paste API key..." />
        <button class="btn-save" onclick={saveApiKey} disabled={apiKeySaving}>
          {apiKeySaved ? "✓ Saved!" : apiKeySaving ? "Saving..." : "Save"}
        </button>
      </div>
    </div>
  </section>

  <section class="settings-group">
    <h3>Restrictions</h3>
    <div class="restriction-info">
      <p>Protected folders: {$settings.restrictions?.protected_folders?.length || 0} configured</p>
      <p>Protected packages: {$settings.restrictions?.protected_packages?.length || 0} configured</p>
      <p>Blocked commands: {$settings.restrictions?.blocked_commands?.length || 0} configured</p>
    </div>
  </section>

  <section class="settings-group">
    <h3>Debug</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Notifications</span>
        <span class="setting-desc">Test native OS desktop popup</span>
      </div>
      <button class="btn-save" onclick={testNotification}>Test Popup</button>
    </div>
  </section>

  <section class="settings-group danger-zone">
    <h3>Danger Zone</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label text-danger">Factory Reset</span>
        <span class="setting-desc">Clear all settings and local data</span>
      </div>
      <button class="btn-danger" onclick={() => showResetModal = true} disabled={resetting}>
        {resetting ? "Resetting..." : "Reset to Defaults"}
      </button>
    </div>
  </section>
</div>

<ResetConfirmModal 
  show={showResetModal} 
  onconfirm={handleReset} 
  oncancel={() => showResetModal = false} 
/>

<style>
  .settings-panel {
    height: 100%; 
    overflow-y: auto;
    padding: 16px;
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 16px;
  }

  .settings-group {
    margin-bottom: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
  }

  h3 {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  .setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
  }

  .setting-row:last-child {
    border-bottom: none;
  }

  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .setting-label {
    font-size: 13px;
    font-weight: 500;
  }

  .setting-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  .toggle {
    width: 40px;
    height: 22px;
    border-radius: 11px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    position: relative;
    transition: all 0.2s;
    cursor: pointer;
    flex-shrink: 0;
  }

  .toggle.active {
    background: var(--accent);
    border-color: var(--accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    transition: transform 0.2s;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .btn-group {
    display: flex;
    gap: 2px;
    background: var(--bg-primary);
    border-radius: var(--radius-sm);
    padding: 2px;
  }

  .btn-group button {
    padding: 4px 12px;
    font-size: 11px;
    color: var(--text-secondary);
    background: transparent;
    border-radius: 4px;
    transition: all 0.15s;
  }

  .btn-group button:hover {
    color: var(--text-primary);
  }

  .btn-group button.active {
    background: var(--accent);
    color: white;
  }

  .input-sm {
    width: 80px;
    padding: 5px 8px;
    font-size: 13px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    text-align: right;
  }

  .input-md {
    width: 160px;
    padding: 5px 8px;
    font-size: 13px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .restriction-info {
    padding: 10px 14px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .restriction-info p {
    margin: 0;
  }

  .api-key-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .btn-save {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
    white-space: nowrap;
  }

  .btn-save:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .btn-save:disabled {
    cursor: not-allowed;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }

  .btn-group button:disabled {
    cursor: not-allowed;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
  }

  .text-danger {
    color: #ef4444;
  }

  .danger-zone {
    border-color: rgba(239, 68, 68, 0.3);
  }

  .danger-zone h3 {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border-bottom-color: rgba(239, 68, 68, 0.2);
  }

  .btn-danger {
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: #ef4444;
    border: none;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
    cursor: pointer;
  }

  .btn-danger:hover:not(:disabled) {
    background: #dc2626;
  }

  .btn-danger:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }
</style>