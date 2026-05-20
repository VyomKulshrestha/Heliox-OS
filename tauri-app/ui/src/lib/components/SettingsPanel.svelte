<script lang="ts">
  import { settings } from "../stores/settings";
  import { _, locale } from 'svelte-i18n';
  import { session } from "../stores/session";
  import { call } from "../api/daemon";
  import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";

  let apiKeyInput = $state("");
  let apiKeySaved = $state(false);
  let apiKeySaving = $state(false);

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

  function toggleTheme() {
    const currentTheme = $settings.theme || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    settings.updateSection("", { theme: nextTheme });
  }

  async function testNotification() {
    let permissionGranted = await isPermissionGranted();
    if (!permissionGranted) {
      const permission = await requestPermission();
      permissionGranted = permission === "granted";
    }
    if (permissionGranted) {
      sendNotification({ title: "Heliox OS", body: "Notifications are working!" });
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
</script>

<div class="settings-panel">
  <h2>{$_('settings.title')}</h2>

  <section class="settings-group">
    <h3>{$_('settings.appearance')}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.light_mode')}</span>
        <span class="setting-desc">{$_('settings.light_mode_desc')}</span>
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

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.language')}</span>
        <span class="setting-desc">{$_('settings.language_desc')}</span>
      </div>
      <select class="input-md" bind:value={$locale}>
        <option value="en">English</option>
        <option value="hi">हिन्दी</option>
      </select>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.security')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.root_access')}</span>
        <span class="setting-desc">{$_('settings.root_access_desc')}</span>
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
        <span class="setting-label">{$_('settings.auto_snapshot')}</span>
        <span class="setting-desc">{$_('settings.auto_snapshot_desc')}</span>
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
        <span class="setting-label">{$_('settings.dry_run')}</span>
        <span class="setting-desc">{$_('settings.dry_run_desc')}</span>
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
        <span class="setting-label">{$_('settings.snapshot_retention')}</span>
        <span class="setting-desc">{$_('settings.snapshot_retention_desc')}</span>
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
    <h3>{$_('settings.usage')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.total_tokens')}</span>
        <span class="setting-desc">{$_('settings.total_tokens_desc')}</span>
      </div>
      <span>{$session.totalTokens}</span>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.estimated_cost')}</span>
        <span class="setting-desc">{$_('settings.estimated_cost_desc')}</span>
      </div>
      <span>
        {$settings.model.provider === "ollama" ? $_('settings.free_local') : `$${$session.estimatedCost.toFixed(4)}`}
      </span>
    </div>

    <div class="setting-row">
      <button class="btn-save" onclick={() => session.resetUsage()}>{$_('settings.reset_usage')}</button>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.screen_vision')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.capture_interval')}</span>
        <span class="setting-desc">{$_('settings.capture_interval_desc')}</span>
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
    <h3>{$_('settings.model')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.provider')}</span>
        <span class="setting-desc">{$_('settings.provider_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.provider === "ollama"} onclick={() => setProvider("ollama")}>Ollama</button>
        <button class:active={$settings.model.provider === "cloud"} onclick={() => setProvider("cloud")}>Cloud</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.reasoning_mode')}</span>
        <span class="setting-desc">{$_('settings.reasoning_mode_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.mode === "lightweight"} onclick={() => setMode("lightweight")}>{$_('settings.light')}</button>
        <button class:active={$settings.model.mode === "full"} onclick={() => setMode("full")}>{$_('settings.full')}</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.ollama_model')}</span>
        <span class="setting-desc">{$_('settings.ollama_model_desc')}</span>
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
        <span class="setting-label">{$_('settings.gpu_memory')}</span>
        <span class="setting-desc">{$_('settings.gpu_memory_desc')}</span>
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
    <h3>{$_('settings.cloud_api')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.cloud_provider')}</span>
        <span class="setting-desc">{$_('settings.cloud_provider_desc')}</span>
      </div>
      <div class="btn-group">
        <button class:active={$settings.model.cloud_provider === "gemini"} onclick={() => setCloudProvider("gemini")}>Gemini</button>
        <button class:active={$settings.model.cloud_provider === "openai"} onclick={() => setCloudProvider("openai")}>OpenAI</button>
        <button class:active={$settings.model.cloud_provider === "claude"} onclick={() => setCloudProvider("claude")}>Claude</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.cloud_model')}</span>
        <span class="setting-desc">{$_('settings.cloud_model_desc')}</span>
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
        <span class="setting-label">{$_('settings.api_key')}</span>
        <span class="setting-desc">{$_('settings.api_key_desc')}</span>
      </div>
      <div class="api-key-row">
        <input type="password" class="input-md" bind:value={apiKeyInput} placeholder={$_('settings.api_key_placeholder')} />
        <button class="btn-save" onclick={saveApiKey} disabled={apiKeySaving}>
          {apiKeySaved ? $_('settings.saved') : apiKeySaving ? $_('settings.saving') : $_('settings.save')}
        </button>
      </div>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.restrictions')}</h3>
    <div class="restriction-info">
      <p>{$_('settings.protected_folders')}: {$settings.restrictions?.protected_folders?.length || 0} {$_('settings.configured')}</p>
      <p>{$_('settings.protected_packages')}: {$settings.restrictions?.protected_packages?.length || 0} {$_('settings.configured')}</p>
      <p>{$_('settings.blocked_commands')}: {$settings.restrictions?.blocked_commands?.length || 0} {$_('settings.configured')}</p>
    </div>
  </section>

  <section class="settings-group">
    <h3>{$_('settings.debug')}</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.notifications')}</span>
        <span class="setting-desc">{$_('settings.notifications_desc')}</span>
      </div>
      <button class="btn-save" onclick={testNotification}>{$_('settings.test_popup')}</button>
    </div>
  </section>
</div>

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
</style>