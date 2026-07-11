<script lang="ts">
  import { settings } from "../stores/settings";
  import { _, locale } from 'svelte-i18n';
  import { session } from "../stores/session";
  import { call } from "../api/daemon";
  import { invoke } from "../api/invoke";
  import { isPermissionGranted, requestPermission, sendNotification } from "@tauri-apps/plugin-notification";
  let apiKeyInput = $state("");
  let apiKeySaved = $state(false);
  let apiKeySaving = $state(false);
  let rootToast = $state("");
  let rootToastType = $state<"success" | "warning">("success");

  $effect(() => {
    if ($locale) {
      localStorage.setItem("locale", $locale);
    }
  });

  let hotkeyInput = $state("");
  let hotkeySaved = $state(false);
  let hotkeyError = $state("");

  const cloudModels: Record<string, string[]> = {
    gemini: ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    openai: ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
    claude: ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"]
  };

  let availableOllamaModels = $state<string[]>([]);

  $effect(() => {
    call("list_ollama_models").then((res: any) => {
      if (res && res.models) {
        availableOllamaModels = res.models;
      }
    }).catch(console.error);
  });

  // Load the current hotkey when component mounts
  invoke("get_hotkey").then((val) => {
    hotkeyInput = val as string;
  });

  async function saveHotkey() {
    hotkeyError = "";
    const trimmed = hotkeyInput.trim();
    if (!trimmed) return;
    try {
      await invoke("set_hotkey", { shortcut: trimmed });
      settings.updateSection("", { hotkey: trimmed });
      hotkeySaved = true;
      setTimeout(() => (hotkeySaved = false), 3000);
    } catch (err) {
      hotkeyError = "Invalid shortcut. Try: Ctrl+Space or Alt+H";
    }
  }

  function toggleRoot() {
    const turningOn = !$settings.security.root_enabled;
    if (turningOn) {
      const confirmed = confirm(
        "⚠️ ENABLE ROOT ACCESS?\n\n" +
        "This unlocks Heliox's full power:\n" +
        "• Admin/sudo shell commands\n" +
        "• System service management\n" +
        "• Protected file modifications\n" +
        "• Registry & disk-level operations\n\n" +
        "Actions requiring elevated privileges will no longer be blocked.\n" +
        "Only enable this if you trust the AI agent with system-level access."
      );
      if (!confirmed) return;
      rootToast = "🔓 Root access ENABLED — Heliox now has full system privileges";
      rootToastType = "warning";
    } else {
      rootToast = "🔒 Root access DISABLED — elevated actions will be blocked";
      rootToastType = "success";
    }
    settings.updateSection("security", { root_enabled: turningOn });
    setTimeout(() => (rootToast = ""), 5000);
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
  
  // Function to reset all settings to defaults
  async function handleReset() {
    const confirmed = confirm($_('settings.reset_confirm'));
    if (!confirmed) return;
    await settings.reset();
  }
  function toggleTheme() {
    const currentTheme = $settings.theme || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";

    // Update the central store root section directly
    // The store's internal side-effects will automatically manage document classes and localStorage synchronization
    settings.updateSection("", { theme: nextTheme });
  }

  async function testNotification() {
    try {
      // Try Tauri native notification first
      let granted = await isPermissionGranted();
      if (!granted) {
        const permission = await requestPermission();
        granted = permission === "granted";
      }
      if (granted) {
        sendNotification({
          title: "Heliox OS",
          body: "Test notification — desktop notifications are working! 🚀",
        });
        return;
      }
    } catch {
      // Tauri plugin not available (running in browser) — fall through
    }

    // Fallback: browser Notification API
    try {
      if (!("Notification" in window)) {
        alert("Heliox OS — Notifications are not supported in this browser.");
        return;
      }
      let perm = Notification.permission;
      if (perm === "default") {
        perm = await Notification.requestPermission();
      }
      if (perm === "granted") {
        new Notification("Heliox OS", {
          body: "Test notification — desktop notifications are working! 🚀",
          icon: "/favicon.png",
        });
      } else {
        alert("Notification permission was denied. Enable it in your browser settings.");
      }
    } catch (err) {
      console.error("Browser notification failed:", err);
      alert("Heliox OS — Test notification triggered! (popups blocked by browser)");
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
        <option value="hi">Hindi</option>
      </select>
    </div>
  </section>

  <section class="settings-group">
    <h3>Keyboard Shortcut</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Global Hotkey</span>
        <span class="setting-desc">Summon Heliox from anywhere (default: Ctrl+Space)</span>
      </div>
      <div class="api-key-row">
        <input
          type="text"
          class="input-md"
          bind:value={hotkeyInput}
          placeholder="Ctrl+Space"
        />
        <button class="btn-save" onclick={saveHotkey}>
          {hotkeySaved ? "✓ Saved!" : "Save"}
        </button>
      </div>
    </div>
    {#if hotkeyError}
      <div style="padding: 6px 14px; font-size: 11px; color: var(--accent);">
        {hotkeyError}
      </div>
    {/if}
  </section>

  <section class="settings-group">
    <h3>{$_('settings.security')}</h3>

    {#if rootToast}
      <div class="root-toast" class:root-toast-warning={rootToastType === 'warning'} class:root-toast-success={rootToastType === 'success'}>
        {rootToast}
      </div>
    {/if}

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

    {#if $settings.security.root_enabled}
      <div class="root-status-banner">
        <span class="root-icon">⚡</span>
        <div class="root-status-info">
          <span class="root-status-title">Root Mode Active</span>
          <span class="root-status-desc">Admin commands, service control, registry writes and protected file access are now unlocked.</span>
        </div>
      </div>
    {/if}

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
      {#if availableOllamaModels.length > 0}
        <select
          class="input-md"
          value={$settings.model.ollama_model}
          onchange={updateOllamaModel}
        >
          {#each availableOllamaModels as modelOption}
            <option value={modelOption}>{modelOption}</option>
          {/each}
        </select>
      {:else}
        <input
          type="text"
          class="input-md"
          value={$settings.model.ollama_model}
          onchange={updateOllamaModel}
          placeholder="llama3.1:8b"
        />
      {/if}
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
      <select
        class="input-md"
        value={$settings.model.cloud_model}
        onchange={updateCloudModel}
      >
        <option value="">Default for provider</option>
        {#each (cloudModels[$settings.model.cloud_provider || "gemini"] || []) as modelOption}
          <option value={modelOption}>{modelOption}</option>
        {/each}
      </select>
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
    <h3>Data & Export</h3>
    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Export Chat History</span>
        <span class="setting-desc">Save the current session history and tool logs to your Downloads folder.</span>
      </div>
      <div class="btn-group">
        <button onclick={() => session.exportChat("markdown")}>Markdown</button>
        <button onclick={() => session.exportChat("json")}>JSON</button>
      </div>
    </div>
  </section>

  <section class="settings-group">
    <h3>Budget &amp; Limits</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Budget enforcement</span>
        <span class="setting-desc">Master switch for all budget checks. When off, none of the limits below are enforced.</span>
      </div>
      <div class="btn-group">
        <button
          class:active={$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: true })}
        >On</button>
        <button
          class:active={!$settings.model.budget_enabled}
          onclick={() => settings.updateSection("model", { budget_enabled: false })}
        >Off</button>
      </div>
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Monthly USD limit</span>
        <span class="setting-desc">Cumulative cap across all cloud calls for the current calendar month.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="0.5"
        value={$settings.model.budget_monthly_limit_usd}
        onchange={(e) => settings.updateSection("model", { budget_monthly_limit_usd: parseFloat((e.target as HTMLInputElement).value) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-action token cap</span>
        <span class="setting-desc">Maximum estimated input tokens for a single LLM call. Oversized prompts are blocked before being sent.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="500"
        value={$settings.model.max_tokens_per_action}
        onchange={(e) => settings.updateSection("model", { max_tokens_per_action: parseInt((e.target as HTMLInputElement).value, 10) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task token cap</span>
        <span class="setting-desc">Cumulative token budget for a single user task. Halts the task cleanly when exceeded.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="5000"
        value={$settings.model.max_tokens_per_task}
        onchange={(e) => settings.updateSection("model", { max_tokens_per_task: parseInt((e.target as HTMLInputElement).value, 10) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Per-task USD cap</span>
        <span class="setting-desc">Cumulative USD cap per task. Useful for bounding cloud spend on autonomous loops.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="0"
        step="0.01"
        value={$settings.model.max_usd_per_task}
        onchange={(e) => settings.updateSection("model", { max_usd_per_task: parseFloat((e.target as HTMLInputElement).value) || 0 })}
      />
    </div>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">Failure threshold</span>
        <span class="setting-desc">Circuit breaker trips after this many consecutive action failures in a task.</span>
      </div>
      <input
        type="number"
        class="input-sm"
        min="1"
        max="20"
        step="1"
        value={$settings.model.max_consecutive_failures}
        onchange={(e) => settings.updateSection("model", { max_consecutive_failures: parseInt((e.target as HTMLInputElement).value, 10) || 1 })}
      />
    </div>
  </section>

  <!--Adding a reset button to clear all settings and return to defaults, with a confirmation prompt to prevent accidental resets -->
  <section class="settings-group">
    <h3>{$_('settings.reset_header')}</h3>

    <div class="setting-row">
      <div class="setting-info">
        <span class="setting-label">{$_('settings.reset_label')}</span>
        <span class="setting-desc">
          {$_('settings.reset_desc')}
        </span>
      </div>

      <button class="btn-save" onclick={handleReset}>
        {$_('settings.reset_button')}
      </button>
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

  /* Root Access Toast & Banner */
  .root-toast {
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 0;
    animation: toastSlide 0.3s ease-out;
  }

  .root-toast-warning {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border-bottom: 1px solid rgba(245, 158, 11, 0.3);
  }

  .root-toast-success {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border-bottom: 1px solid rgba(16, 185, 129, 0.3);
  }

  @keyframes toastSlide {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .root-status-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    background: rgba(245, 158, 11, 0.08);
    border-bottom: 1px solid var(--border);
  }

  .root-icon {
    font-size: 18px;
    flex-shrink: 0;
  }

  .root-status-info {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .root-status-title {
    font-size: 12px;
    font-weight: 600;
    color: #f59e0b;
  }

  .root-status-desc {
    font-size: 11px;
    color: var(--text-muted);
    line-height: 1.3;
  }
</style>
