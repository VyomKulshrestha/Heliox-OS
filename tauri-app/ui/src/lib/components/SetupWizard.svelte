<script lang="ts">
  import { settings } from "../stores/settings";
  import { call, requireOkResult, requireResultStatus, type DaemonStatusResult } from "../api/daemon";
  import { onMount } from "svelte";

  interface Props {
    oncomplete: () => void | Promise<void>;
  }

  let { oncomplete }: Props = $props();
  let finishing = $state(false);
  let finishError = $state("");
  let step = $state(0);

  let modelProvider = $state("ollama");
  let ollamaModel = $state("");
  let ollamaModels = $state<string[]>([]);
  let ollamaAvailable = $state(false);
  let loadingModels = $state(true);
  let cloudProvider = $state("");
  let cloudModel = $state("");
  let cloudApiKey = $state("");
  let subscriptionProvider = $state<"codex" | "claude">("codex");
  let subscriptionModel = $state("");
  let subscriptionChecking = $state(false);
  let subscriptionMessage = $state("");
  let subscriptionConnected = $state(false);
  let protectedFolders = $state("");
  let protectedPackages = $state("firefox, nautilus");

  const steps = ["Welcome", "Model", "Security", "Ready"];

  onMount(async () => {
    try {
      const result = requireOkResult(
        (await call("list_ollama_models")) as DaemonStatusResult & { models?: string[]; available?: boolean },
        "Ollama model discovery is unavailable.",
      );
      if (!Array.isArray(result.models) || typeof result.available !== "boolean") {
        throw new Error("Ollama model discovery is incomplete.");
      }
      ollamaModels = result.models ?? [];
      ollamaAvailable = result.available ?? false;
      if (ollamaModels.length > 0) {
        ollamaModel = ollamaModels[0];
      }
    } catch {
      ollamaAvailable = false;
    } finally {
      loadingModels = false;
    }
    await refreshSubscriptionStatus(false);
  });

  async function refreshSubscriptionStatus(refresh = true): Promise<boolean> {
    subscriptionChecking = true;
    try {
      const result = requireOkResult(
        await call<DaemonStatusResult & { subscription?: boolean; message?: string }>("subscription_status", {
          provider: subscriptionProvider,
          refresh,
        }),
        "Subscription status is unavailable.",
      );
      if (typeof result.subscription !== "boolean") throw new Error("Subscription status is incomplete.");
      subscriptionConnected = result.subscription;
      subscriptionMessage = result.message ?? "";
      return result.subscription;
    } catch (error) {
      subscriptionConnected = false;
      subscriptionMessage = error instanceof Error ? error.message : "Cannot connect to the Heliox daemon.";
      return false;
    } finally {
      subscriptionChecking = false;
    }
  }

  async function changeSubscriptionProvider(provider: "codex" | "claude") {
    subscriptionProvider = provider;
    subscriptionModel = "";
    subscriptionConnected = false;
    await refreshSubscriptionStatus(true);
  }

  async function startSubscriptionLogin() {
    subscriptionMessage = "";
    try {
      const result = requireResultStatus(
        await call<DaemonStatusResult>("subscription_login", { provider: subscriptionProvider }),
        "started",
        "The official subscription sign-in flow did not start.",
      );
      subscriptionMessage = result.message ?? "";
    } catch (error) {
      subscriptionMessage = error instanceof Error ? error.message : "Could not start the official sign-in flow.";
    }
  }

  async function finish() {
    if (finishing) return;
    finishing = true;
    finishError = "";

    try {
      if (modelProvider === "subscription") {
        const connected = await refreshSubscriptionStatus(true);
        if (!connected) {
          throw new Error(
            `${subscriptionProvider === "codex" ? "Codex" : "Claude Code"} must be installed and signed in through an eligible subscription before setup can finish.`,
          );
        }
      }

      // Setup is only complete once the daemon has saved each choice.
      const modelSaved = await settings.updateSectionDetailed("model", {
        provider: modelProvider,
        ollama_model: ollamaModel,
        cloud_provider: cloudProvider,
        cloud_model: cloudModel,
        subscription_provider: subscriptionProvider,
        subscription_model: subscriptionModel,
      });
      if (!modelSaved.ok) throw new Error(`The model choice was not saved: ${modelSaved.message}`);

      const folders = protectedFolders
        .split("\n")
        .map((f) => f.trim())
        .filter(Boolean);
      const packages = protectedPackages
        .split(",")
        .map((p) => p.trim())
        .filter(Boolean);

      // Add to what is already protected (a reset keeps restrictions); never drop entries here.
      const current = $settings.restrictions;
      const restrictionsSaved = await settings.updateSectionDetailed("restrictions", {
        protected_folders: [...new Set([...(current?.protected_folders ?? []), ...folders])],
        protected_packages: [...new Set([...(current?.protected_packages ?? []), ...packages])],
      });
      if (!restrictionsSaved.ok) {
        throw new Error(`The protected folders and packages were not saved: ${restrictionsSaved.message}`);
      }

      // Cloud setup is incomplete until the daemon confirms that the key
      // reached a secure operating-system credential store.
      if (cloudApiKey && cloudProvider) {
        const result = await call<{ status: string; message?: string }>("store_api_key", {
          provider: cloudProvider,
          key: cloudApiKey,
        });
        if (result.status !== "ok") {
          throw new Error(result.message || "The API key could not be stored securely.");
        }
      }

      // Mark first run complete in localStorage
      localStorage.setItem("heliox_first_run_complete", "true");

      await oncomplete();
    } catch (error) {
      finishError = error instanceof Error ? error.message : "Setup could not be completed.";
    } finally {
      finishing = false;
    }
  }
</script>

<div class="wizard-overlay">
  <div class="wizard">
    <div class="wizard-header">
      <h1>Heliox OS Setup</h1>
      <div class="progress">
        {#each steps as s, i}
          <div class="progress-step" class:active={i === step} class:done={i < step}>
            <span class="step-num">{i + 1}</span>
            <span class="step-label">{s}</span>
          </div>
          {#if i < steps.length - 1}
            <div class="progress-line" class:filled={i < step}></div>
          {/if}
        {/each}
      </div>
    </div>

    <div class="wizard-body">
      {#if step === 0}
        <div class="wizard-step">
          <h2>Welcome to Heliox OS</h2>
          <p>
            Heliox OS is your AI system control agent. It lets you control your computer using natural language, voice,
            or gestures while keeping you in full control.
          </p>
          <p>This setup will configure a few essentials:</p>
          <ul>
            <li>Choose your AI model backend</li>
            <li>Set security boundaries</li>
            <li>Define protected folders and packages</li>
          </ul>
          <p class="note">You can change all of these later in Settings.</p>
        </div>
      {:else if step === 1}
        <div class="wizard-step">
          <h2>Model Configuration</h2>

          <div class="field" role="group" aria-labelledby="provider-label">
            <span id="provider-label" class="field-label">Primary Provider</span>
            <div class="radio-group">
              <label class="radio-option" class:selected={modelProvider === "ollama"}>
                <input type="radio" bind:group={modelProvider} value="ollama" />
                <div>
                  <strong>Ollama (Local)</strong>
                  <span>Private, runs on your GPU. Requires Ollama to be installed.</span>
                </div>
              </label>
              <label class="radio-option" class:selected={modelProvider === "cloud"}>
                <input type="radio" bind:group={modelProvider} value="cloud" />
                <div>
                  <strong>Cloud API</strong>
                  <span>Uses OpenAI, OpenRouter, Claude, or Gemini. Requires API key.</span>
                </div>
              </label>
              <label class="radio-option" class:selected={modelProvider === "subscription"}>
                <input type="radio" bind:group={modelProvider} value="subscription" />
                <div>
                  <strong>Existing Subscription</strong>
                  <span>Use the official Codex or Claude Code CLI login. No API key is stored by Heliox.</span>
                </div>
              </label>
            </div>
          </div>

          {#if modelProvider === "ollama"}
            <div class="field">
              <label for="ollama-model">Ollama Model</label>
              {#if loadingModels}
                <div class="model-status">Detecting models...</div>
              {:else if ollamaModels.length > 0}
                <select id="ollama-model" bind:value={ollamaModel}>
                  {#each ollamaModels as m}
                    <option value={m}>{m}</option>
                  {/each}
                </select>
                <span class="hint"
                  >{ollamaModels.length} model{ollamaModels.length === 1 ? "" : "s"} detected from Ollama</span
                >
              {:else if ollamaAvailable}
                <input id="ollama-model-input" type="text" bind:value={ollamaModel} placeholder="llama3.1:8b" />
                <span class="hint warning"
                  >Ollama is running but no models found. Run <code>ollama pull qwen2.5:7b</code></span
                >
              {:else}
                <input id="ollama-model-input-2" type="text" bind:value={ollamaModel} placeholder="llama3.1:8b" />
                <span class="hint warning">Ollama is not running. Start it first, or choose Cloud.</span>
              {/if}
            </div>
          {:else if modelProvider === "cloud"}
            <div class="field">
              <label for="cloud-provider">Cloud Provider</label>
              <select id="cloud-provider" bind:value={cloudProvider} onchange={() => (cloudModel = "")}>
                <option value="">Select...</option>
                <option value="openai">OpenAI</option>
                <option value="openrouter">OpenRouter</option>
                <option value="claude">Anthropic (Claude)</option>
                <option value="gemini">Google (Gemini)</option>
              </select>
            </div>
            {#if cloudProvider}
              {#if cloudProvider === "openrouter"}
                <div class="field">
                  <label for="openrouter-model">OpenRouter Model</label>
                  <input
                    id="openrouter-model"
                    type="text"
                    list="setup-openrouter-model-options"
                    bind:value={cloudModel}
                    placeholder="openrouter/auto"
                  />
                  <datalist id="setup-openrouter-model-options">
                    <option value="openrouter/auto"></option>
                    <option value="deepseek/deepseek-v4-pro"></option>
                    <option value="~deepseek/deepseek-v4-flash-latest"></option>
                    <option value="~anthropic/claude-sonnet-latest"></option>
                    <option value="~google/gemini-pro-latest"></option>
                    <option value="~openai/gpt-latest"></option>
                  </datalist>
                  <span class="hint">Choose a suggestion or paste any model slug from the OpenRouter catalog.</span>
                </div>
              {/if}
              <div class="field">
                <label for="cloud-api-key">API Key</label>
                <input
                  id="cloud-api-key"
                  type="password"
                  bind:value={cloudApiKey}
                  placeholder={cloudProvider === "openrouter" ? "sk-or-v1-..." : "sk-..."}
                />
                <span class="hint"
                  >Stored only in Windows Credential Manager, macOS Keychain, or Linux Secret Service.</span
                >
              </div>
            {/if}
          {:else}
            <div class="field">
              <span class="field-label">Subscription Provider</span>
              <div class="radio-group compact">
                <label class="radio-option" class:selected={subscriptionProvider === "codex"}>
                  <input
                    type="radio"
                    name="subscription-provider"
                    checked={subscriptionProvider === "codex"}
                    onchange={() => changeSubscriptionProvider("codex")}
                  />
                  <div>
                    <strong>Codex</strong>
                    <span>ChatGPT plan through the official Codex CLI.</span>
                  </div>
                </label>
                <label class="radio-option" class:selected={subscriptionProvider === "claude"}>
                  <input
                    type="radio"
                    name="subscription-provider"
                    checked={subscriptionProvider === "claude"}
                    onchange={() => changeSubscriptionProvider("claude")}
                  />
                  <div>
                    <strong>Claude Code</strong>
                    <span>Claude subscription through the official Claude Code CLI.</span>
                  </div>
                </label>
              </div>
            </div>
            <div class="field">
              <label for="subscription-model">Model</label>
              <input
                id="subscription-model"
                type="text"
                bind:value={subscriptionModel}
                placeholder={subscriptionProvider === "codex" ? "Default Codex model" : "Default Claude model"}
              />
              <span class="hint">Leave blank for the CLI default or enter a model included by your plan.</span>
            </div>
            <div class="subscription-connection" class:connected={subscriptionConnected}>
              <strong>{subscriptionConnected ? "Connected" : "Connection required"}</strong>
              <span>{subscriptionChecking ? "Checking the official CLI..." : subscriptionMessage}</span>
              <div class="subscription-buttons">
                <button type="button" class="btn-inline" onclick={startSubscriptionLogin}>Connect</button>
                <button type="button" class="btn-inline secondary" onclick={() => refreshSubscriptionStatus(true)}
                  >Refresh</button
                >
              </div>
            </div>
            <p class="note">
              Heliox uses a tool-free, read-only planning process. The official CLI owns credentials and plan limits;
              Heliox does not copy browser sessions or OAuth tokens.
            </p>
          {/if}
        </div>
      {:else if step === 2}
        <div class="wizard-step">
          <h2>Security Boundaries</h2>

          <div class="field">
            <label for="protected-folders">Protected Folders</label>
            <textarea
              id="protected-folders"
              bind:value={protectedFolders}
              placeholder={"~/Documents/private\n~/ssh"}
              rows={4}></textarea>
            <span class="hint">One path per line. Heliox OS will never modify files in these folders.</span>
          </div>

          <div class="field">
            <label for="protected-packages">Protected Packages</label>
            <input
              id="protected-packages"
              type="text"
              bind:value={protectedPackages}
              placeholder="firefox, nautilus, gnome-shell"
            />
            <span class="hint">Comma-separated. Heliox OS will refuse to uninstall these.</span>
          </div>

          <div class="field">
            <label class="checkbox-label">
              <input type="checkbox" checked disabled />
              <span>Root access is <strong>OFF</strong> by default (enable in Settings when needed)</span>
            </label>
          </div>
        </div>
      {:else}
        <div class="wizard-step">
          <h2>All Set</h2>
          <p>Heliox OS is configured and ready to use.</p>
          <div class="summary">
            <div class="summary-item">
              <span class="summary-label">Provider</span>
              <span
                >{modelProvider === "ollama"
                  ? "Ollama"
                  : modelProvider === "cloud"
                    ? `Cloud (${cloudProvider})`
                    : `Subscription (${subscriptionProvider === "codex" ? "Codex" : "Claude Code"})`}</span
              >
            </div>
            <div class="summary-item">
              <span class="summary-label">Model</span>
              <span
                >{modelProvider === "ollama"
                  ? ollamaModel
                  : modelProvider === "cloud"
                    ? cloudModel || cloudProvider
                    : subscriptionModel || "Provider default"}</span
              >
            </div>
            <div class="summary-item">
              <span class="summary-label">Protected Folders</span>
              <span>{protectedFolders.split("\n").filter(Boolean).length} configured</span>
            </div>
            <div class="summary-item">
              <span class="summary-label">Protected Packages</span>
              <span>{protectedPackages.split(",").filter((p) => p.trim()).length} configured</span>
            </div>
            <div class="summary-item">
              <span class="summary-label">Root Access</span>
              <span>Disabled</span>
            </div>
          </div>
          <p class="note">Press Super+J to toggle the Heliox OS window at any time.</p>
        </div>
      {/if}
    </div>

    <div class="wizard-footer">
      {#if finishError}
        <p class="finish-error" role="alert">{finishError}</p>
      {/if}
      {#if step > 0}
        <button class="btn-back" onclick={() => step--}>Back</button>
      {:else}
        <div></div>
      {/if}

      {#if step < steps.length - 1}
        <button class="btn-next" onclick={() => step++}>Continue</button>
      {:else}
        <button class="btn-finish" onclick={finish} disabled={finishing}>
          {finishing ? "Launching..." : "Launch Heliox OS"}
        </button>
      {/if}
    </div>
  </div>
</div>

<style>
  .wizard-overlay {
    position: fixed;
    inset: 0;
    background: var(--bg-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .wizard {
    width: 100%;
    max-width: 560px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  .wizard-header {
    padding: 24px 28px 20px;
    border-bottom: 1px solid var(--border);
  }

  h1 {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 16px;
  }

  .progress {
    display: flex;
    align-items: center;
    gap: 0;
  }

  .progress-step {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .step-num {
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    border-radius: 50%;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    border: 1px solid var(--border);
  }

  .progress-step.active .step-num {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
  }

  .progress-step.done .step-num {
    background: var(--success);
    color: white;
    border-color: var(--success);
  }

  .step-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .progress-step.active .step-label {
    color: var(--text-primary);
    font-weight: 500;
  }

  .progress-line {
    flex: 1;
    height: 1px;
    background: var(--border);
    margin: 0 8px;
  }

  .progress-line.filled {
    background: var(--success);
  }

  .wizard-body {
    flex: 1;
    overflow-y: auto;
    padding: 24px 28px;
  }

  .wizard-step h2 {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .wizard-step p {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    margin-bottom: 10px;
  }

  .wizard-step ul {
    padding-left: 20px;
    margin-bottom: 12px;
  }

  .wizard-step li {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .note {
    font-size: 12px;
    color: var(--text-muted);
    font-style: italic;
  }

  .field {
    margin-bottom: 16px;
  }

  .field label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .field-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .field input[type="text"],
  .field input[type="password"],
  .field select,
  .field textarea {
    width: 100%;
    padding: 8px 12px;
    font-size: 13px;
    background: var(--bg-primary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: inherit;
  }

  .field textarea {
    resize: vertical;
    font-family: var(--font-mono);
    font-size: 12px;
  }

  .field select {
    cursor: pointer;
  }

  .hint {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  .hint code {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
  }

  .hint.warning {
    color: var(--warning);
  }

  .model-status {
    padding: 10px 12px;
    font-size: 13px;
    color: var(--text-muted);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .radio-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .radio-option {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .radio-option.selected {
    border-color: var(--accent);
    background: var(--accent-muted);
  }

  .radio-option input {
    margin-top: 2px;
  }

  .radio-option div {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .radio-option strong {
    font-size: 13px;
  }

  .radio-option span {
    font-size: 11px;
    color: var(--text-muted);
  }

  .radio-group.compact {
    display: grid;
    grid-template-columns: 1fr 1fr;
  }

  .subscription-connection {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 12px;
    padding: 12px;
    color: var(--warning);
    background: var(--bg-primary);
    border: 1px solid var(--warning);
    border-radius: var(--radius-sm);
  }

  .subscription-connection.connected {
    color: var(--success);
    border-color: var(--success);
  }

  .subscription-connection span {
    color: var(--text-muted);
    font-size: 11px;
  }

  .subscription-buttons {
    display: flex;
    gap: 6px;
    margin-top: 4px;
  }

  .btn-inline {
    padding: 5px 10px;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    font-size: 11px;
  }

  .btn-inline.secondary {
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
  }

  .checkbox-label {
    display: flex !important;
    align-items: center;
    gap: 8px;
    cursor: default;
  }

  .summary {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 4px 0;
    margin: 16px 0;
  }

  .summary-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 14px;
    font-size: 13px;
  }

  .summary-label {
    color: var(--text-muted);
  }

  .wizard-footer {
    display: flex;
    align-items: center;
    gap: 12px;
    justify-content: space-between;
    padding: 16px 28px;
    border-top: 1px solid var(--border);
  }

  .finish-error {
    margin: 0 auto 0 0;
    max-width: 58%;
    color: var(--danger, #ef4444);
    font-size: 11px;
    line-height: 1.35;
  }

  .btn-back {
    padding: 8px 20px;
    font-size: 13px;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .btn-back:hover {
    background: var(--bg-hover);
  }

  .btn-next,
  .btn-finish {
    padding: 8px 24px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
  }

  .btn-next:hover,
  .btn-finish:hover {
    background: var(--accent-hover);
  }
</style>
