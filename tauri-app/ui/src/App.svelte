<script lang="ts">
  import CommandInput from "./lib/components/CommandInput.svelte";
  import ConfirmDialog from "./lib/components/ConfirmDialog.svelte";
  import ActivityLog from "./lib/components/ActivityLog.svelte";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import SetupWizard from "./lib/components/SetupWizard.svelte";
  import VoiceControl from "./lib/components/VoiceControl.svelte";
  import GestureControl from "./lib/components/GestureControl.svelte";
  import AmbientHUD from "./lib/components/AmbientHUD.svelte";
  import ArcReactor from "./lib/components/ArcReactor.svelte";
  import NeuralBackground from "./lib/components/NeuralBackground.svelte";
  import ParticleBurst from "./lib/components/ParticleBurst.svelte";
  import { session } from "./lib/stores/session";
  import type { Message } from "./lib/stores/session";
  import { settings } from "./lib/stores/settings";
  import { tick } from "svelte";

  let activeTab: "chat" | "log" | "settings" = $state("chat");
  let isDragging = $state(false);
  let showWizard = $derived(!$settings.first_run_complete);
  let resultsEl: HTMLDivElement | undefined = $state();
  let particleBurst: ParticleBurst | undefined = $state();

  async function onSetupComplete() {
    await settings.updateSection("", { first_run_complete: true });
    await tick();
    session.addSystemMessage(getJarvisGreeting());
  }

  // JARVIS-style proactive greeting based on time
  function getJarvisGreeting(): string {
    const hour = new Date().getHours();
    if (hour < 6) return "Good evening. Cortex-OS is online. All systems nominal.";
    if (hour < 12) return "Good morning. Cortex-OS is online and ready for your commands.";
    if (hour < 17) return "Good afternoon. Cortex-OS at your service. What shall we tackle?";
    if (hour < 21) return "Good evening. Cortex-OS is ready. How can I assist you tonight?";
    return "Burning the midnight oil? Cortex-OS is standing by.";
  }

  // Handle gesture events for particle effects
  function onGestureDetected(gesture: string) {
    particleBurst?.gestureBurst(gesture);
  }

  function scrollToBottom() {
    tick().then(() => {
      if (resultsEl) resultsEl.scrollTop = resultsEl.scrollHeight;
    });
  }

  $effect(() => {
    $session.messages;
    $session.loading;
    scrollToBottom();
  });

  function formatActionType(t: string): string {
    return t.replace(/_/g, " ");
  }

  function tierLabel(action: { requires_root?: boolean; destructive?: boolean }): string {
    if (action.requires_root) return "ROOT";
    if (action.destructive) return "DESTRUCTIVE";
    return "SAFE";
  }

  function tierClass(action: { requires_root?: boolean; destructive?: boolean }): string {
    if (action.requires_root) return "tier-root";
    if (action.destructive) return "tier-destructive";
    return "tier-safe";
  }
  // Trigger particle burst on new results
  let prevMsgLen = 0;
  $effect(() => {
    const msgs = $session.messages;
    if (msgs.length > prevMsgLen) {
      const last = msgs[msgs.length - 1];
      if (last.type === "result") particleBurst?.confirmBurst();
      else if (last.type === "error") particleBurst?.errorBurst();
    }
    prevMsgLen = msgs.length;
  });
</script>

{#if showWizard}
  <SetupWizard oncomplete={onSetupComplete} />
{/if}

<main
  class="window"
  class:dragging={isDragging}
  class:hidden-behind-wizard={showWizard}
>
  <header
    class="titlebar"
    data-tauri-drag-region
    onmousedown={() => isDragging = true}
    onmouseup={() => isDragging = false}
  >
    <div class="titlebar-left">
      <ArcReactor />
      <span class="title">Cortex-OS</span>
      <span class="badge" class:connected={$session.daemonConnected}>
        {$session.daemonConnected ? "Online" : "Connecting..."}
      </span>
    </div>
    <nav class="tabs">
      <button class="tab" class:active={activeTab === "chat"} onclick={() => activeTab = "chat"}>Command</button>
      <button class="tab" class:active={activeTab === "log"} onclick={() => activeTab = "log"}>Activity</button>
      <button class="tab" class:active={activeTab === "settings"} onclick={() => activeTab = "settings"}>Settings</button>
    </nav>
    <div class="titlebar-right">
      <AmbientHUD />
    </div>
  </header>

  <div class="content">
    {#if activeTab === "chat"}
      <div class="chat-panel">
        <NeuralBackground />
        {#if $session.confirmRequired}
          <ConfirmDialog
            actions={$session.confirmActions}
            onconfirm={() => session.confirm(true)}
            ondeny={() => session.confirm(false)}
          />
        {/if}

        <div class="results" bind:this={resultsEl}>
          {#if $session.messages.length === 0 && !$session.loading}
            <div class="empty-state">
              <div class="empty-logo">C</div>
              <h2>Cortex-OS</h2>
              <p>Your AI system control agent. Type, speak, or gesture — I'm ready.</p>
              <div class="suggestions">
                <button class="suggestion" onclick={() => session.sendCommand("Show system information")}>Show system info</button>
                <button class="suggestion" onclick={() => session.sendCommand("Take a screenshot and describe it")}>Screenshot + OCR</button>
                <button class="suggestion" onclick={() => session.sendCommand("What processes are running?")}>List processes</button>
              </div>
            </div>
          {:else}
            {#each $session.messages as msg (msg.timestamp)}
              {@render messageBlock(msg)}
            {/each}

            {#if $session.loading}
              <div class="message system">
                <div class="msg-header">
                  <span class="msg-label">CORTEX</span>
                  <span class="phase-badge">{$session.phase || "thinking"}</span>
                </div>
                <span class="msg-text loading-dots">
                  {$session.phase ? `${$session.phase}` : "Thinking"}
                </span>
              </div>
            {/if}
          {/if}
        </div>

        <div class="input-row">
          <VoiceControl />
          <CommandInput />
          <GestureControl onGesture={onGestureDetected} />
        </div>
      </div>
    {:else if activeTab === "log"}
      <ActivityLog />
    {:else}
      <SettingsPanel />
    {/if}
  </div>

  <!-- Particle Burst Overlay -->
  <ParticleBurst bind:this={particleBurst} />
</main>

{#snippet messageBlock(msg: Message)}
  {#if msg.type === "user"}
    <div class="message user-msg">
      <span class="msg-label">YOU</span>
      <span class="msg-text">{msg.text}</span>
    </div>

  {:else if msg.type === "plan" && msg.plan}
    <div class="message plan-msg">
      <div class="msg-header">
        <span class="msg-label">PLAN</span>
        <span class="phase-badge">planning</span>
      </div>
      {#if msg.plan.explanation}
        <p class="plan-explanation">{msg.plan.explanation}</p>
      {/if}
      <div class="action-list">
        {#each msg.plan.actions as action, i}
          <div class="action-item">
            <span class="action-index">{i + 1}</span>
            <div class="action-detail">
              <span class="action-type">{formatActionType(action.action_type)}</span>
              <span class="action-target">{action.target}</span>
            </div>
            <span class="tier-badge {tierClass(action)}">{tierLabel(action)}</span>
          </div>
        {/each}
      </div>
    </div>

  {:else if msg.type === "result"}
    <div class="message result-msg">
      <div class="msg-header">
        <span class="msg-label">RESULT</span>
        {#if msg.verification}
          <span class="status-badge" class:passed={msg.verification.passed} class:failed={!msg.verification.passed}>
            {msg.verification.passed ? "Verified" : "Issues detected"}
          </span>
        {/if}
      </div>

      {#if msg.actionResults && msg.actionResults.length > 0}
        {#each msg.actionResults as ar, i}
          <div class="action-result" class:action-success={ar.success} class:action-failure={!ar.success}>
            <div class="ar-header">
              <span class="ar-type">{formatActionType(ar.action_type)}</span>
              {#if ar.target}
                <code class="ar-target">{ar.target}</code>
              {/if}
              <span class="ar-status" class:success={ar.success} class:failure={!ar.success}>
                {ar.success ? "OK" : "FAILED"}
              </span>
            </div>
            {#if ar.success && ar.output}
              <pre class="ar-output">{ar.output.trim()}</pre>
            {/if}
            {#if !ar.success && ar.error}
              <pre class="ar-error">{ar.error}</pre>
            {/if}
          </div>
        {/each}
      {:else}
        <span class="msg-text">{msg.text || "Done."}</span>
      {/if}

      {#if msg.verification && msg.verification.details.length > 0}
        <div class="verification-section">
          <span class="verification-label">Verification</span>
          {#each msg.verification.details as detail}
            <span class="verification-detail" class:v-pass={detail.includes("VERIFIED")} class:v-fail={detail.includes("FAILED") || detail.includes("MISMATCH")}>
              {detail}
            </span>
          {/each}
        </div>
      {/if}
    </div>

  {:else if msg.type === "error"}
    <div class="message error-msg">
      <span class="msg-label">ERROR</span>
      <span class="msg-text">{msg.text}</span>
    </div>

  {:else}
    <div class="message system-msg">
      <span class="msg-label">CORTEX</span>
      <span class="msg-text">{msg.text}</span>
    </div>
  {/if}
{/snippet}

<style>
  .window {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  .window.hidden-behind-wizard {
    visibility: hidden;
  }

  .titlebar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    user-select: none;
    -webkit-app-region: drag;
  }

  .titlebar-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .titlebar-right {
    display: flex;
    align-items: center;
    gap: 6px;
    -webkit-app-region: no-drag;
  }

  .logo {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #00c8ff, #7c3aed);
    color: white;
    font-weight: 700;
    font-size: 13px;
    border-radius: var(--radius-sm);
    box-shadow: 0 0 10px rgba(0, 200, 255, 0.3);
  }

  .title { font-weight: 600; font-size: 14px; letter-spacing: 0.5px; }

  .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 20px;
    background: var(--danger-bg);
    color: var(--danger);
    transition: all 0.3s;
  }

  .badge.connected {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .tabs {
    display: flex;
    gap: 2px;
    -webkit-app-region: no-drag;
  }

  /* Input Row — voice + text + gesture */
  .input-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
    position: relative;
  }

  .tab {
    padding: 5px 14px;
    font-size: 12px;
    color: var(--text-secondary);
    background: transparent;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .tab:hover { color: var(--text-primary); background: var(--bg-hover); }
  .tab.active { color: var(--accent); background: var(--accent-muted); }

  .content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }

  .results {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* Empty state */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    text-align: center;
    padding: 32px 24px;
  }

  .empty-logo {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--accent);
    color: white;
    font-weight: 700;
    font-size: 22px;
    border-radius: 12px;
    margin-bottom: 4px;
  }

  .empty-state h2 { font-size: 18px; font-weight: 700; color: var(--text-primary); }
  .empty-state p { font-size: 13px; color: var(--text-muted); max-width: 320px; line-height: 1.5; }

  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
    margin-top: 8px;
  }

  .suggestion {
    padding: 6px 14px;
    font-size: 12px;
    color: var(--accent);
    background: var(--accent-muted);
    border-radius: 20px;
    transition: all 0.15s;
  }

  .suggestion:hover { background: var(--accent); color: white; }

  /* All messages */
  .message {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    animation: fadeIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .msg-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .msg-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .msg-text {
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* User messages */
  .user-msg {
    background: var(--bg-secondary);
  }

  /* System messages */
  .system-msg {
    background: var(--bg-tertiary);
  }

  /* Error messages */
  .error-msg {
    border-color: var(--danger);
    background: var(--danger-bg);
  }

  .error-msg .msg-label { color: var(--danger); }

  /* Plan messages */
  .plan-msg {
    background: var(--bg-tertiary);
    border-color: var(--accent-muted);
  }

  .plan-msg .msg-label { color: var(--accent); }

  .phase-badge {
    font-size: 10px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    background: var(--accent-muted);
    color: var(--accent);
    text-transform: lowercase;
  }

  .plan-explanation {
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.4;
    margin: 0;
  }

  .action-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin-top: 4px;
  }

  .action-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 5px 8px;
    border-radius: var(--radius-sm);
  }

  .action-item:hover { background: var(--bg-hover); }

  .action-index {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    background: var(--bg-primary);
    border-radius: 50%;
    flex-shrink: 0;
  }

  .action-detail {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .action-type {
    font-size: 12px;
    font-weight: 500;
    font-family: var(--font-mono);
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .action-target {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .tier-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    flex-shrink: 0;
  }

  .tier-safe { background: rgba(74, 222, 128, 0.1); color: var(--success); }
  .tier-destructive { background: rgba(251, 191, 36, 0.1); color: var(--warning); }
  .tier-root { background: var(--danger-bg); color: var(--danger); }

  /* Result messages */
  .result-msg {
    background: var(--bg-secondary);
    gap: 8px;
  }

  .result-msg .msg-label { color: var(--success); }

  .status-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
  }

  .status-badge.passed {
    background: rgba(74, 222, 128, 0.1);
    color: var(--success);
  }

  .status-badge.failed {
    background: var(--danger-bg);
    color: var(--danger);
  }

  .action-result {
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    overflow: hidden;
  }

  .ar-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
  }

  .ar-type {
    font-size: 12px;
    font-weight: 600;
    font-family: var(--font-mono);
    color: var(--text-primary);
    text-transform: capitalize;
  }

  .ar-target {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-muted);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .ar-status {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 8px;
    border-radius: 10px;
    flex-shrink: 0;
  }

  .ar-status.success { background: rgba(74, 222, 128, 0.15); color: var(--success); }
  .ar-status.failure { background: var(--danger-bg); color: var(--danger); }

  .ar-output {
    padding: 8px 10px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--text-primary);
    background: var(--bg-primary);
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.5;
  }

  .ar-error {
    padding: 8px 10px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--danger);
    background: var(--danger-bg);
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
  }

  /* Verification section */
  .verification-section {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
  }

  .verification-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 2px;
  }

  .verification-detail {
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    padding: 2px 0;
  }

  .verification-detail.v-pass { color: var(--success); }
  .verification-detail.v-fail { color: var(--danger); }

  /* Loading */
  .loading-dots::after {
    content: "";
    animation: dots 1.5s infinite;
  }

  @keyframes dots {
    0%, 20% { content: "."; }
    40% { content: ".."; }
    60%, 100% { content: "..."; }
  }
</style>
