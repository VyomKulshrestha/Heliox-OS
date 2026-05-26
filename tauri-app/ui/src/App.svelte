<script lang="ts">
  import CommandInput from "./lib/components/CommandInput.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import { copyMessage as utilCopyMessage } from "./lib/utils/copy";
  import { getJarvisGreeting } from "./lib/utils/greeting";
  import { renderMarkdown } from "./lib/utils/markdown";
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
  import ExecutionGraph from "./lib/components/ExecutionGraph.svelte";
  import ReActPipeline from "./lib/components/ReActPipeline.svelte";
  import VirtualList from "./lib/components/VirtualList.svelte";
  import PluginsTab from "./lib/components/PluginsTab.svelte";
  import GitConflictResolver from "./lib/components/GitConflictResolver.svelte";
  import { session } from "./lib/stores/session";
  import type { Message } from "./lib/stores/session";
  import { settings } from "./lib/stores/settings";
   import { tick, onDestroy } from "svelte";
  import { Copy } from "lucide-svelte";

  import ScrollToBottom from "./lib/components/ScrollToBottom.svelte";
  import ConnectionStatus from "./lib/components/ConnectionStatus.svelte";
  let isDragging = $state(false);
  import { _, isLoading } from 'svelte-i18n';

  let activeTab: "chat" | "log" | "dashboard" | "settings" | "plugins" = $state("chat");
  let showWizard = $derived(
    !$settings.first_run_complete && localStorage.getItem("heliox_first_run_complete") !== "true"
  );
  let showScrollFAB = $state(false);
  let isAtBottom = $state(true);
  let virtualListEl: VirtualList<Message> | undefined = $state();
  let particleBurst: ParticleBurst | undefined = $state();
  $effect(() => {
    showScrollFAB = !isAtBottom;
  });
  async function onSetupComplete() {
    await settings.updateSection("", { first_run_complete: true });
    await tick();
    session.addSystemMessage(getJarvisGreeting());
  }

  // Handle gesture events for particle effects and navigation
  function onGestureDetected(gesture: string) {
    particleBurst?.gestureBurst(gesture);
    if (gesture === "swipe_left" || gesture === "two_finger_swipe_left") {
      if (activeTab === "settings") activeTab = "log";
      else if (activeTab === "log") activeTab = "chat";
    } else if (gesture === "swipe_right" || gesture === "two_finger_swipe_right") {
      if (activeTab === "chat") activeTab = "log";
      else if (activeTab === "log") activeTab = "settings";
    } else if (gesture === "call_me") {
      activeTab = "settings";
    }
  }
  function scrollToBottom() {
    tick().then(() => {
      virtualListEl?.scrollToBottom();
      showScrollFAB = false;
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
  function actionLabel(action: { action_type: string; dry_run?: boolean }, planDryRun = false): string {
    return action.dry_run || planDryRun ? `${formatActionType(action.action_type)} (dry run)` : formatActionType(action.action_type);
  }
  function tierClass(action: { requires_root?: boolean; destructive?: boolean }): string {
    if (action.requires_root) return "tier-root";
    if (action.destructive) return "tier-destructive";
    return "tier-safe";
  }

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
  let copiedMessageId = $state<number | null>(null);

  let copiedTimeout: ReturnType<typeof setTimeout> | null = null;

  async function copyMessage(msg: Message) {
    if (await utilCopyMessage(msg)) {
      copiedMessageId = msg.timestamp;
      if (copiedTimeout) clearTimeout(copiedTimeout);
      copiedTimeout = setTimeout(() => {
        copiedMessageId = null;
        copiedTimeout = null;
      }, 1500);
    }
  }

  onDestroy(() => {
    if (copiedTimeout) clearTimeout(copiedTimeout);
  });
  function exportReActTrace() {
    const traceSteps = $session.messages
      .filter(m => m.type === "plan" || m.type === "result" || m.type === "error")
      .map(m => ({
        type: m.type,
        timestamp: m.timestamp,
        ...(m.plan && { plan: m.plan }),
        ...(m.actionResults && { actionResults: m.actionResults }),
        ...(m.verification && { verification: m.verification }),
        ...(m.text && { text: m.text })
      }));

    if (traceSteps.length === 0) {
      alert("No ReAct trace steps found. Run a command first!");
      return;
    }

    const exportData = {
      exported_at: new Date().toISOString(),
      version: "1.0",
      total_steps: traceSteps.length,
      steps: traceSteps
    };

    const blob = new Blob(
      [JSON.stringify(exportData, null, 2)],
      { type: "application/json" }
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `react_trace_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
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
    onmousedown={() => {isDragging = true}}
    onmouseup={() => {isDragging = false}}
  >
    <div class="titlebar-left">
      <ArcReactor />
      <span class="title">{$_('app.title')}</span>
      <span class="badge" class:connected={$session.daemonConnected}>
        {$session.daemonConnected ? $_('app.online') : $_('app.connecting')}
      </span>
    </div>
    <nav class="tabs">
      <button class="tab" class:active={activeTab === "chat"} title="Open Command Panel" onclick={() => activeTab = "chat"}>{$_('app.tab_command')}</button>
      <button class="tab" class:active={activeTab === "log"} title="Open activity log" onclick={() => activeTab = "log"}>{$_('app.tab_activity')}</button>
      <button class="tab" class:active={activeTab === "dashboard"} title="Open Dashboard" onclick={() => activeTab = "dashboard"}>Dashboard</button>
      <button class="tab" class:active={activeTab === "plugins"} title="Browse plugin marketplace" onclick={() => activeTab = "plugins"}>{$_('app.tab_plugins')}</button>
      <button class="tab" class:active={activeTab === "settings"} title="Open Settings" onclick={() => activeTab = "settings"}>{$_('app.tab_settings')}</button>
    </nav>
    <div class="titlebar-right">
      <ConnectionStatus />
      <AmbientHUD />
    </div>
  </header>

  <div class="content">
    {#if activeTab === "chat"}
      <div class="chat-panel">
        <NeuralBackground />

        <div class="pipeline-container">
          <ReActPipeline />
        </div>
        {#if $session.confirmRequired}
          <ConfirmDialog
            actions={$session.confirmActions}
            onconfirm={() => session.confirm(true)}
            ondeny={() => session.confirm(false)}
          />
        {/if}

        <div class="results">
          {#if $session.messages.length === 0 && !$session.loading}
            <div class="empty-state">
              <div class="empty-logo">C</div>
              <h2>{$_('chat.empty_title')}</h2>
              <p>{$_('chat.empty_subtitle')}</p>
              <div class="suggestions">
                <button class="suggestion" onclick={() => session.sendCommand("Show system information")}>{$_('chat.suggestion_sysinfo')}</button>
                <button class="suggestion" onclick={() => session.sendCommand("Take a screenshot and describe it")}>{$_('chat.suggestion_screenshot')}</button>
                <button class="suggestion" onclick={() => session.sendCommand("What processes are running?")}>{$_('chat.suggestion_processes')}</button>
              </div>
            </div>
          {:else}
            <VirtualList bind:this={virtualListEl} items={$session.messages} bind:atBottom={isAtBottom}>
              {#snippet item(msg)}
                {@render messageBlock(msg)}
              {/snippet}
              {#snippet footer()}
                {#if $session.loading}
                  <ExecutionGraph />
                  {#if $session.streamingText}
                    <div class="message system streaming">
                      <div class="msg-header">
                        <span class="msg-label">HELIOX</span>
                        <span class="phase-badge">{$_('chat.streaming')}</span>
                      </div>
                      <span class="msg-text">{$session.streamingText}</span>
                    </div>
                  {:else}
                    <div class="message system">
                      <div class="msg-header">
                        <span class="msg-label">HELIOX</span>
                        <span class="phase-badge">{$session.phase || $_('chat.thinking')}</span>
                      </div>
                      <span class="msg-text loading-dots">
                        {$session.phase ? `${$session.phase}` : $_('chat.thinking')}
                      </span>
                    </div>
                  {/if}
                {/if}
              {/snippet}
            </VirtualList>
          {/if}
          <ScrollToBottom show={showScrollFAB} onclick={scrollToBottom} />
        </div>
        <div class="input-row">
          <VoiceControl />
          <CommandInput />
          <GestureControl onGesture={onGestureDetected} />
          <button class="tab" type="button" onclick={() => session.exportChat("json")}>{$_('app.export_json')}</button>
          <button class="tab" type="button" onclick={() => session.exportChat("csv")}>{$_('app.export_csv')}</button>
          <button class="tab" type="button" onclick={() => session.exportChat("json")}>Export JSON</button>
          <button class="tab" type="button" onclick={() => session.exportChat("csv")}>Export CSV</button>
          <button class="tab" type="button" onclick={exportReActTrace} title="Export ReAct reasoning trace to JSON">Export Trace</button>
        </div>
      </div>
    {:else if activeTab === "log"}
      <ActivityLog />
    {:else if activeTab === "dashboard"}
     <Dashboard />
    {:else if activeTab === "plugins"}
      <PluginsTab />
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
    <div class="message plan-msg has-copy">
      <div class="msg-header">
        <span class="msg-label">{$_('plan.label')}</span>
        <span class="phase-badge">{msg.plan.dry_run ? $_('plan.dry_run') : $_('plan.planning')}</span>
      </div>
      {#if msg.plan.explanation}
        <p class="plan-explanation">{msg.plan.explanation}</p>
      {/if}
      <div class="action-list">
        {#each msg.plan.actions as action, i}
          <div class="action-item">
            <span class="action-index">{i + 1}</span>
            <div class="action-detail">
              <span class="action-type">{actionLabel(action, Boolean(msg.plan.dry_run))}</span>
              <span class="action-target">{action.target}</span>
            </div>
            <span class="tier-badge {tierClass(action)}">{tierLabel(action)}</span>
          </div>
        {/each}
      </div>
      <button class="copy-button" type="button" aria-label="Copy message" title="Copy" onclick={() => copyMessage(msg)}>
        <Copy size={14} />
      </button>
      <span class="copy-feedback" role="status" aria-live="polite" aria-atomic="true" class:active={copiedMessageId === msg.timestamp}>
        {$_('chat.copied')}
      </span>
    </div>

  {:else if msg.type === "result"}
    <div class="message result-msg has-copy">
      <div class="msg-header">
        <span class="msg-label">{$_('result.label')}</span>
        {#if msg.verification}
          <span class="status-badge" class:passed={msg.verification.passed} class:failed={!msg.verification.passed}>
            {msg.verification.passed ? $_('result.verified') : $_('result.issues')}
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
                {ar.success ? $_('result.ok') : $_('result.failed')}
              </span>
            </div>
            {#if ar.success && ar.output}
              <div class="ar-output">
                {@html renderMarkdown(ar.output.trim())}
              </div>
            {/if}
            {#if !ar.success && ar.error}
              <pre class="ar-error">{ar.error}</pre>
            {/if}
          </div>
        {/each}
      {:else}
        <span class="msg-text">{msg.text || $_('result.done')}</span>
      {/if}
      {#if msg.verification && msg.verification.details.length > 0}
        <div class="verification-section">
          <span class="verification-label">{$_('result.verification')}</span>
          {#each msg.verification.details as detail}
            <span class="verification-detail" class:v-pass={detail.includes("VERIFIED")} class:v-fail={detail.includes("FAILED") || detail.includes("MISMATCH")}>
              {detail}
            </span>
          {/each}
        </div>
      {/if}
      <button class="copy-button" type="button" aria-label="Copy message" title="Copy" onclick={() => copyMessage(msg)}>
        <Copy size={14} />
      </button>
      <span class="copy-feedback" role="status" aria-live="polite" aria-atomic="true" class:active={copiedMessageId === msg.timestamp}>
        {$_('chat.copied')}
      </span>
    </div>
  {:else if msg.type === "error"}
    <div class="message error-msg has-copy">
      <span class="msg-label">{$_('error.label')}</span>
      <span class="msg-text">{msg.text}</span>
      <button class="copy-button" type="button" aria-label="Copy message" title="Copy" onclick={() => copyMessage(msg)}>
        <Copy size={14} />
      </button>
      <span class="copy-feedback" role="status" aria-live="polite" aria-atomic="true" class:active={copiedMessageId === msg.timestamp}>
        {$_('chat.copied')}
      </span>
    </div>

  {:else if msg.type === "git_conflict" && msg.gitConflict}
    <GitConflictResolver payload={msg.gitConflict} />

  {:else}
    <div class="message system-msg has-copy">
      <span class="msg-label">HELIOX</span>
      <div class="msg-text">
        {@html renderMarkdown(msg.text || "")}
      </div>
      <button class="copy-button" type="button" aria-label="Copy message" title="Copy" onclick={() => copyMessage(msg)}>
        <Copy size={14} />
      </button>
      <span class="copy-feedback" role="status" aria-live="polite" aria-atomic="true" class:active={copiedMessageId === msg.timestamp}>
        {$_('chat.copied')}
      </span>
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
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
  }

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

  .message {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    animation: fadeIn 0.2s ease-out;
  }

  .message.has-copy {
    position: relative;
    padding-right: 38px;
  }

  .copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    opacity: 0;
    transform: translateY(-2px);
    pointer-events: none;
    transition: opacity 0.15s ease, transform 0.15s ease, background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  }

  .message.has-copy:hover .copy-button,
  .message.has-copy:focus-within .copy-button {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
  }

  .copy-button:hover {
    background: var(--bg-hover);
    border-color: var(--border);
    color: var(--text-primary);
  }

  .copy-button:active {
    transform: translateY(0) scale(0.98);
  }

  .copy-feedback {
    position: absolute;
    top: 10px;
    right: 36px;
    font-size: 10px;
    font-weight: 600;
    color: var(--success);
    background: rgba(74, 222, 128, 0.12);
    padding: 2px 6px;
    border-radius: 10px;
    opacity: 0;
    transform: translateY(-2px);
    transition: opacity 0.15s ease, transform 0.15s ease;
    pointer-events: none;
  }

  .copy-feedback.active {
    opacity: 1;
    transform: translateY(0);
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

  .user-msg { background: var(--bg-secondary); }
  .system-msg { background: var(--bg-tertiary); }

  .error-msg {
    border-color: var(--danger);
    background: var(--danger-bg);
  }

  .error-msg .msg-label { color: var(--danger); }

  .message.system.streaming {
    background: var(--bg-tertiary);
    border-left: 3px solid var(--accent);
    animation: pulse-glow 2s ease-in-out infinite;
  }

  @keyframes pulse-glow {
    0%, 100% { border-left-color: var(--accent); }
    50% { border-left-color: var(--accent-hover); }
  }

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

  .status-badge.passed { background: rgba(74, 222, 128, 0.1); color: var(--success); }
  .status-badge.failed { background: var(--danger-bg); color: var(--danger); }

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

  .loading-dots::after {
    content: "";
    animation: dots 1.5s infinite;
  }

  @keyframes dots {
    0%, 20% { content: "."; }
    40% { content: ".."; }
    60%, 100% { content: "..."; }
  }

  .hlx-code-wrapper {
    margin: 8px 0;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border);
    background: #1a1a24;
  }

  .hlx-code-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 10px;
    background: rgba(0,0,0,0.3);
    border-bottom: 1px solid var(--border);
    min-height: 26px;
  }

  .hlx-lang-badge {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    opacity: 0.85;
  }

  .hlx-copy-btn {
    font-size: 10px;
    padding: 2px 8px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    font-family: var(--font-sans);
    transition: all 0.15s;
  }

  .hlx-copy-btn:hover {
    background: var(--accent-muted);
    border-color: var(--accent);
    color: var(--accent);
  }

  .hlx-pre {
    margin: 0;
    padding: 12px 14px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.6;
    background: transparent;
  }

  .hlx-pre code {
    font-family: var(--font-mono);
    background: none;
    border: none;
    padding: 0;
  }
</style>