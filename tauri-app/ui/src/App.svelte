<script lang="ts">
  import CommandInput from "./lib/components/CommandInput.svelte";
  import Dashboard from "./lib/components/Dashboard.svelte";
  import { copyMessage as utilCopyMessage } from "./lib/utils/copy";
  import { getJarvisGreeting } from "./lib/utils/greeting";
  import { renderMarkdown } from "./lib/utils/markdown";
  import ConfirmDialog from "./lib/components/ConfirmDialog.svelte";
  import RollbackDialog from "./lib/components/RollbackDialog.svelte";
  import UndoDialog from "./lib/components/UndoDialog.svelte";
  import BudgetExceededDialog from "./lib/components/BudgetExceededDialog.svelte";
  import InterruptDialog from "./lib/components/InterruptDialog.svelte";
  import SupervisionAlertDialog from "./lib/components/SupervisionAlertDialog.svelte";
  import HumanControlBanner from "./lib/components/HumanControlBanner.svelte";
  import ActivityLog from "./lib/components/ActivityLog.svelte";
  import TodayPanel from "./lib/components/TodayPanel.svelte";
  import FocusBanner from "./lib/components/FocusBanner.svelte";
  import CommandPalette from "./lib/components/CommandPalette.svelte";
  import { onNotification, offNotification } from "./lib/api/daemon";
  import SettingsPanel from "./lib/components/SettingsPanel.svelte";
  import SetupWizard from "./lib/components/SetupWizard.svelte";
  import VoiceControl from "./lib/components/VoiceControl.svelte";
  import GestureControl from "./lib/components/GestureControl.svelte";
  import AirHandoffControl from "./lib/components/AirHandoffControl.svelte";
  import AmbientHUD from "./lib/components/AmbientHUD.svelte";
  import ArcReactor from "./lib/components/ArcReactor.svelte";
  import NeuralBackground from "./lib/components/NeuralBackground.svelte";
  import ParticleBurst from "./lib/components/ParticleBurst.svelte";
  import RunStatus from "./lib/components/RunStatus.svelte";
  import AgentRunCard from "./lib/components/AgentRunCard.svelte";
  import VirtualList from "./lib/components/VirtualList.svelte";
  import PluginsTab from "./lib/components/PluginsTab.svelte";
  import GitConflictResolver from "./lib/components/GitConflictResolver.svelte";
  import { airHandoff } from "./lib/stores/airHandoff";
  import { session } from "./lib/stores/session";
  import type { Message } from "./lib/stores/session";
  import { settings } from "./lib/stores/settings";
  import { tick, onDestroy, onMount, untrack } from "svelte";
  import { Circle, CircleCheck, CircleX, Copy, LoaderCircle, ShieldAlert } from "@lucide/svelte";
  import heliosMark from "./assets/heliox-mark.webp";
  import { actionTitle } from "./lib/utils/phase";
  import type { LiveActionState, PlanAction } from "./lib/stores/session";
  import ScrollToBottom from "./lib/components/ScrollToBottom.svelte";
  import { shouldFollowLatest } from "./lib/utils/scrollPolicy";
  import { selectActiveModal } from "./lib/utils/modalPriority";
  import ConnectionBadge from "./lib/components/ConnectionBadge.svelte";
  import HeaderMiniMonitor from "./lib/components/HeaderMiniMonitor.svelte";
  import CommandHistory from "./lib/components/CommandHistory.svelte";
  import ChatSessionControls from "./lib/components/ChatSessionControls.svelte";
  import InteractiveServices from "./lib/components/InteractiveServices.svelte";
  import NeuralControlOverlay from "./lib/components/NeuralControlOverlay.svelte";
  import SelectionStudy from "./lib/components/coadaptation/SelectionStudy.svelte";
  import { narration } from "./lib/stores/narration";
  import { neural } from "./lib/stores/neural";
  import { supervision } from "./lib/stores/supervision";
  import { coadaptation, studyActive } from "./lib/stores/coadaptation";
  import { _, isLoading } from "svelte-i18n";

  let isDragging = $state(false);
  let activeTab: "chat" | "today" | "log" | "dashboard" | "settings" | "plugins" | "study" = $state("chat");
  let showWizard = $derived(
    !$settings.first_run_complete && localStorage.getItem("heliox_first_run_complete") !== "true",
  );
  let showScrollFAB = $state(false);
  let isAtBottom = $state(true);
  let prefillText = $state("");
  let paletteOpen = $state(false);
  let cameraControlsActive = $state(false);
  let virtualListEl: VirtualList<Message> | undefined = $state();
  let particleBurst: ParticleBurst | undefined = $state();
  let daemonConnectionEpoch = $state(0);
  let previousDaemonConnection = $state(false);
  let activeModal = $derived(
    selectActiveModal({
      confirmation: $session.confirmRequired,
      rollback: $session.rollbackPending,
      undo: Boolean($session.undoPreview),
      interrupt: $narration.active,
      neural: Boolean($neural.preview),
      budget: Boolean($session.budget?.exceeded),
      supervision: $supervision.active,
    }),
  );
  let blockingSafetyDecision = $derived(
    activeModal === "confirmation" ||
      activeModal === "rollback" ||
      activeModal === "undo" ||
      activeModal === "interrupt",
  );
  let documentVisible = $state(typeof document === "undefined" || document.visibilityState === "visible");
  let previousStudyActive = false;

  $effect(() => {
    const connected = $session.daemonConnected;
    if (connected && !previousDaemonConnection) {
      daemonConnectionEpoch += 1;
      void coadaptation.load();
    }
    previousDaemonConnection = connected;
  });
  onMount(() => {
    void coadaptation.load();
  });
  // Ctrl+K (Cmd+K on a Mac) and "what can I say?" open the command palette, except over an approval.
  function openPalette() {
    if ($session.confirmRequired) return;
    activeTab = "chat"; // the palette sits over the Command tab, where its choice lands
    paletteOpen = true;
  }
  function onPaletteKey(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (paletteOpen) paletteOpen = false;
      else openPalette();
    }
  }
  function onPaletteNotification(method: string) {
    if (method === "open_command_palette") openPalette();
  }
  onMount(() => {
    window.addEventListener("keydown", onPaletteKey);
    onNotification(onPaletteNotification);
    return () => {
      window.removeEventListener("keydown", onPaletteKey);
      offNotification(onPaletteNotification);
    };
  });
  async function pickFromPalette(text: string) {
    paletteOpen = false;
    activeTab = "chat";
    await handleReplay(text);
  }
  // The study tab exists only while a study session is open: open it when the session starts, and leave it for
  // Settings (where the study is started) when the session ends.
  $effect(() => {
    const active = $studyActive;
    untrack(() => {
      if (active && !previousStudyActive) activeTab = "study";
      else if (!active && previousStudyActive && activeTab === "study") activeTab = "settings";
      previousStudyActive = active;
    });
  });
  // An approval, rollback, narration interrupt or any other App modal takes the person from the study to the
  // Command tab, where the modal is shown; the study is paused through its `paused` prop. From any other tab App
  // does not switch, so unsaved Settings edits are not lost.
  $effect(() => {
    if (activeTab === "study" && activeModal !== null) activeTab = "chat";
  });
  // Gestures may approve only a dialog the person can see (spec 17.2 F2).
  $effect(() => {
    session.setConfirmVisible(activeTab === "chat" && activeModal === "confirmation" && documentVisible && !showWizard);
  });
  $effect(() => {
    showScrollFAB = !isAtBottom;
  });
  $effect(() => {
    if (blockingSafetyDecision && $neural.preview) void neural.deferPreviewForSafetyDecision();
  });
  async function onSetupComplete() {
    await settings.updateSection("", { first_run_complete: true });
    await tick();
    session.addSystemMessage(getJarvisGreeting());
  }

  // Handle gesture events for particle effects and navigation
  function onGestureDetected(gesture: string) {
    particleBurst?.gestureBurst(gesture);
    // A hand moving over the study surface must not switch tabs under the person.
    if (activeTab === "study") return;
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

  async function handleReplay(command: string) {
    prefillText = "";
    await tick();
    prefillText = command; // Prefills input box so the user can edit before sending
  }

  $effect(() => {
    $session.messages;
    $session.loading;
    $session.streamingText;
    $session.phase;
    if (shouldFollowLatest(isAtBottom)) {
      tick().then(() => {
        if (shouldFollowLatest(isAtBottom)) {
          virtualListEl?.scrollToBottom("auto");
        }
      });
    }
  });
  function actionLabel(action: { action_type: string; dry_run?: boolean }, planDryRun = false): string {
    const title = actionTitle(action.action_type);
    return action.dry_run || planDryRun ? `${title} (dry run)` : title;
  }
  function isRisky(action: { requires_root?: boolean; destructive?: boolean }): boolean {
    return Boolean(action.requires_root || action.destructive);
  }
  function tierLabel(action: { requires_root?: boolean; destructive?: boolean }): string {
    return action.requires_root ? "Root" : "Destructive";
  }
  function tierClass(action: { requires_root?: boolean; destructive?: boolean }): string {
    return action.requires_root ? "tier-root" : "tier-destructive";
  }
  /** The live step states for the plan that is running now; older plans show their numbered steps. */
  function liveSteps(plan: { plan_id: string; actions: PlanAction[] }): LiveActionState[] | null {
    const live = $session.liveActions;
    if (!plan.plan_id || plan.plan_id !== $session.currentPlan?.plan_id) return null;
    return live.length === plan.actions.length ? live : null;
  }
  function stepStatusText(status: LiveActionState["status"] | undefined): string {
    if (status === "running") return "running";
    if (status === "success") return "done";
    if (status === "error") return "failed";
    if (status === "pending") return "waiting";
    return "";
  }
  /** A Heliox reply that follows a user message (or opens the chat) starts a new turn with the Heliox name. */
  function startsTurn(index: number): boolean {
    return index === 0 || $session.messages[index - 1]?.type === "user";
  }
  function greeting(): string {
    const hour = new Date().getHours();
    if (hour < 5) return $_("chat.greeting_night");
    if (hour < 12) return $_("chat.greeting_morning");
    if (hour < 17) return $_("chat.greeting_afternoon");
    if (hour < 22) return $_("chat.greeting_evening");
    return $_("chat.greeting_night");
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
</script>

<svelte:document onvisibilitychange={() => (documentVisible = document.visibilityState === "visible")} />
{#if showWizard}
  <SetupWizard oncomplete={onSetupComplete} />
{/if}
<main class="window" class:dragging={isDragging} class:hidden-behind-wizard={showWizard}>
  <header
    class="titlebar"
    role="none"
    data-tauri-drag-region
    onmousedown={() => {
      isDragging = true;
    }}
    onmouseup={() => {
      isDragging = false;
    }}
  >
    <div class="titlebar-left">
      <ArcReactor />
      <span class="title">{$_("app.title")}</span>
    </div>
    <nav class="tabs">
      <button
        class="tab"
        class:active={activeTab === "chat"}
        title="Open Command Panel"
        onclick={() => (activeTab = "chat")}>{$_("app.tab_command")}</button
      >
      <button
        class="tab"
        class:active={activeTab === "log"}
        title="Open activity log"
        onclick={() => (activeTab = "log")}>{$_("app.tab_activity")}</button
      >
      <button
        class="tab"
        class:active={activeTab === "today"}
        title="What Heliox did today and where your time went"
        onclick={() => (activeTab = "today")}>{$_("app.tab_today")}</button
      >
      <button
        class="tab"
        class:active={activeTab === "dashboard"}
        title="Open Dashboard"
        onclick={() => (activeTab = "dashboard")}>Dashboard</button
      >
      <button
        class="tab"
        class:active={activeTab === "plugins"}
        title="Browse plugin marketplace"
        onclick={() => (activeTab = "plugins")}>{$_("app.tab_plugins")}</button
      >
      <button
        class="tab"
        class:active={activeTab === "settings"}
        title="Open Settings"
        onclick={() => (activeTab = "settings")}>{$_("app.tab_settings")}</button
      >
      {#if $studyActive}
        <button
          class="tab"
          class:active={activeTab === "study"}
          title="Open the selection study"
          onclick={() => (activeTab = "study")}>{$_("app.tab_study")}</button
        >
      {/if}
    </nav>
    <div class="titlebar-right">
      <ChatSessionControls onactivate={() => (activeTab = "chat")} />
      <HeaderMiniMonitor />
      <ConnectionBadge />
      <AmbientHUD />
    </div>
  </header>

  <div class="content">
    <svelte:boundary>
      <div class="chat-panel" class:inactive={activeTab !== "chat"}>
        <NeuralBackground />

        <InteractiveServices onconfigure={() => (activeTab = "settings")} />
        {#if activeModal === "confirmation"}
          <!-- Keyed by plan: a new approval never inherits the previous dialog's checkboxes. -->
          {#key $session.confirmPlanId}
            <ConfirmDialog
              actions={$session.confirmActions}
              reason={$session.confirmReason}
              riskAssessment={$session.confirmRiskAssessment}
              gnomon={$session.confirmGnomon}
              submitting={$session.confirmSubmitting}
              error={$session.confirmError}
              onselectionchange={(indices) => session.setConfirmSelection($session.confirmPlanId, indices)}
              onconfirm={(approvedIndices) => session.confirm(true, approvedIndices, { inputChannel: "dialog" })}
              ondeny={() => session.confirm(false, undefined, { inputChannel: "dialog" })}
            />
          {/key}
        {:else if activeModal === "rollback"}
          <RollbackDialog onconfirm={() => session.confirmRollback()} oncancel={() => session.cancelRollback()} />
        {:else if activeModal === "undo" && $session.undoPreview}
          <UndoDialog
            remaining={$session.undoQueue.length}
            preview={$session.undoPreview}
            submitting={$session.undoSubmitting}
            error={$session.undoError}
            onconfirm={() => session.confirmUndo()}
            oncancel={() => session.cancelUndo()}
          />
        {/if}

        <HumanControlBanner />
        <FocusBanner />
        {#if paletteOpen}
          <CommandPalette onpick={pickFromPalette} onclose={() => (paletteOpen = false)} />
        {/if}
        {#if $session.undoOffer && !$session.undoPreview}
          <div class="undo-banner" data-testid="undo-offer">
            <span>{$_("undo.banner", { values: { count: $session.undoOffer.undoable } })}</span>
            <span class="undo-banner-actions">
              <button class="undo-btn" onclick={() => session.openUndoPreview()}>{$_("undo.banner_open")}</button>
              <button class="undo-dismiss" aria-label={$_("undo.dismiss")} onclick={() => session.dismissUndoOffer()}
                >&times;</button
              >
            </span>
          </div>
        {/if}
        {#each ($airHandoff.requests ?? [])
          .filter((item) => item.status === "waiting")
          .slice(0, 1) as request (request.request_id)}
          <div class="undo-banner" data-testid="phone-request-banner">
            <span><strong>{request.device_name}</strong> asks: {request.text}</span>
            <span class="undo-banner-actions">
              <button class="undo-btn" onclick={() => void airHandoff.runRequest(request.request_id)}>Run</button>
              <button
                class="undo-dismiss"
                aria-label="Dismiss"
                onclick={() => void airHandoff.dismissRequest(request.request_id)}>&times;</button
              >
            </span>
          </div>
        {/each}
        {#if $session.routineOffer}
          <div class="undo-banner routine-banner" data-testid="routine-offer">
            <span>
              {$_("routine.offer", { values: { name: $session.routineOffer.name } })}
              {#if $session.routineOffer.every_minutes}
                {$session.routineOffer.label}
              {:else}
                <input
                  class="routine-time"
                  type="time"
                  aria-label={$_("routine.time")}
                  value={$session.routineOffer.daily_at}
                  onchange={(event) => session.setRoutineTime(event.currentTarget.value)}
                />
                {$session.routineOffer.weekdays?.length ? $_("routine.on_some_days") : $_("routine.every_day")}
              {/if}
              <small class="routine-actions"
                >{$_("routine.allowed", {
                  values: { actions: $session.routineOffer.allowed_actions.join(", ") },
                })}</small
              >
            </span>
            <span class="undo-banner-actions">
              <button class="undo-btn" disabled={$session.routineSaving} onclick={() => session.createRoutine()}
                >{$_("routine.create")}</button
              >
              <button class="undo-dismiss" aria-label={$_("routine.dismiss")} onclick={() => session.dismissRoutine()}
                >&times;</button
              >
            </span>
          </div>
        {/if}
        {#if $session.rollback && !$session.rollbackPending}
          <div class="undo-banner">
            <span>{$_("rollback.banner")}</span>
            <button class="undo-btn" onclick={() => session.requestRollback()}>{$_("rollback.undo")}</button>
          </div>
        {/if}

        {#if activeModal === "interrupt"}
          <InterruptDialog />
        {:else if activeModal === "budget"}
          <BudgetExceededDialog />
        {:else if activeModal === "supervision"}
          <SupervisionAlertDialog />
        {/if}
        <NeuralControlOverlay showPreview={!blockingSafetyDecision} />

        {#if $session.proactiveSuggestion}
          <aside
            class="proactive-card"
            class:high-priority={$session.proactiveSuggestion.priority === "high"}
            aria-live="polite"
          >
            <div class="proactive-copy">
              <span class="proactive-eyebrow">{$_("companion.suggestion_label")}</span>
              <strong>{$session.proactiveSuggestion.title}</strong>
              <span>{$session.proactiveSuggestion.description}</span>
              <small>
                {$_("companion.learned_relevance", {
                  values: {
                    percent: Math.round($session.proactiveSuggestion.learnedRelevance * 100),
                  },
                })}
              </small>
            </div>
            <div class="proactive-actions">
              <button
                class="proactive-dismiss"
                disabled={$session.proactiveSuggestionPending}
                onclick={() => session.dismissProactiveSuggestion()}
              >
                {$_("companion.dismiss")}
              </button>
              <button
                class="proactive-accept"
                disabled={$session.proactiveSuggestionPending}
                onclick={() => session.acceptProactiveSuggestion()}
              >
                {$session.proactiveSuggestionPending ? $_("companion.starting") : $_("companion.accept")}
              </button>
            </div>
          </aside>
        {/if}

        {#if $session.durableTask && !$session.loading}
          <aside class="task-recovery" aria-live="polite">
            <div>
              <strong>Interrupted task saved</strong>
              <span>Heliox will continue from its last safe checkpoint without repeating completed actions.</span>
            </div>
            <button type="button" disabled={!$session.daemonConnected} onclick={() => session.resumeDurableTask()}>
              {$session.daemonConnected ? "Resume safely" : "Waiting for daemon"}
            </button>
          </aside>
        {/if}

        <div class="results">
          {#if $session.messages.length === 0 && !$session.loading}
            <div class="empty-state">
              <img class="empty-mark" src={heliosMark} alt="" width="64" height="64" />
              <h2>{greeting()}</h2>
              <p>{$_("chat.empty_subtitle")}</p>
              <div class="suggestions">
                <button class="suggestion" onclick={() => session.sendCommand("What should I do now?")}
                  >{$_("chat.suggestion_next")}</button
                >
                <button class="suggestion" onclick={() => session.sendCommand("Show system information")}
                  >{$_("chat.suggestion_sysinfo")}</button
                >
                <button class="suggestion" onclick={openPalette}>{$_("chat.suggestion_help")}</button>
              </div>
              <p class="empty-hint"><kbd>Ctrl</kbd> <kbd>K</kbd> {$_("chat.palette_hint")}</p>
            </div>
          {:else}
            <VirtualList bind:this={virtualListEl} items={$session.messages} bind:atBottom={isAtBottom}>
              {#snippet item(msg, index)}
                {@render messageBlock(msg, index)}
              {/snippet}
              {#snippet footer()}
                {#if $session.loading}
                  <RunStatus />
                {/if}
              {/snippet}
            </VirtualList>
          {/if}
          <ScrollToBottom show={showScrollFAB} onclick={scrollToBottom} />
        </div>
        <div class="input-row">
          <CommandHistory onReplay={handleReplay} />
          <VoiceControl />
          <CommandInput prefill={prefillText} />
          <div
            class="gesture-control-host"
            class:camera-active={cameraControlsActive}
            class:study-minimized={$studyActive}
          >
            <GestureControl
              onGesture={onGestureDetected}
              onActiveChange={(active) => (cameraControlsActive = active)}
            />
          </div>
          <AirHandoffControl />
        </div>
      </div>
      <!-- Outside the connection-epoch key, so a daemon reconnect pauses the study instead of destroying it. Kept
           mounted (hidden) on other tabs; any App modal or another tab pauses it. -->
      {#if $studyActive}
        <div class="study-panel" class:inactive={activeTab !== "study"}>
          <svelte:boundary>
            <SelectionStudy
              paused={activeModal !== null || activeTab !== "study"}
              onexit={() => {
                if ($studyActive) void coadaptation.exitStudy();
              }}
            />
            {#snippet failed(error)}
              <div class="empty-state">
                <h2>The selection study stopped</h2>
                <p style="font-family: var(--font-mono); font-size: 11px;">
                  {error instanceof Error ? error.message : String(error)}
                </p>
                <div class="suggestions">
                  <button class="suggestion" onclick={() => void coadaptation.exitStudy()}>Close the study</button>
                </div>
              </div>
            {/snippet}
          </svelte:boundary>
        </div>
      {/if}
      {#key daemonConnectionEpoch}
        {#if activeTab === "log"}
          <ActivityLog />
        {:else if activeTab === "today"}
          <TodayPanel />
        {:else if activeTab === "dashboard"}
          <Dashboard />
        {:else if activeTab === "plugins"}
          <PluginsTab />
        {:else if activeTab === "settings"}
          <SettingsPanel onOpenCommand={() => (activeTab = "chat")} />
        {/if}
      {/key}
      {#snippet failed(error, reset)}
        <div class="empty-state">
          <div class="empty-logo" style="background: var(--danger)">!</div>
          <h2>Something went wrong</h2>
          <p style="font-family: var(--font-mono); font-size: 11px;">
            {error instanceof Error ? error.message : String(error)}
          </p>
          <div class="suggestions">
            <button class="suggestion" onclick={reset}>Try Again</button>
            <button class="suggestion" onclick={() => (activeTab = "chat")}>Go to Chat</button>
          </div>
        </div>
      {/snippet}
    </svelte:boundary>
  </div>
  <!-- Particle Burst Overlay -->
  <ParticleBurst bind:this={particleBurst} />
</main>
{#snippet copyControl(msg: Message)}
  <button class="copy-button" type="button" aria-label="Copy message" title="Copy" onclick={() => copyMessage(msg)}>
    <Copy size={14} />
  </button>
  <span
    class="copy-feedback"
    role="status"
    aria-live="polite"
    aria-atomic="true"
    class:active={copiedMessageId === msg.timestamp}
  >
    {$_("chat.copied")}
  </span>
{/snippet}

{#snippet messageBlock(msg: Message, index: number)}
  {#if msg.type === "user"}
    <div class="turn turn-user">
      <p class="bubble"><span class="sr-only">{$_("chat.you_said")}</span>{msg.text}</p>
    </div>
  {:else}
    <div class="turn turn-heliox" class:continues={!startsTurn(index)}>
      {#if startsTurn(index)}
        <div class="who" aria-hidden="true">
          <img class="who-mark" src={heliosMark} alt="" width="22" height="22" />
          <span>Heliox</span>
        </div>
      {/if}
      <div class="turn-body">
        {#if msg.type === "agent_run" && msg.agentRun}
          <AgentRunCard run={msg.agentRun} />
        {:else if msg.type === "plan" && msg.plan}
          {@const live = liveSteps(msg.plan)}
          {@const risky = msg.plan.actions.filter(isRisky).length}
          <section class="card plan-msg has-copy" aria-label={$_("plan.label")}>
            <header class="card-head">
              <span class="card-title">{$_("plan.label")}</span>
              <span class="card-meta">{$_("plan.steps", { values: { count: msg.plan.actions.length } })}</span>
              {#if msg.plan.dry_run}
                <span class="chip">{$_("plan.dry_run")}</span>
              {/if}
              {#if risky}
                <span class="chip chip-warn">
                  <ShieldAlert size={12} aria-hidden="true" />
                  {$_("plan.needs_care", { values: { count: risky } })}
                </span>
              {/if}
            </header>
            {#if msg.plan.explanation}
              <p class="plan-explanation">{msg.plan.explanation}</p>
            {/if}
            <ol class="steps">
              {#each msg.plan.actions as action, i}
                {@const step = live?.[i]}
                <li
                  class="step"
                  class:running={step?.status === "running"}
                  class:ok={step?.status === "success"}
                  class:bad={step?.status === "error"}
                  class:waiting={step?.status === "pending"}
                  aria-current={step?.status === "running" ? "step" : undefined}
                >
                  <span class="step-icon" aria-hidden="true">
                    {#if step?.status === "running"}
                      <span class="spin"><LoaderCircle size={16} /></span>
                    {:else if step?.status === "success"}
                      <CircleCheck size={16} />
                    {:else if step?.status === "error"}
                      <CircleX size={16} />
                    {:else if step?.status === "pending"}
                      <Circle size={16} />
                    {:else}
                      <span class="step-num">{i + 1}</span>
                    {/if}
                  </span>
                  <span class="step-text">
                    <span class="step-title"
                      >{actionLabel(action, Boolean(msg.plan.dry_run))}{#if step}<span class="sr-only"
                          >, {stepStatusText(step.status)}</span
                        >{/if}</span
                    >
                    {#if action.target}
                      <span class="step-target">{action.target}</span>
                    {/if}
                    {#if step?.status === "error" && step.error}
                      <span class="step-error">{step.error}</span>
                    {/if}
                  </span>
                  {#if isRisky(action)}
                    <span class="tier-badge {tierClass(action)}">{tierLabel(action)}</span>
                  {/if}
                </li>
              {/each}
            </ol>
            {@render copyControl(msg)}
          </section>
        {:else if msg.type === "result"}
          <section class="card result-msg has-copy" aria-label={$_("result.label")}>
            <header class="card-head">
              <span class="card-title result-title"
                ><CircleCheck size={14} aria-hidden="true" />{$_("result.label")}</span
              >
              {#if msg.verification}
                <span class="chip" class:chip-ok={msg.verification.passed} class:chip-bad={!msg.verification.passed}>
                  {msg.verification.passed ? $_("result.verified") : $_("result.issues")}
                </span>
              {/if}
            </header>
            {#if msg.text}
              <div class="result-summary">{@html renderMarkdown(msg.text)}</div>
            {/if}
            {#if msg.actionResults && msg.actionResults.length > 0}
              <ul class="outcomes">
                {#each msg.actionResults as ar}
                  <li class="outcome" class:failed={!ar.success}>
                    <div class="outcome-head">
                      <span class="outcome-icon" aria-hidden="true">
                        {#if ar.success}<CircleCheck size={15} />{:else}<CircleX size={15} />{/if}
                      </span>
                      <span class="outcome-title"
                        >{actionTitle(ar.action_type)}<span class="sr-only"
                          >, {ar.success ? $_("result.ok") : $_("result.failed")}</span
                        ></span
                      >
                      {#if ar.target}
                        <code class="outcome-target">{ar.target}</code>
                      {/if}
                    </div>
                    {#if ar.success && ar.output}
                      <div class="ar-output">
                        {@html renderMarkdown(ar.output.trim())}
                      </div>
                    {/if}
                    {#if !ar.success && ar.error}
                      <pre class="ar-error">{ar.error}</pre>
                    {/if}
                  </li>
                {/each}
              </ul>
            {:else if !msg.text}
              <p class="result-summary">{$_("result.done")}</p>
            {/if}
            {#if msg.verification && msg.verification.details.length > 0}
              <div class="verification-section">
                <span class="verification-label">{$_("result.verification")}</span>
                {#each msg.verification.details as detail}
                  <span
                    class="verification-detail"
                    class:v-pass={detail.includes("VERIFIED")}
                    class:v-fail={detail.includes("FAILED") || detail.includes("MISMATCH")}
                  >
                    {detail}
                  </span>
                {/each}
              </div>
            {/if}
            {#if msg.task}
              <div class="task-footer" data-testid="task-footer">
                {#if msg.task.undo?.available}
                  <button
                    class="task-chip"
                    type="button"
                    onclick={() => session.openUndoPreview(msg.task?.undo?.taskKey)}
                  >
                    {$_("task_footer.undo", { values: { count: msg.task.undo.undoable } })}
                  </button>
                {:else if msg.task.undo?.undone}
                  <span class="task-note">{$_("task_footer.undone")}</span>
                {/if}
                {#if msg.task.receiptId}
                  <button
                    class="task-chip"
                    type="button"
                    onclick={() => msg.task?.receiptId && session.suggestRoutine(msg.task.receiptId)}
                  >
                    {$_("task_footer.routine")}
                  </button>
                  <button
                    class="task-chip"
                    type="button"
                    onclick={() => msg.task?.receiptId && session.verifyReceipt(msg.task.receiptId)}
                  >
                    {$_("task_footer.receipt")}
                  </button>
                {/if}
                {#if msg.task.receiptCheck}
                  <span class="task-note" role="status">{msg.task.receiptCheck}</span>
                {/if}
              </div>
            {/if}
            {@render copyControl(msg)}
          </section>
        {:else if msg.type === "error"}
          <section class="card error-msg has-copy" role="group" aria-label={$_("error.label")}>
            <header class="card-head">
              <span class="card-title error-title"><CircleX size={14} aria-hidden="true" />{$_("error.label")}</span>
            </header>
            <p class="msg-text">{msg.text}</p>
            {@render copyControl(msg)}
          </section>
        {:else if msg.type === "git_conflict" && msg.gitConflict}
          <GitConflictResolver payload={msg.gitConflict} />
        {:else}
          <div class="system-msg has-copy">
            <div class="msg-text prose">
              {@html renderMarkdown(msg.text || "")}
            </div>
            {@render copyControl(msg)}
          </div>
        {/if}
      </div>
    </div>
  {/if}
{/snippet}

<style>
  /* ── Window and header ─────────────────────────────────────────────── */
  .window {
    height: 100%;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    overflow: hidden;
  }

  .window.hidden-behind-wizard {
    visibility: hidden;
  }

  .titlebar {
    display: flex;
    align-items: center;
    gap: 12px;
    min-height: 48px;
    padding: 6px 12px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    user-select: none;
    -webkit-app-region: drag;
  }

  .titlebar-left {
    display: flex;
    flex: none;
    align-items: center;
    gap: 8px;
  }

  .title {
    font-family: var(--font-display);
    font-size: var(--text-md);
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }

  .titlebar-right {
    display: flex;
    flex: none;
    align-items: center;
    gap: 6px;
    margin-left: auto;
    -webkit-app-region: no-drag;
  }

  .tabs {
    display: flex;
    min-width: 0;
    gap: 2px;
    overflow-x: auto;
    scrollbar-width: none;
    -webkit-app-region: no-drag;
  }

  .tabs::-webkit-scrollbar {
    display: none;
  }

  .tab {
    flex: none;
    padding: 6px 11px;
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-secondary);
    border-radius: var(--radius-sm);
    transition:
      color var(--dur-fast) ease,
      background var(--dur-fast) ease;
  }

  .tab:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .tab.active {
    color: var(--accent);
    background: var(--accent-muted);
  }

  /* The default window is 700 px wide: the wordmark, the resource monitor and button labels give way to the tabs. */
  @media (max-width: 880px) {
    .title,
    .titlebar-right :global(.mini-monitor) {
      display: none;
    }

    .titlebar {
      gap: 8px;
      padding: 6px 8px;
    }

    .tab {
      padding: 6px 8px;
    }
  }

  /* ── Panels ────────────────────────────────────────────────────────── */
  .content {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  .chat-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }

  .chat-panel.inactive {
    position: absolute;
    inset: 0;
    z-index: 1001;
    pointer-events: none;
  }

  /* :global so child components (the services strip, the background) are hidden too;
     a scoped selector only matched App's own elements, leaving them over other tabs. */
  .chat-panel.inactive > :global(:not(.input-row)) {
    display: none;
  }

  .chat-panel.inactive .input-row {
    position: fixed;
    right: 16px;
    bottom: 12px;
    z-index: 1002;
    padding: 0;
    border: 0;
    background: transparent;
    pointer-events: none;
  }

  .chat-panel.inactive .input-row > :global(:not(.gesture-control-host)) {
    display: none;
  }

  .chat-panel.inactive .gesture-control-host:not(.camera-active) {
    display: none;
  }

  .chat-panel.inactive .gesture-control-host.camera-active {
    pointer-events: auto;
  }

  .study-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
  }

  .study-panel.inactive {
    display: none;
  }

  /* During a selection study the fixed bottom-right camera preview shrinks to a 64 x 48 thumbnail above the camera
     button, and its badges and the gesture status readouts are hidden, so they take as little of the study surface
     as possible. The camera keeps running for the hand pointer, and the button still stops it. */
  .gesture-control-host.study-minimized :global(.camera-pip) {
    width: 64px;
    height: 48px;
    bottom: 56px;
    border-radius: 8px;
  }

  .gesture-control-host.study-minimized
    :global(
      :is(
        .pip-badge,
        .pip-hand-badge,
        .pip-gaze-badge,
        .pip-gesture-tag,
        .cam-trail,
        .gesture-label,
        .gesture-history,
        .spatial-status
      )
    ) {
    display: none;
  }

  .results {
    flex: 1;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  /* ── Banners above the conversation ────────────────────────────────── */
  .undo-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 12px;
    margin: 8px 12px 0;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    font-size: var(--text-sm);
    color: var(--text-secondary);
  }

  .undo-banner-actions {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .undo-btn {
    padding: 5px 12px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-hover);
    border-radius: var(--radius-sm);
    white-space: nowrap;
    transition: background var(--dur-fast) ease;
  }

  .undo-btn:hover {
    background: var(--border-strong);
  }

  .undo-btn:disabled {
    cursor: wait;
    opacity: 0.55;
  }

  .undo-dismiss {
    display: inline-grid;
    place-items: center;
    width: 28px;
    height: 28px;
    font-size: var(--text-lg);
    color: var(--text-secondary);
    border-radius: var(--radius-sm);
  }

  .undo-dismiss:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .routine-time {
    margin: 0 4px;
    padding: 2px 6px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    color: var(--text-primary);
    font: inherit;
  }

  .routine-actions {
    display: block;
    margin-top: 2px;
    color: var(--text-secondary);
    font-size: var(--text-xs);
  }

  .proactive-card {
    z-index: var(--z-raised);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: 8px 12px 0;
    padding: 12px 14px;
    background: color-mix(in srgb, var(--accent) 7%, var(--bg-secondary));
    border: 1px solid var(--accent-line);
    border-radius: var(--radius-md);
  }

  .proactive-card.high-priority {
    border-color: var(--warning);
  }

  .proactive-copy {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 3px;
    font-size: var(--text-sm);
    color: var(--text-secondary);
  }

  .proactive-copy strong {
    color: var(--text-primary);
    font-size: var(--text-base);
  }

  .proactive-copy small {
    color: var(--text-muted);
    font-size: var(--text-xs);
  }

  .proactive-eyebrow {
    color: var(--accent);
    font-size: var(--text-xs);
    font-weight: 600;
  }

  .proactive-actions {
    display: flex;
    flex-shrink: 0;
    gap: 6px;
  }

  .proactive-actions button {
    padding: 6px 12px;
    font-size: var(--text-sm);
    font-weight: 600;
    border-radius: var(--radius-sm);
  }

  .proactive-dismiss {
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
  }

  .proactive-accept {
    color: var(--on-accent);
    background: var(--accent);
  }

  .proactive-actions button:disabled {
    cursor: wait;
    opacity: 0.55;
  }

  .task-recovery {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    margin: 8px 12px 0;
    padding: 10px 12px;
    border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent);
    border-radius: var(--radius-md);
    background: var(--warning-bg);
    color: var(--text-primary);
  }

  .task-recovery div {
    display: grid;
    gap: 2px;
  }

  .task-recovery span {
    color: var(--text-secondary);
    font-size: var(--text-sm);
  }

  .task-recovery button {
    flex: 0 0 auto;
    border: 1px solid color-mix(in srgb, var(--warning) 55%, transparent);
    border-radius: var(--radius-sm);
    padding: 7px 11px;
    background: var(--warning-bg);
    color: var(--text-primary);
    font-weight: 600;
  }

  .task-recovery button:disabled {
    cursor: wait;
    opacity: 0.55;
  }

  /* ── Empty state ───────────────────────────────────────────────────── */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    text-align: center;
    padding: 24px;
    animation: rise var(--dur-slow) var(--ease-out) both;
  }

  .empty-mark {
    width: 64px;
    height: 64px;
    margin-bottom: 4px;
    border-radius: 50%;
    border: 1px solid var(--accent-line);
    object-fit: cover;
  }

  .empty-state h2 {
    font-size: var(--text-2xl);
    color: var(--text-primary);
  }

  .empty-state p {
    max-width: 46ch;
    font-size: var(--text-base);
    line-height: 1.55;
    color: var(--text-secondary);
  }

  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 6px;
  }

  .suggestion {
    padding: 7px 14px;
    font-size: var(--text-sm);
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    transition:
      border-color var(--dur-fast) ease,
      background var(--dur-fast) ease,
      transform var(--dur-fast) ease;
  }

  .suggestion:hover {
    border-color: var(--accent-line);
    background: var(--accent-muted);
  }

  .suggestion:active {
    transform: translateY(1px);
  }

  .empty-state .empty-hint {
    margin-top: 6px;
    font-size: var(--text-xs);
    color: var(--text-muted);
  }

  /* ── Conversation turns ────────────────────────────────────────────── */
  .turn {
    display: flex;
    flex-direction: column;
    padding: 4px 4px 2px;
    animation: rise var(--dur) var(--ease-out) both;
  }

  .turn-user {
    align-items: flex-end;
    padding-top: 12px;
  }

  .bubble {
    max-width: min(78%, 68ch);
    padding: 8px 13px;
    font-size: var(--text-base);
    line-height: 1.5;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg) var(--radius-lg) var(--radius-xs) var(--radius-lg);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .turn-heliox {
    gap: 6px;
    padding-top: 10px;
  }

  .turn-heliox.continues {
    padding-top: 4px;
  }

  .who {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-display);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-secondary);
  }

  .who-mark {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1px solid var(--accent-line);
    object-fit: cover;
  }

  .turn-body {
    min-width: 0;
    padding-left: 30px;
  }

  .msg-text {
    color: var(--text-primary);
    font-size: var(--text-base);
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .msg-text.prose {
    max-width: 72ch;
    white-space: normal;
  }

  .system-msg {
    position: relative;
  }

  /* Cards: plans, results and errors. */
  .card {
    display: grid;
    gap: 8px;
    padding: 10px 12px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }

  .card-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    min-height: 22px;
  }

  .card-title {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-display);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .card-meta {
    font-size: var(--text-xs);
    color: var(--text-muted);
  }

  .chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-pill);
  }

  .chip-warn {
    color: var(--warning);
    background: var(--warning-bg);
  }

  .chip-ok {
    color: var(--success);
    background: var(--success-bg);
  }

  .chip-bad {
    color: var(--danger);
    background: var(--danger-bg);
  }

  .plan-explanation {
    max-width: 72ch;
    font-size: var(--text-base);
    line-height: 1.5;
    color: var(--text-secondary);
  }

  .steps {
    display: grid;
    gap: 1px;
    list-style: none;
  }

  .step {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    transition: background var(--dur) ease;
  }

  .step.running {
    background: var(--accent-muted);
  }

  .step-icon {
    display: inline-grid;
    flex: none;
    place-items: center;
    width: 18px;
    height: 18px;
    margin-top: 1px;
    color: var(--text-muted);
  }

  .step.running .step-icon {
    color: var(--accent);
  }

  .step.ok .step-icon {
    color: var(--success);
  }

  .step.bad .step-icon {
    color: var(--danger);
  }

  .step.waiting .step-title {
    color: var(--text-secondary);
  }

  .step-num {
    display: inline-grid;
    place-items: center;
    width: 18px;
    height: 18px;
    font-size: var(--text-2xs);
    font-weight: 600;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: 50%;
  }

  .spin {
    display: inline-flex;
    animation: spin 0.9s linear infinite;
  }

  .step-text {
    display: grid;
    flex: 1;
    min-width: 0;
    gap: 1px;
  }

  .step-title {
    font-size: var(--text-base);
    font-weight: 500;
    color: var(--text-primary);
  }

  .step-target,
  .outcome-target {
    overflow: hidden;
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .step-error {
    font-size: var(--text-xs);
    color: var(--danger);
  }

  .tier-badge {
    flex: none;
    padding: 2px 8px;
    font-size: var(--text-2xs);
    font-weight: 600;
    border-radius: var(--radius-pill);
  }

  .tier-destructive {
    color: var(--warning);
    background: var(--warning-bg);
  }

  .tier-root {
    color: var(--danger);
    background: var(--danger-bg);
  }

  .result-title {
    color: var(--success);
  }

  .result-summary {
    max-width: 72ch;
    font-size: var(--text-base);
    line-height: 1.6;
    color: var(--text-primary);
  }

  .outcomes {
    display: grid;
    gap: 6px;
    list-style: none;
  }

  .outcome {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .outcome-head {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    padding: 6px 10px;
    background: var(--bg-tertiary);
  }

  .outcome-icon {
    display: inline-flex;
    flex: none;
    color: var(--success);
  }

  .outcome.failed .outcome-icon {
    color: var(--danger);
  }

  .outcome-title {
    flex: none;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .outcome-target {
    flex: 1;
  }

  .ar-output {
    max-height: 200px;
    overflow-y: auto;
    padding: 8px 10px;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    line-height: 1.55;
    color: var(--text-primary);
    background: var(--bg-primary);
    word-break: break-word;
  }

  /* Rendered markdown: paragraphs and code blocks sit flush inside the output box. */
  .ar-output :global(:is(p, pre, ul, ol)) {
    margin: 0;
  }

  .ar-output :global(* + :is(p, pre, ul, ol)) {
    margin-top: 6px;
  }

  .ar-error {
    padding: 8px 10px;
    font-family: var(--font-mono);
    font-size: var(--text-sm);
    color: var(--danger);
    background: var(--danger-bg);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .verification-section {
    display: grid;
    gap: 2px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
  }

  .verification-label {
    margin-bottom: 2px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-secondary);
  }

  .verification-detail {
    font-family: var(--font-mono);
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .verification-detail.v-pass {
    color: var(--success);
  }

  .verification-detail.v-fail {
    color: var(--danger);
  }

  .error-msg {
    border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
    background: var(--danger-bg);
  }

  .error-title {
    color: var(--danger);
  }

  .task-footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    padding-top: 2px;
  }

  .task-chip {
    padding: 4px 10px;
    font-size: var(--text-xs);
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    transition:
      color var(--dur-fast) ease,
      background var(--dur-fast) ease;
  }

  .task-chip:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .task-note {
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  /* Copy appears on hover or keyboard focus. */
  .has-copy {
    position: relative;
  }

  .card.has-copy {
    padding-right: 40px;
  }

  .copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    color: var(--text-muted);
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    opacity: 0;
    transition:
      opacity var(--dur-fast) ease,
      background var(--dur-fast) ease,
      color var(--dur-fast) ease;
  }

  .has-copy:hover .copy-button,
  .has-copy:focus-within .copy-button {
    opacity: 1;
  }

  .copy-button:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
    border-color: var(--border);
  }

  .copy-feedback {
    position: absolute;
    top: 11px;
    right: 40px;
    padding: 2px 7px;
    font-size: var(--text-2xs);
    font-weight: 600;
    color: var(--success);
    background: var(--success-bg);
    border-radius: var(--radius-pill);
    opacity: 0;
    transition: opacity var(--dur-fast) ease;
    pointer-events: none;
  }

  .copy-feedback.active {
    opacity: 1;
  }

  /* ── Composer row ──────────────────────────────────────────────────── */
  .input-row {
    position: relative;
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
  }

  .input-row > :global(.command-input) {
    flex: 1;
    min-width: 0;
  }

  @keyframes rise {
    from {
      opacity: 0;
      transform: translateY(4px);
    }
    to {
      opacity: 1;
      transform: none;
    }
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
