<script lang="ts">
  /**
   * ReActPipeline — Real-time visualization of the AI reasoning process.
   *
   * Shows a dynamic graph of the full agent loop:
   *   User Request → Memory Recall → Agent Routing → Planning → Execution → Verification → Reflection → Memory Update
   *
   * Each node lights up, pulses, and transitions as the backend emits phase events.
   */

  import { session } from "../stores/session";
  import { onNotification, offNotification } from "../api/daemon";
  import { fade, slide } from "svelte/transition";
  import { onDestroy } from "svelte";

  // ── Pipeline Stage Types ──

  type StageStatus = "idle" | "active" | "success" | "error" | "skipped";

  interface PipelineStage {
    id: string;
    label: string;
    emoji: string;
    status: StageStatus;
    detail: string;
    startTime: number;
    endTime: number;
    children?: PipelineStage[];
  }

  interface ThoughtEntry {
    seq: number;
    stage: string;
    stageId: string;
    text: string;
    type: string;
  }

  // ── State ──

  let isVisible = $state(false);
  let totalDuration = $state(0);
  let agentRouting = $state<{ assigned_agents: string[]; is_multi_agent: boolean } | null>(null);
  let showSkeleton = $state(false);

  const skeletonStages = ["Memory", "Planning", "Routing", "Execution", "Verification", "Reflection"];
  const pipelineMethods = new Set([
    "status",
    "agent_routing",
    "plan_preview",
    "action_start",
    "action_complete",
    "confirm_required",
    "reasoning_event",
  ]);

  let stages = $state<PipelineStage[]>([
    { id: "user_input", label: "User Request", emoji: "💬", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "memory_recall", label: "Memory Recall", emoji: "🧠", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "agent_routing", label: "Agent Routing", emoji: "🔀", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "planning", label: "Planning", emoji: "📐", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "confirmation", label: "Confirmation Gate", emoji: "🔐", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "executing", label: "Execution", emoji: "⚡", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "verifying", label: "Verification", emoji: "✅", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "reflection", label: "Reflection", emoji: "🪞", status: "idle", detail: "", startTime: 0, endTime: 0 },
    { id: "memory_update", label: "Memory Update", emoji: "💾", status: "idle", detail: "", startTime: 0, endTime: 0 },
  ]);

  let executionActions = $state<{ type: string; target: string; status: string }[]>([]);
  let pipelineStartTime = 0;
  let showThoughts = $state(false);
  let expandedThoughtStages = $state<Record<string, boolean>>({});
  let collapsed = $state(false);
  let autoCollapseTimer: ReturnType<typeof setTimeout> | null = null;

  // Reasoning event state
  let stageDecisions = $state<Record<string, string>>({});
  let stageTiming = $state<Record<string, number>>({});
  let thoughtStream = $state<ThoughtEntry[]>([]);

  function resetPipeline() {
    stages = stages.map(s => ({ ...s, status: "idle" as StageStatus, detail: "", startTime: 0, endTime: 0 }));
    executionActions = [];
    totalDuration = 0;
    agentRouting = null;
    stageDecisions = {};
    stageTiming = {};
    thoughtStream = [];
    showSkeleton = false;
    showThoughts = false;
    expandedThoughtStages = {};
    collapsed = false;
    if (autoCollapseTimer) { clearTimeout(autoCollapseTimer); autoCollapseTimer = null; }
  }

  function setStage(id: string, status: StageStatus, detail: string = "") {
    const now = Date.now();
    stages = stages.map(s => {
      if (s.id === id) {
        return {
          ...s,
          status,
          detail: detail || s.detail,
          startTime: s.startTime || now,
          endTime: status === "success" || status === "error" ? now : 0,
        };
      }
      return s;
    });
  }

  // ── Listen to WebSocket Notifications ──

  function handleNotification(method: string, params: unknown) {
    const p = params as Record<string, any>;

    if (showSkeleton && pipelineMethods.has(method)) {
      showSkeleton = false;
    }

    switch (method) {
      case "status": {
        const phase = String(p.phase || "");
        isVisible = true;

        if (phase === "receiving input") {
          if (pipelineStartTime === 0) pipelineStartTime = Date.now();
          setStage("user_input", "active", "Processing...");
        }

        if (phase === "recalling memory") {
          setStage("user_input", "success", "Received");
          setStage("memory_recall", "active", "Searching context...");
        }

        if (phase === "routing agents") {
          setStage("memory_recall", "success", "Context loaded");
          setStage("agent_routing", "active", "Analyzing...");
        }

        if (phase === "planning" || phase.startsWith("re-planning")) {
          if (pipelineStartTime === 0) pipelineStartTime = Date.now();
          setStage("user_input", "success", "Received");
          setStage("memory_recall", "success", "Context loaded");
          setStage("agent_routing", "success",
            agentRouting ? agentRouting.assigned_agents.join(", ") : "Routed");
          setStage("planning", "active", phase.includes("re-planning") ? "Re-planning..." : "Generating plan...");
          // Reset downstream
          setStage("confirmation", "idle");
          setStage("executing", "idle");
          setStage("verifying", "idle");
          setStage("reflection", "idle");
          setStage("memory_update", "idle");
        }

        if (phase === "awaiting_confirmation") {
          setStage("planning", "success", "Plan created");
          setStage("confirmation", "active", "Waiting for user...");
        }

        if (phase === "executing") {
          setStage("planning", "success", "Plan created");
          setStage("confirmation", "success", "Approved");
          setStage("executing", "active", "Running actions...");
        }

        if (phase === "verifying") {
          setStage("executing", "success", `${executionActions.length} action(s) done`);
          setStage("verifying", "active", "Checking results...");
        }

        if (phase.includes("retrying")) {
          setStage("verifying", "error", "Mismatch detected — retrying");
          setStage("planning", "active", "Re-planning...");
        }
        break;
      }

      case "agent_routing": {
        agentRouting = {
          assigned_agents: p.assigned_agents || [],
          is_multi_agent: p.is_multi_agent || false,
        };
        setStage("agent_routing", "success",
          (p.assigned_agents || []).join(", ") || "general");
        break;
      }

      case "plan_preview": {
        setStage("planning", "success", p.explanation?.substring(0, 80) || "Plan ready");
        const needsConfirm = (p.actions || []).some((a: any) => a.requires_confirmation);
        if (needsConfirm) {
          setStage("confirmation", "active", "User approval required");
        } else {
          setStage("confirmation", "skipped", "Auto-approved");
        }
        break;
      }

      case "action_start": {
        const action = p.action || {};
        executionActions = [...executionActions, {
          type: action.action_type || "unknown",
          target: action.target || "",
          status: "running",
        }];
        setStage("executing", "active", `Running: ${action.action_type}`);
        break;
      }

      case "action_complete": {
        const result = p.result || {};
        const action = result.action || {};
        const success = result.success;
        executionActions = executionActions.map(a =>
          a.type === (action.action_type || "") && a.status === "running"
            ? { ...a, status: success ? "success" : "error" }
            : a
        );
        break;
      }

      case "confirm_required": {
        setStage("confirmation", "active", "Waiting for user decision...");
        break;
      }

      case "reasoning_event": {
        const evt = p as {
          event_type: string;
          event_name: string;
          stage: string;
          duration_ms: number;
          data: Record<string, any>;
          sequence: number;
        };

        // Map backend stage names to our stage IDs
        const stageMap: Record<string, string> = {
          user_input: "user_input",
          memory_recall: "memory_recall",
          agent_routing: "agent_routing",
          planning: "planning",
          confirmation: "confirmation",
          orchestration: "executing",
          execution: "executing",
          verification: "verifying",
          reflection: "reflection",
          memory_update: "memory_update",
        };
        const stageId = stageMap[evt.stage] || evt.stage;

        // Handle different event types

        // ── Phase lifecycle: update stage status indicators ──
        if (evt.event_type === "phase_start") {
          setStage(stageId, "active", evt.event_name.replace(/_/g, " ") || "Working...");
        }

        if (evt.event_type === "phase_complete") {
          const detail = evt.data?.reason || evt.event_name.replace(/_/g, " ") || "Done";
          setStage(stageId, "success", detail);
          if (evt.duration_ms > 0) {
            stageTiming = { ...stageTiming, [stageId]: Math.round(evt.duration_ms) };
          }
        }

        if (evt.event_type === "phase_error") {
          const errText = evt.data?.error || "Error";
          setStage(stageId, "error", errText);
          thoughtStream = [
            ...thoughtStream.slice(-19),
            { seq: evt.sequence, stage: evt.stage, stageId, text: `❌ ${errText}`, type: "error" },
          ];
        }

        // ── Thoughts, decisions, progress, metrics ──
        if (evt.event_type === "thought") {
          const text = evt.data?.text || "";
          thoughtStream = [
            ...thoughtStream.slice(-19),
            { seq: evt.sequence, stage: evt.stage, stageId, text, type: "thought" },
          ];
        }

        if (evt.event_type === "decision") {
          const desc = evt.data?.description || "";
          const chosen = evt.data?.chosen || "";
          stageDecisions = { ...stageDecisions, [stageId]: `${desc}: ${chosen}` };
          thoughtStream = [
            ...thoughtStream.slice(-19),
            { seq: evt.sequence, stage: evt.stage, stageId, text: `Decision: ${desc} → ${chosen}`, type: "decision" },
          ];
        }

        if (evt.event_type === "progress") {
          const pct = evt.data?.percent || 0;
          const label = evt.data?.label || "";
          setStage(stageId, "active", `${label} (${pct}%)`);
        }

        if (evt.event_type === "metric") {
          thoughtStream = [
            ...thoughtStream.slice(-19),
            {
              seq: evt.sequence,
              stage: evt.stage,
              stageId,
              text: `📊 ${evt.data?.name}: ${evt.data?.value}${evt.data?.unit || ""}`,
              type: "metric",
            },
          ];
        }

        break;
      }
    }
  }

  onNotification(handleNotification);
  onDestroy(() => offNotification(handleNotification));

  // ── React to session phase changes for completion ──

  $effect(() => {
    const s = $session;

    // When loading stops, mark final stages
    if (!s.loading && isVisible && pipelineStartTime > 0) {
      const lastMsg = s.messages[s.messages.length - 1];
      if (lastMsg) {
        // Defer state mutations out of the $effect tracking context
        queueMicrotask(() => {
          showSkeleton = false;
          if (lastMsg.type === "result") {
            setStage("executing", "success", `${executionActions.length} action(s) completed`);
            setStage("verifying", lastMsg.verification?.passed ? "success" : "error",
              lastMsg.verification?.passed ? "All checks passed" : "Verification failed");
            setStage("reflection", "success", "Performance analyzed");
            setStage("memory_update", "success", "History saved");
            totalDuration = Date.now() - pipelineStartTime;
          } else if (lastMsg.type === "error") {
            setStage("executing", "error", "Failed");
            setStage("verifying", "error", "N/A");
            setStage("reflection", "success", "Failure recorded");
            setStage("memory_update", "success", "Error logged");
            totalDuration = Date.now() - pipelineStartTime;
          }
          pipelineStartTime = 0;
        });
      }
    }

    // When a new command starts, reset
    if (s.loading && s.phase === "" && !s.currentPlan) {
      queueMicrotask(() => {
        resetPipeline();
        isVisible = true;
        showSkeleton = true;
        pipelineStartTime = Date.now();
      });
    }
  });

  // Auto-collapse 2s after reaching 100%
  $effect(() => {
    if (progress === 100 && !collapsed && isVisible) {
      queueMicrotask(() => {
        if (autoCollapseTimer) clearTimeout(autoCollapseTimer);
        autoCollapseTimer = setTimeout(() => { collapsed = true; }, 2000);
      });
    }
  });

  let dismiss = () => { isVisible = false; };
  let toggleCollapse = () => { collapsed = !collapsed; };
  let toggleThoughts = () => { showThoughts = !showThoughts; };
  let toggleThoughtStage = (stageId: string) => {
    expandedThoughtStages = {
      ...expandedThoughtStages,
      [stageId]: !expandedThoughtStages[stageId],
    };
  };

  const thoughtStageLabels: Record<string, string> = {
    user_input: "User Request",
    memory_recall: "Memory Recall",
    agent_routing: "Agent Routing",
    planning: "Planning",
    confirmation: "Confirmation",
    executing: "Execution",
    verifying: "Verification",
    reflection: "Reflection",
    memory_update: "Memory Update",
  };

  function thoughtTypeLabel(type: string) {
    return type.replace(/_/g, " ");
  }

  // Computed: progress percentage
  let progress = $derived(
    Math.round(
      (stages.filter(s => s.status === "success" || s.status === "skipped").length / stages.length) * 100
    )
  );

  let thoughtGroups = $derived.by(() => {
    const grouped = new Map<string, ThoughtEntry[]>();
    for (const thought of thoughtStream) {
      const key = thought.stageId || thought.stage;
      grouped.set(key, [...(grouped.get(key) || []), thought]);
    }
    return [...grouped.entries()].map(([stageId, entries]) => ({
      stageId,
      label: thoughtStageLabels[stageId] || stageId.replace(/_/g, " "),
      entries,
      latest: entries[entries.length - 1],
    }));
  });
</script>

{#if isVisible}
  <div class="react-pipeline" class:collapsed transition:slide={{ duration: 300 }}>
    <!-- Header (always visible, clickable to expand/collapse) -->
    <div class="pipeline-header" role="button" tabindex="0" onclick={toggleCollapse} onkeydown={(e) => e.key === 'Enter' && toggleCollapse()}>
      <div class="pipeline-title">
        <span class="reactor-icon">⚛️</span>
        ReAct Pipeline
        <span class="progress-chip">{progress}%</span>
        {#if collapsed && totalDuration > 0}
          <span class="collapsed-summary">completed in {(totalDuration / 1000).toFixed(1)}s — click to expand</span>
        {/if}
      </div>
      <div class="pipeline-controls">
        {#if totalDuration > 0}
          <span class="duration-badge">{(totalDuration / 1000).toFixed(1)}s</span>
        {/if}
        {#if agentRouting?.is_multi_agent}
          <span class="multi-agent-badge">Multi-Agent</span>
        {/if}
        <button class="collapse-toggle" onclick={(e) => { e.stopPropagation(); toggleCollapse(); }} title={collapsed ? 'Expand' : 'Collapse'}>
          {collapsed ? '▼' : '▲'}
        </button>
        <button
          class="thought-toggle"
          class:active={showThoughts}
          onclick={(e) => { e.stopPropagation(); toggleThoughts(); }}
          title={showThoughts ? "Hide agent thoughts" : "Show agent thoughts"}
          aria-label={showThoughts ? "Hide agent thoughts" : "Show agent thoughts"}
          aria-expanded={showThoughts}
        >
          ◫
        </button>
        <button class="dismiss-btn" onclick={(e) => { e.stopPropagation(); dismiss(); }} title="Dismiss">✕</button>
      </div>
    </div>

    <!-- Progress bar -->
    <div class="progress-track" class:skeleton={showSkeleton}>
      <div class="progress-fill" style="width: {showSkeleton ? 0 : progress}%"></div>
    </div>

  {#if !collapsed}
    {#if showSkeleton}
      <div class="pipeline-graph skeleton-graph" aria-label="Preparing ReAct pipeline">
        {#each skeletonStages as stage, i}
          <div class="stage-wrapper skeleton-stage" transition:fade={{ duration: 180, delay: i * 30 }}>
            {#if i > 0}
              <div class="connector skeleton-connector"></div>
            {/if}

            <div class="stage-node skeleton-node">
              <div class="skeleton-emoji" aria-hidden="true"></div>
              <div class="stage-info">
                <div class="stage-label">{stage}</div>
                <div class="skeleton-detail" aria-hidden="true"></div>
              </div>
              <div class="stage-indicator">
                <span class="skeleton-dot" aria-hidden="true"></span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {:else}
    <!-- Pipeline nodes -->
    <div class="pipeline-graph">
      {#each stages as stage, i (stage.id)}
        <div class="stage-wrapper" transition:fade={{ duration: 200, delay: i * 30 }}>
          {#if i > 0}
            <div class="connector" class:active={stage.status !== "idle"} class:success={stage.status === "success"} class:error={stage.status === "error"}></div>
          {/if}

          <div class="stage-node {stage.status}" class:pulse={stage.status === "active"}>
            <div class="stage-emoji">{stage.emoji}</div>
            <div class="stage-info">
              <div class="stage-label">{stage.label}</div>
              {#if stage.detail}
                <div class="stage-detail" transition:fade={{ duration: 150 }}>{stage.detail}</div>
              {/if}
              <!-- Decision badge -->
              {#if showThoughts && stageDecisions[stage.id]}
                <div class="decision-badge" transition:fade={{ duration: 200 }}>
                  <span class="decision-icon">⚖️</span>
                  <span class="decision-text">{stageDecisions[stage.id]}</span>
                </div>
              {/if}
              <!-- Stage timing -->
              {#if stageTiming[stage.id]}
                <div class="stage-timing">{stageTiming[stage.id]}ms</div>
              {/if}
            </div>
            <div class="stage-indicator">
              {#if stage.status === "active"}
                <div class="spinner"></div>
              {:else if stage.status === "success"}
                <span class="check">✓</span>
              {:else if stage.status === "error"}
                <span class="cross">✗</span>
              {:else if stage.status === "skipped"}
                <span class="skip">⤸</span>
              {:else}
                <span class="dot"></span>
              {/if}
            </div>
          </div>

          <!-- Execution sub-actions -->
          {#if stage.id === "executing" && executionActions.length > 0}
            <div class="sub-actions" transition:slide={{ duration: 200 }}>
              {#each executionActions as act, j}
                <div class="sub-action {act.status}" transition:fade={{ duration: 150 }}>
                  <span class="sub-dot {act.status}"></span>
                  <span class="sub-type">{act.type.replace(/_/g, " ")}</span>
                  {#if act.target}
                    <span class="sub-target">{act.target}</span>
                  {/if}
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
    {/if}

    <!-- Collapsible Thought Stream -->
    {#if !showSkeleton && showThoughts && thoughtStream.length > 0}
      <div class="thought-stream" transition:slide={{ duration: 200 }}>
        <div class="thought-stream-header">
          <span class="thought-stream-icon">◫</span>
          <span class="thought-stream-title">Agent Thoughts</span>
          <span class="thought-count">{thoughtStream.length}</span>
        </div>
        <div class="thought-accordion-list">
          {#each thoughtGroups as group (group.stageId)}
            <section class="thought-accordion" class:open={expandedThoughtStages[group.stageId]}>
              <button
                type="button"
                class="thought-accordion-header"
                onclick={() => toggleThoughtStage(group.stageId)}
                aria-expanded={Boolean(expandedThoughtStages[group.stageId])}
              >
                <span class="thought-stage-chip">{group.label}</span>
                <span class="thought-preview">{group.latest.text}</span>
                <span class="thought-count">{group.entries.length}</span>
                <span class="thought-chevron">{expandedThoughtStages[group.stageId] ? "▲" : "▼"}</span>
              </button>
              {#if expandedThoughtStages[group.stageId]}
                <div class="thought-stream-list" transition:slide={{ duration: 160 }}>
                  {#each group.entries as thought (thought.seq)}
                    <div class="thought-entry {thought.type}" transition:fade={{ duration: 120 }}>
                      <span class="thought-seq">#{thought.seq}</span>
                      <span class="thought-type">{thoughtTypeLabel(thought.type)}</span>
                      <span class="thought-content">{thought.text}</span>
                    </div>
                  {/each}
                </div>
              {/if}
            </section>
          {/each}
        </div>
      </div>
    {:else if showThoughts}
      <div class="thought-stream empty" transition:fade={{ duration: 160 }}>
        <div class="thought-stream-header">
          <span class="thought-stream-icon">◫</span>
          <span class="thought-stream-title">Agent Thoughts</span>
        </div>
        <div class="thought-empty">
          Intermediate agent thoughts will appear here.
        </div>
      </div>
    {/if}

    {#if !showThoughts && thoughtStream.length > 0}
      <button
        type="button"
        class="thought-summary"
        onclick={toggleThoughts}
        aria-label="Show grouped agent thoughts"
      >
        <span>Agent thoughts hidden</span>
        <span class="thought-count">{thoughtStream.length}</span>
      </button>
    {/if}

    <!-- Agent routing info -->
    {#if !showSkeleton && agentRouting}
      <div class="routing-info" transition:fade>
        <span class="routing-label">Agents:</span>
        {#each agentRouting.assigned_agents as agent}
          <span class="agent-chip">{agent.replace(/_/g, " ")}</span>
        {/each}
      </div>
    {/if}
  {/if}
  </div>
{/if}

<style>
  .react-pipeline {
    background: linear-gradient(135deg, rgba(8, 12, 22, 0.85), rgba(15, 23, 42, 0.85));
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    backdrop-filter: blur(12px);
    font-family: 'Inter', 'JetBrains Mono', monospace;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: padding 0.3s ease;
  }

  .react-pipeline.collapsed {
    padding: 0.5rem 1rem;
  }

  .pipeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    cursor: pointer;
    user-select: none;
  }

  .react-pipeline.collapsed .pipeline-header {
    margin-bottom: 0;
  }

  .collapsed-summary {
    font-size: 0.65rem;
    color: rgba(0, 240, 255, 0.5);
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    margin-left: 4px;
  }

  .collapse-toggle {
    background: none;
    border: 1px solid rgba(0, 240, 255, 0.2);
    color: rgba(0, 240, 255, 0.6);
    border-radius: 4px;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.6rem;
    transition: all 0.2s;
  }

  .collapse-toggle:hover {
    border-color: rgba(0, 240, 255, 0.5);
    color: rgba(0, 240, 255, 1);
  }

  .pipeline-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #00f0ff;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
  }

  .reactor-icon {
    font-size: 1.1rem;
    filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.6));
  }

  .progress-chip {
    background: rgba(0, 240, 255, 0.15);
    color: #00f0ff;
    padding: 0.1rem 0.4rem;
    border-radius: 6px;
    font-size: 0.7rem;
    border: 1px solid rgba(0, 240, 255, 0.25);
  }

  .pipeline-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .duration-badge {
    background: rgba(0, 255, 170, 0.1);
    color: #00ffaa;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid rgba(0, 255, 170, 0.2);
  }

  .multi-agent-badge {
    background: rgba(255, 170, 0, 0.15);
    color: #ffaa00;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    border: 1px solid rgba(255, 170, 0, 0.3);
  }

  .dismiss-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.7rem;
    transition: all 0.2s;
  }

  .dismiss-btn:hover {
    background: rgba(255, 50, 50, 0.2);
    color: #ff5555;
    border-color: rgba(255, 50, 50, 0.3);
  }

  /* Progress bar */
  .progress-track {
    height: 3px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
    margin-bottom: 1rem;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00f0ff, #00ffaa);
    border-radius: 3px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 8px rgba(0, 240, 255, 0.4);
  }

  .progress-track.skeleton {
    background: rgba(255, 255, 255, 0.07);
  }

  /* Pipeline graph */
  .pipeline-graph {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .stage-wrapper {
    display: flex;
    flex-direction: column;
  }

  .connector {
    width: 2px;
    height: 12px;
    background: rgba(255, 255, 255, 0.07);
    margin-left: 19px;
    transition: all 0.4s ease;
  }

  .connector.active {
    background: rgba(0, 240, 255, 0.3);
    box-shadow: 0 0 4px rgba(0, 240, 255, 0.2);
  }

  .connector.success {
    background: rgba(0, 255, 170, 0.4);
  }

  .connector.error {
    background: rgba(255, 50, 50, 0.4);
  }

  .stage-node {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
  }

  .skeleton-node {
    background: rgba(255, 255, 255, 0.035);
    border-color: rgba(255, 255, 255, 0.08);
    opacity: 0.72;
    overflow: hidden;
  }

  .skeleton-node::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.08), transparent);
    animation: skeleton-shimmer 1.4s ease-in-out infinite;
    transform: translateX(-100%);
  }

  .skeleton-connector {
    background: rgba(255, 255, 255, 0.09);
  }

  .skeleton-emoji,
  .skeleton-dot {
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    flex-shrink: 0;
  }

  .skeleton-emoji {
    width: 28px;
    height: 28px;
  }

  .skeleton-dot {
    width: 8px;
    height: 8px;
  }

  .skeleton-node .stage-label {
    color: rgba(255, 255, 255, 0.42);
  }

  .skeleton-detail {
    width: min(140px, 55%);
    height: 6px;
    margin-top: 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
  }

  .stage-node.active {
    background: rgba(0, 240, 255, 0.08);
    border-color: rgba(0, 240, 255, 0.35);
    box-shadow: 0 0 24px rgba(0, 240, 255, 0.08), inset 0 0 20px rgba(0, 240, 255, 0.03);
  }

  .stage-node.success {
    background: rgba(0, 255, 170, 0.04);
    border-color: rgba(0, 255, 170, 0.15);
  }

  .stage-node.error {
    background: rgba(255, 50, 50, 0.05);
    border-color: rgba(255, 50, 50, 0.2);
  }

  .stage-node.skipped {
    opacity: 0.45;
    border-style: dashed;
  }

  .stage-node.pulse {
    animation: node-pulse 1.5s infinite;
  }

  .stage-emoji {
    font-size: 1.2rem;
    width: 28px;
    text-align: center;
    flex-shrink: 0;
    filter: drop-shadow(0 0 3px rgba(0, 240, 255, 0.3));
  }

  .stage-info {
    flex: 1;
    min-width: 0;
  }

  .stage-label {
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
  }

  .stage-node.idle .stage-label { color: rgba(255, 255, 255, 0.35); }
  .stage-node.active .stage-label { color: #00f0ff; }
  .stage-node.success .stage-label { color: #00ffaa; }
  .stage-node.error .stage-label { color: #ff5555; }

  .stage-detail {
    color: rgba(255, 255, 255, 0.45);
    font-size: 0.7rem;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }

  .stage-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    flex-shrink: 0;
  }

  .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid rgba(0, 240, 255, 0.2);
    border-top-color: #00f0ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .check {
    color: #00ffaa;
    font-size: 0.9rem;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(0, 255, 170, 0.5);
  }

  .cross {
    color: #ff5555;
    font-size: 0.9rem;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(255, 50, 50, 0.5);
  }

  .skip {
    color: rgba(255, 255, 255, 0.3);
    font-size: 0.85rem;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.15);
  }

  /* Sub-actions (expanded execution details) */
  .sub-actions {
    margin-left: 40px;
    padding: 0.25rem 0;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }

  .sub-action {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.6);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.02);
  }

  .sub-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .sub-dot.running { background: #00f0ff; box-shadow: 0 0 6px #00f0ff; }
  .sub-dot.success { background: #00ffaa; }
  .sub-dot.error { background: #ff5555; }

  .sub-type {
    color: rgba(255, 255, 255, 0.75);
    text-transform: capitalize;
    font-weight: 500;
  }

  .sub-target {
    color: rgba(255, 255, 255, 0.35);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 150px;
  }

  /* Agent routing info */
  .routing-info {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.75rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    flex-wrap: wrap;
  }

  .routing-label {
    color: rgba(255, 255, 255, 0.4);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .agent-chip {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.1), rgba(120, 100, 255, 0.1));
    color: #b8b0ff;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: capitalize;
    border: 1px solid rgba(120, 100, 255, 0.2);
  }

  @keyframes node-pulse {
    0% { box-shadow: 0 0 8px rgba(0, 240, 255, 0.1), inset 0 0 8px rgba(0, 240, 255, 0.02); }
    50% { box-shadow: 0 0 20px rgba(0, 240, 255, 0.2), inset 0 0 16px rgba(0, 240, 255, 0.05); }
    100% { box-shadow: 0 0 8px rgba(0, 240, 255, 0.1), inset 0 0 8px rgba(0, 240, 255, 0.02); }
  }

  @keyframes skeleton-shimmer {
    100% { transform: translateX(100%); }
  }

  @keyframes spin {
    100% { transform: rotate(360deg); }
  }

  /* ── Thought Visualization Styles ── */

  .thought-toggle {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    width: 22px;
    height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.75rem;
    transition: all 0.2s;
  }

  .thought-toggle:hover, .thought-toggle.active {
    background: rgba(120, 100, 255, 0.15);
    border-color: rgba(120, 100, 255, 0.4);
  }

  .decision-badge {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin-top: 3px;
    padding: 0.2rem 0.4rem;
    background: rgba(255, 170, 0, 0.06);
    border: 1px solid rgba(255, 170, 0, 0.15);
    border-radius: 4px;
    font-size: 0.6rem;
    color: rgba(255, 200, 80, 0.9);
  }

  .decision-icon { font-size: 0.65rem; }

  .stage-timing {
    font-size: 0.6rem;
    color: rgba(0, 255, 170, 0.5);
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Collapsible Thought Stream */
  .thought-stream {
    margin-top: 0.75rem;
    padding: 0.5rem;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(120, 100, 255, 0.15);
    border-radius: 8px;
    max-height: 260px;
    overflow-y: auto;
  }

  .thought-stream.empty {
    max-height: none;
  }

  .thought-stream-header {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .thought-stream-title {
    color: rgba(180, 160, 255, 0.9);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .thought-stream-icon { font-size: 0.85rem; }

  .thought-count {
    background: rgba(120, 100, 255, 0.15);
    color: rgba(180, 160, 255, 0.9);
    padding: 0.05rem 0.35rem;
    border-radius: 8px;
    font-size: 0.6rem;
    font-weight: 600;
  }

  .thought-accordion-list {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .thought-accordion {
    border: 1px solid rgba(120, 100, 255, 0.12);
    border-radius: 6px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.02);
  }

  .thought-accordion.open {
    border-color: rgba(120, 100, 255, 0.28);
    background: rgba(120, 100, 255, 0.05);
  }

  .thought-accordion-header {
    width: 100%;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto auto;
    align-items: center;
    gap: 0.45rem;
    padding: 0.4rem 0.5rem;
    border: 0;
    background: transparent;
    color: inherit;
    cursor: pointer;
    text-align: left;
  }

  .thought-accordion-header:hover {
    background: rgba(120, 100, 255, 0.08);
  }

  .thought-preview {
    min-width: 0;
    color: rgba(255, 255, 255, 0.52);
    font-size: 0.64rem;
    line-height: 1.3;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .thought-chevron {
    color: rgba(180, 160, 255, 0.65);
    font-size: 0.55rem;
    width: 12px;
    text-align: center;
  }

  .thought-stream-list {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 0.2rem 0.45rem 0.45rem;
  }

  .thought-entry {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    padding: 0.2rem 0.3rem;
    font-size: 0.65rem;
    border-radius: 3px;
    color: rgba(255, 255, 255, 0.6);
    line-height: 1.3;
  }

  .thought-entry.thought { color: rgba(180, 160, 255, 0.7); }
  .thought-entry.decision { color: rgba(255, 200, 80, 0.8); }
  .thought-entry.metric { color: rgba(0, 255, 170, 0.7); }
  .thought-entry.error { color: rgba(255, 80, 80, 0.8); }

  .thought-seq {
    color: rgba(255, 255, 255, 0.25);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    min-width: 20px;
    flex-shrink: 0;
  }

  .thought-stage-chip {
    background: rgba(0, 240, 255, 0.08);
    color: rgba(0, 240, 255, 0.6);
    padding: 0 0.3rem;
    border-radius: 3px;
    font-size: 0.55rem;
    font-weight: 600;
    text-transform: uppercase;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .thought-type {
    color: rgba(255, 255, 255, 0.35);
    font-size: 0.55rem;
    font-weight: 700;
    text-transform: uppercase;
    min-width: 48px;
    flex-shrink: 0;
  }

  .thought-content {
    flex: 1;
    min-width: 0;
    word-break: break-word;
  }

  .thought-empty {
    color: rgba(255, 255, 255, 0.42);
    font-size: 0.68rem;
    padding: 0.2rem 0.1rem;
  }

  .thought-summary {
    margin-top: 0.65rem;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    border: 1px solid rgba(120, 100, 255, 0.16);
    border-radius: 999px;
    background: rgba(120, 100, 255, 0.06);
    color: rgba(180, 160, 255, 0.72);
    padding: 0.25rem 0.55rem;
    font-size: 0.66rem;
    cursor: pointer;
  }

  .thought-summary:hover {
    border-color: rgba(120, 100, 255, 0.35);
    background: rgba(120, 100, 255, 0.1);
  }
</style>
