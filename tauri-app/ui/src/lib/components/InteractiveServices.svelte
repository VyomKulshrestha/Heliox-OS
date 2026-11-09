<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";
  import { companion } from "../stores/companion";
  import { ChevronDown } from "@lucide/svelte";

  let { onconfigure } = $props<{ onconfigure: () => void }>();

  type NarrationStatus = {
    enabled?: boolean;
    narrate_steps?: boolean;
    interrupt_on_risk?: boolean;
    proactive_review_enabled?: boolean;
    live_corrections_enabled?: boolean;
    follow_up_enabled?: boolean;
  };

  type RiskGateStatus = {
    enabled?: boolean;
    weights_loaded?: boolean;
    model_version?: string;
    last_evaluation?: {
      risk_score?: number;
      worst_action_type?: string | null;
    } | null;
  };

  // `reason` explains an unavailable model, e.g. "Needs 2.5 GB of free memory, 1.8 GB free now."
  type GnomonStatus = { state?: string; last_error?: string; reason?: string };

  let narration = $state<NarrationStatus>({});
  let gnomon = $state<GnomonStatus>({ state: "disabled" });
  let lastGnomonEscalated = $state<boolean | null>(null);
  let riskGate = $state<RiskGateStatus>({});
  let reachable = $state(false);
  let lastCompanionDecision = $state("");
  let refreshTimer: ReturnType<typeof setInterval> | null = null;

  async function refresh() {
    try {
      const [nextNarration, nextRiskGate] = await Promise.all([call("narration_status"), call("risk_gate_status")]);
      narration = nextNarration as NarrationStatus;
      riskGate = nextRiskGate as RiskGateStatus;
      reachable = true;
    } catch {
      reachable = false;
    }
    try {
      const review = (await call("autonomy_review_status")) as { gnomon?: GnomonStatus };
      gnomon = review.gnomon ?? { state: "disabled" };
    } catch {
      // Older daemons have no autonomy review; the other services still show.
    }
  }

  function onDaemonNotification(method: string, params: unknown) {
    const payload = (params ?? {}) as Record<string, unknown>;
    if (method === "world_model_assessment") {
      riskGate = {
        ...riskGate,
        enabled: true,
        last_evaluation: {
          risk_score: Number(payload.combined_score ?? payload.risk_score ?? 0),
          worst_action_type: String(payload.worst_action_type ?? "") || null,
        },
      };
      const review = payload.gnomon as { escalate?: boolean } | null | undefined;
      if (review && typeof review === "object") {
        lastGnomonEscalated = Boolean(review.escalate);
        gnomon = { ...gnomon, state: "ready" };
      }
    } else if (method === "companion_plan_review") {
      lastCompanionDecision = String(payload.decision ?? "").toUpperCase();
    }
  }

  let worldModelDetail = $derived.by(() => {
    if (!reachable) return "Daemon unavailable";
    if (!riskGate.enabled) return "Disabled in Settings";
    const score = Number(riskGate.last_evaluation?.risk_score);
    if (Number.isFinite(score)) {
      return `Last plan ${Math.round(score * 100)}% risk`;
    }
    return riskGate.weights_loaded ? "Learned weights ready" : "Rule fallback ready";
  });

  let gnomonDetail = $derived.by(() => {
    if (!reachable) return "Daemon unavailable";
    switch (gnomon.state) {
      case "ready":
        return lastGnomonEscalated === null
          ? "Reviewing autonomous plans"
          : lastGnomonEscalated
            ? "Last review asked for approval"
            : "Last review: no concerns";
      case "loading":
        return "Loading on the GPU";
      case "cold":
        return "Loads when a task starts";
      case "unavailable":
        return gnomon.reason || `Unavailable: ${(gnomon.last_error || "not loaded").slice(0, 60)}`;
      default:
        return "Off in config";
    }
  });

  let voiceActive = $derived(Boolean(narration.enabled && narration.narrate_steps));
  let interferenceActive = $derived(Boolean(narration.proactive_review_enabled || narration.live_corrections_enabled));
  let suggestionActive = $derived(Boolean(narration.follow_up_enabled));
  let coordinatorDetail = $derived(
    $companion.humanSpeaking
      ? "Listening · companion speech paused"
      : $companion.activeChannel
        ? `Speaking · ${$companion.activeChannel.replaceAll("_", " ")}`
        : $companion.queued > 0
          ? `${$companion.queued} message${$companion.queued === 1 ? "" : "s"} queued`
          : "Single priority voice ready",
  );
  let interferenceDetail = $derived(
    lastCompanionDecision
      ? `Last review: ${lastCompanionDecision}`
      : narration.live_corrections_enabled
        ? "Review and live correction armed"
        : narration.proactive_review_enabled
          ? "Plan review armed"
          : "Disabled in Settings",
  );

  onMount(() => {
    void refresh();
    onNotification(onDaemonNotification);
    refreshTimer = setInterval(() => void refresh(), 15_000);
  });

  onDestroy(() => {
    offNotification(onDaemonNotification);
    if (refreshTimer) clearInterval(refreshTimer);
  });

  type ServiceState = "on" | "off" | "down";
  type ServiceRow = { name: string; detail: string; state: ServiceState };

  // Off by choice is neutral; red is kept for "the daemon cannot be reached".
  function rowState(on: boolean): ServiceState {
    return !reachable ? "down" : on ? "on" : "off";
  }

  let services = $derived.by((): ServiceRow[] => {
    const rows: ServiceRow[] = [
      { name: "World model", detail: worldModelDetail, state: rowState(Boolean(riskGate.enabled)) },
    ];
    if (gnomon.state && gnomon.state !== "disabled") {
      rows.push({
        name: "Gnomon",
        detail: gnomonDetail,
        state: rowState(gnomon.state === "ready" || gnomon.state === "loading"),
      });
    }
    rows.push(
      {
        name: "Voice",
        detail: voiceActive ? "Step narration ready" : "Narration is off",
        state: rowState(voiceActive),
      },
      { name: "Interference", detail: interferenceDetail, state: rowState(interferenceActive) },
      {
        name: "Suggestions",
        detail: suggestionActive ? "Verified follow-ups ready" : "Follow-ups are off",
        state: rowState(suggestionActive),
      },
      { name: "Coordinator", detail: coordinatorDetail, state: "on" },
    );
    return rows;
  });
  let onCount = $derived(services.filter((row) => row.state === "on").length);
  let expanded = $state(false);
</script>

<section class="service-strip" aria-label="Interactive services">
  <div class="service-summary">
    <button
      type="button"
      class="summary-toggle"
      aria-expanded={expanded}
      aria-controls="service-list"
      onclick={() => (expanded = !expanded)}
    >
      <span class="dots" aria-hidden="true">
        {#each services as row (row.name)}
          <span class="service-dot {row.state}"></span>
        {/each}
      </span>
      <span class="summary-text">
        {#if !reachable}
          Daemon unavailable
        {:else}
          {onCount} of {services.length} services on
        {/if}
      </span>
      <ChevronDown size={14} aria-hidden="true" class={expanded ? "chevron open" : "chevron"} />
    </button>
    <button type="button" class="configure" onclick={onconfigure}>Configure</button>
  </div>

  <ul class="service-list" id="service-list" hidden={!expanded}>
    {#each services as row (row.name)}
      <li class="service {row.state}">
        <span class="service-dot {row.state}" aria-hidden="true"></span>
        <span class="service-copy">
          <strong>{row.name}</strong>
          <small title={row.detail}>{row.detail}</small>
        </span>
      </li>
    {/each}
  </ul>
</section>

<style>
  .service-strip {
    position: relative;
    z-index: var(--z-raised);
    margin: 8px 12px 0;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--bg-secondary);
  }

  .service-summary {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 3px 6px 3px 3px;
  }

  .summary-toggle {
    display: inline-flex;
    flex: 1;
    align-items: center;
    gap: 9px;
    min-width: 0;
    padding: 4px 8px;
    color: var(--text-secondary);
    font-size: var(--text-xs);
    border-radius: var(--radius-sm);
    text-align: left;
  }

  .summary-toggle:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .summary-toggle :global(.chevron) {
    flex: none;
    transition: transform var(--dur) var(--ease-out);
  }

  .summary-toggle :global(.chevron.open) {
    transform: rotate(180deg);
  }

  .summary-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .dots {
    display: inline-flex;
    gap: 4px;
  }

  .configure {
    flex: none;
    padding: 4px 8px;
    color: var(--accent);
    font-size: var(--text-xs);
    font-weight: 600;
    border-radius: var(--radius-sm);
  }

  .configure:hover {
    background: var(--accent-muted);
  }

  .service-list[hidden] {
    display: none;
  }

  .service-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 8px 14px;
    padding: 6px 12px 10px;
    list-style: none;
    border-top: 1px solid var(--border);
  }

  .service {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    padding-top: 4px;
  }

  .service-dot {
    flex: none;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
    opacity: 0.6;
  }

  .service-dot.on {
    background: var(--success);
    opacity: 1;
  }

  .service-dot.down {
    background: var(--danger);
    opacity: 1;
  }

  .service-copy {
    display: grid;
    min-width: 0;
    gap: 1px;
  }

  .service-copy strong {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .service.off .service-copy strong {
    color: var(--text-secondary);
  }

  .service-copy small {
    overflow: hidden;
    font-size: var(--text-xs);
    color: var(--text-muted);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
</style>
