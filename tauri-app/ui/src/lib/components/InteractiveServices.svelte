<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";
  import { companion } from "../stores/companion";

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

  type GnomonStatus = { state?: string; last_error?: string };

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
        return `Unavailable: ${(gnomon.last_error || "not loaded").slice(0, 60)}`;
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
</script>

<section class="service-strip" aria-label="Interactive services">
  <div class="service-heading">
    <span>Interactive services</span>
    <button type="button" onclick={onconfigure}>Configure</button>
  </div>

  <div class="service-list">
    <div class="service" class:active={Boolean(riskGate.enabled && reachable)}>
      <span class="service-dot"></span>
      <span class="service-copy">
        <strong>World model</strong>
        <small>{worldModelDetail}</small>
      </span>
    </div>
    {#if gnomon.state && gnomon.state !== "disabled"}
      <div class="service" class:active={reachable && (gnomon.state === "ready" || gnomon.state === "loading")}>
        <span class="service-dot"></span>
        <span class="service-copy">
          <strong>Gnomon</strong>
          <small>{gnomonDetail}</small>
        </span>
      </div>
    {/if}
    <div class="service" class:active={voiceActive && reachable}>
      <span class="service-dot"></span>
      <span class="service-copy">
        <strong>Voice</strong>
        <small>{voiceActive ? "Step narration ready" : "Narration is off"}</small>
      </span>
    </div>
    <div class="service" class:active={interferenceActive && reachable}>
      <span class="service-dot"></span>
      <span class="service-copy">
        <strong>Interference</strong>
        <small>{interferenceDetail}</small>
      </span>
    </div>
    <div class="service" class:active={suggestionActive && reachable}>
      <span class="service-dot"></span>
      <span class="service-copy">
        <strong>Suggestions</strong>
        <small>{suggestionActive ? "Verified follow-ups ready" : "Follow-ups are off"}</small>
      </span>
    </div>
    <div class="service active">
      <span class="service-dot"></span>
      <span class="service-copy">
        <strong>Coordinator</strong>
        <small>{coordinatorDetail}</small>
      </span>
    </div>
  </div>
</section>

<style>
  .service-strip {
    position: relative;
    z-index: 2;
    margin: 0 12px 8px;
    padding: 8px 10px;
    border: 1px solid color-mix(in srgb, var(--border) 88%, var(--accent));
    border-radius: 8px;
    background: color-mix(in srgb, var(--bg-secondary) 92%, transparent);
  }

  .service-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 7px;
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .service-heading button {
    border: 0;
    background: transparent;
    color: var(--accent);
    font: inherit;
    cursor: pointer;
  }

  .service-list {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 8px;
  }

  .service {
    display: flex;
    align-items: center;
    min-width: 0;
    gap: 7px;
    color: var(--text-muted);
  }

  .service-dot {
    width: 7px;
    height: 7px;
    flex: 0 0 auto;
    border-radius: 50%;
    background: var(--danger);
    box-shadow: 0 0 7px color-mix(in srgb, var(--danger) 70%, transparent);
  }

  .service.active .service-dot {
    background: var(--success);
    box-shadow: 0 0 7px color-mix(in srgb, var(--success) 70%, transparent);
  }

  .service-copy {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 1px;
  }

  .service-copy strong {
    color: var(--text);
    font-size: 11px;
    font-weight: 650;
  }

  .service-copy small {
    overflow: hidden;
    font-size: 9px;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  @media (max-width: 920px) {
    .service-list {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
