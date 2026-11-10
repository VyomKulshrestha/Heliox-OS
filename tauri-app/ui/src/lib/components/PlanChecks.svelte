<script lang="ts">
  import { ShieldAlert, ShieldCheck, ShieldOff } from "@lucide/svelte";
  import type { GnomonCheck, MachineState, PlanChecks, WorldModelCheck } from "../stores/session";

  interface Props {
    checks?: PlanChecks | null;
    /** The confirm dialog already shows the world model's interruption. */
    hideWorldModel?: boolean;
    /** The confirm dialog already shows Gnomon's reasons. */
    hideGnomon?: boolean;
  }

  let { checks = null, hideWorldModel = false, hideGnomon = false }: Props = $props();

  type Line = { text: string; raised: boolean; ran: boolean };

  const pct = (fraction: number) => `${Math.round(fraction * 100)}%`;
  const known = (value: number | null | undefined): value is number =>
    typeof value === "number" && Number.isFinite(value);

  function machineText(check: WorldModelCheck): string {
    const before: MachineState | null | undefined = check.observed;
    const after: MachineState | null | undefined = check.predicted;
    const parts: string[] = [];
    const sizes = check.measured_sizes ?? [];
    if (sizes.length) {
      // A measured copy or write says more about the disk than the per-step state does.
      for (const size of sizes) {
        const usage =
          known(size.usage_before) && known(size.usage_after)
            ? `, ${size.volume} ${pct(size.usage_before)} to ${pct(size.usage_after)}`
            : "";
        parts.push(`${size.path_label} measured at ${size.size_text}${usage}`);
      }
    } else if (known(before?.disk_usage_fraction) && known(after?.disk_usage_fraction)) {
      parts.push(`disk ${pct(before.disk_usage_fraction)} to ${pct(after.disk_usage_fraction)}`);
    }
    if (known(before?.process_count) && known(after?.process_count)) {
      parts.push(`processes ${before.process_count} to ${after.process_count}`);
    }
    if (!parts.length) return "";
    const joined = parts.join(", ");
    return `${joined.charAt(0).toUpperCase()}${joined.slice(1)}.`;
  }

  function worldModelLine(check: WorldModelCheck): Line {
    if (!check.enabled) return { text: "Off in Settings, no prediction made.", raised: false, ran: false };
    if (check.unavailable) {
      return {
        text: check.raised ? "Could not assess this plan, so it needs your approval." : "Could not assess this plan.",
        raised: Boolean(check.raised),
        ran: false,
      };
    }
    const score = known(check.score) ? check.score : 0;
    const reasons = (check.reasons ?? []).join("; ");
    let verdict: string;
    if (check.raised) verdict = `Raised risk (${pct(score)})${reasons ? `: ${reasons}` : ""}.`;
    else if (score > 0) verdict = `Low risk (${pct(score)})${reasons ? `: ${reasons}` : ""}.`;
    else verdict = "No machine risk predicted.";
    const machine = machineText(check);
    return { text: machine ? `${verdict} ${machine}` : verdict, raised: Boolean(check.raised), ran: true };
  }

  function gnomonLine(check: GnomonCheck): Line {
    const took = known(check.review_seconds) ? ` Took ${check.review_seconds.toFixed(1)} s.` : "";
    switch (check.state) {
      case "reviewed": {
        if (check.escalate) {
          const reasons = (check.reasons ?? []).join(" ");
          return { text: `Asked for your approval. ${reasons || ""}`.trim() + took, raised: true, ran: true };
        }
        const inScope = check.scores?.in_scope;
        const verdict = known(inScope)
          ? `Matches what you asked (${pct(inScope)} in scope).`
          : "Reviewed, nothing raised.";
        return { text: verdict + took, raised: false, ran: true };
      }
      case "loading":
        return { text: "Still loading, did not review this plan.", raised: false, ran: false };
      case "cold":
        return { text: "Not loaded yet, did not review this plan.", raised: false, ran: false };
      case "unavailable": {
        const why = (check.reason ?? "").trim();
        return {
          text: `Not loaded, did not review this plan.${why ? ` ${why}${/[.!?]$/.test(why) ? "" : "."}` : ""}`,
          raised: false,
          ran: false,
        };
      }
      case "timed_out":
        return { text: "Timed out, did not review this plan.", raised: false, ran: false };
      default:
        return { text: "Did not review this plan.", raised: false, ran: false };
    }
  }

  let worldModel = $derived(!hideWorldModel && checks?.worldModel ? worldModelLine(checks.worldModel) : null);
  let gnomon = $derived(!hideGnomon && checks?.gnomon ? gnomonLine(checks.gnomon) : null);
</script>

{#if worldModel || gnomon}
  <div class="plan-checks" role="group" aria-label="Checked before running">
    {#if worldModel}
      <p class="check" class:raised={worldModel.raised}>
        <span class="check-icon" aria-hidden="true">
          {#if worldModel.raised}<ShieldAlert size={13} />{:else if worldModel.ran}<ShieldCheck
              size={13}
            />{:else}<ShieldOff size={13} />{/if}
        </span>
        <span><span class="check-name">World model:</span> {worldModel.text}</span>
      </p>
    {/if}
    {#if gnomon}
      <p class="check" class:raised={gnomon.raised}>
        <span class="check-icon" aria-hidden="true">
          {#if gnomon.raised}<ShieldAlert size={13} />{:else if gnomon.ran}<ShieldCheck size={13} />{:else}<ShieldOff
              size={13}
            />{/if}
        </span>
        <span
          ><span class="check-name">Gnomon:</span>
          {gnomon.text}
          <span class="check-rule">It can ask for approval, never grant it.</span></span
        >
      </p>
    {/if}
  </div>
{/if}

<style>
  .plan-checks {
    display: grid;
    gap: 2px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
  }

  .check {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    max-width: 90ch;
    font-size: var(--text-xs);
    line-height: 1.45;
    color: var(--text-secondary);
  }

  .check-icon {
    display: inline-grid;
    flex: none;
    place-items: center;
    margin-top: 2px;
    color: var(--text-muted);
  }

  .check-name {
    font-weight: 600;
  }

  .check-rule {
    color: var(--text-muted);
  }

  .check.raised {
    color: var(--warning);
  }

  .check.raised .check-icon {
    color: var(--warning);
  }
</style>
