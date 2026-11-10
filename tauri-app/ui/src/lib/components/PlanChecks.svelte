<script lang="ts" module>
  import type { CriticCheck, CriticFinding } from "../stores/session";

  /** Whether the safety review's concerns should look like a warning (only WARN and BLOCK, or a failed review). */
  export function criticRaised(critic: CriticCheck | null | undefined): boolean {
    if (!critic || critic.verdict === "SKIPPED") return false;
    return !critic.ran || critic.verdict === "WARN" || critic.verdict === "BLOCK";
  }

  /** The concerns the confirm dialog shows under plan step `step` (1-based); process notes stay in the row. */
  export function findingsForStep(critic: CriticCheck | null | undefined, step: number): CriticFinding[] {
    if (!critic?.ran) return [];
    return critic.findings.filter((finding) => finding.step === step && finding.severity !== "low");
  }
</script>

<script lang="ts">
  import { ShieldAlert, ShieldCheck, ShieldOff } from "@lucide/svelte";
  import type { GnomonCheck, MachineState, PlanChecks, WorldModelCheck } from "../stores/session";

  interface Props {
    checks?: PlanChecks | null;
    /** The confirm dialog already shows the world model's interruption. */
    hideWorldModel?: boolean;
    /** The confirm dialog already shows Gnomon's reasons; its notes still show here. */
    hideGnomon?: boolean;
    /** Plan steps (1-based) whose safety-review concerns the confirm dialog shows next to the step. */
    stepsShownElsewhere?: number[];
    /** The confirm dialog: one line per check, with concerns and notes behind one closed "Show details". */
    compact?: boolean;
  }

  let {
    checks = null,
    hideWorldModel = false,
    hideGnomon = false,
    stepsShownElsewhere = [],
    compact = false,
  }: Props = $props();

  /** Concerns listed before "Show more"; the rest, and every process note, wait behind it. */
  const SHORT_LIST = 2;
  const listId = $props.id();
  const notesId = `${listId}-gnomon`;
  const detailsId = `${listId}-details`;

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
      // Each part opens with a fixed word, so capitalising it never changes a file name's case.
      for (const size of sizes) {
        const usage =
          known(size.usage_before) && known(size.usage_after)
            ? `, ${size.volume} ${pct(size.usage_before)} to ${pct(size.usage_after)}`
            : "";
        parts.push(`measured ${size.path_label} at ${size.size_text}${usage}`);
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

  type GnomonLine = Line & { notes: string[] };

  /** The daemon's note giving the reason scope was not reviewed; the line says it once, in the daemon's words. */
  const SCOPE_NOTE = /^\s*scope not reviewed\b/i;

  /** What Gnomon's scores say it checked, besides scope. */
  function gnomonChecked(check: GnomonCheck): string[] {
    const checked: string[] = [];
    if (known(check.scores?.severe_harm)) checked.push("harm if wrong");
    if (known(check.scores?.permitted_by_user)) checked.push("instructions found in tool or screen text");
    return checked;
  }

  function gnomonReviewed(check: GnomonCheck, scopeNote: string): string {
    const took = known(check.review_seconds) ? ` Took ${check.review_seconds.toFixed(1)} s.` : "";
    const cutShort = check.complete === false;
    const scopeSkipped = check.scope_reviewed === false;
    const scope = scopeSkipped ? ` ${scopeNote || "Scope not reviewed."}` : "";
    if (check.escalate) {
      const reasons = (check.reasons ?? []).join(" ");
      const lead = cutShort ? "Cut short, but asked for your approval." : "Asked for your approval.";
      return `${`${lead} ${reasons}`.trim()}${scope}${took}`;
    }
    const inScope = check.scores?.in_scope;
    if (!scopeSkipped && !cutShort) {
      const verdict = known(inScope)
        ? `Matches what you asked (${pct(inScope)} in scope).`
        : "Reviewed, nothing raised.";
      return verdict + took;
    }
    const checked = [
      ...(!scopeSkipped && known(inScope) ? [`scope (${pct(inScope)} in scope)`] : []),
      ...gnomonChecked(check),
    ];
    const found = checked.length ? `checked ${checked.join(" and ")}, nothing raised.` : "";
    const lead = cutShort
      ? found
        ? `Cut short. So far it ${found}`
        : "Cut short before any check finished."
      : found
        ? `${found.charAt(0).toUpperCase()}${found.slice(1)}`
        : "";
    return `${lead}${scope}`.trim() + took;
  }

  function gnomonLine(check: GnomonCheck, reasonsShownAbove: boolean): GnomonLine | null {
    const notes = check.state === "reviewed" ? (check.notes ?? []).filter((note) => note.trim()) : [];
    const scopeNote = check.scope_reviewed === false ? (notes.find((note) => SCOPE_NOTE.test(note)) ?? "") : "";
    const others = notes.filter((note) => note !== scopeNote);
    if (reasonsShownAbove) {
      // The dialog shows Gnomon's reasons above the row; the line keeps only what is not shown there.
      if (!others.length && check.scope_reviewed !== false) return null;
      const scope = check.scope_reviewed === false ? ` ${scopeNote || "Scope not reviewed."}` : "";
      return { text: `Asked for your approval, reasons above.${scope}`, raised: true, ran: true, notes: others };
    }
    return { ...gnomonStateLine(check, scopeNote), notes: others };
  }

  function gnomonStateLine(check: GnomonCheck, scopeNote: string): Line {
    switch (check.state) {
      case "reviewed":
        return { text: gnomonReviewed(check, scopeNote), raised: Boolean(check.escalate), ran: true };
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

  /** `listed`: concerns not shown beside a step, most important first; `notes`: process notes. */
  type CriticLine = Line & { listed: CriticFinding[]; notes: CriticFinding[]; elsewhere: boolean };

  const SKIPPED_TEXT: Record<string, string> = {
    low_risk_heuristic: "Did not run, this plan scored low risk.",
    no_destructive_authority: "Did not run, no step in this plan is destructive.",
  };

  function criticLine(critic: CriticCheck): CriticLine {
    const none = { listed: [], notes: [], elsewhere: false };
    if (critic.verdict === "SKIPPED") {
      return { text: SKIPPED_TEXT[critic.skipped ?? ""] ?? "Did not run.", raised: false, ran: false, ...none };
    }
    if (!critic.ran) {
      return { text: "Could not run, so check each step yourself.", raised: true, ran: false, ...none };
    }
    const shownElsewhere = new Set(stepsShownElsewhere);
    const besideStep = (finding: CriticFinding) =>
      finding.step !== null && finding.severity !== "low" && shownElsewhere.has(finding.step);
    const main = critic.findings.filter((finding) => finding.severity !== "low");
    if (critic.verdict === "APPROVE" && !main.length) {
      return { text: "Nothing to flag.", raised: false, ran: true, ...none };
    }
    const verdict =
      critic.verdict === "BLOCK"
        ? "Stopped this plan."
        : critic.verdict === "APPROVE"
          ? "Approved, with notes."
          : "Check before approving.";
    // A downgraded BLOCK's recommendation was written by the daemon, so it is left out.
    const advice = !critic.downgraded && critic.recommendation ? ` ${critic.recommendation}` : "";
    const listed = main.filter((finding) => !besideStep(finding));
    return {
      text: `${verdict}${advice}`,
      raised: criticRaised(critic),
      ran: true,
      listed,
      notes: critic.findings.filter((finding) => finding.severity === "low"),
      elsewhere: listed.length < main.length,
    };
  }

  /** "Step 3: ..." unless the critic already opened with the step. */
  function findingText(finding: CriticFinding): string {
    if (finding.step === null || /^\s*step\s+\d+\b/i.test(finding.text)) return finding.text;
    return `Step ${finding.step}: ${finding.text}`;
  }

  let showMore = $state(false);
  let showMoreNotes = $state(false);
  let showDetails = $state(false);
  let worldModel = $derived(!hideWorldModel && checks?.worldModel ? worldModelLine(checks.worldModel) : null);
  let gnomon = $derived(checks?.gnomon ? gnomonLine(checks.gnomon, hideGnomon) : null);
  let critic = $derived(checks?.critic ? criticLine(checks.critic) : null);
  // Plan card: the first concerns and notes show, the rest wait behind their own "Show more".
  let criticShown = $derived(critic ? critic.listed.slice(0, SHORT_LIST) : []);
  let criticMore = $derived(critic ? [...critic.listed.slice(SHORT_LIST), ...critic.notes] : []);
  let notesShown = $derived(gnomon ? gnomon.notes.slice(0, SHORT_LIST) : []);
  let notesMore = $derived(gnomon ? gnomon.notes.slice(SHORT_LIST) : []);
  // Dialog: every concern not under its step and every Gnomon note sit behind one closed control.
  let detailCount = $derived((critic ? critic.listed.length + critic.notes.length : 0) + (gnomon?.notes.length ?? 0));
</script>

{#snippet concern(finding: CriticFinding)}
  <li class:minor={finding.severity === "low"}>
    {#if finding.severity === "low"}<span class="minor-label">Minor:</span>{/if}
    {findingText(finding)}
  </li>
{/snippet}

{#if worldModel || gnomon || critic}
  <div class="plan-checks" role="group" aria-label="Checked before running">
    {#if critic}
      <div class="check" class:raised={critic.raised}>
        <span class="check-icon" aria-hidden="true">
          {#if !critic.ran}<ShieldOff size={13} />{:else if critic.raised}<ShieldAlert size={13} />{:else}<ShieldCheck
              size={13}
            />{/if}
        </span>
        <div class="check-body">
          <p><span class="check-name">Safety review:</span> {critic.text}</p>
          {#if !compact}
            {#if criticShown.length || (showMore && criticMore.length)}
              <ul class="findings" id={listId} aria-label="Safety review concerns">
                {#each criticShown as finding, i (i)}
                  {@render concern(finding)}
                {/each}
                {#if showMore}
                  {#each criticMore as finding, i (i)}
                    {@render concern(finding)}
                  {/each}
                {/if}
              </ul>
            {/if}
            {#if critic.elsewhere}
              <p class="check-rule">Some concerns are shown with the steps they are about.</p>
            {/if}
            {#if criticMore.length}
              <button
                type="button"
                class="more"
                aria-expanded={showMore}
                aria-controls={listId}
                onclick={() => (showMore = !showMore)}
              >
                {showMore ? "Show fewer" : `Show ${criticMore.length} more`}
              </button>
            {/if}
          {/if}
        </div>
      </div>
    {/if}
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
      <div class="check" class:raised={gnomon.raised}>
        <span class="check-icon" aria-hidden="true">
          {#if gnomon.raised}<ShieldAlert size={13} />{:else if gnomon.ran}<ShieldCheck size={13} />{:else}<ShieldOff
              size={13}
            />{/if}
        </span>
        <div class="check-body">
          <p>
            <span class="check-name">Gnomon:</span>
            {gnomon.text}
            <span class="check-rule">It can ask for approval, never grant it.</span>
          </p>
          {#if !compact && notesShown.length}
            <ul class="findings notes" id={notesId} aria-label="Gnomon notes">
              {#each notesShown as note, i (i)}
                <li>{note}</li>
              {/each}
              {#if showMoreNotes}
                {#each notesMore as note, i (i)}
                  <li>{note}</li>
                {/each}
              {/if}
            </ul>
          {/if}
          {#if !compact && notesMore.length}
            <button
              type="button"
              class="more"
              aria-expanded={showMoreNotes}
              aria-controls={notesId}
              onclick={() => (showMoreNotes = !showMoreNotes)}
            >
              {showMoreNotes ? "Show fewer" : `Show ${notesMore.length} more`}
            </button>
          {/if}
        </div>
      </div>
    {/if}
    {#if compact && detailCount}
      <button
        type="button"
        class="more details-toggle"
        aria-expanded={showDetails}
        aria-controls={detailsId}
        onclick={() => (showDetails = !showDetails)}
      >
        {showDetails ? "Hide details" : "Show details"}
      </button>
      {#if showDetails}
        <div class="details" id={detailsId}>
          {#if critic && (critic.listed.length || critic.notes.length)}
            <ul class="findings" aria-label="Safety review concerns">
              {#each [...critic.listed, ...critic.notes] as finding, i (i)}
                {@render concern(finding)}
              {/each}
            </ul>
          {/if}
          {#if gnomon?.notes.length}
            <ul class="findings notes" aria-label="Gnomon notes">
              {#each gnomon.notes as note, i (i)}
                <li>{note}</li>
              {/each}
            </ul>
          {/if}
        </div>
      {/if}
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

  .check-body {
    display: grid;
    gap: 2px;
    min-width: 0;
  }

  .findings {
    display: grid;
    gap: 2px;
    margin: 0;
    padding-left: 16px;
    overflow-wrap: anywhere;
  }

  .check.raised .findings {
    color: var(--text-secondary);
  }

  .findings.notes {
    color: var(--text-secondary);
  }

  .findings .minor {
    color: var(--text-muted);
  }

  .minor-label {
    font-weight: 600;
  }

  .more {
    justify-self: start;
    padding: 0;
    font-size: var(--text-xs);
    color: var(--accent);
    background: none;
    text-decoration: underline;
    text-underline-offset: 2px;
  }

  .details {
    display: grid;
    gap: 4px;
    padding-left: 19px;
    font-size: var(--text-xs);
    line-height: 1.45;
  }

  .details-toggle {
    margin-left: 19px;
  }

  .more:hover {
    color: var(--accent-hover);
  }

  .check.raised {
    color: var(--warning);
  }

  .check.raised .check-icon {
    color: var(--warning);
  }
</style>
