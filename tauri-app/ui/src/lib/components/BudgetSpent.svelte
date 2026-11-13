<script lang="ts">
  // Settings, beside the monthly limit: what this month has cost so far (budget_stats), and every model that was
  // costed without a current price (pilot.models.prices), so a figure that is high on purpose says why.
  import { onMount } from "svelte";
  import { call } from "../api/daemon";

  type Stats = { cost_usd?: number; limit_usd?: number; estimated_high?: string[] };

  let stats = $state<Stats | null>(null);
  let note = $state("");

  function usd(value: number | undefined): string {
    const n = Number(value ?? 0);
    return `$${n.toFixed(n < 1 ? 4 : 2)}`;
  }

  onMount(async () => {
    try {
      const found = (await call("budget_stats")) as Stats | undefined;
      stats = found && typeof found.cost_usd === "number" ? found : null;
      note = stats ? "" : "Spending is not counted yet.";
    } catch {
      note = "Heliox did not answer, so this month's spending could not be read.";
    }
  });
</script>

<div class="setting-row" data-testid="budget-spent">
  <div class="setting-info">
    <span class="setting-label">Spent this month</span>
    {#if stats}
      <span class="setting-desc">
        {usd(stats.cost_usd)}{stats.limit_usd ? ` of ${usd(stats.limit_usd)}` : ""}, as Heliox counts it. The provider's
        own bill is the final word.
      </span>
      {#each stats.estimated_high ?? [] as line (line)}
        <span class="setting-desc">{line[0].toUpperCase() + line.slice(1)}.</span>
      {/each}
    {:else if note}
      <span class="setting-desc">{note}</span>
    {/if}
  </div>
</div>
