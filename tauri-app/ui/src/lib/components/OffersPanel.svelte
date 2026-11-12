<script lang="ts" module>
  import type { OffersStatus } from "../stores/offers";

  export const MODES: { value: OffersStatus["mode"]; label: string; hint: string }[] = [
    { value: "auto", label: "Once it knows you", hint: "On after you tell Heliox something about yourself" },
    { value: "on", label: "On", hint: "Offers from your habits, goals and usual apps" },
    { value: "off", label: "Off", hint: "No offers; you can still ask what to do next" },
  ];

  /** One line on what is happening now, in plain words. */
  export function stateLine(status: OffersStatus | null): string {
    if (!status) return "";
    if (status.mode === "off") return "Offers are off.";
    if (status.waiting_for_profile)
      return "Waiting until Heliox knows something about you: tell it your name, or what you use the computer for.";
    if (status.active)
      return `On: at most ${status.per_hour} an hour and ${status.per_day} a day, and only when you're not busy.`;
    return "";
  }
</script>

<script lang="ts">
  import { onMount } from "svelte";
  import { offersStatus, setOffersMode, unsilenceOffer } from "../stores/offers";

  let status = $state<OffersStatus | null>(null);
  let error = $state("");
  let busy = $state(false);

  async function run(action: () => Promise<OffersStatus>) {
    busy = true;
    try {
      const result = await action();
      if (result.status === "ok" && typeof result.mode === "string") {
        // A daemon that answers without the list (an older one) must not break Settings.
        status = { ...result, silenced: Array.isArray(result.silenced) ? result.silenced : [] };
        error = "";
      } else if (result.status !== "ok") {
        error = result.message || "That did not work.";
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Heliox is not connected.";
    } finally {
      busy = false;
    }
  }

  onMount(() => void run(offersStatus));
</script>

<section class="offers-panel" aria-labelledby="offers-heading" data-testid="offers-panel">
  <div class="section-title">
    <div>
      <h3 id="offers-heading">Offers</h3>
      <p>
        Now and then Heliox offers something you usually do at about this time, like opening the app you start the day
        with, logging a habit or a reminder for a goal that has gone quiet. Everything is worked out on this computer.
        Ask "why did you offer that?" any time; nothing happens unless you say yes.
      </p>
    </div>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}

  <div class="block">
    <span class="block-label" id="offers-mode">When to offer</span>
    <div class="modes" role="radiogroup" aria-labelledby="offers-mode">
      {#each MODES as mode (mode.value)}
        <label class="mode" class:selected={status?.mode === mode.value}>
          <input
            type="radio"
            name="offers-mode"
            value={mode.value}
            checked={status?.mode === mode.value}
            disabled={busy || !status}
            onchange={() => void run(() => setOffersMode(mode.value))}
          />
          <span class="mode-name">{mode.label}</span>
          <span class="mode-hint">{mode.hint}</span>
        </label>
      {/each}
    </div>
    {#if stateLine(status)}<p class="state" data-testid="offers-state">{stateLine(status)}</p>{/if}
  </div>

  <div class="block">
    <span class="block-label">Offers you turned down for good</span>
    {#if status && status.silenced.length}
      <ul class="silenced">
        {#each status.silenced as item (item.key)}
          <li>
            <span>{item.label}</span>
            <button type="button" disabled={busy} onclick={() => void run(() => unsilenceOffer(item.key))}
              >Offer again</button
            >
          </li>
        {/each}
      </ul>
    {:else}
      <p>None. "Don't offer this" on an offer, or saying "don't offer that again", puts it here.</p>
    {/if}
  </div>
</section>

<style>
  .offers-panel {
    margin-top: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-secondary);
    overflow: hidden;
  }

  .section-title {
    padding: 16px 20px;
  }

  h3 {
    margin: 0 0 5px;
    font-size: var(--text-lg);
  }

  p {
    max-width: 72ch;
    margin: 0;
    color: var(--text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: var(--danger);
    background: var(--danger-bg);
    font-size: var(--text-sm);
  }

  .block {
    display: grid;
    gap: 8px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }

  .block-label {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .modes {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 8px;
  }

  .mode {
    display: grid;
    gap: 2px;
    padding: 9px 11px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) ease,
      background var(--dur-fast) ease;
  }

  .mode input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  .mode:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .mode.selected {
    border-color: var(--accent-line);
    background: var(--accent-muted);
  }

  .mode-name {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .mode-hint {
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .state {
    font-size: var(--text-xs);
  }

  .silenced {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .silenced li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    font-size: var(--text-sm);
    color: var(--text-primary);
  }

  .silenced button {
    flex: none;
    height: 28px;
    padding: 0 10px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .silenced button:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  @media (max-width: 760px) {
    .section-title,
    .block {
      padding-inline: 14px;
    }
  }
</style>
