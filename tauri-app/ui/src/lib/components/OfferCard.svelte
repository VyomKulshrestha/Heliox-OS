<script lang="ts">
  /** One offer Heliox made on its own: what it offers, and the four answers a person would give. */
  import { Sparkles } from "@lucide/svelte";
  import { offers, type OfferCard } from "../stores/offers";

  let { card }: { card: OfferCard } = $props();
  const why = $derived($offers.whys[card.id]);
</script>

<article class="offer-card" role="status" aria-live="polite" data-testid="offer-card">
  <header>
    <span class="icon" aria-hidden="true"><Sparkles size={15} /></span>
    <strong class="title"><span class="sr-only">Offer: </span>{card.text}</strong>
  </header>
  {#if why}
    <p class="why" data-testid="offer-why">{why}</p>
  {/if}
  <div class="actions">
    <button type="button" class="primary" onclick={() => offers.answer(card, "yes")}>Yes</button>
    <button type="button" onclick={() => offers.answer(card, "not_today")}>Not today</button>
    {#if !why}
      <button type="button" onclick={() => offers.answer(card, "why")}>Why?</button>
    {/if}
    <button type="button" class="quiet" onclick={() => offers.answer(card, "silence")}>Don't offer this</button>
  </div>
</article>

<style>
  .offer-card {
    display: grid;
    gap: 6px;
    padding: 10px 12px 12px;
    pointer-events: auto;
    background: var(--bg-elevated);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow);
    animation: arrive var(--dur-slow) var(--ease-out) both;
  }

  header {
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  .icon {
    display: inline-flex;
    flex: none;
    padding-top: 2px;
    color: var(--accent);
  }

  .title {
    flex: 1;
    min-width: 0;
    font-size: var(--text-sm);
    font-weight: 600;
    line-height: 1.45;
    color: var(--text-primary);
  }

  .why {
    font-size: var(--text-sm);
    line-height: 1.5;
    color: var(--text-secondary);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 2px;
  }

  .actions button {
    height: 28px;
    padding: 0 10px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    transition: background var(--dur-fast) ease;
  }

  .actions button:hover {
    background: var(--bg-hover);
  }

  .actions .primary {
    color: var(--on-accent);
    background: var(--accent);
    border-color: transparent;
  }

  .actions .primary:hover {
    background: var(--accent-hover);
  }

  .actions .quiet {
    color: var(--text-secondary);
    background: transparent;
    border-color: transparent;
  }

  @keyframes arrive {
    from {
      opacity: 0;
      transform: translateY(8px) scale(0.98);
    }
    to {
      opacity: 1;
      transform: none;
    }
  }
</style>
