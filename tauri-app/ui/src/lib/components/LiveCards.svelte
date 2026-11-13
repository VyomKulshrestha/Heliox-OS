<script lang="ts">
  /** The live companion's corrections and Heliox's offers: small cards over any tab. A correction carries the
   * evidence and a fix; an offer carries Yes, Not today, Why and Don't offer this. */
  import { CircleAlert, Lightbulb, ShieldAlert, X } from "@lucide/svelte";
  import { live, type LiveCard } from "../stores/live";
  import { offers } from "../stores/offers";
  import OfferCard from "./OfferCard.svelte";

  function fixLabel(card: LiveCard): string {
    if (card.fix_kind === "ask") return "Ask Heliox";
    if (card.fix_kind === "command") return "Copy safer command";
    return "Copy fix";
  }

  /** Said before the title for screen readers; sighted readers get the icon's shape and colour. */
  const URGENCY_WORDS: Record<LiveCard["urgency"], string> = {
    urgent: "Urgent:",
    normal: "Worth a look:",
    low: "Minor:",
  };

  function useFix(card: LiveCard) {
    if (card.fix_kind === "ask") void live.askHeliox(card);
    else void live.copyFix(card);
  }
</script>

{#if $live.cards.length || $offers.cards.length}
  <div class="live-cards" aria-label="Live companion">
    {#each $offers.cards as offer (offer.id)}
      <OfferCard card={offer} />
    {/each}
    {#each $live.cards as card (card.id)}
      {@const note = $live.notes[card.id]}
      {@const held = $live.held[card.id]}
      {#if held}
        <article class="live-card held" role="status" aria-live="polite" data-testid="live-card-held">
          <p class="message">{held.sentence}</p>
          <div class="actions">
            <button type="button" onclick={() => live.undoHold(card, held)}>Undo</button>
          </div>
        </article>
      {:else}
        <article
          class="live-card {card.urgency}"
          class:done={note === "Fixed"}
          role={card.urgency === "urgent" ? "alert" : "status"}
          aria-live={card.urgency === "urgent" ? "assertive" : "polite"}
          data-testid="live-card"
        >
          <header>
            <span class="icon" aria-hidden="true">
              {#if card.urgency === "urgent"}<ShieldAlert size={15} />{:else if card.urgency === "normal"}<CircleAlert
                  size={15}
                />{:else}<Lightbulb size={15} />{/if}
            </span>
            <strong class="title"
              ><span class="sr-only">{`${URGENCY_WORDS[card.urgency] ?? ""} `}</span>{card.title}</strong
            >
            {#if note}<span class="note">{note}</span>{:else if card.spoken}<span
                class="said"
                data-testid="live-card-said">Said aloud</span
              >{/if}
            <button
              type="button"
              class="close"
              aria-label="Not now"
              title="Not now"
              onclick={() => live.feedback(card, "dismissed")}
            >
              <X size={14} />
            </button>
          </header>
          <p class="message">{card.message}</p>
          {#if card.evidence}
            <p class="evidence">
              <span class="sr-only">About: </span><q>{card.evidence}</q>{#if card.app}<span class="where"
                  >in {card.app}</span
                >{/if}
            </p>
          {/if}
          {#if $live.explanations[card.id]}
            <p class="explanation">{$live.explanations[card.id]}</p>
          {/if}
          <div class="actions">
            {#if card.fix && note !== "Fixed"}
              <button type="button" class="primary" onclick={() => useFix(card)}>{fixLabel(card)}</button>
            {/if}
            {#if card.detail && !$live.explanations[card.id]}
              <button type="button" onclick={() => live.explain(card)}>Why?</button>
            {/if}
            <button type="button" onclick={() => live.feedback(card, "helpful")}>Thanks</button>
            <button type="button" class="quiet" onclick={() => live.feedback(card, "mute")}>Don't mention these</button>
          </div>
        </article>
      {/if}
    {/each}
  </div>
{/if}

<style>
  .live-cards {
    position: fixed;
    right: 14px;
    bottom: 70px;
    z-index: var(--z-overlay);
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: min(360px, calc(100vw - 28px));
    pointer-events: none;
  }

  .live-card {
    display: grid;
    gap: 6px;
    padding: 10px 12px 12px;
    pointer-events: auto;
    background: var(--bg-elevated);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow);
    animation: arrive var(--dur-slow) var(--ease-out) both;
    transition: opacity var(--dur) ease;
  }

  /* Urgency shows in the header icon (shape and colour) and, for urgent cards, a 1 px danger edge. */
  .live-card.urgent {
    border-color: var(--danger);
  }

  .live-card.done {
    opacity: 0.55;
  }

  .live-card.held {
    grid-template-columns: 1fr auto;
    align-items: center;
  }

  .live-card.held .actions {
    margin-top: 0;
  }

  header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .icon {
    display: inline-flex;
    flex: none;
    color: var(--accent);
  }

  .urgent .icon {
    color: var(--danger);
  }

  .low .icon {
    color: var(--text-muted);
  }

  .title {
    flex: 1;
    min-width: 0;
    font-family: var(--font-display);
    font-size: var(--text-base);
    font-weight: 600;
    color: var(--text-primary);
  }

  .note {
    flex: none;
    padding: 1px 7px;
    font-size: var(--text-2xs);
    font-weight: 600;
    color: var(--success);
    background: var(--success-bg);
    border-radius: var(--radius-pill);
  }

  /* Heliox said this card aloud: a quiet mark, so a person who heard something can find what it was. */
  .said {
    flex: none;
    font-size: var(--text-2xs);
    color: var(--text-muted);
  }

  .close {
    display: inline-grid;
    flex: none;
    place-items: center;
    width: 24px;
    height: 24px;
    color: var(--text-muted);
    border-radius: var(--radius-sm);
  }

  .close:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .message,
  .explanation {
    font-size: var(--text-sm);
    line-height: 1.5;
    color: var(--text-primary);
  }

  .explanation {
    color: var(--text-secondary);
  }

  .evidence {
    overflow: hidden;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .evidence q {
    font-family: var(--font-mono);
    color: var(--text-primary);
  }

  .where {
    margin-left: 0.35em;
    color: var(--text-muted);
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
