<script lang="ts">
  /** The live companion's corrections: small cards over any tab, each with the evidence and a fix. */
  import { CircleAlert, Lightbulb, ShieldAlert, X } from "@lucide/svelte";
  import { live, type LiveCard } from "../stores/live";

  function fixLabel(card: LiveCard): string {
    if (card.fix_kind === "ask") return "Ask Heliox";
    if (card.fix_kind === "command") return "Copy safer command";
    return "Copy fix";
  }

  function useFix(card: LiveCard) {
    if (card.fix_kind === "ask") void live.askHeliox(card);
    else void live.copyFix(card);
  }
</script>

{#if $live.cards.length}
  <div class="live-cards" aria-label="Live companion">
    {#each $live.cards as card (card.id)}
      {@const note = $live.notes[card.id]}
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
          <strong class="title">{card.title}</strong>
          {#if note}<span class="note">{note}</span>{/if}
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
          {#if card.fix}
            <button type="button" class="primary" onclick={() => useFix(card)}>{fixLabel(card)}</button>
          {/if}
          {#if card.detail && !$live.explanations[card.id]}
            <button type="button" onclick={() => live.explain(card)}>Why?</button>
          {/if}
          <button type="button" onclick={() => live.feedback(card, "helpful")}>Thanks</button>
          <button type="button" class="quiet" onclick={() => live.feedback(card, "mute")}>Don't mention these</button>
        </div>
      </article>
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
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    box-shadow: var(--shadow);
    animation: arrive var(--dur-slow) var(--ease-out) both;
    transition: opacity var(--dur) ease;
  }

  .live-card.urgent {
    border-left-color: var(--danger);
  }

  .live-card.low {
    border-left-color: var(--border-strong);
  }

  .live-card.done {
    opacity: 0.55;
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
