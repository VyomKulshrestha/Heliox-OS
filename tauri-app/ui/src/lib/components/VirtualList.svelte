<!-- VirtualList.svelte — Svelte 5 virtualised scroller for chat messages -->
<script lang="ts" generics="T">
  /**
   * VirtualList — Svelte 5 virtualised scroller.
   *
   * Only the messages visible in the viewport (± overscan) are mounted in the
   * DOM.  Top and bottom padding spacers keep the scroll-bar thumb at the right
   * position without rendering off-screen nodes.
   *
   * Usage:
   *   <VirtualList bind:this={ref} items={messages}>
   *     {#snippet item(msg, index)}
   *       <MessageRow {msg} />
   *     {/snippet}
   *   </VirtualList>
   */

  import { tick, onDestroy } from "svelte";
  import type { Snippet } from "svelte";

  // ── Props ──────────────────────────────────────────────────────────────────

  interface Props<T> {
    /** Full list of items to virtualise. */
    items: T[];
    /** Fallback height (px) used before a row's real height is measured. */
    estimatedItemHeight?: number;
    /** Extra items rendered above and below the visible window. */
    overscan?: number;
    /** Slot — receives (item, index). */
    item: Snippet<[T, number]>;
    /** Optional footer rendered below all items. */
    footer?: Snippet;
  }

  let {
    items,
    estimatedItemHeight = 80,
    overscan = 5,
    item: itemSnippet,
    footer: footerSnippet,
  }: Props<T> = $props();

  // ── Internal state ─────────────────────────────────────────────────────────

  /** Measured heights for every item (may be estimated until observed). */
  let heights = $state<number[]>([]);

  /** Index of the first rendered item. */
  let visStart = $state(0);

  /** Index of the last rendered item (inclusive). */
  let visEnd = $state(-1);

  /** The scrollable container element. */
  let scrollerEl: HTMLDivElement | undefined = $state();

  /** Whether the user was at the bottom before the last items update. */
  let wasAtBottom = true;

  /** ResizeObserver watching rendered row heights. */
  let ro: ResizeObserver | null = null;

  // ── Derived geometry ───────────────────────────────────────────────────────

  /** Pre-calculated cumulative heights for O(1) lookups. */
  let cumulativeHeights = $derived.by(() => {
    let sum = 0;
    return heights.map(h => {
      sum += h;
      return sum;
    });
  });

  /** Total virtual height of all items (px). */
  let totalHeight = $derived(
    cumulativeHeights.length > 0 ? cumulativeHeights[cumulativeHeights.length - 1] : 0
  );

  /** Pixels of invisible content above the rendered window. */
  let paddingTop = $derived(
    visStart > 0 && cumulativeHeights.length > 0 ? cumulativeHeights[visStart - 1] : 0
  );

  /** Pixels of invisible content below the rendered window. */
  let paddingBottom = $derived(
    visEnd >= 0 && visEnd < cumulativeHeights.length - 1
      ? totalHeight - cumulativeHeights[visEnd]
      : 0
  );

  /** Slice of items currently in the DOM. */
  let visibleItems = $derived(
    visEnd >= visStart ? items.slice(visStart, visEnd + 1).map((value, i) => ({
      value,
      index: visStart + i,
    })) : []
  );

  // ── Height initialisation ──────────────────────────────────────────────────

  /**
   * Keep `heights` in sync with `items`.
   * Newly appended items get the estimated height until observed.
   */
  $effect(() => {
    const len = items.length;
    if (heights.length === len) return;

    if (len > heights.length) {
      // Items added — extend with estimates
      const extra = Array(len - heights.length).fill(estimatedItemHeight);
      heights = [...heights, ...extra];
    } else {
      // Items removed — truncate
      heights = heights.slice(0, len);
    }
    
    // Recalculate window so new items can be rendered
    tick().then(recalcWindow);
  });

  // ── Scroll-to-bottom on new messages ──────────────────────────────────────

  let prevItemCount = 0;

  $effect(() => {
    const newCount = items.length;
    if (newCount !== prevItemCount) {
      if (wasAtBottom) {
        tick().then(() => scrollToBottom());
      }
      prevItemCount = newCount;
    }
  });

  // ── Window calculation ─────────────────────────────────────────────────────

  function recalcWindow() {
    if (!scrollerEl) return;

    const scrollTop = scrollerEl.scrollTop;
    const clientH = scrollerEl.clientHeight;

    // Check if we are at (or very near) the bottom before recalculating
    wasAtBottom =
      scrollerEl.scrollTop + scrollerEl.clientHeight >=
      scrollerEl.scrollHeight - 8;

    // Walk the heights array to find the first and last visible item
    let cumulative = 0;
    let newStart = 0;
    let newEnd = heights.length - 1;
    let foundStart = false;

    for (let i = 0; i < heights.length; i++) {
      const bottom = cumulativeHeights[i];

      if (!foundStart && bottom > scrollTop) {
        newStart = Math.max(0, i - overscan);
        foundStart = true;
      }

      if (foundStart && bottom > scrollTop + clientH) {
        newEnd = Math.min(heights.length - 1, i + overscan);
        break;
      }
    }

    // Guard: if list is shorter than viewport just render everything
    if (!foundStart) {
      newStart = 0;
      newEnd = heights.length - 1;
    }

    visStart = newStart;
    visEnd = newEnd;
  }

  // ── ResizeObserver — measure real heights ─────────────────────────────────

  function attachResizeObserver() {
    if (!scrollerEl) return;

    ro?.disconnect();
    ro = new ResizeObserver((entries) => {
      let changed = false;
      for (const entry of entries) {
        const el = entry.target as HTMLElement;
        const idx = Number(el.dataset.vlIndex);
        if (isNaN(idx)) continue;
        const h = entry.contentRect.height;
        if (heights[idx] !== h) {
          heights[idx] = h;
          changed = true;
        }
      }
      if (changed) recalcWindow();
    });

    // Observe all currently rendered rows
    observeRenderedRows();
  }

  function observeRenderedRows() {
    if (!scrollerEl || !ro) return;
    const rows = scrollerEl.querySelectorAll<HTMLElement>("[data-vl-index]");
    rows.forEach((el) => ro!.observe(el));
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  $effect(() => {
    if (!scrollerEl) return;

    scrollerEl.addEventListener("scroll", recalcWindow, { passive: true });
    attachResizeObserver();
    recalcWindow();

    return () => {
      scrollerEl?.removeEventListener("scroll", recalcWindow);
    };
  });

  // Re-observe whenever the visible slice changes (new rows mounted)
  $effect(() => {
    // Depend on visStart + visEnd so this re-runs when the window shifts
    void visStart;
    void visEnd;
    tick().then(observeRenderedRows);
  });

  onDestroy(() => {
    ro?.disconnect();
  });

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Scroll the list to the very bottom. */
  export function scrollToBottom() {
    if (scrollerEl) {
      scrollerEl.scrollTop = scrollerEl.scrollHeight;
      wasAtBottom = true;
    }
  }
</script>

<!-- Scrollable container -->
<div class="vl-scroller" bind:this={scrollerEl}>
  <!-- Top spacer: pushes rendered rows to the correct vertical position -->
  <div class="vl-spacer" style="height: {paddingTop}px;" aria-hidden="true"></div>

  <!-- Rendered rows (only the visible window + overscan) -->
  {#each visibleItems as { value, index } (index)}
    <div class="vl-row" data-vl-index={index}>
      {@render itemSnippet(value, index)}
    </div>
  {/each}

  <!-- Bottom spacer: keeps scrollbar thumb proportional -->
  <div class="vl-spacer" style="height: {paddingBottom}px;" aria-hidden="true"></div>

  {#if footerSnippet}
    {@render footerSnippet()}
  {/if}
</div>

<style>
  .vl-scroller {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    /* Inherit padding from the parent .results rule in App.svelte */
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
  }

  .vl-spacer {
    flex-shrink: 0;
    width: 100%;
  }

  .vl-row {
    flex-shrink: 0;
    margin-bottom: 8px;
  }
</style>
