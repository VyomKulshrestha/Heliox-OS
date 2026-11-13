<script lang="ts">
  import { onMount, type Component } from "svelte";

  /**
   * A Settings panel that lives in its own chunk and loads when it is about to be shown: when it comes within a
   * screen of the visible part of Settings, or once Settings has been open and quiet for a moment, whichever is first.
   * Settings shows everything in one scrolling page, and its single chunk outgrew the production build's 500 KiB
   * budget (vite.config.ts) with the panels far down the page in it. Until it loads, the panel takes no room and
   * shows nothing: the panel's own loading, empty and error states follow as before. If its chunk cannot be loaded at
   * all (the app's files damaged), one quiet line says so.
   */
  let { load }: { load: () => Promise<{ default: Component }> } = $props();

  let Panel: Component | null = $state(null);
  let failed = $state(false);
  let anchor: HTMLDivElement | undefined = $state();

  /** How long Settings stays open and quiet before the rest of its panels load anyway. */
  const IDLE_MS = 1200;

  onMount(() => {
    let started = false;
    const begin = () => {
      if (started) return;
      started = true;
      load()
        .then((module) => (Panel = module.default))
        .catch(() => (failed = true));
    };
    const timer = setTimeout(begin, IDLE_MS);
    if (typeof IntersectionObserver === "undefined" || !anchor) {
      begin();
      return () => clearTimeout(timer);
    }
    const observer = new IntersectionObserver((entries) => entries.some((e) => e.isIntersecting) && begin(), {
      rootMargin: "100% 0px",
    });
    observer.observe(anchor);
    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  });
</script>

{#if Panel}
  <Panel />
{:else if failed}
  <p class="lazy-failed">This part of Settings could not be loaded. Close Settings and open it again.</p>
{:else}
  <div class="lazy-anchor" bind:this={anchor} aria-hidden="true"></div>
{/if}

<style>
  .lazy-anchor {
    height: 1px;
  }

  .lazy-failed {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-muted, inherit);
  }
</style>
