<script lang="ts">
  import WidgetCard from "./WidgetCard.svelte";
  import { onMount } from "svelte";
  import { invoke } from "../api/invoke";
  type FeedItem = {
    title: string;
    time: string;
  };
  let pinned = false;
  function togglePin() {
    pinned = !pinned;
  }
  let feed: FeedItem[] = [];
  async function loadFeed() {
    try {
      const res = await invoke("get_rss_feed");
      if (Array.isArray(res)) feed = res;
    } catch (err) {
      console.error("RSS feed failed", err);
    }
  }
  onMount(() => {
    loadFeed();
    const interval = setInterval(loadFeed, 3000);
    return () => clearInterval(interval);
  });
</script>

<div class="rss-card">
  <WidgetCard title="RSS Feed">
    {#each feed as item}
      <div class="feed-item">
        <p class="title">
          {item.title}
        </p>
        <span>
          {item.time}
        </span>
      </div>
    {/each}
  </WidgetCard>
</div>

<style>
  .rss-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    color: var(--text-primary);
    height: fit-content;
  }
  .feed-item {
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  .title {
    margin: 0;
    color: var(--text-primary);
    font-size: 14px;
    line-height: 1.5;
  }
  span {
    color: var(--text-secondary);
    font-size: 12px;
  }
</style>
