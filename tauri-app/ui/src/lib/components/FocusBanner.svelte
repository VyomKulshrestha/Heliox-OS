<script lang="ts">
  import { onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";

  type FocusStatus = { status?: string; active: boolean; until: number | null; held: number };

  let focus = $state<FocusStatus | null>(null);

  function apply(next: Partial<FocusStatus> | null | undefined) {
    if (next && typeof next.active === "boolean")
      focus = { active: next.active, until: next.until ?? null, held: next.held ?? 0 };
  }

  function onChange(method: string, params: unknown) {
    if (method === "focus_status") apply(params as FocusStatus);
    if (method === "focus_ended") focus = { active: false, until: null, held: 0 };
  }

  async function stop() {
    await call("focus_stop");
  }

  function until(epoch: number): string {
    return new Date(epoch * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  onMount(() => {
    onNotification(onChange);
    void call<FocusStatus>("focus_status")
      .then(apply)
      .catch(() => {});
    return () => offNotification(onChange);
  });
</script>

{#if focus?.active && focus.until}
  <div class="focus-banner" data-testid="focus-banner" role="status">
    <span
      >Focusing until {until(focus.until)}. Reminders and the daily brief wait{focus.held
        ? ` (${focus.held} held)`
        : ""}; approvals still come through.</span
    >
    <button type="button" onclick={stop}>Stop</button>
  </div>
{/if}

<style>
  .focus-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 14px;
    margin: 0 0 8px 0;
    border: 1px solid var(--accent);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--accent) 10%, var(--bg-tertiary));
    color: var(--text-primary);
    font-size: 12px;
  }
  button {
    padding: 4px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-hover);
    color: var(--text-primary);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
</style>
