<script lang="ts">
  import { _ } from "svelte-i18n";
  import { humanControl } from "../stores/humanControl";
</script>

{#if $humanControl.holder === "human"}
  <div class="human-control-banner holding" role="status" aria-live="polite">
    <span class="copy">
      <strong>{$_("human_control.you_have_control")}</strong>
      {#if $humanControl.reason}
        <small>{$_("human_control.reason", { values: { reason: $humanControl.reason } })}</small>
      {/if}
    </span>
    <button class="control-btn" disabled={$humanControl.busy} onclick={() => humanControl.release()}>
      {$_("human_control.resume")}
    </button>
  </div>
{:else if $humanControl.activeJobs.length > 0}
  <div class="human-control-banner working" role="status" aria-live="polite">
    <span class="copy">{$_("human_control.agent_working")}</span>
    <button class="control-btn" disabled={$humanControl.busy} onclick={() => humanControl.take()}>
      {$_("human_control.take")}
    </button>
  </div>
{/if}
{#if $humanControl.error}
  <div class="human-control-error" role="alert">
    {$_("human_control.error", { values: { message: $humanControl.error } })}
  </div>
{/if}

<style>
  .human-control-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin: 6px 12px;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
  }

  .human-control-banner.holding {
    background: rgba(234, 179, 8, 0.14);
    border: 1px solid rgba(234, 179, 8, 0.5);
  }

  .human-control-banner.working {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.4);
  }

  .copy {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .control-btn {
    flex-shrink: 0;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid currentColor;
    background: transparent;
    color: inherit;
    cursor: pointer;
  }

  .control-btn:disabled {
    opacity: 0.6;
    cursor: wait;
  }

  .human-control-error {
    margin: 0 12px 6px;
    font-size: 12px;
    color: #f87171;
  }
</style>
