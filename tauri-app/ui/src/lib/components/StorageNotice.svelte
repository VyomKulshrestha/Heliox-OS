<script lang="ts">
  import { storageNotice } from "../stores/storageNotice";
</script>

<!-- A full disk or a read-only data folder, in the daemon's own words, under the title bar on every screen. It is only
  there while the problem lasts, so no screen moves otherwise (stores/storageNotice.ts). -->
{#if $storageNotice}
  <div class="storage-notice" role="alert" data-testid="storage-notice">{$storageNotice.message}</div>
{/if}

<style>
  .storage-notice {
    padding: 6px 16px;
    border-bottom: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);
    background: var(--warning-bg);
    color: var(--warning);
    font-size: 12px;
    line-height: 1.45;
  }
</style>
