<script lang="ts">
  import { ArrowDown } from "lucide-svelte";

  interface Props {
    show: boolean;
    onclick: () => void;
  }

  let { show, onclick }: Props = $props();
</script>

<button
  class="scroll-fab"
  class:visible={show}
  aria-label="Scroll to bottom"
  title="Scroll to bottom"
  {onclick}
>
  <ArrowDown size={18} />
</button>

<style>
  .scroll-fab {
    position: absolute;
    bottom: 16px;
    right: 20px;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: var(--accent);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.2s ease, transform 0.2s ease;
    pointer-events: none;
    z-index: 10;
  }

  .scroll-fab.visible {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
  }

  .scroll-fab:hover {
    background: var(--accent-hover);
  }
</style>
