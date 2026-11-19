<script lang="ts">
  // Where the face sits in the conversation view. With face.pack "off" (the default) this renders nothing and loads
  // nothing: the face's code is its own chunk and its images live in public/faces, fetched only once a face is chosen.
  // Choosing another face replaces the component, so the old pack's images and timers go with it.
  import type { Component } from "svelte";
  import { settings } from "../stores/settings";
  import { OFF, validPackId } from "../face/pack";

  let { onshown }: { onshown?: (shown: boolean) => void } = $props();

  const pack = $derived.by(() => {
    const chosen = $settings.face?.pack ?? OFF;
    return chosen !== OFF && validPackId(chosen) ? chosen : "";
  });

  let Face: Component<{ pack: string }> | null = $state(null);
  let failed = $state(false);

  $effect(() => {
    onshown?.(Boolean(pack) && !failed);
    if (!pack || Face) return;
    import("./Face.svelte")
      .then((module) => (Face = module.default as Component<{ pack: string }>))
      .catch(() => (failed = true));
  });
</script>

{#if pack && Face}
  <div class="face-host">
    {#key pack}
      <Face {pack} />
    {/key}
  </div>
{/if}

<style>
  .face-host {
    position: absolute;
    top: 10px;
    right: 12px;
    width: var(--face-size, 120px);
    z-index: var(--z-raised);
    pointer-events: none;
  }
</style>
