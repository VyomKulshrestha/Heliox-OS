<script lang="ts">
  // Settings, Appearance: which face the conversation view shows (face.pack), off by default. The choices are the
  // installed face packs by the names their manifests give; any face goes with any voice. Choosing reads only the
  // packs' small manifest files: no face image loads until a face is shown.
  import { onMount } from "svelte";
  import { settings } from "../stores/settings";
  import { installedPacks, OFF, type PackEntry } from "../face/pack";

  let packs: PackEntry[] = $state([]);
  let note = $state("");
  let busy = $state(false);

  const chosen = $derived($settings.face?.pack ?? OFF);
  const credit = $derived(packs.find((pack) => pack.id === chosen)?.credit ?? "");
  const engine = $derived($settings.voice?.tts_engine ?? "");
  // What moves the mouth with the chosen voice, in one sentence (only once a face is chosen).
  const voiceNote = $derived(
    chosen === OFF
      ? ""
      : engine === "kokoro_tts"
        ? ""
        : engine === "pocket_tts"
          ? "With this voice the mouth follows how loud Heliox speaks."
          : "With the system voice the face stays still while Heliox talks, because that voice plays by itself and Heliox cannot time the mouth.",
  );

  onMount(() => {
    void installedPacks().then((found) => (packs = found));
  });

  async function choose(select: HTMLSelectElement) {
    const before = chosen;
    busy = true;
    note = "";
    try {
      const result = await settings.updateSectionDetailed("face", { pack: select.value });
      if (!result.ok) {
        note = result.message || "Not changed.";
        select.value = before; // the list shows what is saved, not the choice that was refused
      }
    } finally {
      busy = false;
    }
  }
</script>

<div class="setting-row" data-testid="face-choice">
  <div class="setting-info">
    <span class="setting-label">Face</span>
    <span class="setting-desc">
      A face beside the conversation that moves its mouth as Heliox speaks. Off by default; works with any voice.
    </span>
    {#if credit}<span class="setting-desc">{credit}</span>{/if}
    {#if voiceNote}<span class="setting-desc">{voiceNote}</span>{/if}
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <select
    class="input-md"
    aria-label="Face"
    disabled={busy}
    value={chosen}
    onchange={(event) => choose(event.currentTarget)}
  >
    <option value={OFF}>Off</option>
    {#each packs as pack (pack.id)}
      <option value={pack.id}>{pack.name}</option>
    {/each}
    {#if chosen !== OFF && !packs.some((pack) => pack.id === chosen)}
      <option value={chosen}>{chosen}</option>
    {/if}
  </select>
</div>
