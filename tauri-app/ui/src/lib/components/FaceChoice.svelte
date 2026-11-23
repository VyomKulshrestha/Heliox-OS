<script lang="ts">
  // Settings, Appearance: which face the conversation view shows (face.pack), off by default. The choices are the
  // installed face packs by the names their manifests give. Choosing reads only the packs' small manifest files: no
  // face image loads until a face is shown. A pack may name the voice that suits the face: the daemon itself switches
  // to it whenever face.pack is set (here, by another client, by voice or in the config file) while the person has
  // never chosen a voice. This row only shows what happened and otherwise offers the voice (face_voice).
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";
  import { installedPacks, OFF, type PackEntry } from "../face/pack";

  interface FaceVoicePlan {
    status?: string;
    message?: string;
    switched?: boolean;
    voice?: string;
    action?: "none" | "same" | "offer" | "fetch" | "unavailable";
    say?: string;
  }

  let packs: PackEntry[] = $state([]);
  let note = $state("");
  let busy = $state(false);
  let plan = $state<FaceVoicePlan | null>(null);
  let switched = $state({ voice: "", say: "" }); // what the last switch said, shown while that voice speaks
  let asked = 0; // the newest face_voice question: an older answer arriving late is dropped

  const chosen = $derived($settings.face?.pack ?? OFF);
  const entry = $derived(packs.find((pack) => pack.id === chosen));
  const credit = $derived(entry?.credit ?? "");
  const engine = $derived($settings.voice?.tts_engine ?? "");
  const currentVoice = $derived($settings.voice?.tts_voice ?? "");
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
  const switchedSay = $derived(chosen !== OFF && switched.voice === currentVoice ? switched.say : "");
  const pairing = $derived(switchedSay || plan?.say || "");
  const offered = $derived(
    !switchedSay && plan && (plan.action === "offer" || plan.action === "fetch") ? (plan.voice ?? "") : "",
  );

  onMount(() => {
    void installedPacks().then((found) => (packs = found));
  });

  async function askFaceVoice(apply: "" | "accept"): Promise<FaceVoicePlan | null> {
    const voice = entry?.voice?.kokoro;
    const mine = ++asked;
    if (chosen === OFF || !voice) {
      plan = null;
      return null;
    }
    try {
      const result = await call<FaceVoicePlan>("face_voice", { voice, hint: entry?.voice?.hint ?? "", apply });
      if (result?.status === "error") {
        if (apply) note = result.message || "The voice was not changed.";
        return null;
      }
      if (mine === asked) plan = result; // a switch is still reported to its caller when a newer question came
      return result;
    } catch {
      if (mine === asked) plan = null;
      return null;
    }
  }

  // The offer follows the chosen face and the voice settings (another window or Settings, Voice may change them).
  $effect(() => {
    void chosen;
    void entry;
    void engine;
    void currentVoice;
    void askFaceVoice("");
  });

  async function choose(select: HTMLSelectElement) {
    const before = chosen;
    busy = true;
    note = "";
    switched = { voice: "", say: "" };
    try {
      const result = await settings.updateSectionDetailed("face", { pack: select.value });
      if (!result.ok) {
        note = result.message || "Not changed.";
        select.value = before; // the list shows what is saved, not the choice that was refused
        return;
      }
      if (select.value === OFF) return; // turning the face off changes no voice
      // The daemon has already applied the face's voice if it should; the Voice rows and this one show it at once,
      // and the plan asked again below says what happened.
      await settings.reloadFromDaemon();
      const answer = await askFaceVoice("");
      if (answer?.switched) switched = { voice: answer.voice ?? "", say: answer.say ?? "" };
    } finally {
      busy = false;
    }
  }

  async function useFaceVoice() {
    busy = true;
    note = "";
    try {
      const answer = await askFaceVoice("accept");
      if (answer?.switched) {
        switched = { voice: answer.voice ?? "", say: answer.say ?? "" };
        await settings.reloadFromDaemon(); // the Voice rows and this one show the new voice at once
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
    {#if pairing}
      <span class="setting-desc" aria-live="polite" data-testid="face-voice">
        {pairing}
        {#if offered}
          <button type="button" class="btn-secondary face-voice-use" disabled={busy} onclick={useFaceVoice}>
            Use {offered}
          </button>
        {/if}
      </span>
    {/if}
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

<style>
  .face-voice-use {
    margin-left: 6px;
  }
</style>
