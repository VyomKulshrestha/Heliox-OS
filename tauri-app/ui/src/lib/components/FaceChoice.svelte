<script lang="ts">
  // Settings, Appearance: which face the conversation view shows (face.pack), off by default. The choices are the
  // installed face packs by the names their manifests give. Choosing reads only the packs' small manifest files: no
  // face image loads until a face is shown. A pack may name the voice that suits the face: the daemon itself switches
  // to it whenever face.pack is set (here, by another client, by voice or in the config file) while the person has
  // never chosen a voice and the voice is on the computer. A face whose voice is not on the computer says so in the
  // list before it is chosen, and choosing it asks one question in place, for any face and any engine: download its
  // voice (the current voice speaks until the new one is ready, and the switch comes between two replies), or use the
  // face with the voice the person has (recorded, so nothing asks again). face_voice does both; nothing is fetched
  // unless the first is pressed.
  import { onDestroy, onMount } from "svelte";
  import { call, offNotification, onNotification } from "../api/daemon";
  import { settings } from "../stores/settings";
  import { installedPacks, OFF, type PackEntry } from "../face/pack";

  interface FaceVoicePlan {
    status?: string;
    message?: string;
    switched?: boolean;
    voice?: string;
    action?: "none" | "same" | "offer" | "fetch" | "unavailable" | "kept" | "downloading";
    say?: string;
    download_bytes?: number;
    download_ok?: boolean;
    download_reason?: string;
  }

  let packs: PackEntry[] = $state([]);
  let note = $state("");
  let busy = $state(false);
  let plan = $state<FaceVoicePlan | null>(null);
  // Each pack's plan before it is chosen, so the list can say which face needs a download for its voice.
  let ahead = $state<Record<string, FaceVoicePlan>>({});
  // A face picked in the list whose voice is not on the computer: not saved until one of the two buttons is pressed.
  let pending = $state("");
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
  // The face the question is about: the one just picked in the list, or the chosen face when its voice is not here.
  const deciding = $derived(pending || (!switchedSay && plan?.action === "fetch" ? chosen : ""));
  const decidingPlan = $derived(pending ? ahead[pending] : deciding ? plan : null);
  const decidingName = $derived(packs.find((pack) => pack.id === deciding)?.name ?? deciding);
  const pairing = $derived(switchedSay || (deciding ? "" : plan?.say || ""));
  const offered = $derived(!switchedSay && plan && plan.action === "offer" ? (plan.voice ?? "") : "");

  /** "about 330 MB", "about half a megabyte": a download's size in words. */
  function sizeWords(bytes: number | undefined): string {
    const size = bytes ?? 0;
    return size < 1_000_000 ? "about half a megabyte" : `about ${Math.round(size / 1_000_000)} MB`;
  }

  function label(pack: PackEntry): string {
    const before = ahead[pack.id];
    return before?.action === "fetch" && pack.id !== chosen
      ? `${pack.name} (its voice needs a download of ${sizeWords(before.download_bytes)})`
      : pack.name;
  }

  const ready = (method: string, params: unknown) => {
    if (method !== "face_voice_ready") return;
    const done = params as { status?: string; voice?: string; say?: string; message?: string };
    if (done.status === "ok") {
      switched = { voice: done.voice ?? "", say: done.say ?? "" };
      void settings.reloadFromDaemon();
    } else {
      note = `The voice was not downloaded: ${done.message ?? "the download stopped."}`;
    }
    void askAhead();
  };

  onMount(() => {
    onNotification(ready);
    void installedPacks().then((found) => (packs = found));
  });
  onDestroy(() => offNotification(ready));

  async function faceVoice(pack: PackEntry | undefined, apply: string): Promise<FaceVoicePlan | null> {
    const voice = pack?.voice?.kokoro;
    if (!pack || !voice) return null;
    try {
      return await call<FaceVoicePlan>("face_voice", { voice, hint: pack.voice?.hint ?? "", pack: pack.id, apply });
    } catch {
      return null;
    }
  }

  async function askFaceVoice(apply: "" | "accept"): Promise<FaceVoicePlan | null> {
    const mine = ++asked;
    if (chosen === OFF || !entry?.voice?.kokoro) {
      plan = null;
      return null;
    }
    const result = await faceVoice(entry, apply);
    if (result?.status === "error") {
      if (apply) note = result.message || "The voice was not changed.";
      return null;
    }
    if (mine === asked) plan = result; // a switch is still reported to its caller when a newer question came
    return result;
  }

  async function askAhead() {
    const found: Record<string, FaceVoicePlan> = {};
    for (const pack of packs) {
      const result = pack.voice?.kokoro ? await faceVoice(pack, "") : null;
      if (result && result.status !== "error") found[pack.id] = result;
    }
    ahead = found;
  }

  // The offer follows the chosen face and the voice settings (another window or Settings, Voice may change them).
  $effect(() => {
    void chosen;
    void entry;
    void engine;
    void currentVoice;
    void askFaceVoice("");
  });
  $effect(() => {
    void packs;
    void engine;
    void currentVoice;
    void askAhead();
  });

  async function save(pack: string, select?: HTMLSelectElement): Promise<boolean> {
    const before = chosen;
    const result = await settings.updateSectionDetailed("face", { pack });
    if (!result.ok) {
      note = result.message || "Not changed.";
      if (select) select.value = before; // the list shows what is saved, not the choice that was refused
      return false;
    }
    return true;
  }

  async function choose(select: HTMLSelectElement) {
    busy = true;
    note = "";
    pending = "";
    switched = { voice: "", say: "" };
    try {
      if (select.value !== OFF && ahead[select.value]?.action === "fetch") {
        pending = select.value; // asked in place first; nothing is saved or fetched yet
        return;
      }
      if (!(await save(select.value, select))) return;
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

  // "Use this face and download its voice" (download) or "Use this face with the voice I have" (keep).
  async function decide(apply: "download" | "keep") {
    const face = deciding;
    const pack = packs.find((p) => p.id === face);
    busy = true;
    note = "";
    try {
      if (pending && !(await save(pending))) return;
      pending = "";
      const answer = await faceVoice(pack, apply);
      if (!answer) note = "Heliox did not answer, so nothing changed.";
      else if (answer.status === "error") note = answer.message || "Not changed.";
      else plan = answer;
      await settings.reloadFromDaemon();
      void askAhead();
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
    {#if deciding && decidingPlan}
      <span class="setting-desc" aria-live="polite" data-testid="face-voice-question">
        {decidingName}'s voice, {decidingPlan.voice}, is not on this computer. Its voice needs a download of
        {sizeWords(decidingPlan.download_bytes)}. Until it is ready, {currentVoice || "the voice you have"} keeps speaking.
      </span>
      <span class="setting-desc">
        <button
          type="button"
          class="btn-secondary"
          disabled={busy || decidingPlan.download_ok === false}
          onclick={() => decide("download")}>Use this face and download its voice</button
        >
        <button type="button" class="btn-secondary face-voice-use" disabled={busy} onclick={() => decide("keep")}
          >Use this face with the voice I have</button
        >
      </span>
      {#if decidingPlan.download_ok === false && decidingPlan.download_reason}
        <span class="setting-desc">Cannot download now: {decidingPlan.download_reason}</span>
      {/if}
    {/if}
    {#if note}<span class="setting-desc" aria-live="polite">{note}</span>{/if}
  </div>
  <select
    class="input-md"
    aria-label="Face"
    disabled={busy}
    value={pending || chosen}
    onchange={(event) => choose(event.currentTarget)}
  >
    <option value={OFF}>Off</option>
    {#each packs as pack (pack.id)}
      <option value={pack.id}>{label(pack)}</option>
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
