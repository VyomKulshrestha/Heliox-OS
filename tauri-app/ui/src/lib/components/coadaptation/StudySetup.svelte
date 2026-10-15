<script module lang="ts">
  import type { Modality } from "../../coadaptation/types";

  /** What the Setup screen hands to the study host. Nothing of it is written to browser storage (spec 10.1). */
  export interface StudySetupChoice {
    participantKey: string;
    modality: Modality;
    gaze: boolean;
    pointerTypes: Array<"mouse" | "pen" | "touch">;
  }
</script>

<script lang="ts">
  /**
   * Setup screen (spec 12.2 step 2): who is selecting (by display label, or key when there is none), which input, and
   * the gaze hint when it is available. The input is preselected from the pointer type seen on this screen every
   * time; no choice is remembered. Start refusals are listed with their messages and keep Start disabled; a changed
   * consent text sends the person back to the consent sheet.
   */
  import { onMount } from "svelte";
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { coadaptation, refusalMessage } from "../../stores/coadaptation";
  import { studyPointer } from "../../stores/studyPointer";
  import { gazeRuntime } from "../../stores/gazeRuntime";
  import { consentTextSha256, englishConsentText } from "../../coadaptation/consentText";
  import type { Translate } from "./AssistChip.svelte";
  import CoAdaptationConsent from "./CoAdaptationConsent.svelte";

  interface Props {
    onstart: (setup: StudySetupChoice) => void;
    oncancel: () => void;
    /** A refusal from the last session_begin, shown above Start. */
    message?: string;
    /** Refusals the host knows about (a running task, armed neural control, a small window, a recording). */
    refusals?: string[];
    /** Set while session_begin runs. */
    starting?: boolean;
    /** session_begin refused because the consent text or the protocol changed: offer the consent sheet. */
    reviewConsent?: boolean;
  }

  let { onstart, oncancel, message = "", refusals = [], starting = false, reviewConsent = false }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  type InputChoice = "pointer" | "keyboard" | "hand_pointer";

  let participantKey = $state("");
  let input = $state<InputChoice>("pointer");
  let pointerType = $state<"mouse" | "pen" | "touch">("mouse");
  let pointerTypes = $state<Array<"mouse" | "pen" | "touch">>([]);
  let useGaze = $state(false);
  let consentHash = $state("");
  let consentSheet = $state<{ participantKey?: string } | null>(null);

  const status = $derived($coadaptation.status);
  const config = $derived(status?.config ?? null);
  const participants = $derived((status?.participants ?? []).filter((entry) => entry.consent?.active));
  const chosen = $derived(participants.find((entry) => entry.participant_key === participantKey) ?? null);
  const purposes = $derived(new Set<string>(chosen?.consent.purposes ?? []));
  const handPointerAllowed = $derived(
    purposes.has("hand_pointer") && Boolean(config?.hand_pointer_enabled) && $studyPointer.available,
  );
  const gazeAvailable = $derived(
    purposes.has("gaze_prior") && Boolean(config?.gaze_prior_enabled) && $gazeRuntime.cameraActive,
  );
  const consentChanged = $derived(Boolean(chosen && consentHash && chosen.consent.consent_text_sha256 !== consentHash));
  const blockers = $derived.by(() => {
    const list: string[] = [];
    if (!config?.enabled) list.push(refusalMessage("disabled"));
    if (!chosen) list.push(refusalMessage("no_consent"));
    if (consentChanged) list.push(refusalMessage("consent_text_changed"));
    if (input === "hand_pointer" && !handPointerAllowed) list.push(t("study.setup.hand_unavailable"));
    return [...list, ...refusals];
  });

  onMount(() => {
    void consentTextSha256(englishConsentText()).then((hash) => (consentHash = hash));
  });

  // Default to the first participant with an active consent.
  $effect(() => {
    if (!chosen && participants.length) participantKey = participants[0].participant_key;
  });

  $effect(() => {
    if (!gazeAvailable) useGaze = false;
  });

  function notePointer(event: PointerEvent) {
    const type = event.pointerType;
    if (type !== "mouse" && type !== "pen" && type !== "touch") return;
    pointerType = type;
    if (!pointerTypes.includes(type)) pointerTypes = [...pointerTypes, type];
  }

  function modality(): Modality {
    if (input === "keyboard") return "keyboard";
    if (input === "hand_pointer") return "hand_pointer";
    return pointerType;
  }

  function start() {
    if (blockers.length || !chosen || starting) return;
    onstart({
      participantKey: chosen.participant_key,
      modality: modality(),
      gaze: useGaze && gazeAvailable,
      pointerTypes: pointerTypes.length ? [...pointerTypes] : [pointerType],
    });
  }

  async function granted(key: string) {
    consentSheet = null;
    await coadaptation.load();
    participantKey = key;
  }
</script>

<!-- The pointer type seen here preselects the input; it is not stored anywhere. -->
<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<section class="setup" aria-labelledby="study-setup-title" onpointerdown={notePointer}>
  {#if consentSheet}
    <CoAdaptationConsent
      participantKey={consentSheet.participantKey}
      enable={Boolean(config?.enabled)}
      onclose={() => (consentSheet = null)}
      ongranted={granted}
    />
  {:else}
    <h2 class="title">{t("study.setup.title")}</h2>
    <h3 id="study-setup-title">{t("study.setup.who")}</h3>
    <div class="participants" role="radiogroup" aria-labelledby="study-setup-title">
      {#each participants as entry (entry.participant_key)}
        <label class="choice">
          <input type="radio" name="study-participant" value={entry.participant_key} bind:group={participantKey} />
          {entry.label || entry.participant_key}
        </label>
      {/each}
      <button class="secondary" onclick={() => (consentSheet = {})}>{t("study.setup.new_participant")}</button>
    </div>
    {#if (consentChanged || reviewConsent) && chosen}
      <button class="secondary" onclick={() => (consentSheet = { participantKey: chosen?.participant_key })}>
        {t("study.setup.needs_consent")}
      </button>
    {/if}

    <h3>{t("study.setup.how")}</h3>
    <div class="inputs" role="radiogroup">
      <label class="choice">
        <input type="radio" name="study-input" value="pointer" bind:group={input} />
        {t("study.setup.mouse")}
      </label>
      <label class="choice">
        <input type="radio" name="study-input" value="keyboard" bind:group={input} />
        {t("study.setup.keyboard")}
      </label>
      <label class="choice" class:unavailable={!handPointerAllowed}>
        <input type="radio" name="study-input" value="hand_pointer" bind:group={input} disabled={!handPointerAllowed} />
        {t("study.setup.hand")}
      </label>
    </div>

    {#if gazeAvailable}
      <label class="choice">
        <input type="checkbox" bind:checked={useGaze} />
        {t("study.setup.gaze")}
      </label>
    {/if}

    <p class="duration">{t("study.setup.duration")}</p>

    {#if message}
      <p class="refusal" role="alert">{message}</p>
    {/if}
    {#if blockers.length}
      <ul class="refusals" role="status">
        {#each blockers as blocker, index (index)}
          <li>{blocker}</li>
        {/each}
      </ul>
    {/if}

    <div class="actions">
      <button class="secondary" onclick={() => oncancel()}>{t("study.setup.cancel")}</button>
      <button class="primary" onclick={start} disabled={blockers.length > 0 || !chosen || starting}>
        {starting ? t("study.setup.starting") : t("study.setup.start")}
      </button>
    </div>
  {/if}
</section>

<style>
  .setup {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 560px;
    margin: 0 auto;
    padding: 16px;
    color: var(--text-primary);
  }

  .title {
    margin: 0;
    font-size: 16px;
  }

  h3 {
    margin: 4px 0 0;
    font-size: 14px;
  }

  .participants,
  .inputs {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .choice {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    cursor: pointer;
  }

  .choice.unavailable {
    color: var(--text-muted);
    cursor: default;
  }

  .choice input {
    accent-color: var(--accent);
  }

  .duration {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .refusal,
  .refusals {
    margin: 0;
    font-size: 12px;
    color: var(--danger);
  }

  .refusals {
    padding-left: 18px;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  button {
    align-self: flex-start;
    padding: 6px 14px;
    font-size: 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  .secondary {
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
  }

  .primary {
    font-weight: 600;
    color: white;
    background: var(--accent);
    border: 1px solid var(--accent);
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
</style>
