<script lang="ts">
  /** Settings for Heliox Live: on or off, how readily it speaks, what it may perceive, which catches it makes. */
  import { onMount } from "svelte";
  import { call } from "../api/daemon";
  import { settings } from "../stores/settings";

  type Kind = { kind: string; label: string };
  type Timing = {
    mid_sentence_seconds: number;
    boundary_seconds: number;
    default_seconds: number;
    learned_from: number;
    cut_offs: number;
    well_timed: number;
  };
  type Pack = {
    name: string;
    ok: boolean;
    problems: string[];
    admitted: number;
    held_back: { id: string; problems: string[] }[];
  };
  type Catches = { packs: Pack[]; rules: number; held_back: number };
  type ReaderState = { state: string; message: string; settings_url: string };
  type Status = {
    status: string;
    kinds?: Kind[];
    reads_focused_text?: boolean;
    focused_text?: ReaderState;
    timing?: Timing;
    catches?: Catches;
  };

  let kinds = $state<Kind[]>([]);
  let timing = $state<Timing | null>(null);
  let catches = $state<Catches | null>(null);
  let readsFocusedText = $state(true);
  let reader = $state<ReaderState | null>(null);
  let error = $state("");
  let saving = $state(false);

  let cfg = $derived(
    $settings.live ?? {
      enabled: false,
      level: "balanced",
      read_focused_text: true,
      ocr_fallback: false,
      listen_to_speech: true,
      personal_timing: true,
      grounded_checks: false,
      grounded_model: "qwen3.5:4b",
      muted_kinds: [],
    },
  );
  let learnsTiming = $derived(cfg.personal_timing !== false);
  // What is listened to, in one plain sentence for the model this person chose (pilot.live, pilot.system.addressing).
  let listeningNote = $derived(listenedTo($settings.model?.provider ?? "", $settings.model?.cloud_provider ?? ""));

  function listenedTo(provider: string, cloud: string): string {
    const heard =
      "Live checks what you say to Heliox. While a conversation is open it also checks sentences said near the " +
      "microphone that were not for Heliox; those give at most a silent card, their words are kept in memory for " +
      "90 seconds, and they are never stored as conversation or sent to any model.";
    if (provider === "ollama" || provider === "local")
      return `${heard} Your language model runs on this computer, so nothing you say leaves it.`;
    const where = cloud ? `the cloud model you chose (${cloud})` : "the cloud model you chose";
    return `${heard} What you say to Heliox goes to ${where} with your request, as it always does; speech that was not for Heliox never does.`;
  }

  function timingSummary(t: Timing): string {
    if (!t.learned_from) return `Still learning how you type. Until then it waits the usual ${t.default_seconds} s.`;
    const pauses = t.learned_from === 1 ? "1 pause" : `${t.learned_from} pauses`;
    let text = `Mid-sentence it now waits ${t.mid_sentence_seconds} s before speaking up, learned from ${pauses}.`;
    if (t.cut_offs)
      text += ` It cut in on you ${t.cut_offs === 1 ? "once" : `${t.cut_offs} times`} and waits longer since.`;
    return text;
  }

  function catchSummary(c: Catches): string {
    const packs = c.packs.length === 1 ? "1 catch pack" : `${c.packs.length} catch packs`;
    const held = c.held_back ? `; ${c.held_back} held back` : "";
    return `${c.rules} rules from ${packs} passed the precision check${held}.`;
  }

  function heldBack(pack: Pack): string {
    const first = pack.held_back[0];
    const why = pack.problems[0] ?? (first ? `${first.id}: ${first.problems[0] ?? ""}` : "");
    return `${pack.name || "a pack"}, ${why}`;
  }

  async function resetTiming() {
    try {
      const status = await call<Status>("live_set", { reset_timing: true });
      if (status.status === "ok") timing = status.timing ?? null;
    } catch {
      error = "Couldn't reset the timing. Is the daemon running?";
    }
  }

  const LEVELS = [
    { value: "quiet", label: "Quiet", hint: "Speaks only for risky things; the rest are cards." },
    { value: "balanced", label: "Balanced", hint: "Speaks at your pauses, about once every two minutes at most." },
    { value: "proactive", label: "Proactive", hint: "Speaks up sooner and more often." },
  ] as const;

  async function save(values: Record<string, unknown>) {
    saving = true;
    error = "";
    const result = await settings.updateSectionDetailed("live", values);
    saving = false;
    if (!result.ok) error = result.message;
  }

  // macOS only: shows the system's Accessibility prompt (at most once per run) and reports the new state.
  async function askPermission() {
    try {
      const status = await call<Status>("live_set", { request_permission: true });
      if (status.status === "ok") reader = status.focused_text ?? reader;
    } catch {
      /* the note above still says where to turn it on */
    }
  }

  function toggleKind(kind: string, on: boolean) {
    const muted = new Set(cfg.muted_kinds ?? []);
    if (on) muted.delete(kind);
    else muted.add(kind);
    void save({ muted_kinds: [...muted] });
  }

  onMount(async () => {
    try {
      const status = await call<Status>("live_status");
      if (status.status === "ok") {
        kinds = status.kinds ?? [];
        readsFocusedText = status.reads_focused_text !== false;
        reader = status.focused_text ?? null;
        timing = status.timing ?? null;
        catches = status.catches ?? null;
      }
    } catch {
      /* shown without the per-kind list until the daemon answers */
    }
  });
</script>

<section class="live-panel" aria-labelledby="live-heading" data-testid="live-panel">
  <div class="section-title">
    <div>
      <h3 id="live-heading">Live companion</h3>
      <p>
        Always on: Heliox notices slips while you work and says so at your next pause, like a colleague looking over
        your shoulder. A weekday that doesn't match its date, a wrong sum, a risky command before you press Enter, a key
        pasted into a chat, an error with its fix. It never acts on its own, and it stays quiet while you talk.
      </p>
    </div>
    <button
      class="toggle"
      class:active={cfg.enabled}
      onclick={() => save({ enabled: !cfg.enabled })}
      disabled={saving}
      aria-label="Live companion"
      aria-pressed={cfg.enabled}
      title="Live companion"
    >
      <span class="toggle-knob"></span>
    </button>
  </div>

  {#if error}<div class="message" role="alert">{error}</div>{/if}

  <div class="block">
    <span class="block-label" id="live-level">How readily it speaks</span>
    <div class="levels" role="radiogroup" aria-labelledby="live-level">
      {#each LEVELS as level (level.value)}
        <label class="level" class:selected={cfg.level === level.value}>
          <input
            type="radio"
            name="live-level"
            value={level.value}
            checked={cfg.level === level.value}
            onchange={() => save({ level: level.value })}
          />
          <span class="level-name">{level.label}</span>
          <span class="level-hint">{level.hint}</span>
        </label>
      {/each}
    </div>
  </div>

  <div class="block">
    <span class="block-label">Your timing</span>
    <label class="check">
      <input
        type="checkbox"
        checked={learnsTiming}
        onchange={(e) => save({ personal_timing: (e.currentTarget as HTMLInputElement).checked })}
      />
      <span
        >Learn how long you pause in the middle of a sentence, so it waits for a real break instead of cutting in while
        you think. Risky commands still get flagged at the first short pause.</span
      >
    </label>
    {#if timing && learnsTiming}
      <div class="timing" data-testid="live-timing">
        <span>{timingSummary(timing)}</span>
        {#if timing.learned_from || timing.cut_offs}
          <button class="text-button" onclick={resetTiming}>Start over</button>
        {/if}
      </div>
    {/if}
  </div>

  <div class="block">
    <span class="block-label">What it may notice</span>
    <label class="check">
      <input
        type="checkbox"
        checked={cfg.read_focused_text}
        onchange={(e) => save({ read_focused_text: (e.currentTarget as HTMLInputElement).checked })}
      />
      <span
        >The text field you're typing in{readsFocusedText ? "" : " (not available on this system)"}. Never password
        fields, password managers, Heliox itself, or apps you excluded from the timeline.</span
      >
    </label>
    {#if cfg.read_focused_text && reader?.message}
      <p class="reader-note">
        {reader.message}
        {#if reader.state === "needs_permission"}
          <button type="button" class="link" onclick={askPermission}>Ask macOS for access</button>
        {/if}
      </p>
    {/if}
    <label class="check">
      <input
        type="checkbox"
        checked={cfg.listen_to_speech}
        onchange={(e) => save({ listen_to_speech: (e.currentTarget as HTMLInputElement).checked })}
      />
      <span>What you say to Heliox, so it can catch a wrong date before planning.</span>
    </label>
    {#if cfg.listen_to_speech}<p class="listening-note" data-testid="live-listening-note">{listeningNote}</p>{/if}
    <label class="check">
      <input
        type="checkbox"
        checked={cfg.ocr_fallback}
        onchange={(e) => save({ ocr_fallback: (e.currentTarget as HTMLInputElement).checked })}
      />
      <span>Read the screen when no text field is readable (for errors only; uses more CPU).</span>
    </label>
    <label class="check">
      <input
        type="checkbox"
        checked={cfg.grounded_checks === true}
        onchange={(e) => save({ grounded_checks: (e.currentTarget as HTMLInputElement).checked })}
      />
      <span
        >Check what you write against your own calendar, to-dos and reminders: a plan on top of a meeting, or the wrong
        day for something you have booked. Clashes and named events are checked by Heliox itself;
        {cfg.grounded_model || "qwen3.5:4b"} in Ollama, on this computer, helps with paraphrases if it is installed.</span
      >
    </label>
  </div>

  {#if kinds.length}
    <div class="block">
      <span class="block-label">What it points out</span>
      <div class="kinds">
        {#each kinds as item (item.kind)}
          <label class="check">
            <input
              type="checkbox"
              checked={!(cfg.muted_kinds ?? []).includes(item.kind)}
              onchange={(e) => toggleKind(item.kind, (e.currentTarget as HTMLInputElement).checked)}
            />
            <span>{item.label}</span>
          </label>
        {/each}
      </div>
      {#if catches && catches.packs.length}
        <div class="packs" data-testid="live-catch-packs">
          <span>{catchSummary(catches)}</span>
          {#each catches.packs.filter((pack) => !pack.ok) as pack (pack.name + pack.problems.join())}
            <span class="held">Held back: {heldBack(pack)}</span>
          {/each}
        </div>
      {/if}
    </div>
  {/if}

  <p class="privacy">
    What it reads stays on this computer and in memory. Two things are kept: which kinds of correction you found useful,
    so it speaks up less about the ones you wave away, and how long your pauses last (numbers, never text). Clear either
    in Your data.
  </p>
</section>

<style>
  .live-panel {
    margin-top: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-secondary);
    overflow: hidden;
  }

  .section-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 16px 20px;
  }

  h3 {
    margin: 0 0 5px;
    font-size: var(--text-lg);
  }

  p {
    max-width: 72ch;
    margin: 0;
    color: var(--text-secondary);
    font-size: var(--text-sm);
    line-height: 1.5;
  }

  .listening-note {
    padding-left: 26px;
    font-size: var(--text-xs);
  }

  .reader-note {
    padding-left: 26px;
  }

  .reader-note .link {
    padding: 0;
    border: 0;
    color: var(--accent);
    background: none;
    font: inherit;
    text-decoration: underline;
    cursor: pointer;
  }

  .message {
    padding: 10px 20px;
    border-top: 1px solid var(--border);
    color: var(--danger);
    background: var(--danger-bg);
    font-size: var(--text-sm);
  }

  .block {
    display: grid;
    gap: 8px;
    padding: 14px 20px;
    border-top: 1px solid var(--border);
  }

  .block-label {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .levels {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 8px;
  }

  .level {
    display: grid;
    gap: 2px;
    padding: 9px 11px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition:
      border-color var(--dur-fast) ease,
      background var(--dur-fast) ease;
  }

  .level input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  .level:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .level.selected {
    border-color: var(--accent-line);
    background: var(--accent-muted);
  }

  .level-name {
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
  }

  .level-hint {
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .check {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    font-size: var(--text-sm);
    line-height: 1.45;
    color: var(--text-secondary);
    cursor: pointer;
  }

  .check input {
    flex: none;
    margin-top: 2px;
    accent-color: var(--accent);
  }

  .timing {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 4px 12px;
    padding-left: 25px;
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .text-button {
    padding: 0;
    border: none;
    background: none;
    color: var(--accent);
    font: inherit;
    font-weight: 600;
    cursor: pointer;
  }

  .text-button:hover {
    text-decoration: underline;
  }

  .text-button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    border-radius: 2px;
  }

  .packs {
    display: grid;
    gap: 3px;
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .packs .held {
    color: var(--text-tertiary, var(--text-secondary));
  }

  .kinds {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 6px 16px;
  }

  .privacy {
    padding: 12px 20px 16px;
    border-top: 1px solid var(--border);
    font-size: var(--text-xs);
  }

  .toggle {
    position: relative;
    flex-shrink: 0;
    width: 40px;
    height: 22px;
    border-radius: 11px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    cursor: pointer;
  }

  .toggle.active {
    background: var(--accent);
    color: var(--on-accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: white;
    transition: transform var(--dur-fast) ease;
  }

  .toggle.active .toggle-knob {
    transform: translateX(18px);
  }

  .toggle:disabled {
    cursor: default;
    opacity: 0.6;
  }
</style>
