<script module lang="ts">
  import { VARIANT_FAMILIES, type PreferenceV1 } from "../../coadaptation/types";
  import { variantLabel, type Translate } from "./AssistChip.svelte";

  /** A forbidden variant or family by name. */
  function forbiddenName(t: Translate, value: string): string {
    return (VARIANT_FAMILIES as readonly string[]).includes(value)
      ? t(`coadaptation.rules.family.${value}`)
      : variantLabel(t, value);
  }

  /** The copy of spec 16.4 for a max_interruptions_per_block rule (the count of section 9.3). */
  export function maxInterruptionsText(t: Translate, n: number): string {
    return t("coadaptation.rules.max_interruptions_copy", { values: { n } });
  }

  /** One rule in words. */
  export function ruleSummary(t: Translate, pref: Pick<PreferenceV1, "kind" | "value">): string {
    const value = pref.value as unknown;
    switch (pref.kind) {
      case "forbid_variant":
        return t("coadaptation.rules.summary.forbid_variant", { values: { value: forbiddenName(t, String(value)) } });
      case "min_caution":
      case "pin_regime":
        return t(`coadaptation.rules.summary.${pref.kind}`, { values: { value: variantLabel(t, String(value)) } });
      case "pin_act":
        return t("coadaptation.rules.summary.pin_act");
      case "max_interruptions_per_block":
        return maxInterruptionsText(t, Number(value));
      case "freeze_adaptation": {
        const regime = (value as { regime?: string } | null)?.regime ?? "";
        return t("coadaptation.rules.summary.freeze_adaptation", { values: { value: variantLabel(t, regime) } });
      }
      case "priorities": {
        const weights = (value ?? {}) as Record<string, number>;
        return t("coadaptation.rules.summary.priorities", {
          values: {
            mistakes: (weights.fewer_mistakes ?? 0).toFixed(2),
            faster: (weights.faster ?? 0).toFixed(2),
            interruptions: (weights.fewer_interruptions ?? 0).toFixed(2),
          },
        });
      }
      default:
        return String(pref.kind);
    }
  }
</script>

<script lang="ts">
  /**
   * The person's own assistance rules (spec 9.6, 16.4). Rules are explicit, retractable and outrank learning; nothing
   * here is written without a click, proposals included. A rule that conflicts with an active one (freezes and pins
   * included) is rejected by the daemon with a message naming that rule: "Remove that rule" retracts it and then saves
   * the new rule, "Keep it" changes nothing. Rules apply to study targets only; no rule can change an approval.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import { coadaptation, PreferenceConflictError } from "../../stores/coadaptation";
  import { CAUTION_ORDER, L1_ARMS } from "../../coadaptation/types";
  import type {
    Modality,
    PreferenceConflictItemV1,
    PreferenceKind,
    PreferenceScopeV1,
    ProposalV1,
  } from "../../coadaptation/types";

  interface Props {
    participantKey: string | null;
  }

  let { participantKey }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  const MODALITIES: Modality[] = ["mouse", "touchpad", "pen", "touch", "keyboard", "hand_pointer"];
  const PIN_REGIMES = [...new Set([...L1_ARMS("mouse"), ...L1_ARMS("hand_pointer")])].filter(
    (variant) => variant !== "assistance_off",
  );
  const FORBID_VARIANTS = [
    ...CAUTION_ORDER.filter((variant) => variant !== "wait_for_evidence"),
    "offer_modality:keyboard",
    "offer_modality:pointer",
  ];
  const FORM_KINDS = [
    "forbid_variant",
    "min_caution",
    "pin_regime",
    "pin_act",
    "max_interruptions_per_block",
    "priorities",
  ] as const;
  type FormKind = (typeof FORM_KINDS)[number];

  interface PendingRule {
    kind: PreferenceKind;
    scope: PreferenceScopeV1;
    value: unknown;
  }

  interface Conflict {
    message: string;
    conflicting: PreferenceConflictItemV1[];
    rule: PendingRule;
  }

  let kind = $state<FormKind>("forbid_variant");
  let forbidValue = $state("ask_clarification:3");
  let minCautionValue = $state("confirm_selection:second_click");
  let pinValue = $state("enlarge_target:1.5");
  let maxInterruptions = $state(4);
  let fewerMistakes = $state(0.5);
  let faster = $state(0.5);
  let fewerInterruptions = $state(0.5);
  let scopeChoice = $state<"participant" | "device">("participant");
  let context = $state<"study" | "all">("study");
  let modality = $state<Modality | "">("");

  let saving = $state(false);
  let error = $state("");
  let notice = $state("");
  let notes = $state<string[]>([]);
  let conflict = $state<Conflict | null>(null);
  let dismissedProposals = $state<string[]>([]);

  // Without a participant only a device-wide rule can be made.
  $effect(() => {
    if (!participantKey) scopeChoice = "device";
  });

  const effectiveScope = $derived<"participant" | "device">(participantKey ? scopeChoice : "device");
  const cap = $derived(Math.max(0, Math.min(8, Math.round(Number(maxInterruptions) || 0))));

  const activeRules = $derived(
    ($coadaptation.status?.preferences ?? []).filter(
      (pref) =>
        !pref.retracted_at && (pref.scope.participant_key === null || pref.scope.participant_key === participantKey),
    ),
  );

  const proposals = $derived(
    ($coadaptation.status?.proposals ?? []).filter(
      (proposal) =>
        (participantKey === null || proposal.participant_key === participantKey) &&
        !dismissedProposals.includes(proposalId(proposal)),
    ),
  );

  // A step cap with a minimum caution that leaves only confirm or hold regimes: the cap does not remove them (spec
  // 9.2 C1), and the editor says so for as long as both rules are active.
  const capKeepsConfirm = $derived(
    activeRules.some((pref) => pref.kind === "max_interruptions_per_block") &&
      activeRules.some(
        (pref) =>
          pref.kind === "min_caution" &&
          CAUTION_ORDER.indexOf(String(pref.value)) >= CAUTION_ORDER.indexOf("confirm_selection:second_click"),
      ),
  );

  function proposalId(proposal: ProposalV1): string {
    return `${proposal.participant_key ?? "device"}|${proposal.kind}|${proposal.variant}`;
  }

  function scopeText(scope: PreferenceScopeV1): string {
    const who = scope.participant_key
      ? t("coadaptation.rules.participant", { values: { participant: scope.participant_key } })
      : t("coadaptation.rules.everyone");
    const where = t(scope.context === "all" ? "coadaptation.rules.context_all" : "coadaptation.rules.context_study");
    const input = scope.modality ? t(`coadaptation.modality.${scope.modality}`) : t("coadaptation.rules.modality_any");
    return `${who} · ${where} · ${input}`;
  }

  function formRule(): PendingRule {
    const scope: PreferenceScopeV1 = {
      context,
      modality: modality === "" ? null : modality,
      participant_key: effectiveScope === "participant" ? participantKey : null,
    };
    switch (kind) {
      case "forbid_variant":
        return { kind, scope, value: forbidValue };
      case "min_caution":
        return { kind, scope, value: minCautionValue };
      case "pin_regime":
        return { kind, scope, value: pinValue };
      case "pin_act":
        return { kind, scope, value: true };
      case "max_interruptions_per_block":
        return { kind, scope, value: cap };
      default:
        return {
          kind,
          scope,
          value: { fewer_mistakes: fewerMistakes, faster, fewer_interruptions: fewerInterruptions },
        };
    }
  }

  function failure(cause: unknown): string {
    return t("coadaptation.rules.failed", {
      values: { message: cause instanceof Error ? cause.message : String(cause) },
    });
  }

  async function save(rule: PendingRule): Promise<boolean> {
    saving = true;
    error = "";
    notice = "";
    notes = [];
    conflict = null;
    try {
      const result = await coadaptation.setPreference(rule.kind, rule.scope, rule.value);
      notes = result.notes;
      notice = t("coadaptation.rules.saved");
      return true;
    } catch (cause) {
      if (cause instanceof PreferenceConflictError) {
        conflict = { message: cause.message, conflicting: cause.conflicting, rule };
      } else {
        error = failure(cause);
      }
      return false;
    } finally {
      saving = false;
    }
  }

  async function retract(prefId: string) {
    saving = true;
    error = "";
    notice = "";
    try {
      await coadaptation.retractPreference(prefId);
      notice = t("coadaptation.rules.removed");
    } catch (cause) {
      error = failure(cause);
    } finally {
      saving = false;
    }
  }

  /** "Remove that rule": retracts every rule the conflict names, then saves the new rule again. */
  async function removeConflicting() {
    const pending = conflict;
    if (!pending) return;
    saving = true;
    error = "";
    try {
      for (const entry of pending.conflicting) await coadaptation.retractPreference(entry.pref_id);
    } catch (cause) {
      error = failure(cause);
      saving = false;
      return;
    }
    await save(pending.rule);
  }

  async function acceptProposal(proposal: ProposalV1) {
    const saved = await save({
      kind: "forbid_variant",
      scope: { context: "study", modality: null, participant_key: proposal.participant_key },
      value: proposal.variant,
    });
    if (saved) dismissedProposals = [...dismissedProposals, proposalId(proposal)];
  }
</script>

<div class="rules">
  <h4>{t("coadaptation.rules.title")}</h4>
  <p class="intro">{t("coadaptation.rules.intro")}</p>

  {#if activeRules.length}
    <ul class="rule-list">
      {#each activeRules as pref (pref.pref_id)}
        <li>
          <div class="rule-text">
            <span class="summary">{ruleSummary(t, pref)}</span>
            <span class="scope-text">{scopeText(pref.scope)}</span>
          </div>
          <button onclick={() => retract(pref.pref_id)} disabled={saving}>{t("coadaptation.rules.retract")}</button>
        </li>
      {/each}
    </ul>
  {:else}
    <p class="empty">{t("coadaptation.rules.none")}</p>
  {/if}

  {#if capKeepsConfirm}
    <p class="note" role="note">
      {t("coadaptation.rules.cap_min_caution_note", {
        default:
          "Your minimum caution leaves only confirm or hold steps, so your step limit does not remove them. Both rules stay as you set them.",
      })}
    </p>
  {/if}

  {#if proposals.length}
    <h5>{t("coadaptation.rules.proposals_title")}</h5>
    {#each proposals as proposal (proposalId(proposal))}
      <div class="proposal" role="note">
        <span>
          {t("coadaptation.rules.proposal_three_vetoes", { values: { variant: variantLabel(t, proposal.variant) } })}
        </span>
        <button onclick={() => acceptProposal(proposal)} disabled={saving}>{t("coadaptation.rules.stop_using")}</button>
        <button onclick={() => (dismissedProposals = [...dismissedProposals, proposalId(proposal)])}>
          {t("coadaptation.rules.keep_it")}
        </button>
      </div>
    {/each}
  {/if}

  <div class="add-rule">
    <h5>{t("coadaptation.rules.add_title")}</h5>
    <label class="field">
      <span>{t("coadaptation.rules.kind_label")}</span>
      <select bind:value={kind} disabled={saving}>
        {#each FORM_KINDS as entry (entry)}
          <option value={entry}>{t(`coadaptation.rules.kind.${entry}`)}</option>
        {/each}
      </select>
    </label>

    {#if kind === "forbid_variant"}
      <label class="field">
        <span>{t("coadaptation.rules.variant_label")}</span>
        <select bind:value={forbidValue} disabled={saving}>
          {#each VARIANT_FAMILIES as family (family)}
            <option value={family}>{t(`coadaptation.rules.family.${family}`)}</option>
          {/each}
          {#each FORBID_VARIANTS as variant (variant)}
            <option value={variant}>{variantLabel(t, variant)}</option>
          {/each}
        </select>
      </label>
    {:else if kind === "min_caution"}
      <label class="field">
        <span>{t("coadaptation.rules.variant_label")}</span>
        <select bind:value={minCautionValue} disabled={saving}>
          {#each CAUTION_ORDER as variant (variant)}
            <option value={variant}>{variantLabel(t, variant)}</option>
          {/each}
        </select>
      </label>
    {:else if kind === "pin_regime"}
      <label class="field">
        <span>{t("coadaptation.rules.variant_label")}</span>
        <select bind:value={pinValue} disabled={saving}>
          {#each PIN_REGIMES as variant (variant)}
            <option value={variant}>{variantLabel(t, variant)}</option>
          {/each}
        </select>
      </label>
    {:else if kind === "pin_act"}
      <p class="explain">{t("coadaptation.rules.pin_act_detail")}</p>
    {:else if kind === "max_interruptions_per_block"}
      <label class="field">
        <span>{t("coadaptation.rules.max_interruptions_label")}</span>
        <input type="number" min="0" max="8" step="1" bind:value={maxInterruptions} disabled={saving} />
      </label>
      <p class="explain">{maxInterruptionsText(t, cap)}</p>
    {:else}
      <label class="field slider">
        <span>{t("coadaptation.rules.priorities_mistakes")} {fewerMistakes.toFixed(2)}</span>
        <input type="range" min="0" max="1" step="0.05" bind:value={fewerMistakes} disabled={saving} />
      </label>
      <label class="field slider">
        <span>{t("coadaptation.rules.priorities_faster")} {faster.toFixed(2)}</span>
        <input type="range" min="0" max="1" step="0.05" bind:value={faster} disabled={saving} />
      </label>
      <label class="field slider">
        <span>{t("coadaptation.rules.priorities_interruptions")} {fewerInterruptions.toFixed(2)}</span>
        <input type="range" min="0" max="1" step="0.05" bind:value={fewerInterruptions} disabled={saving} />
      </label>
    {/if}

    <fieldset class="scope">
      <legend>{t("coadaptation.rules.scope_label")}</legend>
      <label>
        <input
          type="radio"
          name="rule-scope"
          value="participant"
          bind:group={scopeChoice}
          disabled={saving || !participantKey}
        />
        {t("coadaptation.rules.scope_participant")}
      </label>
      <label>
        <input type="radio" name="rule-scope" value="device" bind:group={scopeChoice} disabled={saving} />
        {t("coadaptation.rules.scope_device")}
      </label>
    </fieldset>

    <div class="row">
      <select
        bind:value={context}
        disabled={saving}
        aria-label={`${t("coadaptation.rules.context_study")} / ${t("coadaptation.rules.context_all")}`}
      >
        <option value="study">{t("coadaptation.rules.context_study")}</option>
        <option value="all">{t("coadaptation.rules.context_all")}</option>
      </select>
      <label class="field">
        <span>{t("coadaptation.rules.modality_label")}</span>
        <select bind:value={modality} disabled={saving}>
          <option value="">{t("coadaptation.rules.modality_any")}</option>
          {#each MODALITIES as entry (entry)}
            <option value={entry}>{t(`coadaptation.modality.${entry}`)}</option>
          {/each}
        </select>
      </label>
    </div>

    <div class="actions">
      <button class="primary" onclick={() => save(formRule())} disabled={saving}>
        {saving ? t("coadaptation.rules.saving") : t("coadaptation.rules.save")}
      </button>
    </div>
  </div>

  {#if conflict}
    <div class="conflict" role="alert">
      <p>{conflict.message}</p>
      {#if conflict.conflicting.length}
        <ul>
          {#each conflict.conflicting as entry (entry.pref_id)}
            <li>{entry.summary}</li>
          {/each}
        </ul>
        <button onclick={removeConflicting} disabled={saving}>{t("coadaptation.rules.remove_that_rule")}</button>
      {/if}
      <button onclick={() => (conflict = null)} disabled={saving}>{t("coadaptation.rules.keep_it")}</button>
    </div>
  {/if}

  {#if notice || notes.length}
    <div class="note" role="status">
      {#if notice}<p>{notice}</p>{/if}
      {#each notes as note, index (index)}
        <p>{t("coadaptation.rules.note", { values: { note } })}</p>
      {/each}
    </div>
  {/if}

  {#if error}
    <p class="error" role="alert">{error}</p>
  {/if}
</div>

<style>
  .rules {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 14px;
  }

  h4,
  h5 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  h5 {
    font-size: 12px;
  }

  .intro,
  .empty,
  .explain {
    margin: 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .rule-list {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .rule-list li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }

  .rule-text {
    display: flex;
    min-width: 0;
    flex-direction: column;
    gap: 2px;
  }

  .summary {
    font-size: 12px;
  }

  .scope-text {
    font-size: 11px;
    color: var(--text-muted);
  }

  .proposal,
  .conflict,
  .note {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 10px;
    padding: 8px 10px;
    font-size: 12px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
  }

  .conflict {
    border-left: 2px solid var(--warning);
  }

  .conflict p,
  .note p {
    flex-basis: 100%;
    margin: 0;
  }

  .conflict ul {
    flex-basis: 100%;
    margin: 0;
    padding-left: 18px;
  }

  .add-rule {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 10px;
  }

  .field {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .field.slider input {
    accent-color: var(--accent);
  }

  select,
  input[type="number"] {
    max-width: 320px;
    padding: 4px 8px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .scope {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 0;
    padding: 6px 8px;
    font-size: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .scope legend {
    padding: 0 4px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .actions {
    display: flex;
    justify-content: flex-end;
  }

  button {
    padding: 4px 10px;
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button.primary {
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-color: var(--accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .error {
    margin: 0;
    font-size: 11px;
    color: var(--danger);
  }
</style>
