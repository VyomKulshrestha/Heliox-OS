<script module lang="ts">
  import { VARIANT_LABEL_KEYS } from "../../coadaptation/types";

  export type MessageValues = Record<string, string | number | boolean | null | undefined>;
  export type Translate = (
    key: string,
    options?: { values?: MessageValues; locale?: string; default?: string },
  ) => string;

  function hasLabel(variant: string): variant is keyof typeof VARIANT_LABEL_KEYS {
    return variant in VARIANT_LABEL_KEYS;
  }

  /** A variant's plain-language name (`coadaptation.variant.*`), or the id for an unknown variant. */
  export function variantLabel(t: Translate, variant: string | null | undefined): string {
    if (!variant) return "";
    return hasLabel(variant) ? t(VARIANT_LABEL_KEYS[variant]) : variant;
  }

  /** A reason label of spec 16.6 (`coadaptation.reason.*`). */
  export function reasonLabel(t: Translate, reason: string): string {
    return t(`coadaptation.reason.${reason}`);
  }

  /** The explanation sentence (`coadaptation.explain.*`); variant ids among its values are shown by name. */
  export function explanationText(
    t: Translate,
    explanation: { summary_key: string; values?: Record<string, string | number> } | null | undefined,
  ): string {
    if (!explanation?.summary_key) return "";
    const values: MessageValues = {};
    for (const [name, value] of Object.entries(explanation.values ?? {})) {
      values[name] = typeof value === "string" && hasLabel(value) ? variantLabel(t, value) : value;
    }
    return t(explanation.summary_key, { values });
  }

  /** The short chip label of spec 16.6 for the help a block or an ambiguous click gets. */
  export function chipLabel(t: Translate, variant: string): string {
    if (variant.startsWith("enlarge_target")) return t("study.chip.larger_targets");
    if (variant === "confirm_selection:second_click") return t("study.chip.confirm");
    if (variant.startsWith("confirm_selection:hold")) return t("study.chip.hold");
    if (variant.startsWith("ask_clarification")) return t("study.chip.which_one");
    return variantLabel(t, variant);
  }
</script>

<script lang="ts">
  /**
   * The chip of an applied kind of help (spec 9.7, 16.4): its name with "Why?", "Not this time" and "Never" (Never
   * has its own confirm step). wait_for_evidence is the absence of system action, so it gets a dismiss-only hint with
   * no veto, from any source and in any mode. In pinned mode (a frozen or pinned block, or a pin_act collapse) "Not
   * this time" reverts the selection only, and the chip then offers to remove the rule behind the pinned mode.
   */
  import { _ } from "svelte-i18n";
  import "../../coadaptation/messages";
  import type { ExplanationV1 } from "../../coadaptation/types";

  type PinnedRule = "freeze" | "pin" | "pin_act";

  interface Props {
    variant: string;
    explanation?: ExplanationV1 | null;
    /** The decision's mode; "pinned" offers to retract the rule after "Not this time". */
    mode?: string;
    /** The rule behind pinned mode (defaults to a freeze). */
    pinnedRule?: PinnedRule | null;
    /** The pinned regime, named in "You pinned X". */
    pinnedVariant?: string | null;
    onwhy?: () => void;
    onnotthistime?: () => void;
    onnever?: () => void | Promise<void>;
    /** True when the Never confirm step opens, false when it closes (the study pauses input meanwhile). */
    onneverstep?: (open: boolean) => void;
    ondismiss?: () => void;
    onretract?: () => void;
    onkeep?: () => void;
  }

  let {
    variant,
    explanation = null,
    mode = "",
    pinnedRule = null,
    pinnedVariant = null,
    onwhy,
    onnotthistime,
    onnever,
    onneverstep,
    ondismiss,
    onretract,
    onkeep,
  }: Props = $props();

  const t: Translate = (key, options) => $_(key, options);

  let showWhy = $state(false);
  let confirmingNever = $state(false);
  let neverBusy = $state(false);
  let retractOffer = $state(false);
  let vetoed = $state(false);

  const isHint = $derived(variant === "wait_for_evidence");
  const label = $derived(chipLabel(t, variant));
  const rule = $derived<PinnedRule>(pinnedRule ?? "freeze");
  const why = $derived(explanationText(t, explanation));

  function toggleWhy() {
    showWhy = !showWhy;
    if (showWhy) onwhy?.();
  }

  function notThisTime() {
    vetoed = true;
    onnotthistime?.();
    if (mode === "pinned") retractOffer = true;
  }

  function openNever() {
    confirmingNever = true;
    onneverstep?.(true);
  }

  function cancelNever() {
    confirmingNever = false;
    onneverstep?.(false);
  }

  async function confirmNever() {
    neverBusy = true;
    try {
      await onnever?.();
    } finally {
      neverBusy = false;
      confirmingNever = false;
      onneverstep?.(false);
    }
  }

  function retract() {
    retractOffer = false;
    onretract?.();
  }

  function keep() {
    retractOffer = false;
    onkeep?.();
  }
</script>

{#if isHint}
  <div class="assist-chip hint" role="status">
    <span class="chip-label">{t("study.chip.hint")}</span>
    <button class="dismiss" aria-label={t("study.chip.dismiss")} onclick={() => ondismiss?.()}>×</button>
  </div>
{:else}
  <div class="assist-chip" role="group" aria-label={label}>
    <span class="chip-label">{label}</span>
    {#if retractOffer}
      <div class="retract-offer" role="status">
        {#if rule === "freeze"}
          <span>{t("study.retract.freeze")}</span>
          <button onclick={retract}>{t("study.retract.resume_adapting")}</button>
          <button onclick={keep}>{t("study.retract.keep_paused")}</button>
        {:else if rule === "pin"}
          <span>{t("study.retract.pin", { values: { variant: variantLabel(t, pinnedVariant ?? variant) } })}</span>
          <button onclick={retract}>{t("study.retract.remove_rule")}</button>
          <button onclick={keep}>{t("study.retract.keep_it")}</button>
        {:else}
          <span>{t("study.retract.pin_act")}</span>
          <button onclick={retract}>{t("study.retract.remove_rule")}</button>
          <button onclick={keep}>{t("study.retract.keep_it")}</button>
        {/if}
      </div>
    {:else if confirmingNever}
      <div class="never-confirm" role="alertdialog" aria-label={t("study.chip.never")}>
        <span>{t("study.chip.never_confirm", { values: { variant: variantLabel(t, variant) } })}</span>
        <button onclick={confirmNever} disabled={neverBusy}>{t("study.chip.yes")}</button>
        <button onclick={cancelNever} disabled={neverBusy}>{t("study.chip.cancel")}</button>
      </div>
    {:else}
      <div class="chip-actions">
        <button onclick={toggleWhy} aria-expanded={showWhy}>{t("study.chip.why")}</button>
        <button onclick={notThisTime} disabled={vetoed}>{t("study.chip.not_this_time")}</button>
        <button onclick={openNever}>{t("study.chip.never")}</button>
      </div>
    {/if}
    {#if showWhy && why && !confirmingNever}
      <p class="why">{why}</p>
    {/if}
  </div>
{/if}

<style>
  .assist-chip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 10px;
    max-width: 100%;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  }

  .assist-chip.hint {
    border-style: dashed;
  }

  .chip-label {
    font-weight: 600;
  }

  .chip-actions,
  .never-confirm,
  .retract-offer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }

  button {
    padding: 3px 9px;
    font-size: 12px;
    color: var(--text-secondary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
  }

  button:hover:not(:disabled) {
    color: var(--text-primary);
    border-color: var(--accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .dismiss {
    padding: 0 7px;
    font-size: 14px;
    line-height: 20px;
  }

  .why {
    flex-basis: 100%;
    margin: 0;
    font-size: 11px;
    line-height: 1.4;
    color: var(--text-secondary);
  }
</style>
