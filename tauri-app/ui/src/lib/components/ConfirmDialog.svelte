<script lang="ts">
  import type { PlanAction } from "../stores/session";
  import { describeAction } from "../utils/actionDetails";
  import { _ } from "svelte-i18n";
  import { onMount, tick, untrack } from "svelte";

  interface Props {
    actions: PlanAction[];
    onconfirm: (approvedIndices: number[]) => void;
    ondeny: () => void;
    /** Called on mount and on every toggle with the indices that are checked, so a gesture approves only these. */
    onselectionchange?: (approvedIndices: number[]) => void;
    reason?: string;
    riskAssessment?: Record<string, unknown> | null;
    gnomon?: Record<string, unknown> | null;
    submitting?: boolean;
    error?: string;
  }

  let {
    actions,
    onconfirm,
    ondeny,
    onselectionchange,
    reason = "",
    riskAssessment = null,
    gnomon = null,
    submitting = false,
    error = "",
  }: Props = $props();

  /** Key events this soon after mount are ignored, so typing carried over from the chat box cannot answer. */
  const KEY_GUARD_MS = 300;
  const titleId = $props.id();

  let dialogEl: HTMLDivElement | undefined = $state();
  let denyButton: HTMLButtonElement | undefined = $state();
  let mountedAt = Number.POSITIVE_INFINITY;
  // Keys whose keydown this dialog accepted; a keyup without one (a key held since before the dialog opened) is
  // swallowed so it cannot activate the focused button.
  const acceptedKeys = new Set<string>();

  let approved = $state<boolean[]>([]);
  let anyApproved = $derived(approved.some((v) => v));

  $effect(() => {
    if (approved.length !== actions.length) {
      approved = actions.map(() => true);
    }
  });

  // Compared by value: `actions` comes from the session store, which the report itself updates, so reporting on
  // every store change would loop. The report runs only when the checked set really changes (and once on mount).
  const checkedKey = $derived(checkedIndices().join(","));
  $effect(() => {
    const key = checkedKey;
    untrack(() => onselectionchange?.(key ? key.split(",").map(Number) : []));
  });

  /** Input types that take no typed text; focus on one of them is not someone typing. */
  const NON_TEXT_INPUT_TYPES = new Set([
    "button",
    "checkbox",
    "color",
    "file",
    "hidden",
    "image",
    "radio",
    "range",
    "reset",
    "submit",
  ]);

  function isTextEntry(element: Element | null): boolean {
    if (element instanceof HTMLTextAreaElement) return true;
    if (element instanceof HTMLInputElement) return !NON_TEXT_INPUT_TYPES.has(element.type);
    return element instanceof HTMLElement && element.isContentEditable;
  }

  onMount(() => {
    mountedAt = performance.now();
    // Someone typing in a text field (the chat box) when the approval arrives would press a focused Deny with their
    // next Space or Enter, long after the 300 ms guard. Their focus goes to the dialog box instead, where no key
    // answers: Tab reaches the buttons and Escape still denies. Otherwise the initial focus is on Deny.
    const typing = isTextEntry(document.activeElement);
    // After the pending update, so a tab switch to Command made for this approval has shown the dialog first.
    void tick().then(() => (typing || isTextEntry(document.activeElement) ? dialogEl : denyButton)?.focus());
  });

  function checkedIndices(): number[] {
    return actions.map((a, i) => (approved[i] ? (a.index ?? i) : -1)).filter((i) => i !== -1);
  }

  function toggle(i: number) {
    if (submitting) return;
    approved[i] = !approved[i];
  }

  function handleConfirm() {
    if (submitting) return;
    onconfirm(checkedIndices());
  }

  function handleDeny() {
    if (submitting) return;
    ondeny();
  }

  function swallow(event: KeyboardEvent) {
    event.preventDefault();
    event.stopPropagation();
  }

  function focusableElements(): HTMLElement[] {
    if (!dialogEl) return [];
    return Array.from(
      dialogEl.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex="-1"])',
      ),
    );
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.repeat || performance.now() - mountedAt < KEY_GUARD_MS) {
      swallow(event);
      return;
    }
    acceptedKeys.add(event.key);
    if (event.key === "Escape") {
      swallow(event);
      handleDeny();
    } else if (event.key === "Tab") {
      const focusable = focusableElements();
      if (focusable.length === 0) {
        swallow(event);
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      // Focus on the dialog box itself (a click on its text) counts as outside the cycle of its controls.
      const inside = active instanceof HTMLElement && active !== dialogEl && Boolean(dialogEl?.contains(active));
      if (!inside || (event.shiftKey && active === first)) {
        swallow(event);
        (event.shiftKey ? last : first).focus();
      } else if (!event.shiftKey && active === last) {
        swallow(event);
        first.focus();
      }
    }
    // Enter and Space keep their native meaning: they press the focused button or toggle the focused checkbox.
  }

  function handleKeyup(event: KeyboardEvent) {
    if (!acceptedKeys.delete(event.key)) swallow(event);
  }
</script>

<div class="confirm-overlay">
  <div
    class="confirm-dialog"
    role="dialog"
    aria-modal="true"
    aria-labelledby={titleId}
    tabindex="-1"
    bind:this={dialogEl}
    onkeydowncapture={handleKeydown}
    onkeyupcapture={handleKeyup}
  >
    <div class="confirm-header">
      <span class="warn-icon">&#9888;</span>
      <span id={titleId}>{$_("confirm.title")}</span>
    </div>

    <p class="confirm-body">
      {$_("confirm.body")}
    </p>

    {#if riskAssessment}
      <div class="world-model-warning" role="alert">
        <strong>
          World model interruption ·
          {Math.round(Number(riskAssessment.world_model_score ?? 0) * 100)}% predicted risk
        </strong>
        <span>{reason}</span>
        <small>
          Prediction: {Array.isArray(riskAssessment.prediction_sources)
            ? riskAssessment.prediction_sources.join(" + ")
            : "world model"}
        </small>
      </div>
    {/if}

    {#if reason && !riskAssessment}
      <p class="confirm-reason">{reason}</p>
    {/if}

    {#if gnomon && gnomon.escalate}
      <div class="gnomon-review" role="note">
        <strong>Gnomon review</strong>
        {#each Array.isArray(gnomon.reasons) ? gnomon.reasons : [] as line}
          <span>{line}</span>
        {/each}
        <small>Advisory: Gnomon can ask for approval but never grants it.</small>
      </div>
    {/if}

    <ul class="confirm-list">
      {#each actions as action, i}
        <li>
          <div class="confirm-row">
            <label class="confirm-checkbox">
              <input type="checkbox" checked={approved[i]} disabled={submitting} onchange={() => toggle(i)} />
            </label>
            <strong>{action.action_type}</strong> on
            <code>{action.target}</code>
            {#if action.requires_root}
              <span class="root-tag">{$_("tier.root")}</span>
            {/if}
            {#if action.destructive}
              <span class="destructive-tag">{$_("tier.destructive")}</span>
            {/if}
            {#if action.irreversible}
              <span class="irreversible-tag">{$_("tier.irreversible")}</span>
            {/if}
          </div>
          {#each describeAction(action) as detail (detail.label)}
            <div class="action-detail">
              <span class="detail-label">{detail.label}</span>
              {#if detail.code}<pre>{detail.value}</pre>{:else}<span class="detail-value">{detail.value}</span>{/if}
            </div>
          {/each}
        </li>
      {/each}
    </ul>

    {#if error}
      <p class="confirm-error" role="alert">{error}</p>
    {/if}

    <div class="confirm-actions">
      <button
        class="btn-deny"
        title={$_("confirm.deny")}
        disabled={submitting}
        bind:this={denyButton}
        onclick={handleDeny}>{$_("confirm.deny")}</button
      >
      <button
        class="btn-confirm"
        title={$_("confirm.approve")}
        disabled={!anyApproved || submitting}
        onclick={handleConfirm}
      >
        {submitting ? $_("confirm.submitting") : $_("confirm.approve")}
      </button>
    </div>
  </div>
</div>

<style>
  .confirm-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    padding: 24px;
  }

  .confirm-dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    max-width: 560px;
    max-height: calc(100vh - 48px);
    overflow-y: auto;
    width: 100%;
    box-shadow: var(--shadow);
  }

  .confirm-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
  }

  .warn-icon {
    color: var(--warning);
    font-size: 20px;
  }

  .confirm-body {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 14px;
    line-height: 1.5;
  }

  .confirm-list {
    list-style: none;
    padding: 0;
    margin-bottom: 20px;
  }

  .confirm-list li {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 13px;
  }

  .confirm-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .action-detail {
    display: grid;
    gap: 3px;
    margin-top: 8px;
  }

  .detail-label {
    color: var(--text-muted);
    font-size: 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .detail-value {
    overflow-wrap: anywhere;
  }

  .action-detail pre {
    max-height: 180px;
    margin: 0;
    overflow: auto;
    padding: 8px;
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    font: 11px/1.45 var(--font-mono);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .confirm-list code {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
  }

  .root-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--danger-bg);
    color: var(--danger);
  }

  .destructive-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 10px;
    background: rgba(251, 191, 36, 0.1);
    color: var(--warning);
  }

  .irreversible-tag {
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 10px;
    background: var(--danger);
    color: white;
  }

  .confirm-checkbox {
    display: flex;
    align-items: center;
    margin-right: 2px;
  }

  .confirm-checkbox input {
    width: 15px;
    height: 15px;
    accent-color: var(--accent);
    cursor: pointer;
  }

  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
  }

  .confirm-error {
    margin: -8px 0 16px;
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    background: var(--danger-bg);
    color: var(--danger);
    font-size: 12px;
    line-height: 1.4;
  }

  .world-model-warning {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin: 0 0 14px;
    padding: 10px 12px;
    color: var(--warning);
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: var(--radius-sm);
    font-size: 12px;
    line-height: 1.4;
  }

  .confirm-reason {
    margin: 0 0 14px;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text-primary);
  }

  .gnomon-review {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 0 0 14px;
    padding: 10px 12px;
    border: 1px solid var(--accent-muted);
    border-radius: var(--radius-sm);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    font-size: 12px;
    line-height: 1.4;
  }

  .gnomon-review strong {
    color: var(--accent);
  }

  .gnomon-review small {
    color: var(--text-muted);
  }

  .world-model-warning span {
    color: var(--text-primary);
  }

  .world-model-warning small {
    color: var(--text-muted);
  }

  .btn-deny {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .btn-deny:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .btn-confirm {
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .btn-confirm:hover {
    background: var(--accent-hover);
  }

  .btn-confirm:disabled {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    cursor: not-allowed;
  }
</style>
