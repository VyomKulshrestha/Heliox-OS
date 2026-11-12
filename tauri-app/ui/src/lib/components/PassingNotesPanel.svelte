<script lang="ts">
  /** One switch for notes said in passing ("my flight is on Friday"), and what is never kept that way. */
  import { settings } from "../stores/settings";

  let error = $state("");
  let saving = $state(false);
  // Settings may not have loaded yet, or may come back without a memory section: the default (on) applies.
  const on = $derived(
    (($settings as unknown as { memory?: { passing_notes?: boolean } } | null)?.memory?.passing_notes ?? true) !==
      false,
  );

  async function toggle(value: boolean) {
    saving = true;
    try {
      const result = await settings.updateSectionDetailed("memory", { passing_notes: value });
      error = result?.ok ? "" : result?.message || "The setting was not saved.";
    } catch (err) {
      error = err instanceof Error ? err.message : "Heliox is not connected, so the setting was not saved.";
    } finally {
      saving = false;
    }
  }
</script>

<section class="notes-panel" aria-labelledby="passing-notes-heading" data-testid="passing-notes-panel">
  <div class="section-title">
    <h3 id="passing-notes-heading">Things said in passing</h3>
    <p>
      When you mention something to Heliox, like "my flight is on Friday" or "I moved the meeting to three", it keeps a
      short note so it still knows tomorrow, and tells you so ("Noted: ..."). A note goes by itself the day after the
      date it names, or after a week. Say "forget about my flight" to drop one sooner. "Remember that ..." works either
      way.
    </p>
  </div>
  {#if error}<div class="message" role="alert">{error}</div>{/if}
  <div class="block">
    <label class="check">
      <input
        type="checkbox"
        checked={on}
        disabled={saving}
        onchange={(event) => void toggle(event.currentTarget.checked)}
      />
      <span>Keep notes of things I mention in passing</span>
    </label>
    <p class="never">
      Never kept this way, whatever the wording: passwords, keys and codes; health, including appointments with a doctor
      or dentist; money amounts, rent and pay; and other people's private details beyond a name and how you know them.
    </p>
  </div>
</section>

<style>
  .notes-panel {
    margin-top: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-secondary);
    overflow: hidden;
  }

  .section-title {
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

  .check {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    font-size: var(--text-sm);
    line-height: 1.45;
    color: var(--text-primary);
    cursor: pointer;
  }

  .check input {
    margin-top: 3px;
    accent-color: var(--accent);
  }

  .never {
    padding-left: 26px;
    font-size: var(--text-xs);
  }

  @media (max-width: 760px) {
    .section-title,
    .block {
      padding-inline: 14px;
    }
  }
</style>
