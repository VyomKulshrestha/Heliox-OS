<script lang="ts" module>
  export interface StoredLine {
    id: string;
    text: string;
  }

  type Choice<T extends string> = { value: T; label: string };
  export const LENGTHS: Choice<"short" | "detailed">[] = [
    { value: "short", label: "Short answers" },
    { value: "detailed", label: "More detail" },
  ];
  export const SPEAKING: Choice<"speak_up" | "quiet">[] = [
    { value: "speak_up", label: "Speak up when you notice something" },
    { value: "quiet", label: "Stay quiet unless I ask" },
  ];
</script>

<script lang="ts">
  /** The three getting-to-know questions as a form: every field optional, every stored line shown and removable. */
  import { call } from "../api/daemon";

  let { ondone }: { ondone?: () => void } = $props();

  let name = $state("");
  let use = $state("");
  let length = $state<"" | "short" | "detailed">("");
  let speaking = $state<"" | "speak_up" | "quiet">("");
  let stored = $state<StoredLine[] | null>(null);
  let problems = $state<string[]>([]);
  let error = $state("");
  let busy = $state(false);

  async function save(event: SubmitEvent) {
    event.preventDefault();
    busy = true;
    try {
      const result = await call<{
        status: string;
        stored?: StoredLine[];
        problems?: string[];
        message?: string;
      }>("getting_to_know", { action: "save", name, use, length, speaking });
      if (result.status === "ok") {
        stored = result.stored ?? [];
        problems = result.problems ?? [];
        error = "";
      } else {
        error = result.message || "That did not work.";
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Heliox is not connected.";
    } finally {
      busy = false;
    }
  }

  async function remove(line: StoredLine) {
    try {
      const result = await call<{ status: string; message?: string }>("getting_to_know", {
        action: "remove",
        id: line.id,
      });
      if (result.status === "ok") stored = (stored ?? []).filter((item) => item.id !== line.id);
      else error = result.message || "That did not work.";
    } catch (err) {
      error = err instanceof Error ? err.message : "Heliox is not connected.";
    }
  }
</script>

{#if stored === null}
  <form class="gtk-form" onsubmit={save} data-testid="gtk-form">
    <label>
      <span>What should I call you?</span>
      <input type="text" bind:value={name} maxlength="40" autocomplete="given-name" />
    </label>
    <label>
      <span>What do you mainly use this computer for?</span>
      <input type="text" bind:value={use} maxlength="160" placeholder="Writing, video calls, coding..." />
    </label>
    <fieldset>
      <legend>How should I talk to you?</legend>
      <div class="choices">
        {#each LENGTHS as choice (choice.value)}
          <label class="choice" class:selected={length === choice.value}>
            <input type="radio" name="gtk-length" value={choice.value} bind:group={length} />
            {choice.label}
          </label>
        {/each}
      </div>
      <div class="choices">
        {#each SPEAKING as choice (choice.value)}
          <label class="choice" class:selected={speaking === choice.value}>
            <input type="radio" name="gtk-speaking" value={choice.value} bind:group={speaking} />
            {choice.label}
          </label>
        {/each}
      </div>
    </fieldset>
    <p class="note">Every field is optional. It stays on this computer.</p>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="actions">
      <button type="submit" class="primary" disabled={busy}>Save</button>
      <button type="button" class="quiet" onclick={() => ondone?.()}>Cancel</button>
    </div>
  </form>
{:else}
  <div class="gtk-stored" data-testid="gtk-stored">
    {#if stored.length}
      <p>Here's what I kept:</p>
      <ul>
        {#each stored as line (line.id)}
          <li>
            <span>{line.text}</span>
            <button type="button" onclick={() => void remove(line)} aria-label={`Remove: ${line.text}`}>Remove</button>
          </li>
        {/each}
      </ul>
    {:else}
      <p>Nothing was kept. You can tell me about yourself any time.</p>
    {/if}
    {#each problems as problem (problem)}<p class="error">{problem}</p>{/each}
    {#if error}<p class="error" role="alert">{error}</p>{/if}
    <div class="actions">
      <button type="button" class="primary" onclick={() => ondone?.()}>Done</button>
    </div>
  </div>
{/if}

<style>
  .gtk-form,
  .gtk-stored {
    display: grid;
    gap: 8px;
  }

  label > span,
  legend {
    display: block;
    margin-bottom: 4px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-primary);
  }

  input[type="text"] {
    width: 100%;
    height: 30px;
    padding: 0 8px;
    font-size: var(--text-sm);
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  input[type="text"]:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }

  fieldset {
    display: grid;
    gap: 6px;
    margin: 0;
    padding: 0;
    border: 0;
  }

  .choices {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .choice {
    padding: 4px 9px;
    font-size: var(--text-xs);
    color: var(--text-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    cursor: pointer;
  }

  .choice input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  .choice:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .choice.selected {
    color: var(--text-primary);
    border-color: var(--accent-line);
    background: var(--accent-muted);
  }

  p {
    margin: 0;
    font-size: var(--text-sm);
    line-height: 1.45;
    color: var(--text-primary);
  }

  .note {
    font-size: var(--text-xs);
    color: var(--text-secondary);
  }

  .error {
    font-size: var(--text-xs);
    color: var(--danger);
  }

  ul {
    display: grid;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  li {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: var(--text-sm);
    color: var(--text-primary);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  button {
    height: 28px;
    padding: 0 10px;
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  button:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .primary {
    color: var(--on-accent);
    background: var(--accent);
    border-color: transparent;
  }

  .primary:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .quiet {
    color: var(--text-secondary);
    background: transparent;
    border-color: transparent;
  }
</style>
