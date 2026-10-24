<script lang="ts">
  import { onMount, tick } from "svelte";
  import { call } from "../api/daemon";

  type Group = { group: string; about: string; examples: string[] };
  type Item = { text: string; label: string };

  interface Props {
    /** Puts the chosen text in the command box; nothing runs until the person presses Enter there. */
    onpick: (text: string) => void;
    onclose: () => void;
  }

  let { onpick, onclose }: Props = $props();

  let query = $state("");
  let items = $state<Item[]>([]);
  let selected = $state(0);
  let input: HTMLInputElement | undefined = $state();

  let shown = $derived.by(() => {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const matches = items.filter((item) =>
      words.every((word) => `${item.text} ${item.label}`.toLowerCase().includes(word)),
    );
    return matches.slice(0, 40);
  });

  $effect(() => {
    if (selected >= shown.length) selected = Math.max(0, shown.length - 1);
  });

  async function load() {
    const loaded: Item[] = [];
    try {
      const recent = await call<{ status: string; receipts?: { request: string }[] }>("receipts_list", { limit: 20 });
      const seen = new Set<string>();
      for (const row of recent.receipts ?? []) {
        const text = row.request?.trim();
        if (text && !seen.has(text.toLowerCase())) {
          seen.add(text.toLowerCase());
          loaded.push({ text, label: "Recent" });
        }
      }
    } catch {
      // Recent requests are optional; the guide below still works.
    }
    try {
      const guide = await call<{ status: string; groups?: Group[] }>("command_guide");
      for (const group of guide.groups ?? []) {
        for (const example of group.examples) loaded.push({ text: example, label: group.group });
      }
    } catch {
      // Without the daemon the palette only shows what it has.
    }
    items = loaded;
  }

  function pick(item: Item | undefined) {
    if (!item) return;
    onpick(item.text);
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      onclose();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      selected = Math.min(selected + 1, shown.length - 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      selected = Math.max(selected - 1, 0);
    } else if (event.key === "Enter") {
      event.preventDefault();
      pick(shown[selected] ?? (query.trim() ? { text: query.trim(), label: "" } : undefined));
    }
  }

  onMount(() => {
    void load();
    void tick().then(() => input?.focus());
  });
</script>

<div class="palette-overlay" role="presentation" onclick={onclose}>
  <div
    class="palette"
    role="dialog"
    aria-modal="true"
    aria-label="Command palette"
    tabindex="-1"
    onclick={(event) => event.stopPropagation()}
    onkeydown={onKey}
  >
    <input
      bind:this={input}
      bind:value={query}
      oninput={() => (selected = 0)}
      placeholder="Type a command or search what you asked before"
      aria-label="Search commands"
      autocomplete="off"
    />
    <ul class="results" role="listbox" aria-label="Commands">
      {#each shown as item, index (item.label + item.text)}
        <li role="option" aria-selected={index === selected}>
          <button
            type="button"
            class:selected={index === selected}
            onmouseenter={() => (selected = index)}
            onclick={() => pick(item)}
          >
            <span class="text">{item.text}</span>
            <span class="label">{item.label}</span>
          </button>
        </li>
      {/each}
      {#if shown.length === 0}
        <li class="empty">Press Enter to put "{query}" in the command box.</li>
      {/if}
    </ul>
    <p class="hint">Enter puts it in the command box to check or edit before it runs. Esc closes.</p>
  </div>
</div>

<style>
  .palette-overlay {
    position: absolute;
    inset: 0;
    z-index: 90;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 72px 24px 24px;
    background: rgba(0, 0, 0, 0.45);
  }
  .palette {
    width: 100%;
    max-width: 560px;
    max-height: 70vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--bg-secondary);
    box-shadow: var(--shadow);
  }
  input {
    padding: 14px 16px;
    border: 0;
    border-bottom: 1px solid var(--border);
    outline: none;
    background: transparent;
    color: var(--text-primary);
    font-size: 14px;
  }
  .results {
    flex: 1;
    margin: 0;
    padding: 6px;
    overflow-y: auto;
    list-style: none;
  }
  .results button {
    width: 100%;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 8px 10px;
    border: 0;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-primary);
    font-size: 13px;
    text-align: left;
    cursor: pointer;
  }
  .results button.selected {
    background: var(--bg-hover);
  }
  .text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .label {
    flex-shrink: 0;
    color: var(--text-secondary);
    font-size: 11px;
  }
  .empty,
  .hint {
    margin: 0;
    padding: 8px 12px;
    color: var(--text-secondary);
    font-size: 11px;
  }
  .hint {
    border-top: 1px solid var(--border);
  }
</style>
