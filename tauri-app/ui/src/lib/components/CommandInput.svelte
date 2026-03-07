<script lang="ts">
  import { session } from "../stores/session";

  let input = $state("");

  function handleSubmit(e: Event) {
    e.preventDefault();
    const text = input.trim();
    if (!text) return;
    session.sendCommand(text);
    input = "";
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }
</script>

<form class="command-input" onsubmit={handleSubmit}>
  <div class="input-wrapper">
    <span class="prompt">&gt;</span>
    <input
      type="text"
      bind:value={input}
      placeholder="Tell Cortex-OS what to do..."
      onkeydown={handleKeydown}
      autocomplete="off"
      spellcheck="false"
    />
    <button type="submit" class="send-btn" disabled={!input.trim()}>
      Send
    </button>
  </div>
</form>

<style>
  .command-input {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
  }

  .input-wrapper {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    transition: border-color 0.15s;
  }

  .input-wrapper:focus-within {
    border-color: var(--accent);
  }

  .prompt {
    color: var(--accent);
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 15px;
  }

  input {
    flex: 1;
    background: transparent;
    color: var(--text-primary);
    font-size: 14px;
  }

  input::placeholder {
    color: var(--text-muted);
  }

  .send-btn {
    padding: 5px 16px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border-radius: var(--radius-sm);
    transition: background 0.15s;
  }

  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: default;
  }
</style>
