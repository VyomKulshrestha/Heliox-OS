<script lang="ts">
  import { session } from "../stores/session";

  let input = $state("");
  const MAX_CHARS = 20000;

  type Attachment = {
    name: string;
    type: string;
    content: string;
  };

  let attachments = $state<Attachment[]>([]);
  let isDragging = $state(false);

  const MAX_FILE_SIZE = 500 * 1024;

  const ALLOWED_TEXT_TYPES = ["text/plain", "text/markdown", "application/json", "application/xml"];

  function isTextLikeFile(file: File): boolean {
    return (
      file.type.startsWith("text/") ||
      ALLOWED_TEXT_TYPES.includes(file.type) ||
      /\.(ts|js|jsx|tsx|py|java|cpp|c|h|hpp|cs|go|rs|md|txt|json|xml|yaml|yml)$/i.test(file.name)
    );
  }

  function handleSubmit(e: Event) {
    e.preventDefault();

    const text = input.trim();

    if ((!text && attachments.length === 0) || input.length >= MAX_CHARS) return;

    session.sendCommand(text, attachments);
    input = "";
    attachments = [];
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  }
  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    isDragging = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    isDragging = false;
  }

  async function handleDrop(e: DragEvent) {
    e.preventDefault();
    isDragging = false;

    const droppedFiles = e.dataTransfer?.files;

    if (!droppedFiles?.length) return;

    for (const file of droppedFiles) {
      try {
        if (file.size > MAX_FILE_SIZE) {
          console.warn(`Skipping ${file.name}: file too large`);
          continue;
        }

        if (!isTextLikeFile(file)) {
          console.warn(`Skipping ${file.name}: unsupported file type`);
          continue;
        }

        const content = await file.text();

        attachments = [
          ...attachments,
          {
            name: file.name,
            type: file.type,
            content,
          },
        ];
      } catch (err) {
        console.error("Failed to read file:", err);
      }
    }
  }
</script>

<form class="command-input" onsubmit={handleSubmit}>
  {#if attachments.length}
    <div class="attachment-row">
      {#each attachments as file}
        <div class="attachment-chip">
          📄 {file.name}
        </div>
      {/each}
    </div>
  {/if}

  <div
    class="input-wrapper"
    class:dragging={isDragging}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    ondrop={handleDrop}
  >
    <span class="prompt">&gt;</span>

    <input
      type="text"
      bind:value={input}
      maxlength="20000"
      placeholder="Tell Heliox OS what to do..."
      onkeydown={handleKeydown}
      autocomplete="off"
      spellcheck="false"
    />
    <div
      class="char-counter"
      style:color={input.length >= MAX_CHARS ? "red" : "#888"}
    >
      {input.length}/{MAX_CHARS}
    </div>

    <button
      type="submit"
      class="send-btn"
      title="Send"
      disabled={(!input.trim() && attachments.length === 0) || input.length >= MAX_CHARS}
    >
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

  .char-counter {
    display: flex;
    align-items: center;
    font-size: 12px;
    text-align: right;
    white-space: nowrap;
  }

  .attachment-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
    padding-left: 2px;
  }

  .attachment-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: var(--radius-sm);
    background: var(--bg-primary);
    border: 1px solid var(--border);
    font-size: 12px;
    color: var(--text-secondary);
  }

  .input-wrapper.dragging {
    border-color: var(--accent);
    background: rgba(100, 150, 255, 0.08);
  }
</style>
