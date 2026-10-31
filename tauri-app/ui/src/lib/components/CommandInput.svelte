<script lang="ts">
  import { session } from "../stores/session";
  import { invoke } from "@tauri-apps/api/core";
  import { getCurrentWebview } from "@tauri-apps/api/webview";
  import { ArrowUp, FileText, Square, X } from "@lucide/svelte";

  let input = $state("");
  let inTaskMode = $state<"correct" | "ask" | "next">("correct");
  let field: HTMLTextAreaElement | undefined = $state();
  const MAX_CHARS = 20000;
  /** The counter appears only when a message gets close to the limit. */
  const COUNTER_FROM = 18000;
  const MAX_FIELD_HEIGHT = 140;

  // Accept prefill text from parent (e.g. CommandHistory replay)
  let { prefill = "" }: { prefill?: string } = $props();

  // When prefill changes, populate the input box without running
  $effect(() => {
    if (prefill) {
      input = prefill;
      field?.focus();
    }
  });

  // Grow with the text up to a few lines, then scroll.
  $effect(() => {
    void input;
    if (!field) return;
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, MAX_FIELD_HEIGHT)}px`;
  });

  type Attachment = {
    name: string;
    type: string;
    content: string;
  };

  let attachments = $state<Attachment[]>([]);
  let isDragging = $state(false);

  $effect(() => {
    let unlisten: () => void;

    // Attempt to register Tauri native drag/drop listener (WebView API)
    async function setupTauri() {
      try {
        unlisten = await getCurrentWebview().onDragDropEvent(async (event) => {
          const payload = event.payload;

          if (payload.type === "enter" || payload.type === "over") {
            isDragging = true;
          } else if (payload.type === "leave") {
            isDragging = false;
          } else if (payload.type === "drop") {
            isDragging = false;
            const paths = payload.paths || [];

            for (const path of paths) {
              try {
                // Register path in the allowlist before reading
                await invoke("register_allowed_path", { path });
                // Extract filename
                const filename = path.split("\\").pop()?.split("/").pop() || "unknown";
                const content = await invoke("extract_file_text", { path });

                attachments = [...attachments, { name: filename, type: "file", content: content as string }];
              } catch (err) {
                console.warn(`Failed to extract text from ${path}:`, err);
              }
            }
          }
        });
      } catch (e) {
        // Not running in Tauri (e.g. browser), ignore gracefully
      }
    }

    setupTauri();

    return () => {
      if (unlisten) unlisten();
    };
  });

  function handleSubmit(e: Event) {
    e.preventDefault();

    const text = input.trim();

    if ((!text && attachments.length === 0) || input.length >= MAX_CHARS) return;

    session.sendCommand(text, attachments, inTaskMode);
    input = "";
    attachments = [];
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      handleSubmit(e);
    }
  }

  function removeAttachment(index: number) {
    attachments = attachments.filter((_, i) => i !== index);
  }

  // Esc stops a running task, unless a dialog or a select has the key.
  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key !== "Escape" || e.defaultPrevented || !$session.loading) return;
    if (document.querySelector('[role="dialog"], [role="alertdialog"], [aria-modal="true"]')) return;
    if (e.target instanceof HTMLSelectElement) return;
    e.preventDefault();
    void session.abort();
  }

  let placeholder = $derived(
    $session.loading
      ? inTaskMode === "ask"
        ? "Ask about the task or screen..."
        : inTaskMode === "next"
          ? "Describe the next task..."
          : "Correct or stop the running task..."
      : "Tell Heliox what to do...",
  );
</script>

<svelte:window onkeydown={onWindowKeydown} />

<form class="command-input" onsubmit={handleSubmit}>
  {#if attachments.length}
    <div class="attachment-row">
      {#each attachments as file, i}
        <span class="attachment-chip">
          <FileText size={13} aria-hidden="true" />
          <span class="attachment-name">{file.name}</span>
          <button type="button" aria-label={`Remove ${file.name}`} onclick={() => removeAttachment(i)}>
            <X size={12} />
          </button>
        </span>
      {/each}
    </div>
  {/if}

  <div class="input-wrapper" class:dragging={isDragging} class:busy={$session.loading}>
    {#if $session.loading}
      <select class="in-task-mode" aria-label="While task runs" bind:value={inTaskMode}>
        <option value="correct">Correct task</option>
        <option value="ask">Ask without stopping</option>
        <option value="next">Queue next task</option>
      </select>
    {/if}

    <textarea
      bind:this={field}
      bind:value={input}
      rows="1"
      maxlength="20000"
      aria-label={placeholder}
      {placeholder}
      onkeydown={handleKeydown}
      autocomplete="off"
      spellcheck="false"></textarea>

    {#if input.length >= COUNTER_FROM}
      <span class="char-counter tabular" class:over={input.length >= MAX_CHARS}>
        {input.length}/{MAX_CHARS}
      </span>
    {/if}

    {#if $session.loading}
      <button
        type="button"
        class="stop-btn"
        title="Stop the running task (Esc)"
        aria-label="Stop the running task"
        onclick={() => void session.abort()}
      >
        <Square size={11} fill="currentColor" aria-hidden="true" />
        <span>Stop</span>
      </button>
    {/if}

    <button
      type="submit"
      class="send-btn"
      class:labelled={$session.loading}
      title={$session.loading
        ? inTaskMode === "ask"
          ? "Ask without interrupting"
          : inTaskMode === "next"
            ? "Queue another task"
            : "Correct the running task"
        : "Send (Enter)"}
      aria-label={$session.loading ? undefined : "Send"}
      disabled={(!input.trim() && attachments.length === 0) || input.length >= MAX_CHARS}
    >
      {#if $session.loading}
        {inTaskMode === "ask" ? "Ask" : inTaskMode === "next" ? "Queue" : "Correct"}
      {:else}
        <ArrowUp size={16} aria-hidden="true" />
      {/if}
    </button>
  </div>

  {#if $session.queuedTasks.some((task) => task.sessionId === $session.activeSessionId)}
    <div class="queued-tasks" aria-label="Queued tasks">
      {#each $session.queuedTasks.filter((task) => task.sessionId === $session.activeSessionId) as task (task.taskId)}
        <div class="queued-task">
          <span class="queued-task-label" title={task.input}
            >{task.status === "running" ? "Running" : "Queued"}: {task.input}</span
          >
          {#if task.status === "queued" && task.resumeToken}
            <button
              type="button"
              class="cancel-queued"
              aria-label={`Cancel queued task: ${task.input}`}
              onclick={() => session.cancelQueuedTask(task.taskId)}
            >
              Cancel
            </button>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</form>

<style>
  .command-input {
    min-width: 0;
  }

  .input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    min-height: 40px;
    padding: 4px 4px 4px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    transition:
      border-color var(--dur-fast) ease,
      box-shadow var(--dur-fast) ease;
  }

  .input-wrapper:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-muted);
  }

  .input-wrapper.busy {
    padding-left: 4px;
  }

  textarea {
    flex: 1;
    align-self: center;
    min-width: 0;
    max-height: 140px;
    padding: 5px 0;
    resize: none;
    overflow-y: auto;
    font-size: var(--text-md);
    line-height: 1.45;
    color: var(--text-primary);
    background: transparent;
  }

  /* The wrapper draws the focus ring for the field. */
  textarea:focus-visible {
    outline: none;
  }

  textarea::placeholder {
    color: var(--text-muted);
  }

  .in-task-mode {
    flex: none;
    height: 30px;
    max-width: 150px;
    padding: 0 6px;
    font-size: var(--text-sm);
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
  }

  .send-btn {
    display: inline-flex;
    flex: none;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--on-accent);
    background: var(--accent);
    border-radius: var(--radius-md);
    transition:
      background var(--dur-fast) ease,
      transform var(--dur-fast) ease,
      opacity var(--dur-fast) ease;
  }

  .send-btn.labelled {
    width: auto;
    padding: 0 12px;
  }

  .send-btn:hover:not(:disabled) {
    background: var(--accent-hover);
    color: var(--on-accent);
  }

  .send-btn:active:not(:disabled) {
    transform: translateY(1px);
  }

  .send-btn:disabled {
    opacity: 0.35;
    cursor: default;
  }

  .stop-btn {
    display: inline-flex;
    flex: none;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 11px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--danger);
    background: var(--danger-bg);
    border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
    border-radius: var(--radius-md);
    transition: background var(--dur-fast) ease;
  }

  .stop-btn:hover {
    background: color-mix(in srgb, var(--danger) 20%, transparent);
  }

  .char-counter {
    flex: none;
    align-self: center;
    font-size: var(--text-xs);
    color: var(--text-muted);
  }

  .char-counter.over {
    color: var(--danger);
  }

  .queued-tasks {
    display: grid;
    gap: 4px;
    margin-top: 8px;
  }

  .queued-task {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-size: var(--text-sm);
    color: var(--text-secondary);
  }

  .queued-task-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cancel-queued {
    flex: none;
    min-height: 28px;
    padding: 4px 8px;
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .attachment-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .attachment-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    max-width: 240px;
    padding: 4px 4px 4px 9px;
    font-size: var(--text-sm);
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
  }

  .attachment-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .attachment-chip button {
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    color: var(--text-muted);
    border-radius: 50%;
  }

  .attachment-chip button:hover {
    color: var(--text-primary);
    background: var(--bg-hover);
  }

  .input-wrapper.dragging {
    border-color: var(--accent);
    background: var(--accent-muted);
  }
</style>
