<script lang="ts">
  import { session } from "../stores/session";
  import { invoke } from "@tauri-apps/api/core";
  import { getCurrentWebview } from "@tauri-apps/api/webview";

  let input = $state("");
  let inTaskMode = $state<"correct" | "ask" | "next">("correct");
  const MAX_CHARS = 20000;

  // Accept prefill text from parent (e.g. CommandHistory replay)
  let { prefill = "" }: { prefill?: string } = $props();

  // When prefill changes, populate the input box without running
  $effect(() => {
    if (prefill) {
      input = prefill;
    }
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
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
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

  <div class="input-wrapper" class:dragging={isDragging}>
    <span class="prompt">&gt;</span>

    {#if $session.loading}
      <select class="in-task-mode" aria-label="While task runs" bind:value={inTaskMode}>
        <option value="correct">Correct task</option>
        <option value="ask">Ask without stopping</option>
        <option value="next">Queue next task</option>
      </select>
    {/if}

    <input
      type="text"
      bind:value={input}
      maxlength="20000"
      placeholder={$session.loading
        ? inTaskMode === "ask"
          ? "Ask about the task or screen..."
          : inTaskMode === "next"
            ? "Describe the next task..."
            : "Correct or stop the running task..."
        : "Tell Heliox OS what to do..."}
      onkeydown={handleKeydown}
      autocomplete="off"
      spellcheck="false"
    />
    <div class="char-counter" style:color={input.length >= MAX_CHARS ? "red" : "#888"}>
      {input.length}/{MAX_CHARS}
    </div>

    <button
      type="submit"
      class="send-btn"
      title={$session.loading
        ? inTaskMode === "ask"
          ? "Ask without interrupting"
          : inTaskMode === "next"
            ? "Queue another task"
            : "Correct the running task"
        : "Send"}
      disabled={(!input.trim() && attachments.length === 0) || input.length >= MAX_CHARS}
    >
      {$session.loading ? (inTaskMode === "ask" ? "Ask" : inTaskMode === "next" ? "Queue" : "Correct") : "Send"}
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

  .in-task-mode {
    width: min(170px, 40vw);
    min-height: 44px;
    background: var(--bg-secondary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-size: 12px;
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
    font-size: 12px;
    color: var(--text-secondary);
  }

  .queued-task-label {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cancel-queued {
    flex: none;
    min-height: 32px;
    padding: 4px 8px;
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .cancel-queued:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  .in-task-mode:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
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
