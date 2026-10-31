<script lang="ts">
  import { Download, History, MessageSquarePlus, X } from "@lucide/svelte";
  import { session } from "../stores/session";
  import type { ChatSessionSummary } from "../utils/chatSessions";

  let { onactivate }: { onactivate: () => void } = $props();
  let open = $state(false);
  let chats: ChatSessionSummary[] = $state([]);

  function refresh() {
    chats = session.listChatSessions();
  }

  function openDialog() {
    refresh();
    open = true;
  }

  function startNewChat() {
    if (session.newChat()) {
      onactivate();
      open = false;
    }
  }

  function selectChat(sessionId: string) {
    if (session.switchChatSession(sessionId)) {
      onactivate();
      open = false;
    }
  }

  function formatUpdated(timestamp: number): string {
    const date = new Date(timestamp);
    const today = new Date();
    return date.toDateString() === today.toDateString()
      ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : date.toLocaleDateString([], { month: "short", day: "numeric" });
  }
</script>

<svelte:window
  onkeydown={(event) => {
    if (event.key === "Escape") open = false;
  }}
/>

<div class="session-controls">
  <button
    class="session-button"
    disabled={$session.loading}
    title={$session.loading ? "Finish or stop the current task before starting a new chat" : "Start a new chat"}
    aria-label="Start a new chat"
    onclick={startNewChat}
  >
    <MessageSquarePlus size={15} />
    <span>New chat</span>
  </button>
  <button class="session-button icon-only" title="View all chats" aria-label="View all chats" onclick={openDialog}>
    <History size={16} />
  </button>
</div>

{#if open}
  <div class="dialog-backdrop" role="presentation" onclick={() => (open = false)}>
    <div
      class="chat-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="chat-dialog-title"
      tabindex="-1"
      onclick={(event) => event.stopPropagation()}
      onkeydown={(event) => {
        if (event.key === "Escape") open = false;
        event.stopPropagation();
      }}
    >
      <header>
        <div>
          <span class="eyebrow">Conversation history</span>
          <h2 id="chat-dialog-title">Your chats</h2>
        </div>
        <button class="close-button" aria-label="Close chat history" onclick={() => (open = false)}>
          <X size={18} />
        </button>
      </header>

      <button class="new-chat-card" disabled={$session.loading} onclick={startNewChat}>
        <MessageSquarePlus size={18} />
        <span>
          <strong>New chat</strong>
          <small>Start a clean task while keeping learned preferences</small>
        </span>
      </button>

      <div class="chat-list">
        {#each chats as chat (chat.id)}
          <button
            class="chat-row"
            class:active={chat.id === $session.activeSessionId}
            disabled={$session.loading && chat.id !== $session.activeSessionId}
            onclick={() => selectChat(chat.id)}
          >
            <span class="chat-copy">
              <strong>{chat.title}</strong>
              <small>{chat.preview}</small>
            </span>
            <span class="chat-meta">
              <time datetime={new Date(chat.updatedAt).toISOString()}>{formatUpdated(chat.updatedAt)}</time>
              <small>{chat.messageCount} messages</small>
            </span>
          </button>
        {/each}
      </div>

      {#if $session.loading}
        <p class="busy-note">Finish or stop the active task before changing chats.</p>
      {/if}

      <footer class="export-row">
        <span>Export this chat</span>
        <button type="button" aria-label="Export this chat as JSON" onclick={() => session.exportChat("json")}>
          <Download size={14} aria-hidden="true" />JSON
        </button>
        <button type="button" aria-label="Export this chat as CSV" onclick={() => session.exportChat("csv")}>
          <Download size={14} aria-hidden="true" />CSV
        </button>
      </footer>
    </div>
  </div>
{/if}

<style>
  .session-controls {
    display: flex;
    align-items: center;
    gap: 4px;
    -webkit-app-region: no-drag;
  }

  .session-button,
  .close-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    color: var(--text-secondary);
    border: 1px solid var(--border);
    background: var(--bg-tertiary);
    border-radius: 8px;
    transition:
      color 0.15s,
      border-color 0.15s,
      background 0.15s;
  }

  .session-button {
    height: 30px;
    padding: 0 10px;
    font-size: 11px;
    font-weight: 600;
  }

  .session-button.icon-only {
    width: 30px;
    padding: 0;
  }

  /* In the default 700 px window the button keeps its icon and tooltip and drops the label. */
  @media (max-width: 880px) {
    .session-button:not(.icon-only) {
      width: 30px;
      padding: 0;
    }

    .session-button span {
      display: none;
    }
  }

  .export-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    font-size: var(--text-sm);
    color: var(--text-secondary);
  }

  .export-row span {
    margin-right: auto;
  }

  .export-row button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 12px;
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--text-primary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
  }

  .export-row button:hover {
    border-color: var(--accent-line);
    background: var(--accent-muted);
  }

  .session-button:hover:not(:disabled),
  .close-button:hover {
    color: var(--accent);
    border-color: var(--accent);
    background: var(--accent-muted);
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.45;
  }

  .dialog-backdrop {
    position: fixed;
    inset: 0;
    z-index: 1000;
    display: grid;
    place-items: center;
    padding: 24px;
    background: rgba(3, 5, 12, 0.7);
    backdrop-filter: blur(8px);
    -webkit-app-region: no-drag;
  }

  .chat-dialog {
    width: min(620px, 92vw);
    max-height: min(680px, 86vh);
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px;
    color: var(--text-primary);
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.5);
  }

  header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
  }

  .eyebrow {
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  h2 {
    margin: 3px 0 0;
    font-size: 21px;
  }

  .close-button {
    width: 32px;
    height: 32px;
  }

  .new-chat-card,
  .chat-row {
    width: 100%;
    display: flex;
    align-items: center;
    text-align: left;
    border: 1px solid var(--border);
    border-radius: 10px;
  }

  .new-chat-card {
    gap: 12px;
    padding: 12px 14px;
    color: var(--accent);
    background: var(--accent-muted);
    border-color: color-mix(in srgb, var(--accent) 38%, transparent);
  }

  .new-chat-card span,
  .chat-copy,
  .chat-meta {
    display: flex;
    flex-direction: column;
  }

  .new-chat-card strong {
    color: var(--text-primary);
  }

  small {
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 400;
  }

  .chat-list {
    min-height: 100px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .chat-row {
    justify-content: space-between;
    gap: 16px;
    padding: 11px 12px;
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  .chat-row:hover:not(:disabled),
  .chat-row.active {
    background: var(--bg-hover);
    border-color: var(--accent);
  }

  .chat-row.active {
    box-shadow: inset 3px 0 0 var(--accent);
  }

  .chat-copy {
    min-width: 0;
    gap: 4px;
  }

  .chat-copy strong,
  .chat-copy small {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .chat-meta {
    flex: none;
    align-items: flex-end;
    gap: 4px;
    color: var(--text-secondary);
    font-size: 11px;
  }

  .busy-note {
    margin: 0;
    color: var(--warning);
    font-size: 11px;
    text-align: center;
  }
</style>
