<!--
  LogMessage.svelte
  -----------------
  Renders a single agent log message with full Markdown support and
  syntax-highlighted code blocks.

  Place at: tauri-app/ui/src/components/LogMessage.svelte

  Props
  ──────────────────────────────────────────────────────────────────────────
  content   string   Raw message text (may contain Markdown)
  role      string   "agent" | "user" | "system"
  timestamp string   ISO timestamp string (optional)

  Features
  ──────────────────────────────────────────────────────────────────────────
  • Parses full Markdown via marked.js
  • Syntax highlights fenced code blocks via highlight.js (atom-one-dark)
  • Auto-detects language when no language tag is present
  • One-click copy button on every code block
  • Sanitises HTML output with DOMPurify before injection (XSS safety)
  • Graceful fallback: plain-text messages render without any markdown overhead
  • Re-uses Heliox's existing clipboard Tauri command for the copy action
-->

<script lang="ts">
  import { onMount, tick } from "svelte";
  import { marked, type Renderer } from "marked";
  import DOMPurify from "dompurify";
  import { highlight } from "../lib/highlighter";
  import { writeText } from "@tauri-apps/plugin-clipboard-manager";

  // ── Props ────────────────────────────────────────────────────────────────
  export let content: string = "";
  export let role: "agent" | "user" | "system" = "agent";
  export let timestamp: string = "";

  // ── State ────────────────────────────────────────────────────────────────
  let renderedHtml = "";
  let messageEl: HTMLElement;
  /** Tracks which code block UIDs have been copied (for button feedback) */
  let copiedBlocks = new Set<string>();

  // ── Markdown detection ───────────────────────────────────────────────────
  /**
   * Only run through the full marked pipeline when the message actually
   * contains Markdown constructs. Plain-text messages skip the parser
   * entirely to avoid unnecessary work.
   */
  function looksLikeMarkdown(text: string): boolean {
    return (
      text.includes("```") ||
      text.includes("`") ||
      text.includes("**") ||
      text.includes("# ") ||
      text.includes("- ") ||
      text.includes("1. ") ||
      text.includes("[") ||
      text.includes("> ")
    );
  }

  // ── Custom marked renderer ────────────────────────────────────────────────
  /**
   * Override the code block renderer so highlight.js handles colouring and
   * we inject our copy-button wrapper around each block.
   */
  function buildRenderer(): Partial<Renderer> {
    return {
      code({ text, lang }: { text: string; lang?: string }): string {
        const language = lang ?? "";
        const { value: highlighted, language: detected } = highlight(text, language);

        // Stable UID for this block (used by the copy handler)
        const uid = `hlx-${Math.random().toString(36).slice(2, 9)}`;

        const langLabel = detected !== "plaintext" ? detected : "";
        const langBadge = langLabel
          ? `<span class="hlx-lang-badge">${langLabel}</span>`
          : "";

        return `
          <div class="hlx-code-wrapper" data-uid="${uid}">
            <div class="hlx-code-header">
              ${langBadge}
              <button
                class="hlx-copy-btn"
                data-uid="${uid}"
                data-code="${encodeURIComponent(text)}"
                aria-label="Copy code to clipboard"
                title="Copy"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                     stroke="currentColor" stroke-width="2"
                     stroke-linecap="round" stroke-linejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                <span class="hlx-copy-label">Copy</span>
              </button>
            </div>
            <pre class="hlx-pre"><code class="hljs language-${detected}">${highlighted}</code></pre>
          </div>
        `;
      },
    };
  }

  // ── Render pipeline ──────────────────────────────────────────────────────
  async function render(text: string): Promise<string> {
    if (!text) return "";

    if (!looksLikeMarkdown(text)) {
      // Fast path: escape HTML and return plain text
      return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    marked.use({
      renderer: buildRenderer() as Renderer,
      gfm: true,        // GitHub Flavored Markdown
      breaks: true,     // Single newlines become <br>
    });

    const raw = await marked.parse(text);

    // Sanitise with DOMPurify — defence-in-depth even for local agent output
    return DOMPurify.sanitize(raw, {
      ADD_TAGS: ["svg", "path", "rect"],
      ADD_ATTR: [
        "viewBox", "fill", "stroke", "stroke-width",
        "stroke-linecap", "stroke-linejoin",
        "width", "height", "x", "y", "rx", "ry", "d",
        "data-uid", "data-code",
      ],
    });
  }

  // ── Copy handler ─────────────────────────────────────────────────────────
  async function handleCopyClick(e: MouseEvent): Promise<void> {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>(".hlx-copy-btn");
    if (!btn) return;

    const encoded = btn.dataset.code ?? "";
    const uid     = btn.dataset.uid ?? "";
    const code    = decodeURIComponent(encoded);

    try {
      // Use Tauri's clipboard API (matches existing Heliox clipboard_write action)
      await writeText(code);
    } catch {
      // Fallback for browser dev mode (npm run dev without Tauri shell)
      await navigator.clipboard.writeText(code);
    }

    copiedBlocks.add(uid);
    copiedBlocks = copiedBlocks; // trigger Svelte reactivity

    const label = btn.querySelector<HTMLSpanElement>(".hlx-copy-label");
    if (label) label.textContent = "Copied!";

    setTimeout(() => {
      copiedBlocks.delete(uid);
      copiedBlocks = copiedBlocks;
      if (label) label.textContent = "Copy";
    }, 2000);
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────
  onMount(async () => {
    renderedHtml = await render(content);
    await tick();
    // Attach click delegation on the message container
    messageEl?.addEventListener("click", handleCopyClick);
  });

  // Re-render if content prop changes (streaming messages)
  $: (async () => {
    renderedHtml = await render(content);
  })();
</script>

<!-- ── Template ─────────────────────────────────────────────────────────── -->
<div
  class="hlx-message hlx-message--{role}"
  bind:this={messageEl}
>
  {#if timestamp}
    <span class="hlx-timestamp">{new Date(timestamp).toLocaleTimeString()}</span>
  {/if}

  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div
    class="hlx-content"
    role="log"
    aria-live="polite"
  >
    <!-- eslint-disable-next-line svelte/no-at-html-tags -->
    {@html renderedHtml}
  </div>
</div>

<!-- ── Styles ────────────────────────────────────────────────────────────── -->
<style>
  /* ── Message wrapper ──────────────────────────────────────────────────── */
  .hlx-message {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 0;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 13px;
    line-height: 1.6;
    color: var(--color-text, #e2e8f0);
  }

  .hlx-message--agent  { border-left: 2px solid var(--accent); padding-left: 10px; }
  .hlx-message--user   { border-left: 2px solid var(--color-muted,  #64748b); padding-left: 10px; }
  .hlx-message--system { border-left: 2px solid var(--color-warn,   #f59e0b); padding-left: 10px; opacity: 0.75; }

  .hlx-timestamp {
    font-size: 10px;
    color: var(--color-muted, #64748b);
    letter-spacing: 0.04em;
  }

  /* ── Prose ────────────────────────────────────────────────────────────── */
  .hlx-content :global(p)      { margin: 4px 0; }
  .hlx-content :global(ul),
  .hlx-content :global(ol)     { margin: 4px 0; padding-left: 20px; }
  .hlx-content :global(li)     { margin: 2px 0; }
  .hlx-content :global(strong) { color: var(--accent); }
  .hlx-content :global(em)     { color: var(--color-text-dim, #94a3b8); }
  .hlx-content :global(a)      { color: var(--color-link, #7dd3fc); text-decoration: underline; }

  /* Inline code */
  .hlx-content :global(:not(pre) > code) {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 12px;
    color: var(--color-inline-code, #f472b6);
    font-family: inherit;
  }

  /* Blockquote */
  .hlx-content :global(blockquote) {
    margin: 6px 0;
    padding: 4px 12px;
    border-left: 3px solid var(--accent);
    background: rgba(56, 189, 248, 0.06);
    color: var(--color-text-dim, #94a3b8);
    font-style: italic;
  }

  /* Tables */
  .hlx-content :global(table) {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 12px;
  }
  .hlx-content :global(th),
  .hlx-content :global(td) {
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 5px 10px;
    text-align: left;
  }
  .hlx-content :global(th) {
    background: rgba(255, 255, 255, 0.05);
    color: var(--accent);
    font-weight: 600;
  }

  /* ── Code block wrapper ────────────────────────────────────────────────── */
  .hlx-content :global(.hlx-code-wrapper) {
    margin: 10px 0;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: #282c34; /* atom-one-dark base */
  }

  /* Header bar (lang badge + copy button) */
  .hlx-content :global(.hlx-code-header) {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 10px;
    background: rgba(0, 0, 0, 0.25);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    min-height: 28px;
  }

  .hlx-content :global(.hlx-lang-badge) {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    opacity: 0.8;
  }

  /* Copy button */
  .hlx-content :global(.hlx-copy-btn) {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: var(--color-text-dim, #94a3b8);
    font-size: 11px;
    font-family: inherit;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
    line-height: 1;
  }
  .hlx-content :global(.hlx-copy-btn:hover) {
    background: rgba(56, 189, 248, 0.12);
    border-color: rgba(56, 189, 248, 0.3);
    color: var(--accent);
  }
  .hlx-content :global(.hlx-copy-btn:active) {
    transform: scale(0.96);
  }

  /* Highlighted <pre> block */
  .hlx-content :global(.hlx-pre) {
    margin: 0;
    padding: 14px 16px;
    overflow-x: auto;
    font-size: 12.5px;
    line-height: 1.65;
    background: transparent;

    /* Custom scrollbar to match dark UI */
    scrollbar-width: thin;
    scrollbar-color: rgba(255,255,255,0.15) transparent;
  }
  .hlx-content :global(.hlx-pre::-webkit-scrollbar) { height: 5px; }
  .hlx-content :global(.hlx-pre::-webkit-scrollbar-track) { background: transparent; }
  .hlx-content :global(.hlx-pre::-webkit-scrollbar-thumb) {
    background: rgba(255,255,255,0.15);
    border-radius: 3px;
  }

  .hlx-content :global(.hlx-pre code) {
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    background: none;
    border: none;
    padding: 0;
    color: inherit;
  }
</style>
