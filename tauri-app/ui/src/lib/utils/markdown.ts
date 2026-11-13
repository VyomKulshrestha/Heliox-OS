import { marked } from "marked";
import DOMPurify from "dompurify";
import { highlight } from "../highlighter";
const renderer = new marked.Renderer();
renderer.code = function ({ text: code, lang: language }: { text: string; lang?: string }): string {
  const lang = language || "";
  const result = highlight(code, lang);
  const langLabel = result.language !== "plaintext" ? result.language : "";
  const langBadge = langLabel ? `<span class="hlx-lang-badge">${langLabel}</span>` : "";
  return `
  <div class="hlx-code-wrapper">
    <div class="hlx-code-header">
      ${langBadge}
      <button
        class="hlx-copy-btn"
        data-code="${encodeURIComponent(code)}"
      >
        Copy
      </button>
    </div>
    <pre class="hlx-pre">
      <code class="hljs language-${result.language}">
        ${result.value}
      </code>
    </pre>
  </div>
  `;
};
marked.setOptions({
  renderer,
  gfm: true,
  breaks: true,
});
export function renderMarkdown(text: string): string {
  if (!text) return "";
  const raw = marked.parse(text);
  // A form in text the window did not write could post somewhere (the policy's form-action also stops it).
  return DOMPurify.sanitize(raw as string, { FORBID_TAGS: ["form"], FORBID_ATTR: ["action", "formaction"] });
}

/**
 * Markdown for text that is still arriving (a streamed reply). Whatever the stream has cut in half is closed or held
 * back so the growing text renders the way it will when complete, without a code block swallowing the rest of the
 * page or a half-typed `**` showing as stars and then jumping to bold:
 * - an open code fence is closed, so a half-written block already shows as code;
 * - an unclosed inline code span or bold run at the end is shown without its marker until it closes.
 */
export function closeOpenMarkdown(text: string): string {
  if (!text) return "";
  const fences = text.match(/^\s*(```|~~~)/gm) ?? [];
  if (fences.length % 2 === 1) {
    return `${text}${text.endsWith("\n") ? "" : "\n"}${fences[fences.length - 1].trim()}`;
  }
  let tail = text;
  const lastLine = tail.slice(tail.lastIndexOf("\n") + 1);
  if ((lastLine.match(/`/g) ?? []).length % 2 === 1) {
    const index = tail.lastIndexOf("`");
    tail = tail.slice(0, index) + tail.slice(index + 1);
  }
  const strongs = tail.match(/\*\*/g) ?? [];
  if (strongs.length % 2 === 1) {
    const index = tail.lastIndexOf("**");
    tail = tail.slice(0, index) + tail.slice(index + 2);
  }
  return tail;
}

export function renderStreamingMarkdown(text: string): string {
  return renderMarkdown(closeOpenMarkdown(text));
}
