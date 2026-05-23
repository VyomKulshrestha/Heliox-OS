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
export async function renderMarkdown(
  text: string
): Promise<string> {
  if (!text) return "";
  const raw = await marked.parse(text);
  return DOMPurify.sanitize(raw as string);
}