/**
 * highlighter.ts
 * --------------
 * Thin wrapper around highlight.js that registers only the language packs
 * relevant to Heliox OS agent output, keeping the bundle lean.
 *
 * Place at: tauri-app/ui/src/lib/highlighter.ts
 */

import hljs from "highlight.js/lib/core";

// ── Language packs (add more here as needed) ──────────────────────────────
import python     from "highlight.js/lib/languages/python";
import javascript from "highlight.js/lib/languages/javascript";
import typescript from "highlight.js/lib/languages/typescript";
import json       from "highlight.js/lib/languages/json";
import bash       from "highlight.js/lib/languages/bash";
import shell      from "highlight.js/lib/languages/shell";
import xml        from "highlight.js/lib/languages/xml";   // HTML / XML
import css        from "highlight.js/lib/languages/css";
import yaml       from "highlight.js/lib/languages/yaml";
import toml       from "highlight.js/lib/languages/ini";   // hljs uses ini for TOML
import powershell from "highlight.js/lib/languages/powershell";
import plaintext  from "highlight.js/lib/languages/plaintext";

hljs.registerLanguage("python",      python);
hljs.registerLanguage("javascript",  javascript);
hljs.registerLanguage("js",          javascript);
hljs.registerLanguage("typescript",  typescript);
hljs.registerLanguage("ts",          typescript);
hljs.registerLanguage("json",        json);
hljs.registerLanguage("bash",        bash);
hljs.registerLanguage("sh",          bash);
hljs.registerLanguage("shell",       shell);
hljs.registerLanguage("xml",         xml);
hljs.registerLanguage("html",        xml);
hljs.registerLanguage("css",         css);
hljs.registerLanguage("yaml",        yaml);
hljs.registerLanguage("yml",         yaml);
hljs.registerLanguage("toml",        toml);
hljs.registerLanguage("powershell",  powershell);
hljs.registerLanguage("ps1",         powershell);
hljs.registerLanguage("plaintext",   plaintext);
hljs.registerLanguage("text",        plaintext);

export default hljs;

/**
 * Highlight a code string.
 *
 * @param code     Raw source code string.
 * @param language Language tag from the fenced code block (may be empty/unknown).
 * @returns        Object with `value` (highlighted HTML) and `language` (detected).
 */
export function highlight(
  code: string,
  language: string
): { value: string; language: string } {
  const lang = language.toLowerCase().trim();

  // Known language → deterministic highlight
  if (lang && hljs.getLanguage(lang)) {
    const result = hljs.highlight(code, { language: lang, ignoreIllegals: true });
    return { value: result.value, language: lang };
  }

  // Unknown / missing tag → auto-detect (handles raw JSON dumps etc.)
  const result = hljs.highlightAuto(code, [
    "python", "javascript", "typescript", "json",
    "bash", "shell", "powershell", "yaml", "toml", "xml", "css",
  ]);
  return {
    value: result.value,
    language: result.language ?? "plaintext",
  };
}
