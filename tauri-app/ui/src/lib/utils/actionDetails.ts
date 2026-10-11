/**
 * What an action will actually do, for the approval dialog: the command line, script,
 * code, recipients, tool arguments and so on, taken from the action's parameters.
 * Approving "shell_script on powershell" without seeing the script is not informed consent.
 */
import type { PlanAction } from "../stores/session";

export interface ActionDetail {
  label: string;
  value: string;
  code: boolean;
}

const MAX_CHARS = 2000;
const SECRET_KEY = /(pass(word)?|secret|token|api[_-]?key|authorization|cookie|credential)/i;
// Parameters that are settings rather than substance; shown only when nothing else is.
const NOISE_KEYS = new Set([
  "timeout",
  "timeout_seconds",
  "max_depth",
  "max_entries",
  "interval",
  "presses",
  "wait_until",
  "state",
  "exact",
  "full_page",
  "clear_first",
  "press_enter",
  "allow_private_network",
  "hash_algorithm",
  "recursive",
  "multiple",
  "mark_as_read",
  "max_emails",
  "html",
  "snapshot_id",
]);

function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined || value === "" || value === false) return true;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === "object") return Object.keys(value as object).length === 0;
  return false;
}

/** Replace secret-looking fields with dots, at any depth. */
export function maskSecrets(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(maskSecrets);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [
        key,
        SECRET_KEY.test(key) && !isEmpty(item) ? "••••••" : maskSecrets(item),
      ]),
    );
  }
  return value;
}

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(maskSecrets(value), null, 2);
}

function clip(text: string): string {
  return text.length > MAX_CHARS ? `${text.slice(0, MAX_CHARS)}\n… ${text.length - MAX_CHARS} more characters` : text;
}

export function describeAction(action: PlanAction): ActionDetail[] {
  const p = (action.parameters ?? {}) as Record<string, unknown>;
  const details: ActionDetail[] = [];
  const add = (label: string, value: unknown, code = false) => {
    if (!isEmpty(value)) details.push({ label, value: clip(asText(value)), code });
  };
  const str = (key: string) => (typeof p[key] === "string" ? (p[key] as string) : "");

  switch (action.action_type) {
    case "shell_command":
      add("Command", [str("command"), ...((p.args as string[] | undefined) ?? [])].join(" ").trim(), true);
      add("In folder", p.working_directory);
      if (p.elevated) add("Runs as", "administrator");
      break;
    case "shell_script":
      add(`Script (${str("interpreter") || "shell"})`, p.script, true);
      add("In folder", p.working_directory);
      if (p.elevated) add("Runs as", "administrator");
      break;
    case "pty_exec":
      add("Command", p.command, true);
      break;
    case "code_execute":
    case "code_generate_and_run":
      add(`Code (${str("language") || "code"})`, p.code, true);
      add("Task", p.task_description);
      break;
    case "ssh_command":
      add("Host", p.host);
      add("Command", p.command, true);
      break;
    case "ssh_script":
      add("Host", p.host);
      add("Script", p.script, true);
      break;
    case "browser_execute_js":
      add("Page", p.url);
      add("Page script", p.script, true);
      break;
    case "mcp_call_tool":
      add("Tool", [str("mcp_server"), str("mcp_tool")].filter(Boolean).join("."));
      add("Arguments", p.mcp_arguments, true);
      break;
    case "api_send_email":
    case "api_slack":
    case "api_discord":
    case "api_webhook":
      add("To", p.to || p.channel || p.webhook_url || p.url);
      add("Subject", p.subject);
      add("Message", p.message || p.body, true);
      break;
    case "email_reply":
      add("To", p.to);
      add("Subject", p.subject);
      add("Reply", p.reply_body, true);
      break;
    case "api_request":
    case "api_github":
      add("Request", `${str("method") || "GET"} ${str("url") || str("endpoint")}`.trim());
      add("Body", p.body, true);
      break;
    case "registry_write":
    case "registry_delete":
      add("Key", p.key_path);
      add("Value", p.value_name ? `${str("value_name")} = ${asText(p.value_data ?? "")}` : "");
      break;
    case "keyboard_type":
      add("Types", p.text, true);
      break;
    case "file_write":
      add("File", p.path);
      add("Content", p.content, true);
      break;
    case "file_move":
    case "file_copy":
      add("From", p.path);
      add("To", p.destination);
      break;
    case "file_delete":
      add("Deletes", p.path);
      if (p.recursive) add("Includes", "everything inside the folder");
      break;
  }

  if (details.length === 0) {
    const substance = Object.fromEntries(
      Object.entries(p).filter(([key, value]) => !NOISE_KEYS.has(key) && !isEmpty(value)),
    );
    if (Object.keys(substance).length > 0) add("Parameters", substance, true);
  }
  return details;
}
