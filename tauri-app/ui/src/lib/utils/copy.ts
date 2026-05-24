import { writeText } from "@tauri-apps/plugin-clipboard-manager";
import type { Message } from "../stores/session";
export function getCopyText(msg: Message): string {
  if (msg.type === "plan") {
    const parts: string[] = [];
    if (msg.plan?.explanation) {
      parts.push(msg.plan.explanation);
    }
    if (msg.plan?.actions?.length) {
      parts.push(
        msg.plan.actions
          .map(
            (a: any) =>
              `• ${a.action_type}: ${a.target || ""}`
          )
          .join("\n")
      );
    }
    return parts.join("\n\n");
  }
  if (msg.type === "result") {
    const parts: string[] = [];
    if (msg.text) {
      parts.push(msg.text);
    }
    if (msg.actionResults?.length) {
      parts.push(
        msg.actionResults
          .map(
            (r: any) =>
              r.output || r.error || ""
          )
          .filter(Boolean)
          .join("\n")
      );
    }
    if (msg.verification) {
      parts.push(
        `Verification: ${
          msg.verification.passed
            ? "passed"
            : "failed"
        }`
      );
    }
    return parts.join("\n\n");
  }
  return msg.text || "";
}
export async function copyMessage(
  msg: Message
): Promise<boolean> {
  const text = getCopyText(msg).trim();
  if (!text) return false;
  try {
    await writeText(text);
    return true;
  } catch {
    return false;
  }
}