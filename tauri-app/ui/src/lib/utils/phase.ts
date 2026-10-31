/**
 * Turns the daemon's phase strings ("recalling memory", "critic review", "re-planning (attempt 2)", the
 * interaction runtime's own sentences) into the short status Heliox shows while it works, plus the stage of the
 * four-step track: understand, plan, act, check.
 */

export const RUN_STAGES = ["Understand", "Plan", "Act", "Check"] as const;
export type RunStage = 0 | 1 | 2 | 3;

export interface PhaseView {
  label: string;
  stage: RunStage;
  /** Heliox is paused until the person answers (an approval). */
  waitingOnYou: boolean;
}

type Rule = [pattern: RegExp, label: string, stage: RunStage, waitingOnYou?: boolean];

const RULES: Rule[] = [
  [/^(receiving input|understanding)$/, "Reading your request", 0],
  [/^recalling memory$/, "Recalling what's relevant", 0],
  [/^routing agents$/, "Choosing the right specialist", 0],
  [/^running queued task$/, "Starting your queued task", 0],
  [/^planning voice command$/, "Planning your voice request", 1],
  [/^planning$/, "Planning the steps", 1],
  [/^re-planning \(attempt (\d+)\)$/, "Re-planning, attempt $1", 1],
  [/^companion reviewing plan$/, "Double-checking the plan", 1],
  [/^companion revising plan \((\d+)\/(\d+)\)$/, "Revising the plan, $1 of $2", 1],
  [/^revising plan \(revision (\d+)\)$/, "Revising the plan, revision $1", 1],
  [/^critic review$/, "Reviewing the plan for risks", 1],
  [/^(awaiting[ _]approval|awaiting visible approval)$/, "Waiting for your approval", 1, true],
  [
    /^(revising after your correction|applying your correction|correcting|sending live correction)$/,
    "Applying your correction",
    1,
  ],
  [/^(executing|acting)$/, "Working on it", 2],
  [/^retrying failed actions \(attempt (\d+)\)$/, "Retrying failed steps, attempt $1", 2],
  [/^retrying\b/, "Retrying after a failed attempt", 2],
  [/^answering without interrupting$/, "Answering without interrupting", 2],
  [/^stopping current step and revising$/, "Stopping this step to revise", 2],
  [/^(stopping|aborted)$/, "Stopping", 2],
  [/^(recovering interrupted task|resuming)$/, "Resuming your task", 2],
  [/^connection interrupted\b/, "Connection dropped, resuming safely", 2],
  [/^detecting conflicts$/, "Checking for merge conflicts", 2],
  [/^resume needs attention$/, "Resume needs your attention", 2, true],
  [/^verifying$/, "Checking the result", 3],
];

/** Unknown phases are placed by their leading verb ("Planning the safest useful action" is planning, not acting). */
function stageFromWords(text: string): RunStage {
  if (/\bcorrection\b/.test(text)) return 1;
  const verb = text.split(/\s+/)[0] ?? "";
  if (/^(verif|check)/.test(verb)) return 3;
  if (/^(execut|run|act|apply|open|click|typ|writ|delet|mov)/.test(verb)) return 2;
  if (/^(plan|review|revis|re-plan|waiting|approv)/.test(verb)) return 1;
  return 0;
}

function sentenceCase(text: string): string {
  const cleaned = text
    .replace(/\s*[—–]\s*/g, ", ")
    .replaceAll("_", " ")
    .trim();
  return cleaned ? cleaned[0].toUpperCase() + cleaned.slice(1) : cleaned;
}

export function describePhase(raw: string | null | undefined): PhaseView {
  const phase = String(raw ?? "").trim();
  if (!phase) return { label: "Thinking", stage: 0, waitingOnYou: false };
  const key = phase.toLowerCase();
  for (const [pattern, label, stage, waitingOnYou] of RULES) {
    const match = key.match(pattern);
    if (match) {
      return {
        label: label.replace(/\$(\d)/g, (_, n: string) => match[Number(n)] ?? ""),
        stage,
        waitingOnYou: Boolean(waitingOnYou),
      };
    }
  }
  return {
    label: sentenceCase(phase),
    stage: stageFromWords(key),
    waitingOnYou: /\bwaiting for your\b/.test(key),
  };
}

/** "0:07", "1:42", "12:05": elapsed seconds as minutes and seconds. */
export function formatElapsed(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

const ACRONYMS = new Set(["ocr", "url", "ui", "api", "pdf", "dns", "cpu", "gpu", "ram", "ssh", "http", "json", "csv"]);

/** "open_application" -> "Open application", "screen_ocr" -> "Screen OCR". */
export function actionTitle(actionType: string): string {
  const words = actionType.replaceAll("_", " ").toLowerCase().split(/\s+/).filter(Boolean);
  return sentenceCase(words.map((word) => (ACRONYMS.has(word) ? word.toUpperCase() : word)).join(" "));
}
