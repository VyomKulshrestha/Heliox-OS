// First-run install: what the person can choose and how the setup's state is shown (see docs/FIRST_RUN.md).
// The desktop app runs the steps (src-tauri/src/setup.rs); this file is pure so every state is tested.

export type GroupId = "platform" | "voice" | "screen" | "browser" | "local" | "more";

export interface Group {
  id: GroupId;
  label: string;
  gives: string;
  /** Approximate download, in MB; the CI first-run job measures the core and the platform readers. */
  downloadMb: number;
  source: string;
  note?: string;
}

export const CORE_DOWNLOAD_MB = 110;

export function catalog(platform: string): Group[] {
  const groups: Group[] = [];
  const mac = platform.includes("Mac");
  const linux = platform.includes("Linux");
  if (mac || linux) {
    groups.push({
      id: "platform",
      label: mac ? "macOS readers" : "Linux readers",
      gives: mac
        ? "Lets Live read the field you type in, name the app in front, read the screen and move windows."
        : "Lets Live read the field you type in through the desktop's accessibility service.",
      downloadMb: mac ? 12 : 1,
      source: mac ? "PyPI (pyobjc)" : "PyPI (jeepney)",
    });
  }
  groups.push(
    {
      id: "voice",
      label: "Voice",
      gives: "Talk to Heliox and hear it answer: local speech recognition and local voices.",
      downloadMb: 900,
      source: "PyPI (faster-whisper, Kokoro, Pocket TTS, PyTorch)",
      note: "Speech models are offered separately, with their size, when you first turn voice on.",
    },
    {
      id: "screen",
      label: "Screen reading extras",
      gives: "Reads text and finds things on screen where the system's own text recognition is not enough.",
      downloadMb: 450,
      source: "PyPI (EasyOCR, OpenCV, MediaPipe)",
    },
    {
      id: "browser",
      label: "Browser automation",
      gives: "Lets Heliox open and work in web pages for you, in its own Chromium.",
      downloadMb: 200,
      source: "PyPI (Playwright) and Playwright's Chromium download",
    },
    {
      id: "local",
      label: "Built-in local model runner",
      gives: "Runs GGUF models inside Heliox. Not needed if you use Ollama or a cloud provider.",
      downloadMb: 60,
      source: "PyPI (llama-cpp-python)",
      note: "Builds llama.cpp on this computer: needs a C and C++ compiler and takes several minutes.",
    },
    {
      id: "more",
      label: "Everything else",
      gives: "Documents, network discovery, remote machines, plugin sandbox, typing supervision and memory search.",
      downloadMb: 700,
      source: "PyPI",
    },
  );
  return groups;
}

/** Preselected: the platform readers (Live needs them) and voice when the computer has a microphone. */
export function defaultChoice(groups: Group[], hasMicrophone: boolean): GroupId[] {
  return groups.filter((g) => g.id === "platform" || (g.id === "voice" && hasMicrophone)).map((g) => g.id);
}

export function formatMb(mb: number): string {
  return mb >= 1000 ? `about ${(mb / 1000).toFixed(1)} GB` : `about ${Math.max(1, Math.round(mb))} MB`;
}

export interface PythonPlan {
  kind: string;
  version: string;
  path: string;
  download_bytes: number;
  source: string;
}

export interface SetupStep {
  id: string;
  status: string;
  message: string;
  hint: string;
  detail: string;
}

export interface SetupState {
  phase: string;
  python: PythonPlan;
  steps: SetupStep[];
  log_tail: string[];
}

/** One sentence about Python for the first screen. */
export function pythonSentence(plan: PythonPlan | undefined): string {
  if (!plan || !plan.kind) return "Checking this computer for Python…";
  if (plan.kind === "system") return `Heliox will use Python ${plan.version} already on this computer (${plan.path}).`;
  if (plan.kind === "download") {
    const mb = (plan.download_bytes || 0) / 1_000_000 + 30;
    return (
      `This computer has no Python 3.11 or later, so Heliox will download its own private copy (${formatMb(mb)}: ` +
      "uv from github.com/astral-sh/uv, checksum-checked, then Python from python-build-standalone). " +
      "It stays in Heliox's folder and does not change your system."
    );
  }
  return "No Python 3.11 or later was found, and Heliox has no private Python for this kind of computer. Install Python 3.11 or later, then open Heliox again.";
}

export type Screen = "hidden" | "choose" | "installing" | "failed" | "done";

/** Which screen to show for a setup state. "installed" means there was nothing to set up. */
export function screenFor(state: SetupState | null): Screen {
  if (!state) return "hidden";
  switch (state.phase) {
    case "needs_choice":
      return "choose";
    case "running":
      return "installing";
    case "failed":
      return "failed";
    case "done":
      return state.steps.some((s) => s.status === "failed") ? "done" : "hidden";
    default:
      return "hidden";
  }
}

const STEP_LABELS: Record<string, string> = {
  uv: "Download uv",
  python: "Python",
  core: "Heliox core",
  start: "Start Heliox",
};

export function stepLabel(id: string, groups: Group[]): string {
  return STEP_LABELS[id] ?? groups.find((g) => g.id === id)?.label ?? id;
}

/** Optional groups that failed, for "Retry" (the core and Python are retried as part of setup itself). */
export function failedGroups(state: SetupState): GroupId[] {
  const ids = new Set<string>(["platform", "voice", "screen", "browser", "local", "more"]);
  return state.steps.filter((s) => s.status === "failed" && ids.has(s.id)).map((s) => s.id as GroupId);
}

export function totalMb(groups: Group[], chosen: GroupId[], plan: PythonPlan | undefined): number {
  const python = plan?.kind === "download" ? (plan.download_bytes || 0) / 1_000_000 + 30 : 0;
  return (
    CORE_DOWNLOAD_MB + python + groups.filter((g) => chosen.includes(g.id)).reduce((sum, g) => sum + g.downloadMb, 0)
  );
}
