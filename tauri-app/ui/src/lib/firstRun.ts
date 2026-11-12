// First-run install: what the person can choose and how the setup's state is shown (see docs/FIRST_RUN.md).
// The desktop app runs the steps (src-tauri/src/setup.rs); this file is pure so every state is tested.
// Sizes come from data, never from this file's text: firstRunSizes.json (refreshed from the first-run CI job) for
// the install groups, and the daemon's speech_models.json for speech models.
import sizeData from "./firstRunSizes.json";
import speechManifest from "../../../../daemon/pilot/system/speech_models.json";

export type GroupId = "platform" | "voice" | "screen" | "browser" | "local" | "more";

export interface Group {
  id: GroupId;
  label: string;
  gives: string;
  /** Download, in MB, from firstRunSizes.json for this system. */
  downloadMb: number;
  source: string;
  note?: string;
}

type SystemKey = "windows" | "macos" | "linux";
type SizeEntry = { mb: number; disk_mb?: number; measured?: string };
const SIZES = (sizeData as { sizes: Record<string, Partial<Record<SystemKey | "all", SizeEntry>>> }).sizes;

export function systemKey(platform: string): SystemKey {
  if (platform.includes("Mac")) return "macos";
  if (platform.includes("Linux")) return "linux";
  return "windows";
}

function entryFor(id: string, platform: string): SizeEntry | undefined {
  return SIZES[id]?.[systemKey(platform)] ?? SIZES[id]?.all;
}

/** The download size in MB for an id ("core", "private_python" or a group) on this system; 0 when unknown. */
export function sizeMb(id: string, platform: string): number {
  return entryFor(id, platform)?.mb ?? 0;
}

/** What it takes on disk once installed, in MB, when the data file has it (measured by CI); 0 otherwise. */
export function diskMb(id: string, platform: string): number {
  return entryFor(id, platform)?.disk_mb ?? 0;
}

/** "about 110 MB to download, about 530 MB on disk", or only the download when the disk size is not known. */
export function sizePhrase(id: string, platform: string): string {
  const disk = diskMb(id, platform);
  const download = `${formatMb(sizeMb(id, platform))} to download`;
  return disk ? `${download}, ${formatMb(disk)} on disk` : download;
}

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
      downloadMb: sizeMb("platform", platform),
      source: mac ? "PyPI (pyobjc)" : "PyPI (jeepney)",
    });
  }
  groups.push(
    {
      id: "voice",
      label: "Voice",
      gives: "Talk to Heliox and hear it answer: local speech recognition and local voices.",
      downloadMb: sizeMb("voice", platform),
      source: "PyPI (faster-whisper, Kokoro, Pocket TTS, PyTorch)",
      note: "Speech models are offered below, each with its size; Whisper base comes with voice.",
    },
    {
      id: "screen",
      label: "Screen reading extras",
      gives: "Reads text and finds things on screen where the system's own text recognition is not enough.",
      downloadMb: sizeMb("screen", platform),
      source: "PyPI (EasyOCR, OpenCV, MediaPipe)",
    },
    {
      id: "browser",
      label: "Browser automation",
      gives: "Lets Heliox open and work in web pages for you, in its own Chromium.",
      downloadMb: sizeMb("browser", platform),
      source: "PyPI (Playwright) and Playwright's Chromium download",
    },
    {
      id: "local",
      label: "Built-in local model runner",
      gives: "Runs GGUF models inside Heliox. Not needed if you use Ollama or a cloud provider.",
      downloadMb: sizeMb("local", platform),
      source: "PyPI (llama-cpp-python)",
      note: "Builds llama.cpp on this computer: needs a C and C++ compiler and takes several minutes.",
    },
    {
      id: "more",
      label: "Everything else",
      gives: "Documents, network discovery, remote machines, plugin sandbox, typing supervision and memory search.",
      downloadMb: sizeMb("more", platform),
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

export interface PackageChange {
  name: string;
  from: string;
  to: string;
  bytes: number;
}

/** What an update of an older installed daemon will do; nothing changes until the person applies it. */
export interface UpdatePlan {
  installed: string;
  target: string;
  groups: string[];
  planned: boolean;
  changes: PackageChange[];
  total_bytes: number;
}

export interface SetupState {
  phase: string;
  python: PythonPlan;
  steps: SetupStep[];
  log_tail: string[];
  update?: UpdatePlan;
}

/** One sentence about Python for the first screen. */
export function pythonSentence(plan: PythonPlan | undefined, platform = ""): string {
  if (!plan || !plan.kind) return "Checking this computer for Python…";
  if (plan.kind === "system") return `Heliox will use Python ${plan.version} already on this computer (${plan.path}).`;
  if (plan.kind === "download") {
    const mb = privatePythonMb(plan, platform);
    return (
      `This computer has no Python Heliox can use, so Heliox will download its own private copy (${formatMb(mb)}` +
      `${diskMb("private_python", platform) ? `, ${formatMb(diskMb("private_python", platform))} on disk` : ""}: ` +
      "uv from github.com/astral-sh/uv, checksum-checked, then Python from python-build-standalone). " +
      "It stays in Heliox's folder and does not change your system."
    );
  }
  return "No Python Heliox can use was found, and Heliox has no private Python for this kind of computer. Install Python 3.12, then open Heliox again.";
}

export type Screen = "hidden" | "choose" | "installing" | "failed" | "done" | "update" | "planning";

/** Which screen to show for a setup state. "installed" means there was nothing to set up. */
export function screenFor(state: SetupState | null): Screen {
  if (!state) return "hidden";
  switch (state.phase) {
    case "needs_choice":
      return "choose";
    case "needs_update":
      return "update";
    case "planning":
      return "planning";
    case "updating":
      return "installing";
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

/** uv's exact size (from the probe) plus the private Python's size from the data file. */
export function privatePythonMb(plan: PythonPlan | undefined, platform: string): number {
  if (plan?.kind !== "download") return 0;
  return (plan.download_bytes || 0) / 1_000_000 + sizeMb("private_python", platform);
}

export function totalMb(
  groups: Group[],
  chosen: GroupId[],
  plan: PythonPlan | undefined,
  platform = "",
  models: SpeechModel[] = [],
  chosenModels: string[] = [],
): number {
  const groupMb = groups.filter((g) => chosen.includes(g.id)).reduce((sum, g) => sum + g.downloadMb, 0);
  const modelMb = chosen.includes("voice")
    ? models.filter((m) => chosenModels.includes(m.id)).reduce((sum, m) => sum + m.size_bytes / 1_000_000, 0)
    : 0;
  return sizeMb("core", platform) + privatePythonMb(plan, platform) + groupMb + modelMb;
}

// ── Speech models: the daemon downloads them once it is running (speech_model_download) ──

export interface SpeechModel {
  id: string;
  title: string;
  purpose: string;
  language: string;
  size_bytes: number;
  licence: string;
  source: string;
  part_of_voice?: boolean;
}

/** The speech models offered under Voice: everything in the daemon's manifest that does not come with voice. */
export function speechModels(): SpeechModel[] {
  const models = (speechManifest as { models: SpeechModel[] }).models;
  return models.filter((m) => !m.part_of_voice);
}

export function languageNote(model: SpeechModel): string {
  return model.language === "en" ? "English only" : "Many languages";
}

export const PENDING_MODELS_KEY = "heliox_first_run_speech_models";
/** Set when setup ran here, so the first conversation is offered once the daemon is up. */
export const GREET_KEY = "heliox_first_run_greet";

/** Speech models the person ticked and the daemon has not been asked for yet (kept across a restart). */
export function pendingModels(storage: Pick<Storage, "getItem"> | null): string[] {
  try {
    const raw = storage?.getItem(PENDING_MODELS_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list.filter((id): id is string => typeof id === "string") : [];
  } catch {
    return [];
  }
}

export function savePendingModels(storage: Pick<Storage, "setItem" | "removeItem"> | null, ids: string[]): void {
  try {
    if (ids.length) storage?.setItem(PENDING_MODELS_KEY, JSON.stringify(ids));
    else storage?.removeItem(PENDING_MODELS_KEY);
  } catch {
    /* private window or blocked storage: the choice lasts until the app closes */
  }
}

type Call = (method: string, params: Record<string, unknown>) => Promise<{ status?: string } | undefined>;

/**
 * Once the daemon is up after setup: ask it for each ticked speech model, then start the getting-to-know
 * conversation if the profile is empty. Returns the models still to ask for ("busy" means one download is
 * running; those are asked again on the next call).
 */
export async function afterSetup(call: Call, models: string[], greet = true): Promise<string[]> {
  const left: string[] = [];
  for (const id of models) {
    try {
      const answer = await call("speech_model_download", { id });
      if (answer?.status === "busy" || answer?.status === "error") left.push(id);
    } catch {
      left.push(id);
    }
  }
  if (!greet) return left;
  try {
    const status = (await call("getting_to_know", { action: "status" })) as
      { profile_empty?: boolean; active?: boolean } | undefined;
    if (status?.profile_empty && !status.active) await call("getting_to_know", { action: "start" });
  } catch {
    /* the daemon offers it later as a card */
  }
  return left;
}

/** "0.15.0 to 0.16.0, 12 packages change, about 18 MB to download"; the size only when the index gave it. */
export function updateSummary(plan: UpdatePlan | undefined): string {
  if (!plan) return "";
  const versions = `${plan.installed || "an older version"} to ${plan.target}`;
  if (!plan.planned) return versions;
  const count = plan.changes.length === 1 ? "1 package changes" : `${plan.changes.length} packages change`;
  const size = plan.total_bytes > 0 ? `, ${formatMb(plan.total_bytes / 1_000_000)} to download` : "";
  return `${versions}: ${count}${size}`;
}

/** The kept groups in words for the update screen. */
export function keptGroups(plan: UpdatePlan | undefined, groups: Group[]): string {
  const names = (plan?.groups ?? []).map((id) => groups.find((g) => g.id === id)?.label ?? id);
  return names.length
    ? `Your parts stay installed: ${names.join(", ")}.`
    : "Only the core is installed; it stays as it is.";
}
