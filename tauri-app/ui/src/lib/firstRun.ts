// First-run install: what the person can choose and how the setup's state is shown (see docs/FIRST_RUN.md).
// The desktop app runs the steps (src-tauri/src/setup.rs); this file is pure so every state is tested.
// Sizes come from data, never from this file's text: firstRunSizes.json (refreshed from the first-run CI job) for
// the install groups, and the daemon's speech_models.json for speech models.
import sizeData from "./firstRunSizes.json";
import speechManifest from "../../../../daemon/pilot/system/speech_models.json";

export type GroupId = "platform" | "voice" | "voicegpu" | "screen" | "browser" | "local" | "more";

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

/**
 * Example names for the setup wizard's protected packages, as this system's package manager spells them: winget ids
 * on Windows, Homebrew names on macOS, the distribution's names on Linux. Only a placeholder: the field starts empty,
 * so nothing is protected that the person did not name.
 */
export function protectedPackageExample(platform: string): string {
  const examples: Record<SystemKey, string> = {
    windows: "Mozilla.Firefox, Git.Git",
    macos: "firefox, git",
    linux: "firefox, nautilus, gnome-shell",
  };
  return examples[systemKey(platform)];
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

/** The groups this computer can install. ``gpu``: the NVIDIA card the setup probe found ("" when none). */
export function catalog(platform: string, gpu = ""): Group[] {
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
  groups.push({
    id: "voice",
    label: "Voice",
    gives: "Talk to Heliox and hear it answer: local speech recognition and local voices.",
    downloadMb: sizeMb("voice", platform),
    source: "PyPI (faster-whisper, Kokoro, Pocket TTS, PyTorch)",
    note: "Speech models are offered below, each with its size; Whisper base comes with voice.",
  });
  if (gpu && !mac) {
    // Only with a card, never preselected: it is large, and the voice works on the processor without it.
    groups.push({
      id: "voicegpu",
      label: "Voice on the graphics card",
      gives: `Speech recognition and the local voice run on your ${gpu}, so Heliox answers sooner. Needs Voice.`,
      downloadMb: sizeMb("voicegpu", platform),
      source: linux
        ? "PyPI (NVIDIA cuBLAS and cuDNN, already part of Voice's PyTorch on Linux)"
        : "PyPI (NVIDIA cuBLAS and cuDNN) and download.pytorch.org (PyTorch with CUDA)",
    });
  }
  groups.push(
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
  /** The NVIDIA graphics card the setup probe found ("" or absent when none). */
  gpu?: string;
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

export type Screen = "hidden" | "choose" | "installing" | "failed" | "done" | "update" | "planning" | "repair";

/** Which screen to show for a setup state. "installed" means there was nothing to set up. */
export function screenFor(state: SetupState | null): Screen {
  if (!state) return "hidden";
  switch (state.phase) {
    case "needs_choice":
      return "choose";
    case "needs_update":
      return "update";
    case "needs_repair":
      return "repair";
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
  const ids = new Set<string>(["platform", "voice", "voicegpu", "screen", "browser", "local", "more"]);
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

// ── Heliox's own voice: picked by the computer and the language, from the daemon's table in speech_models.json ──

export type VoiceEngine = "kokoro_tts" | "pocket_tts";
export type VoiceMachine = "nvidia_card" | "apple_silicon" | "processor";

export interface LocalVoice {
  engine: VoiceEngine;
  title: string;
  default_voice: string;
  model: string;
  download_bytes: number;
  size_note: string;
  licence: string;
  source: string;
  url: string;
  attribution: string;
  languages: string[];
  voice_terms?: Record<string, VoiceTerms>;
}

/** A built-in voice's source and terms (speech_models.json, voice_terms). */
export interface VoiceTerms {
  source: string;
  licence: string;
  attribution: string;
  use: "allowed" | "never" | "unknown";
  listened: boolean;
}

interface VoiceRule {
  machine: VoiceMachine;
  languages: string[];
  engine: VoiceEngine;
  why: string;
}

const voiceData = speechManifest as unknown as { voices: LocalVoice[]; voice_rules: { rows: VoiceRule[] } };

/** The local voices Heliox can speak with, as the daemon lists them (size, licence, source, attribution). */
export function localVoices(): LocalVoice[] {
  return voiceData.voices;
}

/**
 * The kind of computer the voice table reads, as far as the first screen can tell before anything is installed:
 * a Mac is taken as Apple Silicon (PyTorch publishes no wheels for Intel Macs since 2.3); an NVIDIA card counts on
 * Linux, where Voice's PyTorch is the CUDA build, and on Windows only with "Voice on the graphics card".
 */
export function voiceMachine(platform: string, gpu: string, chosen: GroupId[]): VoiceMachine {
  if (platform.includes("Mac")) return "apple_silicon";
  if (!gpu) return "processor";
  return platform.includes("Linux") || chosen.includes("voicegpu") ? "nvidia_card" : "processor";
}

/** "en-GB" -> "en": the voice language as the table names it. */
export function voiceLanguage(locale: string): string {
  return (locale || "en").split(/[-_]/)[0].toLowerCase() || "en";
}

export interface VoicePick {
  voice: LocalVoice;
  why: string;
}

/** The first row of the table that matches; Kokoro when none does (the daemon's voice_default.pick). */
export function voicePick(machine: VoiceMachine, language: string): VoicePick {
  const rule = voiceData.voice_rules.rows.find(
    (row) => row.machine === machine && (row.languages.includes("*") || row.languages.includes(language)),
  );
  const engine = rule?.engine ?? "kokoro_tts";
  const voice = localVoices().find((v) => v.engine === engine) ?? localVoices()[0];
  return { voice, why: rule?.why ?? "" };
}

/** The voices the person can pick between: the table's pick first, then any other that speaks their language. */
export function voiceOptions(machine: VoiceMachine, language: string): LocalVoice[] {
  const picked = voicePick(machine, language).voice;
  return [picked, ...localVoices().filter((v) => v.engine !== picked.engine && v.languages.includes(language))];
}

/** Where a voice runs on this kind of computer, in a few words. */
export function voiceWhere(voice: LocalVoice, machine: VoiceMachine): string {
  if (voice.engine === "kokoro_tts" && machine === "nvidia_card") return "on the graphics card";
  if (voice.engine === "kokoro_tts" && machine === "apple_silicon") return "on this Mac";
  return voice.engine === "pocket_tts" ? "on the processor, streaming" : "on the processor";
}

/** A Pocket voice's terms; undefined for a name the list does not know. */
export function pocketVoiceTerms(name: string): VoiceTerms | undefined {
  return localVoices().find((v) => v.engine === "pocket_tts")?.voice_terms?.[name];
}

/**
 * The Pocket voices Settings offers: those whose terms allow any use and that were listened to and read back. A voice
 * the person already speaks with stays in the list (``current``), so a choice made before is never broken.
 */
export function pocketVoicesOffered(current = ""): string[] {
  const terms = localVoices().find((v) => v.engine === "pocket_tts")?.voice_terms ?? {};
  const offered = Object.keys(terms).filter((name) => terms[name].use === "allowed" && terms[name].listened);
  return current && !offered.includes(current) ? [...offered, current] : offered;
}

/** The credit line for a Pocket voice: its own attribution, CC0 said plainly, or why Heliox does not offer it. */
export function pocketVoiceCredit(name: string): string {
  const terms = pocketVoiceTerms(name);
  if (!terms || terms.use !== "allowed")
    return `Voice ${name}: its terms do not allow every use or are not established, so Heliox does not offer it; it stays because it was chosen here.`;
  if (terms.attribution) return `Voice ${name}: ${terms.attribution}`;
  return `Voice ${name}: ${licenceName(terms.licence)}, from ${terms.source}.`;
}

/** "CC-BY-4.0" and "CC0-1.0" as people write them: "CC BY 4.0", "CC0 1.0". */
export function licenceName(licence: string): string {
  return licence.startsWith("CC") ? licence.replace(/-/g, " ") : licence;
}

export const VOICE_KEY = "heliox_first_run_voice";

/** A voice the person picked on the first screen instead of the table's, kept until the daemon is up. */
export function pendingVoice(storage: Pick<Storage, "getItem"> | null): { engine: string; voice: string } | null {
  try {
    const raw = storage?.getItem(VOICE_KEY);
    const kept = raw ? JSON.parse(raw) : null;
    return kept && typeof kept.engine === "string" && typeof kept.voice === "string" ? kept : null;
  } catch {
    return null;
  }
}

export function savePendingVoice(
  storage: Pick<Storage, "setItem" | "removeItem"> | null,
  voice: { engine: string; voice: string } | null,
): void {
  try {
    if (voice) storage?.setItem(VOICE_KEY, JSON.stringify(voice));
    else storage?.removeItem(VOICE_KEY);
  } catch {
    /* storage unavailable: the table's pick stands */
  }
}

/**
 * Once the daemon is up: the voice the person picked over the table's. Saved as a voice they chose, so nothing
 * switches it later. True when the daemon took it.
 */
export async function applyPendingVoice(call: Call, voice: { engine: string; voice: string }): Promise<boolean> {
  try {
    const answer = await call("update_config", {
      section: "voice",
      values: { tts_engine: voice.engine, tts_voice: voice.voice },
    });
    return answer?.status === "ok";
  } catch {
    return false;
  }
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
