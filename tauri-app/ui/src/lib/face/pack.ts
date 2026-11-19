/**
 * Face packs: a manifest and its images in public/faces/<id>/, listed in public/faces/packs.json. The format is
 * written down for pack makers in docs/FACE_PACKS.md. Nothing here loads an image: the face component does that,
 * and only while a face is chosen. Reading manifests (small JSON files) is all Settings does.
 */

import { SHAPES, type Shape } from "./mouthPlayer";

export const OFF = "off";
const ID = /^[a-z0-9][a-z0-9-]{0,39}$/;
const FILE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.(png|webp|jpg|jpeg|svg|avif)$/;

export interface FaceState {
  tilt?: number;
  lift?: number;
  scale?: number;
  /** The mouth while the face rests in this state (an image drawn in mouth.rect); missing means the `sil` image. */
  mouth?: string;
}

export interface FaceManifest {
  format: 1;
  id: string;
  name: string;
  credit: string;
  kind?: "drawing" | "photo";
  size: [number, number];
  tint?: { idle?: string; eyes?: string; mouth?: string };
  breathe?: { amount: number; seconds: number };
  idle: { frames: string[]; fps?: number; loop?: "pingpong" | "forward"; mouth_offsets?: [number, number][] };
  eyes?: {
    open?: string;
    blink?: string[];
    /** A small shift of the eyes, from slight to further, for glances between phrases. */
    glance?: string[];
    blink_fps?: number;
    blink_every?: [number, number];
    listening?: string;
    thinking?: string;
    rect?: [number, number, number, number];
  };
  mouth: { shapes: Partial<Record<Shape, string>>; rect?: [number, number, number, number] };
  states?: { idle?: FaceState; listening?: FaceState; thinking?: FaceState; speaking?: FaceState };
  /** How much larger than the card the picture is drawn (0.04 is 4 %). A photo gets enough by default. */
  overscan?: number;
  /**
   * In-between pictures of a shape, from nearly closed to nearly full, made with the same crop and alpha as the
   * shape's own picture (the last step, the full shape, is that picture; sil is step zero). Optional.
   */
  mouth_steps?: Partial<Record<Shape, string[]>>;
  /**
   * The voice that suits the face: a Kokoro voice id, and a plain hint ("feminine" or "masculine") for when that
   * voice cannot speak on the computer. Optional; Settings switches to it with the face only while the person has
   * never chosen a voice.
   */
  voice?: FaceVoice;
}

export interface FaceVoice {
  kokoro?: string;
  hint?: "feminine" | "masculine";
}

const VOICE_HINTS = ["feminine", "masculine"];
const KOKORO_VOICE = /^[a-z][fm]_[a-z]{1,20}$/;

export const STATE_NAMES = ["idle", "listening", "thinking", "speaking"] as const;

export function validPackId(id: unknown): id is string {
  return typeof id === "string" && ID.test(id);
}

export function facesUrl(path: string): string {
  const base = (import.meta.env?.BASE_URL as string | undefined) ?? "/";
  return `${base.endsWith("/") ? base : base + "/"}faces/${path}`;
}

export function packUrl(id: string, file = ""): string {
  return facesUrl(`${id}/${file}`);
}

function isNumberPair(value: unknown): value is [number, number] {
  return Array.isArray(value) && value.length === 2 && value.every((n) => typeof n === "number" && Number.isFinite(n));
}

function isRect(value: unknown): value is [number, number, number, number] {
  return Array.isArray(value) && value.length === 4 && value.every((n) => typeof n === "number" && Number.isFinite(n));
}

function isFile(value: unknown): value is string {
  return typeof value === "string" && FILE.test(value);
}

function isToken(value: unknown): boolean {
  return value === undefined || (typeof value === "string" && /^--[a-z0-9-]{1,40}$/.test(value));
}

/** The reasons a manifest cannot be played; [] when it can. `id` is the folder it was read from. */
export function manifestProblems(raw: unknown, id: string): string[] {
  const problems: string[] = [];
  if (!raw || typeof raw !== "object") return ["the manifest is not a JSON object"];
  const m = raw as Record<string, unknown>;
  if (m.format !== 1) problems.push("format must be 1");
  if (m.id !== id) problems.push(`id must be "${id}", the folder's name`);
  if (typeof m.name !== "string" || !m.name.trim() || m.name.length > 40) problems.push("name: 1 to 40 characters");
  if (typeof m.credit !== "string" || !m.credit.trim() || m.credit.length > 200)
    problems.push("credit: 1 to 200 characters");
  if (!isNumberPair(m.size) || m.size[0] < 16 || m.size[1] < 16) problems.push("size: [width, height]");
  const idle = m.idle as Record<string, unknown> | undefined;
  if (!idle || !Array.isArray(idle.frames) || !idle.frames.length || !idle.frames.every(isFile))
    problems.push("idle.frames: one or more image files");
  if (idle && idle.mouth_offsets !== undefined) {
    const offsets = idle.mouth_offsets;
    if (!Array.isArray(offsets) || !offsets.every(isNumberPair) || offsets.length !== (idle.frames as unknown[]).length)
      problems.push("idle.mouth_offsets: one [dx, dy] per idle frame");
  }
  const mouth = m.mouth as Record<string, unknown> | undefined;
  const shapes = (mouth?.shapes ?? {}) as Record<string, unknown>;
  if (!mouth || typeof shapes !== "object") problems.push("mouth.shapes is missing");
  else {
    for (const [name, file] of Object.entries(shapes)) {
      if (!(SHAPES as readonly string[]).includes(name)) problems.push(`mouth.shapes: unknown shape ${name}`);
      else if (!isFile(file)) problems.push(`mouth.shapes.${name}: not an image file name`);
    }
    if (!isFile(shapes.sil)) problems.push("mouth.shapes.sil (the closed mouth) is required");
    if (mouth.rect !== undefined && !isRect(mouth.rect)) problems.push("mouth.rect: [x, y, width, height]");
  }
  const eyes = m.eyes as Record<string, unknown> | undefined;
  if (eyes) {
    for (const key of ["open", "listening", "thinking"]) {
      if (eyes[key] !== undefined && !isFile(eyes[key])) problems.push(`eyes.${key}: not an image file name`);
    }
    if (eyes.blink !== undefined && (!Array.isArray(eyes.blink) || !eyes.blink.every(isFile)))
      problems.push("eyes.blink: image files");
    if (eyes.glance !== undefined && (!Array.isArray(eyes.glance) || !eyes.glance.length || !eyes.glance.every(isFile)))
      problems.push("eyes.glance: image files");
    if (eyes.rect !== undefined && !isRect(eyes.rect)) problems.push("eyes.rect: [x, y, width, height]");
  }
  const states = m.states as Record<string, unknown> | undefined;
  if (states !== undefined) {
    if (!states || typeof states !== "object") problems.push("states: an object");
    else
      for (const [name, value] of Object.entries(states)) {
        if (!(STATE_NAMES as readonly string[]).includes(name)) continue; // a later format's state: not drawn
        const state = (value ?? {}) as Record<string, unknown>;
        for (const key of ["tilt", "lift", "scale"]) {
          const n = state[key];
          if (n !== undefined && (typeof n !== "number" || !Number.isFinite(n)))
            problems.push(`states.${name}.${key}: a number`);
        }
        if (typeof state.scale === "number" && (state.scale < 0.5 || state.scale > 2))
          problems.push(`states.${name}.scale: 0.5 to 2`);
        if (state.mouth !== undefined && !isFile(state.mouth))
          problems.push(`states.${name}.mouth: not an image file name`);
      }
  }
  if (m.overscan !== undefined && (typeof m.overscan !== "number" || !(m.overscan >= 0 && m.overscan <= 0.25)))
    problems.push("overscan: 0 to 0.25");
  if (m.mouth_steps !== undefined) {
    const steps = m.mouth_steps as Record<string, unknown>;
    if (!steps || typeof steps !== "object" || Array.isArray(steps)) problems.push("mouth_steps: an object");
    else
      for (const [name, files] of Object.entries(steps)) {
        if (!(SHAPES as readonly string[]).includes(name)) problems.push(`mouth_steps: unknown shape ${name}`);
        else if (!Array.isArray(files) || !files.length || files.length > 8 || !files.every(isFile))
          problems.push(`mouth_steps.${name}: 1 to 8 image files`);
        else if (!(shapes as Record<string, unknown>)[name])
          problems.push(`mouth_steps.${name}: the shape has no picture`);
      }
  }
  if (m.voice !== undefined) {
    const voice = m.voice as Record<string, unknown>;
    if (!voice || typeof voice !== "object" || Array.isArray(voice)) problems.push("voice: an object");
    else {
      if (voice.kokoro !== undefined && (typeof voice.kokoro !== "string" || !KOKORO_VOICE.test(voice.kokoro)))
        problems.push("voice.kokoro: a Kokoro voice id, such as af_heart");
      if (voice.hint !== undefined && !VOICE_HINTS.includes(voice.hint as string))
        problems.push("voice.hint: feminine or masculine");
    }
  }
  const tint = (m.tint ?? {}) as Record<string, unknown>;
  if (!isToken(tint.idle) || !isToken(tint.eyes) || !isToken(tint.mouth))
    problems.push("tint: names of the window's colour tokens, such as --text-primary");
  return problems;
}

/** Which image stands in for a shape the pack does not have: the nearest in how the mouth looks, then E, then sil. */
const STAND_INS: Record<Shape, Shape[]> = {
  sil: [],
  PP: ["sil"],
  FF: ["PP", "DD"],
  TH: ["DD", "E"],
  DD: ["nn", "E"],
  kk: ["E", "aa"],
  CH: ["SS", "U", "O"],
  SS: ["I", "DD", "E"],
  nn: ["DD", "E"],
  RR: ["U", "O", "E"],
  aa: ["E"],
  E: ["aa", "DD"],
  I: ["E", "SS"],
  O: ["U", "aa"],
  U: ["O", "RR"],
};

export function imageForShape(manifest: FaceManifest, shape: Shape): string {
  return manifest.mouth.shapes[shapeShown(manifest, shape)] as string;
}

/** The shape whose picture stands for `shape`: its own, the nearest one the pack has, then E, then sil. */
export function shapeShown(manifest: FaceManifest, shape: Shape): Shape {
  const shapes = manifest.mouth.shapes;
  for (const name of [shape, ...STAND_INS[shape], "E" as Shape, "sil" as Shape]) {
    if (shapes[name]) return name;
  }
  return "sil";
}

/** The pictures of a shape from closed to full: the in-between steps the pack has (mouth_steps), then the shape. */
export function stepsOf(manifest: FaceManifest, shape: Shape): string[] {
  const shown = shapeShown(manifest, shape);
  const full = manifest.mouth.shapes[shown] as string;
  if (shown === "sil") return [];
  return [...(manifest.mouth_steps?.[shown] ?? []), full];
}

/** The shapes a pack has a picture of. */
export function packShapes(manifest: FaceManifest): Shape[] {
  return SHAPES.filter((shape) => Boolean(manifest.mouth.shapes[shape]));
}

export interface PackEntry {
  id: string;
  name: string;
  credit: string;
  /** The voice the pack names, if any. */
  voice?: FaceVoice;
}

type Fetcher = (url: string) => Promise<{ ok: boolean; json(): Promise<unknown> }>;

export async function readManifest(id: string, fetcher: Fetcher = fetch): Promise<FaceManifest> {
  if (!validPackId(id)) throw new Error("Not a face pack name.");
  const response = await fetcher(packUrl(id, "manifest.json"));
  if (!response.ok) throw new Error("This face is not installed.");
  const raw = await response.json();
  const problems = manifestProblems(raw, id);
  if (problems.length) throw new Error(`This face pack cannot be shown: ${problems[0]}.`);
  return raw as FaceManifest;
}

/** The installed packs (public/faces/packs.json) with the names and credits their manifests give. */
export async function installedPacks(fetcher: Fetcher = fetch): Promise<PackEntry[]> {
  let ids: unknown[] = [];
  try {
    const response = await fetcher(facesUrl("packs.json"));
    if (response.ok) {
      const raw = (await response.json()) as { packs?: unknown };
      ids = Array.isArray(raw?.packs) ? raw.packs : [];
    }
  } catch {
    ids = [];
  }
  const entries: PackEntry[] = [];
  for (const id of ids.filter(validPackId)) {
    try {
      const manifest = await readManifest(id, fetcher);
      const entry: PackEntry = { id, name: manifest.name.trim(), credit: manifest.credit.trim() };
      if (manifest.voice?.kokoro) entry.voice = { kokoro: manifest.voice.kokoro, hint: manifest.voice.hint };
      entries.push(entry);
    } catch {
      // A pack that cannot be read is not offered.
    }
  }
  return entries;
}
