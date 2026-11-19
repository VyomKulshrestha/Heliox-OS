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
    blink_fps?: number;
    blink_every?: [number, number];
    listening?: string;
    thinking?: string;
    rect?: [number, number, number, number];
  };
  mouth: { shapes: Partial<Record<Shape, string>>; rect?: [number, number, number, number] };
  states?: { listening?: FaceState; thinking?: FaceState; speaking?: FaceState };
}

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
    if (eyes.rect !== undefined && !isRect(eyes.rect)) problems.push("eyes.rect: [x, y, width, height]");
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
  const shapes = manifest.mouth.shapes;
  for (const name of [shape, ...STAND_INS[shape], "E" as Shape, "sil" as Shape]) {
    const file = shapes[name];
    if (file) return file;
  }
  return shapes.sil as string;
}

export interface PackEntry {
  id: string;
  name: string;
  credit: string;
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
      entries.push({ id, name: manifest.name.trim(), credit: manifest.credit.trim() });
    } catch {
      // A pack that cannot be read is not offered.
    }
  }
  return entries;
}
