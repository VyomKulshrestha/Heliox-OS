import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const gestureControl = readFileSync(
  join(root, "src", "lib", "components", "GestureControl.svelte"),
  "utf8",
);
const tauriConfig = readFileSync(
  join(root, "..", "src-tauri", "tauri.conf.json"),
  "utf8",
);
const viteConfigPath = join(root, "vite.config.ts");
const viteConfig = existsSync(viteConfigPath) ? readFileSync(viteConfigPath, "utf8") : "";

assert(!gestureControl.includes("cdn.jsdelivr.net"));
assert(!gestureControl.includes("https://"));
assert(!tauriConfig.includes("cdn.jsdelivr.net"));

// The @mediapipe/tasks-vision backend (HandLandmarker, real-metric-scale
// worldLandmarks — see GESTURES.md's "3D World-Model Layer" section) must
// stay self-hosted the same way @mediapipe/hands already is: WASM loader
// files and the hand_landmarker.task model are served locally via a Vite
// plugin (mediapipeTasksVisionAssets), never fetched from a live CDN at
// runtime. The .task model itself is vendored into vendor/mediapipe/ (see
// GESTURES.md for provenance/checksum) rather than downloaded on demand.
assert(!viteConfig.includes("cdn.jsdelivr.net"));
assert(!viteConfig.includes("storage.googleapis.com"));
