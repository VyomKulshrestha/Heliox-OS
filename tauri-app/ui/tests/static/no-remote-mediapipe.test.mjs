import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

assert(!gestureControl.includes("cdn.jsdelivr.net"));
assert(!gestureControl.includes("https://"));
assert(!tauriConfig.includes("cdn.jsdelivr.net"));
