// What a script running in the app's window could reach, kept small (see src-tauri/src/navigation.rs and
// src-tauri/src/file_access.rs). Fails when the content-security policy allows eval or framing again, when the
// capabilities name a remote origin or a permission the UI does not use, or when a Rust command is registered that the
// UI never calls.
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const root = process.cwd();
const tauri = join(root, "..", "src-tauri");
const config = JSON.parse(readFileSync(join(tauri, "tauri.conf.json"), "utf8"));

// The content-security policy.
const csp = config.app.security.csp;
const directives = Object.fromEntries(
  csp
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => {
      const [name, ...values] = part.split(/\s+/);
      return [name, values];
    }),
);
assert(
  !/'unsafe-eval'/.test(csp),
  "the policy must not allow eval (MediaPipe's WebAssembly needs only 'wasm-unsafe-eval')",
);
assert(!directives["script-src"].includes("'unsafe-inline'"), "no inline script");
assert.deepEqual(directives["object-src"], ["'none'"]);
assert.deepEqual(directives["frame-src"], ["'none'"]);
assert.deepEqual(directives["frame-ancestors"], ["'none'"]);
assert.deepEqual(directives["form-action"], ["'none'"]);
assert.deepEqual(directives["base-uri"], ["'self'"]);
for (const [name, values] of Object.entries(directives)) {
  for (const value of values) {
    assert(!/^https?:\/\/(?!127\.0\.0\.1|localhost)/.test(value), `${name} names a remote origin: ${value}`);
    assert(value !== "*" && value !== "https:" && value !== "http:", `${name} allows any origin: ${value}`);
  }
}
assert.equal(config.app.windows.length, 1, "one window: anything else needs its own, narrower capability");

const ui = [];
const walk = (folder) => {
  for (const name of readdirSync(folder)) {
    const path = join(folder, name);
    if (statSync(path).isDirectory()) walk(path);
    else if (/\.(ts|svelte)$/.test(name) && !/\.test\./.test(name)) ui.push(readFileSync(path, "utf8"));
  }
};
walk(join(root, "src"));
const source = ui.join("\n");

// The capabilities: no remote origin, and exactly the permissions the UI's own calls need. Each entry is a call the
// UI makes into Tauri, the IPC command it sends (read from Tauri's code: @tauri-apps/api and the plugins' guest-js,
// and tauri/src/window/scripts/drag.js for the drag region), and the permission that allows that command. A Tauri
// import or attribute the table does not know fails the test, so a new call cannot go in without its permission.
const USES = [
  // getCurrentWebview().onDragDropEvent and listen() send plugin:event|listen, and unlisten when they end.
  {
    pattern: /\bonDragDropEvent\(|\blisten(<[^>]*>)?\(/,
    permissions: ["core:event:allow-listen", "core:event:allow-unlisten"],
  },
  // data-tauri-drag-region: Tauri's drag.js sends plugin:window|start_dragging, and on a double click
  // plugin:window|internal_toggle_maximize.
  {
    pattern: /data-tauri-drag-region/,
    permissions: ["core:window:allow-start-dragging", "core:window:allow-internal-toggle-maximize"],
  },
  // The notification plugin: isPermissionGranted sends is_permission_granted (also sent by its init script on
  // macOS and Linux), requestPermission request_permission, sendNotification notify.
  { pattern: /\bisPermissionGranted\(/, permissions: ["notification:allow-is-permission-granted"] },
  { pattern: /\brequestPermission\(/, permissions: ["notification:allow-request-permission"] },
  { pattern: /\bsendNotification\(/, permissions: ["notification:allow-notify"] },
  // The clipboard plugin's writeText sends write_text; readText would need allow-read-text.
  { pattern: /\bwriteText\(/, permissions: ["clipboard-manager:allow-write-text"] },
  { pattern: /\breadText\(/, permissions: ["clipboard-manager:allow-read-text"] },
];
// Tauri imports whose calls the table covers. invoke() of the app's own commands needs no permission for the app's
// own pages (tauri/src/webview/mod.rs checks the ACL only for plugin commands, an app manifest, or a remote origin).
const KNOWN_IMPORTS = new Map([
  ["@tauri-apps/api/core", ["invoke"]],
  ["@tauri-apps/api/webview", ["getCurrentWebview"]],
  ["@tauri-apps/api/event", ["listen"]],
  ["@tauri-apps/plugin-notification", ["isPermissionGranted", "requestPermission", "sendNotification"]],
  ["@tauri-apps/plugin-clipboard-manager", ["writeText", "readText"]],
]);
for (const match of source.matchAll(/import\s*\{([^}]*)\}\s*from\s*"(@tauri-apps\/[^"]+)"/g)) {
  const names = match[1]
    .split(",")
    .map((name) => name.trim().split(/\s+as\s+/)[0])
    .filter(Boolean);
  const known = KNOWN_IMPORTS.get(match[2]);
  assert(known, `the UI imports ${match[2]}, which the permission table does not know`);
  for (const name of names)
    assert(known.includes(name), `the UI uses ${name} from ${match[2]}, not in the permission table`);
}
for (const attribute of source.matchAll(/data-tauri-[a-z-]+/g)) {
  assert.equal(attribute[0], "data-tauri-drag-region", `the permission table does not know ${attribute[0]}`);
}
const required = new Set(USES.filter((use) => use.pattern.test(source)).flatMap((use) => use.permissions));
const capabilities = readdirSync(join(tauri, "capabilities")).filter((name) => name.endsWith(".json"));
const granted = new Set();
for (const name of capabilities) {
  const capability = JSON.parse(readFileSync(join(tauri, "capabilities", name), "utf8"));
  assert(!("remote" in capability), `${name} lets a remote origin call the app`);
  assert.deepEqual(capability.windows, ["main"], `${name} windows`);
  for (const permission of capability.permissions) {
    const id = typeof permission === "string" ? permission : permission.identifier;
    assert(!id.endsWith(":default"), `${name} grants a whole default set (${id}); name what is used`);
    granted.add(id);
  }
}
for (const id of required) assert(granted.has(id), `the UI needs ${id}, which no capability grants`);
for (const id of granted) assert(required.has(id), `a capability grants ${id}, which the UI does not use`);

// Every Rust command the window can call is one the UI calls.
const main = readFileSync(join(tauri, "src", "main.rs"), "utf8");
const handler = main.slice(main.indexOf("generate_handler!["), main.indexOf("])", main.indexOf("generate_handler![")));
const commands = handler
  .slice("generate_handler![".length)
  .split(",")
  .map((item) => item.trim().split("::").pop())
  .filter(Boolean);

for (const command of commands) {
  assert(source.includes(`"${command}"`), `the window can call ${command}, which the UI never calls`);
}
assert(!commands.includes("register_allowed_path"), "only the native drop event grants a file (file_access.rs)");
console.log(`window surface: ${commands.length} commands, ${granted.size} permissions, policy without eval or framing`);
