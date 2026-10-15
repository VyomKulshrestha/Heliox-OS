import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

// Co-adaptation changes presentation only (spec sections 3 and 22.5). Its UI code must never reach the authority
// path (approvals, plan execution, task control, cursor commands, Tauri invoke) and must keep nothing in browser
// storage. The pure library modules must also stay free of Svelte, Tauri, store and API imports and of top-level DOM
// access, so they run unchanged in tests and in SDK hosts.

const root = fileURLToPath(new URL("../..", import.meta.url));
const src = join(root, "src", "lib");

function sourceFiles(dir) {
  if (!existsSync(dir)) return [];
  const out = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) out.push(...sourceFiles(path));
    else if (/\.(ts|svelte|js|mjs)$/.test(name) && !/\.test\.(ts|js|mjs)$/.test(name)) out.push(path);
  }
  return out;
}

const scanned = [
  ...sourceFiles(join(src, "coadaptation")),
  ...sourceFiles(join(src, "components", "coadaptation")),
  join(src, "stores", "coadaptation.ts"),
  join(src, "stores", "studyPointer.ts"),
].filter((path) => existsSync(path));

assert(
  scanned.some((path) => path.endsWith(join("stores", "coadaptation.ts"))),
  "the co-adaptation store is scanned",
);
assert(
  scanned.some((path) => path.endsWith(join("coadaptation", "api.ts"))),
  "the co-adaptation API module is scanned",
);

const forbidden = [
  "session.confirm",
  "sendCommand",
  ".abort(",
  "invoke(",
  "move_gesture_cursor",
  "click_gesture_cursor",
  "cursor_move",
  "cursor_click",
  'call("confirm"',
  'call("execute"',
  'call("interject"',
  'call("queue_execute"',
  "https://",
  "localStorage",
  "sessionStorage",
  "indexedDB",
];

for (const path of scanned) {
  const text = readFileSync(path, "utf8");
  for (const needle of forbidden) {
    assert(!text.includes(needle), `${relative(root, path)} must not contain ${needle}`);
  }
}

// Every co-adaptation RPC is a literal call in lib/coadaptation/api.ts.
const api = readFileSync(join(src, "coadaptation", "api.ts"), "utf8");
for (const method of [
  "status",
  "config_update",
  "consent_grant",
  "consent_revoke",
  "preference_set",
  "preference_retract",
  "learning_reset",
  "erase",
  "results",
  "export",
  "session_begin",
  "session_resume",
  "recommend",
  "record_trial",
  "report_burden",
  "session_end",
]) {
  assert(api.includes(`"co_adaptation_${method}"`), `api.ts calls co_adaptation_${method} literally`);
}
for (const path of scanned) {
  if (path.endsWith(join("coadaptation", "api.ts"))) continue;
  const text = readFileSync(path, "utf8");
  assert(!/\bcall\s*[(<]/.test(text), `${relative(root, path)} calls the daemon only through lib/coadaptation/api.ts`);
}

const pureModules = [
  "types.ts",
  "geometry.ts",
  "targetPosterior.ts",
  "decisionTable.ts",
  "presentation.ts",
  "commitRules.ts",
  "trialRecorder.ts",
  "studySchedule.ts",
  "inputAdapter.ts",
];

function isFunctionLike(node) {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node)
  );
}

/** Identifiers `window` or `document` evaluated at module load (outside every function body). */
function topLevelDomAccess(sourceFile) {
  const found = [];
  const visit = (node) => {
    if (isFunctionLike(node)) return;
    if (ts.isTypeNode(node) || ts.isInterfaceDeclaration(node) || ts.isTypeAliasDeclaration(node)) return;
    if (ts.isIdentifier(node) && (node.text === "window" || node.text === "document")) {
      const parent = node.parent;
      const isPropertyName = parent && ts.isPropertyAccessExpression(parent) && parent.name === node;
      if (!isPropertyName) found.push(node.getStart(sourceFile));
    }
    ts.forEachChild(node, visit);
  };
  visit(sourceFile);
  return found;
}

for (const name of pureModules) {
  const path = join(src, "coadaptation", name);
  assert(existsSync(path), `pure module ${name} exists`);
  const text = readFileSync(path, "utf8");
  const sourceFile = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const specifiers = [];
  const collect = (node) => {
    if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
      specifiers.push(node.moduleSpecifier.text);
    }
    if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
      const [arg] = node.arguments;
      specifiers.push(arg && ts.isStringLiteral(arg) ? arg.text : "<dynamic>");
    }
    ts.forEachChild(node, collect);
  };
  collect(sourceFile);
  for (const specifier of specifiers) {
    assert(
      !/^svelte(\/|$)/.test(specifier) &&
        !specifier.startsWith("@tauri-apps") &&
        !specifier.includes("../stores") &&
        !specifier.includes("../api") &&
        specifier !== "<dynamic>",
      `${name} must not import ${specifier}`,
    );
  }
  assert.deepEqual(topLevelDomAccess(sourceFile), [], `${name} must not touch window or document at module load`);
}
