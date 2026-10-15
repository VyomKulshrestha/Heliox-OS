// Static checks on GestureControl.svelte for the selection study (spec sections 17.2 F2, 17.4 and 22.6):
// no gaze_event RPC while a study session is active, every discrete gesture dispatch passes the re-arm gate,
// and the study pointer edges disarm gestures and reopen the study pinch.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const source = readFileSync(join(process.cwd(), "src", "lib", "components", "GestureControl.svelte"), "utf8");

/** Index of the matching closing bracket for the opening bracket at `open`. */
function matching(text, open, opener, closer) {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    if (text[i] === opener) depth += 1;
    else if (text[i] === closer) {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  throw new Error(`Unbalanced ${opener}${closer} from index ${open}`);
}

/** The body of `function name(...) {...}` as [start, end) offsets of the text between its braces. */
function functionBody(name) {
  const declaration = new RegExp(`function ${name}\\(`).exec(source);
  assert.ok(declaration, `GestureControl.svelte defines ${name}()`);
  const paramsOpen = declaration.index + declaration[0].length - 1;
  const paramsClose = matching(source, paramsOpen, "(", ")");
  const bodyOpen = source.indexOf("{", paramsClose);
  const bodyClose = matching(source, bodyOpen, "{", "}");
  return { start: bodyOpen + 1, end: bodyClose, text: source.slice(bodyOpen + 1, bodyClose) };
}

/** Offsets of every call of `callee` that is not its own declaration. */
function callSites(callee) {
  const escaped = callee.replace(/[.$]/g, "\\$&");
  return [...source.matchAll(new RegExp(`(?<![\\w.$])${escaped}\\(`, "g"))]
    .map((match) => match.index)
    .filter((index) => !source.slice(Math.max(0, index - 9), index).endsWith("function "));
}

// 1. The single sendGazeEvent call is guarded by gazeEventAllowed($studyActive).
const gazeCalls = callSites("sendGazeEvent");
assert.equal(gazeCalls.length, 1, "exactly one sendGazeEvent( call site");
const beforeGazeCall = source.slice(source.lastIndexOf("\n", gazeCalls[0]) + 1, gazeCalls[0]);
assert.match(
  beforeGazeCall,
  /^\s*if \(gazeEventAllowed\(\$studyActive\)\) void $/,
  "the sendGazeEvent call is guarded by gazeEventAllowed($studyActive)",
);
assert.match(source, /import \{ studyActive \} from "\.\.\/stores\/coadaptation";/);
assert.match(source, /gazeEventAllowed,[\s\S]*?\} from "\.\.\/gesture\/studyPointerTracker";/);

// 2. Every discrete dispatch site is reached only through the discrete arm's allows() check.
const dispatch = functionBody("dispatchRecognizedGesture");
assert.match(
  dispatch.text,
  /^\s*\/\/[^\n]*\n(?:\s*\/\/[^\n]*\n)*\s*if \(!discreteArm\.allows\(gestureStartedAt\)\) \{[\s\S]*?return;\s*\}/,
  "dispatchRecognizedGesture starts with the discrete arm's allows(gestureStartedAt) check",
);
for (const callee of [
  "executeGestureAction",
  "onGesture",
  "startBoundWorkflow",
  "dispatchWorkflowControl",
  "airHandoff.handleGesture",
]) {
  const sites = callSites(callee);
  assert.ok(sites.length > 0, `${callee}( is called`);
  for (const index of sites) {
    assert.ok(
      index > dispatch.start && index < dispatch.end,
      `${callee}( at offset ${index} is inside dispatchRecognizedGesture, behind its allows() check`,
    );
  }
}
const dispatchCalls = callSites("dispatchRecognizedGesture");
assert.ok(dispatchCalls.length > 0, "recognised gestures are dispatched");
for (const index of dispatchCalls) {
  assert.ok(
    source.startsWith("dispatchRecognizedGesture(gesture, candidateStartedAt,", index),
    "each dispatch passes the time the gesture's recognition began",
  );
}

// 3. The code that observes $studyPointer.requested disarms on the falling edge and reopens the pinch on the
// rising edge.
const sync = functionBody("syncStudyPointerRequested");
const rising = /if \(requested\) \{([\s\S]*?)\} else \{([\s\S]*)$/.exec(sync.text);
assert.ok(rising, "syncStudyPointerRequested branches on the new value of requested");
assert.match(rising[1], /studyPinch\.reset\(\)/, "requested turning true resets the study pinch hysteresis");
assert.match(rising[2], /discreteArm\.disarm\(/, "requested turning false disarms every discrete gesture");
assert.match(source, /const studyPointerRequested = \$derived\(\$studyPointer\.requested\);/);
assert.match(source, /syncStudyPointerRequested\(requested\)/, "an effect applies changes of requested");
assert.match(
  source,
  /syncStudyPointerRequested\(\$studyPointer\.requested\);/,
  "every camera frame applies a change of requested before it can dispatch",
);

// 4. The OS cursor is never moved or clicked while the study pointer is requested.
for (const name of ["moveGestureCursor", "clickGestureCursor"]) {
  assert.match(
    functionBody(name).text,
    /^\s*(?:\/\/[^\n]*\n\s*)*if \(\$studyPointer\.requested\) return;/,
    `${name} returns at once while the study pointer is requested`,
  );
}

console.log("gesture-gaze-study static checks passed");
