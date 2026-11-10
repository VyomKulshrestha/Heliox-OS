import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import "../../i18n";
import type { CriticCheck, PlanAction, PlanChecks as PlanChecksData, WorldModelCheck } from "../stores/session";
import ConfirmDialog from "./ConfirmDialog.svelte";
import PlanChecks from "./PlanChecks.svelte";

const quietWorldModel: WorldModelCheck = {
  enabled: true,
  unavailable: false,
  raised: false,
  score: 0,
  reasons: [],
  sources: ["rule"],
  observed: { disk_usage_fraction: 0.72, process_count: 350 },
  predicted: { disk_usage_fraction: null, process_count: 351 },
  measured_sizes: [],
};

function row(checks: PlanChecksData | null) {
  const view = render(PlanChecks, { checks });
  return view.container.querySelector(".plan-checks");
}

describe("PlanChecks", () => {
  it("says quietly what both checks did for a routine plan", () => {
    const group = row({
      worldModel: quietWorldModel,
      gnomon: { state: "reviewed", escalate: false, reasons: [], scores: { in_scope: 0.97 }, review_seconds: 0.42 },
    });

    expect(screen.getByRole("group", { name: "Checked before running" })).toBe(group);
    expect(group?.textContent).toContain("World model: No machine risk predicted. Processes 350 to 351.");
    expect(group?.textContent).toContain("Gnomon: Matches what you asked (97% in scope). Took 0.4 s.");
    expect(group?.textContent).toContain("It can ask for approval, never grant it.");
    expect(group?.querySelectorAll(".raised")).toHaveLength(0);
  });

  it("shows a measured copy instead of the per-step disk value", () => {
    const group = row({
      worldModel: {
        ...quietWorldModel,
        measured_sizes: [
          {
            path_label: "Photos",
            size_text: "about 3.2 GB",
            volume: "D:\\",
            usage_before: 0.5,
            usage_after: 0.55,
          },
        ],
      },
    });
    expect(group?.textContent).toContain("Measured Photos at about 3.2 GB, D:\\ 50% to 55%, processes 350 to 351.");
    expect(group?.textContent).not.toContain("Disk 72%");
    expect(group?.textContent).not.toContain("Gnomon");
  });

  it("never changes the case of a measured file name", () => {
    const sizes = [
      {
        path_label: "post-final.md",
        size_text: "about 225 bytes",
        volume: "C:\\",
        usage_before: null,
        usage_after: null,
      },
      { path_label: "screenshots", size_text: "about 267 KB", volume: "C:\\", usage_before: null, usage_after: null },
    ];
    const text = row({ worldModel: { ...quietWorldModel, observed: null, measured_sizes: sizes } })?.textContent;
    expect(text).toContain("Measured post-final.md at about 225 bytes, measured screenshots at about 267 KB.");
    expect(text).not.toContain("Post-final.md");
  });

  it("gives the world model's own reason when it raised risk", () => {
    const group = row({
      worldModel: {
        ...quietWorldModel,
        raised: true,
        score: 0.45,
        reasons: ["predicted disk usage 96% exceeds the safe threshold"],
      },
    });
    const line = group?.querySelector(".check");
    expect(line?.classList.contains("raised")).toBe(true);
    expect(line?.textContent).toContain("Raised risk (45%): predicted disk usage 96% exceeds the safe threshold.");
  });

  it("lists Gnomon's real reasons when it asked for approval", () => {
    const group = row({
      gnomon: {
        state: "reviewed",
        escalate: true,
        reasons: ["Gnomon judged this may go beyond what you asked (8% in scope)."],
        scores: { in_scope: 0.08 },
      },
    });
    const line = group?.querySelector(".check");
    expect(line?.classList.contains("raised")).toBe(true);
    expect(line?.textContent).toContain(
      "Gnomon: Asked for your approval. Gnomon judged this may go beyond what you asked (8% in scope).",
    );
    expect(line?.textContent).not.toContain("Matches");
  });

  it.each([
    [{ state: "loading" }, "Gnomon: Still loading, did not review this plan."],
    [{ state: "cold" }, "Gnomon: Not loaded yet, did not review this plan."],
    [
      { state: "unavailable", reason: "Needs 2.5 GB of free memory, 1.8 GB free now. Tries again on its own." },
      "Gnomon: Not loaded, did not review this plan. Needs 2.5 GB of free memory, 1.8 GB free now.",
    ],
    [{ state: "unavailable", reason: "" }, "Gnomon: Not loaded, did not review this plan. It can"],
    [{ state: "timed_out" }, "Gnomon: Timed out, did not review this plan."],
  ])("says when Gnomon did not review (%o)", (gnomon, text) => {
    const group = row({ gnomon });
    expect(group?.textContent).toContain(text);
    expect(group?.textContent).not.toContain("Matches");
    expect(group?.textContent).not.toContain("Took");
  });

  it("says the gate is off instead of showing a prediction", () => {
    const group = row({ worldModel: { enabled: false } });
    expect(group?.textContent).toContain("World model: Off in Settings, no prediction made.");
    expect(group?.textContent).not.toMatch(/Disk|processes|risk/);
  });

  it("says the world model could not assess the plan", () => {
    const group = row({
      worldModel: { enabled: true, unavailable: true, raised: true, score: 1, reasons: [], measured_sizes: [] },
    });
    expect(group?.textContent).toContain("World model: Could not assess this plan, so it needs your approval.");
  });

  it("leaves out values the daemon did not send", () => {
    const group = row({
      worldModel: { ...quietWorldModel, observed: null, predicted: { disk_usage_fraction: 0.7, process_count: null } },
    });
    expect(group?.textContent).toContain("World model: No machine risk predicted.");
    expect(group?.textContent).not.toMatch(/Disk|processes/);
  });

  it("states no disk figures when the daemon measured nothing", () => {
    const group = row({
      worldModel: { ...quietWorldModel, predicted: { disk_usage_fraction: null, process_count: null } },
    });
    expect(group?.textContent).toContain("World model: No machine risk predicted.");
    expect(group?.textContent).not.toMatch(/Disk|processes|72%/);
  });

  it("renders nothing without a payload", () => {
    expect(row(null)).toBeNull();
    expect(row({})).toBeNull();
  });
});

describe("ConfirmDialog checks", () => {
  const actions = [
    {
      index: 0,
      action_type: "file_delete",
      target: "C:\\Temp\\a.txt",
      parameters: {},
      requires_root: false,
      destructive: true,
    },
  ];
  const base = { actions, onconfirm: () => {}, ondeny: () => {} };

  it("shows the row when nothing above it covers the checks", () => {
    const { container } = render(ConfirmDialog, {
      ...base,
      checks: { worldModel: quietWorldModel, gnomon: { state: "timed_out" } },
    });
    const group = container.querySelector(".plan-checks");
    expect(group?.textContent).toContain("World model: No machine risk predicted.");
    expect(group?.textContent).toContain("Gnomon: Timed out, did not review this plan.");
  });

  it("does not repeat the world-model interruption or Gnomon's reasons", () => {
    const { container } = render(ConfirmDialog, {
      ...base,
      reason: "World model paused this plan. Gnomon judged this may go beyond what you asked (8% in scope).",
      riskAssessment: { world_model_score: 0.8, prediction_sources: ["rule"] },
      checks: {
        worldModel: { ...quietWorldModel, raised: true, score: 0.8 },
        gnomon: { state: "reviewed", escalate: true, reasons: ["Gnomon judged this may go beyond what you asked."] },
      },
    });
    expect(container.querySelector(".plan-checks")).toBeNull();
  });
});

// The real run's review of "move the final post and screenshots, delete drafts and build, zip the rest".
const warnReview: CriticCheck = {
  verdict: "WARN",
  ran: true,
  downgraded: false,
  findings: [
    {
      step: 3,
      severity: "high",
      text: "Step 3 deletes launch/drafts, which holds draft-final-FINAL.md. It can be undone from Heliox for 7 days.",
    },
    { step: 4, severity: "medium", text: "launch/build holds old-release.zip (293 KB), maybe a shipped release." },
    { step: null, severity: "medium", text: "The zip may be incomplete if a copy fails." },
    { step: 5, severity: "low", text: "Step 5 uses a script for a simple zip." },
  ],
  recommendation: "Confirm draft-final-FINAL.md is not your real final post.",
};

function reviewText(critic: CriticCheck | undefined) {
  return row(critic ? { critic } : {})?.textContent ?? "";
}

describe("PlanChecks safety review", () => {
  it("stays to one quiet line for a clean approve", () => {
    const group = row({ critic: { verdict: "APPROVE", ran: true, findings: [], recommendation: "Safe to run." } });
    expect(group?.textContent?.trim()).toBe("Safety review: Nothing to flag.");
    expect(group?.querySelector(".raised")).toBeNull();
    expect(group?.querySelector("ul")).toBeNull();
  });

  it("does not list process notes on an approve", () => {
    const text = reviewText({
      verdict: "APPROVE",
      ran: true,
      findings: [{ step: 2, severity: "low", text: "Uses a script for a zip." }],
      recommendation: "Safe to run.",
    });
    expect(text.trim()).toBe("Safety review: Nothing to flag.");
  });

  it("warns with the recommendation and the first concerns, tied to their steps", () => {
    const group = row({ critic: warnReview });
    const line = group?.querySelector(".check");
    expect(line?.classList.contains("raised")).toBe(true);
    expect(line?.textContent).toContain(
      "Safety review: Check before approving. Confirm draft-final-FINAL.md is not your real final post.",
    );
    const items = screen.getAllByRole("listitem").map((item) => item.textContent?.trim());
    expect(items).toEqual([
      "Step 3 deletes launch/drafts, which holds draft-final-FINAL.md. It can be undone from Heliox for 7 days.",
      "Step 4: launch/build holds old-release.zip (293 KB), maybe a shipped release.",
    ]);
    expect(group?.textContent).not.toContain("simple zip");
  });

  it("keeps a long list short until asked", async () => {
    row({ critic: warnReview });
    const more = screen.getByRole("button", { name: "Show 2 more" });
    expect(more.getAttribute("aria-expanded")).toBe("false");
    await fireEvent.click(more);
    expect(more.getAttribute("aria-expanded")).toBe("true");
    expect(more.textContent?.trim()).toBe("Show fewer");
    const items = screen.getAllByRole("listitem").map((item) => item.textContent?.replace(/\s+/g, " ").trim());
    expect(items.slice(2)).toEqual([
      "The zip may be incomplete if a copy fails.",
      "Minor: Step 5 uses a script for a simple zip.",
    ]);
    expect(document.getElementById(more.getAttribute("aria-controls") ?? "")?.tagName).toBe("UL");
  });

  it("says when the review stopped the plan", () => {
    const text = reviewText({
      verdict: "BLOCK",
      ran: true,
      findings: [{ step: 1, severity: "high", text: "Deletes C:/Windows." }],
      recommendation: "Do not run this.",
    });
    expect(text).toContain("Safety review: Stopped this plan. Do not run this.");
    expect(text).toContain("Step 1: Deletes C:/Windows.");
  });

  it("leaves out a recommendation the daemon wrote for a lowered block", () => {
    const text = reviewText({ ...warnReview, downgraded: true, recommendation: "Show the warning." });
    expect(text).toContain("Safety review: Check before approving.");
    expect(text).not.toContain("Show the warning");
  });

  it.each([
    [
      { verdict: "SKIPPED", ran: false, skipped: "low_risk_heuristic" },
      "Safety review: Did not run, this plan scored low risk.",
    ],
    [
      { verdict: "SKIPPED", ran: false, skipped: "no_destructive_authority" },
      "Safety review: Did not run, no step in this plan is destructive.",
    ],
    [{ verdict: "WARN", ran: false }, "Safety review: Could not run, so check each step yourself."],
  ])("says when the review did not run (%o)", (critic, text) => {
    const shown = reviewText({ findings: [], recommendation: "", ...critic });
    expect(shown.trim()).toBe(text);
  });

  it("says nothing when no review was sent", () => {
    expect(reviewText(undefined)).toBe("");
    expect(row({ worldModel: quietWorldModel })?.textContent).not.toContain("Safety review");
  });
});

describe("ConfirmDialog safety review", () => {
  const undoable = { kind: "undoable" as const, reason: "Heliox keeps what it needs to put this back." };
  const confirmActions: PlanAction[] = [
    {
      index: 2,
      action_type: "file_delete",
      target: "C:\\launch\\drafts",
      parameters: {},
      requires_root: false,
      destructive: true,
      undo: undoable,
    },
    {
      index: 3,
      action_type: "file_delete",
      target: "C:\\launch\\build",
      parameters: {},
      requires_root: false,
      destructive: true,
      undo: undoable,
    },
    {
      index: 4,
      action_type: "shell_script",
      target: "C:\\launch-release.zip",
      parameters: {},
      requires_root: false,
      destructive: false,
      undo: { kind: "not_undoable", reason: "Heliox cannot reverse this kind of change automatically." },
    },
  ];

  it("shows each concern next to the step it is about and describes its checkbox", () => {
    const { container } = render(ConfirmDialog, {
      actions: confirmActions,
      onconfirm: () => {},
      ondeny: () => {},
      checks: { critic: warnReview },
    });
    const rows = Array.from(container.querySelectorAll(".confirm-list > li"));
    expect(rows[0].querySelector(".step-review")?.textContent).toContain(
      "Safety review: Step 3 deletes launch/drafts, which holds draft-final-FINAL.md. It can be undone from Heliox for 7 days.",
    );
    expect(rows[1].querySelector(".step-review")?.textContent).toContain("old-release.zip");
    // Process notes stay out of the step list.
    expect(rows[2].querySelector(".step-review")).toBeNull();

    const checkbox = rows[0].querySelector("input[type=checkbox]");
    const described = document.getElementById(checkbox?.getAttribute("aria-describedby") ?? "");
    expect(described?.textContent).toContain("draft-final-FINAL.md");
    expect(rows[2].querySelector("input")?.hasAttribute("aria-describedby")).toBe(false);

    const group = container.querySelector(".plan-checks");
    expect(group?.textContent).toContain(
      "Safety review: Check before approving. Confirm draft-final-FINAL.md is not your real final post.",
    );
    expect(group?.textContent).not.toContain("old-release.zip");
  });

  it("keeps the actions in view: one line per check, details closed until asked", async () => {
    const { container } = render(ConfirmDialog, {
      actions: confirmActions,
      onconfirm: () => {},
      ondeny: () => {},
      checks: {
        critic: warnReview,
        worldModel: quietWorldModel,
        gnomon: {
          state: "reviewed",
          escalate: false,
          scope_reviewed: false,
          notes: [SCOPE_NOTE, UNDO_NOTE],
          scores: { severe_harm: 0.02 },
        },
      },
    });
    const group = container.querySelector(".plan-checks");
    expect(group?.querySelectorAll("li")).toHaveLength(0);
    expect(group?.querySelectorAll("p.check, div.check")).toHaveLength(3);
    expect(group?.textContent).not.toContain("The zip may be incomplete");
    expect(group?.textContent).not.toContain(UNDO_NOTE);
    expect(group?.textContent).not.toContain("Show 2 more");
    // Gnomon says once, in the daemon's words, why scope was not reviewed.
    expect(group?.textContent?.split("Scope not reviewed").length).toBe(2);
    expect(container.querySelectorAll(".confirm-list > li")).toHaveLength(3);

    const details = screen.getByRole("button", { name: "Show details" });
    expect(details.getAttribute("aria-expanded")).toBe("false");
    await fireEvent.click(details);
    expect(details.getAttribute("aria-expanded")).toBe("true");
    expect(document.getElementById(details.getAttribute("aria-controls") ?? "")).not.toBeNull();
    const text = group?.textContent?.replace(/\s+/g, " ") ?? "";
    expect(text).toContain("The zip may be incomplete if a copy fails.");
    expect(text).toContain("Minor: Step 5 uses a script for a simple zip.");
    expect(text).toContain(UNDO_NOTE);
    expect(text.split("Scope not reviewed").length).toBe(2);
    expect(text).not.toContain("old-release.zip");
  });

  it("puts the undo fact the critic wrote beside the step's undo badge", () => {
    const { container } = render(ConfirmDialog, {
      actions: confirmActions,
      onconfirm: () => {},
      ondeny: () => {},
      checks: { critic: warnReview },
    });
    const first = container.querySelector(".confirm-list > li");
    expect(first?.querySelector(".undo-tag.can-undo")).not.toBeNull();
    expect(first?.querySelector(".step-review")?.textContent).toContain("can be undone from Heliox for 7 days");
    expect(container.textContent).not.toMatch(/permanent/i);
  });

  it("adds nothing to the steps when the review approved cleanly", () => {
    const { container } = render(ConfirmDialog, {
      actions: confirmActions,
      onconfirm: () => {},
      ondeny: () => {},
      checks: { critic: { verdict: "APPROVE", ran: true, findings: [], recommendation: "" } },
    });
    expect(container.querySelector(".step-review")).toBeNull();
    expect(container.querySelector(".plan-checks")?.textContent?.trim()).toBe("Safety review: Nothing to flag.");
  });
});

const SCOPE_NOTE =
  "Scope not reviewed: your request has several parts, and Gnomon judges scope reliably only for one-part requests.";
const UNDO_NOTE = "4 file deletes are backed up by Heliox's undo, so they were not graded as irreversible.";
const LATE_NOTE = "Gnomon ran out of time before every check, so this review is incomplete.";

describe("PlanChecks Gnomon notes", () => {
  it("keeps the one-part verdict unchanged", () => {
    const text = row({
      gnomon: {
        state: "reviewed",
        escalate: false,
        scope_reviewed: true,
        complete: true,
        notes: [],
        scores: { in_scope: 0.97, severe_harm: 0.01 },
      },
    })?.textContent;
    expect(text).toContain("Gnomon: Matches what you asked (97% in scope).");
    expect(text).toContain("It can ask for approval, never grant it.");
  });

  it("says scope was not reviewed, why, and what was checked", () => {
    const group = row({
      gnomon: {
        state: "reviewed",
        escalate: false,
        scope_reviewed: false,
        complete: true,
        notes: [SCOPE_NOTE],
        scores: { severe_harm: 0.02, permitted_by_user: 0.9 },
        review_seconds: 1.2,
      },
    });
    const text = group?.textContent ?? "";
    expect(text).toContain(
      `Gnomon: Checked harm if wrong and instructions found in tool or screen text, nothing raised. ${SCOPE_NOTE} Took 1.2 s.`,
    );
    expect(text.split("Scope not reviewed").length).toBe(2);
    expect(text).toContain("It can ask for approval, never grant it.");
    expect(text).not.toMatch(/Matches what you asked|Reviewed, nothing raised/);
  });

  it("shows the undo note", () => {
    const text = row({
      gnomon: {
        state: "reviewed",
        escalate: false,
        scope_reviewed: false,
        notes: [SCOPE_NOTE, UNDO_NOTE],
        scores: { severe_harm: 0.1 },
      },
    })?.textContent;
    expect(text).toContain(UNDO_NOTE);
    expect(text).toContain(`Gnomon: Checked harm if wrong, nothing raised. ${SCOPE_NOTE}`);
    expect(text?.split("Scope not reviewed").length).toBe(2);
  });

  it("says a cut-short review stopped early and what it found so far", async () => {
    const group = row({
      gnomon: {
        state: "reviewed",
        escalate: false,
        scope_reviewed: false,
        complete: false,
        notes: [SCOPE_NOTE, UNDO_NOTE, LATE_NOTE],
        scores: { severe_harm: 0.03 },
      },
    });
    expect(group?.textContent).toContain(
      `Gnomon: Cut short. So far it checked harm if wrong, nothing raised. ${SCOPE_NOTE}`,
    );
    expect(group?.textContent).toContain(UNDO_NOTE);
    expect(group?.textContent).toContain(LATE_NOTE);
    expect(group?.textContent).not.toMatch(/Matches what you asked/);
  });

  it("says when a cut-short review finished no check", () => {
    const text = row({
      gnomon: { state: "reviewed", escalate: false, complete: false, notes: [LATE_NOTE], scores: {} },
    })?.textContent;
    expect(text).toContain("Gnomon: Cut short before any check finished.");
  });

  it("keeps the notes in the dialog when the reasons are shown above", () => {
    const { container } = render(ConfirmDialog, {
      actions: [],
      onconfirm: () => {},
      ondeny: () => {},
      reason: "Gnomon graded the harm if wrong as: severe.",
      checks: {
        gnomon: { state: "reviewed", escalate: true, scope_reviewed: false, notes: [SCOPE_NOTE], reasons: ["x"] },
      },
    });
    const group = container.querySelector(".plan-checks");
    expect(group?.textContent).toContain(`Gnomon: Asked for your approval, reasons above. ${SCOPE_NOTE}`);
    expect(group?.textContent?.split("Scope not reviewed").length).toBe(2);
  });
});
