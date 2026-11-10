import { render, screen } from "@testing-library/svelte";
import { describe, expect, it } from "vitest";
import "../../i18n";
import type { PlanChecks as PlanChecksData, WorldModelCheck } from "../stores/session";
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
  predicted: { disk_usage_fraction: 0.72, process_count: 351 },
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
    expect(group?.textContent).toContain(
      "World model: No machine risk predicted. Disk 72% to 72%, processes 350 to 351.",
    );
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
    expect(group?.textContent).toContain("Photos measured at about 3.2 GB, D:\\ 50% to 55%, processes 350 to 351.");
    expect(group?.textContent).not.toContain("Disk 72%");
    expect(group?.textContent).not.toContain("Gnomon");
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
