import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import AssistChip from "./AssistChip.svelte";

afterEach(cleanup);

const CHIP_VARIANTS = [
  "act",
  "enlarge_target:1.5",
  "enlarge_target:2.0",
  "confirm_selection:second_click",
  "confirm_selection:hold_400",
  "confirm_selection:hold_700",
  "ask_clarification:2",
  "ask_clarification:3",
];

function vetoButtons() {
  return {
    why: screen.queryByRole("button", { name: "Why?" }),
    notThisTime: screen.queryByRole("button", { name: "Not this time" }),
    never: screen.queryByRole("button", { name: "Never" }),
  };
}

describe("wait_for_evidence hint", () => {
  for (const mode of ["adaptive", "randomized", "probe", "pinned", "fallback", ""]) {
    it(`is dismiss-only in mode "${mode}": no Why?, Not this time or Never`, async () => {
      const ondismiss = vi.fn();
      const onnotthistime = vi.fn();
      render(AssistChip, { variant: "wait_for_evidence", mode, ondismiss, onnotthistime, onnever: vi.fn() });

      expect(screen.getByText("Not selected — aim again")).toBeTruthy();
      const buttons = screen.getAllByRole("button");
      expect(buttons).toHaveLength(1);
      expect(vetoButtons()).toEqual({ why: null, notThisTime: null, never: null });

      await fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
      expect(ondismiss).toHaveBeenCalledTimes(1);
      expect(onnotthistime).not.toHaveBeenCalled();
    });
  }
});

describe("chips for every other kind of help", () => {
  for (const variant of CHIP_VARIANTS) {
    it(`${variant} shows Why?, Not this time and Never`, () => {
      render(AssistChip, { variant, mode: "adaptive" });
      const { why, notThisTime, never } = vetoButtons();
      expect(why).toBeTruthy();
      expect(notThisTime).toBeTruthy();
      expect(never).toBeTruthy();
      expect(screen.queryByText("Not selected — aim again")).toBeNull();
    });
  }

  it("uses the chip labels of the study copy", () => {
    render(AssistChip, { variant: "enlarge_target:1.5" });
    expect(screen.getByText("Larger targets")).toBeTruthy();
    cleanup();
    render(AssistChip, { variant: "confirm_selection:hold_700" });
    expect(screen.getByText("Hold to select")).toBeTruthy();
    cleanup();
    render(AssistChip, { variant: "ask_clarification:2" });
    expect(screen.getByText("Which one did you mean?")).toBeTruthy();
  });

  it("Why? shows the explanation and counts the view", async () => {
    const onwhy = vi.fn();
    render(AssistChip, {
      variant: "act",
      mode: "randomized",
      onwhy,
      explanation: {
        summary_key: "coadaptation.explain.randomized",
        values: {},
        contrasts: [],
        excluded: [],
        evidence_source: "literature_prior",
        trials_person: 0,
        trials_today: 0,
      },
    });
    await fireEvent.click(screen.getByRole("button", { name: "Why?" }));
    expect(onwhy).toHaveBeenCalledTimes(1);
    expect(screen.getByText("This block follows the study schedule you agreed to.")).toBeTruthy();
  });

  it("Never asks first and reports the confirm step so the study pauses input", async () => {
    const onnever = vi.fn().mockResolvedValue(undefined);
    const onneverstep = vi.fn();
    render(AssistChip, { variant: "confirm_selection:second_click", mode: "adaptive", onnever, onneverstep });

    await fireEvent.click(screen.getByRole("button", { name: "Never" }));
    expect(onneverstep).toHaveBeenLastCalledWith(true);
    expect(
      screen.getByText("Stop using Confirm each selection in the study? You can undo this in Settings."),
    ).toBeTruthy();
    expect(onnever).not.toHaveBeenCalled();

    await fireEvent.click(screen.getByRole("button", { name: "Yes" }));
    expect(onnever).toHaveBeenCalledTimes(1);
    await vi.waitFor(() => expect(onneverstep).toHaveBeenLastCalledWith(false));
  });

  it("Cancel in the Never step sends nothing", async () => {
    const onnever = vi.fn();
    const onneverstep = vi.fn();
    render(AssistChip, { variant: "act", onnever, onneverstep });
    await fireEvent.click(screen.getByRole("button", { name: "Never" }));
    await fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onnever).not.toHaveBeenCalled();
    expect(onneverstep).toHaveBeenLastCalledWith(false);
    expect(screen.getByRole("button", { name: "Not this time" })).toBeTruthy();
  });
});

describe("Not this time", () => {
  it("outside pinned mode reverts through the callback and offers nothing else", async () => {
    const onnotthistime = vi.fn();
    render(AssistChip, { variant: "enlarge_target:2.0", mode: "adaptive", onnotthistime });
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(onnotthistime).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "Resume adapting" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Remove rule" })).toBeNull();
  });

  it("in pinned mode after a freeze calls onnotthistime and then offers to resume adapting", async () => {
    const calls: string[] = [];
    const onnotthistime = vi.fn(() => calls.push("veto"));
    const onretract = vi.fn(() => calls.push("retract"));
    const onkeep = vi.fn();
    render(AssistChip, {
      variant: "confirm_selection:hold_700",
      mode: "pinned",
      pinnedRule: "freeze",
      onnotthistime,
      onretract,
      onkeep,
    });

    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(onnotthistime).toHaveBeenCalledTimes(1);
    expect(
      screen.getByText("You paused adapting, so Heliox keeps this help in later blocks. Resume adapting?"),
    ).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Resume adapting" }));
    expect(calls).toEqual(["veto", "retract"]);
    expect(onkeep).not.toHaveBeenCalled();
  });

  it("Keep paused changes nothing", async () => {
    const onretract = vi.fn();
    const onkeep = vi.fn();
    render(AssistChip, { variant: "act", mode: "pinned", pinnedRule: "freeze", onretract, onkeep });
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    await fireEvent.click(screen.getByRole("button", { name: "Keep paused" }));
    expect(onkeep).toHaveBeenCalledTimes(1);
    expect(onretract).not.toHaveBeenCalled();
  });

  it("offers to remove a pin, naming the pinned help", async () => {
    render(AssistChip, {
      variant: "enlarge_target:1.5",
      mode: "pinned",
      pinnedRule: "pin",
      pinnedVariant: "enlarge_target:1.5",
    });
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(screen.getByText("You pinned Larger targets. Remove that rule?")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Remove rule" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Keep it" })).toBeTruthy();
  });

  it("offers to remove pin_act", async () => {
    render(AssistChip, { variant: "act", mode: "pinned", pinnedRule: "pin_act" });
    await fireEvent.click(screen.getByRole("button", { name: "Not this time" }));
    expect(screen.getByText("You chose to select exactly what you click. Remove that rule?")).toBeTruthy();
  });
});
