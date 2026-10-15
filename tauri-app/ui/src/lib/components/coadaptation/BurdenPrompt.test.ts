import { cleanup, fireEvent, render, screen, within } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import BurdenPrompt from "./BurdenPrompt.svelte";

afterEach(cleanup);

const EFFORT = "How much effort did selecting take in the block you just finished?";
const INTRUSIVENESS = "How much did Heliox's help get in your way in that block?";
const CONTROL = "How much did you feel in control of what got selected?";

function scale(question: string) {
  return within(screen.getByRole("radiogroup", { name: question }));
}

describe("BurdenPrompt", () => {
  it("asks the three items of the block just finished on a 7-point scale", () => {
    render(BurdenPrompt, { showIntrusiveness: true, onsubmit: vi.fn() });
    for (const question of [EFFORT, INTRUSIVENESS, CONTROL]) {
      expect(scale(question).getAllByRole("radio")).toHaveLength(7);
    }
    expect(screen.getByText("A short custom rating, not a validated clinical scale.")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Skip" })).toHaveLength(3);
  });

  it("sends the answers, and an unanswered or skipped item as skipped", async () => {
    const onsubmit = vi.fn();
    render(BurdenPrompt, { showIntrusiveness: true, onsubmit });
    await fireEvent.click(scale(EFFORT).getByRole("radio", { name: "3" }));
    await fireEvent.click(scale(CONTROL).getByRole("radio", { name: "6" }));
    await fireEvent.click(screen.getAllByRole("button", { name: "Skip" })[2]);
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onsubmit).toHaveBeenCalledWith({
      effort: 3,
      intrusiveness: null,
      control: null,
      skipped: ["intrusiveness", "control"],
    });
  });

  it("hides B2 after an assistance_off block: intrusiveness stays null and is not counted as skipped", async () => {
    const onsubmit = vi.fn();
    render(BurdenPrompt, { showIntrusiveness: false, onsubmit });
    expect(screen.queryByText(INTRUSIVENESS)).toBeNull();
    await fireEvent.click(scale(EFFORT).getByRole("radio", { name: "2" }));
    await fireEvent.click(scale(CONTROL).getByRole("radio", { name: "7" }));
    await fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onsubmit).toHaveBeenCalledWith({ effort: 2, intrusiveness: null, control: 7, skipped: [] });
  });

  it("is keyboard accessible: arrows move the answer", async () => {
    const onsubmit = vi.fn();
    render(BurdenPrompt, { showIntrusiveness: false, onsubmit });
    const first = scale(EFFORT).getByRole("radio", { name: "1" });
    await fireEvent.keyDown(first, { key: "ArrowRight" });
    await fireEvent.keyDown(first, { key: "ArrowRight" });
    await fireEvent.keyDown(first, { key: "ArrowRight" });
    expect(scale(EFFORT).getByRole("radio", { name: "3" }).getAttribute("aria-checked")).toBe("true");
  });
});
