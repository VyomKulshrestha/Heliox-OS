import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import ClarifyChoice from "./ClarifyChoice.svelte";

let now = 1000;

beforeEach(() => {
  now = 1000;
  vi.spyOn(performance, "now").mockImplementation(() => now);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

interface Candidate {
  index: number;
  label: string;
  cx?: number;
  cy?: number;
  w?: number;
  h?: number;
}

const NEIGHBOURS: Candidate[] = [
  { index: 4, label: "apple", cx: 100, cy: 120, w: 28, h: 28 },
  { index: 5, label: "river", cx: 134, cy: 120, w: 28, h: 28 },
];

function renderChoice(candidates: Candidate[] = NEIGHBOURS) {
  const onchoose = vi.fn();
  render(ClarifyChoice, { candidates, onchoose });
  return onchoose;
}

/** A press and its click, as a mouse produces them (`detail` is the click count). */
async function press(button: HTMLElement, detail = 1) {
  await fireEvent.pointerDown(button, { button: 0 });
  await fireEvent.click(button, { detail });
}

describe("ClarifyChoice", () => {
  it("draws each candidate at twice its size over its position, plus None of these", () => {
    renderChoice();
    expect(screen.getByText("Which one did you mean?")).toBeTruthy();
    const apple = screen.getByRole("button", { name: "apple" }) as HTMLButtonElement;
    expect(apple.style.width).toBe("56px");
    expect(apple.style.height).toBe("56px");
    expect(apple.style.left).toBe("72px");
    expect(screen.getByRole("button", { name: "None of these" })).toBeTruthy();
  });

  it("draws the more likely candidate on top where two enlarged buttons overlap", () => {
    renderChoice();
    // apple spans 72-128 px and river 106-162 px; the overlap belongs to apple, the first (most likely) candidate.
    const apple = screen.getByRole("button", { name: "apple" }) as HTMLButtonElement;
    const river = screen.getByRole("button", { name: "river" }) as HTMLButtonElement;
    expect(Number(apple.style.zIndex)).toBeGreaterThan(Number(river.style.zIndex));
  });

  it("commits only an explicit choice; None of these chooses nothing", async () => {
    const onchoose = renderChoice([
      { index: 1, label: "stone" },
      { index: 7, label: "cloud" },
      { index: 2, label: "maple" },
    ]);
    expect(onchoose).not.toHaveBeenCalled();
    now = 1300;
    await press(screen.getByRole("button", { name: "cloud" }));
    expect(onchoose).toHaveBeenCalledTimes(1);
    expect(onchoose).toHaveBeenLastCalledWith(7);
  });

  it("None of these, chosen with a fresh press, chooses nothing", async () => {
    const onchoose = renderChoice();
    now = 1300;
    await press(screen.getByRole("button", { name: "None of these" }));
    expect(onchoose).toHaveBeenCalledWith(null);
  });

  it("ignores the second click of a double-click that lands on the prompt as it appears", async () => {
    const onchoose = renderChoice();
    // The first click's release opened the prompt; the second press comes about 120 ms later.
    now = 1120;
    await press(screen.getByRole("button", { name: "river" }), 2);
    expect(onchoose).not.toHaveBeenCalled();
  });

  it("ignores a click whose press began before the prompt (no fresh press)", async () => {
    const onchoose = renderChoice();
    now = 1400;
    await fireEvent.click(screen.getByRole("button", { name: "river" }), { detail: 2 });
    expect(onchoose).not.toHaveBeenCalled();
  });

  it("a press inside the first 250 ms does not become fresh when its release comes later", async () => {
    const onchoose = renderChoice();
    const river = screen.getByRole("button", { name: "river" });
    now = 1200;
    await fireEvent.pointerDown(river, { button: 0 });
    now = 1600;
    await fireEvent.click(river, { detail: 1 });
    expect(onchoose).not.toHaveBeenCalled();
    // The same holds for None of these: a reflexive press is not a correction either.
    const none = screen.getByRole("button", { name: "None of these" });
    now = 1001;
    await fireEvent.pointerDown(none, { button: 0 });
    await fireEvent.click(none, { detail: 1 });
    expect(onchoose).not.toHaveBeenCalled();
  });

  it("accepts a press that starts 250 ms after the prompt appeared", async () => {
    const onchoose = renderChoice();
    now = 1250;
    await press(screen.getByRole("button", { name: "river" }));
    expect(onchoose).toHaveBeenCalledWith(5);
  });

  it("ignores key repeats and keys inside 250 ms; a fresh Enter chooses the focused candidate", async () => {
    const onchoose = renderChoice();
    const apple = screen.getByRole("button", { name: "apple" });
    // A held Enter from the activation that opened the prompt auto-repeats onto the focused first button.
    now = 1100;
    await fireEvent.keyDown(apple, { key: "Enter" });
    now = 1500;
    await fireEvent.keyDown(apple, { key: "Enter", repeat: true });
    await fireEvent.keyDown(apple, { key: " ", repeat: true });
    await fireEvent.keyUp(apple, { key: " " });
    expect(onchoose).not.toHaveBeenCalled();
    await fireEvent.keyDown(apple, { key: "Enter" });
    expect(onchoose).toHaveBeenCalledTimes(1);
    expect(onchoose).toHaveBeenCalledWith(4);
  });

  it("keeps its presses away from the study surface underneath", async () => {
    const outside = vi.fn();
    document.addEventListener("pointerdown", outside);
    try {
      renderChoice([{ index: 0, label: "lamp" }]);
      await fireEvent.pointerDown(screen.getByRole("button", { name: "lamp" }));
      expect(outside).not.toHaveBeenCalled();
    } finally {
      document.removeEventListener("pointerdown", outside);
    }
  });
});
