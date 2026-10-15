import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import ConfirmChip from "./ConfirmChip.svelte";

let now = 1000;

beforeEach(() => {
  now = 1000;
  vi.spyOn(performance, "now").mockImplementation(() => now);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function renderChip(method: "second_click" | "hold" = "second_click", extra: Record<string, unknown> = {}) {
  const onconfirm = vi.fn();
  const oncancel = vi.fn();
  render(ConfirmChip, { label: "apple", method, holdMs: 700, onconfirm, oncancel, ...extra });
  return { onconfirm, oncancel };
}

async function click(button: HTMLElement) {
  await fireEvent.pointerDown(button, { button: 0 });
  await fireEvent.click(button, { detail: 1 });
}

describe("confirm step: 250 ms and a fresh press", () => {
  it("ignores a confirm click whose press starts within 250 ms of the chip appearing", async () => {
    const { onconfirm } = renderChip();
    now = 1100;
    await click(screen.getByRole("button", { name: "Select" }));
    now = 1249;
    await click(screen.getByRole("button", { name: "Select" }));
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it("confirms on a fresh press that starts at least 250 ms after the chip appeared", async () => {
    const { onconfirm } = renderChip();
    expect(screen.getByText("Select apple?")).toBeTruthy();
    now = 1250;
    await click(screen.getByRole("button", { name: "Select" }));
    expect(onconfirm).toHaveBeenCalledWith({ kind: "click", sinceChipMs: 250 });
  });

  it("ignores the second click of a double-click that began before the chip (no fresh press)", async () => {
    const { onconfirm } = renderChip();
    const select = screen.getByRole("button", { name: "Select" });
    // The press of a double-click's second click landed before the chip appeared; only the release arrives later.
    now = 1400;
    await fireEvent.click(select, { detail: 2 });
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it("a press inside the first 250 ms does not become fresh when its release comes later", async () => {
    const { onconfirm } = renderChip();
    const select = screen.getByRole("button", { name: "Select" });
    now = 1200;
    await fireEvent.pointerDown(select, { button: 0 });
    now = 1600;
    await fireEvent.click(select, { detail: 1 });
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it("ignores key repeats and keys inside 250 ms; a fresh Enter confirms", async () => {
    const { onconfirm } = renderChip();
    const select = screen.getByRole("button", { name: "Select" });
    now = 1100;
    await fireEvent.keyDown(select, { key: "Enter" });
    now = 1500;
    await fireEvent.keyDown(select, { key: "Enter", repeat: true });
    await fireEvent.keyDown(select, { key: " ", repeat: true });
    expect(onconfirm).not.toHaveBeenCalled();
    await fireEvent.keyDown(select, { key: "Enter" });
    expect(onconfirm).toHaveBeenCalledWith({ kind: "key", sinceChipMs: 500 });
  });

  it("Not this and Escape cancel", async () => {
    const { oncancel, onconfirm } = renderChip();
    await fireEvent.click(screen.getByRole("button", { name: "Not this" }));
    expect(oncancel).toHaveBeenCalledTimes(1);
    await fireEvent.keyDown(screen.getByRole("button", { name: "Select" }), { key: "Escape" });
    expect(oncancel).toHaveBeenCalledTimes(2);
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it("has no timer: nothing confirms while the chip simply stays open", async () => {
    vi.useFakeTimers();
    try {
      const { onconfirm, oncancel } = renderChip();
      now = 60000;
      vi.advanceTimersByTime(60000);
      expect(onconfirm).not.toHaveBeenCalled();
      expect(oncancel).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("hold step", () => {
  it("shows a progress ring and no confirm button; the hold itself is the input", () => {
    const { onconfirm } = renderChip("hold", { holdStartedAt: 1000 });
    expect(screen.getByRole("progressbar")).toBeTruthy();
    expect(screen.getByText("Keep holding to select apple")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Select" })).toBeNull();
    expect(onconfirm).not.toHaveBeenCalled();
  });
});
