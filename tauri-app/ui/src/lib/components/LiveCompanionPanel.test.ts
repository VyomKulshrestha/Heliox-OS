import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({ call: vi.fn(), updateSectionDetailed: vi.fn() }));

vi.mock("../api/daemon", () => ({ call: mocks.call }));
vi.mock("../stores/settings", () => {
  const store = writable({
    live: {
      enabled: true,
      level: "balanced",
      read_focused_text: true,
      ocr_fallback: false,
      listen_to_speech: true,
      personal_timing: true,
      muted_kinds: [],
    },
  });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: mocks.updateSectionDetailed } };
});

import LiveCompanionPanel from "./LiveCompanionPanel.svelte";

const learned = {
  mid_sentence_seconds: 3.1,
  boundary_seconds: 1.2,
  default_seconds: 1.2,
  learned_from: 42,
  cut_offs: 2,
  well_timed: 9,
};

describe("LiveCompanionPanel timing", () => {
  beforeEach(() => {
    mocks.call.mockReset();
    mocks.updateSectionDetailed.mockReset();
    mocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "" });
  });

  it("says what it learned about the person's pauses and can start over", async () => {
    mocks.call.mockImplementation(async (method: string) =>
      method === "live_status"
        ? { status: "ok", kinds: [], timing: learned }
        : { status: "ok", timing: { ...learned, mid_sentence_seconds: 1.2, learned_from: 0, cut_offs: 0 } },
    );
    render(LiveCompanionPanel);
    const note = await screen.findByTestId("live-timing");
    expect(note.textContent).toContain("Mid-sentence it now waits 3.1 s before speaking up, learned from 42 pauses.");
    expect(note.textContent).toContain("It cut in on you 2 times and waits longer since.");

    await fireEvent.click(screen.getByRole("button", { name: "Start over" }));
    expect(mocks.call).toHaveBeenCalledWith("live_set", { reset_timing: true });
    await waitFor(() => expect(note.textContent).toContain("Still learning how you type."));
  });

  it("turns the learning off through the live settings", async () => {
    mocks.call.mockResolvedValue({ status: "ok", kinds: [], timing: learned });
    render(LiveCompanionPanel);
    await fireEvent.click(screen.getByRole("checkbox", { name: /pause in the middle of a sentence/ }));
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith("live", { personal_timing: false });
  });

  it("offers checks against the person's own calendar, off until they choose it", async () => {
    mocks.call.mockResolvedValue({ status: "ok", kinds: [], timing: learned });
    render(LiveCompanionPanel);
    const box = screen.getByRole("checkbox", { name: /your own calendar, to-dos and reminders/ }) as HTMLInputElement;
    expect(box.checked).toBe(false);
    expect(box.closest("label")?.textContent).toContain("qwen3.5:4b in Ollama, on this computer");
    await fireEvent.click(box);
    expect(mocks.updateSectionDetailed).toHaveBeenCalledWith("live", { grounded_checks: true });
  });
});
