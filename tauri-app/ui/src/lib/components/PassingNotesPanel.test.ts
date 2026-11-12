import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  update: vi.fn(async () => ({ ok: true, message: "" })),
  store: null as null | { set: (value: { memory: { passing_notes: boolean } }) => void },
}));

vi.mock("../stores/settings", async () => {
  const { writable } = await import("svelte/store");
  const store = writable({ memory: { passing_notes: true } });
  mocks.store = store;
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: mocks.update } };
});

import PassingNotesPanel from "./PassingNotesPanel.svelte";

describe("things said in passing", () => {
  it("is one switch, on by default, and says what is never kept", async () => {
    render(PassingNotesPanel);
    const box = screen.getByRole("checkbox", { name: "Keep notes of things I mention in passing" });
    expect((box as HTMLInputElement).checked).toBe(true);
    expect(screen.getByText(/passwords, keys and codes; health/)).toBeTruthy();
    await fireEvent.click(box);
    await waitFor(() => expect(mocks.update).toHaveBeenCalledWith("memory", { passing_notes: false }));
  });

  it("shows it off when the setting is off", () => {
    mocks.store!.set({ memory: { passing_notes: false } });
    render(PassingNotesPanel);
    expect((screen.getByRole("checkbox") as HTMLInputElement).checked).toBe(false);
  });
});
