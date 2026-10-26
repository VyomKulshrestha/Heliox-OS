import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

const settingsMocks = vi.hoisted(() => ({ updateSectionDetailed: vi.fn() }));
vi.mock("../stores/settings", async () => {
  const { writable } = await import("svelte/store");
  const store = writable({ clipboard: { history: false } });
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: settingsMocks.updateSectionDetailed } };
});

import YourDataPanel from "./YourDataPanel.svelte";

const stores = [
  { key: "timeline", label: "Activity timeline", about: "Apps and titles.", size: "40.0 KB", items: 12 },
  { key: "notes", label: "Notes", about: "Your quick notes.", size: "120 bytes", items: 2 },
  {
    key: "clipboard",
    label: "Clipboard history",
    about: "Text you copied recently, in memory only.",
    size: "0 bytes",
    items: 0,
    memory_only: true,
  },
];

describe("YourDataPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "privacy_inventory" || method === "privacy_clear") return { status: "ok", stores };
      if (method === "backup_list")
        return { status: "ok", backups: [{ name: "heliox-backup-20261024-120000.zip", size: "2.0 KB", at: 1 }] };
      if (method === "backup_create") return { status: "ok", path: "C:/Users/me/Downloads/heliox-backup.zip" };
      if (method === "backup_restore") return { status: "ok", restored: ["notes.json"] };
      if (method === "timeline_forget") return { status: "ok", removed: 3 };
      return { status: "ok" };
    });
  });

  it("clears a store only on the second click", async () => {
    render(YourDataPanel);
    await waitFor(() => expect(screen.getByText("Notes")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Clear Notes" }));
    expect(daemonMocks.call).not.toHaveBeenCalledWith("privacy_clear", expect.anything());
    await fireEvent.click(screen.getByRole("button", { name: "Clear Notes" }));

    await waitFor(() =>
      expect(daemonMocks.call).toHaveBeenCalledWith("privacy_clear", { key: "notes", confirmed: true }),
    );
  });

  it("forgets the last hour, backs up and restores", async () => {
    render(YourDataPanel);
    await waitFor(() => expect(screen.getByText("Activity timeline")).toBeTruthy());

    await fireEvent.click(screen.getByRole("button", { name: "Forget last hour" }));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Removed 3 timeline entries"));

    await fireEvent.click(screen.getByRole("button", { name: "Back up to Downloads" }));
    expect(daemonMocks.call).toHaveBeenCalledWith("backup_create", { include_timeline: false });

    const restore = screen.getByRole("button", { name: "Restore heliox-backup-20261024-120000.zip" });
    await fireEvent.click(restore);
    await fireEvent.click(restore);
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Restored: notes.json."));
  });

  it("turns clipboard history on from its row and shows it is memory only", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true });
    render(YourDataPanel);
    await waitFor(() => expect(screen.getByText("Clipboard history")).toBeTruthy());
    expect(screen.getByText(/in memory only$/)).toBeTruthy();

    await fireEvent.click(screen.getByRole("checkbox", { name: "Keep" }));
    await waitFor(() =>
      expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("clipboard", { history: true }),
    );
    await waitFor(() => expect(screen.getByText("Clipboard history is on.")).toBeTruthy());
  });
});
