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
  const store = writable({});
  return { settings: { subscribe: store.subscribe, updateSectionDetailed: settingsMocks.updateSectionDetailed } };
});

import DataFolderSize from "./DataFolderSize.svelte";

const GB = 1024 ** 3;
const sizes = {
  status: "ok",
  total_bytes: 6 * GB,
  warn_gb: 5,
  over: true,
  stores: [
    { name: "Audit log", bytes: 3 * GB },
    {
      name: "Experience ledger",
      bytes: 200 * 1024 ** 2,
      note: "Grows about 11.8 KB per request: above about 245 requests a day that passes 1 GB a year.",
    },
  ],
};

/** A call that never answers within the test, made to fail the way a lost connection does. */
async function noReply(): Promise<never> {
  throw new Error("Heliox did not answer. Try again in a moment.");
}

describe("DataFolderSize", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    settingsMocks.updateSectionDetailed.mockReset();
  });

  it("shows the folder and each store, and says when it is over the limit", async () => {
    daemonMocks.call.mockResolvedValue(sizes);
    render(DataFolderSize);
    await waitFor(() => expect(screen.getByText("6.0 GB")).toBeTruthy());
    expect(screen.getByText(/over 5 GB/)).toBeTruthy();
    expect(screen.getByText("Audit log")).toBeTruthy();
    expect(screen.getByText("200 MB")).toBeTruthy();
    expect(screen.getByText(/above about 245 requests a day that passes 1 GB a year/)).toBeTruthy();
    expect((screen.getByLabelText("Size limit in GB") as HTMLInputElement).value).toBe("5");
  });

  it("shows a refusal in the daemon's words", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "Heliox is still starting." });
    render(DataFolderSize);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("Heliox is still starting."));
  });

  it("says so when there is no reply", async () => {
    daemonMocks.call.mockImplementation(noReply);
    render(DataFolderSize);
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe("Heliox did not answer. Try again in a moment."),
    );
  });

  it("saves the limit through the settings store and refuses a nonsense number", async () => {
    daemonMocks.call.mockResolvedValue(sizes);
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    render(DataFolderSize);
    await waitFor(() => expect(screen.getByText("6.0 GB")).toBeTruthy());
    const input = screen.getByLabelText("Size limit in GB");
    await fireEvent.input(input, { target: { value: "0" } });
    await fireEvent.click(screen.getByText("Save"));
    expect(screen.getByRole("alert").textContent).toContain("1 or more");
    expect(settingsMocks.updateSectionDetailed).not.toHaveBeenCalled();
    await fireEvent.input(input, { target: { value: "20" } });
    await fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("passes 20 GB"));
    expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith("storage", { data_folder_warn_gb: 20 });
  });

  it("archives only on the second click, with the three replies", async () => {
    daemonMocks.call.mockResolvedValue(sizes);
    render(DataFolderSize);
    await waitFor(() => expect(screen.getByText("6.0 GB")).toBeTruthy());
    await fireEvent.input(screen.getByLabelText("Archive entries older than"), { target: { value: "2026-01-01" } });

    await fireEvent.click(screen.getByText("Archive"));
    expect(daemonMocks.call).not.toHaveBeenCalledWith("audit_archive", expect.anything());

    // Refused (for example not enough free space): the reason, nothing else.
    daemonMocks.call.mockResolvedValueOnce({ status: "error", message: "There is not enough free space." });
    await fireEvent.click(screen.getByText("Click again to archive"));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("There is not enough free space."));
    expect(daemonMocks.call).toHaveBeenCalledWith("audit_archive", { older_than: "2026-01-01" });

    // No reply.
    daemonMocks.call.mockImplementationOnce(noReply);
    await fireEvent.click(screen.getByText("Archive"));
    await fireEvent.click(screen.getByText("Click again to archive"));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe("Heliox did not answer. Try again in a moment."),
    );

    // Done: the daemon's sentence and the new sizes.
    daemonMocks.call.mockResolvedValueOnce({
      status: "ok",
      message: "Moved 40 audit entries from before 2026-01-01 to audit-2026-11-12.jsonl, beside the audit log.",
      sizes: { ...sizes, total_bytes: 4 * GB, over: false },
    });
    await fireEvent.click(screen.getByText("Archive"));
    await fireEvent.click(screen.getByText("Click again to archive"));
    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("Moved 40 audit entries"));
    expect(screen.getByText("4.0 GB")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
