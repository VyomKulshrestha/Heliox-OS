import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});

import ReceiptsPanel from "./ReceiptsPanel.svelte";

const row = {
  receipt_id: "abc123",
  finished_at: "2026-10-23T10:00:00+00:00",
  status: "success",
  summary: "Wrote notes.",
  request: "write my notes",
  source: "voice",
  action_count: 2,
  signed: true,
};

describe("ReceiptsPanel", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("lists receipts with the key fingerprint and verifies one", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "receipts_list")
        return { status: "ok", receipts: [row], key: { available: true, fingerprint: "aaaa:bbbb" } };
      if (method === "receipt_verify")
        return { status: "ok", result: { valid: true, this_computer: true, reasons: [] } };
      return { status: "ok" };
    });
    render(ReceiptsPanel);

    await waitFor(() => expect(screen.getByText("write my notes")).toBeTruthy());
    expect(screen.getByText("aaaa:bbbb")).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Verify" }));

    await waitFor(() => expect(screen.getByText("Valid, signed by this computer")).toBeTruthy());
    expect(daemonMocks.call).toHaveBeenCalledWith("receipt_verify", { receipt_id: "abc123" });
  });

  it("reports a broken chain in plain words", async () => {
    daemonMocks.call.mockImplementation(async (method: string) => {
      if (method === "receipts_list")
        return { status: "ok", receipts: [], key: { available: false, reason: "no key" } };
      return {
        status: "ok",
        result: {
          valid: false,
          count: 3,
          problems: [{ file: "b.json", reason: "The receipt before this one was changed or removed." }],
        },
      };
    });
    render(ReceiptsPanel);

    await waitFor(() => expect(screen.getByText(/Receipts are not being signed/)).toBeTruthy());
    await fireEvent.click(screen.getByRole("button", { name: "Check all" }));

    await waitFor(() => expect(screen.getByRole("status").textContent).toContain("changed or removed"));
  });
});
