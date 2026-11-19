import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  // A vi.fn whose promise rejects fails the test under vitest 4 even when the component catches it, so the mock
  // resolves to an Error and this plain wrapper throws it.
  const call = async (...args: Parameters<typeof actual.call>) => {
    const result = await daemonMocks.call(...args);
    if (result instanceof Error) throw result;
    return result;
  };
  return { ...actual, call };
});

import SpeechDriverErrorsRow, { countWords, showRow } from "./SpeechDriverErrorsRow.svelte";

const moved = { errors: 12, limit: 3, window_days: 7, speech_on_processor: true, speech_despite_driver_errors: false };

function respond(review: unknown, update: unknown = { status: "ok", restart_required: true }) {
  const asked: { method: string; params: unknown }[] = [];
  daemonMocks.call.mockImplementation(async (method: string, params: unknown) => {
    asked.push({ method, params });
    if (method === "autonomy_review_status") return review;
    if (method === "update_config") return update;
    return { status: "ok" };
  });
  return asked;
}

async function settle() {
  for (let i = 0; i < 5; i += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("SpeechDriverErrorsRow", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("shows the switch, off, with the count when the driver's record moved speech recognition", async () => {
    respond({ status: "ok", gpu: { driver_record: moved } });
    render(SpeechDriverErrorsRow);
    const row = await screen.findByTestId("speech-driver-errors");
    expect(row.textContent).toContain("reported 12 errors in the last 7 days");
    expect(row.textContent).toContain("Off: speech recognition uses the processor");
    expect(screen.getByRole("switch").getAttribute("aria-checked")).toBe("false");
  });

  it("saves the switch to [gpu] and says a restart applies it", async () => {
    const asked = respond({ status: "ok", gpu: { driver_record: moved } });
    render(SpeechDriverErrorsRow);
    await fireEvent.click(await screen.findByRole("switch"));
    await waitFor(() => expect(screen.getByText("Saved. Restart Heliox to apply it.")).toBeTruthy());
    expect(asked.find((entry) => entry.method === "update_config")?.params).toEqual({
      section: "gpu",
      values: { speech_despite_driver_errors: true },
    });
    expect(screen.getByRole("switch").getAttribute("aria-checked")).toBe("true");
  });

  it("says the daemon's reason when it refuses the change", async () => {
    respond({ status: "ok", gpu: { driver_record: moved } }, { status: "error", message: "Invalid config key" });
    render(SpeechDriverErrorsRow);
    await fireEvent.click(await screen.findByRole("switch"));
    await waitFor(() => expect(screen.getByText("Invalid config key")).toBeTruthy());
  });

  it("shows nothing with a bare answer, a refusal, or no answer at all", async () => {
    respond({ status: "ok" });
    const bare = render(SpeechDriverErrorsRow);
    await settle();
    expect(screen.queryByTestId("speech-driver-errors")).toBeNull();
    bare.unmount();

    respond(new Error("Method not found: autonomy_review_status"));
    const refused = render(SpeechDriverErrorsRow);
    await settle();
    expect(screen.queryByTestId("speech-driver-errors")).toBeNull();
    refused.unmount();

    // No answer to the status read (the other reads still answer, as in the app).
    daemonMocks.call.mockImplementation(async (method: string) =>
      method === "autonomy_review_status" ? new Promise(() => {}) : { status: "ok" },
    );
    const silent = render(SpeechDriverErrorsRow);
    await settle();
    expect(screen.queryByTestId("speech-driver-errors")).toBeNull();
    silent.unmount();
  });

  it("is shown only when the rule matters: moved, or switched on over a record past the limit", () => {
    expect(showRow(null)).toBe(false);
    expect(showRow({ errors: 0, limit: 3, speech_on_processor: false })).toBe(false);
    expect(showRow({ errors: 2, limit: 3, speech_on_processor: false, speech_despite_driver_errors: true })).toBe(
      false,
    );
    expect(showRow({ errors: 5, limit: 3, speech_on_processor: false, speech_despite_driver_errors: true })).toBe(true);
    expect(showRow(moved)).toBe(true);
    expect(countWords({ errors: 1, window_days: 7 })).toBe(
      "This graphics card's driver reported 1 error in the last 7 days.",
    );
    expect(countWords({ errors: null })).toBe("This graphics card's driver reported errors in the last 7 days.");
  });
});
