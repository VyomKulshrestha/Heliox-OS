import { render, screen, waitFor } from "@testing-library/svelte";
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

import LocalModelsPanel, { detail, gpuLine, stateWords, tone } from "./LocalModelsPanel.svelte";
import type { LocalModel } from "./LocalModelsPanel.svelte";

const nvidia = {
  machine: { kind: "nvidia", gpus: [{ name: "NVIDIA GeForce RTX 4060 Laptop GPU", temperature_c: 85.4 }], error: "" },
  max_temperature_c: 83,
  disabled: false,
  disabled_reason: "",
};

function respond(models: LocalModel[], gpu: unknown) {
  daemonMocks.call.mockImplementation(async (method: string) => {
    if (method === "autonomy_review_status") return { status: "ok", local_models: models, gpu };
    return { status: "ok" };
  });
}

describe("LocalModelsPanel", () => {
  beforeEach(() => daemonMocks.call.mockReset());

  it("shows each model in plain words with where it runs and why it is unavailable", async () => {
    respond(
      [
        {
          name: "Gnomon",
          state: "unavailable",
          reason: "Gnomon's process stopped. Restarting it in 5 s.",
          device: null,
          process: "worker",
          retrying: true,
          last_fault: { at: "2026-11-12T04:24:11", message: "The review model's process stopped (exit code 15)." },
        },
        { name: "Voice", state: "loaded", device: "cpu", reason: "", process: "worker", retrying: false },
        { name: "Screen text", state: "loaded", device: "system", reason: "Windows OCR.", process: "system" },
      ],
      nvidia,
    );
    render(LocalModelsPanel);

    await waitFor(() => expect(screen.getByText("Gnomon")).toBeTruthy());
    expect(screen.getByText("Unavailable")).toBeTruthy();
    expect(screen.getByText("Gnomon's process stopped. Restarting it in 5 s.")).toBeTruthy();
    expect(screen.getByText(/Last problem at 2026-11-12 04:24:11/)).toBeTruthy();
    expect(screen.getByText("On the processor.")).toBeTruthy();
    expect(screen.getByText("Uses what is built into the system. Windows OCR.")).toBeTruthy();
    expect(screen.getByTestId("gpu-line").textContent).toContain(
      "NVIDIA GeForce RTX 4060 Laptop GPU, 85 °C (models start below 83 °C).",
    );
  });

  it("says when the graphics card is switched off for the session", async () => {
    respond([], {
      ...nvidia,
      disabled: true,
      disabled_reason: "Stopped using the graphics card for this session after 3 faults in 10 minutes.",
    });
    render(LocalModelsPanel);
    await waitFor(() => expect(screen.getByRole("status").textContent).toMatch(/after 3 faults in 10 minutes/));
  });

  it("reads as normal on a machine without an NVIDIA card or on Apple silicon", () => {
    expect(gpuLine({ machine: { kind: "cpu", gpus: [] } })).toBe(
      "No NVIDIA graphics card: local models run on the processor.",
    );
    expect(gpuLine({ machine: { kind: "apple" } })).toMatch(/^Apple silicon: the graphics processor shares/);
    const unsupported: LocalModel = {
      name: "Gnomon",
      state: "not_supported",
      reason: "Not supported on this system: Gnomon's 4-bit model needs an NVIDIA graphics card with CUDA.",
    };
    expect(stateWords(unsupported)).toBe("Not on this computer");
    expect(tone(unsupported)).toBe("quiet");
    expect(tone({ name: "Turn detector", state: "off" })).toBe("quiet");
  });

  it("maps every state and adds the retry note only when the reason does not already say it", () => {
    const states = ["loaded", "loading", "not_loaded", "unavailable", "off", "not_supported", "weird"];
    expect(states.map((state) => stateWords({ name: "x", state }))).toEqual([
      "Ready",
      "Loading",
      "Not loaded yet",
      "Unavailable",
      "Off",
      "Not on this computer",
      "Unknown",
    ]);
    expect(
      detail({ name: "Voice", state: "unavailable", reason: "Speaking with the system voice.", retrying: true }),
    ).toBe("Speaking with the system voice. Tries again by itself.");
    expect(
      detail({ name: "Gnomon", state: "unavailable", reason: "Needs 2.5 GB. Tries again on its own.", retrying: true }),
    ).toBe("Needs 2.5 GB. Tries again on its own.");
    expect(tone({ name: "x", state: "unavailable" })).toBe("warn");
  });

  it("says plainly when the daemon cannot be asked", async () => {
    daemonMocks.call.mockResolvedValue(new Error("Daemon unavailable"));
    render(LocalModelsPanel);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toBe("Daemon unavailable"));
  });
});
