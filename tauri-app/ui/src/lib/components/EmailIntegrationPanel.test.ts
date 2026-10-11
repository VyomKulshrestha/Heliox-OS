import { render, screen } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import EmailIntegrationPanel from "./EmailIntegrationPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => ({
  updateSection: vi.fn(),
  updateSectionDetailed: vi.fn(),
  store: {
    subscribe(run: (next: unknown) => void) {
      run({
        email: {
          enabled: false,
          imap_host: "",
          smtp_host: "",
          smtp_port: 587,
          username: "",
        },
      });
      return () => undefined;
    },
  },
}));

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));
vi.mock("../stores/settings", () => ({
  settings: {
    subscribe: settingsMocks.store.subscribe,
    updateSection: settingsMocks.updateSection,
    updateSectionDetailed: settingsMocks.updateSectionDetailed,
  },
}));

describe("EmailIntegrationPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
  });

  it("surfaces unavailable secure credential storage", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "error",
      available: false,
      providers: [],
      message: "Credential store read failed.",
    });

    render(EmailIntegrationPanel);

    expect(await screen.findByText("Credential store read failed.")).toBeTruthy();
  });
});
