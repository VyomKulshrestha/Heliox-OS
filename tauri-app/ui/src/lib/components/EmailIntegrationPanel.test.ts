import { fireEvent, render, screen } from "@testing-library/svelte";
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
    settingsMocks.updateSectionDetailed.mockReset();
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

  it("leaves the account as it was when the person does not approve a new mail server", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok", providers: ["email"], available: true });
    settingsMocks.updateSectionDetailed.mockResolvedValue({
      ok: false,
      cancelled: true,
      message: "Not changed.",
      restartRequired: false,
    });
    render(EmailIntegrationPanel);
    const imap = screen.getByPlaceholderText("imap.example.com") as HTMLInputElement;
    await fireEvent.input(imap, { target: { value: "imap.mail.example" } });
    await fireEvent.input(screen.getByPlaceholderText("smtp.example.com"), { target: { value: "smtp.mail.example" } });
    await fireEvent.input(screen.getByLabelText("Username"), { target: { value: "me" } });

    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    const note = await screen.findByText("Not changed.");
    expect(note.classList.contains("error")).toBe(false);
    expect(note.classList.contains("ok")).toBe(false);
    expect(imap.value).toBe("");
    expect(daemonMocks.call).not.toHaveBeenCalledWith("store_api_key", expect.anything());
  });
});
