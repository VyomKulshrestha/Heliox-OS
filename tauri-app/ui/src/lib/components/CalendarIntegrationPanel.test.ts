import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CalendarIntegrationPanel from "./CalendarIntegrationPanel.svelte";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));
const settingsMocks = vi.hoisted(() => {
  let value: any;
  const subscribers = new Set<(next: any) => void>();
  return {
    updateSection: vi.fn(),
    updateSectionDetailed: vi.fn(),
    store: {
      subscribe(run: (next: any) => void) {
        subscribers.add(run);
        run(value);
        return () => subscribers.delete(run);
      },
    },
    set(next: any) {
      value = next;
      subscribers.forEach((run) => run(value));
    },
  };
});

vi.mock("../api/daemon", () => ({ call: daemonMocks.call }));
vi.mock("../stores/settings", () => ({
  settings: {
    subscribe: settingsMocks.store.subscribe,
    updateSection: settingsMocks.updateSection,
    updateSectionDetailed: settingsMocks.updateSectionDetailed,
  },
}));

describe("CalendarIntegrationPanel", () => {
  beforeEach(() => {
    daemonMocks.call.mockReset();
    daemonMocks.call.mockResolvedValue({ status: "ok", available: true, providers: [] });
    settingsMocks.updateSection.mockReset();
    settingsMocks.updateSection.mockResolvedValue(true);
    settingsMocks.updateSectionDetailed.mockReset();
    settingsMocks.updateSectionDetailed.mockResolvedValue({ ok: true, message: "", restartRequired: false });
    settingsMocks.set({
      calendar: {
        enabled: false,
        caldav_url: "",
        caldav_username: "",
        caldav_password_provider: "caldav",
        ics_files: [],
      },
    });
  });

  it("saves normalized local calendar paths while CalDAV is disabled", async () => {
    render(CalendarIntegrationPanel);

    await fireEvent.input(screen.getByLabelText(/Local \.ics files/), {
      target: { value: " C:\\Calendars\\work.ics \n\nD:\\shared\\team.ics " },
    });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(settingsMocks.updateSectionDetailed).toHaveBeenCalledWith(
        "calendar",
        expect.objectContaining({
          enabled: false,
          ics_files: ["C:\\Calendars\\work.ics", "D:\\shared\\team.ics"],
        }),
      ),
    );
    expect(screen.getByText(/Saved 2 local calendar sources/)).toBeTruthy();
  });

  it("preserves configured local sources when saving CalDAV", async () => {
    settingsMocks.set({
      calendar: {
        enabled: false,
        caldav_url: "",
        caldav_username: "",
        caldav_password_provider: "caldav",
        ics_files: ["C:\\Calendars\\personal.ics"],
      },
    });
    render(CalendarIntegrationPanel);

    expect((screen.getByLabelText(/Local \.ics files/) as HTMLTextAreaElement).value).toBe(
      "C:\\Calendars\\personal.ics",
    );
  });

  it("does not store a new password for an account the daemon refused", async () => {
    settingsMocks.updateSectionDetailed.mockResolvedValue({
      ok: false,
      message: "calendar.caldav_url must be an https URL",
      restartRequired: false,
    });
    const { container } = render(CalendarIntegrationPanel);
    const [url, user, password] = Array.from(container.querySelectorAll("label input")) as HTMLInputElement[];

    await fireEvent.input(url, { target: { value: "http://cal.example" } });
    await fireEvent.input(user, { target: { value: "me" } });
    await fireEvent.input(password, { target: { value: "app-pass" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("calendar.caldav_url must be an https URL")).toBeTruthy();
    expect(daemonMocks.call).not.toHaveBeenCalledWith("store_api_key", expect.anything());
  });

  it("puts the previous account back when the new password cannot be stored", async () => {
    settingsMocks.set({
      calendar: {
        enabled: false,
        caldav_url: "https://old.example/dav",
        caldav_username: "old",
        caldav_password_provider: "caldav",
        ics_files: [],
      },
    });
    daemonMocks.call.mockImplementation((method: string) => {
      if (method === "store_api_key") return Promise.resolve({ status: "error", message: "Keyring is locked." });
      return Promise.resolve({ status: "ok", available: true, providers: ["caldav"] });
    });
    const { container } = render(CalendarIntegrationPanel);
    const [url, , password] = Array.from(container.querySelectorAll("label input")) as HTMLInputElement[];

    await fireEvent.input(url, { target: { value: "https://new.example/dav" } });
    await fireEvent.input(password, { target: { value: "new-pass" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(/Keyring is locked\. The previous calendar settings were kept\./)).toBeTruthy();
    expect(settingsMocks.updateSectionDetailed).toHaveBeenLastCalledWith(
      "calendar",
      expect.objectContaining({ caldav_url: "https://old.example/dav", caldav_username: "old" }),
    );
  });

  it("surfaces unavailable secure credential storage", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "error",
      available: false,
      providers: [],
      message: "Credential store read failed.",
    });

    render(CalendarIntegrationPanel);

    expect(await screen.findByText("Credential store read failed.")).toBeTruthy();
  });
});
