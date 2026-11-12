import { fireEvent, render, screen } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RemoveHelioxFiles from "./RemoveHelioxFiles.svelte";

const plan = {
  home_known: true,
  message: "",
  locations: [
    { path: "/home/sam/.config/heliox-os/env", label: "Setup: env", kind: "app", bytes: 450_000_000 },
    {
      path: "/home/sam/.local/share/heliox-os",
      label: "Memory, timeline, receipts and undo copies",
      kind: "data",
      bytes: 20_000_000,
    },
  ],
  manual_steps: ["Uninstall the Heliox app itself the way your system removes apps."],
  credential_services: ["pilot-ai-command-center"],
};
const report = {
  removed: ["/home/sam/.config/heliox-os/env", "/home/sam/.heliox"],
  failed: [
    "C:\\Users\\sam\\AppData\\Local\\com.helioxos.app: the running app holds this folder open; it goes when you uninstall the app.",
  ],
  freed_bytes: 470_000_000,
  kept_data: false,
  credentials: {
    outcome: "removed",
    removed: ["pilot-ai-command-center / anthropic"],
    remaining: ["pilot-ai-command-center / ssh:prod:private-key"],
    how: "Open Credential Manager > Windows Credentials and remove the entries named above (service@account).",
  },
  manual_steps: ["Uninstall the Heliox app itself the way your system removes apps."],
};

let replies: Record<string, () => Promise<unknown>> = {};
const invoke = vi.fn((command: string, _args?: unknown) => (replies[command] ?? (async () => undefined))());
vi.mock("../api/invoke", () => ({ invoke: (command: string, args?: unknown) => invoke(command, args) }));

async function choose(everything: boolean) {
  await fireEvent.click(screen.getByRole("button", { name: "Show what Heliox stores" }));
  await screen.findByText(/Setup: env/);
  if (everything) await fireEvent.click(screen.getByRole("radio", { name: /Remove everything/ }));
  await fireEvent.click(screen.getByRole("checkbox", { name: /will be deleted/ }));
  await fireEvent.click(
    screen.getByRole("button", { name: everything ? "Remove everything" : "Remove the app's files" }),
  );
}

describe("Remove Heliox's files", () => {
  beforeEach(() => {
    invoke.mockClear();
    replies = { heliox_files_plan: async () => plan, heliox_files_remove: async () => report };
  });

  it("lists what will go with sizes, keeps data by default, and deletes only after confirmation", async () => {
    render(RemoveHelioxFiles, { force: true });
    expect(invoke).not.toHaveBeenCalled();
    await fireEvent.click(screen.getByRole("button", { name: "Show what Heliox stores" }));
    expect(await screen.findByText(/Setup: env/)).toBeTruthy();
    expect(screen.queryByText(/Memory, timeline/)).toBeNull();
    expect((screen.getByRole("button", { name: "Remove the app's files" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/Your saved keys stay in the credential store/)).toBeTruthy();
    await fireEvent.click(screen.getByRole("radio", { name: /Remove everything/ }));
    expect(screen.getByText(/Memory, timeline/)).toBeTruthy();
    expect(screen.getByText(/removed from the credential store first/)).toBeTruthy();
  });

  it("after removal shows what went, what failed and why, the keys left, the manual steps and Quit", async () => {
    render(RemoveHelioxFiles, { force: true });
    await choose(true);
    expect(invoke).toHaveBeenCalledWith("heliox_files_remove", { keepData: false, confirm: "remove" });
    expect(await screen.findByRole("heading", { name: "Heliox's files were removed" })).toBeTruthy();
    expect(screen.getByText(/Removed 2 folders/)).toBeTruthy();
    expect(screen.getByText(/goes when you uninstall the app/)).toBeTruthy();
    expect(screen.getByText("pilot-ai-command-center / ssh:prod:private-key")).toBeTruthy();
    expect(screen.getByText(/Credential Manager/)).toBeTruthy();
    expect(screen.getAllByText(/Uninstall the Heliox app itself/).length).toBeGreaterThan(0);
    await fireEvent.click(screen.getByRole("button", { name: "Quit Heliox" }));
    expect(invoke).toHaveBeenCalledWith("heliox_quit", undefined);
  });

  it("a bare ok from the app is not taken as a list of what was removed", async () => {
    replies.heliox_files_remove = async () => ({ status: "ok" });
    render(RemoveHelioxFiles, { force: true });
    await choose(false);
    expect(await screen.findByText(/did not say what it removed/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Quit Heliox" })).toBeTruthy();
  });

  it("a refusal removes nothing and says so", async () => {
    replies.heliox_files_remove = async () => {
      throw new Error("Heliox cannot tell which folder is your home folder, so it lists and removes nothing.");
    };
    render(RemoveHelioxFiles, { force: true });
    await choose(true);
    expect(
      await screen.findByText(/Nothing was removed: Heliox cannot tell which folder is your home folder/),
    ).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Heliox's files were removed" })).toBeNull();
  });

  it("no reply from the app is said plainly, for the list and for the removal", async () => {
    replies.heliox_files_plan = () => new Promise(() => {});
    const first = render(RemoveHelioxFiles, { force: true, replyTimeoutMs: 20 });
    await fireEvent.click(screen.getByRole("button", { name: "Show what Heliox stores" }));
    expect(await screen.findByText(/did not answer. Nothing was removed/)).toBeTruthy();
    first.unmount();

    replies.heliox_files_plan = async () => plan;
    replies.heliox_files_remove = () => new Promise(() => {});
    render(RemoveHelioxFiles, { force: true, replyTimeoutMs: 20 });
    await choose(false);
    expect(await screen.findByText(/did not say what it removed/)).toBeTruthy();
  });

  it("an unknown home folder lists nothing", async () => {
    replies.heliox_files_plan = async () => ({
      home_known: false,
      message: "Heliox cannot tell which folder is your home folder, so it lists and removes nothing.",
      locations: [],
    });
    render(RemoveHelioxFiles, { force: true });
    await fireEvent.click(screen.getByRole("button", { name: "Show what Heliox stores" }));
    expect(await screen.findByText(/cannot tell which folder is your home folder/)).toBeTruthy();
    expect(screen.queryByRole("radio")).toBeNull();
  });

  it("renders nothing outside the desktop app", () => {
    const { container } = render(RemoveHelioxFiles);
    expect(container.textContent?.trim()).toBe("");
  });
});
