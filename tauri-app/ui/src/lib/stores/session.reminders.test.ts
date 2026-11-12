import { get, writable } from "svelte/store";
import { beforeEach, describe, expect, it, vi } from "vitest";

type NotificationHandler = (method: string, params: unknown) => void;
let notify: NotificationHandler | null = null;

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", () => ({
  call: daemonMocks.call,
  connect: vi.fn().mockResolvedValue(true),
  isConnected: vi.fn(() => true),
  onConnectionState: vi.fn(),
  onNotification: (handler: NotificationHandler) => {
    notify = handler;
  },
}));

vi.mock("./settings", () => ({ settings: writable({ model: { cloud_model: "ollama" } }) }));
vi.mock("./companion", () => ({ companion: { speak: vi.fn() } }));
vi.mock("@tauri-apps/plugin-notification", () => ({
  isPermissionGranted: vi.fn().mockResolvedValue(false),
  requestPermission: vi.fn().mockResolvedValue("denied"),
  sendNotification: vi.fn(),
}));

const MISSED = {
  message: "While this computer was asleep, two reminders came due: call mum and stretch.",
  reminders: [
    { id: "a", text: "call mum", due: 1_800_000_000, repeat: false },
    { id: "b", text: "stretch", due: 1_800_003_600, repeat: false },
  ],
};

describe("reminders that came due while away", () => {
  beforeEach(() => {
    localStorage.clear();
    notify = null;
    daemonMocks.call.mockReset();
    vi.resetModules();
  });

  it("make one card, and the same notification twice adds nothing", async () => {
    const { session } = await import("./session");
    notify!("reminders_missed", MISSED);
    notify!("reminders_missed", MISSED);

    expect(get(session).missedReminders?.reminders.map((item) => item.id)).toEqual(["a", "b"]);
    expect(get(session).missedReminders?.message).toBe(MISSED.message);
  });

  it("leave the card one by one with Done, and the card goes when empty", async () => {
    const { session } = await import("./session");
    notify!("reminders_missed", MISSED);

    session.dismissMissedReminder("a");
    expect(get(session).missedReminders?.reminders.map((item) => item.id)).toEqual(["b"]);
    session.dismissMissedReminder("b");
    expect(get(session).missedReminders).toBeNull();
  });

  it("set the reminder again in ten minutes with Remind me again", async () => {
    daemonMocks.call.mockResolvedValue({ status: "ok" });
    const { session } = await import("./session");
    notify!("reminders_missed", MISSED);

    await session.remindAgain("b");

    expect(daemonMocks.call).toHaveBeenCalledWith("reminder_add", { text: "stretch", in_minutes: 10 });
    expect(get(session).missedReminders?.reminders.map((item) => item.id)).toEqual(["a"]);
  });

  it("stay on the card when setting it again failed, and say why", async () => {
    daemonMocks.call.mockResolvedValue({ status: "error", message: "Reminders are not ready yet." });
    const { session } = await import("./session");
    notify!("reminders_missed", MISSED);

    await session.remindAgain("a");

    expect(get(session).missedReminders?.reminders.map((item) => item.id)).toEqual(["a", "b"]);
    expect(get(session).messages.at(-1)?.text).toBe("Reminders are not ready yet.");
  });

  it("say when each was due", async () => {
    const { describeMissedDue } = await import("./session");
    const now = new Date(2026, 10, 12, 18, 0);
    expect(describeMissedDue(new Date(2026, 10, 12, 9, 30).getTime() / 1000, now)).toMatch(/^Due at /);
    expect(describeMissedDue(new Date(2026, 10, 10, 9, 30).getTime() / 1000, now)).toMatch(/^Due \w+/);
  });
});
