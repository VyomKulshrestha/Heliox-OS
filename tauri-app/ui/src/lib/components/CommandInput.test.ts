import { fireEvent, render, screen } from "@testing-library/svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  store: null as unknown as import("svelte/store").Writable<Record<string, unknown>>,
  sendCommand: vi.fn(),
  abort: vi.fn(async () => {}),
  cancelQueuedTask: vi.fn(),
}));

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }),
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));
vi.mock("../stores/session", async () => {
  const { writable } = await import("svelte/store");
  mocks.store = writable({ loading: false, queuedTasks: [], activeSessionId: "chat-1" });
  return {
    session: {
      subscribe: mocks.store.subscribe,
      sendCommand: mocks.sendCommand,
      abort: mocks.abort,
      cancelQueuedTask: mocks.cancelQueuedTask,
    },
  };
});

import CommandInput from "./CommandInput.svelte";

describe("CommandInput", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.store.set({ loading: false, queuedTasks: [], activeSessionId: "chat-1" });
  });

  it("sends on Enter, keeps Shift+Enter for a new line, and hides the counter for short text", async () => {
    render(CommandInput);
    const field = screen.getByRole("textbox", { name: "Tell Heliox what to do..." });

    await fireEvent.input(field, { target: { value: "open chrome" } });
    await fireEvent.keyDown(field, { key: "Enter", shiftKey: true });
    expect(mocks.sendCommand).not.toHaveBeenCalled();
    expect(screen.queryByText(/\/20000/)).toBeNull();

    await fireEvent.keyDown(field, { key: "Enter" });
    expect(mocks.sendCommand).toHaveBeenCalledWith("open chrome", [], "correct");
  });

  it("offers Stop while a task runs and stops it with Esc", async () => {
    render(CommandInput);
    expect(screen.queryByRole("button", { name: "Stop the running task" })).toBeNull();

    mocks.store.set({ loading: true, queuedTasks: [], activeSessionId: "chat-1" });
    await fireEvent.click(await screen.findByRole("button", { name: "Stop the running task" }));
    expect(mocks.abort).toHaveBeenCalledTimes(1);

    await fireEvent.keyDown(window, { key: "Escape" });
    expect(mocks.abort).toHaveBeenCalledTimes(2);
  });

  it("leaves Esc to an open dialog", async () => {
    mocks.store.set({ loading: true, queuedTasks: [], activeSessionId: "chat-1" });
    render(CommandInput);
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    document.body.appendChild(dialog);

    await fireEvent.keyDown(window, { key: "Escape" });
    expect(mocks.abort).not.toHaveBeenCalled();
    dialog.remove();
  });
});
