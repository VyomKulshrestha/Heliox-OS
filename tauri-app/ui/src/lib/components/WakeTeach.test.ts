import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { tick } from "svelte";
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (method: string, params: unknown) => void;
type Reply = Record<string, unknown>;
const mocks = vi.hoisted(() => ({
  handlers: [] as Handler[],
  teach: (async () => ({ status: "ok" })) as () => Promise<Reply>,
  remove: (async () => ({ status: "ok" })) as () => Promise<Reply>,
  variants: [] as { text: string; taught?: boolean }[],
}));

vi.mock("../api/daemon", () => ({
  call: vi.fn(async (method: string) => {
    if (method === "list_wake_variants") return { status: "ok", variants: mocks.variants };
    if (method === "voice_wake_teach") return mocks.teach();
    if (method === "remove_wake_variant") return mocks.remove();
    return { status: "ok" };
  }),
  onNotification: (handler: Handler) => mocks.handlers.push(handler),
  offNotification: vi.fn(),
}));

import WakeTeach from "./WakeTeach.svelte";

function emit(method: string, params: Record<string, unknown>) {
  mocks.handlers.forEach((handler) => handler(method, params));
}

describe("Teach Heliox how you say its name", () => {
  beforeEach(() => {
    mocks.handlers.length = 0;
    mocks.variants = [];
    mocks.teach = async () => ({ status: "ok" });
    mocks.remove = async () => ({ status: "ok" });
  });

  it("guides each try and says what was added", async () => {
    let finish: (value: Reply) => void = () => {};
    mocks.teach = () => new Promise((resolve) => (finish = resolve));
    render(WakeTeach);
    await fireEvent.click(screen.getByRole("button", { name: "Teach" }));
    expect(screen.getByText(/Say “Hey Heliox” on its own/)).toBeTruthy();
    emit("voice_status", { status: "wake_teach", try: 1, of: 4, heard: "Hey Helios." });
    await tick();
    expect(screen.getByText("Heard “Hey Helios.”. Again, please. (2 of 4)")).toBeTruthy();
    mocks.variants = [{ text: "hey helios", taught: true }];
    finish({ status: "ok", added: ["hey helios"], refused: {}, already_heliox: 1 });
    await waitFor(() => expect(screen.getByText("Heliox now also answers to: “hey helios”.")).toBeTruthy());
    expect(screen.getByRole("button", { name: "Remove" })).toBeTruthy(); // listed, removable
  });

  it("says why nothing was added", async () => {
    mocks.teach = async () => ({
      status: "ok",
      added: [],
      refused: { "hey alex": "“hey alex” is made of ordinary words; it would wake Heliox in everyday talk." },
    });
    render(WakeTeach);
    await fireEvent.click(screen.getByRole("button", { name: "Teach" }));
    await waitFor(() => expect(screen.getByText(/Nothing added. “hey alex” is made of ordinary words/)).toBeTruthy());
  });

  it("a refusal from the daemon is said", async () => {
    mocks.teach = async () => ({ status: "error", message: "Turn on always listening first, then try again." });
    render(WakeTeach);
    await fireEvent.click(screen.getByRole("button", { name: "Teach" }));
    await waitFor(() => expect(screen.getByText("Turn on always listening first, then try again.")).toBeTruthy());
    expect((screen.getByRole("button", { name: "Teach" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("no answer from the daemon is said instead of an unhandled rejection", async () => {
    mocks.teach = async () => {
      throw new Error("socket closed");
    };
    render(WakeTeach);
    await fireEvent.click(screen.getByRole("button", { name: "Teach" }));
    await waitFor(() => expect(screen.getByText("Heliox did not answer. Try again in a moment.")).toBeTruthy());
  });

  it("a taught spelling can be removed, and a failed removal says so", async () => {
    mocks.variants = [{ text: "hey helios", taught: true }];
    render(WakeTeach);
    const button = await screen.findByRole("button", { name: "Remove" });
    mocks.remove = async () => {
      throw new Error("socket closed");
    };
    await fireEvent.click(button);
    await waitFor(() =>
      expect(screen.getByText("Heliox did not answer, so “hey helios” was not removed.")).toBeTruthy(),
    );
    mocks.remove = async () => ({ status: "ok" });
    await fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(screen.getByText("Removed “hey helios”.")).toBeTruthy());
    expect(screen.queryByRole("button", { name: "Remove" })).toBeNull();
  });
});
