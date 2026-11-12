import { render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../i18n";

type SessionSlice = Record<string, unknown>;
const sessionState = vi.hoisted(() => ({
  store: null as unknown as import("svelte/store").Writable<Record<string, unknown>>,
}));

vi.mock("../stores/session", async () => {
  const { writable } = await import("svelte/store");
  sessionState.store = writable({ phase: "", liveActions: [], streamingText: "" });
  return { session: sessionState.store };
});

import RunStatus from "./RunStatus.svelte";

function setSession(patch: Record<string, unknown>) {
  sessionState.store.update((current: SessionSlice) => ({ ...current, ...patch }));
}

const action = (action_type: string) => ({
  action_type,
  target: "",
  requires_root: false,
  destructive: false,
  parameters: {},
});

describe("RunStatus", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    sessionState.store.set({ phase: "", liveActions: [], streamingText: "" });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("names the phase in plain words and marks the current stage", async () => {
    setSession({ phase: "recalling memory" });
    render(RunStatus);

    expect(screen.getByRole("status").textContent).toBe("Recalling what's relevant");
    expect(screen.getByText("Understand").closest("li")?.getAttribute("aria-current")).toBe("step");

    setSession({ phase: "verifying" });
    await vi.advanceTimersByTimeAsync(0);
    expect(screen.getByRole("status").textContent).toBe("Checking the result");
    expect(screen.getByText("Check").closest("li")?.getAttribute("aria-current")).toBe("step");
    expect(screen.getByText("Plan").closest("li")?.classList.contains("done")).toBe(true);
  });

  it("counts plan steps and shows the elapsed time after a few seconds", async () => {
    setSession({
      phase: "executing",
      liveActions: [
        { index: 0, action: action("open_application"), status: "success" },
        { index: 1, action: action("browser_navigate"), status: "running" },
        { index: 2, action: action("screen_ocr"), status: "pending" },
      ],
    });
    render(RunStatus);

    expect(screen.getByText("Step 2 of 3")).toBeTruthy();
    expect(screen.queryByText("0:02")).toBeNull();
    await vi.advanceTimersByTimeAsync(4000);
    expect(screen.getByText("0:04")).toBeTruthy();
  });

  it("shows streamed answer text and the waiting state for an approval", async () => {
    setSession({ phase: "awaiting visible approval", streamingText: "It is 31 degrees" });
    const { container } = render(RunStatus);

    expect(screen.getByRole("status").textContent).toBe("Waiting for your approval");
    expect(container.querySelector(".run-status")?.classList.contains("waiting")).toBe(true);
    expect(screen.getByText("It is 31 degrees")).toBeTruthy();
  });

  it("renders a streamed reply as markdown while it arrives", async () => {
    const fence = "`".repeat(3);
    setSession({
      phase: "generating",
      streamingText: `Two ways:\n- **merge** keeps history\n${fence}bash\ngit rebase main`,
    });
    const { container } = render(RunStatus);
    const stream = container.querySelector(".stream");
    expect(stream?.querySelector("li strong")?.textContent).toBe("merge");
    expect(stream?.querySelector("pre code")?.textContent).toContain("git rebase main");
    expect(stream?.textContent).not.toContain(fence);
    expect(stream?.querySelector(".caret")).toBeTruthy();
  });
});
