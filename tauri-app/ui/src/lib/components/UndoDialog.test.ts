import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { describe, expect, it, vi } from "vitest";

const daemonMocks = vi.hoisted(() => ({ call: vi.fn() }));

vi.mock("../api/daemon", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/daemon")>();
  return { ...actual, call: daemonMocks.call };
});
import "../../i18n";
import UndoDialog from "./UndoDialog.svelte";
import type { UndoPreview } from "../stores/session";

const preview: UndoPreview = {
  task_key: "task-7",
  undone: false,
  ready: 1,
  preview_token: "token-1",
  items: [
    { seq: 2, type: "file_write", target: "notes.txt", state: "ready", description: "Remove notes.txt." },
    {
      seq: 1,
      type: "file_write",
      target: "plan.md",
      state: "conflict",
      description: "plan.md changed after Heliox wrote it, so it is left as it is.",
    },
  ],
  not_undoable: [{ type: "api_send_email", target: "team@example.com", reason: "Heliox cannot take it back." }],
};

describe("UndoDialog", () => {
  it("separates what will be undone, what is left alone and what cannot be undone", () => {
    render(UndoDialog, { preview, onconfirm: vi.fn(), oncancel: vi.fn() });

    expect(screen.getByTestId("undo-ready").textContent).toContain("Remove notes.txt.");
    expect(screen.getByTestId("undo-left-alone").textContent).toContain("plan.md changed after Heliox wrote it");
    expect(screen.getByTestId("undo-not-undoable").textContent).toContain("team@example.com");
  });

  it("confirms with one click and cannot confirm when nothing is ready", async () => {
    const onconfirm = vi.fn();
    const { unmount } = render(UndoDialog, { preview, onconfirm, oncancel: vi.fn() });

    await fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(onconfirm).toHaveBeenCalledOnce();
    unmount();

    render(UndoDialog, {
      preview: { ...preview, ready: 0, items: [preview.items[1]] },
      onconfirm: vi.fn(),
      oncancel: vi.fn(),
    });
    expect((screen.getByRole("button", { name: "Undo" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText("Nothing from this task can be undone now.")).toBeTruthy();
  });
});

describe("UndoDialog changes", () => {
  it("shows each file's line changes on request", async () => {
    daemonMocks.call.mockResolvedValue({
      status: "ok",
      changes: [{ seq: 2, path: "C:/notes.txt", created: false, diff: "--- before\n+++ now\n-eggs\n+bread" }],
    });
    render(UndoDialog, { preview, onconfirm: vi.fn(), oncancel: vi.fn() });

    await fireEvent.click(screen.getByRole("button", { name: "Show what changed in each file" }));

    await waitFor(() => expect(screen.getByTestId("undo-changes").textContent).toContain("+bread"));
    expect(daemonMocks.call).toHaveBeenCalledWith("undo_diff", { task_key: "task-7" });
    expect(screen.getByText("-eggs").className).toContain("removed");
  });
});
