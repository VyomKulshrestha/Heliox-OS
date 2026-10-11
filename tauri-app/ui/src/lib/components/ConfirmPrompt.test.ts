import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import ConfirmPrompt from "./ConfirmPrompt.svelte";

afterEach(cleanup);

function props(overrides: Record<string, unknown> = {}) {
  return { message: "Delete the schedule?", onconfirm: vi.fn(), oncancel: vi.fn(), ...overrides };
}

describe("ConfirmPrompt", () => {
  it("is a modal dialog described by its message", () => {
    render(ConfirmPrompt, props({ confirmLabel: "Delete schedule" }));

    const dialog = screen.getByRole("dialog", { name: "Delete schedule" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    const description = document.getElementById(dialog.getAttribute("aria-describedby") ?? "");
    expect(description?.textContent).toBe("Delete the schedule?");
  });

  it("uses alertdialog for a dangerous action and starts on Cancel", () => {
    render(ConfirmPrompt, props({ danger: true, confirmLabel: "Delete schedule" }));

    const dialog = screen.getByRole("alertdialog", { name: "Delete schedule" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Cancel" }));
  });

  it("starts on the confirm button for an ordinary prompt", () => {
    render(ConfirmPrompt, props({ confirmLabel: "Save" }));

    expect(document.activeElement).toBe(screen.getByRole("button", { name: "Save" }));
  });

  it("cancels on Escape", async () => {
    const options = props();
    render(ConfirmPrompt, options);

    await fireEvent.keyDown(screen.getByRole("button", { name: "Confirm" }), { key: "Escape" });

    expect(options.oncancel).toHaveBeenCalledTimes(1);
    expect(options.onconfirm).not.toHaveBeenCalled();
  });

  it("keeps Tab and Shift+Tab inside the dialog", async () => {
    render(ConfirmPrompt, props());
    const cancel = screen.getByRole("button", { name: "Cancel" });
    const confirm = screen.getByRole("button", { name: "Confirm" });

    expect(document.activeElement).toBe(confirm);
    await fireEvent.keyDown(confirm, { key: "Tab" });
    expect(document.activeElement).toBe(cancel);
    await fireEvent.keyDown(cancel, { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(confirm);
  });

  it("gives focus back to the element that opened it", () => {
    const opener = document.createElement("button");
    opener.textContent = "Delete";
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = render(ConfirmPrompt, props({ danger: true }));
    expect(document.activeElement).not.toBe(opener);
    unmount();

    expect(document.activeElement).toBe(opener);
    opener.remove();
  });
});
