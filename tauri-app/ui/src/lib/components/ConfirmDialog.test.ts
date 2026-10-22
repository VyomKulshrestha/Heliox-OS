import { fireEvent, render, screen, waitFor } from "@testing-library/svelte";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../i18n";
import ConfirmDialog from "./ConfirmDialog.svelte";

const script = {
  action_type: "shell_script",
  target: "powershell",
  requires_root: false,
  destructive: false,
  irreversible: true,
  index: 2,
  parameters: { script: "Remove-Item C:\\Temp\\old -Recurse", interpreter: "powershell" },
};

describe("ConfirmDialog", () => {
  it("shows the script that will run before it is approved", async () => {
    const onconfirm = vi.fn();
    render(ConfirmDialog, { actions: [script], onconfirm, ondeny: vi.fn() });

    expect(screen.getByText("Script (powershell)")).toBeTruthy();
    expect(screen.getByText("Remove-Item C:\\Temp\\old -Recurse")).toBeTruthy();

    await fireEvent.click(screen.getByRole("button", { name: /approve/i }));
    expect(onconfirm).toHaveBeenCalledWith([2]);
  });

  it("leaves actions without parameters as a single line", () => {
    render(ConfirmDialog, {
      actions: [{ ...script, action_type: "screenshot", parameters: {} }],
      onconfirm: vi.fn(),
      ondeny: vi.fn(),
    });

    expect(screen.queryByText("Script (powershell)")).toBeNull();
    expect(screen.getByText("screenshot")).toBeTruthy();
  });
});

describe("ConfirmDialog review notes", () => {
  it("shows why an autonomous plan needs approval and what Gnomon said", () => {
    render(ConfirmDialog, {
      actions: [{ ...script, action_type: "keyboard_type", parameters: { text: "hello team" } }],
      reason: "Gnomon judged this may go beyond what you asked (4% in scope).",
      gnomon: { escalate: true, reasons: ["Gnomon judged this may go beyond what you asked (4% in scope)."] },
      onconfirm: vi.fn(),
      ondeny: vi.fn(),
    });

    expect(screen.getByText("Gnomon review")).toBeTruthy();
    expect(screen.getAllByText("Gnomon judged this may go beyond what you asked (4% in scope).")).toHaveLength(2);
    expect(screen.getByText("Advisory: Gnomon can ask for approval but never grants it.")).toBeTruthy();
  });

  it("keeps the world-model box for world-model escalations and hides Gnomon when it did not escalate", () => {
    render(ConfirmDialog, {
      actions: [script],
      reason: "The world model predicts 90% risk.",
      riskAssessment: { world_model_score: 0.9, prediction_sources: ["structured_transition"] },
      gnomon: { escalate: false, reasons: [] },
      onconfirm: vi.fn(),
      ondeny: vi.fn(),
    });

    expect(screen.getByText(/90% predicted risk/)).toBeTruthy();
    expect(screen.queryByText("Gnomon review")).toBeNull();
  });
});

describe("ConfirmDialog keyboard and selection safety", () => {
  const second = { ...script, action_type: "file_write", target: "C:\\Temp\\notes.txt", index: 5, parameters: {} };
  let clock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    clock = vi.spyOn(performance, "now").mockReturnValue(1_000);
  });

  afterEach(() => {
    clock.mockRestore();
  });

  function renderDialog(overrides: Record<string, unknown> = {}) {
    const onconfirm = vi.fn();
    const ondeny = vi.fn();
    const onselectionchange = vi.fn();
    render(ConfirmDialog, { actions: [script, second], onconfirm, ondeny, onselectionchange, ...overrides });
    return { onconfirm, ondeny, onselectionchange, dialog: screen.getByRole("dialog") };
  }

  it("is a modal dialog labelled by its title", () => {
    const { dialog } = renderDialog();
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(screen.getByRole("dialog", { name: "Confirmation Required" })).toBe(dialog);
  });

  it("puts the initial focus on Deny", async () => {
    renderDialog();
    const deny = screen.getByRole("button", { name: "Deny" });
    await waitFor(() => expect(document.activeElement).toBe(deny));
  });

  it.each([
    ["the chat box", () => Object.assign(document.createElement("input"), { type: "text" })],
    ["a search field", () => Object.assign(document.createElement("input"), { type: "search" })],
    ["a text area", () => document.createElement("textarea")],
  ])("keeps focus off Deny when the approval arrives while someone types in %s", async (_label, makeField) => {
    const field = makeField();
    document.body.appendChild(field);
    try {
      field.focus();
      expect(document.activeElement).toBe(field);
      const { dialog, ondeny, onconfirm } = renderDialog();
      const deny = screen.getByRole("button", { name: "Deny" });

      // The next Space or Enter, long after the 300 ms guard, lands on the dialog box, which no key activates.
      await waitFor(() => expect(document.activeElement).toBe(dialog));
      expect(document.activeElement).not.toBe(deny);
      clock.mockReturnValue(5_000);
      await fireEvent.keyDown(dialog, { key: " " });
      await fireEvent.keyUp(dialog, { key: " " });
      await fireEvent.keyDown(dialog, { key: "Enter" });
      expect(ondeny).not.toHaveBeenCalled();
      expect(onconfirm).not.toHaveBeenCalled();

      // Tab reaches the controls, and Escape still denies.
      expect(await fireEvent.keyDown(dialog, { key: "Tab" })).toBe(false);
      expect(document.activeElement).toBe(screen.getAllByRole("checkbox")[0]);
      await fireEvent.keyDown(dialog, { key: "Escape" });
      expect(ondeny).toHaveBeenCalledTimes(1);
    } finally {
      field.remove();
    }
  });

  it("still puts the initial focus on Deny when focus was on a control that takes no typing", async () => {
    const button = document.createElement("button");
    const checkbox = Object.assign(document.createElement("input"), { type: "checkbox" });
    document.body.append(button, checkbox);
    try {
      for (const control of [button, checkbox]) {
        control.focus();
        const { unmount } = render(ConfirmDialog, { actions: [script], onconfirm: vi.fn(), ondeny: vi.fn() });
        const deny = screen.getByRole("button", { name: "Deny" });
        await waitFor(() => expect(document.activeElement).toBe(deny));
        unmount();
      }
    } finally {
      button.remove();
      checkbox.remove();
    }
  });

  it("denies on Escape once the first 300 ms have passed", async () => {
    const { dialog, ondeny, onconfirm } = renderDialog();
    clock.mockReturnValue(1_300);
    await fireEvent.keyDown(dialog, { key: "Escape" });
    expect(ondeny).toHaveBeenCalledTimes(1);
    expect(onconfirm).not.toHaveBeenCalled();
  });

  it("ignores every key in the first 300 ms after it mounts", async () => {
    const { dialog, ondeny } = renderDialog();
    clock.mockReturnValue(1_299);
    const deny = screen.getByRole("button", { name: "Deny" });
    expect(await fireEvent.keyDown(dialog, { key: "Escape" })).toBe(false);
    // A prevented keydown is how Enter and Space are kept from pressing the focused button.
    expect(await fireEvent.keyDown(deny, { key: "Enter" })).toBe(false);
    expect(await fireEvent.keyDown(deny, { key: " " })).toBe(false);
    expect(ondeny).not.toHaveBeenCalled();
  });

  it("ignores repeated key events, such as a key held down from the chat box", async () => {
    const { dialog, ondeny } = renderDialog();
    clock.mockReturnValue(5_000);
    const deny = screen.getByRole("button", { name: "Deny" });
    expect(await fireEvent.keyDown(dialog, { key: "Escape", repeat: true })).toBe(false);
    expect(await fireEvent.keyDown(deny, { key: "Enter", repeat: true })).toBe(false);
    expect(ondeny).not.toHaveBeenCalled();
  });

  it("swallows a keyup whose keydown it never accepted", async () => {
    renderDialog();
    clock.mockReturnValue(5_000);
    const deny = screen.getByRole("button", { name: "Deny" });
    expect(await fireEvent.keyUp(deny, { key: " " })).toBe(false);
    expect(await fireEvent.keyDown(deny, { key: " " })).toBe(true);
    expect(await fireEvent.keyUp(deny, { key: " " })).toBe(true);
  });

  it("leaves Enter and Space to the focused button after the guard window", async () => {
    renderDialog();
    clock.mockReturnValue(5_000);
    const deny = screen.getByRole("button", { name: "Deny" });
    expect(await fireEvent.keyDown(deny, { key: "Enter" })).toBe(true);
  });

  it("traps Tab and Shift+Tab inside the dialog", async () => {
    const { dialog } = renderDialog();
    clock.mockReturnValue(5_000);
    const checkboxes = screen.getAllByRole("checkbox");
    const approve = screen.getByRole("button", { name: /approve/i });
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "Deny" })));

    approve.focus();
    expect(await fireEvent.keyDown(approve, { key: "Tab" })).toBe(false);
    expect(document.activeElement).toBe(checkboxes[0]);

    expect(await fireEvent.keyDown(checkboxes[0], { key: "Tab", shiftKey: true })).toBe(false);
    expect(document.activeElement).toBe(approve);

    // Inside the cycle Tab keeps its native behaviour.
    checkboxes[0].focus();
    expect(await fireEvent.keyDown(checkboxes[0], { key: "Tab" })).toBe(true);

    // Focus that left the dialog is brought back to its first control.
    (document.activeElement as HTMLElement | null)?.blur();
    document.body.focus();
    expect(await fireEvent.keyDown(dialog, { key: "Tab" })).toBe(false);
    expect(document.activeElement).toBe(checkboxes[0]);

    // Focus on the dialog box itself (after a click on its text): Shift+Tab goes to the last control, not out.
    dialog.focus();
    expect(document.activeElement).toBe(dialog);
    expect(await fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true })).toBe(false);
    expect(document.activeElement).toBe(approve);
  });

  it("reports the checked indices on mount and on every toggle", async () => {
    const { onselectionchange, onconfirm } = renderDialog();
    expect(onselectionchange).toHaveBeenLastCalledWith([2, 5]);

    const checkboxes = screen.getAllByRole("checkbox");
    await fireEvent.click(checkboxes[0]);
    expect(onselectionchange).toHaveBeenLastCalledWith([5]);
    await fireEvent.click(checkboxes[1]);
    expect(onselectionchange).toHaveBeenLastCalledWith([]);
    await fireEvent.click(checkboxes[0]);
    expect(onselectionchange).toHaveBeenLastCalledWith([2]);

    await fireEvent.click(screen.getByRole("button", { name: /approve/i }));
    expect(onconfirm).toHaveBeenCalledWith([2]);
  });

  it("keeps the buttons, labels and checkbox defaults", () => {
    renderDialog();
    const deny = screen.getByRole("button", { name: "Deny" });
    const approve = screen.getByRole("button", { name: "Approve & Execute" });
    expect(deny.className).toContain("btn-deny");
    expect(approve.className).toContain("btn-confirm");
    expect(approve.hasAttribute("disabled")).toBe(false);
    expect(screen.getAllByRole("checkbox").every((box) => (box as HTMLInputElement).checked)).toBe(true);
  });

  it("does not change how an approval is presented: no presentation, scaling or hold prop", () => {
    const source = readFileSync(resolve(process.cwd(), "src/lib/components/ConfirmDialog.svelte"), "utf-8");
    const props = source.match(/interface Props \{([\s\S]*?)\n  \}/)?.[1] ?? "";
    const names = [...props.matchAll(/^\s+(\w+)\??:/gm)].map((match) => match[1]);
    expect(names).toEqual([
      "actions",
      "onconfirm",
      "ondeny",
      "onselectionchange",
      "reason",
      "riskAssessment",
      "gnomon",
      "submitting",
      "error",
    ]);
    const style = source.slice(source.indexOf("<style>"));
    const rule = (selector: string) => style.match(new RegExp(`\\n  ${selector} \\{([^}]*)\\}`))?.[1] ?? "";
    for (const selector of ["\\.btn-deny", "\\.btn-confirm"]) {
      expect(rule(selector)).toContain("padding: 8px 20px;");
      expect(rule(selector)).toContain("font-size: 13px;");
    }
    expect(rule("\\.confirm-checkbox input")).toContain("width: 15px;");
  });
});

describe("ConfirmDialog undo labels", () => {
  it("labels each action by whether Heliox can undo it", () => {
    render(ConfirmDialog, {
      actions: [
        {
          ...script,
          action_type: "file_write",
          irreversible: false,
          index: 0,
          undo: { kind: "undoable", reason: "Heliox keeps what it needs to put this back." },
        },
        {
          ...script,
          action_type: "api_send_email",
          irreversible: false,
          index: 1,
          undo: { kind: "not_undoable", reason: "Heliox cannot take it back." },
        },
      ],
      onconfirm: vi.fn(),
      ondeny: vi.fn(),
    });

    expect(screen.getByText("Can undo").getAttribute("title")).toBe("Heliox keeps what it needs to put this back.");
    expect(screen.getByText("Can't undo").getAttribute("title")).toBe("Heliox cannot take it back.");
  });
});
