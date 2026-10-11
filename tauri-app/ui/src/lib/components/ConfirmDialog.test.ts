import { fireEvent, render, screen } from "@testing-library/svelte";
import { describe, expect, it, vi } from "vitest";
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
