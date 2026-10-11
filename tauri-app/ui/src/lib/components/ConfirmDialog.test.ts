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
