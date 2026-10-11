import { describe, expect, it } from "vitest";
import { describeAction, maskSecrets } from "./actionDetails";
import type { PlanAction } from "../stores/session";

function action(action_type: string, parameters: Record<string, unknown>, target = ""): PlanAction {
  return { action_type, target, requires_root: false, destructive: false, parameters };
}

describe("describeAction", () => {
  it("shows the whole script and the interpreter for shell scripts", () => {
    const details = describeAction(
      action("shell_script", {
        script: "Get-Process | Sort-Object WS -Descending | Select-Object -First 5",
        interpreter: "powershell",
        elevated: true,
        timeout: 30,
      }),
    );

    expect(details).toEqual([
      {
        label: "Script (powershell)",
        value: "Get-Process | Sort-Object WS -Descending | Select-Object -First 5",
        code: true,
      },
      { label: "Runs as", value: "administrator", code: false },
    ]);
  });

  it("joins a shell command with its arguments", () => {
    const [command] = describeAction(action("shell_command", { command: "git", args: ["push", "--force"] }));

    expect(command).toEqual({ label: "Command", value: "git push --force", code: true });
  });

  it("shows the MCP tool and its arguments with secrets masked", () => {
    const details = describeAction(
      action("mcp_call_tool", {
        mcp_server: "github",
        mcp_tool: "create_issue",
        mcp_arguments: { title: "Crash on save", api_token: "ghp_secret" },
      }),
    );

    expect(details[0]).toEqual({ label: "Tool", value: "github.create_issue", code: false });
    expect(details[1].value).toContain('"title": "Crash on save"');
    expect(details[1].value).not.toContain("ghp_secret");
  });

  it("shows who a message goes to and what it says", () => {
    const details = describeAction(
      action("api_send_email", { to: "priya@example.com", subject: "Invoice", message: "Attached is March." }),
    );

    expect(details.map((detail) => [detail.label, detail.value])).toEqual([
      ["To", "priya@example.com"],
      ["Subject", "Invoice"],
      ["Message", "Attached is March."],
    ]);
  });

  it("falls back to the meaningful parameters and skips settings noise", () => {
    const [detail] = describeAction(action("process_kill", { pid: 4242, name: "", signal: "SIGTERM", timeout: 5 }));

    expect(detail.label).toBe("Parameters");
    expect(JSON.parse(detail.value)).toEqual({ pid: 4242, signal: "SIGTERM" });
    expect(describeAction(action("screenshot", {}))).toEqual([]);
  });

  it("caps very long content", () => {
    const [, content] = describeAction(action("file_write", { path: "C:/a.txt", content: "x".repeat(5000) }));

    expect(content.value.length).toBeLessThan(2100);
    expect(content.value).toContain("3000 more characters");
  });
});

describe("maskSecrets", () => {
  it("masks secret-looking keys at any depth and leaves the rest", () => {
    expect(maskSecrets({ user: "me", nested: { password: "hunter2", list: [{ apiKey: "k" }] } })).toEqual({
      user: "me",
      nested: { password: "••••••", list: [{ apiKey: "••••••" }] },
    });
  });
});
