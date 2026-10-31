import { expect, test } from "@playwright/test";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createConnection } from "node:net";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { delimiter, resolve } from "node:path";

const UI_TOKEN = "playwright-ui-token";
const NEURAL_TOKEN = "playwright-neural-token";
const MCP_TOKEN = "playwright-mcp-token";
const WS_PORT = 8785;

type RpcResponse = { id?: string; result?: Record<string, unknown>; error?: { code: number; message: string } };

async function waitForPort(open: boolean, timeoutMs = 15_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const connected = await new Promise<boolean>((done) => {
      const socket = createConnection({ host: "127.0.0.1", port: WS_PORT });
      socket.once("connect", () => {
        socket.destroy();
        done(true);
      });
      socket.once("error", () => done(false));
      socket.setTimeout(300, () => {
        socket.destroy();
        done(false);
      });
    });
    if (connected === open) return;
    await new Promise((done) => setTimeout(done, 50));
  }
  throw new Error(`Port ${WS_PORT} did not become ${open ? "ready" : "closed"}`);
}

async function waitForEvidence(path: string, timeoutMs = 15_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(path) && readFileSync(path, "utf-8").includes('"event": "ready"')) return;
    await new Promise((done) => setTimeout(done, 50));
  }
  throw new Error("Controlled smoke daemon did not publish its ready evidence");
}

async function waitForExit(process: ChildProcessWithoutNullStreams, timeoutMs = 30_000): Promise<number | null> {
  if (process.exitCode !== null) return process.exitCode;
  return await Promise.race([
    new Promise<number | null>((done) => process.once("exit", done)),
    new Promise<never>((_done, reject) =>
      setTimeout(() => reject(new Error("Controlled smoke daemon did not exit")), timeoutMs),
    ),
  ]);
}

test("one UI approval produces exactly one same-WebSocket local navigation", async ({ page }, testInfo) => {
  const repoRoot = resolve(import.meta.dirname, "../../../..");
  const daemonRoot = resolve(repoRoot, "daemon");
  const artifactDir = process.env.HELIOX_APPROVAL_ARTIFACT_DIR
    ? resolve(process.env.HELIOX_APPROVAL_ARTIFACT_DIR)
    : testInfo.outputPath("approval-evidence");
  mkdirSync(artifactDir, { recursive: true });
  const evidencePath = resolve(artifactDir, "approval-trace.jsonl");
  const targetScreenshot = resolve(artifactDir, "approved-navigation.png");
  const uiScreenshot = resolve(artifactDir, "approval-ui.png");
  const summaryPath = resolve(artifactDir, "claim-boundary.md");
  const python = process.env.HELIOX_PYTHON || process.env.PYTHON || "python";
  const script = resolve(daemonRoot, "scripts/confirmation_ui_smoke_server.py");
  const pythonPath = [daemonRoot, process.env.PYTHONPATH].filter(Boolean).join(delimiter);
  const smoke = spawn(
    python,
    [
      script,
      "--evidence",
      evidencePath,
      "--screenshot",
      targetScreenshot,
      "--token",
      UI_TOKEN,
      "--neural-token",
      NEURAL_TOKEN,
      "--mcp-token",
      MCP_TOKEN,
      "--target-url",
      "http://127.0.0.1:1420/approval-smoke.html",
    ],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        PYTHONPATH: pythonPath,
        HELIOX_BROWSER_HEADLESS: process.env.CI ? "1" : process.env.HELIOX_BROWSER_HEADLESS,
      },
    },
  );
  let smokeOutput = "";
  smoke.stdout.on("data", (chunk) => (smokeOutput += chunk.toString()));
  smoke.stderr.on("data", (chunk) => (smokeOutput += chunk.toString()));

  const rpc = async (token: string, method: string, params: Record<string, unknown> = {}) =>
    await page.evaluate(
      async ({ token, method, params }) => {
        return await new Promise<RpcResponse>((done, reject) => {
          const socket = new WebSocket("ws://127.0.0.1:8785");
          const timeout = window.setTimeout(() => reject(new Error(`RPC ${method} timed out`)), 5_000);
          let authenticated = false;
          socket.onopen = () =>
            socket.send(JSON.stringify({ jsonrpc: "2.0", method: "auth", params: { token }, id: "auth" }));
          socket.onmessage = (event) => {
            const message = JSON.parse(String(event.data)) as RpcResponse;
            if (!authenticated && message.id === "auth") {
              authenticated = true;
              socket.send(JSON.stringify({ jsonrpc: "2.0", method, params, id: "call" }));
              return;
            }
            if (message.id === "call") {
              window.clearTimeout(timeout);
              socket.close();
              done(message);
            }
          };
          socket.onerror = () => reject(new Error(`RPC ${method} socket failed`));
        });
      },
      { token, method, params },
    );

  let gracefulCleanup = false;
  try {
    await waitForEvidence(evidencePath);
    await page.route("**/*", async (route) => {
      const url = new URL(route.request().url());
      if (url.pathname === "/api/auth_token") {
        await route.fulfill({ status: 200, contentType: "text/plain", body: UI_TOKEN });
      } else if (url.hostname === "127.0.0.1" || url.hostname === "localhost") await route.continue();
      else await route.abort("blockedbyclient");
    });
    await page.addInitScript(() => {
      localStorage.setItem("heliox_first_run_complete", "true");
      localStorage.setItem(
        "heliox_settings",
        JSON.stringify({
          first_run_complete: true,
          theme: "dark",
          model: { provider: "ollama", ollama_model: "llama3.1:8b", mode: "lightweight" },
          security: { root_enabled: false, dry_run: false, snapshot_on_destructive: true },
          screen_vision: { capture_interval_seconds: 3 },
          restrictions: { protected_folders: [], protected_packages: [], blocked_commands: [] },
        }),
      );
    });
    await page.goto("/");
    await expect(page.locator(".window")).toBeVisible();
    await page.locator(".command-input textarea").fill("Open the controlled Antler approval page");
    await page.locator(".command-input").press("Enter");

    const dialog = page.locator(".confirm-dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText("browser_navigate");
    await expect(dialog).toContainText("http://127.0.0.1:1420/approval-smoke.html");
    await dialog.getByRole("button", { name: "Approve" }).click();
    await expect(dialog).toBeHidden();
    await expect(page.getByText(/APPROVED ui-smoke-1: browser command executed/)).toBeVisible({ timeout: 30_000 });
    await page.locator(".window").screenshot({ path: uiScreenshot });

    const firstStatus = await rpc(UI_TOKEN, "smoke_status");
    await page.waitForTimeout(250);
    const stableStatus = await rpc(UI_TOKEN, "smoke_status");
    expect(firstStatus.result).toMatchObject({
      browser_execute_count: 1,
      confirmation_count: 1,
      same_websocket: true,
    });
    expect(stableStatus.result).toEqual(firstStatus.result);

    const neuralConfirm = await rpc(NEURAL_TOKEN, "confirm", { plan_id: "ui-smoke-1", confirmed: true });
    const mcpConfirm = await rpc(MCP_TOKEN, "confirm", { plan_id: "ui-smoke-1", confirmed: true });
    expect(neuralConfirm.error).toMatchObject({ code: -32601 });
    expect(mcpConfirm.error).toMatchObject({ code: -32601 });

    expect(existsSync(targetScreenshot)).toBe(true);
    const trace = readFileSync(evidencePath, "utf-8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line));
    const approved = trace.filter((entry) => entry.decision === "approved");
    expect(approved).toHaveLength(1);
    expect(approved[0]).toMatchObject({
      browser_execute_count: 1,
      confirmation_count: 1,
      executed: true,
      same_websocket: true,
    });

    const shutdown = await rpc(UI_TOKEN, "smoke_shutdown");
    expect(shutdown.result).toEqual({ status: "stopping" });
    expect(await waitForExit(smoke)).toBe(0);
    await waitForPort(false);
    gracefulCleanup = true;

    const finalTrace = readFileSync(evidencePath, "utf-8")
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line));
    expect(finalTrace.at(-1)).toMatchObject({ event: "cleanup", browser_closed: true, browser_execute_count: 1 });

    writeFileSync(
      summaryPath,
      `# Heliox Antler demo claim boundary\n\n` +
        `## Verified in this run\n\n` +
        `- One UI-authenticated approval and one browser navigation used the same WebSocket.\n` +
        `- The navigation count remained exactly 1 and the target was served only from loopback.\n` +
        `- Neural-sidecar and MCP-local identities were denied the confirmation RPC.\n` +
        `- The controlled browser and smoke daemon closed cleanly.\n` +
        `- Evidence: \`${evidencePath}\`, \`${uiScreenshot}\`, \`${targetScreenshot}\`.\n\n` +
        `## Not established by this demo\n\n` +
        `- No external-site, production deployment, physical-device, human-study, or unrestricted autonomy claim.\n` +
        `- This is deterministic local software evidence for the approval-to-navigation path only.\n`,
      "utf-8",
    );
  } finally {
    if (!gracefulCleanup && smoke.exitCode === null) {
      try {
        await rpc(UI_TOKEN, "smoke_shutdown");
        await waitForExit(smoke);
        await waitForPort(false);
      } catch {
        smoke.kill();
        await waitForExit(smoke).catch(() => null);
      }
    }
    if (!gracefulCleanup && smoke.exitCode !== 0) {
      await testInfo.attach("smoke-daemon-output", { body: smokeOutput, contentType: "text/plain" });
    }
  }
});
