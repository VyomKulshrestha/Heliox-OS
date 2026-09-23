/// <reference types="vitest" />
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { svelteTesting } from "@testing-library/svelte/vite";
import type { Plugin, ResolvedConfig } from "vite";
import {
  mkdirSync,
  readdirSync,
  copyFileSync,
  existsSync,
  createReadStream,
  readFileSync,
  statfsSync,
  writeFileSync,
} from "node:fs";
import { join, basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";
import { execFile, execSync } from "node:child_process";

const MEDIAPIPE_HANDS_ROUTE = "/mediapipe/hands";
const MEDIAPIPE_HANDS_ASSET_DIR = "mediapipe/hands";
const CONFIG_DIR = dirname(fileURLToPath(import.meta.url));
const UI_VERSION = JSON.parse(readFileSync(join(CONFIG_DIR, "package.json"), "utf-8")).version as string;
const MEDIAPIPE_HANDS_DIR = join(CONFIG_DIR, "node_modules", "@mediapipe", "hands");

const MEDIAPIPE_TASKS_VISION_ROUTE = "/mediapipe/tasks-vision";
const MEDIAPIPE_TASKS_VISION_ASSET_DIR = "mediapipe/tasks-vision";
const MEDIAPIPE_TASKS_VISION_WASM_DIR = join(CONFIG_DIR, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const MEDIAPIPE_TASKS_VISION_MODEL_DIR = join(CONFIG_DIR, "vendor", "mediapipe");

function productionChunk(id: string): string | undefined {
  const normalized = id.replace(/\\/g, "/");
  if (normalized.includes("/src/lib/components/SettingsPanel.svelte")) return "settings-panel";
  if (!normalized.includes("/node_modules/")) return undefined;
  if (normalized.includes("/@mediapipe/")) return "vision-runtime";
  if (normalized.includes("/@tauri-apps/")) return "tauri-runtime";
  if (normalized.includes("/chart.js/") || normalized.includes("/svelte-chartjs/")) return "charts";
  if (normalized.includes("/highlight.js/") || normalized.includes("/marked/") || normalized.includes("/dompurify/")) {
    return "content-rendering";
  }
  if (normalized.includes("/lucide-svelte/")) return "icons";
  if (normalized.includes("/svelte/") || normalized.includes("/svelte-i18n/")) return "svelte-runtime";
  return "vendor";
}

function enforceProductionChunkBudget(maxBytes = 500 * 1024): Plugin {
  return {
    name: "production-chunk-budget",
    apply: "build",
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        const bytes = new TextEncoder().encode(output.code).byteLength;
        if (bytes > maxBytes) {
          this.error(
            `${output.fileName} is ${(bytes / 1024).toFixed(1)} KiB, exceeding the ${(maxBytes / 1024).toFixed(0)} KiB production chunk budget`,
          );
        }
      }
    },
  };
}

function contentType(file: string) {
  if (file.endsWith(".js")) return "text/javascript";
  if (file.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}

function mediapipeHandsAssets(): Plugin {
  let config: ResolvedConfig;

  const assetFiles = () =>
    readdirSync(MEDIAPIPE_HANDS_DIR).filter((file) => /\.(binarypb|data|js|tflite|wasm)$/.test(file));

  return {
    name: "mediapipe-hands-assets",
    configResolved(resolvedConfig) {
      config = resolvedConfig;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = new URL(req.url ?? "", "http://localhost").pathname;
        if (!pathname.startsWith(`${MEDIAPIPE_HANDS_ROUTE}/`)) {
          next();
          return;
        }

        const file = basename(decodeURIComponent(pathname.slice(MEDIAPIPE_HANDS_ROUTE.length + 1)));
        const source = join(MEDIAPIPE_HANDS_DIR, file);
        if (!existsSync(source)) {
          next();
          return;
        }

        res.setHeader("Content-Type", contentType(file));
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
        createReadStream(source).pipe(res);
      });
    },
    writeBundle() {
      const targetDir = join(config.build.outDir, MEDIAPIPE_HANDS_ASSET_DIR);
      mkdirSync(targetDir, { recursive: true });
      for (const file of assetFiles()) {
        copyFileSync(join(MEDIAPIPE_HANDS_DIR, file), join(targetDir, file));
      }
    },
  };
}

function mediapipeTasksVisionAssets(): Plugin {
  let config: ResolvedConfig;

  const sourceDirs = [MEDIAPIPE_TASKS_VISION_WASM_DIR, MEDIAPIPE_TASKS_VISION_MODEL_DIR];

  const assetFiles = () =>
    sourceDirs.flatMap((dir) =>
      readdirSync(dir)
        .filter((file) => /\.(js|wasm|task)$/.test(file))
        .map((file) => ({ dir, file })),
    );

  return {
    name: "mediapipe-tasks-vision-assets",
    configResolved(resolvedConfig) {
      config = resolvedConfig;
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = new URL(req.url ?? "", "http://localhost").pathname;
        if (!pathname.startsWith(`${MEDIAPIPE_TASKS_VISION_ROUTE}/`)) {
          next();
          return;
        }

        const file = basename(decodeURIComponent(pathname.slice(MEDIAPIPE_TASKS_VISION_ROUTE.length + 1)));
        const source = sourceDirs.map((dir) => join(dir, file)).find((candidate) => existsSync(candidate));
        if (!source) {
          next();
          return;
        }

        res.setHeader("Content-Type", contentType(file));
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
        createReadStream(source).pipe(res);
      });
    },
    writeBundle() {
      const targetDir = join(config.build.outDir, MEDIAPIPE_TASKS_VISION_ASSET_DIR);
      mkdirSync(targetDir, { recursive: true });
      for (const { dir, file } of assetFiles()) {
        copyFileSync(join(dir, file), join(targetDir, file));
      }
    },
  };
}

function daemonTokenDevPlugin(): Plugin {
  let lastCpus = os.cpus();
  const sendJson = (res: any, status: number, payload: unknown) => {
    res.statusCode = status;
    res.end(JSON.stringify(payload));
  };
  const unavailable = (res: any, command: string, reason: string) => {
    sendJson(res, 501, { error: `${command} is unavailable in browser development: ${reason}` });
  };
  const filesystemUsage = () => {
    const stats = statfsSync(CONFIG_DIR);
    const total = Number(stats.blocks) * stats.bsize;
    const free = Number(stats.bfree) * stats.bsize;
    return { total, used: Math.max(0, total - free) };
  };
  const cpuPercent = () => {
    const currentCpus = os.cpus();
    let totalDiff = 0;
    let idleDiff = 0;
    for (let i = 0; i < currentCpus.length; i++) {
      const current = currentCpus[i];
      const previous = lastCpus[i] || current;
      for (const type in current.times) {
        totalDiff +=
          current.times[type as keyof typeof current.times] - previous.times[type as keyof typeof previous.times];
      }
      idleDiff += current.times.idle - previous.times.idle;
    }
    lastCpus = currentCpus;
    return totalDiff > 0 ? Math.max(0, Math.min(100, Math.round(100 - (idleDiff / totalDiff) * 100))) : null;
  };
  return {
    name: "daemon-token-dev",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const pathname = new URL(req.url ?? "", "http://localhost").pathname;
        if (pathname === "/api/auth_token") {
          let token = "";
          try {
            const localAppData = process.env.LOCALAPPDATA || join(process.env.USERPROFILE || "", "AppData", "Local");
            const candidates = [
              join(localAppData, "heliox-os", "runtime", "auth_token"),
              join(localAppData, "pilot", "runtime", "auth_token"),
              join(localAppData, "heliox-os", "auth_token"),
              join(localAppData, "pilot", "auth_token"),
              "/run/user/1000/heliox-os/auth_token",
              "/run/user/1000/pilot/auth_token",
            ];
            for (const path of candidates) {
              if (existsSync(path)) {
                const content = readFileSync(path, "utf-8").trim();
                if (content) {
                  token = content;
                  break;
                }
              }
            }
          } catch {
            // ignore
          }
          res.setHeader("Content-Type", "text/plain");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.end(token);
          return;
        }

        if (pathname === "/api/tauri_invoke" && req.method === "POST") {
          let body = "";
          req.on("data", (chunk) => {
            body += chunk;
          });
          req.on("end", async () => {
            let command = "";
            let args: Record<string, unknown> = {};
            try {
              const parsed = JSON.parse(body);
              command = parsed.command;
              if (parsed.args && typeof parsed.args === "object" && !Array.isArray(parsed.args)) {
                args = parsed.args;
              }
            } catch {
              // ignore
            }

            res.setHeader("Content-Type", "application/json");
            res.setHeader("Access-Control-Allow-Origin", "*");

            if (command === "get_system_stats") {
              const currentCpus = os.cpus();
              const totalRam = os.totalmem();
              const freeRam = os.freemem();
              const ram = Math.round(((totalRam - freeRam) / totalRam) * 100);
              const total_ram = Math.round(totalRam / (1024 * 1024 * 1024));
              const cpu_name = currentCpus[0]?.model?.split(" @")[0]?.trim() || "Local CPU";
              const filesystem = filesystemUsage();

              sendJson(res, 200, {
                cpu: cpuPercent(),
                ram,
                disk: filesystem.total > 0 ? (filesystem.used / filesystem.total) * 100 : null,
                network_up: null,
                network_down: null,
                cpu_name,
                total_ram,
                disk_size: filesystem.total / 1024 ** 3,
              });
              return;
            }

            if (command === "get_temperature_stats") {
              unavailable(res, command, "the web runtime has no trusted hardware-sensor bridge");
              return;
            }

            if (command === "run_neural_benchmark") {
              const benchmark = args.benchmark;
              const pythonArgs = ["-m", "pilot.neural.benchmark"];
              if (benchmark === "brainflow-synthetic") {
                pythonArgs.push(benchmark, "--seconds", "2");
              } else if (benchmark === "eegbci") {
                const subject = args.subject ?? 1;
                const runs = args.runs ?? [6, 10, 14];
                const allowedRuns = new Set([4, 6, 8, 10, 12, 14]);
                if (
                  !Number.isInteger(subject) ||
                  Number(subject) < 1 ||
                  Number(subject) > 109 ||
                  !Array.isArray(runs) ||
                  runs.length < 2 ||
                  runs.length > 6 ||
                  runs.some((run) => !Number.isInteger(run) || !allowedRuns.has(Number(run)))
                ) {
                  sendJson(res, 400, { error: "Invalid registered EEGBCI benchmark selection" });
                  return;
                }
                pythonArgs.push(benchmark, "--subject", String(subject), "--runs", ...runs.map((run) => String(run)));
              } else {
                sendJson(res, 400, { error: "Unsupported neural benchmark" });
                return;
              }

              const python = process.env.HELIOX_PYTHON || (process.platform === "win32" ? "python" : "python3");
              const daemonDir = join(CONFIG_DIR, "..", "..", "daemon");
              execFile(
                python,
                pythonArgs,
                { cwd: daemonDir, windowsHide: true, timeout: 120_000, maxBuffer: 1024 * 1024 },
                (error, stdout, stderr) => {
                  if (error) {
                    const detail = stderr.trim().split("\n").filter(Boolean).at(-1) || error.message;
                    sendJson(res, 500, { error: `Neural benchmark failed: ${detail}` });
                    return;
                  }
                  try {
                    sendJson(res, 200, JSON.parse(stdout));
                  } catch (parseError) {
                    sendJson(res, 500, { error: `Neural benchmark returned invalid JSON: ${String(parseError)}` });
                  }
                },
              );
              return;
            }

            if (command === "get_neural_sidecar_status") {
              sendJson(res, 200, {
                running: false,
                pid: null,
                available: false,
                reason: "The native neural sidecar is only available in the packaged Heliox desktop app",
              });
              return;
            }

            if (command === "get_uptime") {
              const uptimeSec = Math.round(os.uptime());
              const hours = Math.floor(uptimeSec / 3600);
              const minutes = Math.floor((uptimeSec % 3600) / 60);
              res.end(JSON.stringify(`${hours}h ${minutes}m`));
              return;
            }

            if (command === "get_terminal_logs") {
              const logFile = join(CONFIG_DIR, "system.log");
              let logs: string[] = [];
              if (existsSync(logFile)) {
                try {
                  const diskLogs = readFileSync(logFile, "utf-8").split("\n").filter(Boolean).slice(-30);
                  if (diskLogs.length > 0) logs = diskLogs;
                } catch (e) {}
              }
              res.end(JSON.stringify(logs));
              return;
            }

            if (command === "open_terminal") {
              try {
                execSync(
                  `start powershell -NoProfile -NoExit -Command "cd '${CONFIG_DIR.replace(/\\/g, "/")}'; echo '=== Heliox OS System Terminal Active ==='"`,
                );
                res.end(JSON.stringify("Terminal opened successfully"));
              } catch (primaryError) {
                try {
                  execSync(`start cmd /K echo Heliox OS System Terminal`);
                  res.end(JSON.stringify("Terminal opened successfully"));
                } catch (fallbackError) {
                  sendJson(res, 500, {
                    error: `Could not open a system terminal: ${String(fallbackError || primaryError)}`,
                  });
                }
              }
              return;
            }

            if (command === "clear_logs") {
              const logFile = join(CONFIG_DIR, "system.log");
              try {
                if (!existsSync(logFile)) {
                  res.end(JSON.stringify("No daemon log file exists yet"));
                  return;
                }
                writeFileSync(logFile, "", "utf-8");
                res.end(JSON.stringify("Daemon log cleared"));
              } catch (error) {
                sendJson(res, 500, { error: `Could not clear ${logFile}: ${String(error)}` });
              }
              return;
            }

            if (command === "restart_agents") {
              unavailable(res, command, "no browser-side agent supervisor is configured");
              return;
            }

            if (command === "system_info") {
              const totalMem = os.totalmem();
              const freeMem = os.freemem();
              const usedMem = totalMem - freeMem;
              const filesystem = filesystemUsage();

              sendJson(res, 200, {
                status: "ok",
                cpu_percent: cpuPercent(),
                memory_percent: Math.round((usedMem / totalMem) * 100),
                memory_used: usedMem,
                memory_total: totalMem,
                disk_percent: filesystem.total > 0 ? Math.round((filesystem.used / filesystem.total) * 100) : null,
                disk_used: filesystem.used,
                disk_total: filesystem.total,
                hostname: os.hostname(),
                uptime_seconds: os.uptime(),
              });
              return;
            }

            if (command === "system_scan") {
              const totalMem = Math.round(os.totalmem() / (1024 * 1024 * 1024));
              const freeMem = Math.round(os.freemem() / (1024 * 1024 * 1024));
              const usedMem = totalMem - freeMem;
              const cpus = os.cpus();
              res.end(
                JSON.stringify({
                  scan_scope: "Resource telemetry only; no malware or threat scan was performed",
                  host_os: `${os.type()} ${os.release()} (${os.arch()})`,
                  cpu_processor: cpus[0]?.model?.trim() || "Local CPU",
                  active_threads: cpus.length,
                  memory_utilization: `${usedMem} GB / ${totalMem} GB (${Math.round((usedMem / totalMem) * 100)}%)`,
                  system_uptime: `${Math.round(os.uptime() / 3600)}h ${Math.round((os.uptime() % 3600) / 60)}m`,
                }),
              );
              return;
            }

            if (command === "take_screenshot") {
              const shotPath = join(CONFIG_DIR, `screenshot_${Date.now()}.png`);
              try {
                const psCmd = `Add-Type -AssemblyName System.Windows.Forms,System.Drawing; $bmp = New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen(0,0,0,0,$bmp.Size); $bmp.Save('${shotPath.replace(/\\/g, "\\\\")}'); $g.Dispose(); $bmp.Dispose();`;
                execSync(`powershell -NoProfile -Command "${psCmd}"`, { timeout: 4000 });
              } catch (e) {
                sendJson(res, 500, {
                  error: `Screenshot capture failed: ${e instanceof Error ? e.message : String(e)}`,
                });
                return;
              }
              res.end(JSON.stringify(shotPath));
              return;
            }

            if (command === "get_agent_activity") {
              unavailable(res, command, "agent activity must come from the running daemon");
              return;
            }

            if (command === "get_rss_feed") {
              let feedItems: any[] = [];
              try {
                const pkgPath = join(process.cwd(), "package.json");
                let currentVer = UI_VERSION;
                if (existsSync(pkgPath)) {
                  try {
                    currentVer = JSON.parse(readFileSync(pkgPath, "utf-8")).version || currentVer;
                  } catch (e) {}
                }
                feedItems.push({
                  title: `Heliox OS v${currentVer} Active Release (JARVIS Core Engine)`,
                  url: "https://github.com/VyomKulshrestha/Heliox-OS/releases",
                  source: "Current Build",
                });

                try {
                  const tagOut = execSync(
                    'git tag -l --sort=-creatordate --format="%(refname:short)|%(creatordate:short)|%(subject)"',
                    { cwd: CONFIG_DIR, encoding: "utf-8" },
                  ).trim();
                  tagOut
                    .split("\n")
                    .filter(Boolean)
                    .slice(0, 4)
                    .forEach((line) => {
                      const parts = line.split("|");
                      if (parts[0]) {
                        feedItems.push({
                          title: `Release ${parts[0]}: ${parts[2] || "Official Heliox OS Distribution"}`,
                          url: `https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/${parts[0]}`,
                          source: `Release Tag (${parts[1] || "Published"})`,
                        });
                      }
                    });
                } catch (e) {}
              } catch (e) {
                feedItems = [];
              }
              res.end(JSON.stringify(feedItems));
              return;
            }

            if (command === "get_log_count") {
              const logFile = join(CONFIG_DIR, "system.log");
              let count = 0;
              if (existsSync(logFile)) {
                try {
                  const lines = readFileSync(logFile, "utf-8").split("\n").filter(Boolean);
                  count = lines.length || 0;
                } catch (e) {}
              }
              res.end(JSON.stringify(count));
              return;
            }

            if (command === "get_status_metrics") {
              unavailable(res, command, "latency and active-agent counts require daemon telemetry");
              return;
            }

            if (command === "get_dashboard_status") {
              const totalRam = os.totalmem();
              const ram = Math.round(((totalRam - os.freemem()) / totalRam) * 100);
              const cpu = cpuPercent();
              sendJson(res, 200, {
                connected: false,
                agents: null,
                cpu: cpu == null ? "Unavailable" : `${cpu}%`,
                memory: `${ram}%`,
                network_up: "Unavailable",
                network_down: "Unavailable",
              });
              return;
            }

            unavailable(res, command || "Unknown command", "no development implementation is registered");
          });
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [
    svelte(),
    svelteTesting(),
    mediapipeHandsAssets(),
    mediapipeTasksVisionAssets(),
    daemonTokenDevPlugin(),
    enforceProductionChunkBudget(),
  ],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: "esnext",
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
    rollupOptions: {
      output: {
        manualChunks: productionChunk,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.{test,spec}.{js,ts}"],
    exclude: ["tests/visual/**", "tests/static/**", "node_modules/**"],
  },
});
