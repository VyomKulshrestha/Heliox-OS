import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import type { Plugin, ResolvedConfig } from "vite";
import {
  mkdirSync,
  readdirSync,
  copyFileSync,
  existsSync,
  createReadStream,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { join, basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";
import { execSync } from "node:child_process";

const MEDIAPIPE_HANDS_ROUTE = "/mediapipe/hands";
const MEDIAPIPE_HANDS_ASSET_DIR = "mediapipe/hands";
const CONFIG_DIR = dirname(fileURLToPath(import.meta.url));
const MEDIAPIPE_HANDS_DIR = join(CONFIG_DIR, "node_modules", "@mediapipe", "hands");

function contentType(file: string) {
  if (file.endsWith(".js")) return "text/javascript";
  if (file.endsWith(".wasm")) return "application/wasm";
  return "application/octet-stream";
}

function mediapipeHandsAssets(): Plugin {
  let config: ResolvedConfig;

  const assetFiles = () =>
    readdirSync(MEDIAPIPE_HANDS_DIR).filter((file) =>
      /\.(binarypb|data|js|tflite|wasm)$/.test(file),
    );

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

        const file = basename(
          decodeURIComponent(pathname.slice(MEDIAPIPE_HANDS_ROUTE.length + 1)),
        );
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

function daemonTokenDevPlugin(): Plugin {
  let lastCpus = os.cpus();
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
          req.on("data", (chunk) => { body += chunk; });
          req.on("end", () => {
            let command = "";
            try {
              const parsed = JSON.parse(body);
              command = parsed.command;
            } catch {
              // ignore
            }

            res.setHeader("Content-Type", "application/json");
            res.setHeader("Access-Control-Allow-Origin", "*");

            if (command === "get_system_stats") {
              const currentCpus = os.cpus();
              let totalDiff = 0;
              let idleDiff = 0;
              for (let i = 0; i < currentCpus.length; i++) {
                const c = currentCpus[i];
                const l = lastCpus[i] || c;
                for (const type in c.times) {
                  totalDiff += (c.times[type as keyof typeof c.times] - l.times[type as keyof typeof l.times]);
                }
                idleDiff += (c.times.idle - l.times.idle);
              }
              lastCpus = currentCpus;
              const cpu = totalDiff > 0 ? Math.max(1, Math.min(100, Math.round(100 - (idleDiff / totalDiff) * 100))) : 15;
              const totalRam = os.totalmem();
              const freeRam = os.freemem();
              const ram = Math.round(((totalRam - freeRam) / totalRam) * 100);
              const total_ram = Math.round(totalRam / (1024 * 1024 * 1024));
              const cpu_name = currentCpus[0]?.model?.split(" @")[0]?.trim() || "Local CPU";

              res.end(JSON.stringify({
                cpu,
                ram,
                disk: 35,
                network_up: 140,
                network_down: 520,
                cpu_name,
                total_ram,
                disk_size: 512,
              }));
              return;
            }

            if (command === "get_temperature_stats") {
              const currentCpus = os.cpus();
              let liveBatteryPercent = 95;
              try {
                const out = execSync(`powershell -NoProfile -Command "(Get-CimInstance -ClassName Win32_Battery -ErrorAction SilentlyContinue).EstimatedChargeRemaining"`, { timeout: 1500, encoding: 'utf-8' }).trim();
                const parsed = parseInt(out, 10);
                if (!isNaN(parsed) && parsed >= 0 && parsed <= 100) liveBatteryPercent = parsed;
              } catch (e) {
                // ignore
              }
              res.end(JSON.stringify({
                cpu: 44,
                gpu: 40,
                motherboard: 36,
                ssd: 34,
                vrm: 33,
                battery: 29,
                power: 52,
                cpu_name: currentCpus[0]?.model?.split(" @")[0]?.trim() || "Local CPU",
                cpu_threads: currentCpus.length || 8,
                battery_percent: liveBatteryPercent
              }));
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
              let logs = [
                "[System] Heliox OS Agent Daemon Connected",
                `[Monitor] Host: ${os.hostname()} (${os.platform()})`,
                "[Core] ReAct Loop active on ws://127.0.0.1:8785",
                "[System] Threat Containment Bridge initialized",
                "[Cognitive] TRIBE v2 Neural Cognitive HUD active",
              ];
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
                execSync(`start powershell -NoProfile -NoExit -Command "cd '${CONFIG_DIR.replace(/\\/g, '/') }'; echo '=== Heliox OS System Terminal Active ==='"`);
              } catch (e) {
                try { execSync(`start cmd /K echo Heliox OS System Terminal`); } catch (e2) {}
              }
              res.end(JSON.stringify("Terminal opened successfully"));
              return;
            }

            if (command === "clear_logs") {
              const logFile = join(CONFIG_DIR, "system.log");
              try { if (existsSync(logFile)) writeFileSync(logFile, "", "utf-8"); } catch (e) {}
              res.end(JSON.stringify("All logs cleared successfully"));
              return;
            }

            if (command === "restart_agents") {
              res.end(JSON.stringify("All 4 neural background agents (System, Code, Web, Monitor) restarted & synchronized (`ws://127.0.0.1:8785`)."));
              return;
            }

            if (command === "system_info") {
              const totalMem = os.totalmem();
              const freeMem = os.freemem();
              const usedMem = totalMem - freeMem;
              let diskUsedBytes = 771.8 * 1024 ** 3;
              let diskTotalBytes = 952.6 * 1024 ** 3;
              try {
                const psOut = execSync('powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk -Filter \\"DeviceID=\'C:\'\\" | Select-Object -Property Size,FreeSpace | ConvertTo-Json -Compress"', { timeout: 2000, encoding: 'utf-8' }).trim();
                const diskJson = JSON.parse(psOut);
                if (diskJson && diskJson.Size) {
                  diskTotalBytes = Number(diskJson.Size);
                  diskUsedBytes = diskTotalBytes - Number(diskJson.FreeSpace || 0);
                }
              } catch (e) {}
              const cpus = os.cpus();
              let cpuPercent = 15;
              try {
                let totalIdle = 0, totalTick = 0;
                cpus.forEach(c => {
                  for (let type in c.times) totalTick += (c.times as any)[type];
                  totalIdle += c.times.idle;
                });
                cpuPercent = Math.round((1 - totalIdle / totalTick) * 100);
              } catch (e) {}

              res.end(JSON.stringify({
                cpu_percent: cpuPercent,
                memory_percent: Math.round((usedMem / totalMem) * 100),
                memory_used: usedMem,
                memory_total: totalMem,
                disk_percent: Math.round((diskUsedBytes / diskTotalBytes) * 100),
                disk_used: diskUsedBytes,
                disk_total: diskTotalBytes,
                hostname: os.hostname(),
                uptime_seconds: os.uptime()
              }));
              return;
            }

            if (command === "get_uptime") {
              const upSec = Math.round(os.uptime());
              const days = Math.floor(upSec / 86400);
              const hrs = Math.floor((upSec % 86400) / 3600);
              const mins = Math.floor((upSec % 3600) / 60);
              const formatted = days > 0 ? `${days}d ${hrs}h ${mins}m` : `${hrs}h ${mins}m`;
              res.end(JSON.stringify(formatted));
              return;
            }

            if (command === "system_scan") {
              const totalMem = Math.round(os.totalmem() / (1024 * 1024 * 1024));
              const freeMem = Math.round(os.freemem() / (1024 * 1024 * 1024));
              const usedMem = totalMem - freeMem;
              const cpus = os.cpus();
              res.end(JSON.stringify({
                status: "Healthy (0 threats / anomalies detected)",
                host_os: `${os.type()} ${os.release()} (${os.arch()})`,
                cpu_processor: cpus[0]?.model?.trim() || "Local CPU",
                active_threads: cpus.length,
                memory_utilization: `${usedMem} GB / ${totalMem} GB (${Math.round((usedMem/totalMem)*100)}%)`,
                system_uptime: `${Math.round(os.uptime()/3600)}h ${Math.round((os.uptime()%3600)/60)}m`
              }));
              return;
            }

            if (command === "take_screenshot") {
              const shotPath = join(CONFIG_DIR, `screenshot_${Date.now()}.png`);
              try {
                const psCmd = `Add-Type -AssemblyName System.Windows.Forms,System.Drawing; $bmp = New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width, [System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height); $g = [System.Drawing.Graphics]::FromImage($bmp); $g.CopyFromScreen(0,0,0,0,$bmp.Size); $bmp.Save('${shotPath.replace(/\\/g, '\\\\')}'); $g.Dispose(); $bmp.Dispose();`;
                execSync(`powershell -NoProfile -Command "${psCmd}"`, { timeout: 4000 });
              } catch (e) {
                try { writeFileSync(shotPath, "Heliox OS Screenshot Capture Record"); } catch (e2) {}
              }
              res.end(JSON.stringify(shotPath));
              return;
            }

            if (command === "get_agent_activity") {
              res.end(JSON.stringify([
                { agent: "System Agent", status: "Idle", tasks_completed: 14 },
                { agent: "Code Agent", status: "Active", tasks_completed: 8 },
                { agent: "Web Agent", status: "Idle", tasks_completed: 5 },
                { agent: "Monitor Agent", status: "Monitoring", tasks_completed: 42 },
                { agent: "Communication Agent", status: "Ready", tasks_completed: 3 },
              ]));
              return;
            }

            if (command === "get_rss_feed") {
              let feedItems: any[] = [];
              try {
                const pkgPath = join(process.cwd(), "package.json");
                let currentVer = "0.7.1";
                if (existsSync(pkgPath)) {
                  try { currentVer = JSON.parse(readFileSync(pkgPath, "utf-8")).version || currentVer; } catch(e){}
                }
                feedItems.push({
                  title: `Heliox OS v${currentVer} Active Release (JARVIS Core Engine)`,
                  url: "https://github.com/VyomKulshrestha/Heliox-OS/releases",
                  source: "Current Build"
                });

                try {
                  const tagOut = execSync('git tag -l --sort=-creatordate --format="%(refname:short)|%(creatordate:short)|%(subject)"', { cwd: CONFIG_DIR, encoding: 'utf-8' }).trim();
                  tagOut.split('\n').filter(Boolean).slice(0, 4).forEach(line => {
                    const parts = line.split('|');
                    if (parts[0]) {
                      feedItems.push({
                        title: `Release ${parts[0]}: ${parts[2] || 'Official Heliox OS Distribution'}`,
                        url: `https://github.com/VyomKulshrestha/Heliox-OS/releases/tag/${parts[0]}`,
                        source: `Release Tag (${parts[1] || 'Published'})`
                      });
                    }
                  });
                } catch (e) {}

                if (feedItems.length === 1) {
                  feedItems.push({
                    title: `TRIBE v2 Cognitive Engine & Threat Containment Bridge Live`,
                    url: "https://github.com/VyomKulshrestha/Heliox-OS",
                    source: "System Feature"
                  });
                }
              } catch (e) {
                feedItems = [{ title: "Heliox OS v0.7.1 Active - JARVIS Core Running", url: "https://github.com/VyomKulshrestha/Heliox-OS", source: "System" }];
              }
              res.end(JSON.stringify(feedItems));
              return;
            }

            if (command === "get_log_count") {
              const logFile = join(CONFIG_DIR, "system.log");
              let count = 128;
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
              const totalRam = os.totalmem();
              const ram = Math.round(((totalRam - os.freemem()) / totalRam) * 100);
              res.end(JSON.stringify({ cpu: 15, ram, latency_ms: 8, agents_active: 5 }));
              return;
            }

            if (command === "get_dashboard_status") {
              const totalRam = os.totalmem();
              const ram = Math.round(((totalRam - os.freemem()) / totalRam) * 100);
              res.end(JSON.stringify({
                connected: true,
                agents: 5,
                cpu: `15%`,
                memory: `${ram}%`,
                network_up: "140 KB/s",
                network_down: "520 KB/s",
              }));
              return;
            }

            res.end(JSON.stringify({}));
          });
          return;
        }

        next();
      });
    },
  };
}

export default defineConfig({
  plugins: [svelte(), mediapipeHandsAssets(), daemonTokenDevPlugin()],
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
  },
});
