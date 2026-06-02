import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import type { Plugin, ResolvedConfig } from "vite";
import {
  mkdirSync,
  readdirSync,
  copyFileSync,
  existsSync,
  createReadStream,
} from "node:fs";
import { join, basename, dirname } from "node:path";
import { fileURLToPath } from "node:url";

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

export default defineConfig({
  plugins: [svelte(), mediapipeHandsAssets()],
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
