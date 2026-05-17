/**
 * Shared helpers for visual regression tests.
 *
 * Because the app connects to a live Tauri/WebSocket daemon that won't be
 * running in CI, we mock the Tauri IPC bridge and the WebSocket session so
 * every panel renders in a deterministic, offline state.
 */

import type { Page } from "@playwright/test";

/**
 * Inject a minimal Tauri IPC stub so the app doesn't crash when it tries
 * to call `window.__TAURI_INTERNALS__` or `window.__TAURI__`.
 *
 * Must be called before page.goto().
 */
export async function mockTauriIpc(page: Page): Promise<void> {
  await page.addInitScript(() => {
    // Minimal Tauri v2 internals stub
    (window as any).__TAURI_INTERNALS__ = {
      invoke: async (_cmd: string, _args?: unknown) => null,
      transformCallback: (cb: Function) => cb,
      convertFileSrc: (src: string) => src,
    };

    // Stub the plugin APIs used by the app
    (window as any).__TAURI__ = {
      core: { invoke: async () => null },
      event: {
        listen: async () => () => {},
        once: async () => () => {},
        emit: async () => {},
      },
    };

    // Prevent WebSocket connection attempts from throwing
    const OrigWS = window.WebSocket;
    (window as any).WebSocket = class extends OrigWS {
      constructor(url: string, protocols?: string | string[]) {
        // Redirect to a no-op URL that will silently fail to connect
        super("ws://localhost:0", protocols);
      }
    };
  });
}

/**
 * Navigate to the app root and wait for the main window to be visible.
 * Skips the SetupWizard by pre-seeding localStorage.
 */
export async function gotoApp(page: Page): Promise<void> {
  await mockTauriIpc(page);

  // Pre-seed localStorage so the SetupWizard is skipped
  await page.addInitScript(() => {
    localStorage.setItem("heliox_first_run_complete", "true");
    // Minimal settings so the app doesn't crash on undefined reads
    localStorage.setItem(
      "heliox_settings",
      JSON.stringify({
        first_run_complete: true,
        theme: "dark",
        model: {
          provider: "ollama",
          ollama_model: "llama3.1:8b",
          mode: "lightweight",
          cloud_provider: "gemini",
          cloud_model: "",
          gpu_memory_limit_mb: 0,
        },
        security: {
          root_enabled: false,
          dry_run: false,
          snapshot_on_destructive: true,
          snapshot_retention_count: 10,
        },
        screen_vision: { capture_interval_seconds: 3 },
        restrictions: {
          protected_folders: [],
          protected_packages: [],
          blocked_commands: [],
        },
      })
    );
  });

  await page.goto("/");
  // Wait for the main window chrome to appear
  await page.waitForSelector(".window", { timeout: 15_000 });
}

/**
 * Click a top-level tab by its visible label.
 */
export async function clickTab(page: Page, label: string): Promise<void> {
  await page.click(`nav.tabs button:has-text("${label}")`);
  // Brief settle for Svelte transitions
  await page.waitForTimeout(300);
}

/**
 * Disable all CSS animations and transitions for pixel-stable screenshots.
 */
export async function freezeAnimations(page: Page): Promise<void> {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition-duration: 0s !important;
        transition-delay: 0s !important;
      }
    `,
  });
}
