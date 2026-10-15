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
    // Seed Math.random using mulberry32 to make all random layouts (like canvas nodes) completely deterministic
    const mulberry32 = (a: number) => {
      return () => {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    };
    Math.random = mulberry32(12345);

    // Mock Date.now to return a deterministic incrementing time to stabilize animations/timers
    let mockTime = 1716768000000;
    Date.now = () => {
      mockTime += 100;
      return mockTime;
    };

    const invoke = async (cmd: string) => {
      if (cmd === "get_system_stats") return { cpu: 18, ram: 62 };
      return null;
    };

    // Minimal Tauri v2 internals stub
    (window as any).__TAURI_INTERNALS__ = {
      invoke,
      transformCallback: (cb: Function) => cb,
      convertFileSrc: (src: string) => src,
    };

    // Stub requestAnimationFrame and cancelAnimationFrame to freeze canvas animations
    (window as any).requestAnimationFrame = () => 999;
    (window as any).cancelAnimationFrame = () => {};

    // Stub the plugin APIs used by the app
    (window as any).__TAURI__ = {
      core: { invoke },
      event: {
        listen: async () => () => {},
        once: async () => () => {},
        emit: async () => {},
      },
    };

    // Status reads the Chat and Settings tabs re-send on a timer (every 1.5 to 5 s). Recording them as the last
    // send could replace the request a test is polling for between the click that sent it and the next poll.
    const timerPolledMethods = new Set([
      "air_handoff_status",
      "get_security_status",
      "get_snapshot_status",
      "get_uptime",
      "list_wake_variants",
      "mesh_status",
      "neural_status",
      "voice_listener_stats",
    ]);

    // Mock WebSocket to simulate daemon connection and intercept messages
    (window as any).WebSocket = class {
      static OPEN = 1;
      static CLOSED = 3;

      onopen: any;
      onmessage: any;
      onerror: any;
      onclose: any;
      readyState = 1; // WebSocket.OPEN

      constructor(url: string, protocols?: string | string[]) {
        (window as any).__mock_ws__ = this;

        // Auto-connect after a tiny delay
        setTimeout(() => {
          if (this.onopen) this.onopen(new Event("open"));
        }, 10);
      }

      send(data: string) {
        // Intercept sent messages and store them globally for tests to inspect
        const request = JSON.parse(data);
        if (!timerPolledMethods.has(request.method)) (window as any).__last_ws_send__ = request;

        if (request.method === "risk_gate_status" && (window as any).__mock_risk_gate_status_missing__) {
          setTimeout(() => {
            this.onmessage?.({
              data: JSON.stringify({
                jsonrpc: "2.0",
                id: request.id,
                error: { code: -32601, message: "Method not found: risk_gate_status" },
              }),
            });
          }, 0);
          return;
        }

        if (request.method === "risk_gate_config_update") {
          (window as any).__risk_gate_enabled__ = request.params.enabled;
          (window as any).__risk_gate_update__ = request;
        }
        if (request.method === "self_healing_config_update") {
          (window as any).__self_healing_update__ = request;
        }
        if (request.method === "voice_gesture_workflow_submit") {
          (window as any).__workflow_submit__ = request;
        }
        if (request.method === "gesture_workflow_bindings_update") {
          (window as any).__gesture_workflow_update__ = request;
          (window as any).__gesture_workflow_policy__ = request.params;
        }
        if (
          request.method === "auth" ||
          request.method === "risk_gate_status" ||
          request.method === "risk_gate_config_update" ||
          request.method === "cognitive_state" ||
          request.method === "self_healing_status" ||
          request.method === "self_healing_config_update" ||
          request.method === "voice_gesture_workflow_list" ||
          request.method === "voice_gesture_workflow_submit" ||
          request.method === "gesture_workflow_bindings_get" ||
          request.method === "gesture_workflow_bindings_update" ||
          request.method === "get_security_status" ||
          request.method === "agent_mesh_status" ||
          request.method === "agent_routing" ||
          request.method === "subscription_status" ||
          request.method === "subscription_login" ||
          request.method === "update_config" ||
          request.method === "co_adaptation_status" ||
          request.method.startsWith("air_handoff_")
        ) {
          let result: Record<string, unknown>;
          if (request.method === "auth") {
            result = { status: "ok" };
          } else if (request.method === "co_adaptation_status") {
            // Available, off, default settings, nobody has consented yet.
            result = {
              status: "ok",
              available: true,
              config: {
                enabled: false,
                adaptive_enabled: false,
                learn_across_sessions: false,
                gaze_prior_enabled: false,
                hand_pointer_enabled: false,
                ambiguity_threshold: 0.25,
                unintended_activation_tolerance: 0.05,
                noninferiority_margin: 0.01,
                upper_bound_credibility: 0.95,
                exploration_floor: 0.1,
                switching_penalty: 0.1,
                burden_prompt_every_blocks: 1,
                day_boundary_hour: 4,
              },
              keyring_available: true,
              store: {
                exists: false,
                bytes: 0,
                participants: 0,
                sessions: 0,
                decisions: 0,
                trials: 0,
                burden_reports: 0,
                oldest_recorded_at: null,
                pruned_sessions: 0,
              },
              participants: [],
              preferences: [],
              proposals: [],
              protocol: { protocol_id: "coadapt-study-v1", sha256: "0".repeat(64) },
              versions: {
                strategy: "coadapt-strategies-v1",
                feature_schema: "coadapt-features-v1",
                model: "coadapt-owm-v1",
                prior: "literature-v1",
                policy: "coadapt-policy-v1",
                sdk: "0.1.0",
                consent_text: "coadapt-consent-v1",
              },
              authority: "presentation_only",
              execution_authority: false,
            };
          } else if (request.method === "get_security_status") {
            result = {
              status: "ok",
              root_policy_enabled: false,
              process_elevated: false,
              platform: "win32",
              detail: "Visual-test daemon is not elevated.",
            };
          } else if (request.method === "cognitive_state") {
            result = {
              attention_score: 0.58,
              stress_level: 0.22,
              cognitive_load: 0.37,
              dominant_modality: "visual",
              confidence: 0.64,
              estimate_kind: "behavioral",
              signal_sources: 2,
            };
          } else if (request.method === "agent_mesh_status") {
            const specialist = (displayName: string, role: string, capabilities: string[], devices: string[] = []) => ({
              agent_key: `builtin:pilot.agents.domain_agents.${displayName}`,
              display_name: displayName,
              role,
              source: "builtin",
              description: `${displayName} bounded capability contract`,
              capabilities,
              executable: true,
              handoff_contract: "bounded_context_refs_and_partial_results",
              permissions: {
                action_types: capabilities,
                confirmation_actions: [],
                filesystem_read: [],
                filesystem_write: [],
                network_domains: [],
                process_names: [],
                credential_names: [],
                clipboard: [],
                devices,
                authority: `agent_gateway:${role}`,
              },
              budget: {
                max_tokens_per_task: 6000,
                max_actions_per_task: 12,
                max_latency_ms_per_action: 90000,
                max_concurrency: 1,
              },
              performance: {
                attempts: 0,
                successes: 0,
                failures: 0,
                quality_score: 0.5,
                average_latency_ms: 0,
                in_flight: 0,
              },
            });
            result = {
              status: "ok",
              enabled: true,
              total_specialists: 21,
              executable_specialists: 21,
              external_capability_providers: 0,
              registered_action_types: 157,
              available_action_types: 157,
              coverage_complete: true,
              uncovered_action_types: [],
              sources: { builtin: 21 },
              delegation: {
                maximum_depth: 3,
                maximum_fanout: 4,
                cycle_detection: true,
                full_transcript_handoffs: false,
                cancellation_propagation: true,
                partial_result_recovery: true,
                parallel_only_when_explicitly_independent: true,
              },
              routing: {
                fixed_numeric_ceiling: false,
                selection: "capability_then_verified_outcome",
                self_reported_success_authority: false,
              },
              specialists: [
                specialist("FileOperationsAgent", "file_agent", ["file_hash", "file_compare"]),
                specialist("GitAgent", "git_agent", ["git_status", "git_diff", "git_push"]),
                specialist("VisionAgent", "vision_agent", ["screen_ocr", "screen_detect_elements"], ["screen"]),
              ],
            };
          } else if (request.method === "agent_routing") {
            result = {
              orchestrator: {
                assigned_specialists: [
                  {
                    agent_key: "builtin:pilot.agents.git_agent.GitAgent",
                    role: "git_agent",
                    display_name: "GitAgent",
                    matches: 3,
                    quality_score: 0.5,
                  },
                ],
              },
            };
          } else if (request.method === "subscription_status") {
            result = {
              status: "ok",
              provider: request.params?.provider ?? "codex",
              installed: true,
              authenticated: true,
              subscription: true,
              version: "codex-cli 0.144.5",
              message: "Connected through the Codex ChatGPT subscription login.",
              last_usage: {
                input_tokens: 13876,
                cached_input_tokens: 8960,
                uncached_input_tokens: 4916,
                output_tokens: 12,
                heliox_prompt_chars: 242,
                heliox_estimated_prompt_tokens: 61,
              },
              session_usage: {
                input_tokens: 13876,
                cached_input_tokens: 8960,
                uncached_input_tokens: 4916,
                output_tokens: 12,
              },
            };
          } else if (request.method === "subscription_login") {
            result = {
              status: "started",
              message: "The official browser sign-in flow was started.",
            };
          } else if (request.method.startsWith("risk_gate_")) {
            result = {
              status: "ok",
              enabled: (window as any).__risk_gate_enabled__ ?? true,
              weights_loaded: true,
              model_version: "risk-mlp-v3-calibrated",
              training_samples: 36000,
              validation_samples: 5400,
              calibrated: true,
              validation_mae: {
                disk_delta: 3.3039651015087657e-8,
                process_delta: 1.3025066436966881e-5,
              },
              baseline_mae: {
                disk_delta: 7.18494490570265e-8,
                process_delta: 0.0022166655398905277,
              },
              embedding_size: 23,
              learnable_action_types: ["file_write", "file_delete", "service_start", "service_stop"],
              last_evaluation: {
                evaluated_at: "2026-07-26T10:00:00Z",
                action_count: 2,
                risk_score: 0.8,
                reasons: ["predicted disk usage 96% exceeds the safe threshold"],
                worst_action_type: "file_write",
                prediction_sources: ["learned_calibrated", "rule"],
                prediction_confidence: 0.77,
              },
            };
          } else if (request.method.startsWith("self_healing_")) {
            result = {
              status: "ok",
              enabled: request.params?.enabled ?? true,
              auto_execute_max_tier: request.params?.auto_execute_max_tier ?? 1,
              watched_metrics: request.params?.watched_metrics ?? ["cpu", "memory", "disk"],
              monitors: {
                cpu: {
                  task_id: "monitor_cpu",
                  status: "running",
                  condition: "CPU > 80%",
                  interval_seconds: 10,
                  last_run: 1716768000,
                  run_count: 12,
                  error_count: 0,
                  last_result: { cpu_percent: 24 },
                },
                memory: {
                  task_id: "monitor_memory",
                  status: "running",
                  condition: "RAM > 85%",
                  interval_seconds: 15,
                  last_run: 1716768000,
                  run_count: 8,
                  error_count: 0,
                  last_result: { memory_percent: 58 },
                },
                disk: {
                  task_id: "monitor_disk",
                  status: "running",
                  condition: "Disk > 90%",
                  interval_seconds: 60,
                  last_run: 1716768000,
                  run_count: 2,
                  error_count: 0,
                  last_result: { disk_percent: 71 },
                },
              },
              attempts: [],
            };
          } else if (request.method === "voice_gesture_workflow_list") {
            result = { workflows: [] };
          } else if (request.method === "voice_gesture_workflow_submit") {
            result = {
              status: "submitted",
              workflow: {
                workflow_id: "wf_test",
                goal: request.params.goal,
                invocation_source: request.params.invocation_source,
                steps: [],
                current_step: 0,
                state: "pending",
                updated_at: "2026-07-26T10:00:00Z",
              },
            };
          } else if (request.method.startsWith("gesture_workflow_bindings_")) {
            const policy = (window as any).__gesture_workflow_policy__ ?? {
              enabled: false,
              bindings: [],
            };
            result = {
              status: "ok",
              enabled: policy.enabled,
              bindings: policy.bindings,
              supported_gestures: ["palm", "swipe_up", "thumbs_up"],
            };
          } else if (request.method.startsWith("air_handoff_")) {
            const state = ((window as any).__air_handoff_state__ ??= {
              enabled: false,
              running: false,
              receiver_url: null,
              paired_devices: [],
              pairing: null,
              draft: null,
              ready_transfers: 0,
              secure_storage_available: true,
            });
            if (request.method === "air_handoff_set_enabled") {
              state.enabled = Boolean(request.params?.enabled);
              state.running = state.enabled;
              state.receiver_url = state.enabled ? "http://192.168.1.25:8787" : null;
            } else if (request.method === "air_handoff_start_pairing") {
              state.pairing = {
                session_id: "pair_test",
                expires_at: Date.now() / 1000 + 300,
                pairing_url: "http://192.168.1.25:8787/#pair=test-secret",
              };
            } else if (request.method === "air_handoff_cancel_pairing") {
              state.pairing = null;
            }
            result = { status: "ok", ...state, ...(state.pairing ?? {}) };
          } else {
            result = {
              status: "ok",
              enabled: false,
              bindings: [],
            };
          }
          setTimeout(() => {
            this.onmessage?.({
              data: JSON.stringify({ jsonrpc: "2.0", id: request.id, result }),
            });
          }, 0);
        }
      }

      close() {
        this.readyState = 3; // WebSocket.CLOSED
        if (this.onclose) this.onclose(new Event("close"));
      }
    };
  });
}

/**
 * Helper to emit a daemon notification (e.g. status updates)
 */
export async function emitNotification(page: Page, method: string, params: Record<string, unknown>): Promise<void> {
  await page.evaluate(
    ({ method, params }) => {
      const ws = (window as any).__mock_ws__;
      if (ws && ws.onmessage) {
        ws.onmessage({ data: JSON.stringify({ method, params }) });
      }
    },
    { method, params },
  );
  await page.waitForTimeout(80);
}

/**
 * Navigate to the app root and wait for the main window to be visible.
 * Skips the SetupWizard by pre-seeding localStorage.
 */
export async function gotoApp(page: Page, options: { riskGateStatusMissing?: boolean } = {}): Promise<void> {
  if (options.riskGateStatusMissing) {
    await page.addInitScript(() => {
      (window as any).__mock_risk_gate_status_missing__ = true;
    });
  }
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
      }),
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
      .particle-canvas {
        display: none !important;
      }
    `,
  });
}
