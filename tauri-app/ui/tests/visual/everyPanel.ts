/**
 * The daemon's answers that make every Settings panel show its rows, for gotoApp's everyPanel option.
 *
 * Without them the mock leaves these reads unanswered and the panels stay on "Loading" or their empty state, so a
 * test of Settings' rows would never see them: the Agent Gateway editor's 27 source profiles, narration,
 * supervision, the speech-model list. The shapes are the daemon's own: the speech models come from the manifest
 * the daemon serves (daemon/pilot/system/speech_models.json), the gateway profiles from
 * pilot.security.gateway.DEFAULT_SOURCE_PROFILES (fixtures/gateway-source-profiles.json, written from it).
 */

import { readFileSync } from "node:fs";

type Manifest = {
  models: {
    id: string;
    title: string;
    purpose: string;
    language: string;
    size_bytes: number;
    licence: string;
    source: string;
  }[];
  recommended: Record<string, { ids: string[]; title: string; why: string }>;
};

function readJson<T>(relative: string): T {
  return JSON.parse(readFileSync(new URL(relative, import.meta.url), "utf-8")) as T;
}

/** The settings the conditional Settings rows need: the gesture cursor on, and a headset with its own input. */
export const EVERY_PANEL_SETTINGS = {
  gesture_cursor: { enabled: true },
  voice: { headset_input_choices: { "AirPods Pro": "Core Audio::MacBook Pro Microphone" } },
};

export function everyPanelReplies(): Record<string, unknown> {
  const manifest = readJson<Manifest>("../../../../daemon/pilot/system/speech_models.json");
  const english = manifest.recommended.en;
  const installed = new Set(["whisper-base"]);
  const profiles = readJson<Record<string, unknown>>("./fixtures/gateway-source-profiles.json");
  return {
    speech_models_list: {
      status: "ok",
      models: manifest.models.map((model) => ({
        id: model.id,
        title: model.title,
        purpose: model.purpose,
        languages: [model.language],
        licence: model.licence,
        source: model.source,
        download_bytes: model.size_bytes,
        installed: installed.has(model.id),
        partial_bytes: 0,
      })),
      recommended: {
        title: english.title,
        why: english.why,
        ids: english.ids,
        missing: english.ids.filter((id) => !installed.has(id)),
        missing_bytes: manifest.models
          .filter((model) => english.ids.includes(model.id) && !installed.has(model.id))
          .reduce((sum, model) => sum + model.size_bytes, 0),
      },
      voice: {
        language: "en",
        engine: "kokoro",
        voice: "af_heart",
        change: false,
        say: "Heliox speaks English with Kokoro's af_heart voice.",
      },
    },
    gateway_policy_get: { status: "ok", enabled: true, profiles },
    list_gateway_events: { status: "ok", events: [] },
    narration_status: {
      status: "ok",
      enabled: true,
      narrate_steps: true,
      interrupt_on_risk: true,
      proactive_review_enabled: true,
      live_corrections_enabled: true,
      follow_up_enabled: true,
      confirm_timeout_seconds: 120,
      advisory_timeout_seconds: 5,
    },
    proactive_learning_status: {
      status: "ok",
      patterns: { open_downloads: { shown: 4, accepted: 2, dismissed: 1 } },
    },
    supervision_status: {
      status: "ok",
      enabled: true,
      keyboard_mouse_hook_enabled: true,
      cognitive_coaching_enabled: true,
      risk_pattern_detection_enabled: true,
      hook_healthy: true,
      screen_reading: { available: true, reason: "", retry_in_seconds: null },
    },
  };
}
