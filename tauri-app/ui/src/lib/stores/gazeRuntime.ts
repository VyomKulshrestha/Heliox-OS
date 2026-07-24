import { writable } from "svelte/store";
import type { GazeRegion } from "../gesture/gazeTracking";

export type GazeRuntimePhase = "idle" | "loading" | "scanning" | "active" | "error";
export type GazeDaemonStatus = "idle" | "sending" | "ingested" | "ignored" | "error";

export interface GazeRuntimeState {
  phase: GazeRuntimePhase;
  cameraActive: boolean;
  region: GazeRegion | null;
  confidence: number | null;
  daemonStatus: GazeDaemonStatus;
  message: string;
}

export const initialGazeRuntimeState: GazeRuntimeState = {
  phase: "idle",
  cameraActive: false,
  region: null,
  confidence: null,
  daemonStatus: "idle",
  message: "",
};

export const gazeRuntime = writable<GazeRuntimeState>({ ...initialGazeRuntimeState });

export function updateGazeRuntime(values: Partial<GazeRuntimeState>): void {
  gazeRuntime.update((current) => ({ ...current, ...values }));
}

export function resetGazeRuntime(): void {
  gazeRuntime.set({ ...initialGazeRuntimeState });
}
