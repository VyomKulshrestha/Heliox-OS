import { get, writable } from "svelte/store";
import { call, onNotification, requireOkResult, type DaemonStatusResult } from "../api/daemon";
import { airHandoffGestureCommand } from "../gesture/airHandoffGesture";
import { resolveAirHandoffDelivery, type AirHandoffDeliveryEvent } from "../gesture/airHandoffDelivery";
import { reconcileAirHandoffTarget } from "../gesture/airHandoffTarget";

export interface AirHandoffDevice {
  device_id: string;
  name: string;
  created_at: number;
  last_seen_at: number;
  credential_available: boolean;
}

export interface AirHandoffDraft {
  draft_id: string;
  kind: string;
  filename: string;
  mime_type: string;
  size: number;
  expires_at: number;
}

export interface AirHandoffPairing {
  session_id: string;
  expires_at: number;
  pairing_url: string;
}

/** A task a paired phone asked for. It runs only when Run is pressed here, or at once if read-only requests may. */
export interface PhoneRequest {
  request_id: string;
  device_id: string;
  device_name: string;
  text: string;
  created_at: number;
  status: "waiting" | "running" | "done" | "failed" | "dismissed";
  job_id: string;
  result: string;
  read_only: boolean;
}

export interface AirHandoffState {
  enabled: boolean;
  accept_requests: boolean;
  run_read_only_requests: boolean;
  requests: PhoneRequest[];
  running: boolean;
  receiver_url: string | null;
  paired_devices: AirHandoffDevice[];
  pairing: AirHandoffPairing | null;
  draft: AirHandoffDraft | null;
  ready_transfers: number;
  recent: AirHandoffDeliveryEvent[];
  secure_storage_available: boolean;
  selectedDeviceId: string;
  awaitingTransferId: string;
  awaitingTargetName: string;
  gestureArmed: boolean;
  busy: boolean;
  error: string;
  message: string;
}

const selectedKey = "heliox_air_handoff_target";

const initialState: AirHandoffState = {
  enabled: false,
  accept_requests: false,
  run_read_only_requests: false,
  requests: [],
  running: false,
  receiver_url: null,
  paired_devices: [],
  pairing: null,
  draft: null,
  ready_transfers: 0,
  recent: [],
  secure_storage_available: false,
  selectedDeviceId: typeof localStorage === "undefined" ? "" : localStorage.getItem(selectedKey) || "",
  awaitingTransferId: "",
  awaitingTargetName: "",
  gestureArmed: false,
  busy: false,
  error: "",
  message: "",
};

function createAirHandoffStore() {
  const store = writable<AirHandoffState>(initialState);
  const { subscribe, update } = store;

  function mergeRemote(remote: Partial<AirHandoffState>): void {
    update((current) => {
      const devices = remote.paired_devices ?? current.paired_devices;
      const recent = remote.recent ?? current.recent;
      const target = reconcileAirHandoffTarget(current.selectedDeviceId, devices, current.gestureArmed);
      const { selectedDeviceId } = target;
      const selectedName = devices.find((device) => device.device_id === selectedDeviceId)?.name || "Phone";
      const delivery = resolveAirHandoffDelivery(
        current.awaitingTransferId,
        recent,
        current.awaitingTargetName || selectedName,
      );
      if (typeof localStorage !== "undefined") {
        if (selectedDeviceId) localStorage.setItem(selectedKey, selectedDeviceId);
        else localStorage.removeItem(selectedKey);
      }
      return {
        ...current,
        ...remote,
        paired_devices: devices,
        recent,
        selectedDeviceId,
        awaitingTransferId: delivery?.awaitingTransferId ?? current.awaitingTransferId,
        awaitingTargetName: delivery ? "" : current.awaitingTargetName,
        message:
          delivery?.message ??
          (target.targetChanged && current.gestureArmed
            ? "Selected phone changed; re-arm Air Handoff before sending."
            : (remote.message ?? current.message)),
        gestureArmed:
          remote.enabled === false || remote.running === false || !selectedDeviceId ? false : target.gestureArmed,
      };
    });
  }

  async function invoke<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    update((state) => ({ ...state, busy: true, error: "" }));
    try {
      const result = requireOkResult(
        await call<T & DaemonStatusResult>(method, params),
        `Air Handoff operation '${method}' was not acknowledged.`,
      );
      return result;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Air Handoff failed";
      update((state) => ({ ...state, error: message }));
      throw cause;
    } finally {
      update((state) => ({ ...state, busy: false }));
    }
  }

  async function refresh(): Promise<void> {
    try {
      const result = requireOkResult(
        await call<Partial<AirHandoffState> & DaemonStatusResult>("air_handoff_status"),
        "Air Handoff status is unavailable.",
      );
      if (
        typeof result.enabled !== "boolean" ||
        typeof result.running !== "boolean" ||
        typeof result.secure_storage_available !== "boolean" ||
        !Array.isArray(result.paired_devices) ||
        !Array.isArray(result.recent)
      ) {
        throw new Error("Air Handoff status is incomplete.");
      }
      mergeRemote({ ...result, error: "" });
    } catch (cause) {
      update((state) => ({
        ...state,
        running: false,
        gestureArmed: false,
        error: cause instanceof Error ? cause.message : "Cannot reach the Air Handoff service",
      }));
    }
  }

  async function setEnabled(enabled: boolean): Promise<void> {
    const result = await invoke<AirHandoffState>("air_handoff_set_enabled", { enabled });
    if (result.enabled !== enabled || result.running !== enabled) {
      throw new Error("Air Handoff receiver did not reach the requested state.");
    }
    mergeRemote(result);
  }

  async function configureRequests(accept: boolean, runReadOnly: boolean): Promise<void> {
    const result = await invoke<AirHandoffState>("air_handoff_requests_configure", {
      accept_requests: accept,
      run_read_only_requests: accept && runReadOnly,
    });
    mergeRemote(result);
  }

  function upsertRequest(request: PhoneRequest): void {
    update((state) => {
      const others = state.requests.filter((item) => item.request_id !== request.request_id);
      return { ...state, requests: [request, ...others].slice(0, 20) };
    });
  }

  async function runRequest(requestId: string): Promise<void> {
    const result = await invoke<{ request: PhoneRequest }>("phone_request_run", { request_id: requestId });
    if (result.request) upsertRequest(result.request);
  }

  async function dismissRequest(requestId: string): Promise<void> {
    const result = await invoke<{ request: PhoneRequest }>("phone_request_dismiss", { request_id: requestId });
    if (result.request) upsertRequest(result.request);
  }

  async function startPairing(): Promise<AirHandoffPairing> {
    const result = await invoke<AirHandoffPairing>("air_handoff_start_pairing");
    if (
      typeof result.session_id !== "string" ||
      !result.session_id ||
      typeof result.pairing_url !== "string" ||
      !result.pairing_url ||
      !Number.isFinite(Number(result.expires_at))
    ) {
      throw new Error("Air Handoff pairing response is incomplete.");
    }
    const pairing: AirHandoffPairing = {
      session_id: result.session_id,
      expires_at: result.expires_at,
      pairing_url: result.pairing_url,
    };
    mergeRemote({ pairing });
    return pairing;
  }

  async function cancelPairing(): Promise<void> {
    await invoke("air_handoff_cancel_pairing");
    mergeRemote({ pairing: null });
  }

  async function revokeDevice(deviceId: string): Promise<void> {
    await invoke("air_handoff_revoke_device", { device_id: deviceId });
    await refresh();
  }

  function selectDevice(deviceId: string): void {
    update((state) => ({ ...state, selectedDeviceId: deviceId, gestureArmed: false }));
    if (typeof localStorage !== "undefined") localStorage.setItem(selectedKey, deviceId);
  }

  function setGestureArmed(armed: boolean): void {
    update((state) => ({
      ...state,
      gestureArmed: armed && Boolean(state.selectedDeviceId) && state.running,
      message: armed ? "Make a fist to grab the screen" : "",
      error: "",
    }));
  }

  async function grabScreenshot(): Promise<void> {
    const result = await invoke<{ draft: AirHandoffDraft }>("air_handoff_grab", {
      kind: "screenshot",
    });
    if (!result.draft || typeof result.draft.draft_id !== "string" || !result.draft.draft_id) {
      throw new Error("Air Handoff did not return a captured screen draft.");
    }
    mergeRemote({ draft: result.draft, message: "Screen grabbed. Push an open palm to send it." });
  }

  async function dropToSelected(): Promise<void> {
    const state = get(store);
    if (!state.selectedDeviceId) throw new Error("Select a paired phone first");
    const result = await invoke<{ transfer: { transfer_id: string } }>("air_handoff_drop", {
      target_device_id: state.selectedDeviceId,
    });
    if (!result.transfer || typeof result.transfer.transfer_id !== "string" || !result.transfer.transfer_id) {
      throw new Error("Air Handoff did not return a queued transfer.");
    }
    const targetName =
      state.paired_devices.find((device) => device.device_id === state.selectedDeviceId)?.name || "selected phone";
    mergeRemote({
      draft: null,
      gestureArmed: false,
      awaitingTransferId: result.transfer.transfer_id,
      awaitingTargetName: targetName,
      message: `Queued securely for ${targetName}; waiting for receipt confirmation`,
    });
  }

  async function cancelDraft(): Promise<void> {
    await invoke("air_handoff_cancel");
    mergeRemote({ draft: null, gestureArmed: false, message: "Handoff cancelled" });
  }

  async function handleGesture(gesture: string): Promise<boolean> {
    const state = get(store);
    if (!state.gestureArmed || state.busy) return false;
    const command = airHandoffGestureCommand(state.gestureArmed, Boolean(state.draft), gesture);
    if (command === "grab") {
      await grabScreenshot();
      return true;
    }
    if (command === "drop") {
      await dropToSelected();
      return true;
    }
    if (command === "cancel") {
      await cancelDraft();
      return true;
    }
    return false;
  }

  onNotification((method, params) => {
    if (method === "phone_request" && params && typeof params === "object") {
      const request = params as PhoneRequest;
      if (typeof request.request_id === "string" && typeof request.text === "string") upsertRequest(request);
      return;
    }
    if (method === "air_handoff_state" && params && typeof params === "object") {
      try {
        const remote = requireOkResult(params as DaemonStatusResult, "Air Handoff status update was not acknowledged.");
        mergeRemote(remote as Partial<AirHandoffState>);
      } catch (cause) {
        update((state) => ({
          ...state,
          running: false,
          gestureArmed: false,
          error: cause instanceof Error ? cause.message : "Air Handoff status update failed",
        }));
      }
    }
  });

  return {
    subscribe,
    refresh,
    setEnabled,
    configureRequests,
    runRequest,
    dismissRequest,
    startPairing,
    cancelPairing,
    revokeDevice,
    selectDevice,
    setGestureArmed,
    grabScreenshot,
    dropToSelected,
    cancelDraft,
    handleGesture,
  };
}

export const airHandoff = createAirHandoffStore();
