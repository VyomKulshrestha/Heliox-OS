import { fireEvent, render, screen } from "@testing-library/svelte";
import { tick } from "svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "./i18n";

// App is tested with every child component replaced by a stub that records its props, mounts and unmounts, and
// with the stores it reads replaced by plain writable stores the tests drive.
const harness = vi.hoisted(() => {
  function store<T>(initial: T) {
    let value = initial;
    const subscribers = new Set<(next: T) => void>();
    return {
      subscribe(run: (next: T) => void) {
        subscribers.add(run);
        run(value);
        return () => {
          subscribers.delete(run);
        };
      },
      set(next: T) {
        value = next;
        subscribers.forEach((run) => run(value));
      },
      update(change: (current: T) => T) {
        this.set(change(value));
      },
      get value() {
        return value;
      },
    };
  }

  const baseSession = () => ({
    activeSessionId: "chat-1",
    daemonConnected: false,
    loading: false,
    messages: [],
    currentPlan: null,
    confirmRequired: false,
    confirmPlanId: "",
    confirmActions: [] as Array<Record<string, unknown>>,
    confirmReason: "",
    confirmRiskAssessment: null,
    confirmGnomon: null,
    confirmBackground: false,
    confirmSubmitting: false,
    confirmError: "",
    confirmSelection: null as { planId: string; indices: number[] } | null,
    confirmVisible: false,
    confirmVisibleSince: null as number | null,
    phase: "",
    interaction: null,
    liveActions: [],
    totalTokens: 0,
    estimatedCost: 0,
    streamingText: "",
    budget: null,
    rollback: null,
    rollbackPending: false,
    proactiveSuggestion: null,
    proactiveSuggestionPending: false,
    terminalStatus: "",
    durableTask: undefined,
    queuedTasks: [],
  });

  const sessionStore = store(baseSession());
  const studyActive = store(false);
  const narration = store({ active: false });
  const neural = store<{ preview: unknown }>({ preview: null });
  const supervision = store({ active: false });
  const settings = store<Record<string, unknown>>({ first_run_complete: true });

  const mounts: Record<string, number> = {};
  const unmounts: Record<string, number> = {};
  const props: Record<string, Record<string, any>> = {};

  async function stubModule(name: string) {
    const { createRawSnippet } = await import("svelte");
    function Stub(anchor: unknown, componentProps: Record<string, any>) {
      props[name] = componentProps;
      const snippet = createRawSnippet(() => ({
        render: () => `<div data-stub="${name}"></div>`,
        setup: () => {
          mounts[name] = (mounts[name] ?? 0) + 1;
          return () => {
            unmounts[name] = (unmounts[name] ?? 0) + 1;
          };
        },
      }));
      (snippet as unknown as (target: unknown) => void)(anchor);
      // Instance methods App calls through bind:this (ParticleBurst, VirtualList).
      return { gestureBurst() {}, confirmBurst() {}, errorBurst() {}, scrollToBottom() {} };
    }
    return { default: Stub };
  }

  return {
    baseSession,
    sessionStore,
    studyActive,
    narration,
    neural,
    supervision,
    settings,
    mounts,
    unmounts,
    props,
    stubModule,
    reset() {
      for (const record of [mounts, unmounts, props]) {
        for (const key of Object.keys(record)) delete record[key];
      }
      sessionStore.set(baseSession());
      studyActive.set(false);
      narration.set({ active: false });
      neural.set({ preview: null });
      supervision.set({ active: false });
    },
  };
});

const sessionApi = vi.hoisted(() => ({
  confirm: vi.fn(async () => {}),
  // Both setters write the store like the real ones, so a component that reports on every store change would loop.
  setConfirmSelection: vi.fn((planId: string, indices: number[]) => {
    harness.sessionStore.update((current) => ({ ...current, confirmSelection: { planId, indices: [...indices] } }));
  }),
  setConfirmVisible: vi.fn((visible: boolean) => {
    if (harness.sessionStore.value.confirmVisible === visible) return;
    harness.sessionStore.update((current) => ({ ...current, confirmVisible: visible }));
  }),
  addSystemMessage: vi.fn(),
  sendCommand: vi.fn(async () => {}),
  requestRollback: vi.fn(),
  cancelRollback: vi.fn(),
  confirmRollback: vi.fn(async () => {}),
  exportChat: vi.fn(async () => {}),
  abort: vi.fn(async () => {}),
  resumeDurableTask: vi.fn(async () => {}),
  acceptProactiveSuggestion: vi.fn(async () => {}),
  dismissProactiveSuggestion: vi.fn(async () => {}),
}));

const coadaptationApi = vi.hoisted(() => ({
  load: vi.fn(async () => {}),
  exitStudy: vi.fn(async () => {}),
}));

vi.mock("./lib/stores/session", () => ({
  session: { subscribe: harness.sessionStore.subscribe, ...sessionApi },
}));
vi.mock("./lib/stores/settings", () => ({
  settings: { subscribe: harness.settings.subscribe, updateSection: vi.fn(async () => true) },
}));
vi.mock("./lib/stores/narration", () => ({ narration: { subscribe: harness.narration.subscribe } }));
vi.mock("./lib/stores/neural", () => ({
  neural: { subscribe: harness.neural.subscribe, deferPreviewForSafetyDecision: vi.fn(async () => {}) },
}));
vi.mock("./lib/stores/supervision", () => ({ supervision: { subscribe: harness.supervision.subscribe } }));
vi.mock("./lib/stores/coadaptation", () => ({
  coadaptation: { subscribe: vi.fn(() => () => {}), ...coadaptationApi },
  studyActive: { subscribe: harness.studyActive.subscribe },
}));

vi.mock("./lib/components/CommandInput.svelte", () => harness.stubModule("CommandInput"));
vi.mock("./lib/components/Dashboard.svelte", () => harness.stubModule("Dashboard"));
vi.mock("./lib/components/RollbackDialog.svelte", () => harness.stubModule("RollbackDialog"));
vi.mock("./lib/components/BudgetExceededDialog.svelte", () => harness.stubModule("BudgetExceededDialog"));
vi.mock("./lib/components/InterruptDialog.svelte", () => harness.stubModule("InterruptDialog"));
vi.mock("./lib/components/SupervisionAlertDialog.svelte", () => harness.stubModule("SupervisionAlertDialog"));
vi.mock("./lib/components/HumanControlBanner.svelte", () => harness.stubModule("HumanControlBanner"));
vi.mock("./lib/components/ActivityLog.svelte", () => harness.stubModule("ActivityLog"));
vi.mock("./lib/components/SettingsPanel.svelte", () => harness.stubModule("SettingsPanel"));
vi.mock("./lib/components/SetupWizard.svelte", () => harness.stubModule("SetupWizard"));
vi.mock("./lib/components/VoiceControl.svelte", () => harness.stubModule("VoiceControl"));
vi.mock("./lib/components/GestureControl.svelte", () => harness.stubModule("GestureControl"));
vi.mock("./lib/components/AirHandoffControl.svelte", () => harness.stubModule("AirHandoffControl"));
vi.mock("./lib/components/AmbientHUD.svelte", () => harness.stubModule("AmbientHUD"));
vi.mock("./lib/components/ArcReactor.svelte", () => harness.stubModule("ArcReactor"));
vi.mock("./lib/components/NeuralBackground.svelte", () => harness.stubModule("NeuralBackground"));
vi.mock("./lib/components/ParticleBurst.svelte", () => harness.stubModule("ParticleBurst"));
vi.mock("./lib/components/ExecutionGraph.svelte", () => harness.stubModule("ExecutionGraph"));
vi.mock("./lib/components/AgentRunCard.svelte", () => harness.stubModule("AgentRunCard"));
vi.mock("./lib/components/VirtualList.svelte", () => harness.stubModule("VirtualList"));
vi.mock("./lib/components/PluginsTab.svelte", () => harness.stubModule("PluginsTab"));
vi.mock("./lib/components/GitConflictResolver.svelte", () => harness.stubModule("GitConflictResolver"));
vi.mock("./lib/components/ScrollToBottom.svelte", () => harness.stubModule("ScrollToBottom"));
vi.mock("./lib/components/ConnectionBadge.svelte", () => harness.stubModule("ConnectionBadge"));
vi.mock("./lib/components/HeaderMiniMonitor.svelte", () => harness.stubModule("HeaderMiniMonitor"));
vi.mock("./lib/components/CommandHistory.svelte", () => harness.stubModule("CommandHistory"));
vi.mock("./lib/components/ChatSessionControls.svelte", () => harness.stubModule("ChatSessionControls"));
vi.mock("./lib/components/InteractiveServices.svelte", () => harness.stubModule("InteractiveServices"));
vi.mock("./lib/components/NeuralControlOverlay.svelte", () => harness.stubModule("NeuralControlOverlay"));
vi.mock("./lib/components/coadaptation/SelectionStudy.svelte", () => harness.stubModule("SelectionStudy"));

import App from "./App.svelte";

let visibility: DocumentVisibilityState = "visible";

function setSession(patch: Record<string, unknown>) {
  harness.sessionStore.update((current) => ({ ...current, ...patch }));
}

function requireApproval(planId: string) {
  setSession({
    confirmRequired: true,
    confirmPlanId: planId,
    confirmActions: [{ index: 0, action_type: "file_write", target: "C:/Temp/a.txt", parameters: {} }],
  });
}

function tab(name: string) {
  return screen.getByRole("button", { name });
}

function activeTabName(): string {
  return document.querySelector("nav.tabs .tab.active")?.textContent?.trim() ?? "";
}

async function settle() {
  await tick();
  await tick();
}

describe("App shell and the selection study", () => {
  beforeEach(() => {
    harness.reset();
    vi.clearAllMocks();
    visibility = "visible";
    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => visibility });
  });

  afterEach(() => {
    delete (document as unknown as { visibilityState?: unknown }).visibilityState;
  });

  it("shows the study tab only while a study session is active", async () => {
    render(App);
    expect(screen.queryByRole("button", { name: "Study" })).toBeNull();
    expect(harness.mounts.SelectionStudy).toBeUndefined();

    harness.studyActive.set(true);
    await settle();
    expect(tab("Study").classList.contains("active")).toBe(true);
    expect(harness.mounts.SelectionStudy).toBe(1);
    expect(harness.props.SelectionStudy.paused).toBe(false);

    harness.studyActive.set(false);
    await settle();
    expect(screen.queryByRole("button", { name: "Study" })).toBeNull();
    expect(harness.unmounts.SelectionStudy).toBe(1);
    expect(activeTabName()).toBe("Settings");
  });

  it("asks the study store to exit when the study reports it is done", async () => {
    render(App);
    harness.studyActive.set(true);
    await settle();
    const onexit = harness.props.SelectionStudy.onexit as () => void;
    onexit();
    expect(coadaptationApi.exitStudy).toHaveBeenCalledTimes(1);

    // A study that already ended its session itself is not ended a second time.
    harness.studyActive.set(false);
    await settle();
    onexit();
    expect(coadaptationApi.exitStudy).toHaveBeenCalledTimes(1);
  });

  it("keeps the study mounted but paused on another tab", async () => {
    render(App);
    harness.studyActive.set(true);
    await settle();

    await fireEvent.click(tab("Settings"));
    await settle();
    expect(harness.mounts.SelectionStudy).toBe(1);
    expect(harness.unmounts.SelectionStudy).toBeUndefined();
    expect(harness.props.SelectionStudy.paused).toBe(true);
    expect(document.querySelector(".study-panel")?.classList.contains("inactive")).toBe(true);

    await fireEvent.click(tab("Study"));
    await settle();
    expect(harness.props.SelectionStudy.paused).toBe(false);
  });

  it("switches from the study to Command for a blocking safety decision and pauses the study", async () => {
    render(App);
    harness.studyActive.set(true);
    await settle();
    expect(activeTabName()).toBe("Study");

    requireApproval("plan-safety");
    await settle();
    expect(activeTabName()).toBe("Command");
    expect(harness.props.SelectionStudy.paused).toBe(true);
    expect(harness.mounts.SelectionStudy).toBe(1);
    expect(harness.unmounts.SelectionStudy).toBeUndefined();
  });

  it("switches from the study to Command for any other App modal", async () => {
    render(App);
    harness.studyActive.set(true);
    await settle();

    harness.supervision.set({ active: true });
    await settle();
    expect(activeTabName()).toBe("Command");
    expect(harness.props.SelectionStudy.paused).toBe(true);
  });

  it("does not switch away from Settings for a safety decision, so unsaved edits stay", async () => {
    render(App);
    await fireEvent.click(tab("Settings"));
    await settle();
    expect(harness.mounts.SettingsPanel).toBe(1);

    requireApproval("plan-while-editing");
    await settle();
    expect(activeTabName()).toBe("Settings");
    expect(harness.mounts.SettingsPanel).toBe(1);
    expect(harness.unmounts.SettingsPanel).toBeUndefined();
  });

  it("keeps SelectionStudy mounted across a daemon reconnect", async () => {
    render(App);
    harness.studyActive.set(true);
    await settle();
    await fireEvent.click(tab("Settings"));
    await settle();
    expect(harness.mounts.SettingsPanel).toBe(1);

    setSession({ daemonConnected: true });
    await settle();
    setSession({ daemonConnected: false });
    await settle();
    setSession({ daemonConnected: true });
    await settle();

    // The connection-epoch block re-rendered (Settings remounted) while the study stayed mounted.
    expect(harness.mounts.SettingsPanel).toBe(3);
    expect(harness.mounts.SelectionStudy).toBe(1);
    expect(harness.unmounts.SelectionStudy).toBeUndefined();
  });

  it("loads the co-adaptation status on mount and on every daemon (re)connection", async () => {
    render(App);
    await settle();
    expect(coadaptationApi.load).toHaveBeenCalledTimes(1);

    setSession({ daemonConnected: true });
    await settle();
    expect(coadaptationApi.load).toHaveBeenCalledTimes(2);

    setSession({ daemonConnected: false });
    await settle();
    setSession({ daemonConnected: true });
    await settle();
    expect(coadaptationApi.load).toHaveBeenCalledTimes(3);
  });

  it("ignores swipe and call_me tab navigation while the study tab is shown", async () => {
    render(App);
    const onGesture = harness.props.GestureControl.onGesture as (gesture: string) => void;

    onGesture("swipe_right");
    await settle();
    expect(activeTabName()).toBe("Activity");
    await fireEvent.click(tab("Command"));

    harness.studyActive.set(true);
    await settle();
    for (const gesture of ["swipe_left", "swipe_right", "two_finger_swipe_left", "two_finger_swipe_right", "call_me"]) {
      onGesture(gesture);
      await settle();
      expect(activeTabName()).toBe("Study");
    }
  });

  it("reports the dialog visible only on the Command tab with the document visible", async () => {
    render(App);
    const lastVisible = () => sessionApi.setConfirmVisible.mock.calls.at(-1)?.[0];
    await settle();
    expect(lastVisible()).toBe(false);

    requireApproval("plan-visible");
    await settle();
    expect(lastVisible()).toBe(true);

    await fireEvent.click(tab("Settings"));
    await settle();
    expect(lastVisible()).toBe(false);

    await fireEvent.click(tab("Command"));
    await settle();
    expect(lastVisible()).toBe(true);

    visibility = "hidden";
    document.dispatchEvent(new Event("visibilitychange"));
    await settle();
    expect(lastVisible()).toBe(false);

    visibility = "visible";
    document.dispatchEvent(new Event("visibilitychange"));
    await settle();
    expect(lastVisible()).toBe(true);

    setSession({ confirmRequired: false, confirmPlanId: "" });
    await settle();
    expect(lastVisible()).toBe(false);
  });

  it("remounts ConfirmDialog for a new plan and wires its callbacks with the dialog channel", async () => {
    render(App);
    requireApproval("plan-a");
    await settle();

    // The real dialog reports its selection into the (mocked, store-writing) session without looping.
    const firstDialog = screen.getByRole("dialog");
    expect(sessionApi.setConfirmSelection).toHaveBeenLastCalledWith("plan-a", [0]);
    expect(harness.sessionStore.value.confirmSelection).toEqual({ planId: "plan-a", indices: [0] });
    const callsAfterMount = sessionApi.setConfirmSelection.mock.calls.length;
    setSession({ phase: "awaiting approval" });
    await settle();
    expect(sessionApi.setConfirmSelection.mock.calls.length).toBe(callsAfterMount);

    await fireEvent.click(screen.getByRole("checkbox"));
    expect(sessionApi.setConfirmSelection).toHaveBeenLastCalledWith("plan-a", []);
    await fireEvent.click(screen.getByRole("checkbox"));
    await fireEvent.click(screen.getByRole("button", { name: /approve/i }));
    expect(sessionApi.confirm).toHaveBeenLastCalledWith(true, [0], { inputChannel: "dialog" });
    await fireEvent.click(screen.getByRole("button", { name: "Deny" }));
    expect(sessionApi.confirm).toHaveBeenLastCalledWith(false, undefined, { inputChannel: "dialog" });

    // Uncheck, then a new plan with the same number of actions arrives: a fresh dialog with fresh checkboxes.
    await fireEvent.click(screen.getByRole("checkbox"));
    expect((screen.getByRole("checkbox") as HTMLInputElement).checked).toBe(false);
    requireApproval("plan-b");
    await settle();
    const secondDialog = screen.getByRole("dialog");
    expect(secondDialog).not.toBe(firstDialog);
    expect(firstDialog.isConnected).toBe(false);
    expect((screen.getByRole("checkbox") as HTMLInputElement).checked).toBe(true);
    expect(sessionApi.setConfirmSelection).toHaveBeenLastCalledWith("plan-b", [0]);
  });

  it("minimises the gesture camera picture-in-picture while a study is active", async () => {
    render(App);
    const host = document.querySelector(".gesture-control-host");
    expect(host?.classList.contains("study-minimized")).toBe(false);

    harness.studyActive.set(true);
    await settle();
    expect(host?.classList.contains("study-minimized")).toBe(true);

    harness.studyActive.set(false);
    await settle();
    expect(host?.classList.contains("study-minimized")).toBe(false);
  });
});
