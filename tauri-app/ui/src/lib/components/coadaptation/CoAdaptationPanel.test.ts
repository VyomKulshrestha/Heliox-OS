import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import CoAdaptationPanel from "./CoAdaptationPanel.svelte";
import { coadaptation } from "../../stores/coadaptation";

type NotificationHandler = (method: string, params: unknown) => void;

const daemonMocks = vi.hoisted(() => ({
  call: vi.fn(),
  handlers: [] as Array<(method: string, params: unknown) => void>,
}));

vi.mock("../../api/daemon", () => ({
  call: daemonMocks.call,
  onNotification: (handler: NotificationHandler) => {
    daemonMocks.handlers.push(handler);
  },
  offNotification: vi.fn(),
  onConnectionState: vi.fn(),
}));

const PK = "p-3fa29c1e";

const CONFIG = {
  enabled: true,
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
};

function statusReply(overrides: Record<string, unknown> = {}) {
  return {
    status: "ok",
    available: true,
    config: { ...CONFIG },
    keyring_available: true,
    store: {
      exists: true,
      bytes: 2048,
      participants: 1,
      sessions: 2,
      decisions: 40,
      trials: 30,
      burden_reports: 4,
      oldest_recorded_at: null,
      pruned_sessions: 0,
    },
    participants: [
      {
        participant_key: PK,
        label: "Lab A",
        consent: {
          consent_id: "c-1",
          purposes: ["selection_study", "research_export"],
          granted_at: "2026-10-01T00:00:00+00:00",
          expires_at: "2099-01-01T00:00:00+00:00",
          retention_days: 90,
          consent_text_version: "coadapt-consent-v1",
          consent_text_sha256: "0".repeat(64),
          protocol_id: "coadapt-study-v1",
          active: true,
        },
        sessions: 2,
        last_session_at: null,
      },
    ],
    preferences: [],
    proposals: [],
    protocol: { protocol_id: "coadapt-study-v1", sha256: "a".repeat(64) },
    authority: "presentation_only",
    execution_authority: false,
    ...overrides,
  };
}

let replies: Record<string, (params: Record<string, unknown>) => unknown> = {};

beforeEach(() => {
  daemonMocks.call.mockReset();
  replies = {
    co_adaptation_status: () => statusReply(),
    co_adaptation_config_update: () => ({ status: "ok", config: CONFIG, restart_required: false }),
    co_adaptation_results: () => ({
      status: "ok",
      evidence_kind: "live_human",
      claim: "Your data on this device; not a general result.",
      per_regime: [],
      calibration: {},
      probe_trend: [],
      drift_flags: [],
      allowed_now: { eligible: [], excluded: {} },
      predictions_now: {},
    }),
    co_adaptation_consent_revoke: () => ({
      status: "ok",
      erased: { rows: 1, key_destroyed: true, export_files: 0, rules_deleted: 0 },
      config: CONFIG,
    }),
    co_adaptation_erase: () => ({ status: "ok", erased: { rows: 3, export_files: 1 }, session_ended: false }),
  };
  daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    const reply = replies[method];
    if (!reply) throw new Error(`unexpected call ${method}`);
    return reply(params);
  });
});

afterEach(async () => {
  cleanup();
  await coadaptation.exitStudy();
});

async function renderLoaded() {
  const view = render(CoAdaptationPanel);
  await screen.findByText("Participants and consent");
  return view;
}

function calls(method: string) {
  return daemonMocks.call.mock.calls.filter(([name]) => name === method);
}

describe("CoAdaptationPanel", () => {
  it("loads the status and always shows the authority note", async () => {
    await renderLoaded();
    expect(calls("co_adaptation_status").length).toBeGreaterThan(0);
    expect(screen.getByText("Assistance & co-adaptation")).toBeTruthy();
    expect(
      screen.getByText(
        "Changes how targets are shown and what a selection needs from you. It never approves, clicks or runs anything for you; permissions and approval dialogs are unchanged.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("Lab A")).toBeTruthy();
    expect(screen.getByText("Anyone using Heliox on this device can see and erase these participants.")).toBeTruthy();
  });

  it("has a subtitle that does not claim Heliox learns what works for the person", async () => {
    await renderLoaded();
    const subtitle = screen.getByText(/Tries different kinds of selection help/);
    expect(subtitle.textContent).toContain("After a few sessions the estimate still rests mostly on general data.");
    expect(subtitle.textContent).not.toMatch(/learns? (what|which)/i);
  });

  it("shows the older-daemon message when the status method is missing", async () => {
    replies.co_adaptation_status = () => {
      throw new Error("Method not found: co_adaptation_status");
    };
    render(CoAdaptationPanel);
    expect(
      await screen.findByText("This app is connected to an older Heliox daemon. Restart the daemon, then retry."),
    ).toBeTruthy();
  });

  it("Save sends only the changed Advanced fields, and says changes apply from the next session", async () => {
    const { container } = await renderLoaded();
    await fireEvent.click(screen.getByText("Advanced"));
    const note = screen.getByText(/^Changes apply from the next study session\./);
    // Study sessions pin the floor and the penalty and take the other three only where they are stricter.
    expect(note.textContent).toContain("study protocol's exploration floor and switching penalty");
    expect(note.textContent).toContain("only where it is stricter than the protocol's");
    const tolerance = container.querySelector('input[name="unintended_activation_tolerance"]') as HTMLInputElement;
    await fireEvent.change(tolerance, { target: { value: "0.03" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => expect(calls("co_adaptation_config_update")).toHaveLength(1));
    expect(calls("co_adaptation_config_update")[0][1]).toEqual({ unintended_activation_tolerance: 0.03 });
  });

  it("never lets the ambiguity threshold exceed 0.25", async () => {
    const { container } = await renderLoaded();
    const threshold = container.querySelector('input[name="ambiguity_threshold"]') as HTMLInputElement;
    expect(threshold.max).toBe("0.25");
    await fireEvent.change(threshold, { target: { value: "0.4" } });
    expect(threshold.value).toBe("0.25");
    expect((screen.getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(true);
    await fireEvent.change(threshold, { target: { value: "0.2" } });
    await fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await vi.waitFor(() => expect(calls("co_adaptation_config_update")).toHaveLength(1));
    expect(calls("co_adaptation_config_update")[0][1]).toEqual({ ambiguity_threshold: 0.2 });
  });

  it("opens the consent sheet with every purpose unchecked and no approval purpose", async () => {
    await renderLoaded();
    await fireEvent.click(screen.getByRole("button", { name: "New participant" }));
    const boxes = screen.getAllByRole("checkbox") as HTMLInputElement[];
    expect(boxes).toHaveLength(6);
    expect(boxes.every((box) => !box.checked)).toBe(true);
    for (const box of boxes) expect(box.closest("label")?.textContent ?? "").not.toMatch(/approv/i);
    expect((screen.getByRole("button", { name: "Grant" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("withdraws only after a second click and the typed word, with the exact rule-deletion text", async () => {
    await renderLoaded();
    await fireEvent.click(screen.getByRole("button", { name: "Withdraw and erase" }));
    expect(
      screen.getByText(
        "This erases all study data for this participant and deletes the rules set only for them. Rules set for everyone on this device stay.",
      ),
    ).toBeTruthy();
    expect(calls("co_adaptation_consent_revoke")).toHaveLength(0);
    const confirm = screen.getByRole("button", { name: "Erase everything for this participant" }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    const word = screen.getByLabelText("Type erase to confirm") as HTMLInputElement;
    await fireEvent.input(word, { target: { value: "eras" } });
    expect(confirm.disabled).toBe(true);
    await fireEvent.input(word, { target: { value: "erase" } });
    expect(confirm.disabled).toBe(false);
    await fireEvent.click(confirm);
    await vi.waitFor(() => expect(calls("co_adaptation_consent_revoke")).toHaveLength(1));
    expect(calls("co_adaptation_consent_revoke")[0][1]).toEqual({ participant_key: PK });
  });

  it("says when a withdrawal left the keyring key or files to delete later, until the status reports them gone", async () => {
    replies.co_adaptation_consent_revoke = () => ({
      status: "ok",
      erased: { rows: 4, key_destroyed: false, export_files: 1, rules_deleted: 0, export_files_left: 0, pending: true },
    });
    await renderLoaded();
    replies.co_adaptation_status = () => statusReply({ erasure_pending: true });
    await fireEvent.click(screen.getByRole("button", { name: "Withdraw and erase" }));
    await fireEvent.input(screen.getByLabelText("Type erase to confirm"), { target: { value: "erase" } });
    await fireEvent.click(screen.getByRole("button", { name: "Erase everything for this participant" }));
    expect(await screen.findByText(/this participant's key is still in secure key storage/)).toBeTruthy();
    expect(screen.queryByText(/Some erased data still awaits deletion/)).toBeNull();
    // The next status, whose prune finished the deletion, clears the warning.
    replies.co_adaptation_status = () => statusReply({ erasure_pending: false });
    await fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    await vi.waitFor(() => expect(screen.queryByText(/still in secure key storage/)).toBeNull());
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("names export files or rules when only those are left, and shows the status's pending erasure", async () => {
    replies.co_adaptation_consent_revoke = () => ({
      status: "ok",
      erased: { rows: 4, key_destroyed: true, export_files: 2, rules_deleted: 0, export_files_left: 1, pending: true },
    });
    await renderLoaded();
    replies.co_adaptation_status = () => statusReply({ erasure_pending: true });
    await fireEvent.click(screen.getByRole("button", { name: "Withdraw and erase" }));
    await fireEvent.input(screen.getByLabelText("Type erase to confirm"), { target: { value: "erase" } });
    await fireEvent.click(screen.getByRole("button", { name: "Erase everything for this participant" }));
    expect(
      await screen.findByText(/an export file or a rule for this participant could not be deleted yet/),
    ).toBeTruthy();
    cleanup();
    // Without a withdrawal in this view (an expiry, an erase-today), the status alone shows what is pending.
    render(CoAdaptationPanel);
    expect(await screen.findByText(/Some erased data still awaits deletion/)).toBeTruthy();
  });

  it("Erase today asks twice and its confirmation mentions exports", async () => {
    await renderLoaded();
    await fireEvent.click(screen.getByRole("button", { name: "Erase today" }));
    expect(calls("co_adaptation_erase")).toHaveLength(0);
    expect(
      screen.getByText(/This erases today's sessions and any export saved since today's first session started\./),
    ).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Click again to confirm" }));
    await vi.waitFor(() => expect(calls("co_adaptation_erase")).toHaveLength(1));
    expect(calls("co_adaptation_erase")[0][1]).toEqual({ participant_key: PK, scope: "today" });
  });

  it("Run a selection session opens the study", async () => {
    await renderLoaded();
    await fireEvent.click(screen.getByRole("button", { name: "Run a selection session" }));
    expect((await import("svelte/store")).get(coadaptation).study.active).toBe(true);
  });

  it("disables an option whose purpose was not granted, with an Update consent offer", async () => {
    await renderLoaded();
    const adaptive = screen.getByRole("switch", { name: "Adapt during later sessions" }) as HTMLButtonElement;
    expect(adaptive.disabled).toBe(true);
    expect(screen.getByText("Needs consent for Let Heliox adapt during later sessions")).toBeTruthy();
  });
});
