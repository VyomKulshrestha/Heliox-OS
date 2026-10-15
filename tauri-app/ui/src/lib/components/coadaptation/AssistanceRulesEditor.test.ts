import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import AssistanceRulesEditor from "./AssistanceRulesEditor.svelte";
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
const MAX_COPY =
  "At most 4 extra steps per block. A confirm or hold step counts once per selection, even if you cancel or let go and try again; each question and each extra confirm Heliox adds counts.";

let preferences: Array<Record<string, unknown>> = [];
let proposals: Array<Record<string, unknown>> = [];
let setReplies: Array<Record<string, unknown>> = [];

function statusReply() {
  return {
    status: "ok",
    available: true,
    config: { enabled: true },
    participants: [
      {
        participant_key: PK,
        label: null,
        consent: {
          consent_id: "c",
          purposes: ["selection_study"],
          granted_at: "2026-10-01T00:00:00+00:00",
          expires_at: "2099-01-01T00:00:00+00:00",
          retention_days: 90,
          consent_text_version: "coadapt-consent-v1",
          consent_text_sha256: "0".repeat(64),
          protocol_id: "coadapt-study-v1",
          active: true,
        },
        sessions: 0,
        last_session_at: null,
      },
    ],
    preferences,
    proposals,
    authority: "presentation_only",
    execution_authority: false,
  };
}

function rule(prefId: string, kind: string, value: unknown, participantKey: string | null = PK) {
  return {
    pref_id: prefId,
    kind,
    scope: { context: "study", modality: null, participant_key: participantKey },
    value,
    origin: "explicit_user",
    created_at: "2026-10-15T09:00:00+00:00",
    retracted_at: null,
  };
}

function ok() {
  return {
    status: "ok",
    preference: rule("pref-new", "forbid_variant", "ask_clarification:3"),
    notes: [],
    refresh_required: false,
    refresh_seq: null,
    fallbacks: null,
  };
}

function conflict(message: string, conflicting: Array<{ pref_id: string; summary: string }>) {
  return { status: "error", reason: "preference_conflict", message, conflicting };
}

function calls(method: string) {
  return daemonMocks.call.mock.calls.filter(([name]) => name === method);
}

beforeEach(async () => {
  preferences = [];
  proposals = [];
  setReplies = [];
  daemonMocks.call.mockReset();
  daemonMocks.call.mockImplementation(async (method: string) => {
    if (method === "co_adaptation_status") return statusReply();
    if (method === "co_adaptation_preference_set") return setReplies.shift() ?? ok();
    if (method === "co_adaptation_preference_retract") {
      return { status: "ok", refresh_required: false, refresh_seq: null, fallbacks: null };
    }
    throw new Error(`unexpected call ${method}`);
  });
  await coadaptation.load();
});

afterEach(cleanup);

describe("adding and removing rules", () => {
  it("saves a rule with the chosen scope and retracts an active one", async () => {
    preferences = [rule("pref-1", "pin_act", true, null)];
    await coadaptation.load();
    render(AssistanceRulesEditor, { participantKey: PK });

    expect(screen.getByText("Always select exactly what I click", { selector: ".summary" })).toBeTruthy();
    await fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_retract")).toHaveLength(1));
    expect(calls("co_adaptation_preference_retract")[0][1]).toEqual({ pref_id: "pref-1" });

    await fireEvent.click(await screen.findByRole("button", { name: "Save rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0][1]).toEqual({
      kind: "forbid_variant",
      scope: { context: "study", modality: null, participant_key: PK },
      value: "ask_clarification:3",
    });
    expect(await screen.findByText("Rule saved.")).toBeTruthy();
  });

  it("offers the two scopes in the words of the spec and saves a device-wide rule", async () => {
    render(AssistanceRulesEditor, { participantKey: PK });
    expect(screen.getByLabelText("Only this participant (deleted when they withdraw)")).toBeTruthy();
    await fireEvent.click(screen.getByLabelText("Everyone on this device (kept after withdrawal)"));
    await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0][1]).toMatchObject({
      scope: { context: "study", modality: null, participant_key: null },
    });
  });

  it("shows the step-limit rule with the count of section 9.3", async () => {
    preferences = [rule("pref-cap", "max_interruptions_per_block", 4)];
    await coadaptation.load();
    render(AssistanceRulesEditor, { participantKey: PK });
    expect(screen.getByText(MAX_COPY)).toBeTruthy();

    await fireEvent.change(screen.getByLabelText("Rule"), { target: { value: "max_interruptions_per_block" } });
    expect(screen.getAllByText(MAX_COPY).length).toBe(2);
    await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0][1]).toMatchObject({
      kind: "max_interruptions_per_block",
      value: 4,
    });
  });

  it("shows the notes of an accepted rule", async () => {
    setReplies = [{ ...ok(), notes: ["Your minimum caution leaves only confirm steps, so the limit keeps them."] }];
    render(AssistanceRulesEditor, { participantKey: PK });
    await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
    expect(
      await screen.findByText("Note: Your minimum caution leaves only confirm steps, so the limit keeps them."),
    ).toBeTruthy();
  });

  it("has no control about approvals", () => {
    const { container } = render(AssistanceRulesEditor, { participantKey: PK });
    expect(container.textContent ?? "").not.toMatch(/approv/i);
  });

  it("never claims a rule reaches beyond the study: only the study reads these rules", async () => {
    preferences = [
      { ...rule("pref-all", "pin_act", true), scope: { context: "all", modality: null, participant_key: PK } },
    ];
    await coadaptation.load();
    const { container } = render(AssistanceRulesEditor, { participantKey: PK });
    const options = Array.from(container.querySelectorAll("option")).map((option) => option.textContent ?? "");
    const contextAll = options.find((text) => /study/i.test(text) && text !== "In the study");
    expect(contextAll).toBeTruthy();
    // The saved rule's scope uses the same words.
    expect(container.textContent ?? "").toContain(contextAll!);
    expect(container.textContent ?? "").not.toMatch(/everywhere/i);
  });
});

describe("conflicts", () => {
  const cases = [
    [
      "a freeze against a minimum caution",
      "Your minimum caution is Confirm each selection; Standard selection is below it. Remove that rule and save this one, or keep it.",
      "Minimum caution: Confirm each selection",
    ],
    [
      "a pin against another pin",
      "You already pinned Larger targets; pinning Hold to select would contradict it. Remove that rule and save this one, or keep it.",
      "Always use: Larger targets",
    ],
    [
      "a second freeze naming another regime",
      "You already paused adapting at Hold to select; pausing at Larger targets would contradict it. Remove that rule and save this one, or keep it.",
      "Adapting paused at: Hold to select",
    ],
    [
      "a freeze against a step limit",
      "You paused adapting at Hold to select, which asks for a step in every selection, more than this limit allows.",
      "At most 2 extra steps per block",
    ],
  ] as const;

  for (const [name, message, summary] of cases) {
    it(`shows ${name}, naming the conflicting rule`, async () => {
      setReplies = [conflict(message, [{ pref_id: "pref-old", summary }])];
      render(AssistanceRulesEditor, { participantKey: PK });
      await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
      expect(await screen.findByText(message)).toBeTruthy();
      expect(screen.getByText(summary)).toBeTruthy();
      expect(screen.getByRole("button", { name: "Remove that rule" })).toBeTruthy();
    });
  }

  it("Remove that rule retracts the named rule and then saves the new one", async () => {
    setReplies = [
      conflict("You already pinned Larger targets.", [{ pref_id: "pref-pin", summary: "Always use" }]),
      ok(),
    ];
    render(AssistanceRulesEditor, { participantKey: PK });
    await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Remove that rule" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(2));
    const order = daemonMocks.call.mock.calls
      .map(([name]) => name)
      .filter((name) => name === "co_adaptation_preference_set" || name === "co_adaptation_preference_retract");
    expect(order).toEqual([
      "co_adaptation_preference_set",
      "co_adaptation_preference_retract",
      "co_adaptation_preference_set",
    ]);
    expect(calls("co_adaptation_preference_retract")[0][1]).toEqual({ pref_id: "pref-pin" });
    expect(calls("co_adaptation_preference_set")[1][1]).toEqual(calls("co_adaptation_preference_set")[0][1]);
  });

  it("Keep it changes nothing", async () => {
    setReplies = [conflict("You already pinned Larger targets.", [{ pref_id: "pref-pin", summary: "Always use" }])];
    render(AssistanceRulesEditor, { participantKey: PK });
    await fireEvent.click(screen.getByRole("button", { name: "Save rule" }));
    await fireEvent.click(await screen.findByRole("button", { name: "Keep it" }));
    expect(screen.queryByText("You already pinned Larger targets.")).toBeNull();
    expect(calls("co_adaptation_preference_set")).toHaveLength(1);
    expect(calls("co_adaptation_preference_retract")).toHaveLength(0);
  });
});

describe("proposals", () => {
  it("writes nothing until the person clicks Stop using it", async () => {
    proposals = [
      { participant_key: PK, kind: "forbid_variant", variant: "ask_clarification:3", reason: "three_vetoes" },
    ];
    await coadaptation.load();
    render(AssistanceRulesEditor, { participantKey: PK });
    expect(
      screen.getByText('You chose "Not this time" for Ask which one (3 choices) three times. Stop using it?'),
    ).toBeTruthy();
    expect(calls("co_adaptation_preference_set")).toHaveLength(0);

    await fireEvent.click(screen.getByRole("button", { name: "Stop using it" }));
    await vi.waitFor(() => expect(calls("co_adaptation_preference_set")).toHaveLength(1));
    expect(calls("co_adaptation_preference_set")[0][1]).toEqual({
      kind: "forbid_variant",
      scope: { context: "study", modality: null, participant_key: PK },
      value: "ask_clarification:3",
    });
  });

  it("Keep it dismisses a proposal without writing", async () => {
    proposals = [{ participant_key: PK, kind: "forbid_variant", variant: "act", reason: "three_vetoes" }];
    await coadaptation.load();
    render(AssistanceRulesEditor, { participantKey: PK });
    await fireEvent.click(screen.getByRole("button", { name: "Keep it" }));
    expect(screen.queryByRole("button", { name: "Stop using it" })).toBeNull();
    expect(calls("co_adaptation_preference_set")).toHaveLength(0);
  });
});
