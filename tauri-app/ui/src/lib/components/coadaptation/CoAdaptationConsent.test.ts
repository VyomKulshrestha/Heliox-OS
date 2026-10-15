import { createHash } from "node:crypto";
import { cleanup, fireEvent, render, screen } from "@testing-library/svelte";
import { locale } from "svelte-i18n";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import "../../../i18n";
import hi from "../../../locales/coadaptation.hi.json";
import CoAdaptationConsent from "./CoAdaptationConsent.svelte";
import { coadaptation } from "../../stores/coadaptation";
import { englishConsentText } from "../../coadaptation/consentText";

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
const NEW_KEY = "p-0a1b2c3d";
const EXPIRES = "2026-12-30T12:00:00+00:00";
const PROTOCOL_SHA = "b".repeat(64);

function statusReply() {
  return {
    status: "ok",
    available: true,
    config: { enabled: true },
    keyring_available: true,
    participants: [
      {
        participant_key: PK,
        label: null,
        consent: {
          consent_id: "c-1",
          purposes: ["selection_study", "gaze_prior", "research_export"],
          granted_at: "2026-10-01T12:00:00+00:00",
          expires_at: EXPIRES,
          retention_days: 90,
          consent_text_version: "coadapt-consent-v1",
          consent_text_sha256: "0".repeat(64),
          protocol_id: "coadapt-study-v1",
          active: true,
        },
        sessions: 1,
        last_session_at: null,
      },
    ],
    preferences: [],
    proposals: [],
    protocol: { protocol_id: "coadapt-study-v1", sha256: PROTOCOL_SHA },
    authority: "presentation_only",
    execution_authority: false,
  };
}

function grantCalls() {
  return daemonMocks.call.mock.calls.filter(([name]) => name === "co_adaptation_consent_grant");
}

function expectedHash(): string {
  return createHash("sha256").update(englishConsentText(), "utf8").digest("hex");
}

beforeEach(async () => {
  daemonMocks.call.mockReset();
  daemonMocks.call.mockImplementation(async (method: string, params: Record<string, unknown>) => {
    if (method === "co_adaptation_status") return statusReply();
    if (method === "co_adaptation_consent_grant") {
      return {
        status: "ok",
        participant_key: params.participant_key ?? NEW_KEY,
        consent: statusReply().participants[0].consent,
        config: { enabled: true },
        removed: { purposes: [], export_files: 0, trials_rewritten: 0, sessions_rewritten: 0 },
      };
    }
    throw new Error(`unexpected call ${method}`);
  });
  await coadaptation.load();
});

afterEach(async () => {
  cleanup();
  locale.set("en");
  await coadaptation.exitStudy();
});

describe("new consent", () => {
  it("starts with every purpose unticked and keeps Grant disabled until the study purpose is ticked", async () => {
    render(CoAdaptationConsent, { onclose: vi.fn(), ongranted: vi.fn() });
    const boxes = screen.getAllByRole("checkbox") as HTMLInputElement[];
    expect(boxes.map((box) => box.name)).toEqual([
      "selection_study",
      "adaptive_assistance",
      "cross_session_learning",
      "gaze_prior",
      "hand_pointer",
      "research_export",
    ]);
    expect(boxes.every((box) => !box.checked)).toBe(true);
    const grant = screen.getByRole("button", { name: "Grant" }) as HTMLButtonElement;
    expect(grant.disabled).toBe(true);
    await fireEvent.click(boxes[5]);
    expect(grant.disabled).toBe(true);
    await fireEvent.click(boxes[0]);
    expect(grant.disabled).toBe(false);
  });

  it("sends the text hash, the purposes, the protocol hash and the new participant's label in the grant only", async () => {
    const ongranted = vi.fn();
    const { container } = render(CoAdaptationConsent, { onclose: vi.fn(), ongranted });
    await fireEvent.click(container.querySelector('input[name="selection_study"]')!);
    await fireEvent.click(container.querySelector('input[name="hand_pointer"]')!);
    await fireEvent.change(screen.getByRole("combobox"), { target: { value: "180" } });
    await fireEvent.input(container.querySelector('input[type="text"]')!, { target: { value: "Lab B" } });
    await fireEvent.click(screen.getByRole("button", { name: "Grant" }));

    await vi.waitFor(() => expect(ongranted).toHaveBeenCalledWith(NEW_KEY));
    expect(grantCalls()).toHaveLength(1);
    expect(grantCalls()[0][1]).toEqual({
      participant_key: null,
      label: "Lab B",
      purposes: ["selection_study", "hand_pointer"],
      retention_days: 180,
      consent_text_version: "coadapt-consent-v1",
      consent_text_sha256: expectedHash(),
      protocol_id: "coadapt-study-v1",
      protocol_sha256: PROTOCOL_SHA,
      authorized: true,
      enable: true,
    });
    const otherCalls = daemonMocks.call.mock.calls.filter(([name]) => name !== "co_adaptation_consent_grant");
    expect(JSON.stringify(otherCalls)).not.toContain("Lab B");
  });

  it("closing the sheet grants nothing", async () => {
    const onclose = vi.fn();
    render(CoAdaptationConsent, { onclose, ongranted: vi.fn() });
    await fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onclose).toHaveBeenCalledTimes(1);
    expect(grantCalls()).toHaveLength(0);
  });

  it("shows the exact consent text, with no claim that Heliox learns which help works", () => {
    const { container } = render(CoAdaptationConsent, { onclose: vi.fn(), ongranted: vi.fn() });
    const text = container.querySelector(".consent-text")?.textContent ?? "";
    expect(text).toContain("Assistance study: how Heliox helps you select things");
    expect(text).toContain("estimates which help suits you");
    expect(text).not.toMatch(/learns? (what|which) (help )?works/i);
    expect(englishConsentText()).not.toMatch(/learns? (what|which) (help )?works/i);
  });

  it("shows the reviewed English-only notice in Hindi, above the English consent text", async () => {
    locale.set("hi");
    render(CoAdaptationConsent, { onclose: vi.fn(), ongranted: vi.fn() });
    const notice = (hi as { coadaptation: { consent: { english_only_notice: string } } }).coadaptation.consent
      .english_only_notice;
    expect(screen.getByText(notice)).toBeTruthy();
    expect(screen.getByText("Assistance study: how Heliox helps you select things")).toBeTruthy();
  });
});

describe("update consent", () => {
  it("pre-ticks the current purposes and states the deletion note with the unchanged expiry date", async () => {
    const { container } = render(CoAdaptationConsent, { participantKey: PK, onclose: vi.fn(), ongranted: vi.fn() });
    const checked = (Array.from(container.querySelectorAll('input[type="checkbox"]')) as HTMLInputElement[])
      .filter((box) => box.checked)
      .map((box) => box.name);
    expect(checked).toEqual(["selection_study", "gaze_prior", "research_export"]);
    const date = new Date(EXPIRES).toLocaleDateString();
    expect(
      screen.getByText(
        `Unticking an item deletes what was stored only for it. Your data are still erased on ${date}; updating never extends that date.`,
      ),
    ).toBeTruthy();
    expect(container.querySelector('input[type="text"]')).toBeNull();
  });

  it("lists what each unticked item deletes and sends no label", async () => {
    const ongranted = vi.fn();
    const { container } = render(CoAdaptationConsent, { participantKey: PK, onclose: vi.fn(), ongranted });
    await fireEvent.click(container.querySelector('input[name="research_export"]')!);
    await fireEvent.click(container.querySelector('input[name="gaze_prior"]')!);
    expect(screen.getByText("Unticking deletes:")).toBeTruthy();
    expect(screen.getByText(/Every encrypted export saved for this profile\./)).toBeTruthy();
    expect(screen.getByText(/Your gaze-check score and the gaze region stored with each selection\./)).toBeTruthy();

    await fireEvent.click(screen.getByRole("button", { name: "Update consent" }));
    await vi.waitFor(() => expect(ongranted).toHaveBeenCalledWith(PK));
    expect(grantCalls()[0][1]).toMatchObject({ participant_key: PK, label: null, purposes: ["selection_study"] });
  });

  it("keeps granted items ticked while a study session is open", async () => {
    const { container } = render(CoAdaptationConsent, {
      participantKey: PK,
      sessionOpen: true,
      onclose: vi.fn(),
      ongranted: vi.fn(),
    });
    const exportBox = container.querySelector('input[name="research_export"]') as HTMLInputElement;
    const handBox = container.querySelector('input[name="hand_pointer"]') as HTMLInputElement;
    expect(exportBox.checked).toBe(true);
    expect(exportBox.disabled).toBe(true);
    expect(handBox.disabled).toBe(false);
    expect(screen.getByText("Stop the running session before removing a permission.")).toBeTruthy();
  });
});
