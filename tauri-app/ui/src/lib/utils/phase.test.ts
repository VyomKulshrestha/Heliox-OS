import { describe, expect, it } from "vitest";
import { actionTitle, describePhase, formatElapsed } from "./phase";

describe("describePhase", () => {
  it("names each daemon phase in plain words with its stage", () => {
    expect(describePhase("recalling memory")).toEqual({
      label: "Recalling what's relevant",
      stage: 0,
      waitingOnYou: false,
    });
    expect(describePhase("critic review").stage).toBe(1);
    expect(describePhase("executing")).toMatchObject({ label: "Working on it", stage: 2 });
    expect(describePhase("verifying")).toMatchObject({ label: "Checking the result", stage: 3 });
  });

  it("keeps the numbers in retry and revision phases", () => {
    expect(describePhase("re-planning (attempt 2)").label).toBe("Re-planning, attempt 2");
    expect(describePhase("companion revising plan (1/3)").label).toBe("Revising the plan, 1 of 3");
    expect(describePhase("retrying failed actions (attempt 3)").label).toBe("Retrying failed steps, attempt 3");
  });

  it("names the plan reviews still running, keeping their capitals", () => {
    expect(describePhase("reviewing plan (double-check, safety review, Gnomon)")).toEqual({
      label: "Reviewing the plan: double-check, safety review, Gnomon",
      stage: 1,
      waitingOnYou: false,
    });
    expect(describePhase("reviewing plan (Gnomon)").label).toBe("Reviewing the plan: Gnomon");
  });

  it("flags phases that wait on the person", () => {
    expect(describePhase("awaiting visible approval")).toMatchObject({
      label: "Waiting for your approval",
      waitingOnYou: true,
    });
    expect(describePhase("Waiting for your approval").waitingOnYou).toBe(true);
  });

  it("passes the interaction runtime's own sentences through and still places them", () => {
    expect(describePhase("Planning the safest useful action")).toMatchObject({
      label: "Planning the safest useful action",
      stage: 1,
    });
    expect(describePhase("Executing the approved plan").stage).toBe(2);
    expect(describePhase("Verifying the result").stage).toBe(3);
  });

  it("falls back to sentence case without dashes, and to Thinking when empty", () => {
    expect(describePhase("retrying — previous attempt failed").label).toBe("Retrying after a failed attempt");
    expect(describePhase("warming_up — models").label).toBe("Warming up, models");
    expect(describePhase("").label).toBe("Thinking");
    expect(describePhase(undefined).stage).toBe(0);
  });
});

describe("formatElapsed and actionTitle", () => {
  it("formats minutes and seconds", () => {
    expect(formatElapsed(7)).toBe("0:07");
    expect(formatElapsed(102.9)).toBe("1:42");
    expect(formatElapsed(-3)).toBe("0:00");
  });

  it("turns action types into sentence-case titles", () => {
    expect(actionTitle("open_application")).toBe("Open application");
    expect(actionTitle("SCREEN_OCR")).toBe("Screen OCR");
    expect(actionTitle("browser_navigate_url")).toBe("Browser navigate URL");
  });
});
