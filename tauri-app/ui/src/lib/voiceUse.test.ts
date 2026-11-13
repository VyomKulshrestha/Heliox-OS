import { beforeEach, describe, expect, it, vi } from "vitest";

const call = vi.fn((..._args: unknown[]) => Promise.resolve({ status: "ok" }));
let connected = true;
vi.mock("./api/daemon", () => ({ call: (...args: unknown[]) => call(...args), isConnected: () => connected }));

import { resetVoiceUseForTests, voiceMayBeWanted } from "./voiceUse";

describe("voiceMayBeWanted", () => {
  beforeEach(() => {
    call.mockClear();
    connected = true;
    resetVoiceUseForTests();
  });

  it("tells the daemon once, and not again within two seconds", () => {
    expect(voiceMayBeWanted("focus", 10_000)).toBe(true);
    expect(voiceMayBeWanted("typing", 11_000)).toBe(false);
    expect(voiceMayBeWanted("typing", 12_500)).toBe(true);
    expect(call.mock.calls).toEqual([
      ["voice_prepare", { reason: "focus" }],
      ["voice_prepare", { reason: "typing" }],
    ]);
  });

  it("sends nothing while the daemon is not connected", () => {
    connected = false;
    expect(voiceMayBeWanted("focus", 10_000)).toBe(false);
    expect(call).not.toHaveBeenCalled();
  });
});
