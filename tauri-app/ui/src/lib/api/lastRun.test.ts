import { beforeEach, describe, expect, it } from "vitest";
import { lastRunLine, resetLastRunForTests } from "./lastRun";

function memory(): Pick<Storage, "getItem" | "setItem"> {
  const values = new Map<string, string>();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => void values.set(key, value),
  };
}

const unclean = {
  status: "ok",
  state: "ready",
  last_run: {
    ended: "did not end cleanly",
    notice: "Heliox's last run (started 14:02) did not end cleanly. It is running again.",
    id: "1763000000",
  },
};

describe("the last run's notice", () => {
  beforeEach(() => resetLastRunForTests());

  it("is said once per unclean end, also across reconnects and a reloaded window", () => {
    const store = memory();
    expect(lastRunLine(unclean, store)).toBe(unclean.last_run.notice);
    expect(lastRunLine(unclean, store)).toBeNull();
    resetLastRunForTests(); // a reloaded window: only the stored id remembers
    expect(lastRunLine(unclean, store)).toBeNull();
    const next = { ...unclean, last_run: { ...unclean.last_run, id: "1763000999" } };
    expect(lastRunLine(next, store)).toBe(unclean.last_run.notice);
  });

  it("says nothing for a computer restart, a clean end or an older daemon", () => {
    const store = memory();
    const restarted = { last_run: { ended: "computer restarted", notice: null, id: "1" } };
    expect(lastRunLine(restarted, store)).toBeNull();
    expect(lastRunLine({ status: "ok", state: "ready" }, store)).toBeNull();
    expect(lastRunLine(undefined, store)).toBeNull();
  });

  it("still says it once when storage cannot be used", () => {
    const broken = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };
    expect(lastRunLine(unclean, broken)).toBe(unclean.last_run.notice);
    expect(lastRunLine(unclean, broken)).toBeNull();
  });
});

describe("the time in the last run's notice", () => {
  beforeEach(() => resetLastRunForTests());

  it("is the person's own time zone and format, not the daemon's", () => {
    const zone = process.env.TZ;
    process.env.TZ = "Asia/Kolkata"; // UTC+5:30: the daemon's 22:10 UTC is 03:40 the next morning here
    try {
      const reply = {
        last_run: {
          notice: "Heliox's last run (started 22:10) did not end cleanly. It is running again.",
          id: "a",
          started_at: Date.UTC(2026, 10, 12, 22, 10) / 1000,
        },
      };
      expect(lastRunLine(reply, null, "en-GB")).toBe(
        "Heliox's last run (started 03:40) did not end cleanly. It is running again.",
      );
      resetLastRunForTests();
      expect(lastRunLine({ last_run: { ...reply.last_run, id: "b" } }, null, "en-US")).toBe(
        "Heliox's last run (started 3:40 AM) did not end cleanly. It is running again.",
      );
    } finally {
      if (zone === undefined) delete process.env.TZ;
      else process.env.TZ = zone;
    }
  });

  it("keeps the daemon's words when it sent no time", () => {
    const reply = { last_run: { notice: "Heliox's last run did not end cleanly. It is running again.", id: "c" } };
    expect(lastRunLine(reply, null)).toBe("Heliox's last run did not end cleanly. It is running again.");
  });
});
