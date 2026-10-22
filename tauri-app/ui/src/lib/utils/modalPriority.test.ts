import { describe, expect, it } from "vitest";
import { selectActiveModal, type ModalSignals } from "./modalPriority";

const inactive: ModalSignals = {
  confirmation: false,
  rollback: false,
  undo: false,
  interrupt: false,
  neural: false,
  budget: false,
  supervision: false,
};

describe("Heliox modal priority", () => {
  it("keeps an execution confirmation above every competing feature", () => {
    expect(
      selectActiveModal({
        confirmation: true,
        rollback: true,
        undo: true,
        interrupt: true,
        neural: true,
        budget: true,
        supervision: true,
      }),
    ).toBe("confirmation");
  });

  it("queues advisory and budget modals behind a neural cancellation window", () => {
    expect(selectActiveModal({ ...inactive, neural: true, budget: true, supervision: true })).toBe("neural");
  });

  it("shows an undo preview after a rollback question and before advisories", () => {
    expect(selectActiveModal({ ...inactive, rollback: true, undo: true })).toBe("rollback");
    expect(selectActiveModal({ ...inactive, undo: true, interrupt: true, neural: true })).toBe("undo");
  });

  it("shows queued states in deterministic order as higher priorities clear", () => {
    expect(selectActiveModal({ ...inactive, interrupt: true, budget: true, supervision: true })).toBe("interrupt");
    expect(selectActiveModal({ ...inactive, budget: true, supervision: true })).toBe("budget");
    expect(selectActiveModal({ ...inactive, supervision: true })).toBe("supervision");
    expect(selectActiveModal(inactive)).toBeNull();
  });
});
