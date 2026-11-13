export type HelioxModal = "confirmation" | "undo" | "interrupt" | "neural" | "budget" | "supervision";

export interface ModalSignals {
  confirmation: boolean;
  undo: boolean;
  interrupt: boolean;
  neural: boolean;
  budget: boolean;
  supervision: boolean;
}

const PRIORITY: HelioxModal[] = ["confirmation", "undo", "interrupt", "neural", "budget", "supervision"];

export function selectActiveModal(signals: ModalSignals): HelioxModal | null {
  return PRIORITY.find((modal) => signals[modal]) ?? null;
}
