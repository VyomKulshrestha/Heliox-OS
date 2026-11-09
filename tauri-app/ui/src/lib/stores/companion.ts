import { writable } from "svelte/store";
import { speakText, stopSpeech, type SpeakOptions } from "../utils/tts";

export type CompanionChannel =
  | "emergency_stop"
  | "user_speech"
  | "approval_risk"
  | "live_correction"
  | "task_failure"
  | "final_answer"
  | "task_narration"
  | "proactive_suggestion"
  | "background_insight";

export interface CompanionUtterance {
  channel: CompanionChannel;
  text: string;
  dedupeKey?: string;
  taskId?: string;
  onStart?: () => void;
  onEnd?: () => void;
  onError?: () => void;
}

export interface CompanionState {
  activeChannel: CompanionChannel | "";
  activeText: string;
  queued: number;
  humanSpeaking: boolean;
  spokenCount: number;
  preemptionCount: number;
  duplicateCount: number;
}

interface QueuedUtterance extends CompanionUtterance {
  id: number;
  priority: number;
  key: string;
}

interface SpeechAdapter {
  speak(text: string, options: SpeakOptions): void;
  stop(): void;
}

export const COMPANION_PRIORITY: Record<CompanionChannel, number> = {
  emergency_stop: 700,
  user_speech: 600,
  approval_risk: 500,
  live_correction: 450,
  task_failure: 400,
  final_answer: 350,
  task_narration: 300,
  proactive_suggestion: 200,
  background_insight: 100,
};

const INITIAL_STATE: CompanionState = {
  activeChannel: "",
  activeText: "",
  queued: 0,
  humanSpeaking: false,
  spokenCount: 0,
  preemptionCount: 0,
  duplicateCount: 0,
};

const MAX_QUEUE = 8;
const DEDUPE_WINDOW_MS = 2500;

export function createCompanionCoordinator(
  adapter: SpeechAdapter = {
    speak: speakText,
    stop: stopSpeech,
  },
  now: () => number = Date.now,
) {
  const store = writable<CompanionState>({ ...INITIAL_STATE });
  let state = { ...INITIAL_STATE };
  let active: QueuedUtterance | null = null;
  let queue: QueuedUtterance[] = [];
  let nextId = 0;
  const recentlySubmitted = new Map<string, number>();

  function publish(patch: Partial<CompanionState> = {}) {
    state = {
      ...state,
      ...patch,
      activeChannel: active?.channel ?? "",
      activeText: active?.text ?? "",
      queued: queue.length,
    };
    store.set(state);
  }

  function normalizedKey(utterance: CompanionUtterance): string {
    return (
      utterance.dedupeKey?.trim() ||
      `${utterance.channel}:${utterance.text.trim().toLocaleLowerCase().replace(/\s+/g, " ")}`
    );
  }

  function pruneRecent(timestamp: number) {
    for (const [key, submittedAt] of recentlySubmitted) {
      if (timestamp - submittedAt > DEDUPE_WINDOW_MS) recentlySubmitted.delete(key);
    }
  }

  function finish(id: number, failed = false) {
    if (active?.id !== id) return;
    const completed = active;
    active = null;
    if (failed) completed.onError?.();
    else completed.onEnd?.();
    publish({ spokenCount: state.spokenCount + (failed ? 0 : 1) });
    startNext();
  }

  function start(utterance: QueuedUtterance) {
    active = utterance;
    publish();
    adapter.speak(utterance.text, {
      channel: utterance.channel,
      dedupeKey: utterance.key,
      onStart: utterance.onStart,
      onEnd: () => finish(utterance.id),
      onError: () => finish(utterance.id, true),
    });
  }

  function startNext() {
    if (active || state.humanSpeaking || queue.length === 0) {
      publish();
      return;
    }
    const next = queue.shift()!;
    start(next);
  }

  function enqueue(utterance: QueuedUtterance) {
    if (utterance.channel === "task_narration" && utterance.taskId) {
      queue = queue.filter((queued) => queued.channel !== "task_narration" || queued.taskId !== utterance.taskId);
    }
    queue.push(utterance);
    queue.sort((left, right) => right.priority - left.priority || left.id - right.id);
    if (queue.length > MAX_QUEUE) queue = queue.slice(0, MAX_QUEUE);
    publish();
  }

  function speak(utterance: CompanionUtterance): boolean {
    const text = utterance.text.trim();
    if (!text) return false;
    const timestamp = now();
    pruneRecent(timestamp);
    const key = normalizedKey({ ...utterance, text });
    if (recentlySubmitted.has(key) || active?.key === key || queue.some((queued) => queued.key === key)) {
      publish({ duplicateCount: state.duplicateCount + 1 });
      return false;
    }
    recentlySubmitted.set(key, timestamp);

    const request: QueuedUtterance = {
      ...utterance,
      text,
      id: ++nextId,
      priority: COMPANION_PRIORITY[utterance.channel],
      key,
    };

    if (request.channel === "emergency_stop") {
      queue = [];
      if (active) {
        adapter.stop();
        active = null;
        state.preemptionCount += 1;
      }
      start(request);
      return true;
    }

    if (state.humanSpeaking) {
      enqueue(request);
      return true;
    }

    if (!active) {
      start(request);
      return true;
    }

    if (request.priority > active.priority) {
      adapter.stop();
      active = null;
      state.preemptionCount += 1;
      if (request.priority >= COMPANION_PRIORITY.task_failure) {
        queue = queue.filter((queued) => queued.priority >= request.priority);
      }
      start(request);
      return true;
    }

    enqueue(request);
    return true;
  }

  function humanSpeechStarted() {
    queue = [];
    if (active && active.channel !== "emergency_stop") {
      adapter.stop();
      active = null;
      state.preemptionCount += 1;
    }
    publish({ humanSpeaking: true });
  }

  function humanSpeechEnded() {
    publish({ humanSpeaking: false });
    startNext();
  }

  function stopAll() {
    queue = [];
    if (active) {
      adapter.stop();
      active = null;
    }
    publish({ humanSpeaking: false });
  }

  function resetMetrics() {
    publish({
      spokenCount: 0,
      preemptionCount: 0,
      duplicateCount: 0,
    });
  }

  return {
    subscribe: store.subscribe,
    speak,
    humanSpeechStarted,
    humanSpeechEnded,
    stopAll,
    resetMetrics,
  };
}

export const companion = createCompanionCoordinator();
