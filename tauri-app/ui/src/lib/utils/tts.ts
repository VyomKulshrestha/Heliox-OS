import { call } from "../api/daemon";

/**
 * Shared text-to-speech helper. The daemon is authoritative so the engine
 * and voice selected in Settings apply to voice replies, narration, and
 * supervision. Browser speech remains an availability fallback.
 */

export interface SpeakOptions {
  rate?: number;
  pitch?: number;
  volume?: number;
  onStart?: () => void;
  onEnd?: () => void;
  onError?: () => void;
}

const PREFERRED_VOICE_NAMES = ["Microsoft Mark", "Google UK English Male", "Daniel", "Alex"];
let speechGeneration = 0;

function speakWithBrowser(
  text: string,
  options: SpeakOptions,
  generation: number,
): void {
  if (!window.speechSynthesis) {
    if (generation === speechGeneration) options.onError?.();
    return;
  }
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = options.rate ?? 1.1;
  utterance.pitch = options.pitch ?? 0.9;
  utterance.volume = options.volume ?? 0.9;

  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find((v) => PREFERRED_VOICE_NAMES.some((name) => v.name.includes(name)));
  if (preferred) utterance.voice = preferred;

  utterance.onend = () => {
    if (generation === speechGeneration) options.onEnd?.();
  };
  utterance.onerror = () => {
    if (generation === speechGeneration) options.onError?.();
  };

  window.speechSynthesis.speak(utterance);
}

/** Speak through the configured local daemon engine, falling back to the
 * browser only when the daemon is unavailable. A newer call supersedes any
 * older one, including stale completion callbacks. */
export function speakText(text: string, options: SpeakOptions = {}): void {
  const trimmed = text.trim();
  if (!trimmed) return;

  const generation = ++speechGeneration;
  window.speechSynthesis?.cancel();
  options.onStart?.();

  void call<{ status: string }>("speak_text", { text: trimmed })
    .then((result) => {
      if (generation !== speechGeneration) return;
      if (result.status !== "spoken") throw new Error("Daemon rejected speech");
      options.onEnd?.();
    })
    .catch(() => {
      if (generation !== speechGeneration) return;
      speakWithBrowser(trimmed, options, generation);
    });
}

/** Stop both possible playback paths. */
export function stopSpeech(): void {
  speechGeneration += 1;
  window.speechSynthesis?.cancel();
  void call("stop_speech").catch(() => {});
}
