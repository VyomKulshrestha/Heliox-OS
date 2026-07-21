/**
 * Shared browser text-to-speech helper (Web Speech API), extracted from
 * VoiceControl.svelte so the Live Execution Narrator's narration/interrupt
 * store can speak too, without duplicating the voice-selection logic.
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

/** Speaks `text` aloud via the browser's speechSynthesis, cancelling any
 * currently-speaking utterance first -- calling this again immediately
 * supersedes whatever was being said, which is how a higher-priority
 * narration/interrupt naturally pre-empts an in-progress one. */
export function speakText(text: string, options: SpeakOptions = {}): void {
  if (!window.speechSynthesis) return;

  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = options.rate ?? 1.1;
  utterance.pitch = options.pitch ?? 0.9;
  utterance.volume = options.volume ?? 0.9;

  const voices = window.speechSynthesis.getVoices();
  const preferred = voices.find((v) => PREFERRED_VOICE_NAMES.some((name) => v.name.includes(name)));
  if (preferred) utterance.voice = preferred;

  utterance.onstart = () => options.onStart?.();
  utterance.onend = () => options.onEnd?.();
  utterance.onerror = () => options.onError?.();

  window.speechSynthesis.speak(utterance);
}
