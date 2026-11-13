import { call, isConnected } from "./api/daemon";

// Heliox's window tells the daemon it may be used soon, so a released local voice starts loading before there is
// anything to say (voice_prepare). Only when connected, at most once every two seconds, and a failure is ignored:
// this is a hint, never something to wait for.
const EVERY_MS = 2000;
let last = Number.NEGATIVE_INFINITY;

export function voiceMayBeWanted(reason: "focus" | "typing", now: number = Date.now()): boolean {
  if (now - last < EVERY_MS || !isConnected()) return false;
  last = now;
  void call("voice_prepare", { reason }).catch(() => undefined);
  return true;
}

export function resetVoiceUseForTests(): void {
  last = Number.NEGATIVE_INFINITY;
}
