<script lang="ts">
  /**
   * VoiceControl — JARVIS-like voice input using Web Speech API.
   * Push-to-talk or continuous listening with wake word "Hey Heliox".
   *
   * Features:
   *  - Real-time speech-to-text transcription
   *  - Visual pulse animation while listening
   *  - Auto-submit transcribed text as command
   *  - Text-to-speech for responses
   */

  import { session } from "../stores/session";
  import { companion } from "../stores/companion";
  import { multimodal } from "../stores/multimodal";
  import AudioVisualizer from "./AudioVisualizer.svelte";
  import { call, offNotification, onNotification } from "../api/daemon";
  import { onDestroy, onMount } from "svelte";
  import { get } from "svelte/store";
  import { settings } from "../stores/settings";

  // ── State ──
  let isListening = $state(false);
  let isSpeaking = $state(false);
  let transcript = $state("");
  let interimTranscript = $state("");
  let wakeWordActive = $state(false);
  let pulseIntensity = $state(0);
  let error = $state("");
  let pushToTalkEnabled = $state(true);
  let wakeWordBusy = $state(false);
  const continuousPreferenceKey = "heliox-continuous-listening-enabled";
  let listenerHealthTimer: ReturnType<typeof setInterval> | null = null;
  let listenerRetryAfter = 0;
  let listenerStartFailures = 0;

  function resetListenerRetry() {
    listenerRetryAfter = 0;
    listenerStartFailures = 0;
  }

  function deferListenerRetry() {
    listenerStartFailures += 1;
    const delayMs = Math.min(300_000, 15_000 * 2 ** Math.min(listenerStartFailures - 1, 5));
    listenerRetryAfter = Date.now() + delayMs;
  }

  // ── Speech Recognition ──
  let recognition: any = null;
  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

  function initRecognition() {
    if (!SpeechRecognition) {
      error = "Speech recognition not supported in this browser. Use Chrome or Edge.";
      return null;
    }

    const rec = new SpeechRecognition();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.maxAlternatives = 1;
    rec.onspeechstart = () => {
      // Human speech always owns the audio channel. This stops narration or
      // a task response before the recognized phrase is routed as a live
      // correction, preventing Heliox from talking over the user.
      companion.humanSpeechStarted();
      isSpeaking = false;
    };

    rec.onresult = (event: any) => {
      let interim = "";
      let final = "";

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const text = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += text;
        } else {
          interim += text;
        }
      }

      interimTranscript = interim;

      if (final) {
        companion.humanSpeechEnded();
        // Check for wake word in continuous mode
        if (wakeWordActive) {
          const lower = final.toLowerCase().trim();
          if (lower.includes("hey heliox") || lower.includes("hey heliox,") || lower.startsWith("heliox")) {
            // Extract command after wake word
            const command = lower
              .replace(/hey heliox[,]?\s*/i, "")
              .replace(/^heliox[,]?\s*/i, "")
              .trim();
            if (command) {
              transcript = command;
              submitVoiceCommand(command);
            }
          }
        } else {
          // Push-to-talk mode — submit everything
          transcript = final.trim();
          if (transcript) {
            submitVoiceCommand(transcript);
          }
        }
      }

      // Animate pulse based on interim text length
      pulseIntensity = Math.min(1, interim.length / 30);
    };

    rec.onerror = (event: any) => {
      if (event.error === "no-speech" || event.error === "aborted") {
        companion.humanSpeechEnded();
        return;
      }
      companion.humanSpeechEnded();
      console.error("Speech recognition error:", event.error);
      error = `Mic error: ${event.error}`;
      // Do not retry automatically after a permission error, but keep the
      // explicit push-to-talk action available. This lets the user grant
      // microphone access in OS/app settings and retry without reloading.
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        error = "Microphone access is blocked. Allow it in app or OS settings, then press Push to talk again.";
        recognition = null;
      }
      isListening = false;
    };

    rec.onend = () => {
      // Only restart if in wake word mode AND no error occurred AND voice is enabled
      isListening = false;
      pulseIntensity = 0;
      companion.humanSpeechEnded();
    };

    return rec;
  }

  function toggleListening() {
    if (wakeWordActive) {
      // Always-listening is on: the button means "my next sentence is for you", no wake phrase needed.
      void call("voice_listen_now").catch(() => {});
      pulseIntensity = 0.6;
      return;
    }
    if (isListening) {
      stopListening();
    } else {
      startListening(false);
    }
  }

  function startListening(continuous: boolean) {
    if (!recognition) {
      recognition = initRecognition();
    }
    if (!recognition) return;

    companion.humanSpeechStarted();
    isSpeaking = false;
    wakeWordActive = continuous;
    error = "";
    transcript = "";
    interimTranscript = "";

    try {
      recognition.start();
      isListening = true;
    } catch (e) {
      // Already started, restart
      recognition.stop();
      setTimeout(() => {
        try {
          recognition.start();
          isListening = true;
        } catch {
          /* ignore */
        }
      }, 100);
    }
  }

  function stopListening() {
    wakeWordActive = false;
    if (recognition) {
      recognition.stop();
    }
    isListening = false;
    pulseIntensity = 0;
    companion.humanSpeechEnded();
  }

  async function toggleWakeWord() {
    if (wakeWordBusy) return;
    wakeWordBusy = true;
    error = "";
    try {
      if (wakeWordActive) {
        const result = await call<{ status: string }>("voice_listener_stop");
        if (result.status !== "stopped" && result.status !== "not_running") {
          throw new Error("Daemon rejected wake-word stop");
        }
        wakeWordActive = false;
        isListening = false;
        localStorage.setItem(continuousPreferenceKey, "false");
      } else {
        resetListenerRetry();
        const result = await call<{ status: string; message?: string }>("voice_listener_start", {
          wake_words: ["hey heliox", "heliox", "hey pilot"],
        });
        if (result.status !== "started" && result.status !== "already_running") {
          throw new Error(result.message || "Daemon rejected wake-word start");
        }
        wakeWordActive = true;
        isListening = true;
        resetListenerRetry();
        localStorage.setItem(continuousPreferenceKey, "true");
      }
    } catch (err) {
      error = err instanceof Error ? err.message : "Wake-word listener unavailable";
    } finally {
      wakeWordBusy = false;
    }
  }

  // A live caption while the person talks (streaming engine), and "still listening" when they pause mid-thought.
  let holding = $state(false);
  // The local voice is loading (up to ~25 s from cold): say a spoken reply is coming instead of going silent.
  let preparingVoice = $state("");
  let inCall = $state("");
  // The microphone or speakers failed or changed: what Heliox is doing about it (cleared when sound flows again).
  let deviceNotice = $state("");
  let deviceTimer: ReturnType<typeof setTimeout> | null = null;
  // A headset in Bluetooth call mode: listen with another microphone instead, if the person presses the button.
  let inputOffer = $state<{ headset: string; id: string; name: string } | null>(null);
  // No speech model on this computer: a button to the download list in Settings, Voice (nothing downloads by itself).
  let modelsNeeded = $state(false);
  function openSpeechModels() {
    window.dispatchEvent(new CustomEvent("heliox:open-settings", { detail: { section: "voice-models" } }));
  }
  async function acceptInputOffer() {
    const offer = inputOffer;
    if (!offer) return;
    inputOffer = null;
    const choices = { ...(get(settings).voice?.headset_input_choices ?? {}), [offer.headset]: offer.id };
    const saved = await settings.updateSectionDetailed("voice", { headset_input_choices: choices });
    showDeviceNotice(
      saved.ok
        ? `Listening with ${offer.name} while ${offer.headset} is connected. Undo in Settings, Voice.`
        : `Not changed: ${saved.message}`,
      8000,
    );
  }
  function showDeviceNotice(text: string, clearAfterMs = 0) {
    deviceNotice = text;
    if (deviceTimer) clearTimeout(deviceTimer);
    deviceTimer = clearAfterMs ? setTimeout(() => (deviceNotice = ""), clearAfterMs) : null;
  }
  let preparingTimer: ReturnType<typeof setTimeout> | null = null;
  let captionTimer: ReturnType<typeof setTimeout> | null = null;
  function showCaption(text: string) {
    interimTranscript = text;
    if (captionTimer) clearTimeout(captionTimer);
    captionTimer = setTimeout(() => {
      interimTranscript = "";
      holding = false;
    }, 4000);
  }

  const voiceStatusHandler = (method: string, params: unknown) => {
    const payload = (params ?? {}) as Record<string, unknown>;
    if (method === "voice_partial") {
      holding = false;
      showCaption(String(payload.text ?? ""));
      return;
    }
    if (method === "voice_command") {
      holding = false;
      interimTranscript = "";
      return;
    }
    if (method !== "voice_status") return;
    const status = String(payload.status ?? "");
    if (status === "voice_loading") {
      preparingVoice = String(payload.message ?? "") || "Getting my voice ready…";
      if (preparingTimer) clearTimeout(preparingTimer);
      if (deviceTimer) clearTimeout(deviceTimer);
      preparingTimer = setTimeout(() => (preparingVoice = ""), 90000);
      return;
    }
    if (status === "in_call") {
      inCall = String(payload.message ?? "") || "Quiet while you're in a call";
      return;
    }
    if (status === "call_ended") {
      inCall = "";
      return;
    }
    if (status === "microphone_problem") {
      showDeviceNotice(String(payload.message ?? "") || "The microphone stopped; reopening it.");
      return;
    }
    if (status === "recogniser_missing") {
      showDeviceNotice(String(payload.message ?? "") || "Speech recognition needs a model.");
      modelsNeeded = true;
      return;
    }
    if (status === "microphone_quality") {
      showDeviceNotice(String(payload.message ?? ""));
      const suggested = payload.suggested_input as { id?: string; name?: string } | null | undefined;
      inputOffer =
        payload.reason === "narrowband" && suggested?.id && suggested.name && payload.headset
          ? { headset: String(payload.headset), id: String(suggested.id), name: String(suggested.name) }
          : null;
      return;
    }
    if (status === "microphone_recovered" || status === "speaker_changed") {
      showDeviceNotice(String(payload.message ?? ""), 6000);
      return;
    }
    if (status === "voice_ready" || status === "voice_unavailable") {
      preparingVoice = "";
      return;
    }
    if (status === "holding") {
      holding = true;
      showCaption(interimTranscript);
      pulseIntensity = 0.5;
      return;
    }
    if (status === "wake_detected" || status === "follow_up_detected") {
      companion.humanSpeechStarted();
      companion.humanSpeechEnded();
      transcript = String(payload.transcript ?? "");
      pulseIntensity = 1;
    } else if (status === "follow_up_ready") {
      pulseIntensity = 0.6;
    } else if (status === "timeout") {
      companion.humanSpeechEnded();
      error = String(payload.message ?? "Voice command timed out.");
      pulseIntensity = 0;
    } else if (status === "listening") {
      pulseIntensity = 0.4;
    }
  };

  const handleVoiceCommandRequest = (event: Event) => {
    const respond = (event as CustomEvent<{ respond?: (message: string) => void }>).detail?.respond;
    // The dashboard must reuse this component's recognizer. Starting another
    // Web Speech instance would contend with continuous listening and can
    // deliver the same phrase twice. When the daemon listener is active, it
    // already owns the microphone and the user can speak the wake phrase.
    if (wakeWordActive) {
      error = "";
      pulseIntensity = 0.6;
      respond?.("Always listening is active - say 'Hey Heliox' followed by your command");
      return;
    }
    if (!SpeechRecognition) {
      respond?.("Speech recognition is unavailable in this runtime");
      return;
    }
    startListening(false);
    respond?.("Listening for your Heliox command...");
  };

  onNotification(voiceStatusHandler);
  onDestroy(() => {
    if (captionTimer) clearTimeout(captionTimer);
    if (preparingTimer) clearTimeout(preparingTimer);
    if (deviceTimer) clearTimeout(deviceTimer);
    offNotification(voiceStatusHandler);
    window.removeEventListener("heliox:voice-command-request", handleVoiceCommandRequest);
    if (listenerHealthTimer) clearInterval(listenerHealthTimer);
  });

  async function ensureContinuousListener() {
    const preferred = localStorage.getItem(continuousPreferenceKey) !== "false";
    if (!preferred || wakeWordBusy || Date.now() < listenerRetryAfter) return;
    try {
      const stats = await call<{ running: boolean }>("voice_listener_stats");
      if (!stats.running) {
        const result = await call<{ status: string; message?: string }>("voice_listener_start", {
          wake_words: ["hey heliox", "heliox", "hey pilot"],
        });
        if (result.status !== "started" && result.status !== "already_running") {
          throw new Error(result.message || "Daemon rejected continuous listening");
        }
      }
      wakeWordActive = true;
      isListening = true;
      error = "";
      resetListenerRetry();
    } catch (err) {
      deferListenerRetry();
      error = err instanceof Error ? err.message : "Continuous voice listener unavailable";
    }
  }

  onMount(() => {
    window.addEventListener("heliox:voice-command-request", handleVoiceCommandRequest);
    void ensureContinuousListener();
    // Re-arm after a daemon restart or transient disconnect. The persisted
    // preference remains false when the user explicitly turns listening off.
    listenerHealthTimer = setInterval(() => void ensureContinuousListener(), 5_000);
  });

  async function submitVoiceCommand(text: string) {
    // Stop listening temporarily in push-to-talk mode
    if (!wakeWordActive) {
      stopListening();
    }

    const wasTaskRunning = get(session).loading;

    // Feed the same finalized transcript into the bounded fusion window
    // before execution. Fusion observes context/cancellation only; the
    // session command remains the sole action-producing path.
    multimodal.onVoiceInput(text, 0.9, true);
    try {
      await call("voice_event", { transcript: text, confidence: 0.9, is_final: true });
    } catch (cause) {
      console.warn("Voice fusion context unavailable; command routing continues safely.", cause);
    }

    // The session store routes speech to the same out-of-band correction
    // channel as typed input whenever a task is active.
    await session.sendCommand(text);

    // The original voice command already owns the one final-response
    // subscription. A spoken correction gets an immediate companion
    // acknowledgment from the narration channel but must not install a
    // second subscriber, or the final result would be spoken twice.
    if (wasTaskRunning) {
      transcript = "";
      interimTranscript = "";
      return;
    }

    // Speak the response if voice is enabled
    // Wait for the response, then use the configured daemon TTS engine.
    const unsub = session.subscribe((s: any) => {
      if (!s.loading && s.messages.length > 0) {
        const lastMsg = s.messages[s.messages.length - 1];
        if (lastMsg.type === "result" || lastMsg.type === "system") {
          speakText(lastMsg.text || "Done.", "final_answer");
          unsub();
        } else if (lastMsg.type === "error") {
          speakText("Error: " + (lastMsg.text || "Something went wrong."), "task_failure");
          unsub();
        }
      }
    });

    transcript = "";
    interimTranscript = "";
  }

  // ── Text-to-Speech ──
  function speakText(text: string, channel: "final_answer" | "task_failure") {
    companion.speak({
      channel,
      text,
      onStart: () => {
        isSpeaking = true;
      },
      onEnd: () => {
        isSpeaking = false;
      },
      onError: () => {
        isSpeaking = false;
      },
    });
  }

  function stopSpeaking() {
    companion.stopAll();
    isSpeaking = false;
  }

  // Check support on mount
  $effect(() => {
    if (!SpeechRecognition) {
      pushToTalkEnabled = false;
    }
    // Preload voices
    window.speechSynthesis?.getVoices();

    call<{ running: boolean }>("voice_listener_stats")
      .then((stats) => {
        wakeWordActive = Boolean(stats.running);
        if (stats.running) isListening = true;
      })
      .catch(() => {});
  });
</script>

<div class="voice-control">
  <!-- Push-to-Talk Button -->
  <button
    class="voice-btn"
    class:listening={isListening && !wakeWordActive}
    class:disabled={!pushToTalkEnabled && !wakeWordActive}
    onclick={toggleListening}
    disabled={!pushToTalkEnabled && !wakeWordActive}
    title={wakeWordActive ? "Talk to Heliox now" : isListening ? "Stop listening" : "Push to talk"}
    aria-label={wakeWordActive ? "Talk to Heliox now" : isListening ? "Stop listening" : "Push to talk"}
  >
    <div class="pulse-ring" style="--intensity: {pulseIntensity}"></div>
    <svg class="mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      {#if isListening && !wakeWordActive}
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" />
        <path d="M5 10a7 7 0 0 0 14 0" />
        <line x1="12" y1="17" x2="12" y2="21" />
      {:else}
        <rect x="9" y="2" width="6" height="12" rx="3" />
        <path d="M5 10a7 7 0 0 0 14 0" />
        <line x1="12" y1="17" x2="12" y2="21" />
        <line x1="8" y1="21" x2="16" y2="21" />
      {/if}
    </svg>
  </button>

  <!-- Audio Visualizer -->
  <AudioVisualizer active={isListening} />

  <!-- Wake Word Toggle -->
  <button
    class="wake-btn"
    class:active={wakeWordActive}
    class:disabled={wakeWordBusy}
    onclick={toggleWakeWord}
    disabled={wakeWordBusy}
    title={wakeWordActive ? "Disable continuous listening" : "Enable continuous listening"}
    aria-label={wakeWordActive ? "Disable continuous listening" : "Enable continuous listening"}
    aria-pressed={wakeWordActive}
  >
    <svg class="wave-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
      <path d="M8 12a4 4 0 0 0 8 0" />
      <circle cx="12" cy="10" r="1.5" fill="currentColor" />
    </svg>
    <span class="wake-label">{wakeWordActive ? "Always listening" : "Start listening"}</span>
  </button>

  <!-- Speaking Indicator -->
  {#if isSpeaking}
    <button class="speak-indicator" onclick={stopSpeaking} title="Click to stop speaking">
      <div class="speak-bars">
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
        <span class="bar"></span>
      </div>
    </button>
  {/if}

  <!-- Transcript Preview -->
  {#if interimTranscript || transcript || holding || preparingVoice || inCall || deviceNotice}
    <div class="transcript-preview" aria-live="polite">
      {#if transcript}
        <span class="final">{transcript}</span>
      {/if}
      {#if interimTranscript}
        <span class="interim">{interimTranscript}</span>
      {/if}
      {#if holding}
        <span class="holding">Still listening…</span>
      {/if}
      {#if preparingVoice}
        <span class="holding">{preparingVoice}</span>
      {/if}
      {#if inCall}
        <span class="holding">{inCall}</span>
      {/if}
      {#if deviceNotice}
        <span class="holding">{deviceNotice}</span>
      {/if}
      {#if modelsNeeded}
        <button type="button" class="input-offer" onclick={openSpeechModels}>Download a speech model</button>
      {/if}
      {#if inputOffer}
        <button type="button" class="input-offer" onclick={acceptInputOffer}>Use {inputOffer.name} for input</button>
      {/if}
    </div>
  {/if}

  {#if error}
    <div class="voice-error">{error}</div>
  {/if}
</div>

<style>
  .holding {
    margin-left: 6px;
    font-size: var(--text-xs);
    color: var(--accent);
  }
  @media (max-width: 880px) {
    .wake-label {
      display: none;
    }

    .wake-btn {
      width: 32px;
      padding: 0;
      justify-content: center;
    }
  }
  .voice-control {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 4px;
  }

  /* ── Push to Talk Button ── */
  .voice-btn {
    position: relative;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid var(--accent-line);
    background: var(--accent-muted);
    color: var(--accent);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition:
      border-color var(--dur-fast) ease,
      background var(--dur-fast) ease;
    overflow: visible;
    flex-shrink: 0;
  }

  .voice-btn:hover {
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 20%, transparent);
  }

  .voice-btn.listening {
    border-color: var(--danger);
    background: var(--danger-bg);
    color: var(--danger);
    animation: breathe 1.5s ease-in-out infinite;
  }

  .voice-btn.disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .mic-icon {
    width: 18px;
    height: 18px;
    z-index: 1;
  }

  /* Pulse ring effect */
  .pulse-ring {
    position: absolute;
    inset: -4px;
    border-radius: 50%;
    border: 2px solid color-mix(in srgb, var(--accent) calc(var(--intensity, 0) * 60%), transparent);
    animation: pulse-expand 1s ease-out infinite;
    pointer-events: none;
  }

  .voice-btn.listening .pulse-ring {
    border-color: color-mix(in srgb, var(--danger) calc(var(--intensity, 0) * 80%), transparent);
  }

  @keyframes pulse-expand {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    100% {
      transform: scale(1.6);
      opacity: 0;
    }
  }

  @keyframes breathe {
    0%,
    100% {
    }
    50% {
    }
  }

  /* ── Wake Word Button ── */
  .wake-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    height: 32px;
    border-radius: var(--radius-pill);
    border: 1px solid var(--border);
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    cursor: pointer;
    font-size: var(--text-sm);
    font-weight: 500;
    transition:
      color var(--dur-fast) ease,
      border-color var(--dur-fast) ease,
      background var(--dur-fast) ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .wake-btn:hover {
    border-color: var(--accent-line);
    color: var(--text-primary);
    background: var(--accent-muted);
  }

  .wake-btn.active {
    border-color: color-mix(in srgb, var(--success) 50%, transparent);
    background: var(--success-bg);
    color: var(--success);
  }

  @keyframes gentle-glow {
    0%,
    100% {
    }
    50% {
    }
  }

  .wave-icon {
    width: 14px;
    height: 14px;
  }

  .wake-label {
    letter-spacing: 0.3px;
  }

  .wake-btn.disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  /* ── Speaking Indicator ── */
  .speak-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    cursor: pointer;
    flex-shrink: 0;
  }

  .speak-bars {
    display: flex;
    align-items: center;
    gap: 2px;
    height: 14px;
  }

  .bar {
    display: block;
    width: 2px;
    background: color-mix(in srgb, var(--accent) 80%, transparent);
    border-radius: 1px;
    animation: speak-bar 0.6s ease-in-out infinite;
  }

  .bar:nth-child(1) {
    height: 4px;
    animation-delay: 0s;
  }
  .bar:nth-child(2) {
    height: 8px;
    animation-delay: 0.1s;
  }
  .bar:nth-child(3) {
    height: 12px;
    animation-delay: 0.2s;
  }
  .bar:nth-child(4) {
    height: 8px;
    animation-delay: 0.3s;
  }
  .bar:nth-child(5) {
    height: 4px;
    animation-delay: 0.4s;
  }

  @keyframes speak-bar {
    0%,
    100% {
      transform: scaleY(1);
    }
    50% {
      transform: scaleY(0.3);
    }
  }

  /* ── Transcript Preview ── */
  .input-offer {
    margin-left: 0.5rem;
    padding: 0.15rem 0.5rem;
    border: 1px solid var(--accent, currentColor);
    border-radius: 6px;
    background: transparent;
    color: inherit;
    font: inherit;
    cursor: pointer;
  }

  .transcript-preview {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-primary);
    border: 1px solid color-mix(in srgb, var(--accent) 20%, transparent);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    max-width: 300px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    z-index: 100;
  }

  .transcript-preview .final {
    color: var(--text-primary);
  }

  .transcript-preview .interim {
    color: var(--accent);
    font-style: italic;
  }

  .voice-error {
    font-size: 10px;
    color: var(--danger);
    max-width: 160px;
    text-align: center;
  }
</style>
