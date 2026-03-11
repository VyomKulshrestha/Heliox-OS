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
  import AudioVisualizer from "./AudioVisualizer.svelte";

  // ── State ──
  let isListening = $state(false);
  let isSpeaking = $state(false);
  let transcript = $state("");
  let interimTranscript = $state("");
  let wakeWordActive = $state(false);
  let pulseIntensity = $state(0);
  let error = $state("");
  let voiceEnabled = $state(true);

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
      if (event.error === "no-speech") return; // Ignore no-speech
      if (event.error === "aborted") return;
      console.error("Speech recognition error:", event.error);
      error = `Mic error: ${event.error}`;
      isListening = false;
    };

    rec.onend = () => {
      // Restart if in wake word mode
      if (wakeWordActive && voiceEnabled) {
        try { rec.start(); } catch { /* ignore */ }
      } else {
        isListening = false;
      }
      pulseIntensity = 0;
    };

    return rec;
  }

  function toggleListening() {
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
        } catch { /* ignore */ }
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
  }

  function toggleWakeWord() {
    if (wakeWordActive) {
      stopListening();
    } else {
      startListening(true);
    }
  }

  async function submitVoiceCommand(text: string) {
    // Stop listening temporarily in push-to-talk mode
    if (!wakeWordActive) {
      stopListening();
    }
    
    // Send the command via the session store
    await session.sendCommand(text);
    
    // Speak the response if voice is enabled
    if (voiceEnabled) {
      // Wait for response, then speak it
      const unsub = session.subscribe((s) => {
        if (!s.loading && s.messages.length > 0) {
          const lastMsg = s.messages[s.messages.length - 1];
          if (lastMsg.type === "result" || lastMsg.type === "system") {
            speakText(lastMsg.text || "Done.");
            unsub();
          } else if (lastMsg.type === "error") {
            speakText("Error: " + (lastMsg.text || "Something went wrong."));
            unsub();
          }
        }
      });
    }

    transcript = "";
    interimTranscript = "";
  }

  // ── Text-to-Speech ──
  function speakText(text: string) {
    if (!window.speechSynthesis) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 0.9;
    utterance.volume = 0.9;
    
    // Try to find a good voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => 
      v.name.includes("Microsoft Mark") || 
      v.name.includes("Google UK English Male") ||
      v.name.includes("Daniel") ||
      v.name.includes("Alex")
    );
    if (preferred) utterance.voice = preferred;

    utterance.onstart = () => { isSpeaking = true; };
    utterance.onend = () => { isSpeaking = false; };
    utterance.onerror = () => { isSpeaking = false; };

    window.speechSynthesis.speak(utterance);
  }

  function stopSpeaking() {
    window.speechSynthesis?.cancel();
    isSpeaking = false;
  }

  // Check support on mount
  $effect(() => {
    if (!SpeechRecognition) {
      error = "Speech recognition not supported. Use Chrome or Edge.";
      voiceEnabled = false;
    }
    // Preload voices
    window.speechSynthesis?.getVoices();
  });
</script>

<div class="voice-control">
  <!-- Push-to-Talk Button -->
  <button
    class="voice-btn"
    class:listening={isListening && !wakeWordActive}
    class:disabled={!voiceEnabled}
    onclick={toggleListening}
    title={isListening ? "Stop listening" : "Push to talk"}
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
    class:disabled={!voiceEnabled}
    onclick={toggleWakeWord}
    title={wakeWordActive ? 'Disable "Hey Heliox"' : 'Enable "Hey Heliox" wake word'}
  >
    <svg class="wave-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
      <path d="M8 12a4 4 0 0 0 8 0" />
      <circle cx="12" cy="10" r="1.5" fill="currentColor" />
    </svg>
    <span class="wake-label">{wakeWordActive ? "Heliox Active" : "Hey Heliox"}</span>
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
  {#if interimTranscript || transcript}
    <div class="transcript-preview">
      {#if transcript}
        <span class="final">{transcript}</span>
      {/if}
      {#if interimTranscript}
        <span class="interim">{interimTranscript}</span>
      {/if}
    </div>
  {/if}

  {#if error}
    <div class="voice-error">{error}</div>
  {/if}
</div>

<style>
  .voice-control {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 4px;
  }

  /* ── Push to Talk Button ── */
  .voice-btn {
    position: relative;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid rgba(0, 200, 255, 0.3);
    background: rgba(0, 200, 255, 0.06);
    color: rgba(0, 200, 255, 0.7);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    overflow: visible;
    flex-shrink: 0;
  }

  .voice-btn:hover {
    border-color: rgba(0, 200, 255, 0.6);
    background: rgba(0, 200, 255, 0.12);
    color: rgba(0, 200, 255, 1);
    box-shadow: 0 0 15px rgba(0, 200, 255, 0.2);
  }

  .voice-btn.listening {
    border-color: rgba(255, 60, 60, 0.8);
    background: rgba(255, 60, 60, 0.15);
    color: rgba(255, 60, 60, 0.9);
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
    border: 2px solid rgba(0, 200, 255, calc(var(--intensity, 0) * 0.6));
    animation: pulse-expand 1s ease-out infinite;
    pointer-events: none;
  }

  .voice-btn.listening .pulse-ring {
    border-color: rgba(255, 60, 60, calc(var(--intensity, 0) * 0.8));
  }

  @keyframes pulse-expand {
    0% { transform: scale(1); opacity: 1; }
    100% { transform: scale(1.6); opacity: 0; }
  }

  @keyframes breathe {
    0%, 100% { box-shadow: 0 0 8px rgba(255, 60, 60, 0.3); }
    50% { box-shadow: 0 0 20px rgba(255, 60, 60, 0.6); }
  }

  /* ── Wake Word Button ── */
  .wake-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 16px;
    border: 1px solid rgba(0, 200, 255, 0.2);
    background: rgba(0, 200, 255, 0.04);
    color: rgba(200, 200, 220, 0.6);
    cursor: pointer;
    font-size: 11px;
    font-family: "Inter", sans-serif;
    transition: all 0.3s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .wake-btn:hover {
    border-color: rgba(0, 200, 255, 0.5);
    color: rgba(0, 200, 255, 0.9);
    background: rgba(0, 200, 255, 0.08);
  }

  .wake-btn.active {
    border-color: rgba(0, 255, 136, 0.5);
    background: rgba(0, 255, 136, 0.1);
    color: rgba(0, 255, 136, 0.9);
    animation: gentle-glow 3s ease-in-out infinite;
  }

  @keyframes gentle-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(0, 255, 136, 0.1); }
    50% { box-shadow: 0 0 16px rgba(0, 255, 136, 0.25); }
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
    border: 1px solid rgba(0, 200, 255, 0.3);
    background: rgba(0, 200, 255, 0.08);
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
    background: rgba(0, 200, 255, 0.8);
    border-radius: 1px;
    animation: speak-bar 0.6s ease-in-out infinite;
  }

  .bar:nth-child(1) { height: 4px; animation-delay: 0s; }
  .bar:nth-child(2) { height: 8px; animation-delay: 0.1s; }
  .bar:nth-child(3) { height: 12px; animation-delay: 0.2s; }
  .bar:nth-child(4) { height: 8px; animation-delay: 0.3s; }
  .bar:nth-child(5) { height: 4px; animation-delay: 0.4s; }

  @keyframes speak-bar {
    0%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(0.3); }
  }

  /* ── Transcript Preview ── */
  .transcript-preview {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: rgba(10, 12, 20, 0.95);
    border: 1px solid rgba(0, 200, 255, 0.2);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    max-width: 300px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    z-index: 100;
  }

  .transcript-preview .final {
    color: rgba(255, 255, 255, 0.9);
  }

  .transcript-preview .interim {
    color: rgba(0, 200, 255, 0.6);
    font-style: italic;
  }

  .voice-error {
    font-size: 10px;
    color: rgba(255, 80, 80, 0.8);
    max-width: 160px;
    text-align: center;
  }
</style>
