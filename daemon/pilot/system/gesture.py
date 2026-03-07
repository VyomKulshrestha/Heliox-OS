"""Gesture Recognition — Hand gesture detection via webcam.

Uses MediaPipe Hands if available, falls back to a stub.
Detected gestures are mapped to Cortex-OS actions.

Gesture Map:
  ✋ open_palm   → cancel / stop
  👍 thumbs_up  → confirm action
  ✌️ peace      → toggle voice
  👊 fist       → execute
  👆 point_up   → scroll up
  🤟 rock       → system info
"""

from __future__ import annotations

import asyncio
import logging
import platform
from typing import Callable

logger = logging.getLogger("pilot.system.gesture")

# ── Gesture Detection via MediaPipe ──────────────────────────────────

_mp_available = False
_mp_hands = None
_cap = None
_running = False

try:
    import cv2
    import mediapipe as mp
    _mp_available = True
except ImportError:
    logger.info("Gesture recognition unavailable: install opencv-python and mediapipe")


GESTURE_MAP = {
    "open_palm": "cancel",
    "thumbs_up": "confirm",
    "peace": "toggle_voice",
    "fist": "execute",
    "point_up": "scroll_up",
    "rock": "system_info",
}


def classify_gesture(landmarks) -> tuple[str, float]:
    """Classify hand landmarks into a gesture name + confidence."""
    if not landmarks:
        return ("", 0.0)

    lm = landmarks.landmark

    def is_extended(tip_idx: int, pip_idx: int) -> bool:
        return lm[tip_idx].y < lm[pip_idx].y

    thumb_extended = lm[4].x < lm[3].x  # Right hand approximation
    index_up = is_extended(8, 6)
    middle_up = is_extended(12, 10)
    ring_up = is_extended(16, 14)
    pinky_up = is_extended(20, 18)

    # Fist — nothing extended
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_extended:
        return ("fist", 0.85)

    # Open palm — all extended
    if index_up and middle_up and ring_up and pinky_up and thumb_extended:
        return ("open_palm", 0.9)

    # Thumbs up — only thumb
    if thumb_extended and not index_up and not middle_up and not ring_up and not pinky_up:
        if lm[4].y < lm[0].y:
            return ("thumbs_up", 0.8)

    # Peace — index + middle
    if index_up and middle_up and not ring_up and not pinky_up:
        return ("peace", 0.85)

    # Point up — only index
    if index_up and not middle_up and not ring_up and not pinky_up:
        return ("point_up", 0.8)

    # Rock — index + pinky
    if index_up and not middle_up and not ring_up and pinky_up:
        return ("rock", 0.75)

    return ("", 0.0)


async def start_gesture_listener(
    on_gesture: Callable[[str, float], None] | None = None,
    cooldown_ms: int = 1500,
) -> None:
    """Start webcam gesture detection loop.

    Args:
        on_gesture: Callback(gesture_name, confidence) when a gesture is detected.
        cooldown_ms: Minimum time between gesture triggers.
    """
    global _running, _cap, _mp_hands

    if not _mp_available:
        logger.warning("MediaPipe not installed — gesture recognition disabled")
        return

    if _running:
        logger.info("Gesture listener already running")
        return

    import cv2 as cv
    import mediapipe as _mp

    _cap = cv.VideoCapture(0)
    if not _cap.isOpened():
        logger.error("Cannot open webcam for gesture detection")
        return

    hands = _mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    _mp_hands = hands
    _running = True

    last_gesture = ""
    last_time = 0.0

    logger.info("Gesture listener started")

    try:
        while _running:
            ret, frame = _cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            # Convert BGR to RGB for MediaPipe
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                gesture_name, confidence = classify_gesture(results.multi_hand_landmarks[0])

                if gesture_name and gesture_name != last_gesture:
                    import time
                    now = time.time() * 1000
                    if now - last_time > cooldown_ms:
                        last_time = now
                        last_gesture = gesture_name
                        logger.info("Gesture detected: %s (%.0f%%)", gesture_name, confidence * 100)
                        if on_gesture:
                            on_gesture(gesture_name, confidence)
                elif not gesture_name:
                    last_gesture = ""

            await asyncio.sleep(0.033)  # ~30 FPS
    finally:
        _running = False
        if _cap:
            _cap.release()
            _cap = None
        if _mp_hands:
            _mp_hands.close()
            _mp_hands = None
        logger.info("Gesture listener stopped")


def stop_gesture_listener() -> None:
    """Stop the gesture detection loop."""
    global _running
    _running = False
