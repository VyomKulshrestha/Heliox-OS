Thanks for the PR! Switching from an MD5 hash to a visual MSE delta using NumPy is an excellent architectural improvement to stop the agent from over-triggering on micro-movements. The logic is solid, but there are a couple of small code hygiene issues to address before merging:

1. **Comment Formatting Bug (`screen_vision.py`):** Around line 141, the comment `# Delta-Frame Throttler State` was accidentally pasted twice, corrupting the previous line:
   ```python
   self._enable_llm_describe = False  # Disabled by default (expensive)# Delta-Frame Throttler State
   # Delta-Frame Throttler State
   ```
   Please clean this up.

2. **State Leak in Fallback Screenshot (`screen_vision.py`):** In `_sync_fallback_screenshot_hash()`, you read from the temporary file `tmp_path = self._screenshot_dir / "_latest.png"`. If the subprocess command (e.g., `screencapture`) fails, `tmp_path.exists()` will still evaluate to `True` because the *previous* tick's screenshot is still on disk. The system will silently load the old image, compute an MSE of 0, and assume nothing changed instead of throwing an error. 
   **Fix:** Add `tmp_path.unlink(missing_ok=True)` right before the `try` block or subprocess call to ensure you don't read stale state.

Once these minor details are fixed, this is good to go!
