import logging
from collections import deque

logger = logging.getLogger(__name__)


class ClipboardBufferManager:
    def __init__(self, max_size=5):
        # deque automatically ensures it never grows past 5 items
        self.buffer = deque(maxlen=max_size)

    def push_text(self, text: str):
        """Pushes a unique text snippet onto the rolling high-stress queue."""
        if not text:
            return False

        text_striped = text.strip()
        if not text_striped:
            return False

        # Only add if it's different from the most recent entry to avoid duplicate spam
        if not self.buffer or self.buffer[-1] != text_striped:
            self.buffer.append(text_striped)
            logger.info(f"[ClipboardBufferManager] Cached new entry. Total stored: {len(self.buffer)}")
            return True
        return False

    def get_history(self):
        """Returns the rolling clipboard snapshot history as a list."""
        return list(self.buffer)

    def clear(self):
        """Flushes the buffer cache completely."""
        self.buffer.clear()


# Global instance for easy import across modules
clipboard_buffer = ClipboardBufferManager()
