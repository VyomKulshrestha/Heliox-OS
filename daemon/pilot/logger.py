import logging
import sys

# Standard ANSI colors
COLORS = {
    "DEBUG": "\033[94m",    # Blue
    "INFO": "\033[92m",     # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[1;91m", # Bold Red
}
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = f"%(asctime)s - %(name)s - {COLORS.get(record.levelname, '')}%(levelname)s{RESET} - %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger():
    """Configure the root 'pilot' logger so all sub-loggers inherit this formatting."""
    logger = logging.getLogger("pilot")

    if getattr(logger, "_pilot_configured", False):
        return
    logger._pilot_configured = True

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Prevent propagating to the root Python logger to avoid duplicate logs
        logger.propagate = False
