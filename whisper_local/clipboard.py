"""Clipboard access with graceful fallback for headless environments."""

import logging

logger = logging.getLogger(__name__)


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard.

    Returns:
        True if copied successfully, False if clipboard is unavailable.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.warning("Clipboard unavailable: %s", e)
        return False
