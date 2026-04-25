"""LSEG Workspace session management."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def open_lseg_session(session_name: str = "desktop.workspace") -> None:
    """Open a managed LSEG Workspace desktop session."""
    import lseg.data as ld

    try:
        ld.open_session(name=session_name)
        logger.info("LSEG session opened: %s", session_name)
    except Exception as exc:
        logger.error(
            "Failed to open LSEG session '%s'. "
            "Make sure LSEG Workspace is running before calling this script. "
            "Error: %s",
            session_name,
            exc,
        )
        raise


def close_lseg_session() -> None:
    """Close the active LSEG session gracefully (ignores errors if already closed)."""
    try:
        import lseg.data as ld

        ld.close_session()
        logger.info("LSEG session closed.")
    except Exception as exc:
        logger.warning("Error closing LSEG session (may already be closed): %s", exc)
