"""Disable tqdm progress bars when output is not a terminal, keeping redirected logs short."""

from __future__ import annotations

import sys


def bar_disabled() -> bool:
    """Check whether tqdm progress bars should be disabled.

    Returns:
        True if stderr is not attached to a terminal.
    """
    return not sys.stderr.isatty()
