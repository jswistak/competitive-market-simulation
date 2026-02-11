"""E2B sandbox lifecycle management."""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages E2B sandbox lifecycle for market simulations."""

    def __init__(self, enabled: bool = True, timeout: int = 300) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self._sandbox: Any = None

    def get_sandbox(self) -> Any:
        """Get or create the E2B sandbox (lazy initialization).

        Returns:
            E2B Sandbox instance, or None if unavailable.
        """
        if not self.enabled:
            return None

        if self._sandbox is None:
            try:
                from e2b_code_interpreter import Sandbox
            except ImportError:
                logger.warning(
                    "e2b_code_interpreter not installed. "
                    "Install with: uv add e2b-code-interpreter"
                )
                self.enabled = False
                return None

            api_key = os.getenv("E2B_API_KEY")
            if not api_key:
                logger.warning("E2B_API_KEY not set, code interpreter disabled")
                self.enabled = False
                return None

            try:
                # Use Sandbox.create() with timeout parameter (in seconds)
                self._sandbox = Sandbox.create(timeout=self.timeout)
                # Pre-install common libraries
                self._sandbox.run_code(
                    "import numpy as np; import statistics; print('sandbox ready')"
                )
                logger.info(f"E2B sandbox created (timeout={self.timeout}s)")
            except Exception as e:
                logger.error(f"Failed to create E2B sandbox: {e}")
                self.enabled = False
                return None

        return self._sandbox

    def close(self) -> None:
        """Kill the current sandbox."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
                logger.info("E2B sandbox closed")
            except Exception as e:
                logger.warning(f"Error closing sandbox: {e}")
            self._sandbox = None

    @contextmanager
    def session(self) -> Generator[SandboxManager, None, None]:
        """Context manager for sandbox lifecycle."""
        try:
            yield self
        finally:
            self.close()
