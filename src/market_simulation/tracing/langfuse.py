"""Langfuse tracing integration."""

import os
import logging
from typing import Any

from langfuse.langchain import CallbackHandler

from ..config.schema import TracingConfig

logger = logging.getLogger(__name__)


class TracingManager:
    """Manager for Langfuse tracing integration."""

    def __init__(self, config: TracingConfig):
        """Initialize tracing manager.

        Args:
            config: Tracing configuration.
        """
        self.config = config
        self.enabled = config.enabled
        self._handler: CallbackHandler | None = None

        if self.enabled:
            self._validate_config()

    def _validate_config(self) -> None:
        """Validate that required credentials are available."""
        public_key = self.config.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self.config.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")

        if not public_key or not secret_key:
            logger.warning(
                "Langfuse tracing enabled but credentials not found. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )
            self.enabled = False

    def get_callback_handler(
        self,
        trace_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> CallbackHandler | None:
        """Get a Langfuse callback handler for LangChain integration.

        Args:
            trace_name: Name for the trace.
            session_id: Session identifier for grouping traces.
            user_id: User identifier.
            metadata: Additional metadata to attach.
            tags: Tags for filtering traces.

        Returns:
            CallbackHandler if tracing is enabled, None otherwise.
        """
        if not self.enabled:
            return None

        public_key = self.config.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self.config.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = self.config.langfuse_host

        return CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            trace_name=trace_name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        )

    def create_callbacks_factory(
        self,
        simulation_id: int,
        experiment_name: str | None = None,
    ):
        """Create a factory function for generating callbacks per LLM call.

        Args:
            simulation_id: Current simulation number.
            experiment_name: Name of the experiment.

        Returns:
            Factory function that creates callback list.
        """
        if not self.enabled:
            return lambda: []

        def factory() -> list:
            handler = self.get_callback_handler(
                trace_name=f"market_simulation_{simulation_id}",
                session_id=f"sim_{simulation_id}",
                metadata={
                    "simulation_id": simulation_id,
                    "experiment": experiment_name or "market_simulation",
                },
                tags=["market_simulation", f"sim_{simulation_id}"],
            )
            return [handler] if handler else []

        return factory

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enabled and self._handler:
            try:
                self._handler.flush()
            except Exception as e:
                logger.warning(f"Failed to flush Langfuse traces: {e}")


def create_tracing_manager(config: TracingConfig) -> TracingManager:
    """Create a tracing manager from configuration.

    Args:
        config: Tracing configuration.

    Returns:
        Configured TracingManager instance.
    """
    return TracingManager(config)
