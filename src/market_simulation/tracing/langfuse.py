"""Langfuse tracing integration."""

import os
import uuid
import logging
from contextlib import contextmanager
from typing import Any, Generator

from langfuse import Langfuse
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
        self._langfuse: Langfuse | None = None
        self._current_experiment_id: str | None = None
        self._current_experiment_name: str | None = None
        self._current_trace: Any | None = None  # Current active trace

        if self.enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Langfuse client with credentials."""
        public_key = self.config.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = self.config.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        host = self.config.langfuse_host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            logger.warning(
                "Langfuse tracing enabled but credentials not found. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
            )
            self.enabled = False
            return

        try:
            # Initialize Langfuse client (singleton pattern in v3)
            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            logger.info("Langfuse client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse client: {e}")
            self.enabled = False

    def set_experiment_context(
        self,
        experiment_id: str,
        experiment_name: str,
    ) -> None:
        """Set the current experiment context for tracing.

        Args:
            experiment_id: Unique identifier for the experiment run (e.g., folder name).
            experiment_name: Human-readable experiment name (e.g., config name).
        """
        self._current_experiment_id = experiment_id
        self._current_experiment_name = experiment_name

    def get_callback_handler(self) -> CallbackHandler | None:
        """Get a Langfuse callback handler for LangChain/LangGraph integration.

        Returns:
            CallbackHandler if tracing is enabled, None otherwise.
        """
        if not self.enabled:
            return None

        try:
            return CallbackHandler()
        except Exception as e:
            logger.warning(f"Failed to create Langfuse callback handler: {e}")
            return None

    @contextmanager
    def trace_simulation(
        self,
        simulation_id: int,
        recursion_limit: int = 100,
        initial_state: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Context manager for tracing a simulation run.

        Creates a new Langfuse trace for each simulation with proper naming and attributes.
        Yields the config dict to pass to graph.invoke().

        Args:
            simulation_id: Current simulation number.
            recursion_limit: Maximum recursion depth for graph execution.
            initial_state: Initial simulation state (for input tracking).

        Yields:
            Config dict ready to pass to graph.invoke().
        """
        config: dict[str, Any] = {
            "recursion_limit": recursion_limit,
        }

        if not self.enabled or not self._langfuse:
            yield config
            return

        experiment_id = self._current_experiment_id or "unknown"
        experiment_name = self._current_experiment_name or "experiment"
        trace_name = f"{experiment_name}_sim{simulation_id}"

        tags = [
            "market_simulation",
            experiment_name,
            f"sim_{simulation_id}",
        ]

        # Build input summary from initial state
        input_summary = None
        if initial_state:
            input_summary = {
                "simulation_id": simulation_id,
                "round": initial_state.get("round", 1),
                "max_rounds": initial_state.get("max_rounds"),
                "max_iterations": initial_state.get("max_iterations"),
                "num_agents": len(initial_state.get("agents", [])),
                "buyers": [
                    {"id": a["id"], "reservation_price": a["reservation_price"]}
                    for a in initial_state.get("agents", [])
                    if a.get("type") == "buyer"
                ],
                "sellers": [
                    {"id": a["id"], "reservation_price": a["reservation_price"]}
                    for a in initial_state.get("agents", [])
                    if a.get("type") == "seller"
                ],
            }

        # Generate a unique trace ID for this simulation
        # This ensures each simulation gets its own separate trace
        trace_id = uuid.uuid4().hex

        # Use start_as_current_span with unique trace_context for each simulation
        with self._langfuse.start_as_current_span(
            name=trace_name,
            input=input_summary,
            trace_context={"trace_id": trace_id},
        ) as span:
            # Store span reference for output updates
            self._current_trace = span

            # Set trace-level attributes (name, user_id, session_id, tags)
            span.update_trace(
                name=trace_name,
                user_id=f"sim_{simulation_id}",
                session_id=experiment_id,
                tags=tags,
                metadata={
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "simulation_id": simulation_id,
                },
            )

            # Create handler - it automatically picks up current trace context
            handler = CallbackHandler()
            config["callbacks"] = [handler]

            try:
                yield config
            finally:
                self._current_trace = None

        # Flush to ensure trace is sent before next simulation
        self._langfuse.flush()

    def update_trace_output(self, **output_data: Any) -> None:
        """Update the current trace's output.

        Args:
            **output_data: Key-value pairs to include in the trace output.
        """
        if self._current_trace:
            self._current_trace.update_trace(output=output_data)

    def get_graph_config(
        self,
        simulation_id: int,
        recursion_limit: int = 100,
    ) -> dict[str, Any]:
        """Get the full config dict to pass to graph.invoke().

        Note: For better trace naming, prefer using trace_simulation() context manager.
        This method is kept for simpler use cases.

        Args:
            simulation_id: Current simulation number.
            recursion_limit: Maximum recursion depth for graph execution.

        Returns:
            Config dict ready to pass to graph.invoke().
        """
        config: dict[str, Any] = {
            "recursion_limit": recursion_limit,
        }

        if not self.enabled:
            return config

        handler = self.get_callback_handler()
        if handler:
            config["callbacks"] = [handler]

        # Build metadata for Langfuse trace attributes
        experiment_id = self._current_experiment_id or "unknown"
        experiment_name = self._current_experiment_name or "experiment"

        tags = [
            "market_simulation",
            experiment_name,
            f"sim_{simulation_id}",
        ]

        config["metadata"] = {
            # Langfuse v3 trace attributes via metadata
            "langfuse_session_id": experiment_id,
            "langfuse_user_id": f"sim_{simulation_id}",
            "langfuse_tags": tags,
            # Custom metadata for linking to results
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "simulation_id": simulation_id,
        }

        return config

    def create_callbacks_factory(
        self,
        simulation_id: int,
        experiment_name: str | None = None,
    ):
        """Create a factory function for generating callbacks per LLM call.

        Note: For LangGraph, prefer using get_graph_config() to pass callbacks
        at the graph level. This method is kept for backwards compatibility
        with direct LLM calls inside nodes.

        Args:
            simulation_id: Current simulation number.
            experiment_name: Name of the experiment.

        Returns:
            Factory function that creates callback list.
        """
        if not self.enabled:
            return lambda: []

        def factory() -> list:
            handler = self.get_callback_handler()
            return [handler] if handler else []

        return factory

    def flush(self) -> None:
        """Flush any pending traces to Langfuse."""
        if self.enabled and self._langfuse:
            try:
                self._langfuse.flush()
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
