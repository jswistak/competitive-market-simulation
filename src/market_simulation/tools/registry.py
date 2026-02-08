"""Tool registry for market simulation agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from ..config.schema import ToolConfig
from .definitions import (
    evaluate_trade,
    compute_market_stats,
    classify_trader,
    get_e2b_tool,
)

from .sandbox import SandboxManager

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available tools for market agents."""

    def __init__(
        self,
        config: ToolConfig,
        sandbox_manager: SandboxManager | None = None,
    ) -> None:
        self.config = config
        self.sandbox_manager = sandbox_manager
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._build()

    def _build(self) -> None:
        if self.config.enable_simple_tools:
            self._tools.extend([evaluate_trade, compute_market_stats, classify_trader])
            logger.info(
                "Registered simple tools: evaluate_trade, compute_market_stats, classify_trader"
            )

        if self.config.enable_code_interpreter and self.sandbox_manager is not None:
            e2b_tool = get_e2b_tool(self.sandbox_manager)
            self._tools.append(e2b_tool)
            logger.info("Registered E2B code interpreter tool")

        self._tool_map = {t.name: t for t in self._tools}

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools

    @property
    def tool_map(self) -> dict[str, BaseTool]:
        return self._tool_map

    @property
    def has_tools(self) -> bool:
        return len(self._tools) > 0

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]
