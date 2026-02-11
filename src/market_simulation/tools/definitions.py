"""Tool definitions for market simulation agents."""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING

from langchain_core.tools import tool, Tool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .sandbox import SandboxManager

logger = logging.getLogger(__name__)


# --- Simple Tools ---


class EvaluateTradeInput(BaseModel):
    reservation_price: float = Field(description="Your reservation price (valuation)")
    trade_price: float = Field(description="The proposed trade price")
    agent_type: str = Field(description="Your role: 'buyer' or 'seller'")


# TODO: test calculator tool (less suggestive)
@tool("evaluate_trade", args_schema=EvaluateTradeInput)
def evaluate_trade(
    reservation_price: float, trade_price: float, agent_type: str
) -> str:
    """Evaluate whether a trade is profitable given your reservation price and the proposed price."""
    if agent_type == "buyer":
        surplus = reservation_price - trade_price
        if surplus > 0:
            return (
                f"Profit: ${surplus:.2f} (gain). As a buyer with valuation ${reservation_price:.2f}, "
                f"buying at ${trade_price:.2f} gives you a surplus of ${surplus:.2f}. "
                f"This trade is profitable."
            )
        elif surplus == 0:
            return (
                f"Profit: $0.00 (break-even). As a buyer with valuation ${reservation_price:.2f}, "
                f"buying at ${trade_price:.2f} gives zero surplus. "
                f"Acceptable but no gain."
            )
        else:
            return (
                f"Loss: ${abs(surplus):.2f}. As a buyer with valuation ${reservation_price:.2f}, "
                f"buying at ${trade_price:.2f} exceeds your valuation by ${abs(surplus):.2f}. "
                f"This trade is unprofitable - reject."
            )
    else:  # seller
        surplus = trade_price - reservation_price
        if surplus > 0:
            return (
                f"Profit: ${surplus:.2f} (gain). As a seller with valuation ${reservation_price:.2f}, "
                f"selling at ${trade_price:.2f} gives you a surplus of ${surplus:.2f}. "
                f"This trade is profitable."
            )
        elif surplus == 0:
            return (
                f"Profit: $0.00 (break-even). As a seller with valuation ${reservation_price:.2f}, "
                f"selling at ${trade_price:.2f} gives zero surplus. "
                f"Acceptable but no gain."
            )
        else:
            return (
                f"Loss: ${abs(surplus):.2f}. As a seller with valuation ${reservation_price:.2f}, "
                f"selling at ${trade_price:.2f} is below your valuation by ${abs(surplus):.2f}. "
                f"This trade is unprofitable - reject."
            )


class ComputeMarketStatsInput(BaseModel):
    history_text: str = Field(description="The market history text from the simulation")


@tool("compute_market_stats", args_schema=ComputeMarketStatsInput)
def compute_market_stats(history_text: str) -> str:
    """Analyze market history to compute statistics: average prices, bid-ask spread, trends, and acceptance rate."""
    if not history_text.strip():
        return "No market history available yet. This is the first iteration."

    # Parse history entries
    accepted_prices: list[float] = []
    rejected_prices: list[float] = []
    buy_prices: list[float] = []
    sell_prices: list[float] = []
    no_announcement_count = 0

    for line in history_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Extract price if present
        price_match = re.search(r"\$(\d+\.?\d*)", line)
        price = float(price_match.group(1)) if price_match else None

        if "was accepted" in line and price is not None:
            accepted_prices.append(price)
            if "to buy" in line:
                buy_prices.append(price)
            elif "to sell" in line:
                sell_prices.append(price)
        elif "was made but no one responded" in line and price is not None:
            rejected_prices.append(price)
            if "to buy" in line:
                buy_prices.append(price)
            elif "to sell" in line:
                sell_prices.append(price)
        elif "no announcement was made" in line:
            no_announcement_count += 1

    all_announced_prices = accepted_prices + rejected_prices
    total_announcements = len(all_announced_prices)
    total_entries = total_announcements + no_announcement_count

    parts: list[str] = []

    if accepted_prices:
        avg_transaction = sum(accepted_prices) / len(accepted_prices)
        parts.append(f"Transactions: {len(accepted_prices)}")
        parts.append(f"Average transaction price: ${avg_transaction:.2f}")
        if len(accepted_prices) >= 2:
            import statistics

            std_dev = statistics.stdev(accepted_prices)
            parts.append(f"Price std dev: ${std_dev:.2f}")
            # Trend: compare first half to second half
            mid = len(accepted_prices) // 2
            first_half_avg = sum(accepted_prices[:mid]) / mid if mid > 0 else 0
            second_half_avg = sum(accepted_prices[mid:]) / (len(accepted_prices) - mid)
            if second_half_avg > first_half_avg * 1.02:
                parts.append("Price trend: RISING")
            elif second_half_avg < first_half_avg * 0.98:
                parts.append("Price trend: FALLING")
            else:
                parts.append("Price trend: STABLE")
    else:
        parts.append("No completed transactions yet.")

    # Bid-ask spread (from most recent announcements)
    if buy_prices and sell_prices:
        latest_buy = buy_prices[-1]
        latest_sell = sell_prices[-1]
        spread = latest_sell - latest_buy
        parts.append(f"Latest bid (buy offer): ${latest_buy:.2f}")
        parts.append(f"Latest ask (sell offer): ${latest_sell:.2f}")
        parts.append(f"Bid-ask spread: ${spread:.2f}")

    # Acceptance rate
    if total_announcements > 0:
        acceptance_rate = len(accepted_prices) / total_announcements
        parts.append(
            f"Acceptance rate: {acceptance_rate:.0%} ({len(accepted_prices)}/{total_announcements})"
        )

    if no_announcement_count > 0:
        parts.append(f"Iterations with no announcement: {no_announcement_count}")

    return "\n".join(parts)


class ClassifyTraderInput(BaseModel):
    reservation_price: float = Field(description="Your reservation price (valuation)")
    agent_type: str = Field(description="Your role: 'buyer' or 'seller'")
    estimated_market_price: float = Field(
        description="Estimated market clearing price (e.g. average transaction price)"
    )


@tool("classify_trader", args_schema=ClassifyTraderInput)
def classify_trader(
    reservation_price: float, agent_type: str, estimated_market_price: float
) -> str:
    """Classify whether you are an inframarginal or extramarginal trader and suggest a price range."""
    if agent_type == "buyer":
        if reservation_price > estimated_market_price:
            margin = reservation_price - estimated_market_price
            return (
                f"You are an INFRAMARGINAL buyer. Your valuation (${reservation_price:.2f}) "
                f"is above the estimated market price (${estimated_market_price:.2f}) by ${margin:.2f}. "
                f"You should be able to trade profitably. "
                f"Maximum profitable bid: ${reservation_price:.2f}. "
                f"Suggested bid range: ${estimated_market_price * 0.85:.2f} - ${estimated_market_price * 1.15:.2f}"
            )
        else:
            shortfall = estimated_market_price - reservation_price
            return (
                f"You are an EXTRAMARGINAL buyer. Your valuation (${reservation_price:.2f}) "
                f"is below the estimated market price (${estimated_market_price:.2f}) by ${shortfall:.2f}. "
                f"Trading at current market prices would be unprofitable. "
                f"Consider waiting for prices to drop or abstaining from trading."
            )
    else:  # seller
        if reservation_price < estimated_market_price:
            margin = estimated_market_price - reservation_price
            return (
                f"You are an INFRAMARGINAL seller. Your valuation (${reservation_price:.2f}) "
                f"is below the estimated market price (${estimated_market_price:.2f}) by ${margin:.2f}. "
                f"You should be able to trade profitably. "
                f"Minimum profitable ask: ${reservation_price:.2f}. "
                f"Suggested ask range: ${estimated_market_price * 0.85:.2f} - ${estimated_market_price * 1.15:.2f}"
            )
        else:
            shortfall = reservation_price - estimated_market_price
            return (
                f"You are an EXTRAMARGINAL seller. Your valuation (${reservation_price:.2f}) "
                f"is above the estimated market price (${estimated_market_price:.2f}) by ${shortfall:.2f}. "
                f"Trading at current market prices would be unprofitable. "
                f"Consider waiting for prices to rise or abstaining from trading."
            )


# --- E2B Code Interpreter Tool ---


class CodeInterpreterInput(BaseModel):
    code: str = Field(
        description="Python code to execute for market analysis, price calculations, or statistical reasoning"
    )


def get_e2b_tool(sandbox_manager: SandboxManager) -> Tool:
    """Create an E2B code interpreter tool backed by a managed sandbox.

    Args:
        sandbox_manager: SandboxManager instance for sandbox lifecycle.

    Returns:
        LangChain Tool wrapping E2B code execution.
    """

    def run_code(code: str) -> str:
        sandbox = sandbox_manager.get_sandbox()
        if sandbox is None:
            return "Error: Code execution sandbox is not available."

        try:
            execution = sandbox.run_code(code)
        except Exception as e:
            return f"Error executing code: {e}"

        output_parts: list[str] = []
        if execution.logs.stdout:
            stdout = "".join(execution.logs.stdout)
            if stdout.strip():
                output_parts.append(f"stdout:\n{stdout.strip()}")
        if execution.logs.stderr:
            stderr = "".join(execution.logs.stderr)
            if stderr.strip():
                output_parts.append(f"stderr:\n{stderr.strip()}")
        if execution.error:
            output_parts.append(f"error: {execution.error.message}")
        if execution.results:
            for r in execution.results:
                if hasattr(r, "text") and r.text:
                    output_parts.append(f"result: {r.text}")

        return (
            "\n".join(output_parts)
            if output_parts
            else "Code executed successfully with no output."
        )

    t = Tool(
        name="code_interpreter",
        description=(
            "Execute Python code in a sandboxed environment for market analysis, "
            "Bayesian belief updating, price calculations, expected value computation, "
            "statistical analysis, or counterfactual reasoning. "
            "Has access to numpy, pandas, scipy, and statistics. "
            "Use print() to output results."
        ),
        func=run_code,
    )
    t.args_schema = CodeInterpreterInput
    return t
