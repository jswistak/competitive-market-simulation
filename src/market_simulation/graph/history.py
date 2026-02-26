"""Market history formatting for agent prompts."""

import logging
import statistics
from typing import Literal

logger = logging.getLogger(__name__)


def build_market_history_for_prompt(
    state: dict,
    mode: Literal["full", "summary"] = "full",
    last_n_events: int = 3,
) -> str:
    """Build market history string for agent prompts.

    Args:
        state: Current market state with full history data.
        mode: "full" returns raw text, "summary" returns statistics.
        last_n_events: Number of recent raw events to include in summary mode.

    Returns:
        Formatted history string for prompt injection.
    """
    if mode == "full":
        result = state["market_history_text"]
    else:
        result = _build_summary(state, last_n_events)

    logger.debug("Market history for prompt (mode=%s):\n%s", mode, result)
    return result


def _build_summary(state: dict, last_n_events: int) -> str:
    """Build a statistical summary of market history."""
    transactions = state.get("transactions", [])
    records = state.get("iteration_records", [])
    current_round = state["round"]

    if not records:
        return "No market history available yet. This is the first iteration."

    parts: list[str] = []

    # Transaction statistics
    if transactions:
        prices = [t["price"] for t in transactions]
        avg_price = sum(prices) / len(prices)
        parts.append(f"Completed transactions: {len(transactions)}")
        parts.append(f"Average transaction price: ${avg_price:.2f}")
        parts.append(f"Last transaction price: ${prices[-1]:.2f}")

        if len(prices) >= 4:
            mid = len(prices) // 2
            first_half = sum(prices[:mid]) / mid
            second_half = sum(prices[mid:]) / (len(prices) - mid)
            if second_half > first_half * 1.02:
                parts.append("Price trend: RISING")
            elif second_half < first_half * 0.98:
                parts.append("Price trend: FALLING")
            else:
                parts.append("Price trend: STABLE")

        if len(prices) >= 2:
            std = statistics.stdev(prices)
            parts.append(
                f"Price range: ${min(prices):.2f} - ${max(prices):.2f} (std: ${std:.2f})"
            )
    else:
        parts.append("No completed transactions yet.")

    # Bid-ask spread from recent announcements
    buy_prices = []
    sell_prices = []
    for rec in records:
        if rec.get("announcement_made") and rec.get("price") is not None:
            if rec.get("announcement_type") == "buy":
                buy_prices.append(rec["price"])
            elif rec.get("announcement_type") == "sell":
                sell_prices.append(rec["price"])

    if buy_prices and sell_prices:
        parts.append(f"Latest bid (buy offer): ${buy_prices[-1]:.2f}")
        parts.append(f"Latest ask (sell offer): ${sell_prices[-1]:.2f}")
        spread = sell_prices[-1] - buy_prices[-1]
        if spread < 0:
            parts.append(
                f"Bid-ask spread: ${abs(spread):.2f} (crossed/converged)"
            )
        else:
            parts.append(f"Bid-ask spread: ${spread:.2f}")

    # Acceptance rate – count distinct announcements by (round, iteration)
    # to avoid over-counting when multiple responders are queried per announcement
    distinct_announcements = len(set(
        (r["round"], r["iteration"])
        for r in records
        if r.get("announcement_made")
    ))
    if distinct_announcements > 0:
        n_accepted = len(transactions)
        rate = n_accepted / distinct_announcements
        parts.append(
            f"Acceptance rate: {rate:.0%} ({n_accepted}/{distinct_announcements})"
        )

    # Current round context
    round_txns = [t for t in transactions if t["round"] == current_round]
    parts.append(f"Transactions in current round ({current_round}): {len(round_txns)}")

    # Last N raw events for recency
    if last_n_events > 0:
        raw_lines = state["market_history_text"].strip().split("\n")
        recent = (
            raw_lines[-last_n_events:]
            if len(raw_lines) >= last_n_events
            else raw_lines
        )
        if recent and recent[0]:
            parts.append("")
            parts.append("Recent events:")
            for line in recent:
                if line.strip():
                    parts.append(f"  {line.strip()}")

    return "\n".join(parts)


def build_own_history_for_prompt(
    agent: dict,
    mode: Literal["full", "summary"] = "full",
) -> str:
    """Build agent's own history string for prompts.

    Args:
        agent: Agent state dict with own_history_prompt and own_history_data.
        mode: "full" returns raw text, "summary" returns statistics.

    Returns:
        Formatted own history string.
    """
    if mode == "full":
        result = agent["own_history_prompt"]
        logger.debug("Own history for prompt (mode=full):\n%s", result)
        return result

    history_data = agent.get("own_history_data", [])
    if not history_data:
        result = "No actions taken yet."
        logger.debug("Own history for prompt (mode=summary):\n%s", result)
        return result

    parts: list[str] = []

    announcements = [h for h in history_data if h["action"] == "announce"]
    responses = [h for h in history_data if h["action"] == "respond"]
    accepted_actions = [h for h in history_data if h["outcome"] == "accepted"]

    def _plural(n: int, singular: str) -> str:
        return f"{n} {singular}" if n == 1 else f"{n} {singular}s"

    parts.append(
        f"Total actions: {len(history_data)} "
        f"({_plural(len(announcements), 'announcement')}, {_plural(len(responses), 'response')})"
    )
    parts.append(f"Successful trades: {len(accepted_actions)}")

    if accepted_actions:
        trade_prices = [h["price"] for h in accepted_actions]
        parts.append(
            f"Average trade price: ${sum(trade_prices) / len(trade_prices):.2f}"
        )

    # Last action
    last = history_data[-1]
    parts.append(
        f"Last action: {last['action']} at ${last['price']:.2f} ({last['outcome']})"
    )

    result = "\n".join(parts)
    logger.debug("Own history for prompt (mode=summary):\n%s", result)
    return result
