"""Shared utilities for auction graph nodes."""

import re
import logging
from typing import Any

from ..state import BidderState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bid / price extraction (reuses logic from graph/nodes/announce.py)
# ---------------------------------------------------------------------------


def extract_bid(response: str) -> float | None:
    """Extract a numeric bid from an LLM response.

    Extraction priority:
      1. Plain float parse (after stripping $ and ,)
      2. Last $-prefixed number (e.g. "$3.27")
      3. Last bare decimal number (e.g. "1.50") — requires decimal point
         to avoid extracting round numbers like "round 1"
      4. Last bare integer (e.g. "5") at a word boundary — catches
         responses like "I bid 5" that lack a decimal point
    """
    if not response or not response.strip():
        return None

    # Stage 1: plain parse
    clean = response.strip().replace("$", "").replace(",", "")
    if re.fullmatch(r"\d+\.?\d*", clean):
        return float(clean)

    # Stage 2: $-prefixed numbers
    dollar_matches = re.findall(r"\$([\d]+\.?\d*)", response)
    if dollar_matches:
        try:
            extracted = float(dollar_matches[-1])
            logger.debug(
                f"Bid extracted via $-prefix fallback: '{response}' -> {extracted}"
            )
            return extracted
        except ValueError:
            pass

    # Stage 3: bare decimal numbers only
    bare_matches = re.findall(r"(?<![\w\-])(\d+\.\d+)(?!\w)", response)
    if bare_matches:
        try:
            extracted = float(bare_matches[-1])
            logger.debug(
                f"Bid extracted via bare-decimal fallback: '{response}' -> {extracted}"
            )
            return extracted
        except ValueError:
            pass

    # Stage 4: bare integers (e.g. "I bid 5")
    int_matches = re.findall(r"\b(\d+)\b", response)
    if int_matches:
        try:
            extracted = float(int_matches[-1])
            logger.debug(
                f"Bid extracted via bare-integer fallback: '{response}' -> {extracted}"
            )
            return extracted
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Yes/no extraction (reuses logic from graph/nodes/respond.py)
# ---------------------------------------------------------------------------


def extract_yes_no(response: str) -> bool:
    """Extract a yes/no answer from an LLM response.

    Uses word boundary matching to avoid false positives
    (e.g. "yesterday" should not match as "yes").
    """
    if not response:
        return False
    text = response.strip().lower()
    if not text:
        return False
    if text in ("yes", "yes."):
        return True
    return bool(re.search(r"\byes\b", text))


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_auction_prompt(
    template: str,
    bidder: BidderState,
    state: dict,
    extra_vars: dict[str, Any] | None = None,
) -> str:
    """Render an auction prompt template with bidder and state variables.

    The template can use any of these variables:
      {bidder_id}, {private_value}, {round}, {max_rounds},
      {market_history}, {own_history}, {action_prompt},
      plus anything in extra_vars.
    """
    template_vars = {
        "bidder_id": bidder["id"],
        "private_value": bidder["private_value"],
        "round": state["round"],
        "max_rounds": state["max_rounds"],
        "market_history": state.get("market_history_text", ""),
        "own_history": bidder["own_history_prompt"],
    }
    if extra_vars:
        template_vars.update(extra_vars)

    # Pre-format the action_prompt with all available vars so that
    # nested placeholders like {current_price} in dutch_accept_prompt
    # get substituted before being inserted into the outer template.
    if "action_prompt" in template_vars and isinstance(template_vars["action_prompt"], str):
        try:
            template_vars["action_prompt"] = template_vars["action_prompt"].format(
                **template_vars
            )
        except (KeyError, IndexError):
            pass  # Leave as-is if it has unresolvable placeholders

    return template.format(**template_vars)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_bidder_by_id(
    bidders: list[BidderState], bidder_id: int
) -> BidderState | None:
    """Look up a bidder by ID."""
    for b in bidders:
        if b["id"] == bidder_id:
            return b
    return None
