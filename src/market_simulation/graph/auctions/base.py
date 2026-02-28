"""Shared utilities for auction graph nodes."""

import logging
from typing import Any

from ..state import BidderState

logger = logging.getLogger(__name__)


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
        "persona": bidder.get("persona", ""),
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

    # Use sentinel replacement for persona to avoid str.format() issues with curly braces
    persona_text = template_vars.pop("persona")
    result = template.replace("{persona}", "<<PERSONA>>")
    result = result.format(**template_vars)
    return result.replace("<<PERSONA>>", persona_text)


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
