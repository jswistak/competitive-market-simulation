"""Order-book node for the improvement-rule CDA.

Replaces the former `select_responders` / `respond` loop. After an agent
posts an order via the announce node, this node decides the order's fate:

- **cross** (bid >= standing_ask or ask <= standing_bid): transaction at
  the standing (earlier) price. Both parties deactivate for the round.
- **improving** (buyer: price > standing_bid; seller: price < standing_ask):
  update the book, no transaction.
- **non_improving**: discard. The agent's tick is consumed but no book
  change. This is how ZI-U's out-of-range draws get filtered — the
  paper's "wasted draws" behaviour.
- **no_announcement**: agent chose not to act (ZI-U Bernoulli or LLM
  opted out). Tick still counts.

Pricing convention: when a new bid crosses the standing ask, the trade
executes at the standing_ask (the earlier-posted price); analogously for
asks crossing the standing bid. This matches Gode & Sunder (1993) and
the NYSE order-book convention.
"""

import logging
from typing import Callable

from ..state import MarketState, Transaction

logger = logging.getLogger(__name__)

# Keep in sync with agents.zi.PRICE_INCREMENT — the minimum strict
# improvement distance for the improvement rule.
PRICE_INCREMENT = 0.01


def make_apply_order_node() -> Callable[[MarketState], dict]:
    """Create the node that applies an announced order to the book.

    Returns:
        Node function that updates standing book or records a transaction.
    """

    def apply_order(state: MarketState) -> dict:
        # No announcement made this tick — nothing to apply. The tick
        # still counts (the check_round node will advance).
        if not state["announcement_made"] or state["announced_price"] is None:
            return {
                "transaction_made": False,
                "last_order_outcome": "no_announcement",
            }

        price = state["announced_price"]
        ann_type = state["announcement_type"]
        announcer_id = state["announcing_agent_id"]
        standing_bid = state.get("standing_bid")
        standing_ask = state.get("standing_ask")

        if ann_type == "buy":
            # Crosses the standing ask: execute at standing_ask.
            if standing_ask is not None and price >= standing_ask:
                counterparty = state["standing_ask_agent_id"]
                trade_price = standing_ask
                return _record_trade(
                    state,
                    buyer_id=announcer_id,
                    seller_id=counterparty,
                    price=trade_price,
                    ann_type="buy",
                )
            # Improving bid: must be strictly above standing_bid.
            if standing_bid is None or price > standing_bid:
                logger.info(
                    f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
                    f"posted improving bid @ ${price:.2f} "
                    f"(prev standing_bid={standing_bid})"
                )
                delta = {
                    "transaction_made": False,
                    "standing_bid": price,
                    "standing_bid_agent_id": announcer_id,
                    "last_order_outcome": "posted",
                }
                prior_owner = state.get("standing_bid_agent_id")
                if prior_owner is not None and prior_owner != announcer_id:
                    delta["replaced_standing_owner_id"] = prior_owner
                    delta["replaced_standing_price"] = standing_bid
                    delta["replaced_standing_side"] = "buy"
                return delta
            # Non-improving — discard. Keep announcement_made=True so it
            # consistently means "agent emitted a price"; order_outcome
            # carries the disposition (the order was dropped from the
            # book). Without this, history.py's summary-mode aggregates
            # silently filter the dropped attempts out of the bid/ask
            # statistics and acceptance-rate denominator.
            logger.info(
                f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
                f"bid ${price:.2f} not improving (standing_bid=${standing_bid:.2f}); discarded"
            )
            return {
                "transaction_made": False,
                "last_order_outcome": "non_improving",
            }

        # ann_type == "sell"
        if standing_bid is not None and price <= standing_bid:
            counterparty = state["standing_bid_agent_id"]
            trade_price = standing_bid
            return _record_trade(
                state,
                buyer_id=counterparty,
                seller_id=announcer_id,
                price=trade_price,
                ann_type="sell",
            )
        if standing_ask is None or price < standing_ask:
            logger.info(
                f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
                f"posted improving ask @ ${price:.2f} "
                f"(prev standing_ask={standing_ask})"
            )
            delta = {
                "transaction_made": False,
                "standing_ask": price,
                "standing_ask_agent_id": announcer_id,
                "last_order_outcome": "posted",
            }
            prior_owner = state.get("standing_ask_agent_id")
            if prior_owner is not None and prior_owner != announcer_id:
                delta["replaced_standing_owner_id"] = prior_owner
                delta["replaced_standing_price"] = standing_ask
                delta["replaced_standing_side"] = "sell"
            return delta
        logger.info(
            f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
            f"ask ${price:.2f} not improving (standing_ask=${standing_ask:.2f}); discarded"
        )
        # See the buy-side branch above for why announcement_made stays True.
        return {
            "transaction_made": False,
            "last_order_outcome": "non_improving",
        }

    return apply_order


def _record_trade(
    state: MarketState,
    *,
    buyer_id: int,
    seller_id: int,
    price: float,
    ann_type: str,
) -> dict:
    """Build the state delta for an executed trade.

    Selectively clears the book: orders belonging to deactivated parties
    (buyer_id, seller_id) are removed; standing orders from third-party
    agents that happen to still be live are preserved. This is the NYSE
    / G&S convention — after a cross, the non-trading side's standing
    order remains if its owner is still active, so intra-marginal
    traders can keep transacting without rebuilding the book from
    scratch every time.
    """
    transaction = Transaction(
        round=state["round"],
        iteration=state["iteration"],
        price=price,
        buyer_id=buyer_id,
        seller_id=seller_id,
        announcement_type=ann_type,
    )

    new_active = [
        aid for aid in state["active_agent_ids"]
        if aid != buyer_id and aid != seller_id
    ]

    updated_agents = []
    for agent in state["agents"]:
        if agent["id"] in (buyer_id, seller_id):
            updated_agents.append({**agent, "active": False})
        else:
            updated_agents.append(agent)

    # Preserve the side's standing order only if the resting agent is
    # someone OTHER than the two deactivating parties. On a buy-side
    # cross, the matched counterparty is the standing_ask owner, so
    # standing_ask always clears; standing_bid may persist. On a
    # sell-side cross, standing_bid always clears; standing_ask may
    # persist. The announcer (who crossed) did not have a resting
    # order on their own side, so we only have to check the opposite.
    standing_bid = state.get("standing_bid")
    standing_bid_owner = state.get("standing_bid_agent_id")
    standing_ask = state.get("standing_ask")
    standing_ask_owner = state.get("standing_ask_agent_id")

    # Drop orders whose owner just deactivated.
    if standing_bid_owner in (buyer_id, seller_id):
        standing_bid = None
        standing_bid_owner = None
    if standing_ask_owner in (buyer_id, seller_id):
        standing_ask = None
        standing_ask_owner = None

    logger.info(
        f"R{state['round']}/T{state['iteration']}: trade executed @ ${price:.2f} "
        f"(buyer={buyer_id}, seller={seller_id}, type={ann_type})"
    )

    return {
        "transactions": [transaction],
        "active_agent_ids": new_active,
        "agents": updated_agents,
        "transaction_made": True,
        "counterparty_agent_id": seller_id if ann_type == "buy" else buyer_id,
        "standing_bid": standing_bid,
        "standing_ask": standing_ask,
        "standing_bid_agent_id": standing_bid_owner,
        "standing_ask_agent_id": standing_ask_owner,
        "last_order_outcome": "traded",
    }
