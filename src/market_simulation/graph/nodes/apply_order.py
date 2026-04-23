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
                return {
                    "transaction_made": False,
                    "standing_bid": price,
                    "standing_bid_agent_id": announcer_id,
                    "last_order_outcome": "posted",
                }
            # Non-improving — discard.
            logger.info(
                f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
                f"bid ${price:.2f} not improving (standing_bid=${standing_bid:.2f}); discarded"
            )
            return {
                "transaction_made": False,
                "announcement_made": False,
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
            return {
                "transaction_made": False,
                "standing_ask": price,
                "standing_ask_agent_id": announcer_id,
                "last_order_outcome": "posted",
            }
        logger.info(
            f"R{state['round']}/T{state['iteration']}: agent {announcer_id} "
            f"ask ${price:.2f} not improving (standing_ask=${standing_ask:.2f}); discarded"
        )
        return {
            "transaction_made": False,
            "announcement_made": False,
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

    Multi-unit aware. Each trader's ``current_unit_index`` advances by
    one. If that was the trader's last unit, they deactivate; otherwise
    they stay live and their ``reservation_price`` is updated to the
    next unit's value. Their own standing orders (which referenced the
    just-retired unit) are cleared from the book — the next unit will
    be re-quoted from scratch, which is the G&S convention.

    Standing orders belonging to third-party agents persist unchanged
    (NYSE / G&S convention): an intra-marginal buyer's outstanding bid
    doesn't vanish just because another pair transacted.
    """
    # Look up trader state to read schedules and unit indices.
    agents_by_id = {a["id"]: a for a in state["agents"]}
    buyer = agents_by_id[buyer_id]
    seller = agents_by_id[seller_id]

    buyer_unit_index = buyer.get("current_unit_index", 0)
    seller_unit_index = seller.get("current_unit_index", 0)
    buyer_values = buyer.get("values", [buyer["reservation_price"]])
    seller_values = seller.get("values", [seller["reservation_price"]])

    transaction = Transaction(
        round=state["round"],
        iteration=state["iteration"],
        price=price,
        buyer_id=buyer_id,
        seller_id=seller_id,
        announcement_type=ann_type,
        buyer_unit_index=buyer_unit_index,
        seller_unit_index=seller_unit_index,
        buyer_value=buyer_values[buyer_unit_index],
        seller_cost=seller_values[seller_unit_index],
    )

    # Determine post-trade status per trader: deactivate only when all
    # units are retired; otherwise advance to the next unit.
    def _advance_unit(agent: dict) -> tuple[dict, bool]:
        """Return (updated_agent, still_active)."""
        values = agent.get("values", [agent["reservation_price"]])
        new_index = agent.get("current_unit_index", 0) + 1
        if new_index >= len(values):
            return ({**agent, "active": False, "current_unit_index": new_index}, False)
        return (
            {
                **agent,
                "current_unit_index": new_index,
                "reservation_price": values[new_index],
            },
            True,
        )

    updated_agents: list[dict] = []
    buyer_still_active = True
    seller_still_active = True
    for agent in state["agents"]:
        if agent["id"] == buyer_id:
            new_agent, buyer_still_active = _advance_unit(agent)
            updated_agents.append(new_agent)
        elif agent["id"] == seller_id:
            new_agent, seller_still_active = _advance_unit(agent)
            updated_agents.append(new_agent)
        else:
            updated_agents.append(agent)

    # Active-agent list: drop the trader only if they exhausted their
    # schedule. Under single-unit (values=[one]), this always drops them
    # — matching pre-multi-unit behaviour.
    new_active = list(state["active_agent_ids"])
    if not buyer_still_active and buyer_id in new_active:
        new_active.remove(buyer_id)
    if not seller_still_active and seller_id in new_active:
        new_active.remove(seller_id)

    # Clear the book's orders that belonged to the traders, regardless
    # of whether they stay in the round. Their prior quote referenced
    # the retired unit; the next unit must be re-quoted.
    standing_bid = state.get("standing_bid")
    standing_bid_owner = state.get("standing_bid_agent_id")
    standing_ask = state.get("standing_ask")
    standing_ask_owner = state.get("standing_ask_agent_id")
    if standing_bid_owner in (buyer_id, seller_id):
        standing_bid = None
        standing_bid_owner = None
    if standing_ask_owner in (buyer_id, seller_id):
        standing_ask = None
        standing_ask_owner = None

    logger.info(
        f"R{state['round']}/T{state['iteration']}: trade executed @ ${price:.2f} "
        f"(buyer={buyer_id} unit {buyer_unit_index}, "
        f"seller={seller_id} unit {seller_unit_index}, type={ann_type})"
    )

    return {
        "transactions": [transaction],
        "active_agent_ids": new_active,
        "agents": updated_agents,
        "transaction_made": True,
        "responding_agent_id": seller_id if ann_type == "buy" else buyer_id,
        "response_accepted": True,
        "standing_bid": standing_bid,
        "standing_ask": standing_ask,
        "standing_bid_agent_id": standing_bid_owner,
        "standing_ask_agent_id": standing_ask_owner,
        "last_order_outcome": "traded",
    }
