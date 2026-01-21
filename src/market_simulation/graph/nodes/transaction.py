"""Transaction recording node."""

import logging
from typing import Callable

from ..state import MarketState, Transaction

logger = logging.getLogger(__name__)


def make_record_transaction_node() -> Callable[[MarketState], dict]:
    """Create node that records a successful transaction.

    Returns:
        Node function that records transaction and removes agents from active list.
    """

    def record_transaction(state: MarketState) -> dict:
        """Record the transaction and deactivate the trading agents."""
        if not state["transaction_made"]:
            return {}

        announcing_id = state["announcing_agent_id"]
        responding_id = state["responding_agent_id"]
        price = state["announced_price"]
        announcement_type = state["announcement_type"]

        # Determine buyer and seller
        if announcement_type == "buy":
            buyer_id = announcing_id
            seller_id = responding_id
        else:
            buyer_id = responding_id
            seller_id = announcing_id

        # Create transaction record
        transaction = Transaction(
            round=state["round"],
            iteration=state["iteration"],
            price=price,
            buyer_id=buyer_id,
            seller_id=seller_id,
            announcement_type=announcement_type,
        )

        # Remove both agents from active list
        new_active = [
            aid for aid in state["active_agent_ids"]
            if aid != announcing_id and aid != responding_id
        ]

        # Update agents' active status
        updated_agents = []
        for agent in state["agents"]:
            if agent["id"] in (announcing_id, responding_id):
                updated_agent = {**agent, "active": False}
                updated_agents.append(updated_agent)
            else:
                updated_agents.append(agent)

        logger.info(
            f"Transaction recorded: buyer {buyer_id} <- seller {seller_id} @ ${price:.2f}"
        )

        return {
            "transactions": [transaction],
            "active_agent_ids": new_active,
            "agents": updated_agents,
        }

    return record_transaction
