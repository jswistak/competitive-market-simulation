"""Control flow nodes for managing iterations and rounds."""

import logging
from typing import Callable

from ..state import MarketState, IterationRecord

logger = logging.getLogger(__name__)


def make_update_history_node() -> Callable[[MarketState], dict]:
    """Create node that updates market history after an iteration.

    Returns:
        Node function that updates history text and records.
    """

    def update_history(state: MarketState) -> dict:
        """Update market history based on iteration outcome."""
        round_num = state["round"]
        iteration = state["iteration"]
        price = state["announced_price"]
        announcement_type = state["announcement_type"]
        announcement_made = state["announcement_made"]
        transaction_made = state["transaction_made"]

        # Get reservation prices for record
        announcing_rp = None
        responding_rp = None

        for agent in state["agents"]:
            if agent["id"] == state["announcing_agent_id"]:
                announcing_rp = agent["reservation_price"]
            if agent["id"] == state["responding_agent_id"]:
                responding_rp = agent["reservation_price"]

        # Create iteration record
        record = IterationRecord(
            round=round_num,
            iteration=iteration,
            price=price,
            announcement_made=announcement_made,
            transaction_made=transaction_made,
            announcement_type=announcement_type,
            announcing_agent_id=state["announcing_agent_id"],
            announcing_agent_reservation_price=announcing_rp,
            responding_agent_id=state["responding_agent_id"],
            responding_agent_reservation_price=responding_rp,
            announcement_reasoning=state.get("last_announcement_reasoning", ""),
            response_reasoning=state.get("last_response_reasoning", ""),
        )

        # Check if all responders for the current announcement have been queried
        all_responders_queried = (
            state["current_responder_index"] >= len(state["potential_responder_ids"])
        )

        # Update market history text when announcement outcome is known
        # (not only at iteration end, so subsequent announcers see prior rejections)
        history_update = ""
        if announcement_made and transaction_made:
            history_update = (
                f"In round {round_num} at iteration {iteration}, "
                f"an announcement to {announcement_type} for ${price:.2f} was accepted.\n"
            )
        elif announcement_made and all_responders_queried:
            history_update = (
                f"In round {round_num} at iteration {iteration}, "
                f"an announcement to {announcement_type} for ${price:.2f} was made but no one responded.\n"
            )
        elif not announcement_made and state["iteration_complete"]:
            history_update = (
                f"In round {round_num} at iteration {iteration}, no announcement was made.\n"
            )

        new_history = state["market_history_text"] + history_update

        # Update agent histories if needed
        updated_agents = _update_agent_histories(state)

        return {
            "iteration_records": [record],
            "market_history_text": new_history,
            "agents": updated_agents,
        }

    return update_history


def _update_agent_histories(state: MarketState) -> list[dict]:
    """Update individual agent history records."""
    agents = state["agents"]
    updated = []

    for agent in agents:
        agent_copy = {**agent}

        # Update announcing agent's history (only when announcement outcome is resolved)
        if agent["id"] == state["announcing_agent_id"] and state["announcement_made"]:
            all_responders_queried = (
                state["current_responder_index"] >= len(state["potential_responder_ids"])
            )
            announcement_resolved = state["transaction_made"] or all_responders_queried

            if announcement_resolved:
                outcome = "accepted" if state["transaction_made"] else "rejected"
                history_entry = {
                    "round": state["round"],
                    "iteration": state["iteration"],
                    "action": "announce",
                    "price": state["announced_price"],
                    "outcome": outcome,
                }
                agent_copy["own_history_data"] = agent["own_history_data"] + [history_entry]

                ann_type = "buy" if agent["type"] == "buyer" else "sell"
                agent_copy["own_history_prompt"] = (
                    agent["own_history_prompt"]
                    + f"In round {state['round']} at iteration {state['iteration']}, "
                    f"your offer to {ann_type} for ${state['announced_price']:.2f} was {outcome}.\n"
                )

        # Update responding agent's history (always, so every responder sees their action)
        if agent["id"] == state["responding_agent_id"] and state["announcement_made"]:
            outcome = "accepted" if state["response_accepted"] else "rejected"
            history_entry = {
                "round": state["round"],
                "iteration": state["iteration"],
                "action": "respond",
                "price": state["announced_price"],
                "outcome": outcome,
            }
            agent_copy["own_history_data"] = agent["own_history_data"] + [history_entry]

            opp_type = state["announcement_type"]
            agent_copy["own_history_prompt"] = (
                agent["own_history_prompt"]
                + f"In round {state['round']} at iteration {state['iteration']}, "
                f"you {outcome} a {opp_type} offer for ${state['announced_price']:.2f}.\n"
            )

        updated.append(agent_copy)

    return updated


def make_check_iteration_node() -> Callable[[MarketState], dict]:
    """Create node that checks if iteration should continue.

    Returns:
        Node function that sets iteration_complete flag.
    """

    def check_iteration(state: MarketState) -> dict:
        """Check if the current iteration is complete."""
        # Iteration complete if:
        # 1. Transaction was made
        # 2. All responders have been queried and rejected
        # 3. No announcement was made

        if state["transaction_made"]:
            logger.info(f"R{state['round']}/I{state['iteration']}: Iteration complete (transaction made)")
            return {"iteration_complete": True}

        if not state["announcement_made"]:
            logger.info(f"R{state['round']}/I{state['iteration']}: Iteration complete (no announcement)")
            return {"iteration_complete": True}

        # Check if more responders to query
        responder_idx = state["current_responder_index"]
        total_responders = len(state["potential_responder_ids"])

        if responder_idx >= total_responders:
            logger.info(
                f"R{state['round']}/I{state['iteration']}: Iteration complete "
                f"(all {total_responders} responders queried)"
            )
            return {"iteration_complete": True}

        return {"iteration_complete": False}

    return check_iteration


def make_check_round_node() -> Callable[[MarketState], dict]:
    """Create node that checks if round should continue.

    Returns:
        Node function that sets round_complete flag.
    """

    def check_round(state: MarketState) -> dict:
        """Check if the current round is complete."""
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]

        # Round complete if max iterations reached or no active agents
        if iteration >= max_iterations:
            logger.info(f"Round {state['round']} complete: max iterations reached")
            return {"round_complete": True}

        if len(state["active_agent_ids"]) < 2:
            logger.info(f"Round {state['round']} complete: insufficient active agents")
            return {"round_complete": True}

        return {"round_complete": False}

    return check_round


def make_next_iteration_node() -> Callable[[MarketState], dict]:
    """Create node that advances to the next iteration.

    Returns:
        Node function that increments iteration counter.
    """

    def next_iteration(state: MarketState) -> dict:
        """Advance to the next iteration."""
        new_iteration = state["iteration"] + 1
        logger.info(f"R{state['round']}: Advancing to iteration {new_iteration}")

        return {
            "iteration": new_iteration,
            "iteration_complete": False,
            "transaction_made": False,
            "announcement_made": False,
            "announcing_agent_id": None,
            "announced_price": None,
            "announcement_type": None,
            "responding_agent_id": None,
            "response_accepted": None,
            "potential_responder_ids": [],
            "current_responder_index": 0,
            "announced_this_iteration": [],  # Reset for new iteration
            "last_announcement_reasoning": "",
            "last_response_reasoning": "",
        }

    return next_iteration


def make_next_round_node() -> Callable[[MarketState], dict]:
    """Create node that advances to the next round.

    Returns:
        Node function that increments round counter and resets agents.
    """

    def next_round(state: MarketState) -> dict:
        """Advance to the next round."""
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        if new_round > max_rounds:
            tx_count = len(state.get("transactions", []))
            logger.info(f"Simulation complete: all {max_rounds} rounds finished ({tx_count} total transactions)")
            return {"simulation_complete": True}

        tx_in_round = sum(1 for t in state.get("transactions", []) if t["round"] == state["round"])
        logger.info(f"Round {state['round']} completed with {tx_in_round} transaction(s). Advancing to round {new_round}")

        # Reactivate all agents for new round
        updated_agents = []
        all_agent_ids = []
        for agent in state["agents"]:
            updated_agents.append({**agent, "active": True})
            all_agent_ids.append(agent["id"])

        return {
            "round": new_round,
            "iteration": 1,
            "round_complete": False,
            "iteration_complete": False,
            "transaction_made": False,
            "announcement_made": False,
            "agents": updated_agents,
            "active_agent_ids": all_agent_ids,
            "announcing_agent_id": None,
            "announced_price": None,
            "announcement_type": None,
            "responding_agent_id": None,
            "response_accepted": None,
            "potential_responder_ids": [],
            "current_responder_index": 0,
            "announced_this_iteration": [],  # Reset for new round
            "last_announcement_reasoning": "",
            "last_response_reasoning": "",
        }

    return next_round
