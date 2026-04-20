"""Graph nodes for the Dutch (descending) auction.

Each round: the mechanism starts at a high price and lowers it by
dutch_decrement each tick.  At each price level every bidder is asked
(in sequence) whether they accept.  The first acceptance wins;
if the price hits the floor with no acceptance, there is no winner.
"""

import logging
import random
from typing import Callable

import numpy as np
from langchain_core.runnables import RunnableConfig

from ...state import DutchAuctionState, BidRecord, AuctionResult
from ....agents import zi as zi_decisions
from ....llm.providers.base import LLMProvider
from ....llm.response_schemas import AcceptRejectResponse, AcceptRejectResponseWithReasoning
from ....config.schema import AuctionPromptConfig, ZIConfig
from ..base import render_auction_prompt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Announce price (sets initial price at start of round)
# ------------------------------------------------------------------


def make_announce_price_node(
    random_seed: int | None = None,
) -> Callable[[DutchAuctionState], dict]:
    """Create node that sets the starting price for a Dutch round.

    Args:
        random_seed: Optional seed for deterministic bidder shuffling.
            When set, a seeded ``random.Random`` instance is used
            instead of the module-level PRNG so runs are reproducible.
    """
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    def announce_price(state: DutchAuctionState) -> dict:
        price = state["dutch_start_price"]
        logger.info(
            f"R{state['round']}: Dutch auction starting at ${price:.2f}"
        )
        # Shuffle bidder order to avoid first-mover advantage.
        # True Dutch auctions are simultaneous; since we query LLMs
        # sequentially, randomising the order each price tick is the
        # pragmatic equivalent.
        shuffled_bidders = list(state["bidders"])
        rng.shuffle(shuffled_bidders)
        return {
            "current_price": price,
            "current_bidder_index": 0,
            "accepted": False,
            "accepting_bidder_id": None,
            "all_queried_at_price": False,
            "bids": [],
            "bidders": shuffled_bidders,
        }

    return announce_price


# ------------------------------------------------------------------
# Solicit acceptance
# ------------------------------------------------------------------


def make_solicit_acceptance_node(
    llm: LLMProvider | None,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    response_schema: type[AcceptRejectResponse] = AcceptRejectResponseWithReasoning,
    zi_config: ZIConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Callable[[DutchAuctionState, RunnableConfig], dict]:
    """Create node that asks the current bidder to accept or reject the price."""

    zi_cfg = zi_config or ZIConfig()
    zi_rng = rng if rng is not None else np.random.default_rng()
    include_reasoning = response_schema is AcceptRejectResponseWithReasoning

    def solicit_acceptance(state: DutchAuctionState, config: RunnableConfig) -> dict:
        idx = state["current_bidder_index"]
        bidders = state["bidders"]

        if idx >= len(bidders):
            return {"all_queried_at_price": True}

        bidder = bidders[idx]
        current_price = state["current_price"]
        strategy = bidder.get("strategy", "llm")

        try:
            if strategy == "llm":
                if llm is None:
                    raise RuntimeError(
                        "Bidder has strategy='llm' but no LLM provider was supplied"
                    )
                prompt = render_auction_prompt(
                    template=prompts.system_template,
                    bidder=bidder,
                    state=state,
                    extra_vars={
                        "action_prompt": prompts.dutch_accept_prompt,
                        "auction_type": state["auction_type"],
                        "value_explanation": prompts.value_explanation,
                        "current_price": current_price,
                        "n_bidders": len(bidders),
                    },
                )

                callbacks = config.get("callbacks", []) if config else []
                if not callbacks and callbacks_factory:
                    callbacks = callbacks_factory()

                response = llm.invoke_structured(prompt, response_schema, callbacks=callbacks)
            else:
                response = zi_decisions.decide_dutch_accept(
                    bidder=bidder,
                    current_price=current_price,
                    zi_cfg=zi_cfg,
                    rng=zi_rng,
                    include_reasoning=include_reasoning,
                )
            reasoning = getattr(response, 'reasoning', '')
            logger.debug(
                f"Bidder {bidder['id']} structured response: accept={response.accept}, "
                f"reasoning='{reasoning[:100]}...'"
            )

            # Capture tool log (ZI path has none)
            tool_log_entries = getattr(llm, "last_tool_log", []) if strategy == "llm" else []
            tool_usage_log = [
                {
                    **entry,
                    "agent_id": bidder["id"],
                    "agent_type": "bidder",
                    "action": "dutch_accept",
                    "round": state["round"],
                    "simulation_id": state["simulation_id"],
                }
                for entry in tool_log_entries
            ]

            accepted = response.accept

            if accepted:
                # Check constraint: accepting above private value
                constraint_violations = state.get("constraint_violations", 0)
                if current_price > bidder["private_value"]:
                    logger.warning(
                        f"CONSTRAINT VIOLATION: Bidder {bidder['id']} "
                        f"accepted ${current_price:.2f} above private "
                        f"value ${bidder['private_value']:.2f}"
                    )
                    constraint_violations += 1

                bid_record = BidRecord(
                    bidder_id=bidder["id"],
                    bid_amount=current_price,
                    round=state["round"],
                    bid_step=idx,
                    private_value=bidder["private_value"],
                )

                logger.info(
                    f"R{state['round']}: Bidder {bidder['id']} ACCEPTS "
                    f"at ${current_price:.2f} "
                    f"(value=${bidder['private_value']:.2f})"
                )

                return {
                    "accepted": True,
                    "accepting_bidder_id": bidder["id"],
                    "bids": state["bids"] + [bid_record],
                    "current_bidder_index": idx + 1,
                    "tool_usage_log": tool_usage_log,
                    "constraint_violations": constraint_violations,
                }

            # Rejected
            logger.info(
                f"R{state['round']}: Bidder {bidder['id']} rejects "
                f"${current_price:.2f}"
            )
            new_idx = idx + 1
            return {
                "current_bidder_index": new_idx,
                "all_queried_at_price": new_idx >= len(bidders),
                "tool_usage_log": tool_usage_log,
            }

        except Exception as e:
            if strategy == "llm":
                logger.error(f"LLM call failed for bidder {bidder['id']}: {e}")
            else:
                logger.error(f"ZI decision failed for bidder {bidder['id']} ({strategy}): {e}")
            new_idx = idx + 1
            return {
                "current_bidder_index": new_idx,
                "all_queried_at_price": new_idx >= len(bidders),
            }

    return solicit_acceptance


# ------------------------------------------------------------------
# Check Dutch end
# ------------------------------------------------------------------


def make_check_dutch_end_node() -> Callable[[DutchAuctionState], dict]:
    """Create node that checks if the Dutch round should end or lower price.

    Three outcomes:
      - accepted: someone accepted -> settle
      - lower: no one accepted at this level, price > floor -> lower_price
      - floor: price at/below floor with no acceptance -> settle (no winner)
    """

    def check_dutch_end(state: DutchAuctionState) -> dict:
        # This node just sets flags; routing is done in edges
        return {}

    return check_dutch_end


# ------------------------------------------------------------------
# Lower price
# ------------------------------------------------------------------


def make_lower_price_node(
    random_seed: int | None = None,
) -> Callable[[DutchAuctionState], dict]:
    """Create node that decrements the current price.

    Args:
        random_seed: Optional seed for deterministic bidder shuffling.
            When set, a seeded ``random.Random`` instance is used
            instead of the module-level PRNG so runs are reproducible.
    """
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    def lower_price(state: DutchAuctionState) -> dict:
        new_price = round(state["current_price"] - state["dutch_decrement"], 2)
        logger.info(
            f"R{state['round']}: Lowering price to ${new_price:.2f}"
        )
        # Shuffle bidder order for each price tick to avoid positional bias
        shuffled_bidders = list(state["bidders"])
        rng.shuffle(shuffled_bidders)
        return {
            "current_price": new_price,
            "current_bidder_index": 0,
            "all_queried_at_price": False,
            "bidders": shuffled_bidders,
        }

    return lower_price


# ------------------------------------------------------------------
# Settlement
# ------------------------------------------------------------------


def make_settle_dutch_node() -> Callable[[DutchAuctionState], dict]:
    """Create node that settles a Dutch auction round."""

    def settle_dutch(state: DutchAuctionState) -> dict:
        round_num = state["round"]
        auction_type = state["auction_type"]

        if state["accepted"] and state["accepting_bidder_id"] is not None:
            winner_id = state["accepting_bidder_id"]
            payment = state["current_price"]

            # Find winner's value for surplus
            winner_value: float | None = None
            for b in state["bidders"]:
                if b["id"] == winner_id:
                    winner_value = b["private_value"]
                    break

            surplus = (winner_value - payment) if winner_value is not None else None

            logger.info(
                f"R{round_num} (dutch): Winner=bidder {winner_id} "
                f"at ${payment:.2f}, surplus=${surplus}"
            )
        else:
            winner_id = None
            payment = None
            surplus = None
            logger.info(f"R{round_num} (dutch): No winner — price hit floor")

        result = AuctionResult(
            round=round_num,
            auction_type=auction_type,
            winner_id=winner_id,
            winning_bid=payment,
            payment=payment,
            second_highest_bid=None,
            all_bids=[
                {"bidder_id": b["bidder_id"], "bid_amount": b["bid_amount"]}
                for b in state["bids"]
            ],
            n_active_bidders=len(state["bidders"]),
            surplus=surplus,
        )

        return {"auction_results": [result], "all_bid_records": list(state["bids"])}

    return settle_dutch


# ------------------------------------------------------------------
# Update history
# ------------------------------------------------------------------


def make_update_dutch_history_node() -> Callable[[DutchAuctionState], dict]:
    """Create node that updates histories after a Dutch round."""

    def update_dutch_history(state: DutchAuctionState) -> dict:
        results = state["auction_results"]
        if not results:
            return {}

        latest = results[-1]
        round_num = latest["round"]

        # Market history
        if latest["winner_id"] is not None:
            history_line = (
                f"Round {round_num} (dutch): "
                f"Bidder {latest['winner_id']} accepted at "
                f"${latest['payment']:.2f}.\n"
            )
        else:
            history_line = (
                f"Round {round_num} (dutch): "
                f"No one accepted. Price reached floor.\n"
            )

        new_history = state["market_history_text"] + history_line

        # Per-bidder histories
        updated_bidders = []
        for bidder in state["bidders"]:
            bidder_copy = {**bidder}
            won = latest["winner_id"] == bidder["id"]

            if won:
                entry_text = (
                    f"Round {round_num}: You accepted at "
                    f"${latest['payment']:.2f}.\n"
                )
                bidder_copy["own_history_data"] = bidder["own_history_data"] + [
                    {
                        "round": round_num,
                        "action": "dutch_accept",
                        "price": latest["payment"],
                        "won": True,
                        "payment": latest["payment"],
                    }
                ]
            else:
                entry_text = (
                    f"Round {round_num}: You did not accept. "
                )
                if latest["winner_id"] is not None:
                    entry_text += (
                        f"Bidder {latest['winner_id']} won at "
                        f"${latest['payment']:.2f}.\n"
                    )
                else:
                    entry_text += "No one accepted.\n"
                bidder_copy["own_history_data"] = bidder["own_history_data"] + [
                    {
                        "round": round_num,
                        "action": "dutch_reject",
                        "price": None,
                        "won": False,
                        "payment": 0.0,
                    }
                ]

            bidder_copy["own_history_prompt"] = (
                bidder["own_history_prompt"] + entry_text
            )
            updated_bidders.append(bidder_copy)

        return {
            "market_history_text": new_history,
            "bidders": updated_bidders,
        }

    return update_dutch_history


# ------------------------------------------------------------------
# Next round
# ------------------------------------------------------------------


def make_next_dutch_round_node() -> Callable[[DutchAuctionState], dict]:
    """Create node that resets state for the next Dutch round."""

    def next_dutch_round(state: DutchAuctionState) -> dict:
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        if new_round > max_rounds:
            logger.info(
                f"Dutch auction complete: all {max_rounds} rounds finished"
            )
            return {"round": new_round}

        logger.info(f"Advancing to Dutch round {new_round}/{max_rounds}")

        return {
            "round": new_round,
            "current_price": state["dutch_start_price"],
            "current_bidder_index": 0,
            "accepted": False,
            "accepting_bidder_id": None,
            "all_queried_at_price": False,
            "bids": [],
        }

    return next_dutch_round
