"""Graph nodes for sealed-bid auctions (FPSB, SPSB, All-Pay)."""

import logging
from typing import Callable

import numpy as np
from langchain_core.runnables import RunnableConfig

from ...state import SealedBidState, BidRecord, AuctionResult
from ....agents import zi as zi_decisions
from ....llm.providers.base import LLMProvider
from ....llm.response_schemas import BidResponse, BidResponseWithReasoning
from ....config.schema import AuctionPromptConfig, ZIConfig
from ..base import render_auction_prompt

logger = logging.getLogger(__name__)


def make_collect_bid_node(
    llm: LLMProvider | None,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    response_schema: type[BidResponse] = BidResponseWithReasoning,
    zi_config: ZIConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Callable[[SealedBidState, RunnableConfig], dict]:
    """Create node that collects a bid from the current bidder.

    Prompts one bidder (or samples a ZI bid), appends to bids list,
    and advances current_bidder_index.
    """

    zi_cfg = zi_config or ZIConfig()
    zi_rng = rng if rng is not None else np.random.default_rng()
    include_reasoning = response_schema is BidResponseWithReasoning

    def collect_bid(state: SealedBidState, config: RunnableConfig) -> dict:
        idx = state["current_bidder_index"]
        bidders = state["bidders"]

        if idx >= len(bidders):
            return {"all_bids_collected": True}

        bidder = bidders[idx]
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
                        "action_prompt": prompts.bid_prompt,
                        "auction_type": state["auction_type"],
                        "value_explanation": prompts.value_explanation,
                        "n_bidders": len(bidders),
                    },
                )

                callbacks = config.get("callbacks", []) if config else []
                if not callbacks and callbacks_factory:
                    callbacks = callbacks_factory()

                response = llm.invoke_structured(prompt, response_schema, callbacks=callbacks)
            else:
                response = zi_decisions.decide_sealed_bid(
                    bidder=bidder,
                    zi_cfg=zi_cfg,
                    rng=zi_rng,
                    include_reasoning=include_reasoning,
                )
            bid_amount = response.bid
            reasoning = getattr(response, 'reasoning', '')
            logger.debug(
                f"Bidder {bidder['id']} structured bid: {bid_amount}, "
                f"reasoning: '{reasoning[:100]}...'"
            )

            # Capture tool usage (ZI path has none)
            tool_log_entries = getattr(llm, "last_tool_log", []) if strategy == "llm" else []
            tool_usage_log = [
                {
                    **entry,
                    "agent_id": bidder["id"],
                    "agent_type": "bidder",
                    "action": "sealed_bid",
                    "round": state["round"],
                    "simulation_id": state["simulation_id"],
                }
                for entry in tool_log_entries
            ]

            # Check constraint: bid should not exceed private value
            constraint_violations = state.get("constraint_violations", 0)
            if bid_amount > bidder["private_value"]:
                logger.warning(
                    f"CONSTRAINT VIOLATION: Bidder {bidder['id']} bid "
                    f"${bid_amount:.2f} exceeds private value "
                    f"${bidder['private_value']:.2f}"
                )
                constraint_violations += 1

            bid_record = BidRecord(
                bidder_id=bidder["id"],
                bid_amount=bid_amount,
                round=state["round"],
                bid_step=0,
                private_value=bidder["private_value"],
            )

            logger.info(
                f"R{state['round']}: Bidder {bidder['id']} "
                f"(value=${bidder['private_value']:.2f}) "
                f"bid ${bid_amount:.2f}"
            )

            new_idx = idx + 1
            return {
                "bids": state["bids"] + [bid_record],
                "current_bidder_index": new_idx,
                "all_bids_collected": new_idx >= len(bidders),
                "tool_usage_log": tool_usage_log,
                "constraint_violations": constraint_violations,
            }

        except Exception as e:
            logger.error(f"Decision failed for bidder {bidder['id']} ({strategy}): {e}")
            bid_record = BidRecord(
                bidder_id=bidder["id"],
                bid_amount=0.0,
                round=state["round"],
                bid_step=0,
                private_value=bidder["private_value"],
            )
            new_idx = idx + 1
            return {
                "bids": state["bids"] + [bid_record],
                "current_bidder_index": new_idx,
                "all_bids_collected": new_idx >= len(bidders),
            }

    return collect_bid


def make_determine_winner_node() -> Callable[[SealedBidState], dict]:
    """Create node that determines the winner based on auction type.

    Payment rules:
      - fpsb: winner pays own bid
      - spsb: winner pays second-highest bid
      - all_pay: winner gets item, ALL bidders pay their bids
    """

    def determine_winner(state: SealedBidState) -> dict:
        bids = state["bids"]
        auction_type = state["auction_type"]
        round_num = state["round"]

        if not bids:
            result = AuctionResult(
                round=round_num,
                auction_type=auction_type,
                winner_id=None,
                winning_bid=None,
                payment=None,
                second_highest_bid=None,
                all_bids=[{"bidder_id": b["bidder_id"], "bid_amount": b["bid_amount"]} for b in bids],
                n_active_bidders=len(state["bidders"]),
                surplus=None,
            )
            return {"auction_results": [result], "all_bid_records": list(bids)}

        # Sort by bid amount descending.  Python's sort is stable, so
        # tied bids are resolved by their original collection order
        # (i.e. the bidder who was queried first wins the tie).
        sorted_bids = sorted(bids, key=lambda b: b["bid_amount"], reverse=True)
        winner = sorted_bids[0]
        second_bid = sorted_bids[1]["bid_amount"] if len(sorted_bids) > 1 else None

        # Determine payment based on auction type
        if auction_type == "fpsb":
            payment = winner["bid_amount"]
        elif auction_type == "spsb":
            # When only one bidder, SPSB degenerates to FPSB — bidder pays
            # own bid since there is no second price.
            payment = second_bid if second_bid is not None else winner["bid_amount"]
        elif auction_type == "all_pay":
            payment = winner["bid_amount"]
        else:
            payment = winner["bid_amount"]

        surplus = winner["private_value"] - payment if payment is not None else None

        # For all-pay auctions, every bidder pays their bid regardless of
        # winning.  total_payments captures the aggregate cost for economic
        # analysis (e.g. revenue equivalence checks).
        if auction_type == "all_pay":
            total_payments = sum(b["bid_amount"] for b in bids)
        else:
            total_payments = payment

        result = AuctionResult(
            round=round_num,
            auction_type=auction_type,
            winner_id=winner["bidder_id"],
            winning_bid=winner["bid_amount"],
            payment=payment,
            second_highest_bid=second_bid,
            all_bids=[{"bidder_id": b["bidder_id"], "bid_amount": b["bid_amount"]} for b in bids],
            n_active_bidders=len(state["bidders"]),
            surplus=surplus,
            total_payments=total_payments,
        )

        logger.info(
            f"R{round_num} ({auction_type}): Winner=bidder {winner['bidder_id']} "
            f"bid=${winner['bid_amount']:.2f} payment=${payment:.2f} "
            f"surplus=${surplus:.2f}"
        )

        return {"auction_results": [result], "all_bid_records": list(bids)}

    return determine_winner


def make_update_sealed_history_node() -> Callable[[SealedBidState], dict]:
    """Create node that updates market and bidder histories after a round."""

    def update_sealed_history(state: SealedBidState) -> dict:
        results = state["auction_results"]
        if not results:
            return {}

        latest = results[-1]
        round_num = latest["round"]
        auction_type = latest["auction_type"]

        # Build market history text
        if latest["winner_id"] is not None:
            history_line = (
                f"Round {round_num} ({auction_type}): "
                f"Bidder {latest['winner_id']} won with bid "
                f"${latest['winning_bid']:.2f}, "
                f"payment=${latest['payment']:.2f}"
            )
            if latest["second_highest_bid"] is not None:
                history_line += f", 2nd-highest=${latest['second_highest_bid']:.2f}"
            history_line += ".\n"
        else:
            history_line = f"Round {round_num} ({auction_type}): No bids submitted.\n"

        new_history = state["market_history_text"] + history_line

        # Update per-bidder histories
        updated_bidders = []
        for bidder in state["bidders"]:
            bidder_copy = {**bidder}
            # Find this bidder's bid
            my_bid = None
            for b in state["bids"]:
                if b["bidder_id"] == bidder["id"]:
                    my_bid = b["bid_amount"]
                    break

            won = latest["winner_id"] == bidder["id"]
            if my_bid is not None:
                outcome = "won" if won else "lost"
                entry_text = (
                    f"Round {round_num}: You bid ${my_bid:.2f} and {outcome}."
                )
                if won and latest["payment"] is not None:
                    entry_text += f" Payment: ${latest['payment']:.2f}."
                elif not won and auction_type == "all_pay":
                    # In all-pay auctions, losers also pay their bid.
                    entry_text += f" You paid your bid of ${my_bid:.2f}."
                entry_text += "\n"
                bidder_copy["own_history_prompt"] = bidder["own_history_prompt"] + entry_text
                bidder_copy["own_history_data"] = bidder["own_history_data"] + [
                    {
                        "round": round_num,
                        "action": "bid",
                        "bid_amount": my_bid,
                        "won": won,
                        "payment": latest["payment"] if won else (my_bid if auction_type == "all_pay" else 0.0),
                    }
                ]
            updated_bidders.append(bidder_copy)

        return {
            "market_history_text": new_history,
            "bidders": updated_bidders,
        }

    return update_sealed_history


def make_next_sealed_round_node() -> Callable[[SealedBidState], dict]:
    """Create node that resets state for the next round."""

    def next_sealed_round(state: SealedBidState) -> dict:
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        if new_round > max_rounds:
            logger.info(
                f"Sealed-bid simulation complete: all {max_rounds} rounds finished"
            )
            # Signal completion by setting round past max
            return {"round": new_round}

        logger.info(f"Advancing to sealed-bid round {new_round}/{max_rounds}")

        return {
            "round": new_round,
            "current_bidder_index": 0,
            "all_bids_collected": False,
            "bids": [],
        }

    return next_sealed_round
