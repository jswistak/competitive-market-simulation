"""Graph nodes for the English (ascending) auction.

Each round: bidders take turns in cycles.  On their turn a bidder may
place a bid ≥ standing_bid + min_increment, or pass (drop out).
The auction ends when a full cycle passes with zero new bids, only one
active bidder remains, or the safety limit (max_bidding_rounds) is hit.
"""

import logging
from typing import Callable

from langchain_core.runnables import RunnableConfig

from ...state import EnglishAuctionState, BidRecord, AuctionResult
from ....llm.providers.base import LLMProvider
from ....llm.response_schemas import EnglishBidResponse
from ....config.schema import AuctionPromptConfig
from ..base import render_auction_prompt

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Solicit bid
# ------------------------------------------------------------------


def make_solicit_bid_node(
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> Callable[[EnglishAuctionState, RunnableConfig], dict]:
    """Create node that asks the current active bidder for a bid or pass."""

    def solicit_bid(state: EnglishAuctionState, config: RunnableConfig) -> dict:
        active_ids = state["active_bidder_ids"]
        idx = state["current_bidder_index"]

        if idx >= len(active_ids):
            # Cycle complete — handled by edge routing
            return {}

        bidder_id = active_ids[idx]

        # Find bidder
        bidder = None
        for b in state["bidders"]:
            if b["id"] == bidder_id:
                bidder = b
                break
        if bidder is None:
            logger.error(f"Bidder {bidder_id} not found")
            return {"current_bidder_index": idx + 1}

        standing = state["standing_bid"]
        min_bid = standing + state["min_increment"]

        prompt = render_auction_prompt(
            template=prompts.system_template,
            bidder=bidder,
            state=state,
            extra_vars={
                "action_prompt": prompts.english_bid_prompt,
                "auction_type": state["auction_type"],
                "value_explanation": prompts.value_explanation,
                "standing_bid": standing,
                "min_bid": min_bid,
                "min_increment": state["min_increment"],
                "n_bidders": len(state["bidders"]),
                "n_active": len(active_ids),
            },
        )

        callbacks = config.get("callbacks", []) if config else []
        if not callbacks and callbacks_factory:
            callbacks = callbacks_factory()

        try:
            response = llm.invoke_structured(prompt, EnglishBidResponse, callbacks=callbacks)
            logger.debug(
                f"Bidder {bidder_id} structured response: action={response.action}, "
                f"bid={response.bid}, reasoning='{response.reasoning[:100]}...'"
            )

            # Capture tool log
            tool_log_entries = getattr(llm, "last_tool_log", [])
            tool_usage_log = [
                {
                    **entry,
                    "agent_id": bidder_id,
                    "agent_type": "bidder",
                    "action": "english_bid",
                    "round": state["round"],
                    "simulation_id": state["simulation_id"],
                }
                for entry in tool_log_entries
            ]

            is_pass = response.action == "pass"
            bid_amount = response.bid

            # If bidder chose to bid but amount is below minimum, treat as pass
            if not is_pass and bid_amount is not None and bid_amount < min_bid:
                logger.info(
                    f"Bidder {bidder_id} bid ${bid_amount:.2f} below min "
                    f"${min_bid:.2f} — treating as pass"
                )
                bid_amount = None
                is_pass = True

            if is_pass or bid_amount is None:
                # Bidder drops out
                new_active = [aid for aid in active_ids if aid != bidder_id]
                logger.info(
                    f"R{state['round']} step {state['bid_step']}: "
                    f"Bidder {bidder_id} passes (drops out). "
                    f"{len(new_active)} active remain."
                )

                return {
                    "active_bidder_ids": new_active,
                    "current_bidder_index": idx + 1,
                    "bid_step": state["bid_step"] + 1,
                    "tool_usage_log": tool_usage_log,
                }

            # Valid bid
            constraint_violations = state.get("constraint_violations", 0)
            if bid_amount > bidder["private_value"]:
                logger.warning(
                    f"CONSTRAINT VIOLATION: Bidder {bidder_id} bid "
                    f"${bid_amount:.2f} > private value "
                    f"${bidder['private_value']:.2f}"
                )
                constraint_violations += 1

            bid_record = BidRecord(
                bidder_id=bidder_id,
                bid_amount=bid_amount,
                round=state["round"],
                bid_step=state["bid_step"],
                private_value=bidder["private_value"],
            )

            logger.info(
                f"R{state['round']} step {state['bid_step']}: "
                f"Bidder {bidder_id} bids ${bid_amount:.2f} "
                f"(value=${bidder['private_value']:.2f})"
            )

            return {
                "standing_bid": bid_amount,
                "standing_bidder_id": bidder_id,
                "bids": state["bids"] + [bid_record],
                "bids_this_cycle": state["bids_this_cycle"] + 1,
                "current_bidder_index": idx + 1,
                "bid_step": state["bid_step"] + 1,
                "tool_usage_log": tool_usage_log,
                "constraint_violations": constraint_violations,
            }

        except Exception as e:
            logger.error(f"LLM call failed for bidder {bidder_id}: {e}")
            # Drop bidder on LLM failure
            new_active = [aid for aid in active_ids if aid != bidder_id]
            return {
                "active_bidder_ids": new_active,
                "current_bidder_index": idx + 1,
                "bid_step": state["bid_step"] + 1,
            }

    return solicit_bid


# ------------------------------------------------------------------
# Check auction end
# ------------------------------------------------------------------


def make_check_auction_end_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that checks whether the current round's auction is over.

    Ends when:
      1. Only 1 (or 0) active bidders remain, OR
      2. A full cycle passed with zero new bids, OR
      3. Safety limit (bid_step >= max_bidding_rounds * n_bidders) reached.
    """

    def check_auction_end(state: EnglishAuctionState) -> dict:
        active = state["active_bidder_ids"]
        bids_this_cycle = state["bids_this_cycle"]
        n_bidders = len(state["bidders"])
        limit = state["max_bidding_rounds"] * max(n_bidders, 1)

        if len(active) <= 1:
            logger.info(
                f"R{state['round']}: Auction ended — "
                f"{len(active)} active bidder(s) remain"
            )
            return {"auction_ended": True}

        if bids_this_cycle == 0:
            logger.info(
                f"R{state['round']}: Auction ended — "
                f"full cycle with no new bids"
            )
            return {"auction_ended": True}

        if state["bid_step"] >= limit:
            logger.info(
                f"R{state['round']}: Auction ended — "
                f"safety limit ({limit}) reached"
            )
            return {"auction_ended": True}

        return {"auction_ended": False}

    return check_auction_end


# ------------------------------------------------------------------
# Reset cycle
# ------------------------------------------------------------------


def make_reset_cycle_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that resets for a new bidding cycle."""

    def reset_cycle(state: EnglishAuctionState) -> dict:
        logger.debug(
            f"R{state['round']}: Resetting cycle. "
            f"{len(state['active_bidder_ids'])} active bidders."
        )
        return {
            "current_bidder_index": 0,
            "bids_this_cycle": 0,
        }

    return reset_cycle


# ------------------------------------------------------------------
# Settlement — English (≈ second-price)
# ------------------------------------------------------------------


def make_settle_english_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that settles an English auction round.

    The winner is the last standing bidder (or the holder of the
    standing bid). Payment = their last bid, which naturally
    approximates the second-highest value.
    """

    def settle_english(state: EnglishAuctionState) -> dict:
        return _settle(state)

    return settle_english


# ------------------------------------------------------------------
# Settlement — First-Price Open-Outcry
# ------------------------------------------------------------------


def make_settle_open_outcry_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that settles a first-price open-outcry round.

    Same as English except payment is explicitly the winner's own
    last bid (first-price rule).
    """

    def settle_open_outcry(state: EnglishAuctionState) -> dict:
        return _settle(state)

    return settle_open_outcry


def _settle(state: EnglishAuctionState) -> dict:
    """Shared settlement logic for English and Open-Outcry auctions."""
    round_num = state["round"]
    auction_type = state["auction_type"]
    standing_bid = state["standing_bid"]
    winner_id = state["standing_bidder_id"]

    # Find second-highest bid
    second_bid: float | None = None
    if len(state["bids"]) >= 2:
        sorted_bids = sorted(
            state["bids"], key=lambda b: b["bid_amount"], reverse=True
        )
        # Second-highest is the highest bid NOT from the winner
        for b in sorted_bids:
            if b["bidder_id"] != winner_id:
                second_bid = b["bid_amount"]
                break

    # Payment rule
    payment = standing_bid  # Both English and open-outcry pay the standing bid
    # (In English, the standing bid naturally ≈ second price because
    # the winner only had to outbid the second-highest by min_increment.
    # Open-outcry is explicitly first-price by design.)

    # Winner's surplus
    winner_value: float | None = None
    for b in state["bidders"]:
        if b["id"] == winner_id:
            winner_value = b["private_value"]
            break

    surplus = (winner_value - payment) if winner_value is not None and payment is not None else None

    result = AuctionResult(
        round=round_num,
        auction_type=auction_type,
        winner_id=winner_id,
        winning_bid=standing_bid,
        payment=payment,
        second_highest_bid=second_bid,
        all_bids=[
            {"bidder_id": b["bidder_id"], "bid_amount": b["bid_amount"]}
            for b in state["bids"]
        ],
        n_active_bidders=len(state["active_bidder_ids"]),
        surplus=surplus,
    )

    if winner_id is not None:
        logger.info(
            f"R{round_num} ({auction_type}): Winner=bidder {winner_id} "
            f"bid=${standing_bid:.2f} payment=${payment:.2f} "
            f"surplus=${surplus}"
        )
    else:
        logger.info(f"R{round_num} ({auction_type}): No winner (no bids)")

    return {"auction_results": [result], "all_bid_records": list(state["bids"])}


# ------------------------------------------------------------------
# Update history
# ------------------------------------------------------------------


def make_update_english_history_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that updates histories after an English/Open-Outcry round."""

    def update_english_history(state: EnglishAuctionState) -> dict:
        results = state["auction_results"]
        if not results:
            return {}

        latest = results[-1]
        round_num = latest["round"]
        auction_type = latest["auction_type"]

        # Market history
        if latest["winner_id"] is not None:
            history_line = (
                f"Round {round_num} ({auction_type}): "
                f"Bidder {latest['winner_id']} won at "
                f"${latest['winning_bid']:.2f}, "
                f"payment=${latest['payment']:.2f}.\n"
            )
        else:
            history_line = f"Round {round_num} ({auction_type}): No bids placed.\n"

        new_history = state["market_history_text"] + history_line

        # Per-bidder histories
        updated_bidders = []
        for bidder in state["bidders"]:
            bidder_copy = {**bidder}
            won = latest["winner_id"] == bidder["id"]

            # Find this bidder's highest bid in the round
            my_bids = [
                b["bid_amount"] for b in state["bids"]
                if b["bidder_id"] == bidder["id"]
            ]
            highest_bid = max(my_bids) if my_bids else None

            if highest_bid is not None:
                outcome = "won" if won else "lost"
                entry_text = (
                    f"Round {round_num}: Your highest bid was "
                    f"${highest_bid:.2f} and you {outcome}."
                )
                if won and latest["payment"] is not None:
                    entry_text += f" Payment: ${latest['payment']:.2f}."
                entry_text += "\n"
                bidder_copy["own_history_prompt"] = (
                    bidder["own_history_prompt"] + entry_text
                )
                bidder_copy["own_history_data"] = bidder["own_history_data"] + [
                    {
                        "round": round_num,
                        "action": "english_bid",
                        "highest_bid": highest_bid,
                        "n_bids": len(my_bids),
                        "won": won,
                        "payment": latest["payment"] if won else 0.0,
                    }
                ]
            else:
                # Bidder dropped out immediately or was never active
                bidder_copy["own_history_prompt"] = (
                    bidder["own_history_prompt"]
                    + f"Round {round_num}: You did not bid.\n"
                )
                bidder_copy["own_history_data"] = bidder["own_history_data"] + [
                    {
                        "round": round_num,
                        "action": "english_pass",
                        "highest_bid": None,
                        "n_bids": 0,
                        "won": False,
                        "payment": 0.0,
                    }
                ]

            updated_bidders.append(bidder_copy)

        return {
            "market_history_text": new_history,
            "bidders": updated_bidders,
        }

    return update_english_history


# ------------------------------------------------------------------
# Next round
# ------------------------------------------------------------------


def make_next_english_round_node() -> Callable[[EnglishAuctionState], dict]:
    """Create node that resets state for the next English/Open-Outcry round."""

    def next_english_round(state: EnglishAuctionState) -> dict:
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        if new_round > max_rounds:
            logger.info(
                f"English auction complete: all {max_rounds} rounds finished"
            )
            return {"round": new_round}

        logger.info(f"Advancing to English round {new_round}/{max_rounds}")

        # Reactivate all bidders
        all_ids = [b["id"] for b in state["bidders"]]

        return {
            "round": new_round,
            "active_bidder_ids": all_ids,
            "current_bidder_index": 0,
            "standing_bid": 0.0,
            "standing_bidder_id": None,
            "bids_this_cycle": 0,
            "bid_step": 0,
            "bids": [],
            "auction_ended": False,
        }

    return next_english_round
