"""Control-flow nodes for the improvement-rule CDA graph."""

import logging
from typing import Callable

from ..state import MarketState, IterationRecord
from ...config.schema import PromptConfig, PromptTemplates

logger = logging.getLogger(__name__)


def _get_templates(prompts: PromptConfig | None) -> PromptTemplates:
    """Return prompt templates, falling back to schema defaults."""
    return prompts.general if prompts is not None else PromptTemplates()


def make_update_history_node(
    prompts: PromptConfig | None = None,
) -> Callable[[MarketState], dict]:
    """Node that records per-tick history.

    One row per tick, regardless of outcome — preserves the existing
    iteration_history_*.csv schema (plus two new columns for the
    standing book and one for the outcome tag).
    """

    templates = _get_templates(prompts)

    def update_history(state: MarketState) -> dict:
        round_num = state["round"]
        tick = state["iteration"]
        price = state.get("announced_price")
        announcement_type = state.get("announcement_type")
        announcement_made = state.get("announcement_made", False)
        transaction_made = state.get("transaction_made", False)
        # Resolve the order outcome. Prefer the explicit tag from
        # apply_order; if absent (e.g. legacy callers / hand-built test
        # states), infer it from the (transaction_made, announcement_made)
        # pair so this node stays drop-in compatible.
        if state.get("last_order_outcome"):
            outcome = state["last_order_outcome"]
        elif transaction_made and announcement_made:
            outcome = "traded"
        elif announcement_made:
            outcome = "posted"
        else:
            outcome = "no_announcement"

        announcer_id = state.get("announcing_agent_id")
        responder_id = state.get("counterparty_agent_id")

        announcing_rp = None
        responding_rp = None
        for agent in state["agents"]:
            if agent["id"] == announcer_id:
                announcing_rp = agent["reservation_price"]
            if responder_id is not None and agent["id"] == responder_id:
                responding_rp = agent["reservation_price"]

        record = IterationRecord(
            round=round_num,
            iteration=tick,
            price=price if announcement_made else None,
            announcement_made=announcement_made,
            transaction_made=transaction_made,
            announcement_type=announcement_type if announcement_made else None,
            announcing_agent_id=announcer_id,
            announcing_agent_reservation_price=announcing_rp,
            counterparty_agent_id=responder_id,
            counterparty_reservation_price=responding_rp,
            announcement_reasoning=state.get("last_announcement_reasoning", ""),
            # Under the CDA the counterparty does not take a fresh action
            # on a cross, so this is always "". The pass-through is kept
            # so the column exists in the iteration history CSV alongside
            # announcement_reasoning, even when always empty.
            counterparty_reasoning=state.get("last_counterparty_reasoning", ""),
            standing_bid=state.get("standing_bid"),
            standing_ask=state.get("standing_ask"),
            order_outcome=outcome,
        )

        # Market-history text — one line per tick, branched on the
        # mechanism's outcome tag. Each branch covers one of the four
        # possible outcomes; non_improving used to fall through to the
        # no_announcement template, which conflated "agent passed" with
        # "agent tried and was dropped" in the LLM's view of the market.
        history_update = ""
        if outcome == "traded":
            history_update = templates.market_history_accepted_template.format(
                round=round_num,
                iteration=tick,
                announcement_type=announcement_type,
                price=price,
            )
        elif outcome == "posted":
            history_update = templates.market_history_rejected_template.format(
                round=round_num,
                iteration=tick,
                announcement_type=announcement_type,
                price=price,
            )
        elif outcome == "non_improving":
            history_update = templates.market_history_non_improving_template.format(
                round=round_num,
                iteration=tick,
                announcement_type=announcement_type,
                price=price,
            )
        else:  # no_announcement
            history_update = templates.market_history_no_announcement_template.format(
                round=round_num,
                iteration=tick,
            )

        new_history = state["market_history_text"] + history_update

        updated_agents = _update_agent_histories(state, templates, outcome=outcome)

        return {
            "iteration_records": [record],
            "market_history_text": new_history,
            "agents": updated_agents,
        }

    return update_history


_OUTCOME_LABELS = {
    "traded": "accepted",
    "posted": "posted",
    "non_improving": "rejected",
}


def _update_agent_histories(
    state: MarketState,
    templates: PromptTemplates | None = None,
    outcome: str | None = None,
) -> list[dict]:
    """Append per-agent history rows for the announcer on this tick.

    In the CDA path there is no explicit "responder" action — the
    counterparty on a cross just has their standing order executed, so
    they don't make a new decision. We therefore only log the announcer's
    action here; the counterparty's original posting was logged on an
    earlier tick.

    The announcer's history captures three distinct outcomes the agent
    can experience for an emitted price (``state.last_order_outcome``):
      * ``traded``        → ``"accepted"`` (a cross executed)
      * ``posted``        → ``"posted"`` (improving order entered the book)
      * ``non_improving`` → ``"rejected"`` (mechanism dropped the order)
    Previously a posted-but-not-yet-traded order was also labelled
    ``"rejected"``, conflating it with non-improving drops. The richer
    label lets an LLM tell the difference between "my order is on the
    book waiting for a counterparty" and "my order was dropped, try
    something different next time."
    """
    if templates is None:
        templates = PromptTemplates()

    agents = state["agents"]
    announcer_id = state.get("announcing_agent_id")
    if outcome is None:
        # Direct callers (legacy tests) may invoke without a precomputed
        # outcome. Apply the same fallback as update_history.
        if state.get("last_order_outcome"):
            outcome = state["last_order_outcome"]
        elif state.get("transaction_made") and state.get("announcement_made"):
            outcome = "traded"
        elif state.get("announcement_made"):
            outcome = "posted"
        else:
            outcome = "no_announcement"
    label = _OUTCOME_LABELS.get(outcome)

    updated = []
    for agent in agents:
        agent_copy = {**agent}

        if (
            announcer_id is not None
            and agent["id"] == announcer_id
            and label is not None
        ):
            history_entry = {
                "round": state["round"],
                "iteration": state["iteration"],
                "action": "announce",
                "price": state["announced_price"],
                "outcome": label,
            }
            agent_copy["own_history_data"] = agent["own_history_data"] + [history_entry]

            ann_type = "buy" if agent["type"] == "buyer" else "sell"
            entry = templates.announcement_history_template.format(
                round=state["round"],
                iteration=state["iteration"],
                announcement_type=ann_type,
                price=state["announced_price"],
                outcome=label,
            )
            agent_copy["own_history_prompt"] = agent["own_history_prompt"] + entry

        updated.append(agent_copy)

    return updated


def make_check_iteration_node() -> Callable[[MarketState], dict]:
    """Retained for non-CDA compatibility; not in the CDA graph anymore.

    The old CDA loop needed this to detect "iteration complete when a
    transaction happens or all responders exhaust." Under the tick-based
    CDA, every tick is one step and the round check handles advancement.
    Keeping the factory so nothing outside the CDA path breaks.
    """

    def check_iteration(state: MarketState) -> dict:
        return {"iteration_complete": True}

    return check_iteration


def make_check_round_node() -> Callable[[MarketState], dict]:
    """Node that decides whether to end the current round.

    Ends when:
      1. Tick budget exhausted (``iteration >= max_iterations``).
      2. Fewer than 2 active agents remain (at most one side left).
      3. No active agent can post an improving order under the current
         book (mechanism-level deadlock — keeps running would just burn
         ticks with guaranteed non-improving draws).

    Condition (3) matters because ZI-C refuses to post non-improving
    orders; once every intra-marginal buyer's reservation is below the
    standing bid (and analogously on the sell side), we'd loop to tick
    cap with no further activity. Early exit keeps simulations fast.
    """

    def check_round(state: MarketState) -> dict:
        iteration = state["iteration"]
        max_iterations = state["max_iterations"]
        if iteration >= max_iterations:
            logger.info(f"Round {state['round']} complete: tick budget reached")
            return {"round_complete": True}

        active_ids = state["active_agent_ids"]
        if len(active_ids) < 2:
            logger.info(f"Round {state['round']} complete: insufficient active agents")
            return {"round_complete": True}

        if _deadlocked(state):
            logger.info(
                f"Round {state['round']} complete: no active agent can "
                f"post an improving order (book frozen)"
            )
            return {"round_complete": True}

        return {"round_complete": False}

    return check_round


def _deadlocked(state: MarketState) -> bool:
    """True when no active agent could post an improving non-loss order.

    Only used for early-exit. A conservative predicate — we return True
    only when both sides are provably frozen for every active agent.
    ZI-U is never deadlocked from the mechanism's perspective (it
    samples in ``[u_low, u_high]`` and may still cross), so any active
    ZI-U or LLM agent is assumed live.
    """
    from .apply_order import PRICE_INCREMENT

    standing_bid = state.get("standing_bid")
    standing_ask = state.get("standing_ask")

    # Book is empty — anyone can post.
    if standing_bid is None and standing_ask is None:
        return False

    for agent in state["agents"]:
        if not agent.get("active", True):
            continue
        if agent["id"] not in state["active_agent_ids"]:
            continue

        strategy = agent.get("strategy", "llm")
        # Conservative: only ZI-C has a deterministic non-loss range we
        # can reason about. LLM and ZI-U agents may still post (their
        # choice function is not a pure price interval).
        if strategy != "zi_c":
            return False

        reservation = agent["reservation_price"]
        if agent["type"] == "buyer":
            # Can this buyer cross the ask or improve the bid?
            crosses = standing_ask is not None and standing_ask <= reservation
            improves = (
                standing_bid is None
                or reservation > standing_bid + PRICE_INCREMENT - 1e-9
            )
            if crosses or improves:
                return False
        else:
            crosses = standing_bid is not None and standing_bid >= reservation
            improves = (
                standing_ask is None
                or reservation < standing_ask - PRICE_INCREMENT + 1e-9
            )
            if crosses or improves:
                return False

    return True


def make_next_iteration_node() -> Callable[[MarketState], dict]:
    """Advance to the next tick within a round.

    The standing book persists across ticks within a round, so
    standing_* fields are intentionally NOT reset here.
    """

    def next_iteration(state: MarketState) -> dict:
        new_iteration = state["iteration"] + 1
        logger.info(f"R{state['round']}: advancing to tick {new_iteration}")
        return {
            "iteration": new_iteration,
            "iteration_complete": False,
            "transaction_made": False,
            "announcement_made": False,
            "announcing_agent_id": None,
            "announced_price": None,
            "announcement_type": None,
            "counterparty_agent_id": None,
            "potential_responder_ids": [],
            "current_responder_index": 0,
            "last_announcement_reasoning": "",
            "last_counterparty_reasoning": "",
            "last_order_outcome": None,
        }

    return next_iteration


def make_next_round_node() -> Callable[[MarketState], dict]:
    """Advance to the next round (or end the simulation)."""

    def next_round(state: MarketState) -> dict:
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        if new_round > max_rounds:
            tx_count = len(state.get("transactions", []))
            logger.info(f"Simulation complete: all {max_rounds} rounds finished ({tx_count} total transactions)")
            return {"simulation_complete": True}

        tx_in_round = sum(
            1 for t in state.get("transactions", []) if t["round"] == state["round"]
        )
        logger.info(
            f"Round {state['round']} completed with {tx_in_round} transaction(s). "
            f"Advancing to round {new_round}"
        )

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
            "counterparty_agent_id": None,
            # Order book clears at round boundary — G&S periods are
            # independent, so standing orders do not carry over.
            "standing_bid": None,
            "standing_ask": None,
            "standing_bid_agent_id": None,
            "standing_ask_agent_id": None,
            "last_order_outcome": None,
            "potential_responder_ids": [],
            "current_responder_index": 0,
            "last_announcement_reasoning": "",
            "last_counterparty_reasoning": "",
        }

    return next_round
