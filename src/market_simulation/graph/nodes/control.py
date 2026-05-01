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
            history_update = templates.market_history_posted_template.format(
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
        elif outcome == "no_announcement":
            history_update = templates.market_history_no_announcement_template.format(
                round=round_num,
                iteration=tick,
            )
        else:
            raise ValueError(f"unknown order_outcome {outcome!r}")

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
    they don't make a new decision. The *announcer* gets a row labelled
    by ``state.last_order_outcome``:
      * ``traded``        → ``"accepted"`` (a cross executed)
      * ``posted``        → ``"posted"`` (improving order entered the book)
      * ``non_improving`` → ``"rejected"`` (mechanism dropped the order)

    On top of that, the resting-order owner whose order was just
    affected gets a back-annotation row:
      * ``traded``  → counterparty (whose standing order was crossed)
                      gains a ``"filled"`` entry.
      * ``posted``  → the displaced prior owner (if a different agent's
                      standing order was just replaced on the same side)
                      gains an ``"outbid"`` entry. Without this, the
                      agent permanently sees only their original
                      "posted" line and cannot tell whether their
                      earlier order traded or was outbid.
    Round-end "expired" annotations are written by ``next_round`` —
    they depend on the standing book at round close, not on a tick's
    apply_order outcome.
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
    if outcome == "no_announcement":
        # Nothing to record on a pass; short-circuit without iterating.
        return [{**a} for a in agents]
    if outcome not in _OUTCOME_LABELS:
        raise ValueError(f"unknown order_outcome {outcome!r}")
    label = _OUTCOME_LABELS[outcome]

    # Resolve the back-annotation target (if any) up front so the
    # per-agent loop stays a single pass.
    backann_owner_id: int | None = None
    backann_kind: str | None = None       # "filled" | "outbid"
    backann_price: float | None = None
    backann_side: str | None = None       # "buy" | "sell"

    if outcome == "traded":
        # The counterparty's resting order was just crossed. Trade
        # executes at the standing (i.e. counterparty's) price; the
        # side they were on is the opposite of the announcer's.
        backann_owner_id = state.get("counterparty_agent_id")
        if backann_owner_id is not None:
            backann_kind = "filled"
            backann_price = state["announced_price"]  # equals trade price
            ann_type = state.get("announcement_type")
            backann_side = "sell" if ann_type == "buy" else "buy"
    elif outcome == "posted":
        prior_owner = state.get("replaced_standing_owner_id")
        if prior_owner is not None and prior_owner != announcer_id:
            backann_owner_id = prior_owner
            backann_kind = "outbid"
            backann_price = state.get("replaced_standing_price")
            backann_side = state.get("replaced_standing_side")

    updated = []
    for agent in agents:
        agent_copy = {**agent}

        if announcer_id is not None and agent["id"] == announcer_id:
            history_entry = {
                "round": state["round"],
                "iteration": state["iteration"],
                "action": "announce",
                "price": state["announced_price"],
                "outcome": label,
            }
            agent_copy["own_history_data"] = agent["own_history_data"] + [history_entry]

            ann_type = "buy" if agent["type"] == "buyer" else "sell"
            # Non-improving orders use a dedicated template that includes
            # the rejection reason. Other outcomes use the generic
            # template with the {outcome} label substituted.
            if outcome == "non_improving":
                entry = templates.announcement_history_non_improving_template.format(
                    round=state["round"],
                    iteration=state["iteration"],
                    announcement_type=ann_type,
                    price=state["announced_price"],
                )
            else:
                entry = templates.announcement_history_template.format(
                    round=state["round"],
                    iteration=state["iteration"],
                    announcement_type=ann_type,
                    price=state["announced_price"],
                    outcome=label,
                )
            agent_copy["own_history_prompt"] = agent["own_history_prompt"] + entry

        # Back-annotation: the resting-order owner whose order was just
        # filled or replaced. Same agent can be the announcer (e.g. a
        # buyer crossing) and the back-annotation target (e.g. an
        # earlier seller posting), but only across distinct ticks —
        # within one tick the two roles are always different agents,
        # so this is an `elif` against the announcer branch.
        if backann_owner_id is not None and agent["id"] == backann_owner_id:
            backann_entry = {
                "round": state["round"],
                "iteration": state["iteration"],
                "action": "order_update",
                "price": backann_price,
                "outcome": backann_kind,
            }
            agent_copy["own_history_data"] = (
                agent_copy["own_history_data"] + [backann_entry]
            )
            if backann_kind == "filled":
                tmpl = templates.announcement_history_filled_template
            else:
                tmpl = templates.announcement_history_outbid_template
            entry = tmpl.format(
                round=state["round"],
                iteration=state["iteration"],
                announcement_type=backann_side,
                price=backann_price,
            )
            agent_copy["own_history_prompt"] = (
                agent_copy["own_history_prompt"] + entry
            )

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
            "replaced_standing_owner_id": None,
            "replaced_standing_price": None,
            "replaced_standing_side": None,
        }

    return next_iteration


def make_next_round_node(
    prompts: PromptConfig | None = None,
) -> Callable[[MarketState], dict]:
    """Advance to the next round (or end the simulation).

    Before clearing the order book, any agent who still owns a standing
    bid or ask gets a back-annotation row in their own_history saying
    the order expired uncrossed at round close. Without this, an agent
    sees only their original "posted" line and never learns whether the
    order eventually traded.
    """

    templates = _get_templates(prompts)

    def _annotate_expired(
        agent: dict, side: str, price: float, round_num: int, iteration: int
    ) -> dict:
        entry = {
            "round": round_num,
            "iteration": iteration,
            "action": "order_update",
            "price": price,
            "outcome": "expired",
        }
        prompt_line = templates.announcement_history_expired_template.format(
            round=round_num,
            announcement_type=side,
            price=price,
        )
        return {
            **agent,
            "own_history_data": agent["own_history_data"] + [entry],
            "own_history_prompt": agent["own_history_prompt"] + prompt_line,
        }

    def next_round(state: MarketState) -> dict:
        new_round = state["round"] + 1
        max_rounds = state["max_rounds"]

        # Standing-order expiry: surface to the resting owners before
        # the book clears. Done for both branches (mid-simulation and
        # final-round) so the last-round agent still learns the fate of
        # their unfilled orders even though no new round will follow.
        round_num = state["round"]
        expired_owners: dict[int, list[tuple[str, float]]] = {}
        sb_owner = state.get("standing_bid_agent_id")
        if sb_owner is not None and state.get("standing_bid") is not None:
            expired_owners.setdefault(sb_owner, []).append(
                ("buy", state["standing_bid"])
            )
        sa_owner = state.get("standing_ask_agent_id")
        if sa_owner is not None and state.get("standing_ask") is not None:
            expired_owners.setdefault(sa_owner, []).append(
                ("sell", state["standing_ask"])
            )

        last_iter = state["iteration"]
        annotated_agents = []
        for agent in state["agents"]:
            if agent["id"] in expired_owners:
                ann = agent
                for side, price in expired_owners[agent["id"]]:
                    ann = _annotate_expired(ann, side, price, round_num, last_iter)
                annotated_agents.append(ann)
            else:
                annotated_agents.append(agent)

        if new_round > max_rounds:
            tx_count = len(state.get("transactions", []))
            logger.info(f"Simulation complete: all {max_rounds} rounds finished ({tx_count} total transactions)")
            result: dict = {"simulation_complete": True}
            if expired_owners:
                # Persist the round-end annotations into agent state
                # before the simulation terminates.
                result["agents"] = annotated_agents
            return result

        tx_in_round = sum(
            1 for t in state.get("transactions", []) if t["round"] == state["round"]
        )
        logger.info(
            f"Round {state['round']} completed with {tx_in_round} transaction(s). "
            f"Advancing to round {new_round}"
        )

        updated_agents = []
        all_agent_ids = []
        for agent in annotated_agents:
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
