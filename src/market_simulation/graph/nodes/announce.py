"""Announcement-related graph nodes."""

import random
import logging
from typing import Callable, Any

import numpy as np
from langchain_core.runnables import RunnableConfig

from ..state import MarketState
from ..history import build_market_history_for_prompt, build_own_history_for_prompt
from ...agents import zi as zi_decisions
from ...llm.providers.base import LLMProvider
from ...llm.response_schemas import AnnouncementResponse, AnnouncementResponseWithReasoning
from ...config.schema import PromptConfig, ZIConfig

logger = logging.getLogger(__name__)


def make_select_announcer_node() -> Callable[[MarketState], dict]:
    """Create node that selects the next agent to act on this tick.

    Under the improvement-rule CDA, each tick is one randomly-chosen
    active agent posting an order. Unlike the old mechanism, agents are
    NOT filtered by "already announced this iteration" — the same agent
    can act on multiple ticks per round as long as it is still active.

    Returns:
        Node function that updates announcing_agent_id.
    """

    def select_announcer(state: MarketState) -> dict:
        active_ids = state["active_agent_ids"]

        if not active_ids:
            logger.info("No active agents remaining this round")
            return {
                "announcing_agent_id": None,
                "announcement_made": False,
            }

        shuffled = list(active_ids)
        random.shuffle(shuffled)
        announcer_id = shuffled[0]

        logger.info(f"T{state['iteration']}: selected agent {announcer_id} to act")
        return {
            "announcing_agent_id": announcer_id,
        }

    return select_announcer


def make_announce_node(
    llm: LLMProvider | None,
    prompts: PromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    response_schema: type[AnnouncementResponse] = AnnouncementResponseWithReasoning,
    zi_config: ZIConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Callable[[MarketState, RunnableConfig], dict]:
    """Create node that handles agent price announcements.

    Args:
        llm: LLM provider for generating announcements. May be ``None`` if
            all configured agents use a zero-intelligence strategy.
        prompts: Prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks (deprecated,
            prefer passing callbacks via graph config).
        response_schema: Pydantic schema for structured output.
        zi_config: Hyperparameters for zero-intelligence sampling.
        rng: Seeded NumPy ``Generator`` for ZI randomness. Constructed
            per-factory call so the graph stays deterministic under a seed.

    Returns:
        Node function that generates price announcement.
    """

    zi_cfg = zi_config or ZIConfig()
    zi_rng = rng if rng is not None else np.random.default_rng()
    include_reasoning = response_schema is AnnouncementResponseWithReasoning

    def announce(state: MarketState, config: RunnableConfig) -> dict:
        """Agent announces a price via LLM call or ZI sampling."""
        agent_id = state["announcing_agent_id"]

        if agent_id is None:
            return {"announcement_made": False, "announced_price": None}

        # Find the agent
        agent = None
        for a in state["agents"]:
            if a["id"] == agent_id:
                agent = a
                break

        if agent is None:
            logger.error(f"Agent {agent_id} not found")
            return {
                "announcement_made": False,
                "announced_price": None,
                "last_error": f"Agent {agent_id} not found",
            }

        agent_type = agent["type"]
        strategy = agent.get("strategy", "llm")

        try:
            if strategy == "llm":
                if llm is None:
                    raise RuntimeError(
                        "Agent has strategy='llm' but no LLM provider was supplied"
                    )
                if agent_type == "buyer":
                    agent_prompts = prompts.buyer
                else:
                    agent_prompts = prompts.seller

                if agent_prompts is None:
                    logger.error(f"No prompts configured for {agent_type}")
                    return {"announcement_made": False, "announced_price": None}

                prompt = _render_announcement_prompt(
                    agent=agent,
                    state=state,
                    prompts=prompts,
                    agent_prompts=agent_prompts,
                )
                logger.debug(
                    f"Announcement prompt for agent {agent_id} (truncated): "
                    f"'{prompt[:200]}...'"
                )

                callbacks = config.get("callbacks", []) if config else []
                if not callbacks and callbacks_factory:
                    callbacks = callbacks_factory()

                call_metadata = {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "action": "announce",
                    "round": state["round"],
                    "iteration": state["iteration"],
                    "simulation_id": state["simulation_id"],
                    "strategy": strategy,
                }
                response = llm.invoke_structured(
                    prompt, response_schema, callbacks=callbacks, metadata=call_metadata,
                )
            else:
                response = zi_decisions.decide_announce(
                    agent=agent,
                    zi_cfg=zi_cfg,
                    rng=zi_rng,
                    include_reasoning=include_reasoning,
                    standing_bid=state.get("standing_bid"),
                    standing_ask=state.get("standing_ask"),
                )
            price = response.price
            reasoning = getattr(response, 'reasoning', '')
            logger.debug(
                f"Structured announcement for agent {agent_id}: price={price}, reasoning='{reasoning[:100]}...'"
            )

            # Capture tool usage log if available (ZI path has none)
            tool_log_entries = getattr(llm, "last_tool_log", []) if strategy == "llm" else []
            tool_usage_log = [
                {
                    **entry,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "action": "announce",
                    "round": state["round"],
                    "iteration": state["iteration"],
                    "simulation_id": state["simulation_id"],
                }
                for entry in tool_log_entries
            ]

            if price is None:
                logger.info(
                    f"Agent {agent_id} chose not to announce "
                    f"(R{state['round']}/I{state['iteration']})"
                )
                return {
                    "announcement_made": False,
                    "announced_price": None,
                    "tool_usage_log": tool_usage_log,
                    "last_announcement_reasoning": reasoning,
                }

            # Check for reservation price constraint violation (log only)
            reservation = agent["reservation_price"]
            violation = False
            if agent_type == "buyer" and price > reservation:
                logger.warning(
                    f"CONSTRAINT VIOLATION: Buyer {agent_id} announced ${price:.2f} "
                    f"above reservation ${reservation:.2f}"
                )
                violation = True
            elif agent_type == "seller" and price < reservation:
                logger.warning(
                    f"CONSTRAINT VIOLATION: Seller {agent_id} announced ${price:.2f} "
                    f"below reservation ${reservation:.2f}"
                )
                violation = True

            announcement_type = "buy" if agent_type == "buyer" else "sell"
            logger.info(
                f"Agent {agent_id} ({agent_type}, reservation=${reservation:.2f}) "
                f"announced {announcement_type} at ${price:.2f}"
            )

            result = {
                "announced_price": price,
                "announcement_type": announcement_type,
                "announcement_made": True,
                "tool_usage_log": tool_usage_log,
                "last_announcement_reasoning": reasoning,
            }
            if violation:
                result["constraint_violations"] = (
                    state.get("constraint_violations", 0) + 1
                )
            return result

        except Exception as e:
            if strategy == "llm":
                logger.error(
                    f"LLM call failed for agent {agent_id} "
                    f"(R{state['round']}/I{state['iteration']}): {e}"
                )
            else:
                logger.error(
                    f"ZI decision failed for agent {agent_id} ({strategy}) "
                    f"(R{state['round']}/I{state['iteration']}): {e}"
                )
            return {
                "announcement_made": False,
                "announced_price": None,
                "last_error": str(e),
                "last_announcement_reasoning": "",
            }

    return announce


def _render_announcement_prompt(
    agent: dict,
    state: MarketState,
    prompts: PromptConfig,
    agent_prompts,
) -> str:
    """Render the announcement prompt for an agent."""
    keywords = agent_prompts.main_keywords

    # Render the standing book as human-readable strings so prompt
    # templates can embed them without dealing with None.
    standing_bid = state.get("standing_bid")
    standing_ask = state.get("standing_ask")
    standing_bid_str = f"${standing_bid:.2f}" if standing_bid is not None else "none"
    standing_ask_str = f"${standing_ask:.2f}" if standing_ask is not None else "none"

    template_vars = {
        "role": keywords.role,
        "verb": keywords.verb,
        "preference": keywords.preference,
        "condition": keywords.condition,
        "profit_formula": keywords.profit_formula,
        "order_outcomes": keywords.order_outcomes,
        "reservation_price": agent["reservation_price"],
        "N_ROUNDS": state["max_rounds"],
        "N_ITER": state["max_iterations"],
        "N_BUYERS": sum(1 for a in state["agents"] if a["type"] == "buyer"),
        "N_SELLERS": sum(1 for a in state["agents"] if a["type"] == "seller"),
        "market_history": build_market_history_for_prompt(
            state,
            mode=state.get("history_mode", "full"),
            last_n_events=state.get("history_summary_last_n", 3),
        ),
        "own_history": build_own_history_for_prompt(
            agent,
            mode=state.get("own_history_mode", "full"),
        ),
        "round": state["round"],
        "iteration": state["iteration"],
        "action_prompt": agent_prompts.announcement_prompt,
        "persona": agent.get("persona", ""),
        "tools_preamble": prompts.tools_preamble,
        # Improvement-rule CDA: the standing book is authoritative
        # market state the agent needs to see.
        "standing_bid": standing_bid_str,
        "standing_ask": standing_ask_str,
    }

    # Use sentinel replacement for persona to avoid str.format() issues with curly braces
    persona_text = template_vars.pop("persona")
    template = prompts.general.main_template.replace("{persona}", "<<PERSONA>>")
    result = template.format(**template_vars)
    return result.replace("<<PERSONA>>", persona_text)


