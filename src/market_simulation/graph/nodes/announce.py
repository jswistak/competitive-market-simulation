"""Announcement-related graph nodes."""

import random
import logging
from typing import Callable, Any

from langchain_core.runnables import RunnableConfig

from ..state import MarketState
from ...llm.providers.base import LLMProvider
from ...config.schema import PromptConfig

logger = logging.getLogger(__name__)


def make_select_announcer_node() -> Callable[[MarketState], dict]:
    """Create node that selects the next agent to make an announcement.

    Returns:
        Node function that updates announcing_agent_id.
    """

    def select_announcer(state: MarketState) -> dict:
        """Select a random active agent to announce (who hasn't announced yet this iteration)."""
        active_ids = state["active_agent_ids"]
        already_announced = state.get("announced_this_iteration", [])

        # Filter out agents who already announced this iteration
        eligible_ids = [aid for aid in active_ids if aid not in already_announced]

        if not eligible_ids:
            logger.info("No eligible agents remaining for announcements this iteration")
            return {
                "announcing_agent_id": None,
                "announcement_made": False,
            }

        # Shuffle and pick first
        shuffled = eligible_ids.copy()
        random.shuffle(shuffled)
        announcer_id = shuffled[0]

        logger.info(f"Selected agent {announcer_id} to announce")
        return {
            "announcing_agent_id": announcer_id,
        }

    return select_announcer


def make_announce_node(
    llm: LLMProvider,
    prompts: PromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> Callable[[MarketState, RunnableConfig], dict]:
    """Create node that handles agent price announcements.

    Args:
        llm: LLM provider for generating announcements.
        prompts: Prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks (deprecated,
            prefer passing callbacks via graph config).

    Returns:
        Node function that generates price announcement.
    """

    def announce(state: MarketState, config: RunnableConfig) -> dict:
        """Agent announces a price via LLM call."""
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
            return {"announcement_made": False, "announced_price": None, "last_error": f"Agent {agent_id} not found"}

        # Determine agent type and get appropriate config
        agent_type = agent["type"]
        if agent_type == "buyer":
            agent_prompts = prompts.buyer
        else:
            agent_prompts = prompts.seller

        if agent_prompts is None:
            logger.error(f"No prompts configured for {agent_type}")
            return {"announcement_made": False, "announced_price": None}

        # Render prompt
        prompt = _render_announcement_prompt(
            agent=agent,
            state=state,
            prompts=prompts,
            agent_prompts=agent_prompts,
        )

        # Get callbacks from config (propagated from graph.invoke)
        # Falls back to callbacks_factory for backwards compatibility
        callbacks = config.get("callbacks", []) if config else []
        if not callbacks and callbacks_factory:
            callbacks = callbacks_factory()

        try:
            response = llm.invoke(prompt, callbacks=callbacks)
            price = _extract_price(response)

            if price is None:
                logger.warning(f"Could not parse price from response: {response}")
                return {
                    "announcement_made": False,
                    "announced_price": None,
                    "last_error": f"Could not parse price: {response}",
                }

            announcement_type = "buy" if agent_type == "buyer" else "sell"
            logger.info(f"Agent {agent_id} ({agent_type}) announced {announcement_type} at ${price:.2f}")

            # Track that this agent has announced this iteration
            announced_this_iteration = state.get("announced_this_iteration", []) + [agent_id]

            return {
                "announced_price": price,
                "announcement_type": announcement_type,
                "announcement_made": True,
                "announced_this_iteration": announced_this_iteration,
            }

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "announcement_made": False,
                "announced_price": None,
                "last_error": str(e),
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

    template_vars = {
        "role": keywords.role,
        "verb": keywords.verb,
        "preference": keywords.preference,
        "condition": keywords.condition,
        "reservation_price": agent["reservation_price"],
        "N_ROUNDS": state["max_rounds"],
        "N_ITER": state["max_iterations"],
        "market_history": state["market_history_text"],
        "own_history": agent["own_history_prompt"],
        "round": state["round"],
        "iteration": state["iteration"],
        "action_prompt": agent_prompts.announcement_prompt,
    }

    return prompts.general.main_template.format(**template_vars)


def _extract_price(response: str) -> float | None:
    """Extract price from LLM response."""
    try:
        # Clean and parse
        clean = response.strip().replace("$", "").replace(",", "")
        return float(clean)
    except ValueError:
        return None
