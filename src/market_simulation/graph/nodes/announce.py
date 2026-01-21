"""Announcement-related graph nodes."""

import random
import logging
from typing import Callable

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
        """Select a random active agent to announce."""
        active_ids = state["active_agent_ids"]

        if not active_ids:
            logger.info("No active agents remaining for announcements")
            return {
                "announcing_agent_id": None,
                "announcement_made": False,
                "iteration_complete": True,
            }

        # Shuffle and pick first
        shuffled = active_ids.copy()
        random.shuffle(shuffled)
        announcer_id = shuffled[0]

        logger.info(f"Selected agent {announcer_id} to announce")
        return {
            "announcing_agent_id": announcer_id,
            "active_agent_ids": shuffled,  # Keep shuffled order
        }

    return select_announcer


def make_announce_node(
    llm: LLMProvider,
    prompts: PromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> Callable[[MarketState], dict]:
    """Create node that handles agent price announcements.

    Args:
        llm: LLM provider for generating announcements.
        prompts: Prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.

    Returns:
        Node function that generates price announcement.
    """

    def announce(state: MarketState) -> dict:
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

        # Call LLM
        callbacks = callbacks_factory() if callbacks_factory else []
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

            return {
                "announced_price": price,
                "announcement_type": announcement_type,
                "announcement_made": True,
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
