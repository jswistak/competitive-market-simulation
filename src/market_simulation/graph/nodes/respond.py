"""Response-related graph nodes."""

import random
import logging
from typing import Callable, Any

from langchain_core.runnables import RunnableConfig

from ..state import MarketState
from ...llm.providers.base import LLMProvider
from ...config.schema import PromptConfig

logger = logging.getLogger(__name__)


def make_select_responders_node() -> Callable[[MarketState], dict]:
    """Create node that selects potential responders to an announcement.

    Returns:
        Node function that sets potential_responder_ids.
    """

    def select_responders(state: MarketState) -> dict:
        """Select agents who can respond (opposite type from announcer)."""
        announcement_type = state["announcement_type"]

        if announcement_type is None:
            return {"potential_responder_ids": [], "current_responder_index": 0}

        # Find announcing agent type
        announcing_id = state["announcing_agent_id"]
        announcer_type = None
        for agent in state["agents"]:
            if agent["id"] == announcing_id:
                announcer_type = agent["type"]
                break

        # Get opposite type agents who are still active
        target_type = "seller" if announcer_type == "buyer" else "buyer"

        responders = [
            agent["id"]
            for agent in state["agents"]
            if agent["type"] == target_type and agent["id"] in state["active_agent_ids"]
        ]

        # Shuffle for randomness
        random.shuffle(responders)

        logger.info(f"Found {len(responders)} potential responders ({target_type}s)")
        return {"potential_responder_ids": responders, "current_responder_index": 0}

    return select_responders


def make_respond_node(
    llm: LLMProvider,
    prompts: PromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
) -> Callable[[MarketState, RunnableConfig], dict]:
    """Create node that handles agent responses to announcements.

    Args:
        llm: LLM provider for generating responses.
        prompts: Prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks (deprecated,
            prefer passing callbacks via graph config).

    Returns:
        Node function that generates response.
    """

    def respond(state: MarketState, config: RunnableConfig) -> dict:
        """Agent responds to an announcement via LLM call."""
        responder_ids = state["potential_responder_ids"]
        current_idx = state["current_responder_index"]

        if current_idx >= len(responder_ids):
            logger.info("No more potential responders")
            return {
                "responding_agent_id": None,
                "response_accepted": False,
                "transaction_made": False,
            }

        responder_id = responder_ids[current_idx]

        # Find the responding agent
        responder = None
        for agent in state["agents"]:
            if agent["id"] == responder_id:
                responder = agent
                break

        if responder is None:
            logger.error(f"Responder {responder_id} not found")
            return {
                "responding_agent_id": responder_id,
                "response_accepted": False,
                "current_responder_index": current_idx + 1,
            }

        # Get appropriate prompts
        agent_type = responder["type"]
        if agent_type == "buyer":
            agent_prompts = prompts.buyer
        else:
            agent_prompts = prompts.seller

        if agent_prompts is None:
            return {
                "responding_agent_id": responder_id,
                "response_accepted": False,
                "current_responder_index": current_idx + 1,
            }

        # Render prompt
        prompt = _render_response_prompt(
            agent=responder,
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
            accepted = _extract_response(response)

            logger.info(
                f"Agent {responder_id} ({agent_type}) {'accepted' if accepted else 'rejected'} "
                f"offer at ${state['announced_price']:.2f}"
            )

            return {
                "responding_agent_id": responder_id,
                "response_accepted": accepted,
                "transaction_made": accepted,
                "current_responder_index": current_idx + 1,
            }

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "responding_agent_id": responder_id,
                "response_accepted": False,
                "current_responder_index": current_idx + 1,
                "last_error": str(e),
            }

    return respond


def _render_response_prompt(
    agent: dict,
    state: MarketState,
    prompts: PromptConfig,
    agent_prompts,
) -> str:
    """Render the response prompt for an agent."""
    keywords = agent_prompts.main_keywords

    # Format response prompt with price
    action_prompt = agent_prompts.response_prompt.format(price=state["announced_price"])

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
        "action_prompt": action_prompt,
    }

    return prompts.general.main_template.format(**template_vars)


def _extract_response(response: str) -> bool:
    """Extract yes/no response from LLM output."""
    return "yes" in response.lower()
