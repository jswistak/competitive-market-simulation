"""Response-related graph nodes."""

import random
import logging
from typing import Callable, Any

from langchain_core.runnables import RunnableConfig

from ..state import MarketState
from ..history import build_market_history_for_prompt, build_own_history_for_prompt
from ...llm.providers.base import LLMProvider
from ...llm.response_schemas import AcceptRejectResponse, AcceptRejectResponseWithReasoning
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

        if announcer_type is None:
            logger.error(
                f"Announcing agent {announcing_id} not found in agents list. "
                f"Returning empty responders."
            )
            return {"potential_responder_ids": [], "current_responder_index": 0}

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
    response_schema: type[AcceptRejectResponse] = AcceptRejectResponseWithReasoning,
) -> Callable[[MarketState, RunnableConfig], dict]:
    """Create node that handles agent responses to announcements.

    Args:
        llm: LLM provider for generating responses.
        prompts: Prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks (deprecated,
            prefer passing callbacks via graph config).
        response_schema: Pydantic schema for structured output.

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
            response = llm.invoke_structured(prompt, response_schema, callbacks=callbacks)
            accepted = response.accept
            reasoning = getattr(response, 'reasoning', '')
            logger.debug(
                f"Structured response for agent {responder_id}: accept={accepted}, reasoning='{reasoning[:100]}...'"
            )

            # Capture tool usage log if available
            tool_log_entries = getattr(llm, "last_tool_log", [])
            tool_usage_log = [
                {
                    **entry,
                    "agent_id": responder_id,
                    "agent_type": agent_type,
                    "action": "respond",
                    "round": state["round"],
                    "iteration": state["iteration"],
                    "simulation_id": state["simulation_id"],
                }
                for entry in tool_log_entries
            ]

            # Check for reservation price constraint violation (log only)
            violation = False
            if accepted:
                reservation = responder["reservation_price"]
                announced_price = state["announced_price"]
                if agent_type == "buyer" and announced_price > reservation:
                    logger.warning(
                        f"CONSTRAINT VIOLATION: Buyer {responder_id} accepted buy at "
                        f"${announced_price:.2f} above reservation ${reservation:.2f}"
                    )
                    violation = True
                elif agent_type == "seller" and announced_price < reservation:
                    logger.warning(
                        f"CONSTRAINT VIOLATION: Seller {responder_id} accepted sell at "
                        f"${announced_price:.2f} below reservation ${reservation:.2f}"
                    )
                    violation = True

            logger.info(
                f"R{state['round']}/I{state['iteration']}: Agent {responder_id} ({agent_type}) "
                f"{'accepted' if accepted else 'rejected'} offer at ${state['announced_price']:.2f}"
            )

            result = {
                "responding_agent_id": responder_id,
                "response_accepted": accepted,
                "transaction_made": accepted,
                "current_responder_index": current_idx + 1,
                "tool_usage_log": tool_usage_log,
                "last_response_reasoning": reasoning,
            }
            if violation:
                result["constraint_violations"] = state.get("constraint_violations", 0) + 1
            return result

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {
                "responding_agent_id": responder_id,
                "response_accepted": False,
                "current_responder_index": current_idx + 1,
                "last_error": str(e),
                "last_response_reasoning": "",
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
        "action_prompt": action_prompt,
        "persona": agent.get("persona", ""),
    }

    # Use sentinel replacement for persona to avoid str.format() issues with curly braces
    persona_text = template_vars.pop("persona")
    template = prompts.general.main_template.replace("{persona}", "<<PERSONA>>")
    result = template.format(**template_vars)
    return result.replace("<<PERSONA>>", persona_text)


