"""Announcement-related graph nodes."""

import re
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

        logger.debug(f"Announcement prompt for agent {agent_id} (truncated): '{prompt[:200]}...'")

        # Get callbacks from config (propagated from graph.invoke)
        # Falls back to callbacks_factory for backwards compatibility
        callbacks = config.get("callbacks", []) if config else []
        if not callbacks and callbacks_factory:
            callbacks = callbacks_factory()

        try:
            response = llm.invoke(prompt, callbacks=callbacks)
            logger.debug(f"Raw LLM announcement response for agent {agent_id}: '{response}'")
            price = _extract_price(response)

            # Capture tool usage log if available
            tool_log_entries = getattr(llm, "last_tool_log", [])
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
                logger.warning(
                    f"Could not parse price from agent {agent_id} "
                    f"(R{state['round']}/I{state['iteration']}): '{response}'"
                )
                return {
                    "announcement_made": False,
                    "announced_price": None,
                    "last_error": f"Could not parse price: {response}",
                    "tool_usage_log": tool_usage_log,
                    "parse_failures": state.get("parse_failures", 0) + 1,
                    "announced_this_iteration": state.get("announced_this_iteration", []) + [agent_id],
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

            # Track that this agent has announced this iteration
            announced_this_iteration = state.get("announced_this_iteration", []) + [agent_id]

            result = {
                "announced_price": price,
                "announcement_type": announcement_type,
                "announcement_made": True,
                "announced_this_iteration": announced_this_iteration,
                "tool_usage_log": tool_usage_log,
            }
            if violation:
                result["constraint_violations"] = state.get("constraint_violations", 0) + 1
            return result

        except Exception as e:
            logger.error(f"LLM call failed for agent {agent_id} (R{state['round']}/I{state['iteration']}): {e}")
            return {
                "announcement_made": False,
                "announced_price": None,
                "last_error": str(e),
                "announced_this_iteration": state.get("announced_this_iteration", []) + [agent_id],
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
    """Extract price from LLM response.

    Handles both plain numbers and longer tool-augmented responses
    where the number may be embedded in reasoning text.

    Extraction priority:
      1. Plain float parse (after stripping $ and ,)
      2. Last $-prefixed number (e.g. "$3.27")
      3. Last bare decimal number (e.g. "1.50") — requires decimal point
         to avoid extracting round/iteration numbers like "round 1"
    """
    if not response or not response.strip():
        return None

    # Stage 1: plain parse (most common case without tools)
    # Validate format to reject negative numbers and scientific notation
    clean = response.strip().replace("$", "").replace(",", "")
    if re.fullmatch(r"\d+\.?\d*", clean):
        return float(clean)

    # Stage 2: prefer $-prefixed numbers (most reliable signal)
    dollar_matches = re.findall(r"\$([\d]+\.?\d*)", response)
    if dollar_matches:
        try:
            extracted = float(dollar_matches[-1])
            logger.warning(
                f"Price extracted via Stage 2 ($-prefix fallback). "
                f"Response ({len(response)} chars): '{response}', extracted: {extracted}"
            )
            return extracted
        except ValueError:
            pass

    # Stage 3: bare decimal numbers only (require decimal point to avoid
    # extracting round/iteration numbers like "round 1")
    bare_matches = re.findall(r"(?<![\w\-])(\d+\.\d+)(?!\w)", response)
    if bare_matches:
        try:
            extracted = float(bare_matches[-1])
            logger.warning(
                f"Price extracted via Stage 3 (bare-decimal fallback). "
                f"Response ({len(response)} chars): '{response}', extracted: {extracted}"
            )
            return extracted
        except ValueError:
            pass

    return None
