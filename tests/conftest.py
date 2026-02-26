"""Shared pytest fixtures for market simulation tests."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from market_simulation.config.schema import (
    SimulationConfig,
    ExperimentConfig,
    LLMConfig,
    AgentPricesConfig,
    TracingConfig,
    ToolConfig,
    PromptConfig,
    PromptTemplates,
    AgentPromptConfig,
    AgentKeywords,
)
from market_simulation.graph.state import MarketState, AgentState


MAIN_TEMPLATE = (
    "You are a {role}. Verb: {verb}. Preference: {preference}. "
    "Condition: {condition}. Reservation: {reservation_price}. "
    "Rounds: {N_ROUNDS}. Iters: {N_ITER}. "
    "Buyers: {N_BUYERS}. Sellers: {N_SELLERS}. "
    "Market: {market_history}. Own: {own_history}. "
    "Round {round}/{N_ROUNDS} Iter {iteration}/{N_ITER}. {action_prompt}"
)


@pytest.fixture
def buyer_keywords():
    return AgentKeywords(role="buyer", verb="buy", preference="lowest", condition="above")


@pytest.fixture
def seller_keywords():
    return AgentKeywords(role="seller", verb="sell", preference="highest", condition="below")


@pytest.fixture
def prompt_config(buyer_keywords, seller_keywords):
    return PromptConfig(
        general=PromptTemplates(
            main_template=MAIN_TEMPLATE,
            announcement_history_template=(
                "Round {round} iter {iteration}: {announcement_type} ${price:.2f} {outcome}.\n"
            ),
            response_history_template=(
                "Round {round} iter {iteration}: {outcome} {opposite_announcement_type} ${price:.2f}.\n"
            ),
        ),
        buyer=AgentPromptConfig(
            main_keywords=buyer_keywords,
            response_prompt="Sell at ${price:.2f}. Buy? yes/no.",
            announcement_prompt="Announce bid price as number.",
        ),
        seller=AgentPromptConfig(
            main_keywords=seller_keywords,
            response_prompt="Buy at ${price:.2f}. Sell? yes/no.",
            announcement_prompt="Announce ask price as number.",
        ),
    )


@pytest.fixture
def experiment_config():
    return ExperimentConfig(
        n_rounds=2,
        n_iterations=3,
        n_simulations=1,
        buyers=AgentPricesConfig(min=1.0, max=2.0, num=3),
        sellers=AgentPricesConfig(min=1.0, max=2.0, num=3),
    )


@pytest.fixture
def llm_config():
    return LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=50,
        max_retries=3,
    )


@pytest.fixture
def tracing_config():
    return TracingConfig(enabled=False)


@pytest.fixture
def tool_config():
    return ToolConfig(enabled=False)


@pytest.fixture
def simulation_config(experiment_config, llm_config, tracing_config, prompt_config, tool_config):
    return SimulationConfig(
        experiment=experiment_config,
        llm=llm_config,
        tracing=tracing_config,
        prompts=prompt_config,
        tools=tool_config,
    )


@pytest.fixture
def sample_buyer():
    return AgentState(
        id=0,
        type="buyer",
        reservation_price=2.0,
        active=True,
        own_history_prompt="",
        own_history_data=[],
        persona="",
    )


@pytest.fixture
def sample_seller():
    return AgentState(
        id=3,
        type="seller",
        reservation_price=1.0,
        active=True,
        own_history_prompt="",
        own_history_data=[],
        persona="",
    )


@pytest.fixture
def sample_agents():
    """Three buyers (ids 0-2) and three sellers (ids 3-5)."""
    return [
        AgentState(id=0, type="buyer", reservation_price=2.0, active=True, own_history_prompt="", own_history_data=[], persona=""),
        AgentState(id=1, type="buyer", reservation_price=1.5, active=True, own_history_prompt="", own_history_data=[], persona=""),
        AgentState(id=2, type="buyer", reservation_price=1.0, active=True, own_history_prompt="", own_history_data=[], persona=""),
        AgentState(id=3, type="seller", reservation_price=1.0, active=True, own_history_prompt="", own_history_data=[], persona=""),
        AgentState(id=4, type="seller", reservation_price=1.5, active=True, own_history_prompt="", own_history_data=[], persona=""),
        AgentState(id=5, type="seller", reservation_price=2.0, active=True, own_history_prompt="", own_history_data=[], persona=""),
    ]


@pytest.fixture
def base_market_state(sample_agents):
    """A minimal MarketState for unit-testing graph nodes."""
    all_ids = [a["id"] for a in sample_agents]
    return MarketState(
        round=1,
        iteration=1,
        max_rounds=2,
        max_iterations=3,
        simulation_id=1,
        agents=sample_agents,
        active_agent_ids=all_ids,
        potential_responder_ids=[],
        current_responder_index=0,
        announcing_agent_id=None,
        announced_price=None,
        announcement_type=None,
        responding_agent_id=None,
        response_accepted=None,
        market_history_text="",
        iteration_records=[],
        transactions=[],
        announced_this_iteration=[],
        announcement_made=False,
        transaction_made=False,
        iteration_complete=False,
        round_complete=False,
        simulation_complete=False,
        tool_usage_log=[],
        last_error=None,
        parse_failures=0,
        constraint_violations=0,
    )


@pytest.fixture
def mock_llm():
    """A mock LLM provider that returns configurable responses."""
    llm = MagicMock()
    llm.invoke.return_value = "1.50"
    llm.provider_name = "mock"
    llm.model_name = "mock-model"
    llm.last_tool_log = []
    return llm


@pytest.fixture
def configs_dir():
    """Path to the real configs directory."""
    return Path(__file__).parent.parent / "configs"
