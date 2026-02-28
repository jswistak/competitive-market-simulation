"""Auction graph builders — registry and dispatcher."""

from typing import Callable

from langgraph.graph import StateGraph

from ...llm.providers.base import LLMProvider
from ...config.schema import AuctionType, AuctionPromptConfig

from .sealed_bid import build_sealed_bid_graph
from .english import build_english_graph
from .dutch import build_dutch_graph
from .open_outcry import build_open_outcry_graph


AUCTION_BUILDERS: dict[str, Callable[..., StateGraph]] = {
    AuctionType.FPSB.value: build_sealed_bid_graph,
    AuctionType.SPSB.value: build_sealed_bid_graph,
    AuctionType.ALL_PAY.value: build_sealed_bid_graph,
    AuctionType.ENGLISH.value: build_english_graph,
    AuctionType.DUTCH.value: build_dutch_graph,
    AuctionType.FIRST_PRICE_OPEN_OUTCRY.value: build_open_outcry_graph,
}


def build_auction_graph(
    auction_type: str | AuctionType,
    llm: LLMProvider,
    prompts: AuctionPromptConfig,
    callbacks_factory: Callable[[], list] | None = None,
    random_seed: int | None = None,
    include_reasoning: bool = True,
) -> StateGraph:
    """Build the appropriate auction graph for the given type.

    Args:
        auction_type: Auction mechanism identifier.
        llm: LLM provider for bidder interactions.
        prompts: Auction prompt configuration.
        callbacks_factory: Optional factory for tracing callbacks.
        random_seed: Optional seed for deterministic random operations
            (e.g. bidder shuffling in Dutch auctions).
        include_reasoning: Whether to include reasoning field in LLM responses.

    Returns:
        Compiled StateGraph ready for execution.

    Raises:
        ValueError: If auction_type is not supported.
    """
    type_str = auction_type.value if isinstance(auction_type, AuctionType) else auction_type

    builder = AUCTION_BUILDERS.get(type_str)
    if builder is None:
        supported = ", ".join(sorted(AUCTION_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported auction type: {type_str}. Supported: {supported}"
        )

    if type_str == AuctionType.DUTCH.value:
        return builder(
            type_str, llm, prompts, callbacks_factory,
            random_seed=random_seed, include_reasoning=include_reasoning,
        )

    return builder(type_str, llm, prompts, callbacks_factory, include_reasoning=include_reasoning)
