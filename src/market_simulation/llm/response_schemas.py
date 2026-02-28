"""Pydantic response models for structured LLM output.

Note: Answer fields are placed before 'reasoning' so that models
generating JSON in field order (e.g. Gemini) produce the answer
before potentially running out of output tokens on a long reasoning.
"""

from typing import Literal, NamedTuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base schemas (answer fields only — no reasoning)
# ---------------------------------------------------------------------------


class AnnouncementResponse(BaseModel):
    """Structured response for price announcements."""

    price: float | None = Field(
        description="The price to announce, or null if you do not want to announce."
    )


class AcceptRejectResponse(BaseModel):
    """Structured response for accept/reject decisions."""

    accept: bool = Field(description="Whether you accept (true) or reject (false).")


class BidResponse(BaseModel):
    """Structured response for sealed-bid auctions."""

    bid: float = Field(description="Your bid amount.")


class EnglishBidResponse(BaseModel):
    """Structured response for English/open-outcry auction bids."""

    action: Literal["bid", "pass"] = Field(
        description="Whether to bid or pass (drop out)."
    )
    bid: float | None = Field(
        default=None, description="Your bid amount if action is 'bid'."
    )


# ---------------------------------------------------------------------------
# WithReasoning variants (add a reasoning field)
# ---------------------------------------------------------------------------


class AnnouncementResponseWithReasoning(AnnouncementResponse):
    """Announcement response with chain-of-thought reasoning."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class AcceptRejectResponseWithReasoning(AcceptRejectResponse):
    """Accept/reject response with chain-of-thought reasoning."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class BidResponseWithReasoning(BidResponse):
    """Sealed-bid response with chain-of-thought reasoning."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class EnglishBidResponseWithReasoning(EnglishBidResponse):
    """English auction response with chain-of-thought reasoning."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


# ---------------------------------------------------------------------------
# Schema selector
# ---------------------------------------------------------------------------


class ResponseSchemas(NamedTuple):
    """Container for the four response schema classes."""

    announcement: type[AnnouncementResponse]
    accept_reject: type[AcceptRejectResponse]
    bid: type[BidResponse]
    english_bid: type[EnglishBidResponse]


def get_response_schemas(include_reasoning: bool = True) -> ResponseSchemas:
    """Return the appropriate set of response schema classes.

    Args:
        include_reasoning: If True, return schemas that include a reasoning
            field. If False, return base schemas without reasoning.
    """
    if include_reasoning:
        return ResponseSchemas(
            announcement=AnnouncementResponseWithReasoning,
            accept_reject=AcceptRejectResponseWithReasoning,
            bid=BidResponseWithReasoning,
            english_bid=EnglishBidResponseWithReasoning,
        )
    return ResponseSchemas(
        announcement=AnnouncementResponse,
        accept_reject=AcceptRejectResponse,
        bid=BidResponse,
        english_bid=EnglishBidResponse,
    )
