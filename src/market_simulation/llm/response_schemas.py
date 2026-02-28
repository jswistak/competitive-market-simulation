"""Pydantic response models for structured LLM output.

Note: Answer fields are placed before 'reasoning' so that models
generating JSON in field order (e.g. Gemini) produce the answer
before potentially running out of output tokens on a long reasoning.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnnouncementResponse(BaseModel):
    """Structured response for price announcements."""

    price: float | None = Field(
        description="The price to announce, or null if you do not want to announce."
    )
    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class AcceptRejectResponse(BaseModel):
    """Structured response for accept/reject decisions."""

    accept: bool = Field(description="Whether you accept (true) or reject (false).")
    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class BidResponse(BaseModel):
    """Structured response for sealed-bid auctions."""

    bid: float = Field(description="Your bid amount.")
    reasoning: str = Field(default="", description="Your step-by-step reasoning.")


class EnglishBidResponse(BaseModel):
    """Structured response for English/open-outcry auction bids."""

    action: Literal["bid", "pass"] = Field(
        description="Whether to bid or pass (drop out)."
    )
    bid: float | None = Field(
        default=None, description="Your bid amount if action is 'bid'."
    )
    reasoning: str = Field(default="", description="Your step-by-step reasoning.")
