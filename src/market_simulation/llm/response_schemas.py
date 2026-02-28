"""Pydantic response models for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


class AnnouncementResponse(BaseModel):
    """Structured response for price announcements."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")
    price: float | None = Field(
        description="The price to announce, or null if you do not want to announce."
    )


class AcceptRejectResponse(BaseModel):
    """Structured response for accept/reject decisions."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")
    accept: bool = Field(description="Whether you accept (true) or reject (false).")


class BidResponse(BaseModel):
    """Structured response for sealed-bid auctions."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")
    bid: float = Field(description="Your bid amount.")


class EnglishBidResponse(BaseModel):
    """Structured response for English/open-outcry auction bids."""

    reasoning: str = Field(default="", description="Your step-by-step reasoning.")
    action: Literal["bid", "pass"] = Field(
        description="Whether to bid or pass (drop out)."
    )
    bid: float | None = Field(
        default=None, description="Your bid amount if action is 'bid'."
    )
