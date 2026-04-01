from __future__ import annotations

"""Listing-specific schemas for the HDB simulation."""

from concordia.hdb_simulation.models.schemas.listing.qdrant import ListingRecord
from pydantic import BaseModel, Field

from concordia.hdb_simulation.models.schemas.common import (
    BaseBuyer,
    BaseSeller,
    NegotiationHistoryRecord,
)

class PortalBuyer(BaseBuyer):
    negotiation_history: list[NegotiationHistoryRecord] = Field(
        default_factory=list
    )


class PortalSeller(BaseSeller):
    negotiation_history: list[NegotiationHistoryRecord] = Field(
        default_factory=list
    )


class PortalSearchResult(ListingRecord):
    score: float = Field(ge=0.0)


class ListingSchedulerSnapshot(BaseModel):
    week_number: int = Field(ge=0)
    active_player_names: list[str] = Field(default_factory=list)
    completed_weeks: int = Field(ge=0)
    closed_player_count: int = Field(ge=0)
    open_player_count: int = Field(ge=0)
    max_rounds: int = Field(ge=0)


class NegotiationRequest(BaseModel):
    request_id: str
    buyer_id: str
    buyer_name: str
    listing_id: str
    seller_id: str
    week_submitted: int = Field(ge=1)
    message: str
    market_valuation_notes: str = ''


class NegotiationMatch(BaseModel):
    match_id: str
    buyer_id: str
    buyer_name: str
    seller_id: str
    seller_name: str
    listing_id: str
    week_matched: int = Field(ge=1)

class ListingWeeklyBatchOutcome(BaseModel):
    week_number: int = Field(ge=1)
    active_player_names: list[str] = Field(default_factory=list)
    newly_listed_listing_ids: list[str] = Field(default_factory=list)
    buyers_processed: list[str] = Field(default_factory=list)
    sellers_reviewed: list[str] = Field(default_factory=list)
    matched_pairs: list[NegotiationMatch] = Field(default_factory=list)
    closed_player_names: list[str] = Field(default_factory=list)
