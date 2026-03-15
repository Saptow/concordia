"""Listing-specific schemas for the HDB simulation."""

from typing import Optional

from pydantic import BaseModel, Field

from concordia.hdb_simulation.models.schemas.common import (
    BaseBuyer,
    BaseSeller,
    RoleType,
)

# TODO: This is the entry schema that initialises portal state for buyers/sellers that joins from negotiation/get initialised. 
class PortalBuyer(BaseBuyer):
    pass


class PortalSeller(BaseSeller):
    pass


class PortalSearchResult(BaseModel):
    listing_id: str
    seller_id: str
    seller_name: str
    score: float
    listing_price: float
    flat_type: str
    town: str
    summary: str


class BuyerMarketBeliefState(BaseModel):
    buyer_id: str
    base_reservation_price: float = Field(ge=0.0)
    effective_reservation_price: float = Field(ge=0.0)
    latest_market_feedback: str = 'No market feedback yet.'
    feedback_history: list[str] = Field(default_factory=list)
    latest_observed_min_price: float | None = Field(default=None, ge=0.0)
    latest_observed_avg_price: float | None = Field(default=None, ge=0.0)
    latest_observed_max_price: float | None = Field(default=None, ge=0.0)


class ListingSchedulerSnapshot(BaseModel):
    week_number: int = Field(ge=0)
    active_player_names: list[str] = Field(default_factory=list)
    completed_weeks: int = Field(ge=0)
    closed_player_count: int = Field(ge=0)
    open_player_count: int = Field(ge=0)
    max_rounds: int = Field(ge=0)


class ListingBuyerState(BaseModel):
    role: RoleType = Field(default=RoleType.BUYER)
    player_id: str
    player_name: str
    budget_min_price: float
    budget_max_price: float
    effective_reservation_price: float
    preferred_flat_types: list[str] = Field(default_factory=list)
    preferred_towns: list[str] = Field(default_factory=list)
    latest_search_results: list[PortalSearchResult] = Field(default_factory=list)
    latest_market_feedback: str = 'No market feedback yet.'


class ListingSellerState(BaseModel):
    role: RoleType = Field(default=RoleType.SELLER)
    player_id: str
    player_name: str
    listed: bool
    current_listing_id: str | None = None
    current_listing_price: float | None = None
    open_requests: int = Field(ge=0)
    flat_type: str
    town: str


class ListingPortalSnapshot(BaseModel):
    week_number: int = Field(ge=0)
    buyers: list[ListingBuyerState] = Field(default_factory=list)
    sellers: list[ListingSellerState] = Field(default_factory=list)
    matched_pairs: list['NegotiationMatch'] = Field(default_factory=list)


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


ListingPortalSnapshot.model_rebuild()
