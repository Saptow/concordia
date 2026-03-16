"""Listing-specific schemas for the HDB simulation."""

from concordia.hdb_simulation.models.schemas.listing.qdrant import ListingRecord
from pydantic import BaseModel, Field

from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.hdb_simulation.models.schemas.common import (
    BaseBuyer,
    BaseSeller,
    RoleType,
)

# TODO: This is the entry schema that initialises portal state for buyers/sellers that joins from negotiation/get initialised. 
class PortalBuyer(BaseBuyer):
    negotiation_history: list[str] = Field(default_factory=list) # store ids of past sellers that negotiated but failed to go through with, to avoid rematching with them in the future.
    pass


class PortalSeller(BaseSeller):
    negotiation_history: list[str] = Field(default_factory=list) # store ids of past buyers that negotiated but failed to go through with, to avoid rematching with them in the future.
    pass


class PortalSearchResult(ListingRecord):
    score: float = Field(ge=0.0)
    

class BuyerMarketBeliefState(BaseModel):
    buyer_id: str
    base_reservation_price: float = Field(ge=0.0)
    effective_reservation: uncertain_helper.NormalDistribution
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


class ListingBuyerState(PortalBuyer):
    effective_reservation: uncertain_helper.NormalDistribution
    latest_search_results: list[PortalSearchResult] = Field(default_factory=list)
    latest_market_feedback: str = 'No market feedback yet.'

class ListingSellerState(PortalSeller):
    listed: bool
    current_listing_id: str | None = None
    current_listing_price: float | None = None
    open_requests: int = Field(ge=0)


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


class ListingNegotiationTransferPayload(BaseModel):
    match_id: str
    week_matched: int = Field(ge=1)
    listing_record: ListingRecord
    buyer_state: ListingBuyerState
    seller_state: ListingSellerState

class ListingWeeklyBatchOutcome(BaseModel):
    week_number: int = Field(ge=1)
    active_player_names: list[str] = Field(default_factory=list)
    newly_listed_listing_ids: list[str] = Field(default_factory=list)
    buyers_processed: list[str] = Field(default_factory=list)
    sellers_reviewed: list[str] = Field(default_factory=list)
    matched_pairs: list[NegotiationMatch] = Field(default_factory=list)
    closed_player_names: list[str] = Field(default_factory=list)


ListingPortalSnapshot.model_rebuild()
