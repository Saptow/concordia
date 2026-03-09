"""Listing-specific schemas for the HDB simulation."""

from collections.abc import Sequence
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, RootModel

from concordia.hdb_simulation.models.schemas.common import (
    BaseBuyer,
    BaseSeller,
    BuyerBudgetRange,
    BuyerPreferenceProfile,
    Flat,
    RoleType,
    SellerExpectationRange,
)


class SearchAndRequestNegotiation(BaseModel):
  type: Annotated[
      Literal['SEARCH_AND_REQUEST_NEGOTIATION'],
      Field(
          description=(
              'Search the centralized listing portal, update your market view, '
              'and request negotiations with suitable listings.'
          )
      ),
  ]
  search_query: str = Field(..., description='Natural-language query sent to the listing portal.')
  requested_listing_ids: list[str] = Field(
      default_factory=list,
      description='Listing IDs that the buyer wants to request negotiations with this week.',
  )
  market_valuation_notes: str = Field(..., description='Summary of the market insight learned this week.')


class ListFlatOnPortal(BaseModel):
  type: Annotated[
      Literal['LIST_FLAT_ON_PORTAL'],
      Field(description="Publish the seller's flat on the centralized portal."),
  ]
  listing_price: int = Field(..., gt=0, description='Initial asking price for the flat in SGD.')
  listing_summary: str = Field(..., description='Short listing summary to accompany the flat metadata.')


class ReviewRequestsAndStartNegotiation(BaseModel):
  type: Annotated[
      Literal['REVIEW_REQUESTS_AND_START_NEGOTIATION'],
      Field(
          description=(
              'Review all requests submitted to the flat so far. If any exist, '
              'the portal will uniformly sample one request to begin negotiation.'
          )
      ),
  ]
  seller_response: str = Field(..., description='Short note explaining the seller decision for this week.')


ListingBuyerAction = Annotated[
    SearchAndRequestNegotiation,
    Field(discriminator='type'),
]
ListingSellerUnlistedAction = Annotated[
    ListFlatOnPortal,
    Field(discriminator='type'),
]
ListingSellerListedAction = Annotated[
    ReviewRequestsAndStartNegotiation,
    Field(discriminator='type'),
]


class ListingBuyerActions(RootModel[ListingBuyerAction]):
  pass


class ListingSellerUnlistedActions(RootModel[ListingSellerUnlistedAction]):
  pass


class ListingSellerListedActions(RootModel[ListingSellerListedAction]):
  pass


LISTING_BUYER_ACTIONS = ('SEARCH_AND_REQUEST_NEGOTIATION',)
LISTING_SELLER_UNLISTED_ACTIONS = ('LIST_FLAT_ON_PORTAL',)
LISTING_SELLER_LISTED_ACTIONS = ('REVIEW_REQUESTS_AND_START_NEGOTIATION',)

LISTING_ACTION_TYPE_DESCRIPTIONS: dict[str, str] = {
    'SEARCH_AND_REQUEST_NEGOTIATION': (
        'Search the portal, learn the market valuation of matching flats, and '
        'request negotiations with suitable listings.'
    ),
    'LIST_FLAT_ON_PORTAL': 'Publish the seller flat on the centralized portal.',
    'REVIEW_REQUESTS_AND_START_NEGOTIATION': (
        'Review all accumulated requests on the seller listing and move one '
        'uniformly sampled request into negotiation when available.'
    ),
}


def format_action_type_descriptions(action_types: Sequence[str]) -> str:
  lines = []
  for action_type in action_types:
    key = str(action_type).strip().upper()
    if not key:
      continue
    description = LISTING_ACTION_TYPE_DESCRIPTIONS.get(
        key, 'No description available.'
    )
    lines.append(f'- {key}: {description}')
  return '\n'.join(lines)


def get_action_model(
    role: RoleType,
    seller_is_listed: bool | None = None,
) -> type[RootModel]:
  if role == RoleType.BUYER:
    return ListingBuyerActions
  if role == RoleType.SELLER:
    return (
        ListingSellerListedActions
        if seller_is_listed
        else ListingSellerUnlistedActions
    )
  raise ValueError(f'No portal action model for role: {role}')


def get_allowed_action_types(
    role: RoleType,
    seller_is_listed: bool | None = None,
) -> tuple[str, ...]:
  if role == RoleType.BUYER:
    return LISTING_BUYER_ACTIONS
  if role == RoleType.SELLER:
    return (
        LISTING_SELLER_LISTED_ACTIONS
        if seller_is_listed
        else LISTING_SELLER_UNLISTED_ACTIONS
    )
  raise ValueError(f'No portal action set for role: {role}')


class PortalBuyer(BaseBuyer):
  budget: BuyerBudgetRange
  preferences: BuyerPreferenceProfile
  description: Optional[str] = None


class PortalSeller(BaseSeller):
  flat: Flat
  expectations: SellerExpectationRange
  description: Optional[str] = None


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


class PortalNotification(BaseModel):
  player_name: str
  message: str


class ListingWeeklyBatchOutcome(BaseModel):
  week_number: int = Field(ge=1)
  active_player_names: list[str] = Field(default_factory=list)
  newly_listed_listing_ids: list[str] = Field(default_factory=list)
  buyers_processed: list[str] = Field(default_factory=list)
  sellers_reviewed: list[str] = Field(default_factory=list)
  matched_pairs: list[NegotiationMatch] = Field(default_factory=list)
  closed_player_names: list[str] = Field(default_factory=list)
  notifications: list[PortalNotification] = Field(default_factory=list)


ListingPortalSnapshot.model_rebuild()
