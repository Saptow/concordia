"""Shared reusable schemas for the HDB simulation."""

from enum import StrEnum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Structured Action Schemas (shared)
class ActionReasoningFields(BaseModel):
    internal_reasoning: str = Field(
        ...,
        description=(
            'Private chain-of-thought style rationale for internal debugging/'
            'tracing only. Must never be shown to counterparties.'
        ),
    )


class VerbalExplanationFields(ActionReasoningFields):
    verbal_explanation: str = Field(
        ...,
        description=(
            'Public-facing explanation safe to share with counterparties. '
            'Do not include hidden thresholds, private beliefs, or strategy internals.'
        ),
    )


class ActionChoiceWithRationale(BaseModel):
    chosen_action_type: str = Field(
        ...,
        description=(
            'The single action type selected for this turn. Must match exactly one '
            'allowed action type from the prompt.'
        ),
    )
    decision_rationale: str = Field(
        ...,
        description=(
            'Concise private summary of why this action type was chosen over nearby '
            'alternatives for this turn.'
        ),
    )


# General Object Schemas
class RoleType(StrEnum):
    BUYER = 'buyer'
    SELLER = 'seller'
    PLACEHOLDER = 'placeholder'


class FlatType(StrEnum):
    ONE_ROOM = '1-Room'
    TWO_ROOM = '2-Room'
    THREE_ROOM = '3-Room'
    FOUR_ROOM = '4-Room'
    FIVE_ROOM = '5-Room'
    EXECUTIVE = 'Executive'


class AmenityType(StrEnum):
    MRT = "MRT"
    SCHOOL = "Primary School"
    HAWKER = "Hawker Centre"
    MALL = "Shopping Mall"

class Amenity(BaseModel):
    name: str
    type: AmenityType
    radius: Literal['Within 1km', 'Within 2km']

class PriceTrend(BaseModel):
    transactions_6m: int
    min_price_6m: float
    max_price_6m: float

class Flat(BaseModel):
    flat_type: FlatType
    address: str
    description: str
    town: str
    storey_range: str
    remaining_lease: float
    contra: bool
    extension_of_stay: bool
    minimum_occupancy_period_completed: bool = True
    minimum_occupancy_period_years: Optional[float] = None
    upgrading: Optional[List[str]] = None
    ethnic_eligibility: str
    spr_eligibility: str
    floor_area_sqm: float
    nearby_amenities: Optional[List[Amenity]] = None
    past_price_trends: Optional[PriceTrend] = None
    

    def to_compact_description(self) -> str:
        details = [str(self.flat_type), f'in {self.town}']
        if self.address:
            details.append(f'at {self.address}')
        if self.storey_range:
            details.append(f'storey range {self.storey_range}')
        details.append(f'about {self.floor_area_sqm:.0f} sqm')
        details.append(f'remaining lease {self.remaining_lease:g} years')
        return ' '.join(details)


    
class BuyerBudgetRange(BaseModel):
    min_price: float = Field(ge=0.0)
    max_price: float = Field(ge=0.0)


class SellerExpectationRange(BaseModel):
    min_price: float = Field(ge=0.0)
    max_price: float = Field(ge=0.0)


class BuyerPreferenceProfile(BaseModel):
    flat_type: List[str] = Field(default_factory=list)
    towns: List[str] = Field(default_factory=list)
    features: str = ''

class BaseBuyer(BaseModel):
    id: str
    name: str
    role: RoleType = Field(default=RoleType.BUYER, description='Role of the entity.')
    description: Optional[str] = Field(default=None, description='Optional free-text description of the buyer.')
    # Buyer-specific
    preferences: BuyerPreferenceProfile
    budget: BuyerBudgetRange



class BaseSeller(BaseModel):
    id: str
    name: str
    role: RoleType = Field(default=RoleType.SELLER, description='Role of the entity.')
    description: Optional[str] = Field(default=None, description='Optional free-text description of the seller.')

    # Seller-specific
    flat: Flat
    expectations: SellerExpectationRange


class OfferHistory(BaseModel):
    offer_price: int = Field(..., gt=0, description='The price proposed in this offer.')
    offer_week: int = Field(..., ge=0, description='The week number when this offer was made.')
    offer_turn: int = Field(..., ge=0, description='The turn number from the start of the negotiation when this offer was made.')
    offerer_role: RoleType = Field(..., description='The role (buyer or seller) of the party that made this offer.')


class NegotiationHistoryRecord(BaseModel):
    buyer_id: str
    seller_id: str
    start_week: int = Field(..., ge=0, description='The week number when the negotiation started.')
    end_week: Optional[int] = Field(default=None, ge=0, description='The week number when the negotiation ended. Null if still ongoing.')
    offer_history: list[OfferHistory] = Field(default_factory=list)


class NegotiationOutcome(StrEnum):
    SUCCESS = 'SUCCESS'
    CLOSED = 'CLOSED'
    CLOSED_WITHOUT_SUCCESS = 'CLOSED_WITHOUT_SUCCESS'



