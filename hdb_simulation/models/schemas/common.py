"""Shared reusable schemas for the HDB simulation."""

import copy
import dataclasses
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

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


@dataclasses.dataclass
class NormalDistribution:
    """Normal belief used for an agent's own reservation estimate."""

    name: str
    mean: float
    std: float
    confidence: float
    evidence_count: int = 0
    last_updated: Optional[str] = None

    @property
    def get_expected_mean(self) -> float:
        return self.mean

    @property
    def get_expected_variance(self) -> float:
        return self.std ** 2

    def sample(self, n: int = 1) -> float | list[float]:
        import numpy as np

        samples = np.random.normal(self.mean, self.std, n)
        return samples[0] if n == 1 else samples.tolist()

    def model_copy(self, *, deep: bool = False) -> 'NormalDistribution':
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def update_with_evidence(self, observation: float, reliability: float = 1.0) -> None:
        reliability = max(0.0, min(1.0, reliability))
        observation = max(0.0, observation)
        prior_precision = 1 / (self.std ** 2)
        evidence_precision = reliability / (self.std ** 2)
        total_precision = prior_precision + evidence_precision
        new_mean = (
            (prior_precision * self.mean + evidence_precision * observation)
            / total_precision
        )
        new_std = 1 / math.sqrt(total_precision)

        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.05 * reliability)
        self.mean = max(0.0, new_mean)
        self.std = new_std

    def get_confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        z_score = 1.96 if level == 0.95 else 2.58
        margin = z_score * self.std
        lower = max(0.0, self.mean - margin)
        upper = max(lower, self.mean + margin)
        return (lower, upper)


PREFERENCE_CATEGORY_LABELS: dict[str, str] = {
    'flat_type': 'Flat Type',
    'town': 'Town',
    'transport': 'Transport',
    'schools': 'Schools',
    'shopping': 'Shopping',
    'dining': 'Dining',
    'other': 'Other Priorities',
}


class BuyerPreferenceItem(BaseModel):
    category: Literal[
        'flat_type',
        'town',
        'transport',
        'schools',
        'shopping',
        'dining',
        'other',
    ]
    description: str = Field(min_length=1)


class BuyerPreferenceProfile(BaseModel):
    preferences: List[BuyerPreferenceItem] = Field(min_length=2)

    @model_validator(mode='after')
    def _validate_required_preferences(self) -> 'BuyerPreferenceProfile':
        if not self.values_for('flat_type'):
            raise ValueError('BuyerPreferenceProfile must include at least one flat_type preference.')
        if not self.values_for('town'):
            raise ValueError('BuyerPreferenceProfile must include at least one town preference.')
        return self

    def values_for(self, category: str) -> list[str]:
        return [
            preference.description.strip()
            for preference in self.preferences
            if preference.category == category and preference.description.strip()
        ]

    def grouped_preferences(self) -> list[tuple[str, list[str]]]:
        grouped: list[tuple[str, list[str]]] = []
        for category, label in PREFERENCE_CATEGORY_LABELS.items():
            values = self.values_for(category)
            if values:
                grouped.append((label, values))
        return grouped

    def feature_summary(self) -> str:
        return '; '.join(
            f'{label}: {", ".join(values)}'
            for label, values in self.grouped_preferences()
            if label not in {'Flat Type', 'Town'}
        )


def coerce_buyer_preferences(
    preferences: BuyerPreferenceProfile | Mapping[str, Any] | None,
) -> BuyerPreferenceProfile | None:
    if isinstance(preferences, BuyerPreferenceProfile):
        return preferences
    if isinstance(preferences, Mapping):
        return BuyerPreferenceProfile.model_validate(dict(preferences))
    return None


def summarize_buyer_features(
    preferences: BuyerPreferenceProfile | Mapping[str, Any] | None,
) -> str:
    profile = coerce_buyer_preferences(preferences)
    return profile.feature_summary() if profile is not None else ''


def format_buyer_preferences(
    preferences: BuyerPreferenceProfile | Mapping[str, Any] | None,
) -> list[str]:
    profile = coerce_buyer_preferences(preferences)
    if profile is None:
        return []
    return [
        f'- {label}: {", ".join(values)}'
        for label, values in profile.grouped_preferences()
    ]

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



