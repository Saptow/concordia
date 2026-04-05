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

PREFERENCE_PRIOR_DIMENSIONS: tuple[str, ...] = (
    'intercept',
    'flat_type_match',
    'town_match',
    'transport_access',
    'school_access',
    'shopping_access',
    'dining_access',
    'other_match',
)


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
    strength: float = Field(ge=0.0, le=1.0, description='Relative importance of this preference, between 0 and 1, with 0 indicating no importance and 1 indicating maximum importance.')


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

    def items_for(self, category: str) -> list[BuyerPreferenceItem]:
        return [
            preference
            for preference in self.preferences
            if preference.category == category and preference.description.strip()
        ]

    def strongest_strength_for(self, category: str) -> float:
        items = self.items_for(category)
        if not items:
            return 0.0
        return max(float(item.strength) for item in items)

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


class BuyerPreferencePrior(BaseModel):
    dimension_names: List[str] = Field(
        default_factory=lambda: list(PREFERENCE_PRIOR_DIMENSIONS)
    )
    mean_vector: List[float]
    covariance_matrix: List[List[float]]

    @model_validator(mode='after')
    def _validate_shape(self) -> 'BuyerPreferencePrior':
        size = len(self.dimension_names)
        if len(self.mean_vector) != size:
            raise ValueError('BuyerPreferencePrior mean_vector must match dimension_names.')
        if len(self.covariance_matrix) != size:
            raise ValueError('BuyerPreferencePrior covariance_matrix must match dimension_names.')
        if any(len(row) != size for row in self.covariance_matrix):
            raise ValueError('BuyerPreferencePrior covariance_matrix must be square.')
        return self

    def index_of(self, dimension_name: str) -> int:
        return self.dimension_names.index(dimension_name)

    def project(self, attribute_vector: List[float]) -> tuple[float, float]:
        if len(attribute_vector) != len(self.mean_vector):
            raise ValueError('Attribute vector must match preference-prior dimensions.')
        projected_mean = sum(
            value * weight
            for value, weight in zip(attribute_vector, self.mean_vector, strict=True)
        )
        projected_variance = 0.0
        for row_index, row in enumerate(self.covariance_matrix):
            for col_index, covariance in enumerate(row):
                projected_variance += (
                    attribute_vector[row_index]
                    * covariance
                    * attribute_vector[col_index]
                )
        return (projected_mean, max(1e-9, projected_variance))

    def update_intercept(self, mean: float, variance: float) -> None:
        index = self.index_of('intercept')
        self.mean_vector[index] = max(0.0, float(mean))
        self.covariance_matrix[index][index] = max(1e-9, float(variance))

    def update_from_pseudo_observation(
        self,
        attribute_vector: List[float],
        target_value: float,
        observation_variance: float,
    ) -> None:
        """Bayesian linear update from a pseudo-observed flat valuation target."""
        if len(attribute_vector) != len(self.mean_vector):
            raise ValueError('Attribute vector must match preference-prior dimensions.')

        observation_variance = max(1e-9, float(observation_variance))
        # Sigma x term used in the Bayesian linear-regression / Kalman-style gain.
        covariance_times_attribute = [
            sum(
                covariance * attribute_value
                for covariance, attribute_value in zip(
                    row,
                    attribute_vector,
                    strict=True,
                )
            )
            for row in self.covariance_matrix
        ]
        predicted_mean, predicted_variance = self.project(attribute_vector)
        # x' Sigma x + sigma_obs^2
        innovation_variance = max(1e-9, predicted_variance + observation_variance)
        # K = Sigma x / (x' Sigma x + sigma_obs^2)
        kalman_gain = [
            covariance_value / innovation_variance
            for covariance_value in covariance_times_attribute
        ]
        # y_fail - x' mu
        residual = max(0.0, float(target_value)) - predicted_mean

        # mu_new = mu + K (y_fail - x' mu)
        self.mean_vector = [
            float(mean + (gain * residual))
            for mean, gain in zip(self.mean_vector, kalman_gain, strict=True)
        ]

        # Sigma_new = Sigma - K x' Sigma
        updated_covariance = [
            [
                float(covariance - (kalman_gain[row_index] * covariance_times_attribute[col_index]))
                for col_index, covariance in enumerate(row)
            ]
            for row_index, row in enumerate(self.covariance_matrix)
        ]

        size = len(updated_covariance)
        for row_index in range(size):
            updated_covariance[row_index][row_index] = max(
                1e-9,
                updated_covariance[row_index][row_index],
            )
            for col_index in range(row_index + 1, size):
                symmetric_value = 0.5 * (
                    updated_covariance[row_index][col_index]
                    + updated_covariance[col_index][row_index]
                )
                updated_covariance[row_index][col_index] = symmetric_value
                updated_covariance[col_index][row_index] = symmetric_value

        intercept_index = self.index_of('intercept')
        self.mean_vector[intercept_index] = max(
            0.0,
            float(self.mean_vector[intercept_index]),
        )
        self.covariance_matrix = updated_covariance


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
    lines: list[str] = []
    for category, label in PREFERENCE_CATEGORY_LABELS.items():
        items = profile.items_for(category)
        if not items:
            continue
        formatted_values = [
            f'{item.description.strip()} ({float(item.strength):.2f})'
            for item in items
            if item.description.strip()
        ]
        if formatted_values:
            lines.append(f'- {label}: {", ".join(formatted_values)}')
    return lines


def _budget_bounds(
    budget: BuyerBudgetRange | Mapping[str, Any] | None,
) -> tuple[float, float]:
    if isinstance(budget, BuyerBudgetRange):
        return (float(budget.min_price), float(budget.max_price))
    if isinstance(budget, Mapping):
        return (
            max(0.0, float(budget.get('min_price', 0.0))),
            max(0.0, float(budget.get('max_price', 0.0))),
        )
    return (0.0, 0.0)


def build_buyer_preference_prior(
    preferences: BuyerPreferenceProfile | Mapping[str, Any] | None,
    budget: BuyerBudgetRange | Mapping[str, Any] | None,
) -> BuyerPreferencePrior:
    profile = coerce_buyer_preferences(preferences)
    budget_min, budget_max = _budget_bounds(budget)
    budget_midpoint = max(0.0, 0.5 * (budget_min + budget_max))
    budget_span = max(0.0, budget_max - budget_min)
    base_std = max(25000.0, 0.5 * budget_span, 0.04 * budget_midpoint)

    def _strength(category: str) -> float:
        return profile.strongest_strength_for(category) if profile is not None else 0.0

    means = [
        budget_midpoint,
        0.02 * budget_midpoint * _strength('flat_type'),
        0.015 * budget_midpoint * _strength('town'),
        0.008 * budget_midpoint * _strength('transport'),
        0.007 * budget_midpoint * _strength('schools'),
        0.006 * budget_midpoint * _strength('shopping'),
        0.006 * budget_midpoint * _strength('dining'),
        0.006 * budget_midpoint * _strength('other'),
    ]
    stds = [
        base_std,
        0.45 * base_std,
        0.40 * base_std,
        0.25 * base_std,
        0.25 * base_std,
        0.20 * base_std,
        0.20 * base_std,
        0.30 * base_std,
    ]
    size = len(PREFERENCE_PRIOR_DIMENSIONS)
    covariance = [[0.0 for _ in range(size)] for _ in range(size)]
    for index, std in enumerate(stds):
        covariance[index][index] = float(std ** 2)

    def _set_covariance(left: str, right: str, correlation: float) -> None:
        left_index = PREFERENCE_PRIOR_DIMENSIONS.index(left)
        right_index = PREFERENCE_PRIOR_DIMENSIONS.index(right)
        covariance_value = correlation * stds[left_index] * stds[right_index]
        covariance[left_index][right_index] = covariance_value
        covariance[right_index][left_index] = covariance_value

    _set_covariance('transport_access', 'shopping_access', 0.20)
    _set_covariance('transport_access', 'dining_access', 0.15)
    _set_covariance('shopping_access', 'dining_access', 0.15)
    _set_covariance('town_match', 'school_access', 0.10)

    return BuyerPreferencePrior(
        mean_vector=[float(value) for value in means],
        covariance_matrix=covariance,
    )


def build_buyer_flat_attribute_vector(
    preferences: BuyerPreferenceProfile | Mapping[str, Any] | None,
    flat_payload: Flat | Mapping[str, Any],
) -> List[float]:
    profile = coerce_buyer_preferences(preferences)
    payload = (
        flat_payload.model_dump(mode='python')
        if isinstance(flat_payload, Flat)
        else dict(flat_payload)
    )
    nearby_amenities = payload.get('nearby_amenities') or []
    amenity_types = {
        str(item.type) if isinstance(item, Amenity) else str(item.get('type', ''))
        for item in nearby_amenities
    }

    flat_description = ' '.join(
        str(payload.get(field, ''))
        for field in ('description', 'address', 'storey_range')
    ).casefold()
    other_match = 0.0
    if profile is not None:
        for description in profile.values_for('other'):
            tokens = [
                token for token in description.casefold().replace('/', ' ').split()
                if len(token) > 3
            ]
            if tokens and any(token in flat_description for token in tokens):
                other_match = 1.0
                break

    return [
        1.0,
        1.0
        if profile is not None
        and str(payload.get('flat_type', '')).strip() in profile.values_for('flat_type')
        else 0.0,
        1.0
        if profile is not None
        and str(payload.get('town', '')).strip() in profile.values_for('town')
        else 0.0,
        1.0 if str(AmenityType.MRT) in amenity_types else 0.0,
        1.0 if str(AmenityType.SCHOOL) in amenity_types else 0.0,
        1.0 if str(AmenityType.MALL) in amenity_types else 0.0,
        1.0 if str(AmenityType.HAWKER) in amenity_types else 0.0,
        other_match,
    ]

class BaseBuyer(BaseModel):
    id: str
    name: str
    role: RoleType = Field(default=RoleType.BUYER, description='Role of the entity.')
    description: Optional[str] = Field(default=None, description='Optional free-text description of the buyer.')
    # Buyer-specific
    preferences: BuyerPreferenceProfile
    budget: BuyerBudgetRange
    reservation_price_prior: Optional[float] = Field(
        default=None,
        ge=0.0,
        description='Initial private reservation-price prior before market feedback.',
    )



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
