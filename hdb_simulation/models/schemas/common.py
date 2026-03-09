"""Shared reusable schemas for the HDB simulation."""

from enum import StrEnum
from typing import List, Optional

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

class BaseBuyer(BaseModel):
  id: str
  name: str
  role: RoleType = Field(default=RoleType.BUYER, description='Role of the entity.')


class BaseSeller(BaseModel):
  id: str
  name: str
  role: RoleType = Field(default=RoleType.SELLER, description='Role of the entity.')


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
  nearby_amenities: Optional[List[str]] = None


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
