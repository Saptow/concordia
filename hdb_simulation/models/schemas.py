from enum import StrEnum
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Optional, Annotated, Union

# Enums
class RoleType(StrEnum):
    BUYER = "buyer"
    SELLER = "seller"
    PLACEHOLDER = "placeholder" # just a dummy

class FlatType(StrEnum):
    ONE_ROOM = "1-Room"
    TWO_ROOM = "2-Room"
    THREE_ROOM = "3-Room"
    FOUR_ROOM = "4-Room"
    FIVE_ROOM = "5-Room"
    EXECUTIVE = "Executive"



# Data Models
# Shared reasoning fields across all executable actions.
class ActionReasoningFields(BaseModel):
    internal_reasoning: str = Field(
        ...,
        description=(
            "Private chain-of-thought style rationale for internal debugging/"
            "tracing only. Must never be shown to counterparties."
        ),
    )


# Decisions (actions) that buyers and sellers can take during the negotiation process when no offer is on the table. 
class MakeOffer(ActionReasoningFields):
    type: Literal['MAKE_OFFER']
    offer_price: float = Field(..., gt=0, description="The price proposed in the offer.")
    verbal_explanation: str = Field(
        ...,
        validation_alias='explanation',
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )

class NormalAnswer(ActionReasoningFields):
    type: Literal['NORMAL_ANSWER']
    answer_details: str = Field(..., 
                                validation_alias="explanation",
                                description="Detailed public answer to counterpart's question. Keep aligned with verbal_explanation and avoid private information.")

# Buyer-specific actions when no offer is on the table
class BuyerInquiry(ActionReasoningFields):
    type: Literal['INQUIRE_BUYER']
    inquiry_details: str = Field(..., 
                                 validation_alias="explanation",
                                 description="Public inquiry details to counterpart. Must not include private information.")

class BuyerQuestion(ActionReasoningFields):
    type: Literal['QUESTION_BUYER']
    question_details: str = Field(..., validation_alias="explanation",
                                  description="Specific public question to counterpart regarding negotiation context.")

# Seller-specific actions when no offer is on the table
class SellerInquiry(ActionReasoningFields):
    type: Literal['INQUIRE_SELLER']
    inquiry_details: str = Field(..., validation_alias="explanation",
                                 description="Public inquiry details to counterpart. Must not include private information.")

# Decisions (actions) that can be taken by buyers and sellers when there is an active offer on the table.
class AcceptOffer(ActionReasoningFields):
    type: Literal['ACCEPT_OFFER']
    price_settled: float = Field(..., gt=0, description="The price at which the offer is accepted.")
    verbal_explanation: str = Field(
        ...,
        validation_alias='explanation',
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )

class RejectOffer(ActionReasoningFields):
    type: Literal['REJECT_OFFER']
    verbal_explanation: str = Field(
        ...,
        validation_alias='explanation',
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )

class MakeCounteroffer(ActionReasoningFields):
    type: Literal['MAKE_COUNTEROFFER']
    counteroffer_price: float = Field(..., gt=0, description="The price proposed in the counteroffer.")
    verbal_explanation: str = Field(
        ...,
        validation_alias='explanation',
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )

class BuyerWalkAway(ActionReasoningFields):
    type: Literal['WALK_AWAY']
    verbal_explanation: str = Field(
        ...,
        validation_alias='explanation',
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )


# Union types for buyer and seller actions with discriminators for parsing
BuyerNonOfferActionTypes=Annotated[
    Union[MakeOffer,BuyerInquiry, BuyerQuestion, NormalAnswer, BuyerWalkAway],
    Field(discriminator='type')
]
BuyerOfferActionTypes=Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer, BuyerWalkAway],
    Field(discriminator='type')
]
class BuyerNonOfferActions(RootModel[BuyerNonOfferActionTypes]):
    pass
class BuyerOfferActions(RootModel[BuyerOfferActionTypes]):
    pass

SellerNonOfferActionTypes=Annotated[
    Union[MakeOffer,SellerInquiry, NormalAnswer],
    Field(discriminator='type')
]
SellerOfferActionTypes=Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer],
    Field(discriminator='type')
]
class SellerNonOfferActions(RootModel[SellerNonOfferActionTypes]):
    pass
class SellerOfferActions(RootModel[SellerOfferActionTypes]):
    pass

# Backward-compatible "all actions" schemas.
# These are intentionally broad and can be narrowed dynamically at runtime
# using the helper selectors below.
BuyerActionTypes = Annotated[
    Union[
        MakeOffer,
        BuyerInquiry,
        BuyerQuestion,
        NormalAnswer,
        AcceptOffer,
        RejectOffer,
        MakeCounteroffer,
        BuyerWalkAway,
    ],
    Field(discriminator='type')
]

SellerActionTypes = Annotated[
    Union[
        MakeOffer,
        SellerInquiry,
        NormalAnswer,
        AcceptOffer,
        RejectOffer,
        MakeCounteroffer,
    ],
    Field(discriminator='type')
]

class BuyerActions(RootModel[BuyerActionTypes]):
    pass

class SellerActions(RootModel[SellerActionTypes]):
    pass


BUYER_NON_OFFER_ACTIONS = (
    'MAKE_OFFER',
    'INQUIRE_BUYER',
    'QUESTION_BUYER',
    'NORMAL_ANSWER',
    'WALK_AWAY',
)
BUYER_OFFER_ACTIONS = (
    'ACCEPT_OFFER',
    'REJECT_OFFER',
    'MAKE_COUNTEROFFER',
    'WALK_AWAY',
)
SELLER_NON_OFFER_ACTIONS = (
    'MAKE_OFFER',
    'INQUIRE_SELLER',
    'NORMAL_ANSWER',
)
SELLER_OFFER_ACTIONS = (
    'ACCEPT_OFFER',
    'REJECT_OFFER',
    'MAKE_COUNTEROFFER',
)


def get_action_model(role: RoleType, has_active_offer: bool) -> type[RootModel]:
    """Return the role-constrained action model for current offer state."""
    if role == RoleType.BUYER:
        return BuyerOfferActions if has_active_offer else BuyerNonOfferActions
    if role == RoleType.SELLER:
        return SellerOfferActions if has_active_offer else SellerNonOfferActions
    raise ValueError(f'No action model for role: {role}')


def get_allowed_action_types(role: RoleType, has_active_offer: bool) -> tuple[str, ...]:
    """Return allowed action type literals for prompt/policy use."""
    if role == RoleType.BUYER:
        return BUYER_OFFER_ACTIONS if has_active_offer else BUYER_NON_OFFER_ACTIONS
    if role == RoleType.SELLER:
        return SELLER_OFFER_ACTIONS if has_active_offer else SELLER_NON_OFFER_ACTIONS
    raise ValueError(f'No action set for role: {role}')

class BaseBuyer(BaseModel):
    id: str
    name: str
    role: RoleType = Field(default=RoleType.BUYER, description="Role of the entity.")

class BaseSeller(BaseModel):
    id: str
    name: str
    role: RoleType = Field(default=RoleType.SELLER, description="Role of the entity.")

    
class Flat(BaseModel):
    # TODO: revise flat attributes should more data be given 
    # Basic Attributes
    flat_type: FlatType
    address: str 
    description: str

    # Strict Metadata
    town: str
    storey_range: str # might change to range later
    remaining_lease: float # in years
    contra: bool
    extension_of_stay: bool
    upgrading: Optional[List[str]] = None
    ethnic_eligibility: str # Format: "<race> (Month Year)"
    spr_eligibility: str # Format: "<bool> (Month Year)"
    floor_area_sqm: float

    # Other optional descriptions
    nearby_amenities: Optional[List[str]] = None


    
