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
# Decisions (actions) that buyers and sellers can take during the negotiation process when no offer is on the table. 
class MakeOffer(BaseModel):
    type: Literal['MAKE_OFFER']
    offer_price: float = Field(..., gt=0, description="The price proposed in the offer.")
    reasoning: Optional[str] = Field(None, description="Optional reasoning behind the offer. This reasoning SHOULD NOT include private justifications, but are optional texts that the agent can use to explain the offer to the other party, e.g. 'Based on recent transactions in the area, I believe this is a fair offer.'")

class NormalAnswer(BaseModel):
    type: Literal['NORMAL_ANSWER']
    answer_details: str = Field(..., description="Details of the buyer's/seller's answer to inquiries or questions. Note that these details SHOULD NOT include private justifications, but are optional texts that the agent can use to explain their answer to the other party, e.g. 'I am asking this question because I want to understand your flexibility on the price.'")

# Buyer-specific actions when no offer is on the table
class BuyerInquiry(BaseModel):
    type: Literal['INQUIRE_BUYER']
    inquiry_details: str = Field(..., description="Details of the buyer's inquiry about the flat's conditions/details. Note that these details SHOULD NOT include private justifications, but are optional texts that the buyer can use to explain their inquiry to the seller, e.g. 'I am asking about the flat's conditions because I want to understand if it is worth the price you are asking.'")

class BuyerQuestion(BaseModel):
    type: Literal['QUESTION_BUYER']
    question_details: str = Field(..., description="Specific question posed by the buyer regarding flexibility and urgency of the negotiation. Note that these details SHOULD NOT include private justifications, but are optional texts that the buyer can use to explain their question to the seller, e.g. 'I am asking about your urgency because I want to understand if you are willing to negotiate on the price.'")

# Seller-specific actions when no offer is on the table
class SellerInquiry(BaseModel):
    type: Literal['INQUIRE_SELLER']
    inquiry_details: str = Field(..., description="Details of the seller's inquiry about the buyer's preferences or constraints. Note that these details SHOULD NOT include private justifications, but are optional texts that the seller can use to explain their inquiry to the buyer, e.g. 'I am asking about your preferences because I want to understand what is important to you in this negotiation.'")

# Decisions (actions) that can be taken by buyers and sellers when there is an active offer on the table.
class AcceptOffer(BaseModel):
    type: Literal['ACCEPT_OFFER']
    price_settled: float = Field(..., gt=0, description="The price at which the offer is accepted.")
    reasoning: Optional[str] = Field(None, description="Optional reasoning behind the acceptance of the offer. This reasoning SHOULD NOT include private justifications, but are optional texts that the agent can use to explain the acceptance to the other party, e.g. 'I accept this offer because it is fair and reasonable.'  ")

class RejectOffer(BaseModel):
    type: Literal['REJECT_OFFER']
    reasoning: Optional[str] = Field(None, description="Optional reasoning behind the rejection of the offer. This reasoning SHOULD NOT include private justifications, but are optional texts that the agent can use to explain the rejection to the other party, e.g. 'I reject this offer because it is too low compared to recent transactions in the area.'")

class MakeCounteroffer(BaseModel):
    type: Literal['MAKE_COUNTEROFFER']
    counteroffer_price: float = Field(..., gt=0, description="The price proposed in the counteroffer.")
    reasoning: Optional[str] = Field(None, description="Optional reasoning behind the counteroffer. This reasoning SHOULD NOT include private justifications, but are optional texts that the agent can use to explain the counteroffer to the other party, e.g. 'I make this counteroffer because it is more in line with recent transactions in the area.'")


# Union types for buyer and seller actions with discriminators for parsing
BuyerNonOfferActionTypes=Annotated[
    Union[MakeOffer,BuyerInquiry, BuyerQuestion, NormalAnswer],
    Field(discriminator='type')
]
BuyerOfferActionTypes=Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer],
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
)
BUYER_OFFER_ACTIONS = (
    'ACCEPT_OFFER',
    'REJECT_OFFER',
    'MAKE_COUNTEROFFER',
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


    


