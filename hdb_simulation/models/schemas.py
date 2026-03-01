from enum import StrEnum
from pydantic import BaseModel, Field, RootModel
from typing import List, Literal, Optional, Annotated, Sequence, Union

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


class VerbalExplanationFields(ActionReasoningFields):
    verbal_explanation: str = Field(
        ...,
        description=(
            "Public-facing explanation safe to share with counterparties. "
            "Do not include hidden thresholds, private beliefs, or strategy internals."
        ),
    )


# Decisions (actions) that buyers and sellers can take during the negotiation process when no offer is on the table. 
class MakeOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['MAKE_OFFER'],
        Field(
            description=(
                "Start a new price proposal when there is no active offer on the table."
            )
        ),
    ]
    offer_price: int = Field(..., gt=0, description="The price proposed in the offer.")

class NormalAnswer(ActionReasoningFields):
    type: Annotated[
        Literal['NORMAL_ANSWER'],
        Field(
            description=(
                "Provide a factual/public answer without proposing a new price. Try to have a follow-up question or statement to keep the conversation going, unless you intend to walk away."
            )
        ),
    ]
    answer_details: str = Field(
        ...,
        description=(
            "Detailed public answer to counterpart's question. Keep aligned with "
            "verbal_explanation and avoid private information."
        ),
    )

# Buyer-specific actions when no offer is on the table
class BuyerInquiry(ActionReasoningFields):
    type: Annotated[
        Literal['INQUIRE_BUYER'],
        Field(
            description=(
                "Buyer asks exploratory questions to gather details of the flat without "
                "making a price proposal."
            )
        ),
    ]
    inquiry_details: str = Field(
        ...,
        description="Public inquiry details to counterpart. Must not include private information.",
    )

class BuyerQuestion(ActionReasoningFields):
    type: Annotated[
        Literal['QUESTION_BUYER'],
        Field(
            description=(
                "Buyer asks a direct question relevant to the counterpart, "
                "without making a price proposal."
            )
        ),
    ]
    question_details: str = Field(
        ...,
        description="Specific public question to counterpart regarding themselves.",
    )

# Seller-specific actions when no offer is on the table
class SellerInquiry(ActionReasoningFields):
    type: Annotated[
        Literal['INQUIRE_SELLER'],
        Field(
            description=(
                "Seller asks exploratory questions to gather details without "
                "making a price proposal."
            )
        ),
    ]
    inquiry_details: str = Field(
        ...,
        description="Public inquiry details to counterpart. Must not include private information.",
    )

# Decisions (actions) that can be taken by buyers and sellers when there is an active offer on the table.
class AcceptOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['ACCEPT_OFFER'],
        Field(
            description=(
                "Accept the currently active offer and finalize at the agreed price."
            )
        ),
    ]
    price_settled: int = Field(..., gt=0, description="The price at which the offer is accepted.")

class RejectOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['REJECT_OFFER'],
        Field(
            description=(
                "Reject the currently active offer without proposing a new price."
            )
        ),
    ]

class MakeCounteroffer(VerbalExplanationFields):
    type: Annotated[
        Literal['MAKE_COUNTEROFFER'],
        Field(
            description=(
                "Respond to the currently active offer with an alternative price. IT CANNOT be used to repeat the same offer price from the previous round. "
            )
        ),
    ]
    counteroffer_price: int = Field(..., gt=0, description="The price proposed in the counteroffer.")

class BuyerWalkAway(VerbalExplanationFields):
    type: Annotated[
        Literal['WALK_AWAY'],
        Field(
            description=(
                "Buyer explicitly ends the negotiation now without reaching agreement "
                "(close without success). Use only when intentionally terminating, "
                "not for questions/information gathering or continued negotiation."
            )
        ),
    ]


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
BUYER_TOTAL_ACTIONS = BUYER_OFFER_ACTIONS + BUYER_NON_OFFER_ACTIONS
SELLER_TOTAL_ACTIONS = SELLER_OFFER_ACTIONS + SELLER_NON_OFFER_ACTIONS

ACTION_TYPE_DESCRIPTIONS: dict[str, str] = {
    'MAKE_OFFER': 'Start a new price proposal when there is no active offer.',
    'INQUIRE_BUYER': 'Buyer asks exploratory questions without proposing a price.',
    'QUESTION_BUYER': 'Buyer asks a direct question without proposing a price.',
    'INQUIRE_SELLER': 'Seller asks exploratory questions without proposing a price.',
    'NORMAL_ANSWER': 'Provide a factual/public answer without proposing a new price.',
    'ACCEPT_OFFER': 'Accept the currently active offer and finalize at that price.',
    'REJECT_OFFER': 'Reject the currently active offer without proposing a new price.',
    'MAKE_COUNTEROFFER': 'Respond to an active offer with an alternative price.',
    'WALK_AWAY': 'Buyer explicitly ends the negotiation without agreement.',
}


def format_action_type_descriptions(action_types: Sequence[str]) -> str:
    """Format action-type descriptions for prompt injection."""
    lines = []
    for action_type in action_types:
        key = str(action_type).strip().upper()
        if not key:
            continue
        desc = ACTION_TYPE_DESCRIPTIONS.get(key, 'No description available.')
        lines.append(f'- {key}: {desc}')
    return '\n'.join(lines)


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


    
