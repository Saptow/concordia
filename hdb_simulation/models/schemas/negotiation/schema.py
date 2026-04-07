"""Negotiation-specific schemas for the HDB simulation."""

from collections.abc import Sequence
from typing import Annotated, Literal, Union, Any, Optional

from concordia.hdb_simulation.models.schemas.listing.qdrant import ListingRecord
from concordia.hdb_simulation.models.schemas.listing.schema import (
    NegotiationMatch,
    PortalBuyer,
    PortalSearchResult,
    PortalSeller,
)
from pydantic import Field, RootModel, BaseModel 

from concordia.hdb_simulation.models.schemas.common import (
    ActionReasoningFields,
    BaseBuyer,
    BaseSeller,
    NegotiationHistoryRecord,
    NegotiationOutcome,
    NormalDistribution,
    OfferHistory,
    RoleType,
    VerbalExplanationFields,
)

# Negotiation Entity Schemas
class NegotiationBuyer(BaseBuyer):
    negotiation_config: Optional[dict[str, Any]]= None
'''
Example schema for negotiation_config is:
- own_reservation_: int = Field(..., gt=0, description='The buyer\'s own reservation price for the property.')
- own_reservation_std: int = Field(..., ge=0, description='The standard deviation representing uncertainty in the buyer\'s reservation price.')
- cp_reservation_: int = Field(..., gt=0, description='The buyer\'s estimate of the seller\'s reservation price for the property.')
- lambda_: int = Field(..., ge=0, description='Controls how tightly mean of cp_resservation around the prior (which is the listing price). Higher lambda means tighter around the prior.')
- a: int = Field(..., ge=0, description='Can be interpreted as number of pseudo-observations that the buyer has of seller\'s reservation price.')
- b: int = Field(..., ge=0, description='Represents prior scale for uncertainty in the buyer\'s estimate of the seller\'s reservation price. Higher b means higher uncertainty.')
'''

class NegotiationSeller(BaseSeller):
    negotiation_config: Optional[dict[str, Any]]= None

'''
Example schema for negotiation_config is:
- own_reservation_: int = Field(..., gt=0, description='The seller\'s own reservation price for the property.')
- own_reservation_std: int = Field(..., ge=0, description='The standard deviation representing uncertainty in the seller\'s reservation price.')
- cp_reservation_: int = Field(..., gt=0, description='The seller\'s estimate of the buyer\'s reservation price for the property.')
- lambda_: int = Field(..., ge=0, description='Controls how tightly mean of cp_resservation around the prior (which is the listing price). Higher lambda means tighter around the prior.')
- a: int = Field(..., ge=0, description='Can be interpreted as number of pseudo-observations that the seller has of buyer\'s reservation price.')
- b: int = Field(..., ge=0, description='Represents prior scale for uncertainty in the seller\'s estimate of the buyer\'s reservation price. Higher b means higher uncertainty.')
'''

# Negotiation Hand-off Schemas (for structured outputs of buyer/seller state at the end of negotiation for hand-off to listing agent)
class NegotiationBuyerHandOffPayload(BaseModel):
    buyer_id: str
    effective_reservation: NormalDistribution


class NegotiationSellerHandOffPayload(BaseModel):
    seller_id: str
    effective_reservation: NormalDistribution


class NegotiationToListingPayload(BaseModel):
    negotiation_history: NegotiationHistoryRecord
    buyer_state: NegotiationBuyerHandOffPayload
    seller_state: NegotiationSellerHandOffPayload


class BuyerMarketBeliefState(BaseModel):
    buyer_id: str
    base_reservation_price: float = Field(ge=0.0)
    effective_reservation: NormalDistribution
    latest_market_feedback: str = 'No market feedback yet.'
    feedback_history: list[str] = Field(default_factory=list)
    latest_observed_min_price: float | None = Field(default=None, ge=0.0)
    latest_observed_avg_price: float | None = Field(default=None, ge=0.0)
    latest_observed_max_price: float | None = Field(default=None, ge=0.0)


class SellerMarketBeliefState(BaseModel):
    seller_id: str
    base_reservation_price: float = Field(ge=0.0)
    effective_reservation: NormalDistribution


class ListingBuyerState(PortalBuyer):
    effective_reservation: NormalDistribution
    latest_search_results: list[PortalSearchResult] = Field(default_factory=list)
    latest_market_feedback: str = 'No market feedback yet.'


class ListingSellerState(PortalSeller):
    effective_reservation: NormalDistribution
    listed: bool
    current_listing_id: str | None = None
    current_listing_price: float | None = None
    open_requests: int = Field(ge=0)


class ListingPortalSnapshot(BaseModel):
    week_number: int = Field(ge=0)
    buyers: list[ListingBuyerState] = Field(default_factory=list)
    sellers: list[ListingSellerState] = Field(default_factory=list)
    matched_pairs: list[NegotiationMatch] = Field(default_factory=list)


class ListingNegotiationTransferPayload(BaseModel):
    match_id: str
    week_matched: int = Field(ge=1)
    listing_record: ListingRecord
    buyer_state: ListingBuyerState
    seller_state: ListingSellerState

# Belief Update Schemas (for structured outputs of buyer belief updates during negotiations)
class UpdateOwnBeliefInfoMetadata(BaseModel):
    """Metadata for buyer self-belief updates during negotiations."""
    estimate: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)


class UpdateOwnBeliefInfo(BaseModel):
    """Information used to update the buyer's own reservation belief."""
    reservation_info: Optional[UpdateOwnBeliefInfoMetadata] = Field(
        None,
        description='Information about own reservation value',
    )


class UpdateOpposingBeliefInfoMetadata(BaseModel):
    """Metadata for counterpart-belief updates during negotiations."""
    estimate: float = Field(
        ge=0.0,
        description="Estimate of the counterpart's reservation value.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence level in the estimate of the counterpart's reservation "
            'value.'
        ),
    )


class UpdateOpposingBeliefTrustMetadata(BaseModel):
    """Metadata for trust updates toward the counterpart."""
    trust_level: float = Field(
        ge=-1.0,
        le=1.0,
        description=(
            'Signed trust signal in the counterpart based on the new information '
            '(-1 distrust, 0 neutral, 1 trust)'
        ),
    )


class UpdateOpposingBeliefInfo(BaseModel):
    """Information used to update beliefs about the counterpart."""
    budget_info: Optional[UpdateOpposingBeliefInfoMetadata] = None
    trust_info: Optional[UpdateOpposingBeliefTrustMetadata] = None


class PersonaMemoryWindow(BaseModel):
    """Structured LLM output for persona-conditioned memory retrieval size."""

    num_memories_to_retrieve: int = Field(ge=4, le=12)


class BuyerOwnConfidenceEstimate(BaseModel):
    """Structured LLM output for buyer self-confidence at pair initialization."""

    own_confidence: float = Field(ge=0.0, le=1.0)


class BuyerCounterpartConfidenceEstimate(BaseModel):
    """Structured LLM output for buyer counterpart-confidence at pair initialization."""

    counterpart_confidence: float = Field(ge=0.0, le=1.0)


class InitialBuyerPairingPriors(BaseModel):
    """Initial belief priors to set once a buyer is paired to a real listing."""
    own_confidence: float = Field(ge=0.0, le=1.0)
    counterpart_confidence: float = Field(ge=0.0, le=1.0)


class InitialSellerPairingPriors(BaseModel):
    """Initial seller-side priors to set once a real buyer/listing pair forms."""

    counterpart_confidence: float = Field(ge=0.0, le=1.0)


# Action Schemas
class MakeOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['MAKE_OFFER'],
        Field(
            description='Start a new price proposal when there is no active offer on the table.'
        ),
    ]
    offer_price: int = Field(..., gt=0, description='The price proposed in the offer.')


class NormalAnswer(ActionReasoningFields):
    type: Annotated[
        Literal['NORMAL_ANSWER'],
        Field(
            description=(
                'Provide a factual/public answer without proposing a new price. '
                'Try to keep the conversation moving unless you intend to walk away.'
            )
        ),
    ]
    answer_details: str = Field(..., description='Detailed public answer to the counterpart.')


class BuyerInquiry(ActionReasoningFields):
    type: Annotated[
        Literal['INQUIRE_BUYER'],
        Field(description='Buyer asks exploratory questions without making a price proposal.'),
    ]
    inquiry_details: str = Field(..., description='Public inquiry details to counterpart.')


class BuyerQuestion(ActionReasoningFields):
    type: Annotated[
        Literal['QUESTION_BUYER'],
        Field(description='Buyer asks a direct question without making a price proposal.'),
    ]
    question_details: str = Field(..., description='Specific public question to counterpart.')


class SellerInquiry(ActionReasoningFields):
    type: Annotated[
        Literal['INQUIRE_SELLER'],
        Field(description='Seller asks exploratory questions without making a price proposal.'),
    ]
    inquiry_details: str = Field(..., description='Public inquiry details to counterpart.')


class AcceptOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['ACCEPT_OFFER'],
        Field(description='Accept the currently active offer and finalize at the agreed price.'),
    ]
    price_settled: int = Field(..., gt=0, description='The price at which the offer is accepted.')


class RejectOffer(VerbalExplanationFields):
    type: Annotated[
        Literal['REJECT_OFFER'],
        Field(description='Reject the currently active offer without proposing a new price.'),
    ]


class MakeCounteroffer(VerbalExplanationFields):
    type: Annotated[
        Literal['MAKE_COUNTEROFFER'],
        Field(
            description=(
                'Respond to the currently active offer with an alternative price. '
                'It cannot repeat the same offer price from the previous round.'
            )
        ),
    ]
    counteroffer_price: int = Field(..., gt=0, description='The counteroffer price in SGD.')


class BuyerWalkAway(VerbalExplanationFields):
    type: Annotated[
        Literal['WALK_AWAY'],
        Field(description='Buyer explicitly ends the negotiation without agreement.'),
    ]


NegotiationBuyerNonOfferAction = Annotated[
    Union[MakeOffer, BuyerInquiry, BuyerQuestion, NormalAnswer, BuyerWalkAway],
    Field(discriminator='type'),
]
NegotiationBuyerOfferAction = Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer, BuyerWalkAway],
    Field(discriminator='type'),
]
NegotiationSellerNonOfferAction = Annotated[
    Union[MakeOffer, SellerInquiry, NormalAnswer],
    Field(discriminator='type'),
]
NegotiationSellerOfferAction = Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer],
    Field(discriminator='type'),
]
NegotiationBuyerAction = Annotated[
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
    Field(discriminator='type'),
]
NegotiationSellerAction = Annotated[
    Union[
        MakeOffer,
        SellerInquiry,
        NormalAnswer,
        AcceptOffer,
        RejectOffer,
        MakeCounteroffer,
    ],
    Field(discriminator='type'),
]


class NegotiationBuyerNonOfferActions(RootModel[NegotiationBuyerNonOfferAction]):
    pass


class NegotiationBuyerOfferActions(RootModel[NegotiationBuyerOfferAction]):
    pass


class NegotiationSellerNonOfferActions(RootModel[NegotiationSellerNonOfferAction]):
    pass


class NegotiationSellerOfferActions(RootModel[NegotiationSellerOfferAction]):
    pass


class NegotiationBuyerActions(RootModel[NegotiationBuyerAction]):
    pass


class NegotiationSellerActions(RootModel[NegotiationSellerAction]):
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

NEGOTIATION_ACTION_TYPE_DESCRIPTIONS: dict[str, str] = {
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
    lines = []
    for action_type in action_types:
        key = str(action_type).strip().upper()
        if not key:
            continue
        description = NEGOTIATION_ACTION_TYPE_DESCRIPTIONS.get(
            key, 'No description available.'
        )
        lines.append(f'- {key}: {description}')
    return '\n'.join(lines)


def get_action_model(role: RoleType, has_active_offer: bool) -> type[RootModel]:
    if role == RoleType.BUYER:
        return (
            NegotiationBuyerOfferActions
            if has_active_offer
            else NegotiationBuyerNonOfferActions
        )
    if role == RoleType.SELLER:
        return (
            NegotiationSellerOfferActions
            if has_active_offer
            else NegotiationSellerNonOfferActions
        )
    raise ValueError(f'No action model for role: {role}')


def get_allowed_action_types(role: RoleType, has_active_offer: bool) -> tuple[str, ...]:
    if role == RoleType.BUYER:
        return BUYER_OFFER_ACTIONS if has_active_offer else BUYER_NON_OFFER_ACTIONS
    if role == RoleType.SELLER:
        return SELLER_OFFER_ACTIONS if has_active_offer else SELLER_NON_OFFER_ACTIONS
    raise ValueError(f'No action set for role: {role}')
