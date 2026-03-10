"""Negotiation-specific schemas for the HDB simulation."""

from collections.abc import Sequence
from typing import Annotated, Literal, Union

from pydantic import Field, RootModel

from concordia.hdb_simulation.models.schemas.common import (
    ActionReasoningFields,
    RoleType,
    VerbalExplanationFields,
)


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
