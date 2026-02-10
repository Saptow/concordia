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
# Decisions
class AcceptOffer(BaseModel):
    type: Literal['ACCEPT_OFFER']
    price_settled: float = Field(..., gt=0, description="The price at which the offer is accepted.")

class RejectOffer(BaseModel):
    type: Literal['REJECT_OFFER']

class MakeCounteroffer(BaseModel):
    type: Literal['MAKE_COUNTEROFFER']
    counteroffer_price: float = Field(..., gt=0, description="The price proposed in the counteroffer.")

class BuyerInquiry(BaseModel):
    type: Literal['INQUIRE_BUYER']
    inquiry_details: str = Field(..., description="Details of the buyer's inquiry about the flat's conditions/details.")

class BuyerQuestion(BaseModel):
    type: Literal['QUESTION_BUYER']
    question_details: str = Field(..., description="Specific question posed by the buyer regarding flexibility and urgency of the negotiation.")

# Seller Decisions
class SellerInquiry(BaseModel):
    type: Literal['INQUIRE_SELLER']
    inquiry_details: str = Field(..., description="Details of the seller's inquiry about the buyer's preferences or constraints.")

BuyerActionTypes=Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer, BuyerInquiry, BuyerQuestion],
    Field(discriminator='type')
]

class BuyerActions(RootModel[BuyerActionTypes]):
    pass

SellerActionTypes=Annotated[
    Union[AcceptOffer, RejectOffer, MakeCounteroffer, SellerInquiry],
    Field(discriminator='type')
]
class SellerActions(RootModel[SellerActionTypes]):
    pass

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


    



