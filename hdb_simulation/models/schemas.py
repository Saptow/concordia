from enum import StrEnum
from pydantic import BaseModel, Field
from typing import List, Optional

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

class BuyerActions(StrEnum):
    ACCEPT_OFFER = "Accept Offer"
    REJECT_OFFER = "Reject Offer"
    REVISE_OFFER = "Revise Offer"
    INQUIRE_DETAILS_COUNTERPART = "Inquire Details on counterpart's information/intentions/reservation value"
    INQUIRE_DETAILS_FLAT = "Inquire Details on flat condition/features" # this is on exploration

class SellerActions(StrEnum):
    ACCEPT_OFFER = "Accept Offer"
    REJECT_OFFER = "Reject Offer"
    REVISE_OFFER = "Revise Offer"
    INQUIRE_DETAILS_COUNTERPART = "Inquire Details on counterpart's information/intentions/reservation value"
    DESCRIBE_FLAT = "Describe or show flat condition/features to buyer" 

# Data Models
class ActionDecision(BaseModel):
    action: BuyerActions | SellerActions = Field(description='Choose only one action.')
    
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


    



