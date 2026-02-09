from concordia.hdb_simulation.models.schemas import Flat
'''
Data module for seller configurations and metadata.
TODO: Migrate to pydantic models for validation.
Schema: 
{
    "seller_id": {
        "name": str, 
        "age": int,
        "occupation": str,
        "description": Optional[str], # for additional context that cannot be captured in structured fields
        "flat": Flat,
        "expectations": {
            "min_price": float,
            "max_price": float
        },
    }
}
'''

# Sample seller data
SELLER_DATA = {
    "seller_001": {
        "name": "Xiao Li",
        "age": 45,
        "occupation": "Software Engineer",
        "description": "Xiao Li is a tech-savvy individual looking to upgrade her living space. Not rushing to sell, but open to good offers.",
        "flat": Flat(
            flat_type="3-Room",
            address="Blk 16 Jurong East Singapore 090016",
            description="A cozy 3-room flat with modern amenities and a great view of the city skyline.",
            town="Jurong East",
            storey_range="5-10",
            remaining_lease=75.4,
            contra=False,
            extension_of_stay=False,
            floor_area_sqm=65.0,
            ethnic_eligibility="Chinese (Dec 2020)",
            spr_eligibility="True (Dec 2020)",
        ).model_dump(),
        "expectations": {
            "min_price": 450000.0,
            "max_price": 550000.0
        },
    }
}
