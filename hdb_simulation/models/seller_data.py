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
        "description": "Xiao Li is upgrading her living space and prefers a smooth, timely sale. Open to reasonable counteroffers that can close soon.",
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
            "min_price": 500000.0,
            "max_price": 535000.0
        },
    },
    "seller_002": {
        "name": "Lim Wei Han",
        "age": 52,
        "occupation": "Civil Engineer",
        "description": "Owner of a well-kept flat, motivated to conclude with a serious buyer and flexible within a realistic price band.",
        "flat": Flat(
            flat_type="4-Room",
            address="Blk 512 Tampines Central 7 Singapore 520512",
            description="Renovated 4-room flat near transport and shopping amenities.",
            town="Tampines",
            storey_range="11-15",
            remaining_lease=78.2,
            contra=True,
            extension_of_stay=False,
            floor_area_sqm=93.0,
            ethnic_eligibility="Malay (Jan 2021)",
            spr_eligibility="True (Jan 2021)",
            nearby_amenities=["MRT station", "Shopping mall", "Primary school"],
        ).model_dump(),
        "expectations": {
            "min_price": 700000.0,
            "max_price": 730000.0
        },
    },
    "seller_003": {
        "name": "Goh Hui Jie",
        "age": 39,
        "occupation": "Nurse",
        "description": "Motivated seller with a practical unit that requires only light touch-ups and willing to settle quickly at a fair number.",
        "flat": Flat(
            flat_type="3-Room",
            address="Blk 301 Woodlands Street 31 Singapore 730301",
            description="Bright 3-room corner unit with unblocked views and simple layout.",
            town="Woodlands",
            storey_range="6-10",
            remaining_lease=70.6,
            contra=False,
            extension_of_stay=False,
            floor_area_sqm=67.0,
            ethnic_eligibility="Chinese (Apr 2021)",
            spr_eligibility="True (Apr 2021)",
            nearby_amenities=["Bus interchange", "Market", "Park"],
        ).model_dump(),
        "expectations": {
            "min_price": 460000.0,
            "max_price": 495000.0
        },
    },
    "seller_004": {
        "name": "Rahimah Binte Salleh",
        "age": 50,
        "occupation": "HR Manager",
        "description": "Selling a larger home after children moved out; values smooth and transparent deal execution and is open to practical offers.",
        "flat": Flat(
            flat_type="5-Room",
            address="Blk 118 Toa Payoh Lorong 1 Singapore 310118",
            description="Spacious 5-room flat in a mature estate with strong amenities.",
            town="Toa Payoh",
            storey_range="10-12",
            remaining_lease=74.8,
            contra=False,
            extension_of_stay=True,
            floor_area_sqm=112.0,
            ethnic_eligibility="Indian/Others (Jun 2021)",
            spr_eligibility="True (Jun 2021)",
            nearby_amenities=["MRT station", "Polyclinic", "Food centre"],
        ).model_dump(),
        "expectations": {
            "min_price": 810000.0,
            "max_price": 855000.0
        },
    },
    "seller_005": {
        "name": "Chua Han Rong",
        "age": 44,
        "occupation": "Product Manager",
        "description": "Flexible seller with a modern flat, prioritizing credible buyers and realistic timelines and willing to close promptly.",
        "flat": Flat(
            flat_type="4-Room",
            address="Blk 274B Sengkang East Avenue Singapore 542274",
            description="Modern 4-room unit with efficient layout and renovated kitchen.",
            town="Sengkang",
            storey_range="13-15",
            remaining_lease=81.1,
            contra=True,
            extension_of_stay=False,
            floor_area_sqm=91.0,
            ethnic_eligibility="Chinese (Aug 2021)",
            spr_eligibility="True (Aug 2021)",
            nearby_amenities=["LRT station", "Mall", "Community club"],
        ).model_dump(),
        "expectations": {
            "min_price": 625000.0,
            "max_price": 660000.0
        },
    },
    "seller_006": {
        "name": "Anita Devi",
        "age": 56,
        "occupation": "Business Owner",
        "description": "Owner of a premium executive flat, willing to negotiate with financially ready buyers and close if price is near expectations.",
        "flat": Flat(
            flat_type="Executive",
            address="Blk 9 Queenstown Close Singapore 140009",
            description="Rare executive flat in a mature estate with generous living space.",
            town="Queenstown",
            storey_range="16-20",
            remaining_lease=69.9,
            contra=False,
            extension_of_stay=True,
            floor_area_sqm=142.0,
            ethnic_eligibility="No quota limit (Oct 2021)",
            spr_eligibility="True (Oct 2021)",
            nearby_amenities=["MRT station", "Hawker centre", "Hospital"],
        ).model_dump(),
        "expectations": {
            "min_price": 955000.0,
            "max_price": 1005000.0
        },
    },
}
