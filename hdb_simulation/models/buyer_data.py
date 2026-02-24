'''
Data schema for buyer data.
TODO: Migrate to pydantic models for validation.

Schema:
{
    "buyer_id": {
        "name": str,
        "age": int,
        "occupation": str,
        "description": Optional[str], # for additional context that cannot be captured in structured fields
        "budget": {
            "min_price": float,
            "max_price": float
        },
        "preferences": {
            "flat_type": List[str],
            "towns": List[str],
            "features": str # free text describing other preferences
        }
'''

# Sample buyer data
BUYER_DATA = {
    "buyer_001": {
        "name": "Xiao Ming",
        "age": 32,
        "occupation": "Data Analyst",
        "description": "Xiao Ming, a 32-year-old Data Analyst, is buying with his partner and wants a practical flat near MRT and supermarkets. He is methodical and family-oriented, prioritizing reliable daily convenience over flashy renovation. He can commit quickly if condition and pricing are reasonable.",
        "budget": {
            "min_price": 490000.0,
            "max_price": 545000.0
        },
        "preferences": {
            "flat_type": ["3-Room", "4-Room"],
            "towns": ["Jurong East", "Bukit Batok", "Clementi"],
            "features": "Preferably near a park and with good schools nearby."
        },
    },
    "buyer_002": {
        "name": "Tan Mei Ling",
        "age": 29,
        "occupation": "Marketing Manager",
        "description": "Tan Mei Ling, a 29-year-old Marketing Manager, is a first-time buyer focused on childcare access, safety, and reliable transport. She is organized and forward-looking, balancing lifestyle goals with clear budget discipline. She decides fast on good matches but negotiates firmly on valuation gaps.",
        "budget": {
            "min_price": 690000.0,
            "max_price": 740000.0
        },
        "preferences": {
            "flat_type": ["4-Room", "5-Room"],
            "towns": ["Tampines", "Pasir Ris", "Bedok"],
            "features": "Near MRT, childcare options, and supermarkets."
        },
    },
    "buyer_003": {
        "name": "Arjun Kumar",
        "age": 35,
        "occupation": "Secondary School Teacher",
        "description": "Arjun Kumar, a 35-year-old Secondary School Teacher, prefers functional units with quiet surroundings and a predictable commute. He is pragmatic and routine-driven, valuing stability and low-maintenance living. He is budget-disciplined but can close quickly when condition checks out.",
        "budget": {
            "min_price": 455000.0,
            "max_price": 505000.0
        },
        "preferences": {
            "flat_type": ["3-Room"],
            "towns": ["Woodlands", "Yishun", "Sembawang"],
            "features": "Good ventilation, quiet surroundings, and close to bus interchange."
        },
    },
    "buyer_004": {
        "name": "Nur Aisyah",
        "age": 41,
        "occupation": "Operations Lead",
        "description": "Nur Aisyah, a 41-year-old Operations Lead, is upgrading for a multigenerational household and prioritizes space, layout, and nearby schools and clinics. She is decisive and responsibility-driven, optimizing for smooth family routines. She prefers transparent terms and a near-term close.",
        "budget": {
            "min_price": 805000.0,
            "max_price": 875000.0
        },
        "preferences": {
            "flat_type": ["5-Room", "Executive"],
            "towns": ["Bishan", "Toa Payoh", "Serangoon"],
            "features": "High floor preferred, larger living room, and nearby food options."
        },
    },
    "buyer_005": {
        "name": "Daniel Ong",
        "age": 33,
        "occupation": "UX Designer",
        "description": "Daniel Ong, a 33-year-old UX Designer, wants a move-in-ready flat with good light, sensible storage, and park access. He is detail-oriented and design-conscious, but still practical about trade-offs that improve livability. He can stretch slightly on price for the right unit with quick completion.",
        "budget": {
            "min_price": 620000.0,
            "max_price": 670000.0
        },
        "preferences": {
            "flat_type": ["4-Room"],
            "towns": ["Sengkang", "Punggol", "Hougang"],
            "features": "Near LRT/MRT, park connector access, and good natural lighting."
        },
    },
    "buyer_006": {
        "name": "Priya Nair",
        "age": 46,
        "occupation": "Finance Director",
        "description": "Priya Nair, a 46-year-old Finance Director, is seeking a long-term larger home in a mature estate. She is analytical and long-term focused, weighing both day-to-day comfort and asset quality. She expects data-backed pricing and can proceed decisively once value is clear.",
        "budget": {
            "min_price": 950000.0,
            "max_price": 1015000.0
        },
        "preferences": {
            "flat_type": ["Executive", "5-Room"],
            "towns": ["Queenstown", "Bukit Merah", "Clementi"],
            "features": "Prefer mature estate amenities and within walking distance to MRT."
        },
    },
}
