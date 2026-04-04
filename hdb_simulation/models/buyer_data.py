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
            "preferences": [
                {
                    "category": str,  # one of flat_type, town, transport, schools, shopping, dining, other
                    "description": str
                }
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "3-Room"},
                {"category": "flat_type", "description": "4-Room"},
                {"category": "town", "description": "Jurong East"},
                {"category": "town", "description": "Bukit Batok"},
                {"category": "town", "description": "Clementi"},
                {"category": "schools", "description": "Good schools nearby"},
                {"category": "shopping", "description": "Supermarkets nearby"},
                {"category": "other", "description": "Near a park"},
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "4-Room"},
                {"category": "flat_type", "description": "5-Room"},
                {"category": "town", "description": "Tampines"},
                {"category": "town", "description": "Pasir Ris"},
                {"category": "town", "description": "Bedok"},
                {"category": "transport", "description": "Near MRT"},
                {"category": "shopping", "description": "Supermarkets nearby"},
                {"category": "other", "description": "Childcare options nearby"},
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "3-Room"},
                {"category": "town", "description": "Woodlands"},
                {"category": "town", "description": "Yishun"},
                {"category": "town", "description": "Sembawang"},
                {"category": "transport", "description": "Close to bus interchange"},
                {"category": "other", "description": "Good ventilation"},
                {"category": "other", "description": "Quiet surroundings"},
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "5-Room"},
                {"category": "flat_type", "description": "Executive"},
                {"category": "town", "description": "Bishan"},
                {"category": "town", "description": "Toa Payoh"},
                {"category": "town", "description": "Serangoon"},
                {"category": "dining", "description": "Nearby food options"},
                {"category": "other", "description": "High floor preferred"},
                {"category": "other", "description": "Larger living room"},
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "4-Room"},
                {"category": "town", "description": "Sengkang"},
                {"category": "town", "description": "Punggol"},
                {"category": "town", "description": "Hougang"},
                {"category": "transport", "description": "Near LRT/MRT"},
                {"category": "other", "description": "Park connector access"},
                {"category": "other", "description": "Good natural lighting"},
            ]
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
            "preferences": [
                {"category": "flat_type", "description": "Executive"},
                {"category": "flat_type", "description": "5-Room"},
                {"category": "town", "description": "Queenstown"},
                {"category": "town", "description": "Bukit Merah"},
                {"category": "town", "description": "Clementi"},
                {"category": "transport", "description": "Within walking distance to MRT"},
                {"category": "other", "description": "Mature estate amenities"},
            ]
        },
    },
}
