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
        "description": "Looking for a comfortable home close to amenities and public transport. Motivated to close soon and open to fair counteroffers for a smooth deal.",
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
        "description": "First-time buyer seeking a family-friendly estate with reliable transport links and prepared to decide quickly if terms are reasonable.",
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
        "description": "Looking for a practical HDB flat with minimal renovation needed and easy commute. Ready to make a practical offer and settle if value is fair.",
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
        "description": "Upgrading their HDB flat because of a larger household and values central access to schools and amenities. Prefers to conclude negotiations in the near term.",
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
        "description": "Wants a move-in ready flat with modern finishes and access to green spaces. Willing to compromise slightly on price to secure the right unit quickly.",
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
        "description": "Seeking a spacious flat in a mature estate close to city-side workplaces. Financially prepared and open to realistic price alignment for a timely close.",
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
