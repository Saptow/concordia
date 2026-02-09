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
        "description": "Looking for a comfortable home close to amenities and public transport. Not rushing, but eager to find the right fit.",
        "budget": {
            "min_price": 400000.0,
            "max_price": 600000.0
        },
        "preferences": {
            "flat_type": ["3-Room", "4-Room"],
            "towns": ["Jurong East", "Bukit Batok", "Clementi"],
            "features": "Preferably near a park and with good schools nearby."
        },
    }
}
