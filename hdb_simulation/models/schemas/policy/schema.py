from pydantic import BaseModel, Field
from enum import StrEnum


class PolicyType(StrEnum):
    FINANCING_AND_AFFORDABILITY = "Financing and Affordability Rules"
    TRANSACTION_RULES = "Transaction Rules/Processes"
    GRANTS = "Grants and Subsidies"

class PolicyPage(BaseModel):
    path: str = Field(..., description="The path of the policy page, to be used for retrieval.")
    source: str = Field(..., description="The source of the policy page, put the URL here.")
    summary: str = Field(..., description="A summary of the policy page content. To be done with a summarisation model.")
    tags: list[PolicyType] = Field(..., description="A list of tags associated with the policy page.")

class FullPolicyPage(PolicyPage):
    content: str = Field(..., description="The full content of the policy page, to be used for retrieval and summarisation.")

# This is the main schema for the policy page directory, which contains a list of policy pages.
class PolicyPageDirectory(BaseModel):
    policy_pages: list[PolicyPage] = Field(..., description="A list of policy pages in the directory.")