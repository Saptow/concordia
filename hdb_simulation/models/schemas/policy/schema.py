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


class RetrievedFullPolicyPages(BaseModel):
    policy_pages: list[FullPolicyPage] = Field(
        default_factory=list,
        description="List of retrieved full policy pages with content.",
    )


class RelevantPolicyPathSelection(BaseModel):
    relevant_paths: list[str] = Field(
        default_factory=list,
        description="Exact policy paths from the directory that are relevant to the current negotiation context.",
    )
    retrieval_decision: str = Field(
        ...,
        description="Short explanation of whether relevant policies were found in the directory.",
    )


class PolicyStateEntry(BaseModel):
    """Single policy entry that can be active in the simulation."""

    policy_type: str = Field(min_length=1)
    policy_text: str = Field(min_length=1)
    sources: list[str] = Field(default_factory=list)


class PolicyWeekSchedule(BaseModel):
    """Policies that should become active starting from a specific week."""

    week: int = Field(ge=1)
    policies: list[PolicyStateEntry] = Field(default_factory=list)
    overwrite: bool = False


class PolicyAnnouncementConfig(BaseModel):
    """Top-level YAML config for baseline and scheduled policy changes."""

    initial_state: list[PolicyStateEntry] = Field(default_factory=list)
    policies: list[PolicyWeekSchedule] = Field(default_factory=list)
