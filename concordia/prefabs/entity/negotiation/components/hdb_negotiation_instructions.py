"""Negotiation-specific instructions component."""

from collections.abc import Mapping
from typing import Any, Optional

from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.common import RoleType
from concordia.typing import entity_component


class HDBNegotiationInstructions(action_spec_ignored.ActionSpecIgnored):
    """Instructions component specialized for HDB negotiation contexts."""

    def __init__(
        self,
        agent_name: str,
        role: RoleType,
        flat_listing: Mapping[str, object] | None = None,
        preferences: Mapping[str, Any] | None = None,
        reservation_value: float = 0.0,
        ethical_constraints: Optional[str] = None,
        pre_act_label: str = 'Negotiation instructions',
        verbose: bool = False,
    ):
        """Initialize negotiation instructions."""
        super().__init__(pre_act_label=pre_act_label)
        self._agent_name = agent_name
        self._role = role
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        self._preferences = dict(preferences) if preferences else {}
        self._reservation_value = reservation_value
        self._ethics = ethical_constraints or 'Be honest and fair. Do not deceive.'
        self._pre_act_label = pre_act_label
        self._verbose = verbose
        self._base_instructions = self._generate_base_instructions()

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find('{')
        if start < 0:
            return None
        candidate = text[start:]
        depth = 0
        in_string = False
        escaped = False
        for idx, ch in enumerate(candidate):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return candidate[: idx + 1]
        return None

    def _generate_base_instructions(self) -> str:
        """Generate compact negotiation instructions for action prompts."""
        buyer_preferences_block = ''
        flat_listing_block = ''
        additional_instructions = ''

        if self._flat_listing:
            flat_description = common_schemas.Flat.model_validate(
                self._flat_listing
            ).to_compact_description()
            flat_listing_block = f'Flat: {flat_description}\n'

        if self._role == RoleType.BUYER:
            if self._preferences:
                preference_lines = common_schemas.format_buyer_preferences(
                    self._preferences
                )
                if preference_lines:
                    buyer_preferences_block = (
                        'Preferences:\n' + '\n'.join(preference_lines) + '\n'
                    )
            goal = (
                'Goal: buy on the best overall terms.\n'
                'Track flat value, seller reservation value, and seller constraints.\n'
            )
            additional_instructions = (
                'Buyer guidance:\n'
                '- Follow your preferences when valuing the flat.\n'
                '- Ask targeted questions to reduce uncertainty.\n'
            )
        else:
            goal = (
                'Goal: sell on the best overall terms.\n'
                'Track buyer reservation value and non-price constraints.\n'
            )
            additional_instructions = (
                'Seller guidance:\n'
                '- Use your better knowledge of the flat responsibly.\n'
            )

        instructions = (
            f'You are a {self._role.name} in the Singapore HDB resale market.\n'
            f'{goal}'
            f'{buyer_preferences_block}'
            f'{flat_listing_block}'
            f'Rule: {self._ethics}\n'
            'Rules:\n'
            '- One round = one week; gather missing info and keep the deal moving.\n'
            '- Infer interests behind positions, not just price.\n'
            '- Communicate offers clearly.\n'
            '- Use only listing-stated neighbourhood or amenity facts. If unlisted, treat as unknown and ask.\n'
            f'{additional_instructions}'
        )
        return instructions

    def get_pre_act_label(self) -> str:
        """Label used when other components reference this pre-act context."""
        return self._pre_act_label

    def _make_pre_act_value(self) -> str:
        """Pre-act instruction body used by dependent components."""
        return self._base_instructions

    def apply_listing_handoff(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> None:
        self._flat_listing = listing_payload.listing_record.flat.model_dump(mode='json')
        self._base_instructions = self._generate_base_instructions()

    def pre_act(self, action_spec) -> str:
        """Provide negotiation instructions before action without changing prompt shape."""
        del action_spec
        return self.get_pre_act_value()

    def post_act(self, action_attempt: str) -> str:
        """Update state after action."""
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        del observation
        return ""

    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""

    def update(self) -> None:
        """Update internal state."""
        pass

    @property
    def name(self) -> str:
        """Component name."""
        return 'NegotiationInstructions'

    def get_state(self) -> str:
        """Get the component state for saving/restoring."""
        return ""

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        del state
        pass
