
"""Negotiation-specific instructions component."""

import json
from collections.abc import Mapping
from typing import Optional, Any

from concordia.typing import entity_component
from concordia.hdb_simulation.models.schemas import RoleType

# TODO: This should be the part to inject government policies and HDB-specific constraints
# into the negotiation instructions for agents negotiating HDB resale flats.

class HDBNegotiationInstructions(entity_component.ContextComponent):
    """Instructions component specialized for HDB negotiation contexts.

    This component provides dynamic negotiation guidance based on:
    - Current negotiation phase (opening, middle, closing)
    - Active negotiation style
    - Ethical constraints
    - Contextual constraints specific to HDB resale negotiations
    - Situational factors
    """

    def __init__(
        self,
        agent_name: str,
        role: RoleType,
        description: str,
        flat_listing: Mapping[str, object] | None = None,
        preferences: Mapping[str, Any] | None = None,
        reservation_value: float = 0.0,
        ethical_constraints: Optional[str] = None,
        pre_act_label: str = 'Negotiation instructions',
        verbose: bool = False,
    ):
        """Initialize negotiation instructions.

        Args:
            agent_name: Name of the agent
            description: Brief description of the agent in mind
            flat_listing: Flat listing metadata shared by both buyer and seller
            preferences: Buyer preference metadata used for buyer-side guidance
            negotiation_style: One of 'cooperative', 'competitive', 'integrative', fixed at 'competitive' for HDB negotiations
            reservation_value: Minimum acceptable value (BATNA)
            ethical_constraints: Optional ethical guidelines
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._role = role
        self._description = description
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        self._preferences = dict(preferences) if preferences else {}
        self._reservation_value = reservation_value
        self._ethics = ethical_constraints or 'Be honest and fair. Do not deceive.'
        self._pre_act_label = pre_act_label
        self._verbose = verbose

        # Base instructions
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

    def _format_flat_listing_details(self) -> str:
        """Render listing details into a compact prompt block."""
        if not self._flat_listing:
            return ''
        lines = []
        for key, value in self._flat_listing.items():
            label = str(key).replace('_', ' ').strip().title()
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value) if value else 'None'
            else:
                value_str = str(value)
            lines.append(f'- {label}: {value_str}')
        return 'Flat Details:\n'+'\n'.join(lines) + '\n\n'

    def _generate_base_instructions(self) -> str:
        """Generate base negotiation instructions."""
        additional_instructions = None
        buyer_preferences_block = ''
        
        # Buyer-specifc
        if self._role == RoleType.BUYER:
            if self._preferences:
                preference_lines = []
                for key, value in self._preferences.items():
                    label = str(key).replace('_', ' ').strip().title()
                    if isinstance(value, list):
                        value_str = ', '.join(str(v) for v in value) if value else 'None'
                    else:
                        value_str = str(value)
                    preference_lines.append(f'- {label}: {value_str}')
                buyer_preferences_block = 'Your ideal preferences are:\n' + '\n'.join(preference_lines) + '\n'
            goal = (
                "**Main objective**: Get the best purchase terms for this flat.\n\n"
                "Information to collect:\n"
                "1. Flat value (in SGD): Recent renovations, Commute Options, Nearby Amenities etc \n"
                "2. Seller reservation value (in SGD): Minimum price they are willing to accept.\n"
                "3. Seller constraints: timeline/urgency, flexibility → terms you can offer in exchange for better price/conditions.\n"
            )
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'BUYER-SPECIFIC GUIDANCE:\n'
                f'- Always **adhere** to your own preferences when evaluating the flat and making offers.\n'
                f'- Assume information asymmetry: the seller likely knows more about the flat’. condition and value than you do.'
                f'- Use targeted questions and on-site observations to surface issues, reduce uncertainty, and update your valuation.'
            )
        # Seller-specific
        elif self._role == RoleType.SELLER:
            goal = (
                "**Main objective**: Get the best purchase terms for this flat.\n\n"
                "Information to collect:\n"
                "1. Buyer Reservation Value (in SGD): Maximum price they are willing to pay.\n"
            )
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'SELLER-SPECIFIC GUIDANCE:\n'
                f'You have full knowledge of the flat condition. The counterpart party does not. Always remember this when negotiating price.'
            )

        instructions = ( # Note that description is already provided in the self-perception component.
            f'\nYou are a {self._role.name} in the HDB resale market in Singapore.\n'
            'Your goals:\n'
            f'{goal}\n\n'
            f'{buyer_preferences_block}\n'
            f'{self._ethics}\n\n'
            '## Instructions:\n'
            '1. Treat each round as one week: use the time to gather missing information (if any) and make progress toward your preferred agreement (avoid stalling).\n'
            '2. Infer interests (timeline, certainty, inclusions) behind positions, not just price itself\n'
            '3. Communicate offers naturally and clearly.\n\n'
        )
        instructions += self._format_flat_listing_details()

        if additional_instructions:
            instructions += additional_instructions + '\n'
        # TODO: Add HDB-specific policy constraints and guidelines here
        return instructions


    def get_pre_act_label(self) -> str:
        """Label used when other components reference this pre-act context."""
        return self._pre_act_label

    def get_pre_act_value(self) -> str:
        """Pre-act instruction body used by dependent components."""
        return self._base_instructions
    
    def pre_act(self, action_spec) -> str:
        """Provide negotiation instructions before action."""
        del action_spec
        return self.get_pre_act_value()

    def post_act(self, action_attempt: str) -> str:
        """Update state after action."""
        # Always increment round counter on each action

        return ""
    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""

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
