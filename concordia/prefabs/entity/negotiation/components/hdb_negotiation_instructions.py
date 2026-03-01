
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

        # Track negotiation state
        self._negotiation_phase = 'opening'
        self._rounds_completed = 0
        self._last_offer_made = None
        self._last_offer_received = None

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

    @classmethod
    def _is_offer_action(cls, text: str) -> bool:
        payload_json = cls._extract_first_json_object(text)
        if not payload_json:
            return False
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        action_type = str(payload.get('type', '')).strip().upper()
        return action_type in {'MAKE_OFFER', 'MAKE_COUNTEROFFER'}

    def _format_flat_listing_details(self) -> str:
        """Render listing details into a compact prompt block."""
        if not self._flat_listing:
            return ''
        lines = ['FLAT DETAILS (FACTUAL):']
        for key, value in self._flat_listing.items():
            label = str(key).replace('_', ' ').strip().title()
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value) if value else 'None'
            else:
                value_str = str(value)
            lines.append(f'- {label}: {value_str}')
        return '\n'.join(lines) + '\n\n'

    def _generate_base_instructions(self) -> str:
        """Generate base negotiation instructions."""
        additional_instructions = None
        buyer_preferences_block = ''
        if self._role == RoleType.BUYER:
            if self._preferences:
                preference_lines = ['YOUR FLAT PREFERENCES:']
                for key, value in self._preferences.items():
                    label = str(key).replace('_', ' ').strip().title()
                    if isinstance(value, list):
                        value_str = ', '.join(str(v) for v in value) if value else 'None'
                    else:
                        value_str = str(value)
                    preference_lines.append(f'- {label}: {value_str}')
                buyer_preferences_block = '\n'.join(preference_lines) + '\n'
            goal = (
                "Get the best possible purchase terms for this flat while "
                "keeping the negotiation moving toward agreement, and learn "
                "the flat's true value through targeted information gathering."
            )
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'BUYER-SPECIFIC GUIDANCE:\n'
                f'Always **adhere** to your own preferences when evaluating the flat and making offers.\n'
                f'Remember that the other party has more information about the flat condition than you do. '
                f'Use questions and observations to reduce uncertainty and refine your valuation while negotiating price.'
            )
        elif self._role == RoleType.SELLER:
            goal = (
                "Get the best possible sale terms for this flat while keeping "
                "the negotiation moving toward agreement."
            )
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'SELLER-SPECIFIC GUIDANCE:\n'
                f'You have full knowledge of the flat condition. The counterpart party does not. Always remember this when negotiating price.'
            )

        instructions = ( # Note that description is already provided in the self-perception component.
            f'You are {self._agent_name}.\n\n'
            f'ROLE: {self._role.name}. \n\n'
            f'PRIMARY GOAL: {goal}\n\n'
            f'{buyer_preferences_block}'
            f'ETHICS: {self._ethics}\n\n'
            'CORE NEGOTIATION PRINCIPLES:\n'
            '1. Keep your objective clear and consistent each turn\n'
            '2. Infer interests (timeline, certainty, inclusions) behind positions, not just price itself\n'
            '3. Communicate offers naturally but unambiguously\n\n'
        )
        instructions += self._format_flat_listing_details()

        if additional_instructions:
            instructions += additional_instructions + '\n\n'
        # TODO: Add HDB-specific policy constraints and guidelines here
        return instructions


    def get_pre_act_label(self) -> str:
        """Label used when other components reference this pre-act context."""
        return self._pre_act_label

    def get_pre_act_value(self) -> str:
        """Pre-act instruction body used by dependent components."""
        # Build contextual instructions
        instructions = self._base_instructions
        instructions += (
            f'\nCURRENT ROUND COUNTER: {self._rounds_completed}\n'
            'Perform intentional actions over vague repetition.\n'
        )

        return instructions

    def pre_act(self, action_spec) -> str:
        """Provide negotiation instructions before action."""
        del action_spec
        return self.get_pre_act_value()

    def post_act(self, action_attempt: str) -> str:
        """Update state after action."""
        # Always increment round counter on each action
        self._rounds_completed += 1

        # Track if we made an offer
        if self._is_offer_action(action_attempt):
            self._last_offer_made = action_attempt

        if self._verbose:
            print(f'[{self._agent_name}] Negotiation round {self._rounds_completed}')
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        # Track if we received an offer
        if self._is_offer_action(observation):
            self._last_offer_received = observation
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
        # Persist only objective-relevant state.
        last_made = self._last_offer_made.replace('|', '\\|') if self._last_offer_made else ''
        last_received = (
            self._last_offer_received.replace('|', '\\|')
            if self._last_offer_received
            else ''
        )
        return (
            f'rounds={self._rounds_completed}|'
            f'last_made={last_made}|'
            f'last_received={last_received}'
        )

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        if '|' in state:
            parts = state.split('|')
            if len(parts) == 2 and '=' not in parts[0]:
                _, rounds = parts
                self._rounds_completed = int(rounds)
                self._last_offer_made = None
                self._last_offer_received = None
                return

            parsed: dict[str, str] = {}
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    parsed[key.strip()] = value
            self._rounds_completed = int(parsed.get('rounds', '0'))
            last_made = parsed.get('last_made', '').replace('\\|', '|').strip()
            last_received = parsed.get('last_received', '').replace('\\|', '|').strip()
            self._last_offer_made = last_made or None
            self._last_offer_received = last_received or None
            return

        self._rounds_completed = int(state.strip() or 0)
        self._last_offer_made = None
        self._last_offer_received = None
