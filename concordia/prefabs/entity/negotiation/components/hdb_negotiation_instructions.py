
"""Negotiation-specific instructions component."""

from collections.abc import Mapping
from typing import Optional

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
            negotiation_style: One of 'cooperative', 'competitive', 'integrative', fixed at 'competitive' for HDB negotiations
            reservation_value: Minimum acceptable value (BATNA)
            ethical_constraints: Optional ethical guidelines
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._role = role
        self._description = description
        self._flat_listing = dict(flat_listing) if flat_listing else {}
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

    def _format_flat_listing_details(self) -> str:
        """Render listing details into a compact prompt block."""
        if not self._flat_listing:
            return ''
        lines = ['FLAT LISTING DETAILS (FACTUAL):']
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
        if self._role == RoleType.BUYER:
            goal = (
                "Get the best possible purchase terms for this flat while "
                "keeping the negotiation moving toward agreement, and learn "
                "the flat's true value through targeted information gathering."
            )
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'BUYER-SPECIFIC GUIDANCE:\n'
                f'Always **adhere** to your own preferences when evaluating the flat and making offers.'
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
        instructions = (
            f'You are {self._agent_name}'
            f'ROLE: {self._role.name}. ALWAYS KEEP TO YOUR ROLE.\n'
            f'DESCRIPTION: {self._description}\n'
            f'PRIMARY GOAL: {goal}\n'
            f'ETHICS: {self._ethics}\n\n'
            'CORE NEGOTIATION PRINCIPLES:\n'
            '1. Keep your objective clear and consistent each turn\n'
            '2. Infer interests (timeline, certainty, inclusions) behind positions, not just price itself\n'
            '3. Communicate offers naturally but unambiguously\n'
        )
        instructions += self._format_flat_listing_details()

        if additional_instructions:
            instructions += additional_instructions + '\n\n'
        # TODO: Add HDB-specific policy constraints and guidelines here
        return instructions

    def get_phase_specific_guidance(self) -> str:
        """Get guidance specific to current negotiation phase."""
        if self._negotiation_phase == 'opening':
            return (
                'OPENING PHASE:\n'
                '- Build rapport and establish communication norms\n'
                '- Explore interests and priorities\n'
                '- Set collaborative or competitive tone\n'
                '- Make or solicit initial offers carefully\n'
            )
        elif self._negotiation_phase == 'middle':
            return (
                'MIDDLE PHASE:\n'
                '- Exchange offers and counteroffers\n'
                '- Look for trade-offs and package deals\n'
                '- Test different options and scenarios\n'
                '- Build on areas of agreement\n'
            )
        else:  # closing
            return (
                'CLOSING PHASE:\n'
                '- Finalize remaining issues\n'
                '- Ensure mutual understanding\n'
                '- Document the agreement clearly\n'
                '- Preserve relationship for future\n'
            )

    def get_pre_act_label(self) -> str:
        """Label used when other components reference this pre-act context."""
        return self._pre_act_label

    def get_pre_act_value(self) -> str:
        """Pre-act instruction body used by dependent components."""
        # Build contextual instructions
        instructions = self._base_instructions
        instructions += (
            f'\nCURRENT ROUND COUNTER: {self._rounds_completed}\n'
            'Stay objective-focused: improve your deal while moving toward agreement.\n'
        )

        # Add situational guidance
        if self._last_offer_received:
            instructions += f'\nLAST OFFER RECEIVED: {self._last_offer_received}\n'
            instructions += (
                'Consider how to improve terms from this point and what '
                'concrete next action best advances your objective.\n'
            )

        if self._last_offer_made:
            instructions += f'\nYOUR LAST OFFER: {self._last_offer_made}\n'

        # Add tactical reminders
        instructions += '\nREMEMBER:\n'
        instructions += '- Make each turn advance the negotiation meaningfully\n'
        instructions += '- Every concession should get something in return\n'
        instructions += '- Prefer clear actionable proposals over vague repetition\n'

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
        if 'offer' in action_attempt.lower():
            self._last_offer_made = action_attempt

        if self._verbose:
            print(f'[{self._agent_name}] Negotiation round {self._rounds_completed}')
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        # Track if we received an offer
        if 'offer' in observation.lower():
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
