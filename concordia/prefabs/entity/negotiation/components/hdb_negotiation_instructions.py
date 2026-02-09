
"""Negotiation-specific instructions component."""

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
        negotiation_style: str = 'competitive',
        reservation_value: float = 0.0,
        ethical_constraints: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize negotiation instructions.

        Args:
            agent_name: Name of the agent
            description: Brief description of the agent in mind
            negotiation_style: One of 'cooperative', 'competitive', 'integrative', fixed at 'competitive' for HDB negotiations
            reservation_value: Minimum acceptable value (BATNA)
            ethical_constraints: Optional ethical guidelines
            verbose: Whether to print debug information
        """
        self._agent_name = agent_name
        self._role = role
        self._description = description
        self._style = negotiation_style
        self._reservation_value = reservation_value
        self._ethics = ethical_constraints or 'Be honest and fair. Do not deceive.'
        self._verbose = verbose

        # Track negotiation state
        self._negotiation_phase = 'opening'
        self._rounds_completed = 0
        self._last_offer_made = None
        self._last_offer_received = None

        # Base instructions
        self._base_instructions = self._generate_base_instructions()

    def _generate_base_instructions(self) -> str:
        """Generate base negotiation instructions."""
        additional_instructions = None
        if self._role == RoleType.BUYER:
            goal = "Discover the true details of the flat and negotiate the lowest possible price within budget if possible."
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'BUYER-SPECIFIC GUIDANCE:\n'
                f'Always **adhere** to your own preferences when evaluating the flat and making offers.'
                f'Remember that the other party has more information about the flat condition than you do. Consider this when discovering more about the flat and negotiating prices.'
            )
        elif self._role == RoleType.SELLER:
            goal = "Negotiate highest possible price to sell the flat strategically based on strategy provided."
            additional_instructions = ( # TODO: expand later with policy-specific aspects of the flat
                f'SELLER-SPECIFIC GUIDANCE:\n'
                f'You have full knowledge of the flat condition. The counterpart party does not. Always remember this when negotiating price.'
            )
        instructions = (
            f'You are {self._agent_name}, a potential {self._role} within the HDB resale market. \n\n'
            f'PRIMARY GOAL: {goal}\n'
            f'NEGOTIATION STYLE: {self._style}\n'
            f'RESERVATION VALUE: {self._reservation_value} (your minimum acceptable outcome)\n'
            f'ETHICS: {self._ethics}\n\n'
            'CORE NEGOTIATION PRINCIPLES:\n'
            '1. Know your walk-away thresholds (BATNA/reservation value) at all times\n'
            '2. Infer interests (timeline, certainty, inclusions) behind positions, not just price itself\n'
            '3. Communicate offers naturally but unambiguously\n'
            'COMPETITIVE STYLE GUIDANCE:\n'
            '- Maximize your own value capture\n'
            '- Control information strategically\n'
            '- Start with ambitious positions\n'
            '- Use leverage when available\n'
            '- Be willing to walk away\n\n'
        )

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

    def pre_act(self, action_spec) -> str:
        """Provide negotiation instructions before action."""
        # Update phase based on rounds
        if self._rounds_completed < 3:
            self._negotiation_phase = 'opening'
        elif self._rounds_completed < 10:
            self._negotiation_phase = 'middle'
        else:
            self._negotiation_phase = 'closing'

        # Build contextual instructions
        instructions = self._base_instructions
        instructions += self.get_phase_specific_guidance()

        # Add situational guidance
        if self._last_offer_received:
            instructions += f'\nLAST OFFER RECEIVED: {self._last_offer_received}\n'
            instructions += 'Consider: Is this above your reservation value? '
            instructions += 'Can you find creative ways to improve the deal?\n'

        if self._last_offer_made:
            instructions += f'\nYOUR LAST OFFER: {self._last_offer_made}\n'

        # Add tactical reminders
        instructions += '\nREMEMBER:\n'
        if self._style == 'cooperative':
            instructions += '- Share information to build trust\n'
        elif self._style == 'competitive':
            instructions += '- Every concession should get something in return\n'
        else:
            instructions += '- Look for ways to expand value for both parties\n'

        return instructions

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
        return f'{self._negotiation_phase}|{self._rounds_completed}'

    def set_state(self, state: str) -> None:
        """Set the component state from a saved string."""
        if '|' in state:
            phase, rounds = state.split('|', 1)
            self._negotiation_phase = phase
            self._rounds_completed = int(rounds)
