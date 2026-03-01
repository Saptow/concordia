"""Basic negotiation strategy component."""

import abc
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.hdb_simulation.models.schemas import RoleType
from concordia.prefabs.entity.negotiation.components.uncertain_buyer import UncertainBuyer
from concordia.prefabs.entity.negotiation.components.uncertain_seller import UncertainSeller
from concordia.typing import entity_component

from concordia.prefabs.entity.negotiation.config import StrategyConfig

AVG_NEGOTIATION_LENGTH = 12  # average number of rounds in HDB resale negotiation (in weeks)
MIN_ROUNDS = 8
MAX_ROUNDS = 15

class UrgencyLevel(BaseModel):
    """Schema for urgency level output."""
    urgency: float = Field(..., ge=0.0, le=1.0, description="Urgency level from 0 (not urgent) to 1 (extremely urgent)")

# TODO: think about integrating the evolution strategies mentioned 
@dataclasses.dataclass
class SimpleStrategyState: 
    """Simplified state of negotiation for basic strategies."""
    current_position: float = 0.0
    opponent_position: float = 0.0
    rounds_elapsed: int = 0

@dataclasses.dataclass
class StrategyState:
    """Current state of negotiation from strategic perspective."""
    current_position: float = 0.0
    opponent_position: float = 0.0
    rounds_elapsed: int = 0
    concessions_made: List[float] = dataclasses.field(default_factory=list)
    zone_of_agreement: Optional[Tuple[float, float]] = None
    negotiation_temperature: float = 1.0  # High = early, Low = late stage

class SimpleNegotiationStrategy(abc.ABC):
    '''Abstract base class for simple negotiation strategies.'''

    @abc.abstractmethod
    def should_walk_away(self, state: SimpleStrategyState) -> bool:
        """Decide whether to walk away from negotiation."""
        pass

class NegotiationStrategy(abc.ABC):
    """Abstract base class for negotiation strategies."""

    @abc.abstractmethod
    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Determine opening offer position."""
        pass

    @abc.abstractmethod
    def calculate_concession(self, state: StrategyState) -> float:
        """Calculate next concession amount."""
        pass

    @abc.abstractmethod
    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Decide whether to accept current offer."""
        pass

    @abc.abstractmethod
    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide tactical guidance for current situation."""
        pass


class CooperativeStrategy(NegotiationStrategy):
    """Cooperative negotiation strategy focused on mutual gains."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with reasonable position showing good faith."""
        # Open at 70% of the distance between reservation and target
        return reservation_value + 0.7 * (target_value - reservation_value)

    def calculate_concession(self, state: StrategyState) -> float:
        """Make steady, predictable concessions to build trust."""
        if state.rounds_elapsed < 3:
            # Larger early concessions to signal cooperation
            return 0.15 * (state.current_position - state.opponent_position)
        else:
            # Smaller but consistent concessions
            return 0.10 * (state.current_position - state.opponent_position)

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Accept if offer is reasonable and shows good faith."""
        # Accept if within threshold of our current position
        return offer >= StrategyConfig.COOPERATIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide cooperative tactical guidance."""
        guidance = "COOPERATIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Make meaningful concessions to build trust\n"
            guidance += "- Share information about your priorities\n"
            guidance += "- Acknowledge the other party's interests\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Continue steady concessions\n"
            guidance += "- Explore creative solutions\n"
            guidance += "- Suggest package deals\n"
        else:
            guidance += "- Work toward final agreement\n"
            guidance += "- Emphasize mutual benefits\n"
            guidance += "- Ensure both parties feel satisfied\n"

        return guidance


class CompetitiveStrategy(NegotiationStrategy):
    """Competitive negotiation strategy focused on value claiming."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with ambitious position to anchor high."""
        # Open at 120% of target value
        return target_value * 1.2

    def calculate_concession(self, state: StrategyState) -> float:
        """Make minimal concessions, decreasing over time."""
        base_concession = StrategyConfig.BASE_CONCESSION_RATE * (state.current_position - state.opponent_position)

        # Reduce concessions as negotiation progresses
        time_factor = max(0.3, 1.0 - (state.rounds_elapsed / 20))

        return base_concession * time_factor

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Only accept if offer meets high standards."""
        # Accept only if within competitive threshold of current position
        return offer >= StrategyConfig.COMPETITIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide competitive tactical guidance."""
        guidance = "COMPETITIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Anchor high with ambitious opening\n"
            guidance += "- Emphasize your alternatives (BATNA)\n"
            guidance += "- Make them make the first concession\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Concede slowly and reluctantly\n"
            guidance += "- Extract value for each concession\n"
            guidance += "- Use time pressure to your advantage\n"
        else:
            guidance += "- Hold firm near your target\n"
            guidance += "- Be willing to walk away\n"
            guidance += "- Make 'final' offers credible\n"

        return guidance


class IntegrativeStrategy(NegotiationStrategy):
    """Integrative negotiation strategy focused on expanding value."""

    def get_opening_position(self, reservation_value: float, target_value: float) -> float:
        """Start with exploratory position to enable value creation."""
        # Open at 85% of distance to target
        return reservation_value + 0.85 * (target_value - reservation_value)

    def calculate_concession(self, state: StrategyState) -> float:
        """Make strategic concessions tied to value creation."""
        if state.zone_of_agreement:
            # If we've identified ZOPA, move toward middle
            zopa_middle = sum(state.zone_of_agreement) / 2
            return 0.2 * (state.current_position - zopa_middle)
        else:
            # Exploratory concessions to find ZOPA
            return 0.1 * (state.current_position - state.opponent_position)

    def should_accept_offer(self, offer: float, state: StrategyState) -> bool:
        """Accept if offer represents good integrated value."""
        # Consider total value created, not just distribution
        if state.zone_of_agreement:
            zopa_middle = sum(state.zone_of_agreement) / 2
            # Accept if reasonably close to middle of ZOPA
            return abs(offer - zopa_middle) < 0.2 * (state.zone_of_agreement[1] - state.zone_of_agreement[0])
        else:
            # Standard acceptance criteria
            return offer >= StrategyConfig.INTEGRATIVE_ACCEPTANCE_THRESHOLD * state.current_position

    def get_tactical_guidance(self, state: StrategyState) -> str:
        """Provide integrative tactical guidance."""
        guidance = "INTEGRATIVE TACTICS:\n"

        if state.rounds_elapsed < 3:
            guidance += "- Ask questions to understand their interests\n"
            guidance += "- Identify all negotiable issues\n"
            guidance += "- Look for different valuations to trade on\n"
        elif state.rounds_elapsed < 8:
            guidance += "- Propose creative package deals\n"
            guidance += "- Suggest conditional agreements\n"
            guidance += "- Find ways to expand the pie\n"
        else:
            guidance += "- Optimize the total value created\n"
            guidance += "- Ensure fair distribution\n"
            guidance += "- Document all aspects of complex deal\n"

        return guidance

class HDBNegotiationStrategy(entity_component.ContextComponent):
    """
    Simple Negotiation Strategy for HDB resale market context to prevent long transactions.
    """
    _OFFER_OPEN_ACTIONS = frozenset({'MAKE_OFFER', 'MAKE_COUNTEROFFER'})
    _OFFER_CLOSE_ACTIONS = frozenset({'REJECT_OFFER', 'ACCEPT_OFFER', 'WALK_AWAY'})

    def __init__(
        self,
        model,
        agent_name: str,
        role: RoleType,
        description: str,
        uncertain_context: Union[UncertainBuyer, UncertainSeller],
        memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
        max_observations: int = 80,
        verbose: bool = False,
    ):
        """Initializes the negotiation strategy component.

        Args:
          model: The language model to use for generating the action attempt.
          agent_name: The name of the agent using this strategy.
          verbose: Whether to enable verbose logging.
        """
        self._model = model
        self._agent_name = agent_name
        self._role = role
        self._uncertainty_context = uncertain_context
        self._description = description
        self._memory_component_key = memory_component_key
        self._max_observations = max(1, int(max_observations))
        self._last_numeric_fields: Dict[str, str] = {}
        self.strategy_summary = ""
        self.fields: Dict[str, str] = {'hasActiveOffer': 'False'}
        self._verbose = verbose
        self._initialise_strategy(uncertain_context)
    
    def _initialise_strategy(self, uncertain_context: Union[UncertainBuyer, UncertainSeller]):
        '''Initialise strategy states and parameters.'''
        if self._role==RoleType.BUYER:
            current_position = uncertain_context._beliefs['own_reservation'].get_expected_mean
            counterpart_position = uncertain_context._beliefs['counterpart_reservation'].get_expected_mean

        elif self._role==RoleType.SELLER:
            current_position = uncertain_context._own_reservation
            counterpart_position = uncertain_context._beliefs['counterpart_reservation'].get_expected_mean

        self._state = SimpleStrategyState(current_position=current_position, opponent_position=counterpart_position)

        # based on the description, get an urgency level
        prompt = (
            f"You are given the description of a {self._role} in an HDB resale negotiation:\n"
            f"Description: {self._description}\n"
            f"Determine how urgent {self._agent_name} is to close the negotiation from 0 to 1 (0 = not urgent, 1 = extremely urgent)."
            f"Output using the schema provided."
        )

        response = self._model.sample_text(
            prompt=prompt,
            json_schema=UrgencyLevel.model_json_schema(),
            max_tokens=100,
        )

        # validate and parse response
        try: 
            urgency_output = UrgencyLevel.model_validate_json(response)
            self._urgency_level = urgency_output.urgency
        except Exception as e:
            if self._verbose:
                print(f"[{self._agent_name}] Failed to parse urgency level, defaulting to 0.5. Error: {e}")
            self._urgency_level = 0.5  # default to medium urgency

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find('{')
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
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
                    return text[start: idx + 1]

        return None

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            parsed = float(value)
            if math.isfinite(parsed) and parsed > 0.0:
                return parsed
            return None
        if isinstance(value, str):
            cleaned = (
                value.strip()
                .replace('$', '')
                .replace('SGD', '')
                .replace('sgd', '')
                .replace(',', '')
            )
            if not cleaned:
                return None
            try:
                parsed = float(cleaned)
            except ValueError:
                return None
            if math.isfinite(parsed) and parsed > 0.0:
                return parsed
        return None

    @staticmethod
    def _extract_action_type(payload: Dict[str, Any]) -> str:
        return str(payload.get('type', '')).strip().upper()

    def _extract_action_price(self, payload: Dict[str, Any]) -> float | None:
        for key in ('counteroffer_price', 'offer_price', 'price_settled'):
            parsed = self._coerce_positive_float(payload.get(key))
            if parsed is not None:
                return parsed
        return None

    def _parse_observed_action(self, memory_text: str) -> tuple[str, Dict[str, Any]] | None:
        text = memory_text.strip()
        obs_tag = observation_component.OBSERVATION_TAG
        if text.startswith(f'{obs_tag} '):
            text = text[len(obs_tag) + 1:].strip()

        actor, sep, payload = text.partition(':')
        if not sep:
            return None
        payload_json = self._extract_first_json_object(payload)
        if not payload_json:
            return None
        try:
            action = json.loads(payload_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(action, dict):
            return None
        return actor.strip(), action

    @staticmethod
    def _format_money(value: float | None) -> str:
        if value is None:
            return 'NA'
        if abs(value - round(value)) <= 1e-9:
            return f'{int(round(value))}'
        return f'{value:.2f}'

    def _format_reservation_comparison(
        self,
        own_reservation: float | None,
        opponent_reservation: float | None,
    ) -> str:
        if own_reservation is None or opponent_reservation is None:
            return 'Unknown'
        diff = own_reservation - opponent_reservation
        if abs(diff) <= 1e-9:
            return 'Reservation Prices are equal.'
        if diff > 0:
            return f'OwnAboveOpponent Reservation Price with Difference of {self._format_money(diff)}'
        return f'OwnBelowOpponent Reservation Price with Difference of {self._format_money(diff)}'

    def _resolve_zopa_feasible(
        self,
        own_reservation: float | None,
        opponent_reservation: float | None,
    ) -> bool | None:
        if own_reservation is None or opponent_reservation is None:
            return None
        if self._role == RoleType.BUYER:
            return own_reservation >= opponent_reservation
        return opponent_reservation >= own_reservation

    def _infer_offer_state(
        self,
        recent_memories: List[str],
    ) -> tuple[bool, float | None, str | None]:
        """Infer active offer from the latest relevant structured action."""
        for mem in reversed(recent_memories):
            parsed = self._parse_observed_action(mem)
            if not parsed:
                continue
            _, action = parsed
            action_type = self._extract_action_type(action)
            if not action_type:
                continue
            if action_type in self._OFFER_OPEN_ACTIONS:
                return True, self._extract_action_price(action), action_type
            if action_type in self._OFFER_CLOSE_ACTIONS:
                return False, None, None
        return False, None, None

    def _compute_deterministic_numeric_fields(self) -> Dict[str, str]:
        recent_memories: List[str] = []
        try:
            memory = self.get_entity().get_component(
                self._memory_component_key, type_=memory_component.Memory
            )
            recent_memories = memory.retrieve_recent(limit=self._max_observations)
        except Exception:
            recent_memories = []

        has_active_offer, active_offer_price, active_offer_type = (
            self._infer_offer_state(recent_memories)
        )

        own_reservation = self._coerce_positive_float(self._state.current_position)
        opponent_reservation = self._coerce_positive_float(self._state.opponent_position)
        reservation_comparison = self._format_reservation_comparison(
            own_reservation=own_reservation,
            opponent_reservation=opponent_reservation,
        )
        zopa_feasible = self._resolve_zopa_feasible(
            own_reservation=own_reservation,
            opponent_reservation=opponent_reservation,
        )

        fields: Dict[str, str] = {
            'OwnVsOpponentReservation': reservation_comparison,
            'ZOPAFeasible': str(zopa_feasible) if zopa_feasible is not None else 'Unknown',
            'HasActiveOffer': str(bool(has_active_offer)),
            'ActiveOfferPrice': (
                self._format_money(active_offer_price) if has_active_offer else 'NA'
            ),
        }

        if has_active_offer:
            fields['ActiveOfferType'] = active_offer_type or 'NA'
            if own_reservation is not None and active_offer_price is not None:
                offer_minus_reservation = active_offer_price - own_reservation
                fields['OfferMinusOwnReservation'] = self._format_money(
                    offer_minus_reservation
                )
                if self._role == RoleType.BUYER:
                    fields['OfferWithinOwnReservation'] = str(
                        active_offer_price <= own_reservation
                    )
                else:
                    fields['OfferMeetsOwnReservation'] = str(
                        active_offer_price >= own_reservation
                    )

        self._last_numeric_fields = dict(fields)
        self.fields = {'hasActiveOffer': fields.get('HasActiveOffer', 'False')}
        return fields

    @staticmethod
    def _numeric_fact_summary(fields: Dict[str, str]) -> str:
        return (
            f"HasActiveOffer: {fields.get('HasActiveOffer', 'False')}\n"
            f"ActiveOfferPrice: {fields.get('ActiveOfferPrice', 'NA')}\n"
            f"IsDealPossible: {fields.get('ZOPAFeasible', 'Unknown')}\n"
        )

    def pre_act(self, action_spec) -> str:
        """Provide simple strategy guidance before each action."""
        del action_spec
        # Update state first
        if self._role==RoleType.BUYER:
            self._state.current_position= self._uncertainty_context._beliefs['own_reservation'].get_expected_mean
            self._state.opponent_position = self._uncertainty_context._beliefs['counterpart_reservation'].get_expected_mean
        elif self._role==RoleType.SELLER:
            self._state.current_position = self._uncertainty_context._own_reservation
            self._state.opponent_position = self._uncertainty_context._beliefs['counterpart_reservation'].get_expected_mean
        
        # Compute negotiation state
        numeric_fields = self._compute_deterministic_numeric_fields()
        numeric_summary = self._numeric_fact_summary(numeric_fields)
        negotiation_numbers = (
            f"Current Reservation Price (in SGD):{self._state.current_position}\n"
            f"Opponent Reservation Price (in SGD):{self._display_position(self._state.opponent_position)}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_summary}\n"
        )

        # Get negotiation strategy guidance based on urgency and role
        horizon = self._max_rounds_from_urgency(self._urgency_level)
        rounds_elapsed = self._state.rounds_elapsed
        rounds_left = max(0, horizon - rounds_elapsed)
        urgency = max(0.0, min(1.0, float(self._urgency_level)))
        
        if self._role == RoleType.BUYER:
            if self.should_walk_away():
                urgency_rule = "Patience horizon **exceeded**. Choose WALK_AWAY."

            if rounds_left <= 1:
                urgency_rule = (
                    "HIGH URGENCY: If an offer is active, decide now with ACCEPT_OFFER or REJECT_OFFER. Do not MAKE_COUNTEROFFER or ask more questions.\n"
                )
            elif rounds_left <= 2:
                urgency_rule = (
                    "ELEVATED URGENCY: Prioritize price-closing actions.\n"
                    "If an offer is active, lean toward ACCEPT_OFFER if it's within reservation, otherwise REJECT_OFFER. "
                    "Avoid open-ended inquiries.\n"
                    "If no offer is active, MAKE_OFFER with a price that shows clear progress toward closure.\n"
                )
            else:
                urgency_rule = ""
        else:  # RoleType.SELLER
            if rounds_left <= 1:
                urgency_rule = (
                    "FINAL DECISION TURN: If an offer is active, decide now with "
                    "ACCEPT_OFFER, REJECT_OFFER."
                    "If no offer is active, issue MAKE_OFFER now."
                )
            elif rounds_left <= 2:
                urgency_rule = (
                    "HIGH DEADLINE PRESSURE: Prioritize price-closing actions and avoid "
                    "open-ended inquiries."
                )
            else:
                urgency_rule = (
                    "NORMAL PRESSURE: Keep progressing toward agreement with concrete "
                    "price movement each turn."
                )

        self.strategy_summary = urgency_rule
        return (
            f"{negotiation_numbers}"
            f"Strategy Summary:{self.strategy_summary}\n"
        )

    def post_act(self, action_attempt: str) -> str:
        """Update strategy state after each action."""
        # Simple parsing to update rounds elapsed
        self._state.rounds_elapsed += 1
        return ""
    
    def _max_rounds_from_urgency(self, urgency: float) -> int:
        """
        Map urgency in [0,1] to a patience horizon (max rounds).
        Higher urgency => fewer allowed rounds.
        """
        urgency = max(0.0, min(1.0, urgency))

        # Linear interpolation from AVG down to MIN
        max_rounds = AVG_NEGOTIATION_LENGTH - urgency * (AVG_NEGOTIATION_LENGTH - MIN_ROUNDS)

        # Round to an integer horizon
        horizon = int(math.ceil(max_rounds))

        # Optional clamps
        horizon = max(MIN_ROUNDS, horizon)
        horizon = min(MAX_ROUNDS, horizon)
        return horizon
        
    def should_walk_away(self) -> bool:
        """
        Walk away if we've exceeded our patience horizon, which is determined by:
        - AVG_NEGOTIATION_LENGTH (baseline)
        - urgency level (0..1), higher => fewer tolerated rounds
        """
        if self._role != RoleType.BUYER:
            return False

        horizon = self._max_rounds_from_urgency(self._urgency_level)

        # Walk away when we've hit or exceeded the horizon
        return self._state.rounds_elapsed >= horizon

    # TODO: implement once strategy evolution is being done.
    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        return ""
    
    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""
    
    def update(self) -> None:
        """Periodic updates if needed."""
        pass
    def get_pre_act_label(self) -> str:
        return 'Negotiation Strategy State and Numeric Facts:\n'

    @staticmethod
    def _display_position(value: float | None) -> str:
        if not isinstance(value, (int, float)):
            return 'Unknown'
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            return 'Unknown'
        return str(parsed)
    
    def get_pre_act_value(self) -> str:
        '''Get pre-act value with strategy state and numeric facts for prompting.'''
        numeric_facts = self._numeric_fact_summary(self._last_numeric_fields)
        return (
            f"Current Reservation Price (in SGD):{self._state.current_position}\n"
            f"Opponent Reservation Price (in SGD):{self._display_position(self._state.opponent_position)}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_facts}\n"
            f"Strategy Summary:{self.strategy_summary}\n"
        )

    def get_state(self)-> str:
        '''Get component state for saving /restoring.'''
        numeric_facts = self._numeric_fact_summary(self._last_numeric_fields)
        strategy_summary = getattr(self, 'strategy_summary', '')
        return (
            f"Current Reservation Price (in SGD):{self._state.current_position}\n"
            f"Opponent Reservation Price (in SGD):{self._display_position(self._state.opponent_position)}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_facts}\n"
            f"Strategy Summary:{strategy_summary}\n"
        )

    def set_state(self, state: str) -> None:
        '''Set component state from saved string.'''
        parts = state.split('\n')
        restored_numeric_fields: Dict[str, str] = {}
        for part in parts:
            if ':' not in part:
                continue
            key, value = part.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'Current Reservation Price (in SGD)':
                self._state.current_position = float(value)
            elif key == 'Current Reservation Price':
                self._state.current_position = float(value)
            elif key == 'Opponent Reservation Price (in SGD)':
                if value != 'Unknown':
                    self._state.opponent_position = float(value)
            elif key == 'Opponent Reservation Price':
                self._state.opponent_position = float(value)
            elif key == 'Number of weeks since negotiation started':
                self._state.rounds_elapsed = int(value)
            elif key == 'Current Urgency Level (0-1)':
                self._urgency_level = float(value)
            elif key == 'UrgencyLevel':
                self._urgency_level = float(value)
            elif key == 'HasActiveOffer':
                restored_numeric_fields['HasActiveOffer'] = value
            elif key == 'ActiveOfferPrice':
                restored_numeric_fields['ActiveOfferPrice'] = value
            elif key == 'IsDealPossible':
                restored_numeric_fields['ZOPAFeasible'] = value
            elif key == 'Strategy Summary':
                self.strategy_summary = value

        if restored_numeric_fields:
            self._last_numeric_fields.update(restored_numeric_fields)
            self.fields = {
                'hasActiveOffer': self._last_numeric_fields.get(
                    'HasActiveOffer', 'False'
                )
            }
