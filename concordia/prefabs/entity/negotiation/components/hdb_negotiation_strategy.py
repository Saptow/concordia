"""Basic negotiation strategy component."""

import abc
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.hdb_simulation.models.schemas.common import RoleType
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


class HDBNegotiationStrategy(entity_component.ContextComponent):
    """
    Simple Negotiation Strategy for HDB resale market context to prevent long transactions.
    """
    # Parameters
    _OFFER_OPEN_ACTIONS = frozenset({'MAKE_OFFER', 'MAKE_COUNTEROFFER'})
    _OFFER_CLOSE_ACTIONS = frozenset({'REJECT_OFFER', 'ACCEPT_OFFER', 'WALK_AWAY'})
    _INFO_HAZARD_STEEPNESS = 10.0
    _INFO_HAZARD_MIDPOINT = 0.7
    _ACTIVE_OFFER_INFO_MULTIPLIER = 0.6
    _LATE_STAGE_INFO_BUDGET_CAP = 0.2

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
        self._urgency_level = 0.5
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

        prompt = ( #TODO: rewrite prompt, instead of description, use previous listing scenario and existing scenario condition to determine urgency level instead. 
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

        try:
            urgency_output = UrgencyLevel.model_validate_json(response)
            self._urgency_level = urgency_output.urgency
        except Exception as e:
            if self._verbose:
                print(f"[{self._agent_name}] Failed to parse urgency level, defaulting to 0.5. Error: {e}")
            self._urgency_level = 0.5

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

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == 'true'

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

    def _get_uncertainty_strategy_summary(
        self,
        action_context: str,
        allowed_info_budget: float,
    ) -> Dict[str, Any]:
        try:
            return self._uncertainty_context.get_strategy_uncertainty_summary(
                action_context,
                allowed_info_budget=allowed_info_budget,
            )
        except Exception:
            return {
                'scenario_summary': 'Unknown',
                'info_items': [],
                'recommend_information_gathering': False,
                'avg_uncertainty': 1.0,
            }

    def _get_base_info_budget(self) -> float:
        try:
            base_budget = float(self._uncertainty_context.get_information_gathering_budget())
        except Exception:
            base_budget = 0.0
        return max(0.0, base_budget)

    def _compute_allowed_info_budget(
        self,
        base_budget: float,
        rounds_left: int,
        horizon: int,
        has_active_offer: bool,
    ) -> float:
        """Transform base exploration capacity into a turn-level allowed budget."""
        if base_budget <= 0.0 or rounds_left <= 0:
            return 0.0

        progress_used = 1.0 - (rounds_left / max(1, horizon))
        progress_used = max(0.0, min(1.0, progress_used))

        closure_hazard = 1.0 / (
            1.0
            + math.exp(
                -self._INFO_HAZARD_STEEPNESS
                * (progress_used - self._INFO_HAZARD_MIDPOINT)
            )
        )
        time_multiplier = 1.0 - closure_hazard
        offer_multiplier = (
            self._ACTIVE_OFFER_INFO_MULTIPLIER if has_active_offer else 1.0
        )

        allowed_budget = base_budget * time_multiplier * offer_multiplier
        if rounds_left <= 2:
            allowed_budget = min(
                allowed_budget,
                base_budget * self._LATE_STAGE_INFO_BUDGET_CAP,
            )
        return max(0.0, allowed_budget)

    @staticmethod
    def _build_information_focus(
        info_items: List[str],
        recommend_information_gathering: bool,
        rounds_left: int,
        allowed_info_budget: float,
    ) -> str:
        if rounds_left <= 1 or allowed_info_budget == 0.0 or not info_items:
            return 'No more information gathering; focus entirely on closing the deal with ACCEPT_OFFER, REJECT_OFFER, or WALK_AWAY.'
        if rounds_left <= 2:
            return (
                'If you still need information, ask at most one question: '
                f'{info_items[0]}'
            )
        if recommend_information_gathering:
            return (
                '[IMPORTANT] Information gathering is recommended. Prioritise: '
                + '; '.join(info_items[:2])
            )
        return 'If gathering information, prioritize: ' + '; '.join(info_items[:2])

    @staticmethod
    def _numeric_fact_summary(fields: Dict[str, str]) -> str:
        summary = (
            f"OwnVsOpponentReservation: {fields.get('OwnVsOpponentReservation', 'Unknown')}\n"
            f"HasActiveOffer: {fields.get('HasActiveOffer', 'False')}\n"
            f"ActiveOfferPrice: {fields.get('ActiveOfferPrice', 'NA')}\n"
            f"IsDealPossible: {fields.get('DealScenarios', fields.get('ZOPAFeasible', 'Unknown'))}\n"
        )
        if 'OfferWithinOwnReservation' in fields:
            summary += (
                f"OfferWithinOwnReservation: {fields.get('OfferWithinOwnReservation', 'Unknown')}\n"
            )
        if 'OfferMeetsOwnReservation' in fields:
            summary += (
                f"OfferMeetsOwnReservation: {fields.get('OfferMeetsOwnReservation', 'Unknown')}\n"
            )
        return summary

    def pre_act(self, action_spec) -> str:
        """Provide simple strategy guidance before each action."""
        action_context = action_spec.call_to_action
        # Update state first
        if self._role==RoleType.BUYER:
            self._state.current_position= self._uncertainty_context._beliefs['own_reservation'].get_expected_mean
            self._state.opponent_position = self._uncertainty_context._beliefs['counterpart_reservation'].get_expected_mean
        elif self._role==RoleType.SELLER:
            self._state.current_position = self._uncertainty_context._own_reservation
            self._state.opponent_position = self._uncertainty_context._beliefs['counterpart_reservation'].get_expected_mean
        
        # Compute negotiation state
        numeric_fields = self._compute_deterministic_numeric_fields()
        expected_horizon = AVG_NEGOTIATION_LENGTH
        rounds_elapsed = self._state.rounds_elapsed
        rounds_left = max(0, expected_horizon - rounds_elapsed)
        has_active_offer = self._coerce_bool(numeric_fields.get('HasActiveOffer'))
        allowed_info_budget = self._compute_allowed_info_budget(
            base_budget=self._get_base_info_budget(),
            rounds_left=rounds_left,
            horizon=expected_horizon,
            has_active_offer=has_active_offer,
        )
        uncertainty_summary = self._get_uncertainty_strategy_summary(
            action_context,
            allowed_info_budget=allowed_info_budget,
        )
        numeric_fields['DealScenarios'] = uncertainty_summary.get(
            'scenario_summary', 'Unknown'
        )
        self._last_numeric_fields = dict(numeric_fields)
        numeric_summary = self._numeric_fact_summary(numeric_fields)
        negotiation_numbers = (
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position):.2f}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_summary}\n"
        )

        # Get negotiation strategy guidance based on urgency and role
        information_focus = self._build_information_focus(
            uncertainty_summary.get('info_items', []),
            bool(uncertainty_summary.get('recommend_information_gathering', False)),
            rounds_left,
            allowed_info_budget,
        )
        
        if self._role == RoleType.BUYER:
            base_strategy = (
                "Base Strategy:\n"
                "- Evaluate the offer against your reservation price first; use the opponent position only as supporting context.\n"
                "- If OfferWithinOwnReservation is True, consider ACCEPT_OFFER.\n"
                "- If OfferWithinOwnReservation is False, consider REJECT_OFFER or MAKE_COUNTEROFFER.\n"
                "- Use OwnVsOpponentReservation and IsDealPossible to judge whether further bargaining is worthwhile, not whether the current offer itself is acceptable.\n"
                f"- {information_focus}\n"
            ) if has_active_offer else (
                "Base Strategy:\n"
                f"- {information_focus}\n"
            )
            if self.should_walk_away():
                urgency_rule = "[IMPORTANT] Patience horizon **exceeded**. Choose WALK_AWAY."

            if rounds_left <= 1:
                urgency_rule = (
                    "[IMPORTANT] If an offer is active, decide now with ACCEPT_OFFER or REJECT_OFFER. Do not MAKE_COUNTEROFFER or ask more questions.\n"
                )
            elif rounds_left <= 2:
                urgency_rule = (
                    "[IMPORTANT] Prioritize price-closing actions."
                    "If an offer is active and OfferWithinOwnReservation is True, **HIGHLY** consider ACCEPT_OFFER."
                )
            else:
                urgency_rule = ""
        else:  # RoleType.SELLER
            base_strategy = (
                "Base Strategy:\n"
                "- Evaluate the offer against your reservation price first; use the opponent position only as supporting context.\n"
                "- If OfferMeetsOwnReservation is True, consider ACCEPT_OFFER.\n"
                "- If OfferMeetsOwnReservation is False, consider REJECT_OFFER or MAKE_COUNTEROFFER.\n"
                "- Use OwnVsOpponentReservation and IsDealPossible to judge whether further bargaining is worthwhile, not whether the current offer itself is acceptable.\n"
                f"- {information_focus}\n"
            ) if has_active_offer else (
                "Base Strategy:\n"
                f"- {information_focus}\n"
            )
            if rounds_left <= 1:
                urgency_rule = (
                    "[IMPORTANT] If an offer is active, decide now with "
                    "ACCEPT_OFFER, REJECT_OFFER."
                    "If no offer is active, issue MAKE_OFFER now."
                )
            elif rounds_left <= 2:
                urgency_rule = (
                    "[IMPORTANT] [HIGH DEADLINE PRESSURE] Prioritize price-closing actions and avoid "
                    "open-ended inquiries."
                )
            else:
                urgency_rule = ""


        self.strategy_summary = base_strategy + urgency_rule
        return ('\n'
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
        return 'Negotiation Strategy State and Numeric Facts'

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
        return ('\n'
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position):.2f}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_facts}\n"
            f"Strategy Summary:\n{self.strategy_summary}\n"
        )

    def get_state(self)-> str:
        '''Get component state for saving /restoring.'''
        numeric_facts = self._numeric_fact_summary(self._last_numeric_fields)
        strategy_summary = getattr(self, 'strategy_summary', '')
        return (
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position):.2f}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_facts}\n"
            f"Strategy Summary:\n{strategy_summary}\n"
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
            if key == '(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD)':
                self._state.current_position = float(value)
            elif key == 'Current Reservation Price':
                self._state.current_position = float(value)
            elif key == '(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD)':
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
                restored_numeric_fields['DealScenarios'] = value
            elif key == 'Strategy Summary':
                self.strategy_summary = value

        if restored_numeric_fields:
            self._last_numeric_fields.update(restored_numeric_fields)
            self.fields = {
                'hasActiveOffer': self._last_numeric_fields.get(
                    'HasActiveOffer', 'False'
                )
            }
