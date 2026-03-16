"""Basic negotiation strategy component."""

import abc
import dataclasses
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from concordia.components.agent import action_spec_ignored
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


class HDBNegotiationStrategy(action_spec_ignored.ActionSpecIgnored):
    """
    Simple Negotiation Strategy for HDB resale market context to prevent long transactions.
    """
    # Parameters
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
        super().__init__(pre_act_label='Negotiation Strategy State and Numeric Facts')
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

        prompt = (  # TODO: rewrite prompt, instead of description, use previous listing scenario and existing scenario condition to determine urgency level instead.
            "# Role\n"
            f"You are estimating negotiation urgency for a {self._role} in an HDB resale negotiation.\n\n"
            "# Task\n"
            f"Estimate how urgent **{self._agent_name}** is to close the negotiation.\n\n"
            "# Input\n"
            "## Agent Description\n"
            f"{self._description}\n\n"
            "# Private Reasoning Process\n"
            "Think step by step **privately** before answering:\n\n"
            "1. Identify signals of time pressure, financial pressure, relocation needs, family needs, or willingness to wait.\n"
            "2. Distinguish strong urgency cues from mild preferences.\n"
            "3. Convert the overall urgency into a score from `0` to `1`.\n"
            "4. Return only the final JSON object. Do not reveal your reasoning.\n\n"
            "# Scoring Rubrics\n"
            "The score is continuous between `0` and `1`, where higher values indicate greater urgency to close the deal. For example:\n"
            "- `0.0` = not urgent at all, very patient, can comfortably wait.\n"
            "- `0.25` = low urgency, some preference to close but no real pressure.\n"
            "- `0.5` = moderate urgency, meaningful motivation to close but not time-critical.\n"
            "- `0.75` = high urgency, clear pressure or strong need to close soon.\n"
            "- `1.0` = extremely urgent, immediate pressure to close.\n\n"
            "# Rules\n"
            "- Use only the agent description provided.\n"
            "- Do not invent facts that are not stated or strongly implied.\n"
            "- Output a single urgency value between `0` and `1`.\n\n"
            "# Output\n"
            "- Return JSON only.\n"
            "- Match the provided schema exactly.\n"
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
    ) -> Dict[str, Any]:
        try:
            return self._uncertainty_context.get_strategy_uncertainty_summary(
                action_context,
            )
        except Exception:
            return {
                'scenario_summary': 'Unknown',
                'issue_items': [],
                'top_issue_question': '',
                'top_issue_score': 0.0,
                'action_confidence': 1.0,
                'risk_tolerance': 0.5,
            }

    def _should_ask_question(
        self,
        uncertainty_summary: Dict[str, Any],
        rounds_left: int,
    ) -> bool:
        if rounds_left <= 1:
            return False
        top_issue_question = str(
            uncertainty_summary.get('top_issue_question', '')
        ).strip()
        if not top_issue_question:
            return False

        action_confidence = float(uncertainty_summary.get('action_confidence', 1.0))
        risk_tolerance = float(uncertainty_summary.get('risk_tolerance', 0.5))
        action_confidence = max(0.0, min(1.0, action_confidence))
        risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        return (1-action_confidence) > risk_tolerance

    def _build_information_focus(
        self,
        uncertainty_summary: Dict[str, Any],
        rounds_left: int,
        has_active_offer: bool,
    ) -> str:
        issue_items = list(uncertainty_summary.get('issue_items', []))
        top_issue_question = str(
            uncertainty_summary.get('top_issue_question', '')
        ).strip()
        if rounds_left <= 1:
            return "Time is almost up. Focus on closing the deal."
        if not issue_items:
            return "No information to gather. Follow your goals in this negotiation."
        if self._should_ask_question(
            uncertainty_summary=uncertainty_summary,
            rounds_left=rounds_left,
        ):
            if has_active_offer:
                return (
                    '[IMPORTANT] If clarification is still needed before responding to the active offer, ask at most one targeted question: '
                    f'"{top_issue_question}"'
                )
            return (
                '[IMPORTANT] Ask at most one targeted question before negotiating further: '
                f'"{top_issue_question}"'
            )
        return (
            'Negotiate rather than stall.'
            'If you ask anything, keep it to one targeted question:\n'
            + issue_items[0]
        )

    # TODO: simple deal outlook summary based on numeric fields, can be enhanced with more complex scenario analysis in the future.
    @staticmethod
    def _deal_outlook_summary(fields: Dict[str, str]) -> str:
        scenario_summary = fields.get('DealScenarios')
        if scenario_summary:
            return scenario_summary

        zopa_feasible = fields.get('ZOPAFeasible', 'Unknown')
        if zopa_feasible == 'True':
            return 'Likely feasible'
        if zopa_feasible == 'False':
            return 'Likely not feasible'
        return 'Unknown'

    @staticmethod
    def _numeric_fact_summary(fields: Dict[str, str]) -> str:
        summary = (
            f"OwnVsOpponentReservation: {fields.get('OwnVsOpponentReservation', 'Unknown')}\n"
            f"HasActiveOffer: {fields.get('HasActiveOffer', 'False')}\n"
            f"ActiveOfferPrice: {fields.get('ActiveOfferPrice', 'NA')}\n"
            f"DealOutlook: {HDBNegotiationStrategy._deal_outlook_summary(fields)}\n"
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

    def _make_pre_act_value(self) -> str:
        """Provide simple strategy guidance before each action."""
        action_context = ''
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
        uncertainty_summary = self._get_uncertainty_strategy_summary(
            action_context,
        )
        numeric_fields['DealScenarios'] = uncertainty_summary.get(
            'scenario_summary', 'Unknown'
        )
        self._last_numeric_fields = dict(numeric_fields)
        numeric_summary = self._numeric_fact_summary(numeric_fields)
        negotiation_numbers = (
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position)}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"{numeric_summary}\n"
        )

        # Get negotiation strategy guidance based on urgency and role
        information_focus = self._build_information_focus(
            uncertainty_summary,
            rounds_left,
            has_active_offer,
        )
        
        if self._role == RoleType.BUYER:
            base_strategy = (
                "Base Strategy:\n"
                "- Evaluate the offer against your reservation price first; use the opponent position only as supporting context.\n"
                "- If OfferWithinOwnReservation is True, consider ACCEPT_OFFER.\n"
                "- If OfferWithinOwnReservation is False, consider REJECT_OFFER or MAKE_COUNTEROFFER.\n"
                "- Use OwnVsOpponentReservation and DealOutlook to judge whether further bargaining is worthwhile, not whether the current offer itself is acceptable.\n"
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
                "- Use OwnVsOpponentReservation and DealOutlook to judge whether further bargaining is worthwhile, not whether the current offer itself is acceptable.\n"
                f"- {information_focus}\n"
            ) if has_active_offer else (
                "\n Base Strategy:\n"
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
        super().update()
    def get_pre_act_label(self) -> str:
        return 'Negotiation Strategy State and Numeric Facts'

    @staticmethod
    def _display_position(value: float | None) -> str:
        if not isinstance(value, (int, float)):
            return 'Unknown'
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            return 'Unknown'
        return f'{parsed:.2f}'
    
    def get_pre_act_value(self) -> str:
        '''Get pre-act value with strategy state and numeric facts for prompting.'''
        return super().get_pre_act_value()

    def get_state(self)-> str:
        '''Get component state for saving /restoring.'''
        numeric_facts = self._numeric_fact_summary(self._last_numeric_fields)
        strategy_summary = getattr(self, 'strategy_summary', '')
        return (
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position)}\n"
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
            elif key == 'DealOutlook':
                restored_numeric_fields['DealScenarios'] = value
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
