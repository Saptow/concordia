"""Basic negotiation strategy component."""

import dataclasses
import json
import math
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.common import RoleType
from concordia.prefabs.entity.negotiation import structured_setup_batching
from concordia.prefabs.entity.negotiation.components.uncertain_buyer import UncertainBuyer
from concordia.prefabs.entity.negotiation.components.uncertain_seller import UncertainSeller

# Calibrated threshold: literature supports a monotonic link between high time
# pressure and impasse/exit risk, but not a specific numeric cutoff. We use 0.8
# as the default threshold before buyer-specific inference.
BUYER_WALK_AWAY_URGENCY_THRESHOLD = 0.8
SELLER_EXPLORATION_URGENCY_THRESHOLD = 0.7
DEFAULT_URGENCY_LEVEL = 0.5
MIN_WEEKS_BEFORE_WALK_AWAY = 1
SELF_ACTION_TAG = '[self_action]'
MAX_PRE_ACT_STRATEGY_SUMMARY_CHARS = 360

class UrgencyLevel(BaseModel):
    """Schema for urgency level output."""
    urgency: float = Field(..., ge=0.0, le=1.0, description="Urgency level from 0 (not urgent) to 1 (extremely urgent)")


class WalkAwayThreshold(BaseModel):
    """Schema for buyer-specific walk-away threshold output."""
    walkaway_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Urgency threshold at which the buyer should prefer walking away.",
    )


class SellerExplorationThreshold(BaseModel):
    """Schema for seller-specific exploration threshold output."""
    exploration_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Urgency threshold above which the seller should stop exploratory "
            "questioning and move toward pricing or closure."
        ),
    )

# Tracks only the live pair-local values that this prompt layer uses.
@dataclasses.dataclass
class SimpleStrategyState:
    """Simplified state of negotiation for basic strategies."""
    current_position: float = 0.0
    opponent_position: float = 0.0
    rounds_elapsed: int = 0


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
        buyer_walkaway_threshold: float | None = None,
        seller_exploration_threshold: float | None = None,
        initial_window_position: int | None = None,
        initial_window_size: int | None = None,
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
        self._prefilled_buyer_walkaway_threshold = self._coerce_probability(
            buyer_walkaway_threshold
        )
        self._prefilled_seller_exploration_threshold = self._coerce_probability(
            seller_exploration_threshold
        )
        self._initial_window_position = self._coerce_positive_int(
            initial_window_position
        )
        self._initial_window_size = self._coerce_positive_int(
            initial_window_size
        )
        self._buyer_walkaway_threshold = (
            self._prefilled_buyer_walkaway_threshold
            if self._prefilled_buyer_walkaway_threshold is not None
            else BUYER_WALK_AWAY_URGENCY_THRESHOLD
        )
        self._seller_exploration_threshold = (
            self._prefilled_seller_exploration_threshold
            if self._prefilled_seller_exploration_threshold is not None
            else SELLER_EXPLORATION_URGENCY_THRESHOLD
        )
        self._state = SimpleStrategyState()
        self._urgency_level = DEFAULT_URGENCY_LEVEL
        self._failed_negotiations_count = 0
        self._prefetched_live_strategy_state: Dict[str, Any] | None = None
        self._prefetched_live_urgency_level: float | None = None
        self._sync_positions_from_beliefs()

    def _sync_positions_from_beliefs(self) -> None:
        """Refresh cached reservation positions from the uncertainty component."""
        beliefs = self._uncertainty_context._beliefs
        self._state.current_position = beliefs['own_reservation'].get_expected_mean
        self._state.opponent_position = (
            beliefs['counterpart_reservation'].get_expected_mean
        )

    @staticmethod
    def _coerce_probability(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return min(1.0, max(0.0, parsed))

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _judge_urgency_level(
        self,
        *,
        number_of_failed_negotiations: int = 0,
        current_situation_summary: str = '',
    ) -> float:
        prompt = self._build_urgency_prompt(
            number_of_failed_negotiations=number_of_failed_negotiations,
            current_situation_summary=current_situation_summary,
        )
        response = self._model.sample_text(
            prompt=prompt,
            json_schema=UrgencyLevel.model_json_schema(),
            max_tokens=100,
        )
        return self._parse_urgency_response(response)

    def _build_urgency_prompt(
        self,
        *,
        number_of_failed_negotiations: int = 0,
        current_situation_summary: str = '',
    ) -> str:
        current_situation_block = (
            "## Current Situation\n"
            f"{current_situation_summary.strip()}\n\n"
            if str(current_situation_summary).strip()
            else ''
        )
        prompt = (
            "# Role\n"
            f"You are estimating negotiation urgency for a {self._role} in an HDB resale negotiation.\n\n"
            "# Task\n"
            f"Estimate how urgent **{self._agent_name}** is to close the negotiation.\n\n"
            "# Input\n"
            "## Agent Description\n"
            f"{self._description}\n\n"
            f"{current_situation_block}"
            "## Number Of Failed Negotiations\n"
            f"{max(0, int(number_of_failed_negotiations))}\n\n"
            "# Private Reasoning Process\n"
            "Think step by step **privately** before answering:\n\n"
            "1. Identify signals of time pressure, financial pressure, relocation needs, family needs, willingness to wait, and persona-driven patience.\n"
            "2. Treat the current situation as additional evidence: offer state, price anchoring, and repeated failures can all change urgency.\n"
            "3. For sellers, treat any stated selling context or seller motivations in the description as especially important urgency evidence.\n"
            "4. Distinguish strong urgency cues from mild preferences.\n"
            "5. Convert the overall urgency into a score from `0` to `1`.\n"
            "6. Return only the final JSON object. Do not reveal your reasoning.\n\n"
            "# Scoring Rubrics\n"
            "The score is continuous between `0` and `1`, where higher values indicate greater urgency to close the deal. For example:\n"
            "- `0.0` = not urgent at all, very patient, can comfortably wait.\n"
            "- `0.25` = low urgency, some preference to close but no real pressure.\n"
            "- `0.5` = moderate urgency, meaningful motivation to close but not time-critical.\n"
            "- `0.75` = high urgency, clear pressure or strong need to close soon.\n"
            "- `1.0` = extremely urgent, immediate pressure to close.\n\n"
            "# Rules\n"
            "- Use only the provided agent description, current situation, and failed-negotiation count.\n"
            "- The agent description captures persona, and for sellers may also include selling motivations/context.\n"
            "- Do not invent facts that are not stated or strongly implied.\n"
            "- Output a single urgency value between `0` and `1`.\n\n"
            "# Output\n"
            "- Return JSON only.\n"
            "- Match the provided schema exactly.\n"
        )
        return prompt

    def _parse_urgency_response(self, response: str) -> float:
        try:
            urgency_output = UrgencyLevel.model_validate_json(response)
            return urgency_output.urgency
        except Exception as e:
            if self._verbose:
                print(
                    f"[{self._agent_name}] Failed to parse urgency level, defaulting to 0.5. Error: {e}"
                )
            return 0.5

    @staticmethod
    def _build_walkaway_threshold_prompt(
        *,
        agent_name: str,
        description: str,
    ) -> str:
        return (
            "# Role\n"
            "You infer how much pressure a buyer tolerates before walking away from an HDB resale negotiation.\n\n"
            "# Task\n"
            f"Estimate a buyer-specific `walkaway_threshold` for **{agent_name}** from `0` to `1`.\n\n"
            "# Interpretation\n"
            "- Lower values mean the buyer is quicker to abandon negotiations under pressure.\n"
            "- Higher values mean the buyer is more patient and willing to keep negotiating.\n\n"
            "# Few-shot examples\n"
            "Example 1\n"
            "Description: Needs to move urgently for a new job, dislikes prolonged uncertainty, is decisive, and will quickly leave deals that feel stalled or overpriced.\n"
            'Output: {"walkaway_threshold": 0.34}\n\n'
            "Example 2\n"
            "Description: Practical and balanced, willing to negotiate for a while if the flat seems promising, but does not want endless back-and-forth.\n"
            'Output: {"walkaway_threshold": 0.58}\n\n'
            "Example 3\n"
            "Description: Very patient, methodical, and comfortable waiting for the right deal. Tolerates uncertainty and prefers exhausting negotiation options before exiting.\n"
            'Output: {"walkaway_threshold": 0.86}\n\n'
            "# Input\n"
            "## Buyer Description\n"
            f"{description}\n\n"
            "# Rules\n"
            "- Use only the buyer description.\n"
            "- Focus on patience, decisiveness, flexibility, and tolerance for prolonged uncertainty.\n"
            "- Calibrate your answer relative to the examples above.\n"
            "- Return JSON only.\n"
            "- Do not explain your reasoning.\n\n"
            "# Output\n"
            "- Match the schema exactly.\n"
        )

    @staticmethod
    def _build_seller_exploration_threshold_prompt(
        *,
        agent_name: str,
        description: str,
        initial_window_position: int | None = None,
        initial_window_size: int | None = None,
    ) -> str:
        initial_window_context = "Not provided."
        if (
            initial_window_position is not None
            and initial_window_size is not None
        ):
            initial_window_context = (
                f"Seller entered in initial window position "
                f"{initial_window_position} of {initial_window_size}."
            )
        return (
            "# Role\n"
            "You infer when a seller should stop exploratory questioning and move "
            "toward pricing or closure in an HDB resale negotiation.\n\n"
            "# Task\n"
            f"Estimate a seller-specific `exploration_threshold` for **{agent_name}** from `0` to `1`.\n\n"
            "# Interpretation\n"
            "- Lower values mean the seller should stop exploring sooner under urgency.\n"
            "- Higher values mean the seller can tolerate more urgency before switching from exploration to price-focused action.\n\n"
            "# Few-shot examples\n"
            "Example 1\n"
            "Description: Needs to sell quickly because a new property purchase is already lined up and carrying costs are rising.\n"
            "Initial window position: 1 of 12.\n"
            'Output: {"exploration_threshold": 0.24}\n\n'
            "Example 2\n"
            "Description: Hopes for a good sale but is not under severe time pressure and can wait a bit for the right buyer.\n"
            "Initial window position: 3 of 12.\n"
            'Output: {"exploration_threshold": 0.46}\n\n'
            "Example 3\n"
            "Description: Motivated to transact within a reasonable horizon but still open to learning from buyers before locking into price discussions.\n"
            "Initial window position: 7 of 12.\n"
            'Output: {"exploration_threshold": 0.63}\n\n'
            "Example 4\n"
            "Description: Selling opportunistically, highly patient, and comfortable testing the market before committing.\n"
            "Initial window position: 11 of 12.\n"
            'Output: {"exploration_threshold": 0.84}\n\n'
            "# Input\n"
            "## Seller Description\n"
            f"{description}\n\n"
            "## Initial Window Context\n"
            f"{initial_window_context}\n\n"
            "# Rules\n"
            "- Use the seller description and initial window context when available.\n"
            "- Earlier initial-window positions should generally suggest lower exploration thresholds than later ones, all else equal.\n"
            "- Focus on time pressure, urgency to transact, flexibility, and willingness to keep probing before pricing.\n"
            "- Calibrate your answer relative to the examples above.\n"
            "- Return JSON only.\n"
            "- Do not explain your reasoning.\n\n"
            "# Output\n"
            "- Match the schema exactly.\n"
        )

    @staticmethod
    def _parse_structured_probability_response(
        response: str,
        *,
        schema: type[BaseModel],
        field_name: str,
        fallback: float,
        agent_name: str,
        verbose: bool,
        label: str,
    ) -> float:
        candidates = [str(response or '').strip()]
        extracted_json = HDBNegotiationStrategy._extract_first_json_object(
            candidates[0]
        )
        if extracted_json and extracted_json != candidates[0]:
            candidates.append(extracted_json)

        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed = schema.model_validate_json(candidate)
                value = getattr(parsed, field_name, None)
                if isinstance(value, (int, float)):
                    return float(value)
            except Exception:
                continue

        if verbose:
            print(
                f"[{agent_name}] Failed to parse {label}, defaulting to "
                f"{fallback:.2f}. Raw response: {response}"
            )
        return fallback

    @staticmethod
    def _parse_walkaway_threshold_response(
        response: str,
        *,
        agent_name: str = 'Negotiator',
        verbose: bool = False,
    ) -> float:
        return HDBNegotiationStrategy._parse_structured_probability_response(
            response,
            schema=WalkAwayThreshold,
            field_name='walkaway_threshold',
            fallback=BUYER_WALK_AWAY_URGENCY_THRESHOLD,
            agent_name=agent_name,
            verbose=verbose,
            label='buyer walk-away threshold',
        )

    @staticmethod
    def _parse_seller_exploration_threshold_response(
        response: str,
        *,
        agent_name: str = 'Negotiator',
        verbose: bool = False,
    ) -> float:
        return HDBNegotiationStrategy._parse_structured_probability_response(
            response,
            schema=SellerExplorationThreshold,
            field_name='exploration_threshold',
            fallback=SELLER_EXPLORATION_URGENCY_THRESHOLD,
            agent_name=agent_name,
            verbose=verbose,
            label='seller exploration threshold',
        )

    def apply_listing_handoff(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> None:
        requests = self.build_listing_handoff_requests(listing_payload)
        responses = structured_setup_batching.execute_setup_requests(requests)
        self.apply_listing_handoff_responses(
            listing_payload,
            {
                request.response_key: response
                for request, response in zip(requests, responses)
            },
        )

    def build_listing_handoff_requests(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> list[structured_setup_batching.StructuredSetupRequest]:
        participant_state = (
            listing_payload.buyer_state
            if self._role == RoleType.BUYER
            else listing_payload.seller_state
        )
        return [
            structured_setup_batching.StructuredSetupRequest(
                component=self,
                response_key='urgency',
                prompt_text=self._build_urgency_prompt(
                    number_of_failed_negotiations=len(
                        participant_state.negotiation_history
                    ),
                    current_situation_summary=(
                        self._build_listing_handoff_urgency_context(
                            listing_payload
                        )
                    ),
                ),
                specific_schema=UrgencyLevel,
                max_tokens=100,
            )
        ]

    def apply_listing_handoff_responses(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
        responses_by_key: Dict[str, str],
    ) -> None:
        self._urgency_level = self._parse_urgency_response(
            responses_by_key.get('urgency', '')
        )
        participant_state = (
            listing_payload.buyer_state
            if self._role == RoleType.BUYER
            else listing_payload.seller_state
        )
        negotiation_history = getattr(participant_state, 'negotiation_history', ())
        self._failed_negotiations_count = len(negotiation_history)

    def build_live_urgency_request(
        self,
    ) -> structured_setup_batching.StructuredSetupRequest:
        self._sync_positions_from_beliefs()
        numeric_fields = self._compute_deterministic_numeric_fields()
        has_active_offer = (
            str(numeric_fields.get('HasActiveOffer', '')).strip().lower() == 'true'
        )
        uncertainty_summary = self._get_uncertainty_strategy_summary('')
        self._prefetched_live_strategy_state = {
            'numeric_fields': dict(numeric_fields),
            'has_active_offer': has_active_offer,
            'uncertainty_summary': dict(uncertainty_summary),
        }
        self._prefetched_live_urgency_level = None
        return structured_setup_batching.StructuredSetupRequest(
            component=self,
            response_key='live_urgency',
            prompt_text=self._build_urgency_prompt(
                number_of_failed_negotiations=self._failed_negotiations_count,
                current_situation_summary=self._build_live_urgency_context(
                    has_active_offer=has_active_offer,
                    numeric_fields=numeric_fields,
                ),
            ),
            specific_schema=UrgencyLevel,
            max_tokens=100,
        )

    def apply_live_urgency_response(
        self,
        response: str,
    ) -> None:
        self._prefetched_live_urgency_level = self._parse_urgency_response(response)
        self._urgency_level = self._prefetched_live_urgency_level

    def _build_listing_handoff_urgency_context(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> str:
        listing_price = self._format_money(listing_payload.seller_state.current_listing_price)
        if self._role == RoleType.BUYER:
            buyer_state = listing_payload.buyer_state
            return (
                f"- Negotiation is starting now for a flat listed at {listing_price}.\n"
                f"- Buyer budget range: {self._format_money(buyer_state.budget.min_price)} to "
                f"{self._format_money(buyer_state.budget.max_price)}.\n"
                "- There is no active offer yet.\n"
                f"- Pair week at entry: {listing_payload.week_matched}."
            )
        seller_state = listing_payload.seller_state
        return (
            f"- Negotiation is starting now for your flat listed at {listing_price}.\n"
            f"- Seller expectation range: {self._format_money(seller_state.expectations.min_price)} to "
            f"{self._format_money(seller_state.expectations.max_price)}.\n"
            "- There is no active offer yet.\n"
            f"- Pair week at entry: {listing_payload.week_matched}."
        )

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

    def _extract_action_price(self, payload: Dict[str, Any]) -> float | None:
        for key in ('counteroffer_price', 'offer_price', 'price_settled'):
            parsed = self._coerce_positive_float(payload.get(key))
            if parsed is not None:
                return parsed
        return None

    def _parse_observed_action(self, memory_text: str) -> tuple[str, Dict[str, Any]] | None:
        text = memory_text.strip()
        for tag in (observation_component.OBSERVATION_TAG, SELF_ACTION_TAG):
            if text.startswith(f'{tag} '):
                text = text[len(tag) + 1:].strip()
                break

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
    def _normalize_self_action_payload(action_attempt: str) -> str:
        """Normalize self-action payload while preserving internal fields."""
        text = str(action_attempt).strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if not isinstance(payload, dict):
            return text
        return json.dumps(payload, ensure_ascii=False)

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
            action_type = str(action.get('type', '')).strip().upper()
            if not action_type:
                continue
            if action_type in self._OFFER_OPEN_ACTIONS:
                return True, self._extract_action_price(action), action_type
            if action_type in self._OFFER_CLOSE_ACTIONS:
                return False, None, None
        return False, None, None

    def _compute_deterministic_numeric_fields(self) -> Dict[str, str]:
        # Keep these fields deterministic so the prompt has a stable numeric
        # summary even though the strategy itself is LLM-driven.
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

    def _should_gather_info(
        self,
        uncertainty_summary: Dict[str, Any],
        urgency_level: float,
    ) -> bool:
        top_issue_question = str(
            uncertainty_summary.get('top_issue_question', '')
        ).strip()
        if not top_issue_question:
            return False

        action_confidence = float(uncertainty_summary.get('action_confidence', 1.0))
        risk_tolerance = float(uncertainty_summary.get('risk_tolerance', 0.5))
        action_confidence = max(0.0, min(1.0, action_confidence))
        risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        uncertainty_exceeds_tolerance = (1 - action_confidence) > risk_tolerance
        return uncertainty_exceeds_tolerance

    def _build_live_urgency_context(
        self,
        *,
        has_active_offer: bool,
        numeric_fields: Dict[str, str],
    ) -> str:
        lines = [
            f"- Weeks since negotiation started: {self._state.rounds_elapsed}.",
            f"- Active offer on table: {has_active_offer}.",
            (
                "- Reservation comparison: "
                f"{numeric_fields.get('OwnVsOpponentReservation', 'Unknown')}."
            ),
            f"- Deal outlook: {self._deal_outlook_summary(numeric_fields)}.",
        ]
        active_offer_price = numeric_fields.get('ActiveOfferPrice', 'NA')
        if has_active_offer and active_offer_price not in {'', 'NA'}:
            lines.append(f"- Active offer price: {active_offer_price}.")
        return '\n'.join(lines)

    def _build_information_focus(
        self,
        uncertainty_summary: Dict[str, Any],
        has_active_offer: bool,
        should_gather_info: bool,
        should_close: bool,
        should_walk_away: bool,
    ) -> str:
        issue_items = list(uncertainty_summary.get('issue_items', []))
        top_issue_question = str(
            uncertainty_summary.get('top_issue_question', '')
        ).strip()
        no_material_question_left = not top_issue_question
        if should_walk_away:
            return (
                "[IMPORTANT] Your urgency exceeds your buyer-specific walk-away threshold. "
                "WALK_AWAY NOW."
            )
        if should_close:
            if has_active_offer:
                return "[IMPORTANT] An active offer is on the table and uncertainty is low enough to close. ACCEPT_OFFER or REJECT_OFFER now."
            return "[IMPORTANT] Shift away from open-ended exploration and toward concrete closing actions. MAKE_OFFER or ACCEPT_OFFER now."
        if not has_active_offer and not should_gather_info:
            if (
                self._state.rounds_elapsed >= MIN_WEEKS_BEFORE_WALK_AWAY
                or no_material_question_left
            ):
                return (
                    "[IMPORTANT] No active offer is on the table and the remaining "
                    "uncertainty does not justify further questioning. "
                    "MAKE_OFFER now to establish price discovery."
                )
            return (
                "[IMPORTANT] No active offer is on the table yet. During the first "
                "negotiation week, if you still ask anything, keep it to one lightweight "
                "targeted question and then move toward MAKE_OFFER."
            )
        if not issue_items:
            return "[IMPORTANT] No information to gather. Lean towards MAKE_OFFER after the 1st week of negotiation, otherwise lean toward targeted questioning to resolve the most pressing uncertainty."
        if should_gather_info:
            if has_active_offer:
                return (
                    '[IMPORTANT] If clarification is still needed before responding to the active offer, ask at most one targeted question: '
                    f'"{top_issue_question}"'
                )
            return (
                '[IMPORTANT] Ask at most one targeted question before negotiating further, then lean toward MAKE_OFFER unless the answer creates a new material issue: '
                f'"{top_issue_question}"'
            )
        return (
            'Negotiate rather than stall.'
            'If you ask anything, keep it to one targeted question:\n'
            + issue_items[0]
        )

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

    @staticmethod
    def _compact_strategy_summary(summary: str) -> str:
        normalized_lines = [
            " ".join(str(line).split())
            for line in str(summary or "").splitlines()
            if str(line).strip()
        ]
        normalized = " | ".join(normalized_lines)
        if len(normalized) <= MAX_PRE_ACT_STRATEGY_SUMMARY_CHARS:
            return normalized
        return normalized[: MAX_PRE_ACT_STRATEGY_SUMMARY_CHARS - 3].rstrip() + '...'

    def _make_pre_act_value(self) -> str:
        """Provide simple strategy guidance before each action."""
        action_context = ''
        prefetched_state = self._prefetched_live_strategy_state
        prefetched_urgency = self._prefetched_live_urgency_level
        if prefetched_state is not None and prefetched_urgency is not None:
            numeric_fields = dict(prefetched_state.get('numeric_fields', {}))
            has_active_offer = bool(prefetched_state.get('has_active_offer', False))
            uncertainty_summary = dict(
                prefetched_state.get('uncertainty_summary', {})
            )
            current_urgency_level = prefetched_urgency
        else:
            self._sync_positions_from_beliefs()

            # The prompt policy is state-gated: unresolved uncertainty permits one
            # more information-gathering turn; otherwise active offers should move
            # toward closure, and urgent buyers may walk away after at least one
            # completed negotiation week.
            numeric_fields = self._compute_deterministic_numeric_fields()
            has_active_offer = (
                str(numeric_fields.get('HasActiveOffer', '')).strip().lower()
                == 'true'
            )
            uncertainty_summary = self._get_uncertainty_strategy_summary(
                action_context,
            )
            current_urgency_level = self._judge_urgency_level(
                number_of_failed_negotiations=self._failed_negotiations_count,
                current_situation_summary=self._build_live_urgency_context(
                    has_active_offer=has_active_offer,
                    numeric_fields=numeric_fields,
                ),
            )
        self._urgency_level = current_urgency_level
        should_gather_info = self._should_gather_info(
            uncertainty_summary,
            urgency_level=current_urgency_level,
        )
        should_close = has_active_offer and not should_gather_info
        can_walk_away = (
            self._state.rounds_elapsed >= MIN_WEEKS_BEFORE_WALK_AWAY
        )
        should_walk_away = (
            self._role == RoleType.BUYER
            and can_walk_away
            and current_urgency_level >= self._buyer_walkaway_threshold
            and not should_gather_info
        )
        numeric_fields['DealScenarios'] = uncertainty_summary.get(
            'scenario_summary', 'Unknown'
        )
        self._last_numeric_fields = dict(numeric_fields)
        # Get negotiation strategy guidance based on urgency and role
        information_focus = self._build_information_focus(
            uncertainty_summary,
            has_active_offer,
            should_gather_info,
            should_close,
            should_walk_away,
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
            if should_walk_away:
                urgency_rule = (
                    "[IMPORTANT] Your urgency level is now high enough that "
                    "you should prefer WALK_AWAY instead of extending the negotiation."
                )
            elif (
                current_urgency_level >= self._buyer_walkaway_threshold
                and not can_walk_away
            ):
                urgency_rule = (
                    "[IMPORTANT] Do not choose WALK_AWAY during the first negotiation week. "
                    "Use this period to gather information or make concrete progress instead.\n"
                )
            elif should_close:
                urgency_rule = (
                    "[IMPORTANT] An active offer should now be resolved. "
                    "Choose ACCEPT_OFFER, REJECT_OFFER, or MAKE_COUNTEROFFER based on your reservation value. "
                    "Do not ask more questions first.\n"
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
            if should_close:
                urgency_rule = (
                    "[IMPORTANT] If an offer is active, decide now with "
                    "ACCEPT_OFFER, REJECT_OFFER."
                    "If no offer is active, issue MAKE_OFFER now."
                )
            else:
                urgency_rule = ""


        self.strategy_summary = base_strategy + urgency_rule
        lines = [
            f"WeeksElapsed={self._state.rounds_elapsed}",
            f"FailedNegotiations={self._failed_negotiations_count}",
            f"UrgencyLevel={current_urgency_level:.2f}",
            (
                f"UrgencyThreshold={self._buyer_walkaway_threshold:.2f}"
                if self._role == RoleType.BUYER
                else f"UrgencyThreshold={self._seller_exploration_threshold:.2f}"
            ),
            f"HasActiveOffer={numeric_fields.get('HasActiveOffer', 'False')}",
            f"ActiveOfferPrice={numeric_fields.get('ActiveOfferPrice', 'NA')}",
            f"OwnVsOpponentReservation={numeric_fields.get('OwnVsOpponentReservation', 'Unknown')}",
            f"DealOutlook={self._deal_outlook_summary(numeric_fields)}",
            f"StrategySummary={self._compact_strategy_summary(self.strategy_summary)}",
        ]
        offer_key = (
            'OfferWithinOwnReservation'
            if self._role == RoleType.BUYER
            else 'OfferMeetsOwnReservation'
        )
        if offer_key in numeric_fields:
            lines.insert(
                6,
                f"{offer_key}={numeric_fields.get(offer_key, 'Unknown')}",
            )
        self._prefetched_live_strategy_state = None
        self._prefetched_live_urgency_level = None
        return '\n'.join(lines) + '\n'

    def post_act(self, action_attempt: str) -> str:
        """Update strategy state after each action."""
        action_text = self._normalize_self_action_payload(action_attempt)
        if action_text:
            try:
                memory = self.get_entity().get_component(
                    self._memory_component_key, type_=memory_component.Memory
                )
                memory.add(f'{SELF_ACTION_TAG} {self._agent_name}: {action_text}')
            except Exception:
                if self._verbose:
                    print(
                        f'[{self._agent_name}] Failed to persist self action to memory.'
                    )
        return ""

    def advance_pair_round(self) -> None:
        """Advance elapsed negotiation time once per completed pair-week."""
        self._state.rounds_elapsed += 1

    # TODO: implement once strategy evolution is being done.
    def pre_observe(self, observation: str) -> str:
        """Process incoming observations."""
        return ""
    
    def post_observe(self) -> str:
        """Post-observation processing."""
        return ""
    
    def update(self) -> None:
        """Periodic updates if needed."""
        self._prefetched_live_strategy_state = None
        self._prefetched_live_urgency_level = None
        super().update()
    def get_pre_act_label(self) -> str:
        return 'Negotiation Strategy State and Numeric Facts'
    def get_pre_act_value(self) -> str:
        return super().get_pre_act_value()
    
    @staticmethod
    def _display_position(value: float | None) -> str:
        if not isinstance(value, (int, float)):
            return 'Unknown'
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            return 'Unknown'
        return f'{parsed:.2f}'
    
    def get_state(self)-> str:
        '''Get component state for saving /restoring.'''
        numeric_facts = self._numeric_fact_summary(self._last_numeric_fields)
        strategy_summary = getattr(self, 'strategy_summary', '')
        return (
            f"(DO NOT REVEAL/DISCUSS) Current Reservation Price (in SGD):{self._state.current_position:.2f}\n"
            f"(DO NOT REVEAL/DISCUSS) Opponent Reservation Price (in SGD) :{self._display_position(self._state.opponent_position)}\n"
            f"Number of weeks since negotiation started:{self._state.rounds_elapsed}\n"
            f"Failed Negotiations Count:{self._failed_negotiations_count}\n"
            f"Current Urgency Level (0-1):{self._urgency_level}\n"
            f"Buyer Walk-Away Threshold (0-1):{self._buyer_walkaway_threshold}\n"
            f"Seller Exploration Threshold (0-1):{self._seller_exploration_threshold}\n"
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
            elif key == 'Failed Negotiations Count':
                self._failed_negotiations_count = int(value)
            elif key == 'Buyer Walk-Away Threshold (0-1)':
                self._buyer_walkaway_threshold = float(value)
            elif key == 'Seller Exploration Threshold (0-1)':
                self._seller_exploration_threshold = float(value)
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
