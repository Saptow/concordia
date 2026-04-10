"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.prefabs.entity.negotiation import structured_setup_batching
from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from pydantic import ValidationError

BUYER_AGENT_DESCRIPTION_PROMPT_MAX_CHARS = 700
BUYER_PREFERENCES_PROMPT_MAX_CHARS = 900
BUYER_NEGOTIATION_HISTORY_PROMPT_MAX_CHARS = 700
BUYER_LISTING_CONTEXT_PROMPT_MAX_CHARS = 700
BUYER_OBSERVATION_PROMPT_MAX_CHARS = 1_200
BUYER_BELIEF_UPDATE_MAX_TOKENS = 192


class UncertainBuyer(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (buyer side)"""

    def __init__(
        self,
        model: Any,
        agent_description: str = '',
        own_confidence: float = 0.7,
        counterpart_confidence: float = 0.7,
        risk_tolerance: float = 0.8,
        preferences: Optional[dict] = None,
        flat_listing: Optional[dict] = None,
        own_reservation_: float=0.0,
        own_reservation_std: float=100.0,
        mu: float = 0.0,
        lambda_: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
        recent_memory_window: int = 6,
        emit_pre_act_context: bool = True,
    ):
        """Initialize uncertainty-aware component.

        Args:
            model: Language model for analysis
            own_confidence: Prior confidence in own reservation estimate (0-1)
            counterpart_confidence: Prior confidence in counterpart reservation estimate (0-1)
            risk_tolerance: Tolerance for uncertainty (0-1)
        """
        super().__init__(pre_act_label='uncertain_buyer')
        self._model = model
        self._agent_description = str(agent_description)
        self._risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self._preferences = preferences or {}
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        self._emit_pre_act_context = emit_pre_act_context
        self._memory_component_key = memory_component_key
        self._recent_memory_window = max(1, int(recent_memory_window))
        self._base_own_confidence = max(0.0, min(1.0, float(own_confidence)))
        self._base_counterpart_confidence = max(
            0.0, min(1.0, float(counterpart_confidence))
        )
        self._last_observation_hash: int | None = None
        self._pending_observations: List[str] = []
        self._debug_trace: List[str] = []

        # Belief state tracking
        self._beliefs: Dict[
            str,
            uncertain_helper.NormalDistribution | uncertain_helper.NormalInverseGamma,
        ] = {}
        self._issue_bank: List[uncertain_helper.NegotiationIssue] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(mu, lambda_, a, b, own_reservation_, own_reservation_std, own_confidence, counterpart_confidence)

    def _get_action_confidence(self) -> float:
        own_confidence = float(self._beliefs['own_reservation'].confidence)
        counterpart_confidence = float(self._beliefs['counterpart_reservation'].confidence)
        return max(0.0, min(1.0, min(own_confidence, counterpart_confidence)))

    def _build_debug_chain_of_thought(self) -> List[str]:
        own_reservation = self._beliefs['own_reservation']
        counterpart_reservation = self._beliefs['counterpart_reservation']
        lines = [
            'Belief snapshot:',
            (
                f'- own_reservation.mean='
                f'{uncertain_helper.format_money(own_reservation.get_expected_mean)}'
            ),
            (
                f'- own_reservation.ci95='
                f'{uncertain_helper.format_interval(own_reservation.get_confidence_interval())}'
            ),
            f'- own_reservation.confidence={own_reservation.confidence:.2f}',
            (
                f'- counterpart_reservation.mean='
                f'{uncertain_helper.format_money(counterpart_reservation.get_expected_mean)}'
            ),
            (
                f'- counterpart_reservation.ci95='
                f'{uncertain_helper.format_interval(counterpart_reservation.get_confidence_interval())}'
            ),
            (
                f'- counterpart_reservation.confidence='
                f'{counterpart_reservation.confidence:.2f}'
            ),
            f'- open_issue_count={len(uncertain_helper.get_open_issues(self._issue_bank))}',
            f'- risk_tolerance={self._risk_tolerance:.2f}',
            f'- scenario_outlook={self._format_scenario_summary()}',
            'Latest uncertainty update:',
        ]
        if self._debug_trace:
            lines.append(f'- {self._debug_trace[-1]}')
        else:
            lines.append('- No uncertainty update recorded yet.')
        return lines

    def _format_scenario_summary(self) -> str:
        scenarios = self._generate_scenarios()
        if not scenarios:
            return 'Unknown'
        return uncertain_helper.build_strategy_scenario_summary(scenarios)

    def _make_pre_act_value(self) -> str:
        own_reservation = self._beliefs['own_reservation']
        counterpart_reservation = self._beliefs['counterpart_reservation']
        action_confidence = self._get_action_confidence()

        lines = [
            'Perspective=Buyer',
            f'OwnReservationMean={uncertain_helper.format_money(own_reservation.get_expected_mean)}',
            f'OwnReservationCI95={uncertain_helper.format_interval(own_reservation.get_confidence_interval())}',
            f'OwnReservationConfidence={own_reservation.confidence:.2f}',
            f'CounterpartReservationMean={uncertain_helper.format_money(counterpart_reservation.get_expected_mean)}',
            f'CounterpartReservationCI95={uncertain_helper.format_interval(counterpart_reservation.get_confidence_interval())}',
            f'CounterpartReservationConfidence={counterpart_reservation.confidence:.2f}',
            f'OpenIssueCount={len(uncertain_helper.get_open_issues(self._issue_bank))}',
            f'TopOpenIssue={uncertain_helper.summarize_top_issue(uncertain_helper.get_top_issue(self._issue_bank))}',
            f'IssueBank={uncertain_helper.format_issue_bank(self._issue_bank)}',
            f'RiskTolerance={self._risk_tolerance:.2f}',
            f'ActionConfidence={action_confidence:.2f}',
            f'ScenarioOutlook={self._format_scenario_summary()}',
        ]
        result = '\n'.join(lines)
        self._logging_channel({
            'Key': self.get_pre_act_label(),
            'Summary': 'Buyer uncertainty state',
            'State': result,
            'Chain of thought': self._build_debug_chain_of_thought(),
        })
        return result

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        if self._emit_pre_act_context:
            return super().pre_act(action_spec)
        del action_spec
        _ = self.get_pre_act_value()
        return ""

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

    def apply_listing_handoff_responses(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
        responses_by_key: Dict[str, str],
    ) -> None:
        buyer_state = listing_payload.buyer_state
        negotiation_count = max(0, len(buyer_state.negotiation_history))
        # Each new listing-to-negotiation handoff starts a fresh pair, so we
        # clear pair-local uncertainty artifacts instead of carrying them over.
        self._issue_bank = []
        self._last_observation_hash = None
        self._pending_observations = []
        self._debug_trace = []
        self._flat_listing = listing_payload.listing_record.flat.model_dump(mode='json')
        # Re-estimate confidence at the start of every new negotiation rather
        # than inheriting the previous pair's confidence state.
        priors = self._parse_initial_pairing_priors_response(
            responses_by_key.get('initial_pairing_priors', '')
        )
        own_confidence = float(priors.own_confidence)
        counterpart_confidence = float(priors.counterpart_confidence)
        self._beliefs['own_reservation'] = self._build_flat_specific_own_reservation(
            listing_payload,
            own_confidence=own_confidence,
        )
        self._beliefs['counterpart_reservation'] = (
            uncertain_helper.build_counterpart_reservation_prior(
                name="Counterpart's Reservation Value",
                source_distribution=self._build_observed_seller_signal(
                    listing_payload
                ),
                confidence=counterpart_confidence,
                negotiation_count=negotiation_count,
            )
        )
        uncertain_helper.append_debug_trace(
            self._debug_trace,
            (
                'Initialized counterpart NIG prior from buyer-observable seller '
                f'signals with negotiation_count={negotiation_count} '
                f'and confidence={counterpart_confidence:.2f}.'
            ),
        )

    def get_effective_reservation_distribution(
        self,
    ) -> uncertain_helper.NormalDistribution:
        return self._beliefs['own_reservation'].model_copy(deep=True)

    def _initialize_default_beliefs(self, mu: float = 0.0, lambda_: float = 1.0, a: float = 1.0, b: float = 1.0, own_reservation_: float = 0.0, own_reservation_std: float = 0.0, own_confidence: float = 0.5, counterpart_confidence: float = 0.5):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = uncertain_helper.NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=max(0.0, mu), 
            lambda_=max(1e-6, lambda_),
            a=max(1e-6, a),
            b=max(1e-6, b),
            confidence=max(0.0, min(1.0, counterpart_confidence)),
        )

        self._beliefs['own_reservation'] = uncertain_helper.NormalDistribution(
            name='Your Own Reservation Value',
            mean=max(0.0, own_reservation_),
            std=max(0.0, own_reservation_std),
            confidence=max(0.0, min(1.0, own_confidence)),
        )

    def _build_initial_pairing_priors_prompt(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> str:
        buyer_state = listing_payload.buyer_state
        listing_record = listing_payload.listing_record
        observation_count = len(buyer_state.latest_search_results)
        negotiation_count = len(buyer_state.negotiation_history)
        if self._has_substantive_market_feedback(
            buyer_state.latest_market_feedback
        ):
            observation_count += 1
        agent_description = uncertain_helper.truncate_prompt_text(
            self._agent_description,
            max_chars=BUYER_AGENT_DESCRIPTION_PROMPT_MAX_CHARS,
        )
        failed_history_summary = self._format_failed_negotiation_history(
            buyer_state
        )
        preferences_text = uncertain_helper.truncate_prompt_text(
            self._preferences,
            max_chars=BUYER_PREFERENCES_PROMPT_MAX_CHARS,
            middle=True,
        )
        failed_history_summary = uncertain_helper.truncate_prompt_text(
            failed_history_summary,
            max_chars=BUYER_NEGOTIATION_HISTORY_PROMPT_MAX_CHARS,
            middle=True,
        )
        listing_context = uncertain_helper.truncate_prompt_text(
            uncertain_helper.build_compact_listing_context(listing_record),
            max_chars=BUYER_LISTING_CONTEXT_PROMPT_MAX_CHARS,
            middle=True,
        )

        return (
            "# Role\n"
            "You are calibrating initial buyer-side priors for a buyer who has just "
            "started a new HDB resale negotiation.\n\n"
            "# Task\n"
            "Estimate both `own_confidence` and `counterpart_confidence` between 0 and 1.\n\n"
            "# Input\n"
            "## Buyer Description / Persona\n"
            f"{agent_description}\n\n"
            "## Buyer Preferences\n"
            f"{preferences_text}\n\n"
            "## Buyer Effective Reservation\n"
            f"{buyer_state.effective_reservation}\n\n"
            "## Negotiation History\n"
            f"- failed_negotiation_count: {negotiation_count}\n"
            f"- listing_stage_observation_count: {observation_count}\n"
            f"{failed_history_summary}\n\n"
            "## Listing Snapshot\n"
            f"{listing_context}\n\n"
            "# Calibration Heuristics\n"
            "- `own_confidence` should mainly reflect how grounded, decisive, and valuation-disciplined the buyer is, adjusted upward by more failed negotiations and listing-stage observation count.\n"
            "- `counterpart_confidence` should mainly reflect how informative the listing is and how much the buyer's preferences and failed-negotiation history help infer the seller's likely reservation value.\n\n"
            "# Few-Shot Examples\n"
            "Example 1:\n"
            "- Buyer persona: analytical, budget-disciplined, compares transactions carefully.\n"
            "- Failed negotiations: 4\n"
            "- Listing strongly matches buyer preferences and is clear and specific.\n"
            '- Output: {"own_confidence": 0.82, "counterpart_confidence": 0.77}\n\n'
            "Example 2:\n"
            "- Buyer persona: uncertain first-time buyer, still figuring out trade-offs.\n"
            "- Failed negotiations: 0\n"
            "- Listing is partial, ambiguous, and only weakly aligned with buyer preferences.\n"
            '- Output: {"own_confidence": 0.38, "counterpart_confidence": 0.34}\n\n'
            "# Rules\n"
            "- Return JSON only.\n"
            "- Keep both values within [0, 1].\n"
            "- Use the persona and negotiation history as the main signal for `own_confidence`.\n"
            "- Use buyer preferences, listing clarity, and failed-negotiation history as the main signal for `counterpart_confidence`.\n"
            "- Use the examples as anchors, not as fixed templates.\n"
        )

    def _parse_initial_pairing_priors_response(
        self,
        response: str,
    ) -> negotiation_schemas.InitialBuyerPairingPriors:
        try:
            priors = negotiation_schemas.InitialBuyerPairingPriors.model_validate_json(
                response
            )
        except ValidationError:
            return negotiation_schemas.InitialBuyerPairingPriors(
                own_confidence=self._base_own_confidence,
                counterpart_confidence=self._base_counterpart_confidence,
            )
        except Exception:
            return negotiation_schemas.InitialBuyerPairingPriors(
                own_confidence=self._base_own_confidence,
                counterpart_confidence=self._base_counterpart_confidence,
            )
        return negotiation_schemas.InitialBuyerPairingPriors(
            own_confidence=max(0.0, min(1.0, float(priors.own_confidence))),
            counterpart_confidence=max(
                0.0, min(1.0, float(priors.counterpart_confidence))
            ),
        )

    @staticmethod
    def _has_substantive_market_feedback(feedback: str) -> bool:
        normalized = ' '.join(str(feedback).split()).strip().casefold()
        return bool(normalized and normalized != 'no market feedback yet.')

    @staticmethod
    def _format_failed_negotiation_history(
        buyer_state: negotiation_schemas.ListingBuyerState,
        *,
        limit: int = 3,
    ) -> str:
        history = list(buyer_state.negotiation_history)
        if not history:
            return '- None'

        lines: list[str] = []
        for record in history[-max(1, int(limit)):]:
            offer_count = len(record.offer_history)
            if record.offer_history:
                first_offer = min(
                    int(offer.offer_price) for offer in record.offer_history
                )
                last_offer = int(record.offer_history[-1].offer_price)
                offer_summary = (
                    f'offer_count={offer_count}, '
                    f'first_offer={first_offer}, '
                    f'last_offer={last_offer}'
                )
            else:
                offer_summary = 'offer_count=0'
            end_week = (
                str(record.end_week)
                if record.end_week is not None
                else 'ongoing/unknown'
            )
            lines.append(
                f'- seller_id={record.seller_id}, start_week={record.start_week}, '
                f'end_week={end_week}, {offer_summary}'
            )
        return '\n'.join(lines)

    def build_listing_handoff_requests(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> list[structured_setup_batching.StructuredSetupRequest]:
        return [
            structured_setup_batching.StructuredSetupRequest(
                component=self,
                response_key='initial_pairing_priors',
                prompt_text=self._build_initial_pairing_priors_prompt(
                    listing_payload
                ),
                specific_schema=negotiation_schemas.InitialBuyerPairingPriors,
                max_tokens=120,
            ),
        ]

    def _build_flat_specific_own_reservation(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
        *,
        own_confidence: float,
    ) -> uncertain_helper.NormalDistribution:
        buyer_state = listing_payload.buyer_state
        listing_prior = buyer_state.effective_reservation
        preference_match_score = common_schemas.build_buyer_flat_preference_match_score(
            buyer_state.preferences,
            listing_payload.listing_record.flat,
        )
        base_mean = max(0.0, float(listing_prior.mean))
        base_std = max(1.0, float(listing_prior.std))
        mismatch_discount = (1.0 - preference_match_score) * base_std
        flat_mean = max(0.0, base_mean - mismatch_discount)
        uncertain_helper.append_debug_trace(
            self._debug_trace,
            (
                'Initialized flat-specific buyer own reservation from effective '
                f'reservation and preference match score={preference_match_score:.2f}, '
                f'base_mean={base_mean:.2f}, base_std={base_std:.2f}, '
                f'mismatch_discount={mismatch_discount:.2f}.'
            ),
        )
        return uncertain_helper.NormalDistribution(
            name='Your Reservation Value For This Flat',
            mean=flat_mean,
            std=base_std,
            confidence=max(0.0, min(1.0, own_confidence)),
            # Fresh pair-local belief rebuild for this matched flat.
            evidence_count=0,
            last_updated=None,
        )

    def _build_observed_seller_signal(
        self,
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> uncertain_helper.NormalDistribution:
        buyer_state = listing_payload.buyer_state
        listing_prior = buyer_state.effective_reservation
        listing_price = uncertain_helper.coerce_positive_float(
            listing_payload.listing_record.listing_price,
            default=float(listing_prior.mean),
        )
        observation_count = len(buyer_state.latest_search_results)
        if self._has_substantive_market_feedback(
            buyer_state.latest_market_feedback
        ):
            observation_count += 1
        observed_std = max(
            1.0,
            max(
                float(listing_prior.std),
                0.10 * max(1.0, listing_price),
                20000.0,
            ) / math.sqrt(1.0 + observation_count),
        )
        return uncertain_helper.NormalDistribution(
            name='Observed Seller Signal',
            mean=max(0.0, listing_price),
            std=observed_std,
            confidence=max(0.0, min(1.0, float(listing_prior.confidence))),
            evidence_count=observation_count,
            last_updated=None,
        )

    def _build_belief_summary_for_strategy(self) -> str:
        own_reservation = self._beliefs['own_reservation']
        counterpart_reservation = self._beliefs['counterpart_reservation']
        lines = [
            f'OwnReservationMean={uncertain_helper.format_money(own_reservation.get_expected_mean)}',
            f'OwnReservationCI95={uncertain_helper.format_interval(own_reservation.get_confidence_interval())}',
            f'OwnReservationConfidence={own_reservation.confidence:.2f}',
            f'CounterpartReservationMean={uncertain_helper.format_money(counterpart_reservation.get_expected_mean)}',
            f'CounterpartReservationCI95={uncertain_helper.format_interval(counterpart_reservation.get_confidence_interval())}',
            f'CounterpartReservationConfidence={counterpart_reservation.confidence:.2f}',
            f'OpenIssueCount={len(self._issue_bank)}',
            f'TopOpenIssue={uncertain_helper.summarize_top_issue(uncertain_helper.get_top_issue(self._issue_bank))}',
            f'IssueBank={uncertain_helper.format_issue_bank(self._issue_bank)}',
            f'ScenarioOutlook={self._format_scenario_summary()}',
        ]
        return '\n'.join(lines)

    def _replace_issue_bank(
        self,
        issues: List[uncertain_helper.NegotiationIssue],
    ) -> List[uncertain_helper.NegotiationIssue]:
        self._issue_bank = uncertain_helper.sanitize_issue_bank(issues)
        return list(self._issue_bank)

    def _build_own_reservation_update_prompt(self, context: str) -> str:
        """Build the structured prompt for a buyer self-belief update."""
        truncated_context = uncertain_helper.truncate_prompt_text(
            context,
            max_chars=BUYER_OBSERVATION_PROMPT_MAX_CHARS,
            middle=True,
        )
        preferences_text = uncertain_helper.truncate_prompt_text(
            self._preferences,
            max_chars=BUYER_PREFERENCES_PROMPT_MAX_CHARS,
            middle=True,
        )
        return (
            "# Role\n"
            "You are a buyer in an HDB resale negotiation with imperfect information.\n\n"
            "# Task\n"
            "You observe a new situation (Observation). Given your preferences and current beliefs, decide whether this observation contains material information that should update **your own reservation value** for this flat.\n\n"
            "# Inputs\n"
            "## Observation\n"
            f"{truncated_context}\n\n"
            "## Preferences\n"
            f"{preferences_text}\n\n"
            "## Current Belief\n"
            f"- Current reservation value: {self._beliefs['own_reservation'].get_expected_mean:.2f}\n"
            f"- Current confidence level: {self._beliefs['own_reservation'].confidence:.2f}\n\n"
            "# Private Reasoning Process (MUST FOLLOW)\n"
            "Think step by step **privately** before answering:\n\n"
            "1. Identify any concrete new facts in the observation that affect your valuation of the flat.\n"
            "2. Ignore facts that **DO NOT** materially change your willingness to pay.\n"
            "3. Decide whether these new facts justify changing your reservation value at all.\n"
            "4. If yes, estimate the updated reservation value and assign a confidence level`[0, 1]`.\n"
            "5. If no, do **not** invent a value. Return an empty object instead.\n\n"
            "# Rules\n"
            "- Use only information grounded in the provided observation and preferences.\n"
            "- Do not use `0` or `0.0` as a placeholder estimate.\n"
            "- The updated reservation value must be a valid monetary amount (in Singapore Dollars).\n"
            "- Do not simply return the current reservation value if there is no meaningful update. If there is no meaningful update, return an empty object `{}` instead to indicate that your reservation value remains unchanged.\n"
            "- If there is no meaningful new signal, you are allowed to return an empty object. `{}` \n\n"
            "# Output\n"
            "- Return JSON only.\n"
            "- Match the provided schema exactly.\n"
        )

    def _apply_own_reservation_update_response(self, response: str) -> str:
        """Apply one buyer self-belief update response."""
        try:
            info_update = negotiation_schemas.UpdateOwnBeliefInfo.model_validate_json(
                response
            )
        except ValidationError:
            return 'Own reservation unchanged: model output was invalid.'
        if info_update.reservation_info:
            old_mean = self._beliefs['own_reservation'].get_expected_mean
            self._beliefs['own_reservation'].update_with_evidence(
                info_update.reservation_info.estimate,
                info_update.reservation_info.confidence
            )
            new_mean = self._beliefs['own_reservation'].get_expected_mean
            return (
                'Own reservation updated from '
                f'{uncertain_helper.format_money(old_mean)} to {uncertain_helper.format_money(new_mean)} '
                f'using estimate={uncertain_helper.format_money(info_update.reservation_info.estimate)} '
                f'confidence={info_update.reservation_info.confidence:.2f}.'
            )
        return 'Own reservation unchanged: no reservation signal extracted.'

    def _build_counterpart_reservation_update_prompt(self, context: str) -> str:
        """Build the structured prompt for a buyer counterpart-belief update."""
        truncated_context = uncertain_helper.truncate_prompt_text(
            context,
            max_chars=BUYER_OBSERVATION_PROMPT_MAX_CHARS,
            middle=True,
        )
        return (
            "# Role\n"
            "You are a buyer in an HDB resale negotiation with imperfect information.\n\n"
            "# Task\n"
            "Given an observation, infer the following: \n"
            "- **seller's likely reservation value**, with a confidence level.\n"
            "- (ONLY if there is no usable signal on reservation value) **trust level signal** on the seller based on the new information [-1 to 1], where -1 indicates negative trust and 1 indicates positive trust.\n\n"
            "## Observation\n"
            f"{truncated_context}\n\n"
            "## Current Belief about Seller's Reservation Value\n"
            f"- Reservation Value Estimate: {self._beliefs['counterpart_reservation'].get_expected_mean:.2f}\n"
            f"- Confidence in Reservation Value: `{self._beliefs['counterpart_reservation'].confidence:.2f}\n\n"
            "## Important Prior Guardrail\n"
            "- Your current belief already includes the seller-side listing-price anchor as the prior.\n"
            "- Therefore, a listing price, asking price, or repeated ask is NOT by itself a new reason to update `budget_info`.\n\n"
            "# Private Reasoning Process\n"
            "Think step by step **privately** before answering:\n\n"
            "1. Extract **ONLY** concrete evidence about the seller's minimum acceptable price.\n"
            "2. Ignore any information that is not directly related to the seller's reservation value.\n"
            "3. Treat listing prices, stated asks, offers, counters, urgency, and willingness to compromise as evidence, "
            "but do not assume any single number is automatically the true reservation value.\n"
            "4. If the observation only repeats the existing listing/asking price anchor without new evidence about flexibility, urgency, bottom line, constraints, or willingness to concede, do not update `budget_info`.\n"
            "5. Decide whether there is enough usable evidence to estimate `budget_info`.\n"
            "6. If yes, provide your best estimate of the seller's reservation value and a confidence level (0-1) for that estimate based on the strength of the evidence.\n"
            "7. (IMPORTANT) If there is no available estimate on the seller's reservation value, extract **ANY INFORMATION** about the trustworthiness of the seller based on the given observation.\n"
            "   - For example, if you observe that the seller is intentionally hiding information, being evasive, or providing inconsistent signals, that would be a negative trust signal.\n"
            "   - For example, if you observe the seller being transparent, providing reasonable justifications for their price, or showing willingness to find a mutually beneficial deal, that would be a positive trust signal.\n"
            "8. Decide whether there is enough usable evidence to estimate `trust_info`. If yes, provide a trust level signal between -1 and 1 through the provided schema.\n"
            "9. Return only the final JSON object. Do not reveal your reasoning.\n\n"
            "# Decision Rules\n"
            "- Return `budget_info` only if the context contains a genuine budget, reservation, or flexibility signal.\n"
            "- If `budget_info` is returned, `estimate` must be a plausible positive SGD value and `confidence` must be greater than `0`.\n"
            "- Never use `budget_info` with `estimate=0`, `confidence=0`, or any zero placeholder to mean \"no signal\".\n"
            "- If the observation only echoes the listing price or current ask, return an empty object `{}` for `budget_info` because that anchor is already captured in the prior.\n"
            "- Do not return the same reservation estimate repeatedly if the context does not provide new evidence. If there is no new evidence to update the reservation estimate, return an empty object `{}` for `budget_info` to indicate that the reservation belief remains unchanged.\n"
            "- If there is no usable budget signal but the context suggests how trustworthy the seller is, return `trust_info`.\n"
            "- If there is neither a usable budget signal nor a trust signal, you can return an empty JSON object `{}`. Do not be coerced into returning a value when there is no evidence.\n"
            "- Do not fabricate private numbers.\n\n"
            "# Output\n"
            "- Return JSON only.\n"
            "- Match the provided schema exactly.\n"
        )

    def _apply_counterpart_reservation_update_response(self, response: str) -> str:
        """Apply one buyer counterpart-belief update response."""
        try:
            info_update = (
                negotiation_schemas.UpdateOpposingBeliefInfo.model_validate_json(
                    response
                )
            )
        except ValidationError:
            return 'Counterpart reservation unchanged: model output was invalid.'
        if info_update.budget_info:
            old_mean = self._beliefs['counterpart_reservation'].get_expected_mean
            self._beliefs['counterpart_reservation'].update_with_evidence(
                info_update.budget_info.estimate,
                info_update.budget_info.confidence
            )
            new_mean = self._beliefs['counterpart_reservation'].get_expected_mean
            return (
                'Counterpart reservation updated from '
                f'{uncertain_helper.format_money(old_mean)} to {uncertain_helper.format_money(new_mean)} '
                f'using estimate={uncertain_helper.format_money(info_update.budget_info.estimate)} '
                f'confidence={info_update.budget_info.confidence:.2f}.'
            )
        if info_update.trust_info:
            self._beliefs['counterpart_reservation'].update_trust(
                info_update.trust_info.trust_level
            )
            return (
                'Counterpart reservation confidence updated via trust signal '
                f'trust_level={info_update.trust_info.trust_level:.2f}.'
            )
        return 'Counterpart reservation unchanged: no budget or trust signal extracted.'
    def _update_own_reservation_from_context(self, context: str) -> str:
        # TODO: refine prompt to include more specific examples of the flat (what the LLM should look out for)
        """Update own reservation belief based on new context information."""
        response = self._model.sample_text(
            self._build_own_reservation_update_prompt(context),
            json_schema=negotiation_schemas.UpdateOwnBeliefInfo.model_json_schema(),
            max_tokens=BUYER_BELIEF_UPDATE_MAX_TOKENS,
        )
        return self._apply_own_reservation_update_response(response)

        
    def _update_counterpart_reservation_from_context(self, context: str) -> str:
        """Update beliefs based on new context information."""
        # TODO: we are going to use the LLM as a black box to extract relevant info and give confidence estimates on whether the given price is driven
        # by market sentiments OR private valuations (e.g. urgency, relationship, etc). 
        # We will update the respective beliefs separately based on the estimates given by the LLM output. 
        response = self._model.sample_text(
            self._build_counterpart_reservation_update_prompt(context),
            json_schema=negotiation_schemas.UpdateOpposingBeliefInfo.model_json_schema(),
            max_tokens=BUYER_BELIEF_UPDATE_MAX_TOKENS,
        )
        return self._apply_counterpart_reservation_update_response(response)

    def _generate_scenarios(self) -> List[uncertain_helper.ScenarioAnalysis]:
        """
        Generate scenarios based on current beliefs.
        We will generate 3 types of scenarios: Optimistic, Pessimistic and Realistic. 
        The expected value for each scenario will be calculated based on sampled beliefs and risk tolerance parameter.
        """
        def _estimate_scenario_likelihoods_from_values(
            scenario_values: Dict[str, float],
            dispersion: float,
            risk_tolerance: float,
        ) -> Dict[str, float]:
            """
            Estimate relative scenario likelihoods from the actual scenario values.

            This is intentionally value-aware rather than a fixed pessimistic/
            realistic/optimistic slider. When the surplus numbers strongly favour
            one scenario, the reported likelihoods should reflect that.
            """
            if not scenario_values:
                return {
                    'Pessimistic': 1 / 3,
                    'Realistic': 1 / 3,
                    'Optimistic': 1 / 3,
                }

            scale = max(float(dispersion), 1.0)
            temperature = max(0.35, 1.15 - (0.8 * risk_tolerance))
            scaled_values = {
                name: value / scale
                for name, value in scenario_values.items()
            }
            max_scaled_value = max(scaled_values.values())
            exp_scores = {
                name: math.exp((value - max_scaled_value) / temperature)
                for name, value in scaled_values.items()
            }
            total_score = sum(exp_scores.values())
            if total_score <= 0.0:
                return {
                    'Pessimistic': 1 / 3,
                    'Realistic': 1 / 3,
                    'Optimistic': 1 / 3,
                }
            return {
                name: score / total_score
                for name, score in exp_scores.items()
            }

        scenarios = []
        cp_belief=self._beliefs['counterpart_reservation']
        # calculate student t parameters for predictive distribution
        df=2*cp_belief.a
        if df <= 2:
            var_cp = 1e9 # cap for now
        else:
            scale=cp_belief.b*(cp_belief.lambda_+1)/(cp_belief.lambda_*cp_belief.a)
            tail_inflation = df / (df - 2)
            var_cp = scale * tail_inflation
        # counterpart's distribution summary statistics
        mu_cp = cp_belief.mu

        # own distribution's summary statistics
        own_reservation=self._beliefs['own_reservation']
        mu_own = own_reservation.get_expected_mean
        var_own = max(1e-9, own_reservation.get_expected_variance)

        # find surplus (own-counterpart)
        mu_diff = mu_own - mu_cp
        sigma_diff = math.sqrt(var_own + var_cp)

        # TODO: we assume independence for now (i.e. covariance = 0) but assumption is weak since we are talking about the same product. 
        # However, it is fine for now, since we assume maximum variance between the differences => more conservative estimates for ZOPA. 
        zopa_dist = NormalDist(mu_diff, sigma_diff)
        confidence_threshold = 1-self._risk_tolerance
        # calculate z critical value for the confidence threshold
        p_upper = 0.5 + (confidence_threshold / 2.0)
        
        z_width = NormalDist(mu=0, sigma=1).inv_cdf(p_upper)
        
        val_optimistic = zopa_dist.mean + (z_width * zopa_dist.stdev)
        val_realistic  = zopa_dist.mean
        val_pessimistic = zopa_dist.mean - (z_width * zopa_dist.stdev)
        scenario_values = {
            'Pessimistic': val_pessimistic,
            'Realistic': val_realistic,
            'Optimistic': val_optimistic,
        }
        likelihoods = _estimate_scenario_likelihoods_from_values(
            scenario_values=scenario_values,
            dispersion=zopa_dist.stdev,
            risk_tolerance=self._risk_tolerance,
        )
        # generate scenarios based on mean and std deviations
        for name, likelihood in likelihoods.items():
            scenario_value = scenario_values[name]
            scenarios.append(
                uncertain_helper.ScenarioAnalysis(
                    scenario_type=name,
                    outcome='Deal Possible' if scenario_value > 0 else 'No Deal',
                    likelihood=likelihood,
                )
            )
        return scenarios
    
    def get_strategy_uncertainty_summary(
        self,
        context: str,
        max_info_items: int = 2,
    ) -> Dict[str, Any]:
        """Build a concise uncertainty summary for the strategy component."""
        strategy_context = uncertain_helper.merge_live_context(
            uncertain_helper.build_strategy_context(
                belief_summary=self._build_belief_summary_for_strategy(),
                recent_memories=uncertain_helper.retrieve_recent_memories(
                    self,
                    self._memory_component_key,
                    self._recent_memory_window,
                ),
                extra_sections=[
                    ('Buyer preferences', self._preferences),
                    ('Flat under negotiation', self._flat_listing),
                ],
            ),
            context,
        )
        scenarios = self._generate_scenarios()
        self._replace_issue_bank(
            uncertain_helper.discover_issues(
                self._model,
                role_description='buyer in an HDB resale negotiation with imperfect information',
                answerer_name='counterpart',
                context=strategy_context,
                issue_bank=self._issue_bank,
                recent_memories=uncertain_helper.retrieve_recent_memories(
                    self,
                    self._memory_component_key,
                    self._recent_memory_window,
                ),
            )
        )
        action_confidence = self._get_action_confidence()
        scenario_summary = uncertain_helper.build_strategy_scenario_summary(scenarios)
        open_issues = uncertain_helper.get_open_issues(self._issue_bank)
        top_issue = open_issues[0] if open_issues else None
        top_issue_score = uncertain_helper.compute_issue_score(top_issue)
        issue_items = [
            uncertain_helper.format_issue_item(issue, include_score=False)
            for issue in open_issues[:max(1, max_info_items)]
        ]

        return {
            'scenario_summary': scenario_summary or 'Unknown',
            'issue_items': issue_items,
            'top_issue_question': (
                top_issue.suggested_question
                if top_issue is not None and top_issue.answerable_now == 1
                else ''
            ),
            'top_issue_score': top_issue_score,
            'action_confidence': action_confidence,
            'risk_tolerance': self._risk_tolerance,
        }
    
    def post_act(self, action_attempt: str) -> str:
        """No-op: uncertainty updates are observation-driven (pre_observe only)."""
        del action_attempt
        return ""

    def pre_observe(self, observation: str) -> str:
        """Record incoming observations and defer LLM belief updates."""
        observation_text = observation.strip()
        if not observation_text:
            uncertain_helper.append_debug_trace(self._debug_trace, 'Skipped empty observation.')
            return ""

        observation_hash = hash(observation_text)
        if self._last_observation_hash == observation_hash:
            uncertain_helper.append_debug_trace(self._debug_trace, 'Skipped duplicate observation.')
            return ""
        self._last_observation_hash = observation_hash

        actor = uncertain_helper.extract_observation_actor(observation_text)
        if actor and actor == self.get_entity().name:
            uncertain_helper.append_debug_trace(
                self._debug_trace,
                'Ignored self-authored observation: '
                f'{uncertain_helper.format_observation_summary(observation_text)}',
            )
            return ""

        uncertain_helper.append_debug_trace(
            self._debug_trace,
            f'Observation considered: {uncertain_helper.format_observation_summary(observation_text)}',
        )
        if uncertain_helper.extract_listing_handoff_state(observation_text) is not None:
            uncertain_helper.append_debug_trace(
                self._debug_trace,
                'Listing handoff observation recorded without duplicate belief update.',
            )
            return ""
        self._pending_observations.append(observation_text)
        uncertain_helper.append_debug_trace(
            self._debug_trace,
            'Queued observation for batched belief update.',
        )
        return ""

    def build_observation_requests(
        self,
    ) -> list[structured_setup_batching.StructuredSetupRequest]:
        requests: list[structured_setup_batching.StructuredSetupRequest] = []
        for index, observation in enumerate(self._pending_observations):
            requests.append(
                structured_setup_batching.StructuredSetupRequest(
                    component=self,
                    response_key=f'own_reservation::{index}',
                    prompt_text=self._build_own_reservation_update_prompt(observation),
                    specific_schema=negotiation_schemas.UpdateOwnBeliefInfo,
                    max_tokens=BUYER_BELIEF_UPDATE_MAX_TOKENS,
                )
            )
            requests.append(
                structured_setup_batching.StructuredSetupRequest(
                    component=self,
                    response_key=f'counterpart_reservation::{index}',
                    prompt_text=self._build_counterpart_reservation_update_prompt(
                        observation
                    ),
                    specific_schema=negotiation_schemas.UpdateOpposingBeliefInfo,
                    max_tokens=BUYER_BELIEF_UPDATE_MAX_TOKENS,
                )
            )
        return requests

    def apply_observation_responses(
        self,
        responses_by_key: Dict[str, str],
    ) -> None:
        for index, _ in enumerate(self._pending_observations):
            uncertain_helper.append_debug_trace(
                self._debug_trace,
                self._apply_own_reservation_update_response(
                    responses_by_key.get(f'own_reservation::{index}', '')
                ),
            )
            uncertain_helper.append_debug_trace(
                self._debug_trace,
                self._apply_counterpart_reservation_update_response(
                    responses_by_key.get(f'counterpart_reservation::{index}', '')
                ),
            )
        self._pending_observations = []

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        belief_states = {}
        for name, belief in self._beliefs.items():
            belief_state: Dict[str, Any] = {
                'class': belief.__class__.__name__,
                'confidence': belief.confidence,
                'evidence_count': belief.evidence_count,
                'last_updated': belief.last_updated,
            }
            if isinstance(belief, uncertain_helper.NormalDistribution):
                belief_state.update({
                    'mean': belief.mean,
                    'std': belief.std,
                })
            elif isinstance(belief, uncertain_helper.NormalInverseGamma):
                belief_state.update({
                    'mu': belief.mu,
                    'lambda_': belief.lambda_,
                    'a': belief.a,
                    'b': belief.b,
                })
            belief_states[name] = belief_state

        action_confidence = self._get_action_confidence()

        return {
            'beliefs': belief_states,
            'own_confidence': self._beliefs['own_reservation'].confidence,
            'counterpart_confidence': self._beliefs['counterpart_reservation'].confidence,
            'issue_bank': [
                issue.model_dump()
                for issue in self._issue_bank
            ],
            'pending_observations': list(self._pending_observations),
            'action_confidence': action_confidence,
            'avg_confidence': action_confidence,
            'uncertainty_level': 1.0 - action_confidence,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        own_belief = self._beliefs['own_reservation']
        own_belief.confidence = max(
            0.0,
            min(1.0, state.get('own_confidence', own_belief.confidence)),
        )
        counterpart_belief = self._beliefs['counterpart_reservation']
        counterpart_belief.confidence = max(
            0.0,
            min(
                1.0,
                state.get('counterpart_confidence', counterpart_belief.confidence),
            ),
        )
        for name, belief_data in state.get('beliefs', {}).items():
            if name in self._beliefs:
                belief = self._beliefs[name]
                belief.confidence = max(
                    0.0,
                    min(1.0, belief_data.get('confidence', belief.confidence)),
                )
                belief.evidence_count = belief_data.get('evidence_count', belief.evidence_count)
                belief.last_updated = belief_data.get('last_updated', belief.last_updated)

                if isinstance(belief, uncertain_helper.NormalDistribution):
                    belief.mean = max(0.0, belief_data.get('mean', belief.mean))
                    belief.std = max(0.0, belief_data.get('std', belief.std))
                elif isinstance(belief, uncertain_helper.NormalInverseGamma):
                    belief.mu = max(0.0, belief_data.get('mu', belief.mu))
                    belief.lambda_ = max(1e-6, belief_data.get('lambda_', belief.lambda_))
                    belief.a = max(1e-6, belief_data.get('a', belief.a))
                    belief.b = max(1e-6, belief_data.get('b', belief.b))
        self._issue_bank = uncertain_helper.restore_issue_bank(
            state.get('issue_bank', []),
        )
        self._pending_observations = [
            str(item) for item in state.get('pending_observations', [])
        ]
        
    def update(self) -> None:
        """Update uncertainty-aware component state."""
        super().update()
