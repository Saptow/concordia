"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from pydantic import BaseModel, Field, ValidationError

class InitialSellerPairingPriors(BaseModel):
    """Initial seller-side priors to set once a real buyer/listing pair forms."""
    counterpart_confidence: float = Field(ge=0.0, le=1.0)

class UncertainSeller(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (seller's side)"""

    def __init__(
        self,
        model: Any,
        agent_description: str = '',
        own_confidence: float = 1.0,
        counterpart_confidence: float = 0.7,
        risk_tolerance: float = 0.3,
        flat_listing: Optional[dict] = None,
        own_reservation_: float = 0.0, # note that this is the minimum reservation price for the seller
        own_reservation_std: float = 100.0, # this is the uncertainty in the seller's own reservation price (to simulate cases where the seller is not sure about their own reservation price due to factors like uncertain valuation of the flat, emotional attachment, etc.)
        mu: float = 0.0,
        lambda_: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        listing_price_prior_discount: float = 0.9,
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
        super().__init__(pre_act_label='uncertain_seller')
        self._model = model
        self._agent_description = str(agent_description)
        self._risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        # TODO: think whether we need a separate belief for own reservation price (should not be to simulate information asymmetry of the product)
        self._listing_price_prior_discount = max(
            0.0, min(1.0, float(listing_price_prior_discount))
        )
        self._emit_pre_act_context = emit_pre_act_context
        self._memory_component_key = memory_component_key
        self._recent_memory_window = max(1, int(recent_memory_window))
        self._last_observation_hash: int | None = None
        self._debug_trace: List[str] = []

        # Belief state tracking
        self._beliefs: Dict[
            str,
            uncertain_helper.NormalDistribution | uncertain_helper.NormalInverseGamma,
        ] = {}
        self._issue_bank: List[uncertain_helper.NegotiationIssue] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(
            mu,
            lambda_,
            a,
            b,
            own_reservation_,
            own_reservation_std,
            own_confidence,
            counterpart_confidence,
        )

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
            'Perspective=Seller',
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
            'Summary': 'Seller uncertainty state',
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
        listing_payload: listing_schemas.ListingNegotiationTransferPayload,
    ) -> None:
        self._flat_listing = listing_payload.listing_record.flat.model_dump(mode='json')
        listing_price = uncertain_helper.coerce_positive_float(
            listing_payload.listing_record.listing_price
        )
        if listing_price <= 0.0:
            return
        self._listing_price_prior_discount = 1.0
        self._beliefs['counterpart_reservation'].mu = listing_price
        own_reservation = self._beliefs['own_reservation'].mean
        if abs(float(own_reservation) - float(listing_price)) <= 1e-9:
            self._debug_trace.append(
                'Own reservation equals listing price; seller expectation spread is absent.'
            )
        priors = self._calibrate_initial_pairing_priors(listing_payload)
        if priors is not None:
            self._beliefs['counterpart_reservation'].confidence = (
                priors.counterpart_confidence
            )

    def get_effective_reservation_distribution(
        self,
    ) -> uncertain_helper.NormalDistribution:
        return self._beliefs['own_reservation'].model_copy(deep=True)

    def _initialize_default_beliefs(
        self,
        mu: float = 0.0,
        lambda_: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        own_reservation_: float = 0.0,
        own_reservation_std: float = 1.0,
        own_confidence: float = 1.0,
        counterpart_confidence: float = 0.7,
    ):
        """Initialize default beliefs about negotiation parameters."""
        self._beliefs['own_reservation'] = uncertain_helper.NormalDistribution(
            name='Your Own Reservation Value',
            mean=max(0.0, own_reservation_),
            std=max(0.0, own_reservation_std),
            confidence=max(0.0, min(1.0, own_confidence)),
        )
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = uncertain_helper.NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=max(0.0, mu), #TODO: to determine based on initial logs before initialisation of negotiation
            lambda_=max(1e-6, lambda_),
            a=max(1e-6, a),
            b=max(1e-6, b),
            confidence=max(0.0, min(1.0, counterpart_confidence)),
        )

    def _calibrate_initial_pairing_priors(
        self,
        listing_payload: listing_schemas.ListingNegotiationTransferPayload,
    ) -> Optional[InitialSellerPairingPriors]:
        prompt = (
            "# Role\n"
            "You are calibrating initial uncertainty priors for a seller who has "
            "just been paired to a buyer for an HDB resale negotiation.\n\n"
            "# Task\n"
            "Estimate counterpart_confidence between 0 and 1: how confident the "
            "seller is in their initial estimate of the buyer's reservation value.\n\n"
            "# Input\n"
            "## Seller Description / Persona\n"
            f"{self._agent_description}\n\n"
            "## Listing Context\n"
            f"{listing_payload.listing_record.model_dump(mode='json')}\n\n"
            "## Seller State\n"
            f"{listing_payload.seller_state.model_dump(mode='json')}\n\n"
            "# Rubric\n"
            "- 0.20-0.40: seller persona seems unsure, reactive, or inexperienced in reading buyers.\n"
            "- 0.45-0.65: seller persona seems somewhat informed but still uncertain about buyer limits.\n"
            "- 0.70-0.90: seller persona seems experienced, assertive, market-aware, or confident in reading counterpart signals.\n\n"
            "# Few-Shot Examples\n"
            "Example 1:\n"
            "- Seller persona: seasoned, pragmatic, knows the area well, comfortable pricing strategically.\n"
            "- Interpretation: likely more confident in estimating what buyers can pay.\n"
            '- Output: {"counterpart_confidence": 0.78}\n\n'
            "Example 2:\n"
            "- Seller persona: emotionally attached, conflicted about selling, not clearly market-savvy.\n"
            "- Interpretation: likely less confident in estimating buyer reservation value.\n"
            '- Output: {"counterpart_confidence": 0.36}\n\n'
            "# Rules\n"
            "- Return JSON only.\n"
            "- Keep the value within [0, 1].\n"
            "- Base the estimate primarily on the seller persona.\n"
            "- Higher values should reflect personas that seem experienced, assertive, "
            "or market-aware.\n"
            "- Use the examples as anchors, not as fixed templates.\n"
        )

        try:
            response = self._model.sample_text(
                prompt,
                json_schema=InitialSellerPairingPriors.model_json_schema(),
                max_tokens=120,
            )
            priors = InitialSellerPairingPriors.model_validate_json(response)
        except ValidationError:
            return None
        except Exception:
            return None
        return InitialSellerPairingPriors(
            counterpart_confidence=max(
                0.0,
                min(1.0, priors.counterpart_confidence),
            ),
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

    def _update_counterpart_reservation_from_context(self, context: str) -> str:
        """Update beliefs based on new context information."""
        prompt = (
            "# Role\n"
            "You are a seller in an HDB resale negotiation.\n\n"
            "# Task\n"
            "Given an observation, infer the following:\n"
            "- **buyer's likely reservation value**, with a confidence level.\n"
            "- (ONLY if there is no usable signal on reservation value) **trust level signal** on the buyer "
            "based on the new information `[-1, 1]`, where `-1` indicates negative trust and `1` indicates positive trust.\n\n"
            "# Inputs\n"
            "## Observation\n"
            f"{context}\n\n"
            "## Current Belief about Buyer's Reservation Value\n"
            f"- Reservation value estimate: {self._beliefs['counterpart_reservation'].get_expected_mean:.2f}\n"
            f"- Confidence in reservation value: `{self._beliefs['counterpart_reservation'].confidence:.2f}`\n\n"
            "# Private Reasoning Process\n"
            "Think step by step **privately** before answering:\n\n"
            "1. Extract **ONLY** concrete evidence about the buyer's maximum acceptable price.\n"
            "2. Ignore any information that is not directly related to the buyer's reservation value.\n"
            "3. Treat offers, counters, stated affordability, urgency, willingness to compromise, and financing constraints as evidence, "
            "but do not assume any single number is automatically the true reservation value.\n"
            "4. Decide whether there is enough usable evidence to estimate `budget_info`.\n"
            "5. If yes, provide your best estimate of the buyer's reservation value and a confidence level `(0-1)` for that estimate based on the strength of the evidence.\n"
            "6. (IMPORTANT) If there is no available estimate on the buyer's reservation value, extract **ANY INFORMATION** about the trustworthiness of the buyer based on the given observation.\n"
            "   - For example, if you observe that the buyer is intentionally hiding information, being evasive, or providing inconsistent signals, that would be a negative trust signal.\n"
            "   - For example, if you observe the buyer being transparent, giving reasonable justifications, or showing willingness to find a mutually beneficial deal, that would be a positive trust signal.\n"
            "7. Decide whether there is enough usable evidence to estimate `trust_info`. If yes, provide a trust level signal between `-1` and `1` through the provided schema.\n"
            "8. Return only the final JSON object. Do not reveal your reasoning.\n\n"
            "# Decision Rules\n"
            "- Return `budget_info` only if the context contains a genuine budget, reservation, or flexibility signal.\n"
            "- If `budget_info` is returned, `estimate` must be a plausible positive SGD value and `confidence` must be greater than `0`.\n"
            "- Never use `budget_info` with `estimate=0`, `confidence=0`, or any zero placeholder to mean \"no signal\".\n"
            "- Do not return the same reservation estimate repeatedly if the context does not provide new evidence. If there is no new evidence to update the reservation estimate, return an empty object `{}` for `budget_info` to indicate that the reservation belief remains unchanged.\n"
            "- If there is no usable budget signal but the context suggests how trustworthy the buyer is, return `trust_info`.\n"
            "- If there is neither a usable budget signal nor a trust signal, return `{}`.\n"
            "- Do not fabricate hidden motives or private numbers.\n\n"
            "# Output\n"
            "- Return JSON only.\n"
            "- Match the provided schema exactly.\n"
        )
        response = self._model.sample_text(
            prompt,
            json_schema=negotiation_schemas.UpdateOpposingBeliefInfo.model_json_schema(),
        )

        # Ignore malformed model output so one bad response does not crash the turn.
        try:
            info_update = negotiation_schemas.UpdateOpposingBeliefInfo.model_validate_json(response)
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
        elif info_update.trust_info: # if there is no explicit budget info, we update our beliefs based on the trust level of the counterpart instead.
            self._beliefs['counterpart_reservation'].update_trust(
                info_update.trust_info.trust_level
            )
            return (
                'Counterpart reservation confidence updated via trust signal '
                f'trust_level={info_update.trust_info.trust_level:.2f}.'
            )
        return 'Counterpart reservation unchanged: no budget or trust signal extracted.'

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
        own_reservation = self._beliefs['own_reservation']
        mu_own, var_own = own_reservation.mean, own_reservation.std**2

        # find surplus; for the seller, its 
        mu_diff = mu_cp - mu_own

        # TODO: we assume independence for now (i.e. covariance = 0) but assumption is weak since we are talking about the same product. 
        # However, it is fine for now, since we assume maximum variance between the differences => more conservative estimates for ZOPA. 
        zopa_dist = NormalDist(mu_diff, math.sqrt(var_cp + var_own))
        p_upper = max(1e-6, min(1.0 - 1e-6, 0.5 + ((1 - self._risk_tolerance) / 2.0)))
        
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
                    ('Flat under negotiation', self._flat_listing),
                ],
            ),
            context,
        )
        scenarios = self._generate_scenarios()
        self._replace_issue_bank(
            uncertain_helper.discover_issues(
                self._model,
                role_description='seller in an HDB resale negotiation',
                answerer_name='buyer',
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
            for issue in open_issues
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
        """Process incoming observation text to update beliefs."""
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
        uncertain_helper.append_debug_trace(
            self._debug_trace,
            self._update_counterpart_reservation_from_context(observation),
        )
        return ""

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
            if isinstance(belief, uncertain_helper.NormalInverseGamma):
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
            'own_reservation': self._beliefs['own_reservation'].mean,
            'own_confidence': self._beliefs['own_reservation'].confidence,
            'counterpart_confidence': self._beliefs['counterpart_reservation'].confidence,
            'listing_price_prior_discount': self._listing_price_prior_discount,
            'issue_bank': [
                issue.model_dump()
                for issue in self._issue_bank
            ],
            'action_confidence': action_confidence,
            'avg_confidence': action_confidence,
            'uncertainty_level': 1.0 - action_confidence,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        self._listing_price_prior_discount = max(
            0.0,
            min(
                1.0,
                float(
                    state.get(
                        'listing_price_prior_discount',
                        self._listing_price_prior_discount,
                    )
                ),
            ),
        )
        own_belief = self._beliefs['own_reservation']
        own_belief.mean = max(0.0, state.get('own_reservation', own_belief.mean))
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
                if isinstance(belief, uncertain_helper.NormalInverseGamma):
                    belief.mu = max(0.0, belief_data.get('mu', belief.mu))
                    belief.lambda_ = max(1e-6, belief_data.get('lambda_', belief.lambda_))
                    belief.a = max(1e-6, belief_data.get('a', belief.a))
                    belief.b = max(1e-6, belief_data.get('b', belief.b))
        self._issue_bank = uncertain_helper.restore_issue_bank(state.get('issue_bank', []))


    def update(self) -> None:
        """Update uncertainty-aware component state."""
        super().update()
