"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from pydantic import BaseModel, Field, ValidationError

class UpdateOwnBeliefInfoMetadata(BaseModel):
    '''Metadata for belief info updates during negotiations.'''
    estimate: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)

class UpdateOwnBeliefInfo(BaseModel):
    '''Information to update belief during negotiations.'''
    reservation_info: Optional[UpdateOwnBeliefInfoMetadata] = Field(None, description="Information about own reservation value")

class UpdateOpposingBeliefInfoMetadata(BaseModel):
    '''Metadata for belief info updates during negotiations.'''
    estimate: float = Field(ge=0.0, description="Estimate of the counterpart's reservation value.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level in the estimate of the counterpart's reservation value.")

class UpdateOpposingBeliefTrustMetadata(BaseModel):
    '''Metadata for trust updates during negotiations.'''
    trust_level: float = Field(ge=0.0, le=1.0, description="Trust level in the counterpart based on the new information")

class UpdateOpposingBeliefInfo(BaseModel):
    '''Information to update belief during negotiations.'''
    budget_info: Optional[UpdateOpposingBeliefInfoMetadata] = None
    trust_info: Optional[UpdateOpposingBeliefTrustMetadata] = None

class UncertainBuyer(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (buyer side)"""

    def __init__(
        self,
        model: Any,
        own_confidence: float = 0.7,
        counterpart_confidence: float = 0.7,
        risk_tolerance: float = 0.8,
        preferences: Optional[dict] = None,
        flat_listing: Optional[dict] = None,
        own_reservation_: float=0.0,
        own_reservation_std: float=1000.0,
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
        self._own_confidence = max(0.0, min(1.0, own_confidence))
        self._counterpart_confidence = max(0.0, min(1.0, counterpart_confidence))
        self._risk_tolerance = risk_tolerance
        self._preferences = preferences or {}
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        self._emit_pre_act_context = emit_pre_act_context
        self._memory_component_key = memory_component_key
        self._recent_memory_window = max(1, int(recent_memory_window))
        self._last_observation_hash: int | None = None
        self._debug_trace: List[str] = []

        # Belief state tracking
        self._beliefs: Dict[
            str,
            uncertain_helper.BeliefDistribution | uncertain_helper.NormalInverseGamma,
        ] = {}
        self._issue_bank: List[uncertain_helper.NegotiationIssue] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(mu, lambda_, a, b, own_reservation_, own_reservation_std)

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
            'Recent uncertainty updates:',
        ]
        if self._debug_trace:
            lines.extend(f'- {entry}' for entry in self._debug_trace)
        else:
            lines.append('- No uncertainty updates recorded yet.')
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

    def _initialize_default_beliefs(self, mu: float = 0.0, lambda_: float = 1.0, a: float = 1.0, b: float = 1.0, own_reservation_: float = 0.0, own_reservation_std: float = 0.0):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = uncertain_helper.NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=max(0.0, mu), #TODO: to determine based on initial logs before initialisation of negotiation
            lambda_=max(1e-6, lambda_),
            a=max(1e-6, a),
            b=max(1e-6, b),
            confidence=self._counterpart_confidence,
        )

        self._beliefs['own_reservation'] = uncertain_helper.BeliefDistribution(
            name='Your Own Reservation Value',
            mean=max(0.0, own_reservation_),
            std=max(0.01, own_reservation_std),
            confidence=self._own_confidence,
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
            f'ScenarioOutlook={self._format_scenario_summary()}',
        ]
        return '\n'.join(lines)

    def _replace_issue_bank(
        self,
        issues: List[uncertain_helper.NegotiationIssue],
    ) -> List[uncertain_helper.NegotiationIssue]:
        self._issue_bank = uncertain_helper.sanitize_issue_bank(issues)
        return list(self._issue_bank)
    def _update_own_reservation_from_context(self, context: str) -> str:
        # TODO: refine prompt to include more specific examples of the flat (what the LLM should look out for)
        """Update own reservation belief based on new context information."""
        prompt = f"""
        You are looking from the perspective of a buyer in a negotiation with imperfect information. The counterpart has full information about their own budget and flexibility.
        Given a context and your own preferences, your task is to extract any relevant information that might affect your own reservation value of the HDB flat (in dollars), if there is any:
        
        Preferences: {self._preferences}
        Context: {context}
        Current Reservation Value: {self._beliefs['own_reservation'].get_expected_mean:.2f}
        Current Confidence Level: {self._beliefs['own_reservation'].confidence:.2f}

        Focus on extracting reservation_info, your own reservation value. Determine the confidence level (0-1) through the amount of trust you have in this information.

        Return a response using the JSON schema provided.
        """

        response = self._model.sample_text(prompt, json_schema=UpdateOwnBeliefInfo.model_json_schema())

        # Ignore malformed model output so one bad response does not crash the turn.
        try:
            info_update = UpdateOwnBeliefInfo.model_validate_json(response)
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

        
    def _update_counterpart_reservation_from_context(self, context: str) -> str:
        """Update beliefs based on new context information."""
        # TODO: we are going to use the LLM as a black box to extract relevant info and give confidence estimates on whether the given price is driven
        # by market sentiments OR private valuations (e.g. urgency, relationship, etc). 
        # We will update the respective beliefs separately based on the estimates given by the LLM output. 
        prompt = f"""
        You are a buyer in a negotiation with imperfect information. The counterpart has full information about their own budget and flexibility.
        Given a context, your task is to extract information concerning the counterpart's budget and flexibility in the negotiation, if there is any:
        
        Context: {context}

        First, focus on extracting BUDGET_INFO, the counterpart's budget or reservation value (in dollars), if any. Determine the confidence level (0-1) through the amount of trust you have in this information.
        Should there be no explicit budget information from the counterpart, focus next on extracting your TRUST in the counterpart based on the context, and determine a trust level (0-1) that reflects how much you trust the counterpart based on the given context. 
        Return a response using the JSON schema provided.
        """

        response = self._model.sample_text(prompt, json_schema=UpdateOpposingBeliefInfo.model_json_schema())

        # Ignore malformed model output so one bad response does not crash the turn.
        try:
            info_update = UpdateOpposingBeliefInfo.model_validate_json(response)
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
        own_reservation=self._beliefs['own_reservation']
        mu_own, var_own = own_reservation.mean, own_reservation.std**2

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
            uncertain_helper.format_issue_item(issue)
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
        uncertain_helper.append_debug_trace(
            self._debug_trace,
            self._update_own_reservation_from_context(observation),
        )
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
            if isinstance(belief, uncertain_helper.BeliefDistribution):
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
            'own_confidence': self._own_confidence,
            'counterpart_confidence': self._counterpart_confidence,
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
        self._own_confidence = max(0.0, min(1.0, state.get('own_confidence', self._own_confidence)))
        self._counterpart_confidence = max(0.0, min(1.0, state.get('counterpart_confidence', self._counterpart_confidence)))
        for name, belief_data in state.get('beliefs', {}).items():
            if name in self._beliefs:
                belief = self._beliefs[name]
                belief.confidence = belief_data.get('confidence', belief.confidence)
                belief.evidence_count = belief_data.get('evidence_count', belief.evidence_count)
                belief.last_updated = belief_data.get('last_updated', belief.last_updated)

                if isinstance(belief, uncertain_helper.BeliefDistribution):
                    belief.mean = belief_data.get('mean', belief.mean)
                    belief.std = max(0.01, belief_data.get('std', belief.std))
                elif isinstance(belief, uncertain_helper.NormalInverseGamma):
                    belief.mu = max(0.0, belief_data.get('mu', belief.mu))
                    belief.lambda_ = max(1e-6, belief_data.get('lambda_', belief.lambda_))
                    belief.a = max(1e-6, belief_data.get('a', belief.a))
                    belief.b = max(1e-6, belief_data.get('b', belief.b))
        self._issue_bank = uncertain_helper.restore_issue_bank(
            state.get('issue_bank', []),
        )
        
    def update(self) -> None:
        """Update uncertainty-aware component state."""
        super().update()
