"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import dataclasses
import json
import math
import random
from statistics import NormalDist
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from pydantic import BaseModel, ValidationError, Field

# To model pricing distribution beliefs
@dataclasses.dataclass
class NormalInverseGamma:
    '''Represents a Normal-Inverse-Gamma conjugate prior for Bayesian updating.'''
    name: str 
    mu: float  
    lambda_: float
    a: float
    b: float
    confidence: float = 0.5  # Initial confidence level (0-1)
    evidence_count: int = 0  # Number of observations supporting this belief
    last_updated: Optional[str] = None

    # Helpers
    def _get_t_critical(self, confidence: float, df: float) -> float:
        """
        Approximates the t-critical value without scipy.
        Uses a lookup table for small df and Z-score for large df.
        Confidence should only be 0.90, 0.95, or 0.99.
        """
        t_table_90 = {
            1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
            6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
            12: 1.782, 14: 1.761, 16: 1.746, 18: 1.734, 20: 1.725,
            25: 1.708, 30: 1.697
        }
        t_table_95 = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            12: 2.179, 14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086,
            25: 2.060, 30: 2.042
        }
        t_table_99 = {
            1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
            6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
            12: 3.055, 14: 2.977, 16: 2.921, 18: 2.878, 20: 2.845,
            25: 2.787, 30: 2.750
        }

        if confidence == 0.90:
            t_table = t_table_90
            if df > 30:
                return 1.645  # use normal approximation
        elif confidence == 0.95:
            t_table = t_table_95
            if df > 30:
                return 1.96  # use normal approximation
        elif confidence == 0.99:
            t_table = t_table_99
            if df > 30:
                return 2.576  # use normal approximation
        
        # Find the closest key in the table for small df
        closest_df = max(k for k in t_table if k <= df)
        return t_table[closest_df]

    @property
    def get_expected_mean(self) -> float: 
        '''Calculate the expected mean of the distribution.'''
        return self.mu 

    @property
    def get_expected_variance(self) -> float:
        '''Calculate the expected variance of the distribution.'''
        if self.a > 1:
            return self.b / (self.a - 1)
        else:
            return self.b 
    
    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        '''Calculate the predictive mean range for a given confidence level.'''
        confidence = max(0.90, min(0.99, confidence))
        df = 2 * self.a
        scale = np.sqrt(self.b * (self.lambda_ + 1) / (self.lambda_ * self.a))

        t_value = self._get_t_critical(confidence, df)
        margin = t_value * scale

        lower = max(0.0, self.mu - margin)
        upper = max(lower, self.mu + margin)
        return (lower, upper)
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        '''Sample from the Normal-Inverse-Gamma distribution.'''
        rng=np.random.default_rng()
        tau_sq_samples = 1 / rng.gamma(self.a, 1 / self.b, n)# first sample from inverse gamma distribution
        mu_samples=rng.normal(self.mu, np.sqrt(1 / (self.lambda_ * tau_sq_samples)), n) # then sample from normal distribution
        return mu_samples[0] if n == 1 else mu_samples.tolist()
    
    def update_with_evidence(self, observation: float, reliability: float = 1.0):
        '''Update the distribution with new evidence using standard Bayesian update.'''
        reliability = max(0.0, min(1.0, reliability))
        observation = max(0.0, observation)

        # Update mean and lambda
        new_lambda = self.lambda_ + reliability
        new_mu = (self.lambda_ * self.mu + reliability * observation) / new_lambda

        # Update variance parameters(a and b)
        new_a = self.a + 0.5 * reliability # 1 observation
        diff = observation - self.mu
        new_b = self.b + (self.lambda_ * reliability * (diff **2)) / (2 * new_lambda)

        # update confidence 
        # TODO: refine confidence update logic
        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.05 * reliability)


        # Assign updated parameters
        self.mu = new_mu
        self.lambda_ = new_lambda
        self.a = new_a
        self.b = new_b


    def update_trust(self, trust_level: float, scale: float = 7.0):
        '''
        Update confidence based on trust level of new information. Note that scale is chosen at 7.0 since timestep is in terms of weeks.
        '''
        w = max(0.0, min(1.0, float(trust_level)))
        if w <= 0.0:
            return

        m_eff = scale * w  # >= 0

        # Current expected variance (safe)
        ev = max(1e-9, self.get_expected_variance)

        # Update a (shape)
        a_new = self.a + 0.5 * m_eff

        # Adjust b to keep E[sigma^2] = b/(a-1) constant
        b_new = ev * max(1e-9, (a_new - 1.0))

        self.a = a_new
        self.b = b_new

        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.02 * (1 - math.exp(-m_eff / 5.0)))
        
@dataclasses.dataclass
class BeliefDistribution:
    """Represents a belief about a parameter with uncertainty."""
    name: str
    mean: float
    std: float
    confidence: float  # 0-1, how confident we are in this estimate
    evidence_count: int = 0  # Number of observations supporting this belief
    last_updated: Optional[str] = None

    @property
    def get_expected_mean(self) -> float:
        """Get the expected mean of the belief."""
        return self.mean
    @property
    def get_expected_variance(self) -> float:
        """Get the expected variance of the belief."""
        return self.std ** 2
    
    def sample(self, n: int = 1) -> Union[float, List[float]]:
        """Sample from the belief distribution."""
        samples = np.random.normal(self.mean, self.std, n)
        return samples[0] if n == 1 else samples.tolist()

    def update_with_evidence(self, observation: float, reliability: float = 1.0):
        """Bayesian update with new evidence."""
        reliability = max(0.0, min(1.0, reliability))
        observation = max(0.0, observation)
        std = max(0.01, self.std)

        # Simple Bayesian updating assuming normal distributions (both prior and likelihood are normal)
        prior_precision = 1 / (std ** 2)
        evidence_precision = reliability / (std ** 2)  # Reliability affects precision

        # Update mean (weighted average)
        total_precision = prior_precision + evidence_precision
        new_mean = (prior_precision * self.mean + evidence_precision * observation) / total_precision

        # Update standard deviation (precision increases)
        new_std = 1 / math.sqrt(total_precision)

        # Update confidence based on evidence accumulation
        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.05 * reliability)

        self.mean = max(0.0, new_mean)
        self.std = max(0.01, new_std)  # Prevent std from becoming too small
    
        
    def get_confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for the belief."""
        z_score = 1.96 if level == 0.95 else 2.58  # 95% or 99%
        margin = z_score * self.std
        lower = max(0.0, self.mean - margin)
        upper = max(lower, self.mean + margin)
        return (lower, upper)


# Dataclasses for structured analysis outputs
class UncertainInfo(BaseModel):
    '''Schema for uncertain information.'''
    claim: str
    confidence_level: float # 0-1

class UncertaintyContext(BaseModel):
    '''Schema for uncertainty contexts.'''
    missing_info: List[str]
    uncertainty_sources: List[UncertainInfo]
    valuable_info: List[str]
    scenarios: List[str]

@dataclasses.dataclass
class ScenarioAnalysis:
    """Analysis of different negotiation scenarios under uncertainty."""
    scenario_type: Literal['Pessimistic', 'Realistic', 'Optimistic']
    outcome: Literal['Deal Possible', 'No Deal']
    likelihood: float


class InformationValue(BaseModel):
    """Value of gathering specific information."""
    question: str = Field(description="The specific question to ask to gather information")
    priority_score: float = Field(ge=0, le=1, description="The degree of reduction in uncertainty (0-1); higher means more valuable")
    cost_factor: float = Field(ge=0, le=1, description="Relative cost of obtaining this information (0-1); higher means more costly")

class InformationValueResponse(BaseModel):
    information_values: Optional[List[InformationValue]] = None

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

class UncertainBuyer(action_spec_ignored.ActionSpecIgnored):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (buyer side)"""
    _PLACEHOLDER_INFO_QUESTION_TOKENS = (
        'question text',
        'placeholder',
        'your question here',
        'example question',
        'insert question',
    )

    def __init__(
        self,
        model: Any,
        confidence: float = 0.7,
        risk_tolerance: float = 0.8,
        preferences: Optional[dict] = None,
        flat_listing: Optional[dict] = None,
        information_gathering_budget: float = 0.5, # fraction of value to spend on information gathering
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
            confidence: Prior confidence in beliefs (0-1) # TODO: to determine based on personality metadata; THIS IS NOT TO BE UPDATED DURING THE NEGOTIATION.
            risk_tolerance: Tolerance for uncertainty (0-1)
            information_gathering_budget: Fraction of value to spend on info gathering
        """
        super().__init__(pre_act_label='uncertain_buyer')
        self._model = model
        self._confidence = confidence
        self._risk_tolerance = risk_tolerance #TODO: to determine based on personality metadata; THIS IS NOT TO BE UPDATED DURING THE NEGOTIATION.
        self._preferences = preferences or {}
        self._flat_listing = dict(flat_listing) if flat_listing else {}
        self._info_budget = information_gathering_budget
        self._emit_pre_act_context = emit_pre_act_context
        self._memory_component_key = memory_component_key
        self._recent_memory_window = max(1, int(recent_memory_window))
        self._last_observation_hash: int | None = None

        # Belief state tracking
        self._beliefs: Dict[str, BeliefDistribution | NormalInverseGamma] = {}
        self._uncertainty_sources: List[str] = []
        self._information_gaps: List[str] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(mu, lambda_, a, b, own_reservation_, own_reservation_std)

    @staticmethod
    def _format_money(value: float) -> str:
        return f'{float(value):.2f}'

    @classmethod
    def _format_interval(cls, interval: Tuple[float, float]) -> str:
        return f'{cls._format_money(interval[0])}-{cls._format_money(interval[1])}'

    def _format_scenario_summary(self) -> str:
        scenarios = self._generate_scenarios()
        if not scenarios:
            return 'Unknown'
        return self._build_strategy_scenario_summary(scenarios)

    @staticmethod
    def _normalize_scenario_outcome(outcome: str) -> str:
        return str(outcome).strip().lower()

    @classmethod
    def _build_main_deal_summary(
        cls,
        scenarios: List[ScenarioAnalysis],
    ) -> str:
        if not scenarios:
            return 'Overall assessment unknown.'

        scenario_lookup = {
            scenario.scenario_type: scenario for scenario in scenarios
        }
        realistic = scenario_lookup.get('Realistic')
        pessimistic = scenario_lookup.get('Pessimistic')
        optimistic = scenario_lookup.get('Optimistic')

        realistic_outcome = (
            cls._normalize_scenario_outcome(realistic.outcome)
            if realistic is not None
            else 'unknown'
        )
        pessimistic_outcome = (
            cls._normalize_scenario_outcome(pessimistic.outcome)
            if pessimistic is not None
            else 'unknown'
        )
        optimistic_outcome = (
            cls._normalize_scenario_outcome(optimistic.outcome)
            if optimistic is not None
            else 'unknown'
        )

        deal_possible_probability = sum(
            float(scenario.likelihood)
            for scenario in scenarios
            if cls._normalize_scenario_outcome(scenario.outcome) == 'deal possible'
        )

        if realistic_outcome == 'deal possible':
            if pessimistic_outcome == 'no deal':
                return (
                    'Overall, a deal is possible but uncertain '
                    f'({deal_possible_probability:.0%} weighted chance).'
                )
            return (
                'Overall, a deal is likely possible '
                f'({deal_possible_probability:.0%} weighted chance).'
            )

        if realistic_outcome == 'no deal':
            if optimistic_outcome == 'deal possible':
                return (
                    'Overall, a deal looks difficult but not impossible '
                    f'({deal_possible_probability:.0%} weighted chance).'
                )
            return (
                'Overall, a deal is unlikely '
                f'({deal_possible_probability:.0%} weighted chance).'
            )

        if deal_possible_probability >= 0.67:
            return (
                'Overall, a deal is likely possible '
                f'({deal_possible_probability:.0%} weighted chance).'
            )
        if deal_possible_probability <= 0.33:
            return (
                'Overall, a deal is unlikely '
                f'({deal_possible_probability:.0%} weighted chance).'
            )
        return (
            'Overall, the deal outlook is mixed '
            f'({deal_possible_probability:.0%} weighted chance).'
        )

    @classmethod
    def _build_strategy_scenario_summary(
        cls,
        scenarios: List[ScenarioAnalysis],
    ) -> str:
        if not scenarios:
            return 'Unknown'

        scenario_lookup = {
            scenario.scenario_type: scenario for scenario in scenarios
        }

        def _format_case(
            case_label: str,
            scenario_name: str,
        ) -> str:
            scenario = scenario_lookup.get(scenario_name)
            if scenario is None:
                return f'{case_label}: unknown.'
            return (
                f'{case_label}: {scenario.outcome.lower()} '
                f'({scenario.likelihood:.0%}).'
            )

        return ' '.join([
            cls._build_main_deal_summary(scenarios),
            _format_case('Base case', 'Realistic'),
            _format_case('Downside case', 'Pessimistic'),
            _format_case('Upside case', 'Optimistic'),
        ])

    def _make_pre_act_value(self) -> str:
        own_reservation = self._beliefs['own_reservation']
        counterpart_reservation = self._beliefs['counterpart_reservation']
        belief_confidences = [belief.confidence for belief in self._beliefs.values()]
        avg_confidence = float(np.mean(belief_confidences)) if belief_confidences else 0.0

        lines = [
            'Perspective=Buyer',
            f'OwnReservationMean={self._format_money(own_reservation.get_expected_mean)}',
            f'OwnReservationCI95={self._format_interval(own_reservation.get_confidence_interval())}',
            f'OwnReservationConfidence={own_reservation.confidence:.2f}',
            f'CounterpartReservationMean={self._format_money(counterpart_reservation.get_expected_mean)}',
            f'CounterpartReservationCI95={self._format_interval(counterpart_reservation.get_confidence_interval())}',
            f'CounterpartReservationConfidence={counterpart_reservation.confidence:.2f}',
            f'InformationGatheringBudget={self._info_budget:.2f}',
            f'RiskTolerance={self._risk_tolerance:.2f}',
            f'AverageBeliefConfidence={avg_confidence:.2f}',
            f'ScenarioOutlook={self._format_scenario_summary()}',
        ]
        return '\n'.join(lines)

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        if self._emit_pre_act_context:
            return super().pre_act(action_spec)
        del action_spec
        _ = self.get_pre_act_value()
        return ""

    @staticmethod
    def _extract_observation_actor(observation: str) -> str | None:
        """Extract actor name from observation text like '[observation] Alice: ...'."""
        text = observation.strip()
        if text.startswith('[observation]'):
            text = text[len('[observation]'):].strip()
        actor, sep, _ = text.partition(':')
        if not sep:
            return None
        actor = actor.strip()
        return actor if actor else None
    
    def _initialize_default_beliefs(self, mu: float = 0.0, lambda_: float = 1.0, a: float = 1.0, b: float = 1.0, own_reservation_: float = 0.0, own_reservation_std: float = 0.0):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=max(0.0, mu), #TODO: to determine based on initial logs before initialisation of negotiation
            lambda_=max(1e-6, lambda_),
            a=max(1e-6, a),
            b=max(1e-6, b)
        )

        self._beliefs['own_reservation'] = BeliefDistribution(
            name='Your Own Reservation Value',
            mean=max(0.0, own_reservation_),
            std=max(0.01, own_reservation_std),
            confidence=self._confidence
        )

        # TODO: consider if we are able to segment privsate vs market driven components
        # self._beliefs['market_conditions'] = BeliefDistribution(
        #     name=""
        # )

    def get_information_gathering_budget(self) -> float:
        return max(0.0, float(self._info_budget))

    def _build_belief_summary_for_strategy(self) -> str:
        own_reservation = self._beliefs['own_reservation']
        counterpart_reservation = self._beliefs['counterpart_reservation']
        lines = [
            f'OwnReservationMean={self._format_money(own_reservation.get_expected_mean)}',
            f'OwnReservationCI95={self._format_interval(own_reservation.get_confidence_interval())}',
            f'OwnReservationConfidence={own_reservation.confidence:.2f}',
            f'CounterpartReservationMean={self._format_money(counterpart_reservation.get_expected_mean)}',
            f'CounterpartReservationCI95={self._format_interval(counterpart_reservation.get_confidence_interval())}',
            f'CounterpartReservationConfidence={counterpart_reservation.confidence:.2f}',
            f'RiskTolerance={self._risk_tolerance:.2f}',
            f'InformationGatheringBudget={self._info_budget:.2f}',
            f'ScenarioOutlook={self._format_scenario_summary()}',
        ]
        return '\n'.join(lines)

    def _retrieve_recent_memories(self) -> List[str]:
        try:
            memory = self.get_entity().get_component(
                self._memory_component_key,
                type_=memory_component.Memory,
            )
            return list(memory.retrieve_recent(limit=self._recent_memory_window))
        except Exception:
            return []

    def _build_strategy_context(self) -> str:
        recent_memories = self._retrieve_recent_memories()
        memory_lines = [
            f'- {memory_text}'
            for memory_text in recent_memories
        ] or ['- None']
        listing_context = (
            f'Flat under negotiation:\n{json.dumps(self._flat_listing, ensure_ascii=False)}\n'
            if self._flat_listing
            else ''
        )
        return (
            'Current belief summary:\n'
            f'{self._build_belief_summary_for_strategy()}\n'
            f'{listing_context}'
            'Recent negotiation memories:\n'
            + '\n'.join(memory_lines)
        )

    @staticmethod
    def _merge_live_context(
        strategy_context: str,
        context: str,
    ) -> str:
        live_context = str(context).strip()
        if not live_context:
            return strategy_context
        return (
            f'{strategy_context}\n'
            'Current action context:\n'
            f'{live_context}'
        )

    @classmethod
    def _normalize_information_question(cls, question: str) -> str:
        normalized = ' '.join(str(question).split()).strip().strip('"').strip("'")
        return normalized

    @classmethod
    def _is_placeholder_information_value(cls, info: InformationValue) -> bool:
        question = cls._normalize_information_question(info.question)
        lowered = question.lower().rstrip('.')
        if not question or len(question) < 8:
            return True
        if any(token in lowered for token in cls._PLACEHOLDER_INFO_QUESTION_TOKENS):
            return True
        if lowered in {'question', 'string', 'text', 'n/a', 'none'}:
            return True
        if info.priority_score <= 0.0 and info.cost_factor <= 0.0:
            return True
        return False

    @classmethod
    def _sanitize_information_values(
        cls,
        info_values: List[InformationValue],
    ) -> List[InformationValue]:
        sanitized: List[InformationValue] = []
        seen_questions: set[str] = set()
        for info in info_values:
            if cls._is_placeholder_information_value(info):
                continue
            normalized_question = cls._normalize_information_question(info.question)
            dedupe_key = normalized_question.lower().rstrip('?.!')
            if dedupe_key in seen_questions:
                continue
            seen_questions.add(dedupe_key)
            info.question = normalized_question
            sanitized.append(info)
        return sanitized

    @staticmethod
    def _format_information_item(info: InformationValue) -> str:
        question = info.question
        if len(question) > 90:
            question = question[:87].rstrip() + '...'
        return f'"{question}" [P={info.priority_score:.2f}, C={info.cost_factor:.2f}]'

    def _analyze_uncertainty_context(self, context: str) -> UncertaintyContext:
        """Analyze context for uncertainty indicators and information gaps."""
        formatted_beliefs="\n".join(
            [f"{name}: mean={belief.get_expected_mean:.2f}, std={math.sqrt(belief.get_expected_variance):.2f}" for name, belief in self._beliefs.items()] # TODO: might need to add confidence for NIG, but decide later on
        )
        formatted_preferences = "\n".join(
            [f"{key}: {value}" for key, value in self._preferences.items()]
        )
        prompt = f"""
        You are a buyer in an HDB resale negotiation with imperfect information.
        Use the full context below, including any listing details, recent memories, and current action context.
        Focus on uncertainties that materially affect whether a deal is possible, what price range is plausible, and what questions would reduce uncertainty most efficiently.

        Current beliefs:
        {formatted_beliefs}

        Buyer preferences:
        {formatted_preferences}

        Negotiation context:
        {context}

        Produce a structured uncertainty analysis:
        1. missing_info: the most decision-relevant facts that are still unknown.
        2. uncertainty_sources: concrete uncertain claims together with confidence_level in [0,1], where lower confidence means more uncertainty.
        3. valuable_info: short descriptions of information that would materially improve pricing or deal-feasibility judgment.
        4. scenarios: concise scenario labels or summaries worth considering.

        Prioritize information tied to reservation price, flexibility, urgency, completion timeline, flat condition, financing readiness, and any listing attributes that affect willingness to pay.
        Do not repeat obvious facts already established in the context unless they are still genuinely uncertain.
        Return only JSON that matches the provided schema.
        """

        response = self._model.sample_text(prompt, json_schema=UncertaintyContext.model_json_schema())

        # Load JSON response and validate schema
        try:
            analysis = UncertaintyContext.model_validate_json(response)
        except ValidationError as e:
            # current fallback TODO: improve error handling
            analysis = UncertaintyContext(
                missing_info=[],
                uncertainty_sources=[],
                valuable_info=[],
                scenarios=[]
            )

        return analysis
    def _update_own_reservation_from_context(self, context: str):
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
            return
        if info_update.reservation_info:
            self._beliefs['own_reservation'].update_with_evidence(
                info_update.reservation_info.estimate,
                info_update.reservation_info.confidence
            )

        
    def _update_counterpart_reservation_from_context(self, context: str):
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
            return
        if info_update.budget_info:
            self._beliefs['counterpart_reservation'].update_with_evidence(
                info_update.budget_info.estimate,
                info_update.budget_info.confidence
            )
        elif info_update.trust_info: # if there is no explicit budget info, we update our beliefs based on the trust level of the counterpart instead.
            self._beliefs['counterpart_reservation'].update_trust(
                info_update.trust_info.trust_level
            )

    def _generate_scenarios(self) -> List[ScenarioAnalysis]:
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
                ScenarioAnalysis(
                    scenario_type=name,
                    outcome='Deal Possible' if scenario_value > 0 else 'No Deal',
                    likelihood=likelihood,
                )
            )
        return scenarios
    
    def _calculate_information_values(self, context: str, uncertainty_analysis: UncertaintyContext) -> List[InformationValue]:
        """Calculate value of gathering different types of information."""
        # Current uncertainty level (higher uncertainty = more value from information)

        # Determine the info_opportunities using a LLM. 
        prompt = f"""
        You are a buyer in an HDB resale negotiation with imperfect information.
        Given the negotiation context and the uncertainty analysis below, propose up to 10 specific questions that the buyer could realistically ask the counterpart next.

        Context:
        {context}

        Uncertainty Analysis:
        {uncertainty_analysis.model_dump_json()}
        
        Each proposed question should:
        - be answerable by the counterpart in conversation,
        - reduce uncertainty about reservation price, flexibility, urgency, timing, condition, inclusions, or transaction feasibility,
        - avoid duplicating questions that the context already answers,
        - be phrased as a natural negotiation question, not as analysis notes.

        For each question, estimate:
        1. priority_score (0-1): how much it could improve decision quality or confidence.
        2. cost_factor (0-1): how costly it is in negotiation terms, such as being awkward, tedious, or unlikely to elicit a useful answer.

        Prefer high-value, low-cost questions.
        Return only JSON matching the provided schema.
        Do not output placeholders such as "question text..." or dummy zero-score rows.
        If there are no useful questions, return an empty list.
        """

        response = self._model.sample_text(prompt, json_schema=InformationValueResponse.model_json_schema())
        # Load JSON response and validate schema
        try:
            info_values_response = InformationValueResponse.model_validate_json(response)
            info_opportunities = info_values_response.information_values or []
        except ValidationError as e:
            # current fallback TODO: improve error handling
            info_opportunities = []

        info_opportunities = self._sanitize_information_values(info_opportunities)
        # Sort by rank
        info_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        return info_opportunities

    def _select_budgeted_information_values(
        self,
        info_values: List[InformationValue],
        budget: float | None = None,
        limit: int = 3,
    ) -> List[InformationValue]:
        """Return top information opportunities that fit the current budget."""
        selected: List[InformationValue] = []
        frac_of_budget = max(
            0.0,
            self._info_budget if budget is None else float(budget),
        )
        for info in info_values[:limit]:
            frac_of_budget -= info.cost_factor
            if frac_of_budget > 0:
                selected.append(info)
        return selected

    def _calculate_average_uncertainty(
        self,
        uncertainty_analysis: UncertaintyContext,
    ) -> float:
        """Compute average uncertainty level for current beliefs/context."""
        uncertainty_levels = [
            1 - info.confidence_level
            for info in uncertainty_analysis.uncertainty_sources
        ]
        if uncertainty_levels:
            return float(np.mean(uncertainty_levels))

        belief_confidences = [belief.confidence for belief in self._beliefs.values()]
        if not belief_confidences:
            return 1.0
        return 1.0 - float(np.mean(belief_confidences))

    def get_strategy_uncertainty_summary(
        self,
        context: str,
        max_info_items: int = 2,
        allowed_info_budget: float | None = None,
    ) -> Dict[str, Any]:
        """Build a concise uncertainty summary for the strategy component."""
        strategy_context = self._merge_live_context(
            self._build_strategy_context(),
            context,
        )
        uncertainty_analysis = self._analyze_uncertainty_context(strategy_context)
        scenarios = self._generate_scenarios()
        info_values = self._calculate_information_values(
            strategy_context,
            uncertainty_analysis,
        )
        budgeted_info_values = self._select_budgeted_information_values(
            info_values,
            budget=allowed_info_budget,
        )
        avg_uncertainty = self._calculate_average_uncertainty(uncertainty_analysis)
        scenario_lookup = {
            scenario.scenario_type: scenario
            for scenario in scenarios
        }
        scenario_summary = self._build_strategy_scenario_summary(scenarios)
        info_items = [
            self._format_information_item(info)
            for info in budgeted_info_values[:max(1, max_info_items)]
        ]

        return {
            'scenario_summary': scenario_summary or 'Unknown',
            'info_items': info_items,
            'recommend_information_gathering': (
                avg_uncertainty > self._risk_tolerance and bool(info_items)
            ),
            'avg_uncertainty': avg_uncertainty,
        }
    
    def post_act(self, action_attempt: str) -> str:
        """No-op: uncertainty updates are observation-driven (pre_observe only)."""
        del action_attempt
        return ""

    def pre_observe(self, observation: str) -> str:
        """Process incoming observation text to update beliefs."""
        observation_text = observation.strip()
        if not observation_text:
            return ""

        observation_hash = hash(observation_text)
        if self._last_observation_hash == observation_hash:
            return ""
        self._last_observation_hash = observation_hash

        actor = self._extract_observation_actor(observation_text)
        if actor and actor == self.get_entity().name:
            return ""

        self._update_own_reservation_from_context(observation)
        self._update_counterpart_reservation_from_context(observation)
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
            if isinstance(belief, BeliefDistribution):
                belief_state.update({
                    'mean': belief.mean,
                    'std': belief.std,
                })
            elif isinstance(belief, NormalInverseGamma):
                belief_state.update({
                    'mu': belief.mu,
                    'lambda_': belief.lambda_,
                    'a': belief.a,
                    'b': belief.b,
                })
            belief_states[name] = belief_state

        belief_confidences = [belief.confidence for belief in self._beliefs.values()]
        avg_confidence = float(np.mean(belief_confidences)) if belief_confidences else 0.0

        return {
            'beliefs': belief_states,
            'avg_confidence': avg_confidence,
            'uncertainty_level': 1.0 - avg_confidence,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        for name, belief_data in state.get('beliefs', {}).items():
            if name in self._beliefs:
                belief = self._beliefs[name]
                belief.confidence = belief_data.get('confidence', belief.confidence)
                belief.evidence_count = belief_data.get('evidence_count', belief.evidence_count)
                belief.last_updated = belief_data.get('last_updated', belief.last_updated)

                if isinstance(belief, BeliefDistribution):
                    belief.mean = belief_data.get('mean', belief.mean)
                    belief.std = max(0.01, belief_data.get('std', belief.std))
                elif isinstance(belief, NormalInverseGamma):
                    belief.mu = max(0.0, belief_data.get('mu', belief.mu))
                    belief.lambda_ = max(1e-6, belief_data.get('lambda_', belief.lambda_))
                    belief.a = max(1e-6, belief_data.get('a', belief.a))
                    belief.b = max(1e-6, belief_data.get('b', belief.b))
        
    def update(self) -> None:
        """Update uncertainty-aware component state."""
        super().update()
