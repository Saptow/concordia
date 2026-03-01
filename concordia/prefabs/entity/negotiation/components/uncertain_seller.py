"""Uncertainty-aware component for probabilistic reasoning in negotiations."""

import dataclasses
import math
import random
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

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
        self.confidence= min(0.95, self.confidence + 0.05 * reliability)
        self.evidence_count += 1

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


    # TODO: revise this method to be more principled
    def update_flexibility_multiplicative(self, flexibility_score: float, reliability: float = 1.0):
            """
            Updates variance by scaling the current belief.
            
            Args:
                flexibility_score: 
                    0.0 (Rigid)    -> Shrink variance (Multiplier < 1.0)
                    0.5 (Neutral)  -> No change (Multiplier = 1.0)
                    1.0 (Flexible) -> Grow variance (Multiplier > 1.0)
                reliability: 
                    Multiplier dampening factor (0.0 to 1.0)
            """
            # Define max parameters to avoid extreme scaling
            max_shrink = 0.5
            max_grow = 2.0
            
            # 2. Map flexibility score to some form of multiplier
            if flexibility_score < 0.5:
                # Range [0.0, 0.5] maps to [max_shrink, 1.0]
                raw_multiplier = max_shrink + (flexibility_score * 2 * (1.0 - max_shrink))
            else:
                # Range [0.5, 1.0] maps to [1.0, max_grow]
                raw_multiplier = 1.0 + ((flexibility_score - 0.5) * 2 * (max_grow - 1.0))

            # If reliability is low, pull the multiplier back towards 1.0 (no change)
            # Formula: Final = 1.0 + (Target - 1.0) * Reliability
            final_multiplier = 1.0 + (raw_multiplier - 1.0) * reliability
            
            self.b *= final_multiplier
            
            # to ensure b stays within reasonable bounds (TODO: edit if needed)
            self.b = max(1e4, min(self.b, 1e11))
    
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
    
    def update_trust(self, trust_level: float, scale: float = 7.0):
        '''
        Update confidence based on trust level of new information. Note that scale is chosen at 7.0 since timestep is in terms of weeks.
        '''
        w = max(0.0, min(1.0, float(trust_level)))

        if w <= 0.0:
            return
        
        m_eff = scale * w  # >= 0

        ev = max(1e-9, self.get_expected_variance)

        a_new = self.a + 0.5 * m_eff

        b_new = ev * max(1e-9, (a_new - 1.0))

        self.a = a_new
        self.b = b_new

        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.02 * (1 - math.exp(-m_eff / 5.0)))

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
    scenario_name: str
    likelihood: float
    value: float
    key_assumptions: List[str]
    risk_factors: List[str]


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
    estimate: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)

class UpdateOpposingBeliefTrustMetadata(BaseModel):
    '''Metadata for trust updates during negotiations.'''
    trust_level: float = Field(ge=0.0, le=1.0, description="Trust level in the counterpart based on the new information")

class UpdateOpposingBeliefInfo(BaseModel):
    '''Information to update belief during negotiations.'''
    budget_info: Optional[UpdateOpposingBeliefInfoMetadata] = None
    trust_info: Optional[UpdateOpposingBeliefTrustMetadata] = None
    # flexibility_info: Optional[UpdateOpposingBeliefInfoMetadata] = None # TODO: to implement later if needed

class UncertainSeller(entity_component.ContextComponent):
    """Component for probabilistic reasoning and uncertainty management in negotiations. (seller's side)"""

    def __init__(
        self,
        model: Any,
        confidence: float = 0.7,
        risk_tolerance: float = 0.3,
        information_gathering_budget: float = 0.1,
        own_reservation_: float = 0.0, # note that this is the minimum reservation price for the seller
        mu: float = 0.0,
        lambda_: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        emit_pre_act_context: bool = True,
    ):
        """Initialize uncertainty-aware component.

        Args:
            model: Language model for analysis
            confidence: Initial confidence level (0-1); to not be updated during negotiation
            risk_tolerance: Tolerance for uncertainty (0-1)
            information_gathering_budget: Fraction of value to spend on info gathering
        """
        self._model = model
        self._confidence = confidence
        self._risk_tolerance = risk_tolerance
        # TODO: think whether we need a separate belief for own reservation price (should not be to simulate information asymmetry of the product)
        self._own_reservation = max(0.0, own_reservation_) # for seller, we first assume that they are sure of their reservation price
        self._info_budget = information_gathering_budget
        self._emit_pre_act_context = emit_pre_act_context
        self._pre_act_value: Optional[str] = None
        self._last_observation_hash: int | None = None

        # Belief state tracking
        self._beliefs: Dict[str, BeliefDistribution | NormalInverseGamma] = {}
        self._uncertainty_sources: List[str] = []
        self._information_gaps: List[str] = []

        # Initialize common negotiation beliefs
        self._initialize_default_beliefs(mu, lambda_, a, b, own_reservation_)

    def get_pre_act_label(self) -> str:
        return "uncertain_seller"
    
    def get_pre_act_value(self) -> str:
        return self._pre_act_value if self._pre_act_value else ""

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
    
    def _initialize_default_beliefs(self, mu: float = 0.0, lambda_: float = 1.0, a: float = 1.0, b: float = 1.0, own_reservation_: float = 0.0):
        """Initialize default beliefs about negotiation parameters."""
        # Counterpart's reservation value (start with high uncertainty)
        self._beliefs['counterpart_reservation'] = NormalInverseGamma(
            name="Counterpart's Reservation Value",
            mu=max(0.0, mu), #TODO: to determine based on initial logs before initialisation of negotiation
            lambda_=max(1e-6, lambda_),
            a=max(1e-6, a),
            b=max(1e-6, b)
        )

    def _analyze_uncertainty_context(self, context: str) -> UncertaintyContext:
        """Analyze context for uncertainty indicators and information gaps."""
        formatted_beliefs="\n".join(
            [f"{name}: mean={belief.get_expected_mean:.2f}, std={math.sqrt(belief.get_expected_variance):.2f}" for name, belief in self._beliefs.items()] # TODO: might need to add confidence for NIG, but decide later on
        )
        prompt = f"""
        You are looking from the perspective of a seller in a negotiation with perfect information of your HDB flat that you plan to sell. The counterpart has imperfect information about your own reservation price and flexibility.
        However, there is uncertainty about the buyer's reservation value and flexibility.
        Based off your current beliefs given below, analyze this negotiation context for uncertainty and missing information:
        Beliefs: {formatted_beliefs}
        Context: {context}

        Identify and assess:
        1. What key information is missing?
        2. What are the main sources of uncertainty and how uncertain are they? Determine the degree of uncertainty based on your current beliefs (0 - 1 scale).
        3. What information would be most valuable to gather given our preferences?
        4. What are the potential scenarios we should consider?

        Format your response using the JSON schema provided. It should be as follows:
        missing_info: [List key missing information]
        uncertainty_sources: List of {{"claim": [uncertain claim], "confidence_level": [0-1]}}
        valuable_info: [List of most valuable information to gather given preferences]
        scenarios: [Key scenarios to consider]"""

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

    def _update_counterpart_reservation_from_context(self, context: str):
        """Update beliefs based on new context information."""
        prompt = f"""
        You are a seller in a negotiation with perfect information. The counterpart has imperfect information about your own budget and flexibility regarding the HDB flat. However, there is uncertainty about the buyer's budget and flexibility.
        Given a context, your task is to extract information regarding the counterpart's budget and flexibility in the negotiation, if there is any:
        
        Context: {context}

        Focus on extracting BUDGET_INFO, the counterpart's budget or reservation value (in dollars). Determine the confidence level (0-1) through the amount of trust you have in this information.

        Return a response using the JSON schema provided. Provide estimates with confidence levels:
        budget_info: [value estimate] [confidence 0-1]
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
        def _get_risk_adjusted_weights(risk_tolerance: float) -> Dict[str, float]:
            """
            Calculates scenario weights based on risk tolerance.
            
            Logic:
            - We start with a neutral stance: [25% Pessimistic, 50% Realistic, 25% Optimistic]
            - Risk Tolerance acts as a slider moving mass from Pessimistic to Optimistic.
            """
            # TODO: Refine this logic as needed
            # Define neutral likelihood for now
            base_pessimistic = 0.25
            base_optimistic = 0.25
            
            # Scale based on risk tolerance (max movement of 25% from each tail)
            shift = (risk_tolerance - 0.5) * 0.5  
            
            # Calculate likelihoods
            l_pess = base_pessimistic - shift
            l_opt  = base_optimistic + shift
            l_real = 1.0 - l_pess - l_opt # to ensure sum to 1.0
            
            return {
                'Pessimistic': max(0.0, min(1.0, l_pess)),
                'Realistic': max(0.0, min(1.0, l_real)),
                'Optimistic': max(0.0, min(1.0, l_opt))
            }
        likelihoods = _get_risk_adjusted_weights(self._risk_tolerance)
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
        own_reservation=self._own_reservation # for the seller, we assume they know their own reservation price

        # find surplus; for the seller, its 
        mu_diff = mu_cp - own_reservation

        # TODO: we assume independence for now (i.e. covariance = 0) but assumption is weak since we are talking about the same product. 
        # However, it is fine for now, since we assume maximum variance between the differences => more conservative estimates for ZOPA. 
        zopa_dist = NormalDist(mu_diff, math.sqrt(var_cp + 0)) # variance of own reservation is 0 since we assume full knowledge
        p_upper = 0.5 + ((1 - self._risk_tolerance) / 2.0)
        
        z_width = NormalDist(mu=0, sigma=1).inv_cdf(p_upper)
        
        val_optimistic = zopa_dist.mean + (z_width * zopa_dist.stdev)
        val_realistic  = zopa_dist.mean 
        val_pessimistic = zopa_dist.mean - (z_width * zopa_dist.stdev)
        # generate scenarios based on mean and std deviations
        for name, likelihood in likelihoods.items():
            if name == 'Pessimistic': 
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_pessimistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_pessimistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
            elif name == 'Realistic':
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_realistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_realistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
            else:
                scenarios.append(
                    ScenarioAnalysis(
                        scenario_name=f"{name}: Deal Possible" if val_optimistic > 0 else f"{name}: No Deal",
                        likelihood=likelihood,
                        value=max(0.0, val_optimistic),
                        key_assumptions=[f"Counterpart reservation at {cp_belief.get_expected_mean:.2f}", f"Your reservation at {own_reservation:.2f}"],
                        risk_factors=[f"Counterpart reservation uncertainty (std: {math.sqrt(cp_belief.get_expected_variance):.2f})"]
                    )
                )
        return scenarios
    
    def _calculate_information_values(self, context: str, uncertainty_analysis: UncertaintyContext) -> List[InformationValue]:
        """Calculate value of gathering different types of information."""
        # Current uncertainty level (higher uncertainty = more value from information)

        # Determine the info_opportunities using a LLM. 
        prompt = f"""
        You are looking from the perspective of a seller in a negotiation with perfect information of your HDB flat that you plan to sell. The counterpart has imperfect information about your own reservation price and flexibility.
        However, there is uncertainty about the counterpart's reservation value and flexibility. Given the negotiation context and uncertainty analysis below, identify up to 10 specific questions that can be asked to gather valuable information that affects the counterpart's reservation value and flexibility.

        Context: {context}
        Uncertainty Analysis: {uncertainty_analysis.model_dump_json()}
        
        For each question, estimate 
        1. priority_score (0-1): how much confidence improvement this information could provide.
        2. cost_factor (0-1): the relative cost of obtaining this information based off how tedious/difficult it is to obtain.

        Return a list of information gathering opportunities in the JSON schema provided.
        """

        response = self._model.sample_text(prompt, json_schema=InformationValueResponse.model_json_schema())
        # Load JSON response and validate schema
        try:
            info_opportunities_response = InformationValueResponse.model_validate_json(response)
            info_opportunities = info_opportunities_response.information_values or []
        except ValidationError as e:
            # current fallback TODO: improve error handling
            info_opportunities = []

        # Sort by rank
        info_opportunities.sort(key=lambda x: x.priority_score, reverse=True)
        return info_opportunities

    def _generate_uncertainty_guidance(self, context: str) -> str:
        """Generate comprehensive uncertainty-aware guidance."""
        # Analyze uncertainty in context
        uncertainty_analysis = self._analyze_uncertainty_context(context)

        # Generate scenarios
        scenarios = self._generate_scenarios()

        # Calculate information values
        info_values = self._calculate_information_values(context, uncertainty_analysis)

        # Generate guidance
        guidance = f""" Uncertainty-Aware Analysis
**Current Belief State:**
"""

        for name, belief in self._beliefs.items():
            std = math.sqrt(max(0.0, belief.get_expected_variance))
            guidance += (
                f"• {belief.name}: Mean={belief.get_expected_mean:.1f}, "
                f"Std={std:.1f}\n"
            )
            guidance += (
                f"  95% CI: [{belief.get_confidence_interval()[0]:.1f}, "
                f"{belief.get_confidence_interval()[1]:.1f}]\n"
            )

        guidance += f"\n**Scenario Analysis:**\n"
        for scenario in scenarios:
            guidance += f"• {scenario.scenario_name} ({scenario.likelihood:.1%}): "
            guidance += f"Expected value {scenario.value:.0f}, "

        guidance += f"\n**Information Gathering Opportunities:**\n"
        frac_of_budget = self._info_budget
        for info in info_values[:3]:  # Top 3 opportunities
            frac_of_budget -= info.cost_factor
            if frac_of_budget > 0:
                guidance += f"• \"{info.question[:50]}...\"\n"
                guidance += f"  Priority Score: ${info.priority_score:.0f}, Cost Factor: +{info.cost_factor:.2f}\n"

        guidance += f"\n**Uncertainty Management Strategy:**\n"

        # Calculate overall uncertainty level
        uncertainty_levels = [1 - info.confidence_level for info in uncertainty_analysis.uncertainty_sources]
        if uncertainty_levels:
            avg_uncertainty = float(np.mean(uncertainty_levels))
        else:
            belief_confidences = [belief.confidence for belief in self._beliefs.values()]
            avg_uncertainty = 1.0 - float(np.mean(belief_confidences)) if belief_confidences else 1.0

        if avg_uncertainty > self._risk_tolerance:
            guidance += f"• HIGH UNCERTAINTY DETECTED (uncertainty: {avg_uncertainty:.1%})\n"
            guidance += f"• Recommend information gathering before major commitments\n"
            guidance += f"• Consider contingent offers and flexible terms\n"
            guidance += f"• Focus on robust strategies that work across scenarios\n"
        else:
            guidance += f"• Moderate uncertainty (uncertainty: {1-avg_uncertainty:.1%})\n"
            guidance += f"• Proceed with standard negotiation approach\n"
            guidance += f"• Monitor for new information that could change beliefs\n"

        guidance += f"\n**Robust Decision Principles:**\n"
        guidance += f"• Base decisions on {1-avg_uncertainty:.1%} confidence in current beliefs\n"
        guidance += f"• Maintain flexibility for belief updates\n"
        guidance += f"• Balance information gathering with negotiation progress\n"

        return guidance

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide uncertainty-aware guidance before action."""
        context = action_spec.call_to_action

        # Beliefs are updated only when new observations arrive (pre_observe).
        guidance = self._generate_uncertainty_guidance(context)
        self._pre_act_value=guidance
        if self._emit_pre_act_context:
            return f"\n{guidance}"
        return ""

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
            'own_reservation': self._own_reservation,
            'avg_confidence': avg_confidence,
            'uncertainty_level': 1.0 - avg_confidence,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        self._own_reservation = max(0.0, state.get('own_reservation', self._own_reservation))
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

    # def get_action_attempt(
    #         self,
    #         context: Any, 
    #         action_spec: entity_lib.ActionSpec
    #     ) -> str:
    #     """Generate action attempt with uncertainty considerations."""
    #     situation = action_spec.call_to_action

    #     self._update_counterpart_reservation_from_context(situation)
    #     self._update_own_reservation_from_context(situation)

    #     uncertainty_analysis = self._analyze_uncertainty_context(situation)

    #     scenarios = self._generate_scenarios()

    #     info_values = self._calculate_information_values(situation, uncertainty_analysis)

    #     # generate average uncertainty level
    #     avg_uncertainty = np.mean([1 - confidence_level for info in uncertainty_analysis.uncertainty_sources for confidence_level in [info.confidence_level]])

    #     should_gather_info = avg_uncertainty > (1 - self._confidence_threshold and info_values and info_values[0].net_value > 0)

    #     if should_gather_info:
    #         # generate info gathering action
    #         top_info = info_values[0]
    #         prompt = f""" Based on uncertainty analysis, generate an information gathering action towards the counterpart. 

    #         Situation: {situation}

    #         Uncertainty Analysis: {uncertainty_analysis.model_dump_json()}

    #         Most Valuable Information to Gather:
    #         Question: {top_info.question}
    #         Priority Score: {top_info.priority_score}
    #         Cost Factor: {top_info.cost_factor}

    #         Generate a negotiation action that:
    #         1. Asks the most valuable information-gathering question
    #         2. Explains why this information would help both parties
    #         3. Demonstrates thoughtful preparation and analysis
    #         4. Maintains negotiation momentum while reducing uncertainty
    #         5. Shows professional competence despite information gaps

    #         Action:"""
    #     else:
    #         # generate based on scenarios
    #         best_scenario = max(scenarios, key=lambda x: x.likelihood * x.value)
    #         worst_scenario = min(scenarios, key=lambda x: x.likelihood * x.value)

    #         # get confidence intervals of counterpart
    #         cp_belief=self._beliefs['counterpart_reservation']
    #         cp_ci = cp_belief.get_confidence_interval()
    #         prompt = f""" Based on uncertainty analysis, generate a robust negotiation action towards the counterpart.

    #         Situation: {situation}

    #         Current Belief of counterpart State:
    #         • {cp_belief.name}: Mean = {cp_belief.get_expected_mean:.1f}, Variance = {cp_belief.get_expected_variance:.1f}
    #         • 95% CI: [{cp_ci[0]:.1f}, {cp_ci[1]:.1f}]

    #         Scenario Analysis:
    #         Best Case: {best_scenario.scenario_name} (Value: {best_scenario.value:.0f}, Likelihood: {best_scenario.likelihood:.1%})
    #         Worst Case: {worst_scenario.scenario_name} (Value: {worst_scenario.value:.0f}, Likelihood: {worst_scenario.likelihood:.1%})

    #         Risk Management: 
    #         - Risk Tolerance Level: {self._risk_tolerance:.2f}
    #         - Key Uncertainties: {', '.join([info.claim for info in uncertainty_analysis.uncertainty_sources])}

    #         Generate a negotiation action that:
    #         1. Makes a robust proposal that works across scenarios
    #         2. Acknowledges and manages key uncertainties
    #         3. Includes contingencies for different outcomes  
    #         4. Demonstrates analytical sophistication
    #         5. Balances confidence with appropriate caution given uncertainty level

    #         Action:"""

    #         response = self._model.sample_text(prompt)

    #             # Clean up response
    #     action = response.strip()
    #     if action.lower().startswith('action:'):
    #         action = action[7:].strip()
        
    #     # Add uncertainty framing based on confidence level
    #     if should_gather_info:
    #         action = f"To make the best decision for both of us, {action.lower()}"

    #     return action

    def update(self) -> None:
        """Update uncertainty-aware component state."""
        # # Gradually decay confidence over time if no new evidence
        # for belief in self._beliefs.values():
        #     if belief.evidence_count == 0:
        #         belief.confidence *= 0.99  # Slow decay
        #         belief.std = min(belief.std * 1.01, belief.std * 2)  # Increase uncertainty
