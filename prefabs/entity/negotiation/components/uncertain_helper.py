"""Shared helpers for uncertainty-aware negotiation components."""

import copy
import dataclasses
import json
import math
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from concordia.components.agent import memory as memory_component
from concordia.hdb_simulation.models.schemas.common import NormalDistribution
from pydantic import BaseModel, Field, ValidationError
from scipy import stats

ISSUE_BANK_PROMPT_MAX_CHARS = 1_000
RECENT_MEMORIES_PROMPT_MAX_CHARS = 1_000
FULL_CONTEXT_PROMPT_MAX_CHARS = 1_600
DISCOVER_ISSUES_MAX_TOKENS = 512


def truncate_prompt_text(
    text: Any,
    *,
    max_chars: int,
    middle: bool = False,
) -> str:
    normalized = str(text or '').strip()
    if max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    if max_chars <= 10:
        return normalized[:max_chars]
    if not middle:
        return normalized[: max_chars - 3].rstrip() + '...'

    marker = '\n\n... [truncated] ...\n\n'
    available = max_chars - len(marker)
    if available <= 2:
        return normalized[:max_chars]
    head_chars = available // 2
    tail_chars = available - head_chars
    return (
        normalized[:head_chars].rstrip()
        + marker
        + normalized[-tail_chars:].lstrip()
    )


def build_compact_listing_context(listing_record: Any) -> str:
    """Render a compact listing snapshot for model prompts."""
    if listing_record is None:
        return 'Listing unavailable.'

    flat = getattr(listing_record, 'flat', None)
    if flat is None:
        return 'Listing unavailable.'

    nearby_amenities = getattr(flat, 'nearby_amenities', None) or []
    amenity_bits: list[str] = []
    for amenity in nearby_amenities[:4]:
        amenity_type = str(getattr(amenity, 'type', '')).strip()
        amenity_name = str(getattr(amenity, 'name', '')).strip()
        if amenity_type and amenity_name:
            amenity_bits.append(f'{amenity_type}:{amenity_name}')
        elif amenity_name:
            amenity_bits.append(amenity_name)

    amenity_text = ', '.join(amenity_bits) if amenity_bits else 'unknown'

    trend_text = 'unknown'
    past_price_trends = getattr(flat, 'past_price_trends', None)
    if past_price_trends is not None:
        trend_text = (
            f'{int(past_price_trends.transactions_6m)} txns in 6m, '
            f'${float(past_price_trends.min_price_6m):,.0f}-${float(past_price_trends.max_price_6m):,.0f}'
        )

    return '\n'.join([
        f'Price: SGD {float(getattr(listing_record, "listing_price", 0.0)):,.0f}',
        f'Flat: {flat.to_compact_description()}',
        f'Extension of stay: {"yes" if bool(getattr(flat, "extension_of_stay", False)) else "no"}',
        f'Amenities: {amenity_text}',
        f'Price trends: {trend_text}',
    ])


@dataclasses.dataclass
class NormalInverseGamma:
    """Normal-Inverse-Gamma belief used for reservation-value uncertainty."""

    name: str
    mu: float
    lambda_: float
    a: float
    b: float
    confidence: float = 0.5
    evidence_count: int = 0
    last_updated: Optional[str] = None

    def _get_t_critical(self, confidence: float, df: float) -> float:
        confidence = max(0.90, min(0.99, confidence))
        df = max(1e-6, float(df))
        tail_probability = 0.5 + (confidence / 2.0)
        return float(stats.t.ppf(tail_probability, df))

    @property
    def get_expected_mean(self) -> float:
        return self.mu

    @property
    def get_expected_variance(self) -> float:
        if self.a > 1:
            return self.b / (self.a - 1)
        return self.b

    def get_confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        confidence = max(0.90, min(0.99, confidence))
        df = 2 * self.a
        scale = np.sqrt(self.b * (self.lambda_ + 1) / (self.lambda_ * self.a))
        margin = self._get_t_critical(confidence, df) * scale
        lower = max(0.0, self.mu - margin)
        upper = max(lower, self.mu + margin)
        return (lower, upper)

    def sample(self, n: int = 1) -> Union[float, List[float]]:
        rng = np.random.default_rng()
        tau_sq_samples = 1 / rng.gamma(self.a, 1 / self.b, n)
        mu_samples = rng.normal(
            self.mu,
            np.sqrt(1 / (self.lambda_ * tau_sq_samples)),
            n,
        )
        return mu_samples[0] if n == 1 else mu_samples.tolist()

    def model_copy(self, *, deep: bool = False) -> 'NormalInverseGamma':
        if deep:
            return copy.deepcopy(self)
        return copy.copy(self)

    def update_with_evidence(self, observation: float, reliability: float = 1.0) -> None:
        reliability = max(0.0, min(1.0, reliability))
        observation = max(0.0, observation)
        new_lambda = self.lambda_ + reliability
        new_mu = (self.lambda_ * self.mu + reliability * observation) / new_lambda
        new_a = self.a + 0.5 * reliability
        diff = observation - self.mu
        new_b = self.b + (self.lambda_ * reliability * (diff ** 2)) / (2 * new_lambda)

        self.evidence_count += 1
        self.confidence = min(0.95, self.confidence + 0.05 * reliability)
        self.mu = new_mu
        self.lambda_ = new_lambda
        self.a = new_a
        self.b = new_b

    def update_trust(self, trust_level: float, scale: float = 1.0) -> None:
        trust = max(-1.0, min(1.0, float(trust_level)))
        if math.isclose(trust, 0.0, abs_tol=1e-6):
            return

        effective_scale = 0.25 * min(1.0, max(0.0, float(scale)))
        m_eff = effective_scale * abs(trust)
        if m_eff <= 0.0:
            return

        direction = 1.0 if trust > 0.0 else -1.0
        expected_variance = max(1e-9, self.get_expected_variance)
        a_new = max(1.05, self.a + direction * 0.5 * m_eff)
        b_new = expected_variance * max(1e-9, (a_new - 1.0))

        self.a = a_new
        self.b = b_new
        self.evidence_count += 1
        self.confidence = max(
            0.05,
            min(0.95, self.confidence + (m_eff * direction)),
        )


def kappa_from_negotiation_count(negotiation_count: int) -> float:
    return max(1.0, 1.0 + float(max(0, negotiation_count)))


def adjust_confidence_for_negotiation_count(
    confidence: float,
    negotiation_count: int,
    *,
    step: float = 0.04,
    cap: float = 0.95,
) -> float:
    bounded_confidence = max(0.0, min(1.0, float(confidence)))
    experience_bonus = max(0.0, float(negotiation_count)) * max(0.0, float(step))
    return min(max(0.0, float(cap)), bounded_confidence + experience_bonus)


def alpha_from_uncertainty_level(uncertainty_level: float) -> float:
    bounded_uncertainty = max(0.0, min(1.0, float(uncertainty_level)))
    return 1.5 + (4.0 * (1.0 - bounded_uncertainty))


def beta_from_predictive_variance(
    predictive_variance: float,
    *,
    kappa: float,
    alpha: float,
) -> float:
    variance = max(1e-9, float(predictive_variance))
    return variance * kappa * max(1e-9, alpha - 1.0) / (kappa + 1.0)


def build_counterpart_reservation_prior(
    *,
    name: str,
    source_distribution: NormalDistribution,
    confidence: float,
    negotiation_count: int,
) -> NormalInverseGamma:
    kappa = kappa_from_negotiation_count(negotiation_count)
    uncertainty_level = 1.0 - max(0.0, min(1.0, float(confidence)))
    alpha = alpha_from_uncertainty_level(uncertainty_level)
    beta = beta_from_predictive_variance(
        source_distribution.get_expected_variance,
        kappa=kappa,
        alpha=alpha,
    )
    return NormalInverseGamma(
        name=name,
        mu=max(0.0, float(source_distribution.get_expected_mean)),
        lambda_=kappa,
        a=alpha,
        b=beta,
        confidence=max(0.0, min(1.0, float(confidence))),
        # Fresh pair-level priors should start with zero direct negotiation
        # evidence even if the listing-stage source signal already aggregated
        # earlier market observations.
        evidence_count=1,
        last_updated=None,
    )


@dataclasses.dataclass
class ScenarioAnalysis:
    """Scenario summary used by the strategy layer."""

    scenario_type: Literal['Pessimistic', 'Realistic', 'Optimistic']
    outcome: Literal['Deal Possible', 'No Deal']
    likelihood: float


class NegotiationIssueBucket(StrEnum):
    FLAT_CONDITIONS = 'Flat Conditions'
    FLAT_AMENITIES = 'Flat Amenities'
    FINANCING = 'Financing'
    TIMELINE_AND_PROCESS = 'Timeline and Process'
    COUNTERPART = 'Counterpart\'s Circumstances'
    OTHERS = 'Others'


class LinkedBelief(StrEnum):
    OWN_RESERVATION = 'own_reservation'
    COUNTERPART_RESERVATION = 'counterpart_reservation'


class NegotiationIssue(BaseModel):
    """Open issue that may justify one targeted question."""

    label: NegotiationIssueBucket = Field(description='Categorical label for the issue')
    summary: str = Field(description='Brief summary of the issue in 50 words or less')
    evidence: List[str] = Field(
        default_factory=list,
        description='List of observations that relate to this issue',
    )
    linked_belief: Optional[List[LinkedBelief]] = Field(
        None,
        description='List of related beliefs',
    )
    uncertainty: float = Field(ge=0.0, le=1.0, description='Degree of uncertainty (0-1)')
    answerable_now: Literal[0, 1] = Field(
        description='1 if the counterpart can answer a direct question now, else 0',
    )
    suggested_question: str = Field(
        default='',
        description='One concise question to ask if answerable_now is 1',
    )


class NegotiationIssueResponse(BaseModel):
    issues: Optional[List[NegotiationIssue]] = Field(
        None,
        description='List of identified open issues based on the current context',
    )


def format_money(value: float) -> str:
    return f'{float(value):.2f}'


def format_interval(interval: Tuple[float, float]) -> str:
    return f'{format_money(interval[0])}-{format_money(interval[1])}'


def format_observation_summary(observation: str, max_chars: int = 180) -> str:
    normalized = ' '.join(str(observation).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + '...'


def append_debug_trace(debug_trace: List[str], message: str, limit: int = 12) -> None:
    normalized = ' '.join(str(message).split()).strip()
    if not normalized:
        return
    debug_trace.append(normalized)
    del debug_trace[:-limit]


def extract_observation_actor(observation: str) -> str | None:
    text = observation.strip()
    if text.startswith('[observation]'):
        text = text[len('[observation]'):].strip()
    actor, sep, _ = text.partition(':')
    if not sep:
        return None
    actor = actor.strip()
    return actor if actor else None


def normalize_text(value: str) -> str:
    return ' '.join(str(value).split()).strip().strip('"').strip("'")


def coerce_positive_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0.0 else float(default)


def extract_first_json_object(text: str) -> str | None:
    candidate = str(text or '').strip()
    start = candidate.find('{')
    if start < 0:
        return None
    candidate = candidate[start:]
    depth = 0
    for idx, ch in enumerate(candidate):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return candidate[: idx + 1]
    return None


def extract_listing_handoff_state(observation: str) -> Dict[str, Any] | None:
    marker = 'Listing handoff context for this negotiation:'
    text = str(observation or '')
    if marker not in text:
        return None
    _, _, payload = text.partition(marker)
    payload_text = str(payload).strip()
    if not payload_text:
        return {}
    payload_json = extract_first_json_object(payload)
    if not payload_json:
        return {'listing_summary': payload_text}
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        return {'listing_summary': payload_text}
    return parsed if isinstance(parsed, dict) else None


def sanitize_issue_bank(issues: List[NegotiationIssue]) -> List[NegotiationIssue]:
    sanitized: List[NegotiationIssue] = []
    for issue in issues[:5]:
        issue.summary = normalize_text(issue.summary)
        issue.suggested_question = normalize_text(issue.suggested_question)
        issue.evidence = [
            cleaned
            for item in issue.evidence
            if (cleaned := normalize_text(item))
        ][:5]
        if len(issue.summary) < 6:
            continue
        if issue.answerable_now == 1 and len(issue.suggested_question) < 8:
            continue
        if issue.answerable_now == 0:
            issue.suggested_question = ''
        sanitized.append(issue)
    sanitized.sort(key=compute_issue_score, reverse=True)
    return sanitized


def compute_issue_score(issue: NegotiationIssue | None) -> float:
    if issue is None:
        return 0.0
    return max(0.0, min(1.0, float(issue.uncertainty) * float(issue.answerable_now)))


def get_open_issues(issue_bank: List[NegotiationIssue], limit: int | None = None) -> List[NegotiationIssue]:
    issues = list(issue_bank)
    if limit is None:
        return issues
    return issues[: max(0, int(limit))]


def get_top_issue(issue_bank: List[NegotiationIssue]) -> NegotiationIssue | None:
    return issue_bank[0] if issue_bank else None


def summarize_top_issue(issue: NegotiationIssue | None) -> str:
    if issue is None:
        return 'None'
    return (
        f"Category: {issue.label}\n"
        f"Summary: {issue.summary}\n"
        f"Evidence: {' |\n '.join(issue.evidence) if issue.evidence else 'None'}\n"
        f"Uncertainty: {issue.uncertainty:.2f}\n"
        f"Answerable Now: {'Yes' if issue.answerable_now == 1 else 'No'}\n"
        f"Suggested Question: {issue.suggested_question if issue.answerable_now == 1 else 'N/A'}"
    )


def format_issue_item(
    issue: NegotiationIssue,
    *,
    include_score: bool = False,
) -> str:
    question = issue.suggested_question or 'Not directly answerable now.'
    if len(question) > 90:
        question = question[:87].rstrip() + '...'
    item = (
        f"Category: {issue.label}\n"
        f"Question: {question}"
    )
    if include_score:
        item += f' [S={compute_issue_score(issue):.2f}]'
    return item


def format_issue_bank(
    issue_bank: List[NegotiationIssue],
    *,
    include_score: bool = False,
) -> str:
    open_issues = get_open_issues(issue_bank)
    if not open_issues:
        return 'None'
    return ' | '.join(
        format_issue_item(issue, include_score=include_score)
        for issue in open_issues
    )


def normalize_scenario_outcome(outcome: str) -> str:
    return str(outcome).strip().lower()


def build_main_deal_summary(scenarios: List[ScenarioAnalysis]) -> str:
    if not scenarios:
        return 'Overall assessment unknown.'

    scenario_lookup = {scenario.scenario_type: scenario for scenario in scenarios}
    realistic = scenario_lookup.get('Realistic')
    pessimistic = scenario_lookup.get('Pessimistic')
    optimistic = scenario_lookup.get('Optimistic')

    realistic_outcome = normalize_scenario_outcome(realistic.outcome) if realistic else 'unknown'
    pessimistic_outcome = normalize_scenario_outcome(pessimistic.outcome) if pessimistic else 'unknown'
    optimistic_outcome = normalize_scenario_outcome(optimistic.outcome) if optimistic else 'unknown'

    deal_possible_probability = sum(
        float(scenario.likelihood)
        for scenario in scenarios
        if normalize_scenario_outcome(scenario.outcome) == 'deal possible'
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


def build_strategy_scenario_summary(scenarios: List[ScenarioAnalysis]) -> str:
    if not scenarios:
        return 'Unknown'

    scenario_lookup = {scenario.scenario_type: scenario for scenario in scenarios}

    def _format_case(case_label: str, scenario_name: str) -> str:
        scenario = scenario_lookup.get(scenario_name)
        if scenario is None:
            return f'{case_label}: unknown.'
        return f'{case_label}: {scenario.outcome.lower()} ({scenario.likelihood:.0%}).'

    return ' '.join([
        build_main_deal_summary(scenarios),
        _format_case('Base case', 'Realistic'),
        _format_case('Downside case', 'Pessimistic'),
        _format_case('Upside case', 'Optimistic'),
    ])


def retrieve_recent_memories(owner: Any, memory_component_key: str, limit: int) -> List[str]:
    try:
        memory = owner.get_entity().get_component(
            memory_component_key,
            type_=memory_component.Memory,
        )
        return list(memory.retrieve_recent(limit=max(1, int(limit))))
    except Exception:
        return []


def build_strategy_context(
    belief_summary: str,
    recent_memories: List[str],
    extra_sections: Optional[List[Tuple[str, Any]]] = None,
) -> str:
    section_blocks: List[str] = []
    for title, payload in extra_sections or []:
        if not payload:
            continue
        if isinstance(payload, str):
            section_blocks.append(f'{title}:\n{payload}\n')
        else:
            section_blocks.append(f'{title}:\n{json.dumps(payload, ensure_ascii=False)}\n')

    memory_lines = [f'- {memory_text}' for memory_text in recent_memories] or ['- None']
    return (
        'Current belief summary:\n'
        f'{belief_summary}\n'
        f'{"".join(section_blocks)}'
        'Recent negotiation memories:\n'
        + '\n'.join(memory_lines)
    )


def merge_live_context(strategy_context: str, context: str) -> str:
    live_context = str(context).strip()
    if not live_context:
        return strategy_context
    return f'{strategy_context}\nCurrent action context:\n{live_context}'


def discover_issues(
    model: Any,
    *,
    role_description: str,
    answerer_name: str,
    context: str,
    issue_bank: List[NegotiationIssue],
    recent_memories: List[str],
) -> List[NegotiationIssue]:
    issue_bank_json = truncate_prompt_text(
        json.dumps([issue.model_dump() for issue in issue_bank], ensure_ascii=False),
        max_chars=ISSUE_BANK_PROMPT_MAX_CHARS,
        middle=True,
    )
    recent_memories_json = truncate_prompt_text(
        json.dumps(recent_memories, ensure_ascii=False),
        max_chars=RECENT_MEMORIES_PROMPT_MAX_CHARS,
        middle=True,
    )
    truncated_context = truncate_prompt_text(
        context,
        max_chars=FULL_CONTEXT_PROMPT_MAX_CHARS,
        middle=True,
    )
    prompt = (
        "# Role\n"
        f"You are a {role_description}.\n\n"
        "# Task\n"
        "Given the full context below, as well as the current issue bank and recent memories, identify up to 5 **open issues** that matter right now in this negotiation.\n\n"
        "# Inputs\n"
        "## Current Issue Bank\n"
        f"{issue_bank_json}\n\n"
        "## Recent Memories\n"
        f"{recent_memories_json}\n\n"
        "## Full Context\n"
        f"{truncated_context}\n\n"
        "# Private Reasoning Process\n"
        "Think step by step **privately** before answering:\n\n"
        "1. Review the current issue bank against the Full Context and Recent Memories and keep only issues that are still open and relevant now.\n"
        "2. Look for **NEW** unresolved uncertainties in the recent memories and full context.\n"
        "3. Group these uncertainties into the buckets defined in the schema.\n"
        "4. Merge ANY duplicates or overlapping issues rather than listing the same concern twice.\n"
        "5. For each issue, identify any concrete evidence from the recent memories or full context that relates to this issue and quote them **EXACTLY**, to list them as `evidence`.\n"
        "6. For each issue, give a concise summary in 50 words or less.\n"
        "7. For each issue, based on the concise summary and evidence, give an `uncertainty` score between 0 and 1 to indicate how uncertain you are about the issue (0 means not at all, 1 means extremely uncertain).\n"
        f"8. Mark `answerable_now=1` only if the {answerer_name} can realistically answer a direct question immediately, based on the Full Context.\n"
        "9. If `answerable_now=1`, provide one concise natural question that you would ask the counterpart to clear the uncertainty around this issue. The question should be specific and grounded, based on the evidence and summary you provided.\n"
        "10. Return only the final JSON object. Do not reveal your reasoning.\n\n"
        "# Rules\n"
        "- Return only open issues that are still relevant now.\n"
        "- You may repeat issues from the current issue bank **IF** they are still relevant to the present context.\n"
        "- `summary` should be concise, concrete, and no more than 50 words.\n"
        f"- `answerable_now` must be either `0` or `1`.\n"
        "- If `answerable_now` is `0`, leave `suggested_question` empty.\n"
        "- If `answerable_now` is `1`, provide one concise natural question.\n"
        "- Only take `evidence` for new issues from Recent Memories or Full Context.\n"
        "- If there are no useful open issues, you are allowed to return an empty list.\n\n"
        "- Do not return placeholders or hypothetical issues that are not grounded in the current context or recent memories.\n"
        "# Output\n"
        "- Return JSON only.\n"
        "- Match the provided schema exactly.\n"
    )
    response = model.sample_text(
        prompt,
        json_schema=NegotiationIssueResponse.model_json_schema(),
        max_tokens=DISCOVER_ISSUES_MAX_TOKENS,
    )
    try:
        issue_response = NegotiationIssueResponse.model_validate_json(response)
    except ValidationError:
        return []
    return issue_response.issues or []


def restore_issue_bank(raw_issue_bank: List[Dict[str, Any]]) -> List[NegotiationIssue]:
    issues: List[NegotiationIssue] = []
    for issue_data in raw_issue_bank:
        try:
            issues.append(NegotiationIssue.model_validate(issue_data))
        except ValidationError:
            continue
    return sanitize_issue_bank(issues)


__all__ = [
    'adjust_confidence_for_negotiation_count',
    'LinkedBelief',
    'NegotiationIssue',
    'NegotiationIssueBucket',
    'NegotiationIssueResponse',
    'NormalDistribution',
    'NormalInverseGamma',
    'alpha_from_uncertainty_level',
    'ScenarioAnalysis',
    'append_debug_trace',
    'beta_from_predictive_variance',
    'build_counterpart_reservation_prior',
    'build_main_deal_summary',
    'build_strategy_context',
    'build_strategy_scenario_summary',
    'coerce_positive_float',
    'compute_issue_score',
    'discover_issues',
    'extract_first_json_object',
    'extract_listing_handoff_state',
    'extract_observation_actor',
    'format_interval',
    'format_issue_bank',
    'format_issue_item',
    'format_money',
    'format_observation_summary',
    'get_open_issues',
    'get_top_issue',
    'kappa_from_negotiation_count',
    'merge_live_context',
    'normalize_scenario_outcome',
    'normalize_text',
    'restore_issue_bank',
    'retrieve_recent_memories',
    'sanitize_issue_bank',
    'summarize_top_issue',
    'truncate_prompt_text',
]
