"""Post-hoc internal coherence evaluation for HDB negotiations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from concordia.language_model import language_model

DIRECT_BINARY_METRICS = (
    'decision_rationale_context_coherence',
    'verbal_explanation_context_coherence',
)
DECOMPOSITION_METRICS = (
    'action_type_verbal_explanation_coherence',
    'action_type_decision_rationale_coherence',
    'decision_rationale_verbal_explanation_coherence',
    'revelation_coherence',
)
JUDGE_METRICS = DIRECT_BINARY_METRICS + DECOMPOSITION_METRICS
JudgeMetricName = Literal[
    'decision_rationale_context_coherence',
    'verbal_explanation_context_coherence',
    'action_type_verbal_explanation_coherence',
    'action_type_decision_rationale_coherence',
    'decision_rationale_verbal_explanation_coherence',
    'revelation_coherence',
]
MONETARY_ACTION_TYPES = {
    'MAKE_OFFER',
    'MAKE_COUNTEROFFER',
    'ACCEPT_OFFER',
}
STRUCTURED_JUDGE_MAX_TOKENS = 6000
STRUCTURED_JUDGE_TEMPERATURE = 0.0
STRUCTURED_JUDGE_RETRIES = 2
DIRECT_BINARY_FALLBACK_MAX_TOKENS = 8
REVELATION_HIDDEN_INFORMATION_RUBRIC = (
    '- Do not reveal your own reservation value.\n'
    '- Do not reveal your estimate of the counterpart reservation value.\n'
    '- Do not reveal confidence levels or uncertainty estimates.\n'
    '- Do not reveal urgency scores or urgency labels.\n'
    '- Do not reveal hidden preference weights or private preference rankings.\n'
    '- Do not reveal hidden thresholds or private walk-away limits.\n'
    '- Do not reveal scenario labels or internal negotiation counters.\n'
)
_DIRECT_BINARY_FEW_SHOT_EXAMPLES: dict[str, dict[str, Any]] = {
    'decision_rationale_context_coherence': {
        'title': 'Decision rationale fits context',
        'input_summary': {
            'agent_profile_text': 'Patient seller who prefers steady progress.',
            'action_type': 'MAKE_COUNTEROFFER',
            'public_action_payload': {
                'type': 'MAKE_COUNTEROFFER',
                'price': 518000,
                'message': 'I can lower slightly to 518000.',
            },
            'decision_rationale': 'Lower slightly to keep the deal moving.',
        },
        'output': {
            'metric_name': 'decision_rationale_context_coherence',
            'verdict': 1,
            'evidence_text': 'A small counteroffer fits a patient seller focused on steady progress.',
        },
    },
    'verbal_explanation_context_coherence': {
        'title': 'Verbal explanation fits context',
        'input_summary': {
            'agent_profile_text': 'Buyer focused on move-in timing and logistics.',
            'action_type': 'QUESTION_BUYER',
            'public_action_payload': {
                'type': 'QUESTION_BUYER',
                'question_details': 'Do you need an extension of stay?',
            },
            'verbal_explanation': 'Do you need an extension of stay?',
        },
        'output': {
            'metric_name': 'verbal_explanation_context_coherence',
            'verdict': 1,
            'evidence_text': 'The explanation is a timing question, which matches the buyer context.',
        },
    },
}

_DECOMPOSITION_FEW_SHOT_EXAMPLES: dict[str, dict[str, Any]] = {
    'action_type_verbal_explanation_coherence': {
        'title': 'Action type grounds verbal explanation',
        'input_summary': {
            'action_type': 'QUESTION_BUYER',
            'verbal_explanation': 'Do you need an extension of stay?',
        },
        'output': {
            'metric_name': 'action_type_verbal_explanation_coherence',
            'claims': [
                {
                    'claim_text': 'The verbal explanation should be a direct question consistent with the chosen action type.',
                    'verdict': 1,
                    'evidence_text': 'The verbal explanation is phrased as a direct question.',
                },
            ],
        },
    },
    'action_type_decision_rationale_coherence': {
        'title': 'Action type matches decision rationale',
        'input_summary': {
            'action_type': 'QUESTION_BUYER',
            'decision_rationale': 'Ask directly about the seller timeline.',
        },
        'output': {
            'metric_name': 'action_type_decision_rationale_coherence',
            'claims': [
                {
                    'claim_text': 'The chosen action type should support a direct question about timeline.',
                    'verdict': 1,
                    'evidence_text': 'QUESTION_BUYER matches a rationale about asking directly.',
                },
            ],
        },
    },
    'decision_rationale_verbal_explanation_coherence': {
        'title': 'Decision rationale matches verbal explanation',
        'input_summary': {
            'decision_rationale': 'Ask directly about the seller timeline.',
            'verbal_explanation': 'Do you need an extension of stay?',
        },
        'output': {
            'metric_name': 'decision_rationale_verbal_explanation_coherence',
            'claims': [
                {
                    'claim_text': 'The verbal explanation should express the timing question described in the decision rationale.',
                    'verdict': 1,
                    'evidence_text': 'Both the rationale and the explanation focus on seller timing.',
                },
            ],
        },
    },
    'revelation_coherence': {
        'title': 'No hidden information revealed',
        'input_summary': {
            'hidden_information_rubric': REVELATION_HIDDEN_INFORMATION_RUBRIC,
            'verbal_explanation': 'Let me think about your offer and get back to you.',
        },
        'output': {
            'metric_name': 'revelation_coherence',
            'claims': [
                {
                    'claim_text': 'The public verbal text should not reveal hidden numeric limits.',
                    'verdict': 1,
                    'evidence_text': 'The statement does not include any hidden numbers or limits.',
                },
            ],
        },
    },
}


class EvaluationClaim(BaseModel):
  """One claim-level LLM judgment."""

  claim_text: str = Field(min_length=1)
  verdict: int = Field(ge=0, le=1)
  evidence_text: str = Field(min_length=1)


class DecomposedClaim(BaseModel):
  """One decomposed claim to evaluate in a second step."""

  claim_text: str = Field(min_length=1)


class ClaimDecompositionResponse(BaseModel):
  """Minimal structured output expected from the decomposition step."""

  metric_name: JudgeMetricName
  claims: list[DecomposedClaim] = Field(min_length=1)


class DirectBinaryJudgeResponse(BaseModel):
  """Binary LLM-as-a-judge response for non-decomposed metrics."""

  metric_name: JudgeMetricName
  verdict: int = Field(ge=0, le=1)
  evidence_text: str = Field(min_length=1)


class JudgeMetricResponse(BaseModel):
  """Minimal structured output expected from the judge model."""

  metric_name: JudgeMetricName
  claims: list[EvaluationClaim] = Field(min_length=1)


class TurnEvaluationRecord(BaseModel):
  """One stored turn record used for post-hoc internal coherence evaluation."""

  turn_id: str = Field(min_length=1)
  pair_key: str = Field(min_length=1)
  week_number: int = Field(ge=0)
  pair_round_number: int = Field(ge=0)
  actor_id: str = Field(min_length=1)
  actor_name: str = Field(min_length=1)
  actor_role: str = Field(min_length=1)
  agent_profile_text: str = ''
  action_type: str = ''
  public_action_payload: dict[str, Any] = Field(default_factory=dict)
  public_verbal_text: str = ''
  internal_reasoning: str = ''
  decision_rationale: str = ''
  numeric_action_signature: dict[str, float | int] = Field(default_factory=dict)


def _sample_json_response(
    model: language_model.LanguageModel,
    *,
    prompt: str,
    response_model: type[BaseModel],
) -> str:
  """Sample one structured JSON response with a best-effort schema hint."""
  kwargs = {
      'max_tokens': STRUCTURED_JUDGE_MAX_TOKENS,
      'temperature': STRUCTURED_JUDGE_TEMPERATURE,
      'json_schema': response_model.model_json_schema(),
  }
  try:
    return model.sample_text(prompt=prompt, **kwargs)
  except TypeError:
    return model.sample_text(
        prompt=prompt,
        max_tokens=STRUCTURED_JUDGE_MAX_TOKENS,
        temperature=STRUCTURED_JUDGE_TEMPERATURE,
    )


def _sample_json_responses(
    model: language_model.LanguageModel,
    *,
    prompts: Sequence[str],
    response_model: type[BaseModel],
) -> list[str]:
  """Sample structured JSON responses, batching compatible prompts when possible."""
  prompt_list = list(prompts)
  if not prompt_list:
    return []
  kwargs = {
      'max_tokens': STRUCTURED_JUDGE_MAX_TOKENS,
      'temperature': STRUCTURED_JUDGE_TEMPERATURE,
      'json_schema': response_model.model_json_schema(),
  }
  batch_sampler = getattr(model, 'sample_text_batch', None)
  use_batch = callable(batch_sampler) and len(prompt_list) > 1
  if use_batch:
    try:
      raw_responses = batch_sampler(prompt_list, **kwargs)
      if len(raw_responses) == len(prompt_list):
        return list(raw_responses)
    except TypeError:
      try:
        raw_responses = batch_sampler(
            prompt_list,
            max_tokens=STRUCTURED_JUDGE_MAX_TOKENS,
            temperature=STRUCTURED_JUDGE_TEMPERATURE,
        )
        if len(raw_responses) == len(prompt_list):
          return list(raw_responses)
      except Exception:
        pass
    except Exception:
      pass
  return [
      _sample_json_response(
          model,
          prompt=prompt,
          response_model=response_model,
      )
      for prompt in prompt_list
  ]


def _validate_or_resample_json_response(
    *,
    model: language_model.LanguageModel,
    prompt: str,
    raw_response: str,
    response_model: type[BaseModel],
    error_prefix: str,
) -> BaseModel:
  """Parse one JSON response and retry individually if it is malformed."""
  candidate_response = _normalize_json_response_text(raw_response)
  last_error: ValidationError | None = None
  total_attempts = 1 + max(0, int(STRUCTURED_JUDGE_RETRIES))
  for attempt_index in range(total_attempts):
    try:
      return response_model.model_validate_json(candidate_response)
    except ValidationError as error:
      last_error = error
      if attempt_index >= total_attempts - 1:
        break
      candidate_response = _normalize_json_response_text(
          _sample_json_response(
              model,
              prompt=prompt,
              response_model=response_model,
          )
      )
  raise ValueError(f'{error_prefix}: {last_error}') from last_error


def _normalize_json_response_text(raw_response: str) -> str:
  """Strip common wrappers so valid JSON can still be parsed."""
  text = str(raw_response or '').strip()
  if not text:
    return text
  fenced_match = re.search(
      r'```(?:json)?\s*(.*?)\s*```',
      text,
      flags=re.DOTALL | re.IGNORECASE,
  )
  if fenced_match is not None:
    text = fenced_match.group(1).strip()
  object_start = text.find('{')
  object_end = text.rfind('}')
  if object_start >= 0 and object_end > object_start:
    return text[object_start:object_end + 1]
  return text


def _extract_direct_binary_verdict(raw_response: str) -> int | None:
  """Best-effort extraction of a binary verdict from malformed JSON."""
  match = re.search(r'"verdict"\s*:\s*([01])\b', str(raw_response or ''))
  if match is None:
    return None
  return int(match.group(1))


def _extract_json_string_field_values(
    raw_response: str,
    *,
    field_name: str,
) -> list[str]:
  """Extract closed JSON string field values from malformed responses."""
  matches = re.findall(
      rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"',
      str(raw_response or ''),
      flags=re.DOTALL,
  )
  values: list[str] = []
  for match in matches:
    try:
      values.append(json.loads(f'"{match}"'))
    except json.JSONDecodeError:
      values.append(str(match))
  return values


def _extract_decomposed_claim_texts(raw_response: str) -> list[str]:
  """Extract claim text items from malformed decomposition JSON."""
  claim_texts = _extract_json_string_field_values(
      raw_response,
      field_name='claim_text',
  )
  return [claim_text for claim_text in claim_texts if claim_text.strip()]


def _extract_claim_verdicts(raw_response: str) -> list[int]:
  """Extract verdict integers from malformed evaluation JSON."""
  return [
      int(match)
      for match in re.findall(r'"verdict"\s*:\s*([01])\b', str(raw_response or ''))
  ]


def _parse_fallback_claim_lines(raw_response: str) -> list[str]:
  """Parse plain-text fallback claim lines."""
  claim_lines: list[str] = []
  for raw_line in str(raw_response or '').splitlines():
    line = raw_line.strip()
    if not line:
      continue
    line = re.sub(r'^\s*(?:[-*]|\d+[.)])\s*', '', line)
    if line:
      claim_lines.append(line)
  return claim_lines[:3]


def _default_claim_text(metric_name: str) -> str:
  """Return a conservative default claim for one metric."""
  defaults = {
      'action_type_verbal_explanation_coherence': (
          'The verbal explanation should match the chosen action type.'
      ),
      'action_type_decision_rationale_coherence': (
          'The decision rationale should match the chosen action type.'
      ),
      'decision_rationale_verbal_explanation_coherence': (
          'The verbal explanation should reflect the decision rationale.'
      ),
      'revelation_coherence': (
          'The verbal explanation should not reveal hidden information.'
      ),
  }
  return defaults.get(metric_name, 'The turn should satisfy the requested metric.')


def _fallback_claim_decomposition_response(
    *,
    model: language_model.LanguageModel,
    prompt: str,
    metric_name: str,
    raw_response: str,
) -> ClaimDecompositionResponse:
  """Recover decomposition claims without requiring valid JSON output."""
  extracted_claim_texts = _extract_decomposed_claim_texts(raw_response)
  if extracted_claim_texts:
    return ClaimDecompositionResponse(
        metric_name=metric_name,
        claims=[
            DecomposedClaim(claim_text=claim_text)
            for claim_text in extracted_claim_texts[:3]
        ],
    )

  fallback_prompt = (
      f'{prompt}\n\n'
      '# Final Instruction\n'
      'Return ONLY 1 to 3 short claim lines. No JSON. No bullets. No numbering.\n'
      'Claims:'
  )
  last_sample = ''
  total_attempts = 1 + max(0, int(STRUCTURED_JUDGE_RETRIES))
  for _ in range(total_attempts):
    last_sample = str(
        model.sample_text(
            prompt=fallback_prompt,
            max_tokens=STRUCTURED_JUDGE_MAX_TOKENS,
            temperature=STRUCTURED_JUDGE_TEMPERATURE,
        )
    ).strip()
    parsed_lines = _parse_fallback_claim_lines(last_sample)
    if parsed_lines:
      return ClaimDecompositionResponse(
          metric_name=metric_name,
          claims=[
              DecomposedClaim(claim_text=claim_text)
              for claim_text in parsed_lines
          ],
      )

  return ClaimDecompositionResponse(
      metric_name=metric_name,
      claims=[DecomposedClaim(claim_text=_default_claim_text(metric_name))],
  )


def _fallback_claim_evaluation_response(
    *,
    model: language_model.LanguageModel,
    prompt: str,
    metric_name: str,
    expected_claims: Sequence[str],
    raw_response: str,
) -> JudgeMetricResponse:
  """Recover claim verdicts without requiring valid JSON output."""
  verdicts = _extract_claim_verdicts(raw_response)
  if verdicts:
    extracted_evidence = _extract_json_string_field_values(
        raw_response,
        field_name='evidence_text',
    )
    recovered_claims: list[EvaluationClaim] = []
    for index, claim_text in enumerate(expected_claims):
      verdict = verdicts[index] if index < len(verdicts) else 0
      evidence_text = (
          extracted_evidence[index]
          if index < len(extracted_evidence)
          and extracted_evidence[index].strip()
          else 'Recovered verdict from malformed structured judge output.'
      )
      recovered_claims.append(
          EvaluationClaim(
              claim_text=claim_text,
              verdict=verdict,
              evidence_text=evidence_text,
          )
      )
    return JudgeMetricResponse(
        metric_name=metric_name,
        claims=recovered_claims,
    )

  fallback_prompt = (
      f'{prompt}\n\n'
      '# Final Instruction\n'
      'Return ONLY one line per supplied claim, in the same order, using the '
      'format `0|short evidence` or `1|short evidence`. No JSON.\n'
      'Answers:'
  )
  last_sample = ''
  total_attempts = 1 + max(0, int(STRUCTURED_JUDGE_RETRIES))
  for _ in range(total_attempts):
    last_sample = str(
        model.sample_text(
            prompt=fallback_prompt,
            max_tokens=STRUCTURED_JUDGE_MAX_TOKENS,
            temperature=STRUCTURED_JUDGE_TEMPERATURE,
        )
    ).strip()
    parsed_claims: list[EvaluationClaim] = []
    for index, raw_line in enumerate(last_sample.splitlines()):
      if index >= len(expected_claims):
        break
      line = raw_line.strip()
      match = re.match(r'^([01])\s*\|\s*(.+)$', line)
      if match is None:
        continue
      parsed_claims.append(
          EvaluationClaim(
              claim_text=expected_claims[index],
              verdict=int(match.group(1)),
              evidence_text=match.group(2).strip(),
          )
      )
    if parsed_claims:
      while len(parsed_claims) < len(expected_claims):
        claim_index = len(parsed_claims)
        parsed_claims.append(
            EvaluationClaim(
                claim_text=expected_claims[claim_index],
                verdict=0,
                evidence_text='Missing fallback claim evaluation; defaulted to fail.',
            )
        )
      return JudgeMetricResponse(
          metric_name=metric_name,
          claims=parsed_claims,
      )

  return JudgeMetricResponse(
      metric_name=metric_name,
      claims=[
          EvaluationClaim(
              claim_text=claim_text,
              verdict=0,
              evidence_text='Claim evaluation defaulted to fail after malformed output.',
          )
          for claim_text in expected_claims
      ],
  )


def _fallback_direct_binary_response(
    *,
    model: language_model.LanguageModel,
    prompt: str,
    metric_name: str,
    raw_response: str,
) -> DirectBinaryJudgeResponse:
  """Recover a direct binary verdict without requiring valid JSON output."""
  extracted_verdict = _extract_direct_binary_verdict(raw_response)
  if extracted_verdict is not None:
    return DirectBinaryJudgeResponse(
        metric_name=metric_name,
        verdict=extracted_verdict,
        evidence_text=(
            'Recovered verdict from malformed structured judge output.'
        ),
    )

  fallback_prompt = (
      f'{prompt}\n\n'
      '# Final Instruction\n'
      'Return ONLY one character: `1` if the answer should pass, or `0` if it '
      'should fail. Do not return JSON. Do not explain.\n'
      'Answer:'
  )
  last_sample = ''
  total_attempts = 1 + max(0, int(STRUCTURED_JUDGE_RETRIES))
  for _ in range(total_attempts):
    last_sample = str(
        model.sample_text(
            prompt=fallback_prompt,
            max_tokens=DIRECT_BINARY_FALLBACK_MAX_TOKENS,
            temperature=STRUCTURED_JUDGE_TEMPERATURE,
        )
    ).strip()
    verdict_match = re.search(r'\b([01])\b', last_sample)
    if verdict_match is not None:
      return DirectBinaryJudgeResponse(
          metric_name=metric_name,
          verdict=int(verdict_match.group(1)),
          evidence_text='Fallback scalar direct-judge verdict.',
      )
  raise ValueError(
      'Unable to recover direct binary judge verdict from malformed output. '
      f'Last fallback sample: {last_sample}'
  )


def _default_direct_binary_response(metric_name: str) -> DirectBinaryJudgeResponse:
  """Conservative last-resort response when all direct judge recovery fails."""
  return DirectBinaryJudgeResponse(
      metric_name=metric_name,
      verdict=0,
      evidence_text='Direct judge defaulted to fail after malformed output.',
  )


def _normalize_text(text: str) -> str:
  lowered = str(text or '').casefold()
  lowered = re.sub(r'[^a-z0-9\s]+', ' ', lowered)
  lowered = re.sub(r'\s+', ' ', lowered).strip()
  return lowered


def _text_similarity(
    text_a: str,
    text_b: str,
    *,
    similarity_fn: Callable[[str, str], float] | None = None,
    dense_embedding_model: Any | None = None,
) -> float:
  """Compute similarity using the supplied function or embedding model."""
  normalized_a = _normalize_text(text_a)
  normalized_b = _normalize_text(text_b)
  if normalized_a == normalized_b:
    return 1.0
  if not normalized_a or not normalized_b:
    return 0.0
  if similarity_fn is not None:
    return max(0.0, min(1.0, float(similarity_fn(normalized_a, normalized_b))))
  if dense_embedding_model is not None and hasattr(dense_embedding_model, 'encode'):
    embeddings = dense_embedding_model.encode(
        [normalized_a, normalized_b],
        normalize_embeddings=True,
    )
    return float(np.dot(embeddings[0], embeddings[1]))
  return 0.0


def _markdown_json(value: Any) -> str:
  """Render compact JSON for prompts."""
  return json.dumps(
      value,
      ensure_ascii=False,
      sort_keys=True,
      separators=(',', ':'),
  )


def _few_shot_output(example: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
  """Build the stage-specific few-shot output payload."""
  evaluation_output = dict(example['output'])
  if stage == 'decomposition':
    return {
        'metric_name': evaluation_output['metric_name'],
        'claims': [
            {'claim_text': str(claim['claim_text'])}
            for claim in evaluation_output['claims']
        ],
    }
  if stage == 'evaluation':
    return evaluation_output
  raise ValueError(f'Unsupported few-shot stage: {stage}')


def _decomposition_few_shot_markdown(metric_name: str, *, stage: str) -> str:
  """Render one compact few-shot example for decomposition-style metrics."""
  example = _DECOMPOSITION_FEW_SHOT_EXAMPLES.get(metric_name)
  if example is None:
    return ''
  return '\n'.join([
      '## Few-Shot Examples',
      f'### Example 1: {example["title"]}',
      '**Input summary**',
      '```json',
      _markdown_json(example['input_summary']),
      '```',
      '**Output JSON**',
      '```json',
      _markdown_json(_few_shot_output(example, stage=stage)),
      '```',
  ])


def _direct_binary_few_shot_markdown(metric_name: str) -> str:
  """Render one compact few-shot example for direct binary metrics."""
  example = _DIRECT_BINARY_FEW_SHOT_EXAMPLES.get(metric_name)
  if example is None:
    return ''
  return '\n'.join([
      '## Few-Shot Examples',
      f'### Example 1: {example["title"]}',
      '**Input summary**',
      '```json',
      _markdown_json(example['input_summary']),
      '```',
      '**Output JSON**',
      '```json',
      _markdown_json(example['output']),
      '```',
  ])


def _claim_decomposition_prompt_for_record(
    *,
    metric_name: str,
    record: TurnEvaluationRecord,
) -> str:
  """Build the decomposition prompt for one metric and one turn."""
  if metric_name not in DECOMPOSITION_METRICS:
    raise ValueError(f'Metric {metric_name} does not use claim decomposition.')
  sections = [
      '# Internal Coherence Claim Decomposition',
      '## Output Contract',
      '- Return JSON only.',
      '- Generate claim_text items only.',
      '- Return 1 to 3 short, non-overlapping claims.',
      '## Context',
      f'Metric name: {metric_name}',
      _decomposition_few_shot_markdown(metric_name, stage='decomposition'),
  ]
  if metric_name == 'action_type_verbal_explanation_coherence':
    sections.extend([
      '## Task',
      'Extract claims for whether the action type supports the verbal explanation.',
      '## Decompose This Turn',
      f'**Action type**: {record.action_type}',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "action_type": record.action_type,
          "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  elif metric_name == 'action_type_decision_rationale_coherence':
    sections.extend([
      '## Task',
      'Extract claims for whether the action type supports the decision rationale.',
      '## Decompose This Turn',
      f'**Action type**: {record.action_type}',
      f'**Decision rationale**: {record.decision_rationale}',
      '## Input',
      '```json',
      _markdown_json({
          "action_type": record.action_type,
          "decision_rationale": record.decision_rationale,
        }),
        '```',
    ])
  elif metric_name == 'decision_rationale_verbal_explanation_coherence':
    sections.extend([
      '## Task',
      'Extract claims for whether the decision rationale supports the verbal explanation.',
      '## Decompose This Turn',
      f'**Decision rationale**: {record.decision_rationale}',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "decision_rationale": record.decision_rationale,
          "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  elif metric_name == 'revelation_coherence':
    sections.extend([
      '## Task',
      'Extract must-not-reveal claims implied by the rubric for this verbal explanation.',
      '## Decompose This Turn',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "hidden_information_rubric": REVELATION_HIDDEN_INFORMATION_RUBRIC,
          "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  else:
    raise ValueError(f'Unsupported judge metric: {metric_name}')
  return '\n\n'.join(section for section in sections if section)


def _direct_binary_prompt_for_record(
    *,
    metric_name: str,
    record: TurnEvaluationRecord,
) -> str:
  """Build the prompt for a direct binary coherence judge."""
  if metric_name not in DIRECT_BINARY_METRICS:
    raise ValueError(f'Metric {metric_name} is not a direct binary metric.')
  sections = [
      '# Binary Coherence Judge',
      '## Output Contract',
      '- Return JSON only.',
      '- verdict must be 0 or 1.',
      '- Keep evidence_text short and concrete.',
      '## Context',
      f'Metric name: {metric_name}',
      _direct_binary_few_shot_markdown(metric_name),
  ]
  if metric_name == 'decision_rationale_context_coherence':
    sections.extend([
        '## Task',
        'Decide whether the decision rationale is supported by the turn context.',
        '## Evaluate This Turn',
        '## Input',
        '```json',
        _markdown_json({
            "agent_profile_text": record.agent_profile_text,
            "action_type": record.action_type,
            "public_action_payload": record.public_action_payload,
            "verbal_explanation": record.public_verbal_text,
            "decision_rationale": record.decision_rationale,
        }),
        '```',
    ])
  elif metric_name == 'verbal_explanation_context_coherence':
    sections.extend([
        '## Task',
        'Decide whether the verbal explanation is supported by the turn context.',
        '## Evaluate This Turn',
        '## Input',
        '```json',
        _markdown_json({
            "agent_profile_text": record.agent_profile_text,
            "action_type": record.action_type,
            "public_action_payload": record.public_action_payload,
            "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  else:
    raise ValueError(f'Unsupported direct metric: {metric_name}')
  return '\n\n'.join(section for section in sections if section)


def _claim_evaluation_prompt_for_record(
    *,
    metric_name: str,
    record: TurnEvaluationRecord,
    claims: Sequence[str],
) -> str:
  """Build the evaluation prompt for one metric, one turn, and fixed claims."""
  if metric_name not in DECOMPOSITION_METRICS:
    raise ValueError(f'Metric {metric_name} does not use claim evaluation.')
  sections = [
      '# Internal Coherence Claim Evaluation',
      '## Output Contract',
      '- Return JSON only.',
      '- Evaluate the supplied claims only.',
      '- Return one result per claim in the same order as the supplied claims.',
      '- For each claim return claim_text, verdict (0 or 1), and short evidence_text.',
      '## Context',
      f'Metric name: {metric_name}',
      _decomposition_few_shot_markdown(metric_name, stage='evaluation'),
      '## Claims To Evaluate',
      '```json',
      _markdown_json([{"claim_text": claim_text} for claim_text in claims]),
      '```',
  ]
  if metric_name == 'action_type_verbal_explanation_coherence':
    sections.extend([
      '## Task',
      'Judge whether the action type grounds the verbal explanation for each claim.',
      '## Evaluate This Turn',
      f'**Action type**: {record.action_type}',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "action_type": record.action_type,
          "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  elif metric_name == 'action_type_decision_rationale_coherence':
    sections.extend([
      '## Task',
      'Judge whether the action type grounds the decision rationale for each claim.',
      '## Evaluate This Turn',
      f'**Action type**: {record.action_type}',
      f'**Decision rationale**: {record.decision_rationale}',
      '## Input',
      '```json',
      _markdown_json({
          "action_type": record.action_type,
          "decision_rationale": record.decision_rationale,
        }),
        '```',
    ])
  elif metric_name == 'decision_rationale_verbal_explanation_coherence':
    sections.extend([
      '## Task',
      'Judge whether the decision rationale grounds the verbal explanation for each claim.',
      '## Evaluate This Turn',
      f'**Decision rationale**: {record.decision_rationale}',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "decision_rationale": record.decision_rationale,
          "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  elif metric_name == 'revelation_coherence':
    sections.extend([
      '## Task',
      'Judge the claims against the verbal explanation only.',
      '## Evaluate This Turn',
      f'**Verbal explanation**: {record.public_verbal_text}',
      '## Input',
      '```json',
      _markdown_json({
          "hidden_information_rubric": REVELATION_HIDDEN_INFORMATION_RUBRIC,
            "verbal_explanation": record.public_verbal_text,
        }),
        '```',
    ])
  else:
    raise ValueError(f'Unsupported judge metric: {metric_name}')
  return '\n\n'.join(section for section in sections if section)


def _validate_turn_record(raw_record: Mapping[str, Any]) -> TurnEvaluationRecord:
  """Validate one stored turn record."""
  return TurnEvaluationRecord.model_validate(raw_record)


def _align_evaluated_claims(
    expected_claims: Sequence[str],
    observed_claims: Sequence[EvaluationClaim],
) -> list[dict[str, Any]]:
  """Align evaluated claims to the original decomposition by position."""
  aligned_claims: list[dict[str, Any]] = []
  observed_list = list(observed_claims)
  for index, expected_claim_text in enumerate(expected_claims):
    if index < len(observed_list):
      observed_claim = observed_list[index]
      aligned_claims.append({
          'claim_text': expected_claim_text,
          'verdict': int(observed_claim.verdict),
          'evidence_text': observed_claim.evidence_text,
      })
      continue
    aligned_claims.append({
        'claim_text': expected_claim_text,
        'verdict': 0,
        'evidence_text': 'Missing evaluated claim; defaulted to fail.',
    })
  return aligned_claims


def _judge_metric_for_records(
    *,
    model: language_model.LanguageModel,
    metric_name: str,
    records: Sequence[TurnEvaluationRecord],
) -> dict[str, dict[str, Any]]:
  """Run one judge metric over many records, batching LLM calls when possible."""
  if not records:
    return {}

  if metric_name in DIRECT_BINARY_METRICS:
    prompts = [
        _direct_binary_prompt_for_record(metric_name=metric_name, record=record)
        for record in records
    ]
    raw_responses = _sample_json_responses(
        model,
        prompts=prompts,
        response_model=DirectBinaryJudgeResponse,
    )
    results: dict[str, dict[str, Any]] = {}
    for record, prompt, raw_response in zip(
        records,
        prompts,
        raw_responses,
        strict=True,
    ):
      try:
        parsed = _validate_or_resample_json_response(
            model=model,
            prompt=prompt,
            raw_response=raw_response,
            response_model=DirectBinaryJudgeResponse,
            error_prefix=(
                f'Invalid {metric_name} direct judge output for turn '
                f'{record.turn_id}'
            ),
        )
      except Exception:
        try:
          parsed = _fallback_direct_binary_response(
              model=model,
              prompt=prompt,
              metric_name=metric_name,
              raw_response=raw_response,
          )
        except Exception:
          parsed = _default_direct_binary_response(metric_name)
      if parsed.metric_name != metric_name:
        parsed = DirectBinaryJudgeResponse(
            metric_name=metric_name,
            verdict=int(parsed.verdict),
            evidence_text=str(parsed.evidence_text),
        )
      results[record.turn_id] = {
          'global_verdict': int(parsed.verdict),
          'evidence_text': parsed.evidence_text,
      }
    return results

  decomposition_prompts = [
      _claim_decomposition_prompt_for_record(metric_name=metric_name, record=record)
      for record in records
  ]
  raw_decompositions = _sample_json_responses(
      model,
      prompts=decomposition_prompts,
      response_model=ClaimDecompositionResponse,
  )
  claims_by_turn_id: dict[str, list[str]] = {}
  for record, prompt, raw_decomposition in zip(
      records,
      decomposition_prompts,
      raw_decompositions,
      strict=True,
  ):
    try:
      decomposed = _validate_or_resample_json_response(
          model=model,
          prompt=prompt,
          raw_response=raw_decomposition,
          response_model=ClaimDecompositionResponse,
          error_prefix=(
              f'Invalid {metric_name} decomposition output for turn '
              f'{record.turn_id}'
          ),
      )
    except Exception:
      decomposed = _fallback_claim_decomposition_response(
          model=model,
          prompt=prompt,
          metric_name=metric_name,
          raw_response=raw_decomposition,
      )
    if decomposed.metric_name != metric_name:
      raise ValueError(
          f'Decomposition output mismatch for {metric_name} turn {record.turn_id}.'
      )
    claims_by_turn_id[record.turn_id] = [
        claim.claim_text for claim in decomposed.claims
    ]

  evaluation_prompts = [
      _claim_evaluation_prompt_for_record(
          metric_name=metric_name,
          record=record,
          claims=claims_by_turn_id[record.turn_id],
      )
      for record in records
  ]
  raw_evaluations = _sample_json_responses(
      model,
      prompts=evaluation_prompts,
      response_model=JudgeMetricResponse,
  )
  results: dict[str, dict[str, Any]] = {}
  for record, prompt, raw_evaluation in zip(
      records,
      evaluation_prompts,
      raw_evaluations,
      strict=True,
  ):
    expected_claims = claims_by_turn_id[record.turn_id]
    try:
      parsed = _validate_or_resample_json_response(
          model=model,
          prompt=prompt,
          raw_response=raw_evaluation,
          response_model=JudgeMetricResponse,
          error_prefix=(
              f'Invalid {metric_name} judge output for turn {record.turn_id}'
          ),
      )
    except Exception:
      parsed = _fallback_claim_evaluation_response(
          model=model,
          prompt=prompt,
          metric_name=metric_name,
          expected_claims=expected_claims,
          raw_response=raw_evaluation,
      )
    if parsed.metric_name != metric_name:
      raise ValueError(
          f'Judge output mismatch for {metric_name} turn {record.turn_id}.'
      )
    claims = _align_evaluated_claims(expected_claims, parsed.claims)
    results[record.turn_id] = {
        'claims': claims,
        'global_verdict': int(all(int(claim['verdict']) == 1 for claim in claims)),
    }
  return results


def collect_evaluation_records(
    *,
    week_summaries: Sequence[Mapping[str, Any]] = (),
    archived_pair_records: Sequence[Mapping[str, Any]] = (),
    archive_jsonl_path: str | None = None,
) -> list[dict[str, Any]]:
  """Collect and deduplicate evaluation records from summaries and archives."""
  merged_records: dict[str, dict[str, Any]] = {}

  def _store_many(candidate_records: Sequence[Any]) -> None:
    for candidate in candidate_records:
      if not isinstance(candidate, Mapping):
        continue
      record = _validate_turn_record(candidate)
      merged_records.setdefault(record.turn_id, record.model_dump(mode='json'))

  for summary in week_summaries:
    if not isinstance(summary, Mapping):
      continue
    negotiation = summary.get('negotiation', {})
    if not isinstance(negotiation, Mapping):
      continue
    records = negotiation.get('evaluation_records', ())
    if isinstance(records, Sequence) and not isinstance(records, str):
      _store_many(records)

  for archive_record in archived_pair_records:
    if not isinstance(archive_record, Mapping):
      continue
    records = archive_record.get('evaluation_records', ())
    if isinstance(records, Sequence) and not isinstance(records, str):
      _store_many(records)

  if archive_jsonl_path:
    archive_path = Path(archive_jsonl_path)
    if archive_path.exists():
      for line in archive_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
          continue
        parsed = json.loads(line)
        if not isinstance(parsed, Mapping):
          continue
        records = parsed.get('evaluation_records', ())
        if isinstance(records, Sequence) and not isinstance(records, str):
          _store_many(records)

  return sorted(
      merged_records.values(),
      key=lambda record: (
          int(record.get('week_number', 0)),
          int(record.get('pair_round_number', 0)),
          str(record.get('turn_id', '')),
      ),
  )


def _judge_metric_for_record(
    *,
    model: language_model.LanguageModel,
    metric_name: str,
    record: TurnEvaluationRecord,
) -> dict[str, Any]:
  """Run one judge metric, directly or via decomposition depending on the metric."""
  return _judge_metric_for_records(
      model=model,
      metric_name=metric_name,
      records=(record,),
  )[record.turn_id]


def _is_repeated_turn(
    record: TurnEvaluationRecord,
    prior_records: Sequence[TurnEvaluationRecord],
    *,
    similarity_threshold: float,
    similarity_fn: Callable[[str, str], float] | None = None,
    dense_embedding_model: Any | None = None,
) -> int:
  """Return 1 when the turn repeats a prior same-actor same-pair action."""
  normalized_text = _normalize_text(record.public_verbal_text)
  for prior in prior_records:
    if prior.action_type != record.action_type:
      continue
    if record.action_type in MONETARY_ACTION_TYPES:
      if dict(prior.numeric_action_signature) != dict(record.numeric_action_signature):
        continue
      if normalized_text == _normalize_text(prior.public_verbal_text):
        return 1
      similarity = _text_similarity(
          record.public_verbal_text,
          prior.public_verbal_text,
          similarity_fn=similarity_fn,
          dense_embedding_model=dense_embedding_model,
      )
      if similarity >= similarity_threshold:
        return 1
      continue
    if normalized_text == _normalize_text(prior.public_verbal_text):
      return 1
    similarity = _text_similarity(
        record.public_verbal_text,
        prior.public_verbal_text,
        similarity_fn=similarity_fn,
        dense_embedding_model=dense_embedding_model,
    )
    if similarity >= similarity_threshold:
      return 1
  return 0


def evaluate_internal_coherence(
    *,
    model: language_model.LanguageModel,
    week_summaries: Sequence[Mapping[str, Any]] = (),
    archived_pair_records: Sequence[Mapping[str, Any]] = (),
    archive_jsonl_path: str | None = None,
    similarity_threshold: float = 0.94,
    similarity_fn: Callable[[str, str], float] | None = None,
    dense_embedding_model: Any | None = None,
) -> dict[str, Any]:
  """Evaluate internal coherence across stored negotiation turn records."""
  raw_records = collect_evaluation_records(
      week_summaries=week_summaries,
      archived_pair_records=archived_pair_records,
      archive_jsonl_path=archive_jsonl_path,
  )
  records = [
      TurnEvaluationRecord.model_validate(raw_record)
      for raw_record in raw_records
  ]
  metric_results_by_turn_id: dict[str, dict[str, dict[str, Any]]] = {}
  metric_errors: dict[str, str] = {}
  for metric_name in JUDGE_METRICS:
    try:
      metric_results_by_turn_id[metric_name] = _judge_metric_for_records(
          model=model,
          metric_name=metric_name,
          records=records,
      )
    except Exception as error:  # pylint: disable=broad-exception-caught
      metric_errors[metric_name] = str(error)

  per_turn_metrics: list[dict[str, Any]] = []
  successful_metric_names = tuple(metric_results_by_turn_id.keys())
  judge_scores_by_metric: dict[str, list[int]] = {
      metric_name: [] for metric_name in successful_metric_names
  }
  judge_scores_by_metric_and_actor: dict[str, dict[str, list[int]]] = {
      metric_name: defaultdict(list) for metric_name in successful_metric_names
  }
  per_agent_metrics: dict[str, dict[str, Any]] = {}
  prior_records_by_actor_pair: dict[
      tuple[str, str],
      list[TurnEvaluationRecord],
  ] = defaultdict(list)

  repeated_turns_total = 0
  total_turns = 0
  repetition_rates_by_actor: dict[str, list[int]] = defaultdict(list)

  for record in records:
    agent_metrics = per_agent_metrics.setdefault(
        record.actor_id,
        {
            'actor_id': record.actor_id,
            'actor_name': record.actor_name,
            'actor_role': record.actor_role,
        },
    )
    turn_metrics = {
        'turn_id': record.turn_id,
        'pair_key': record.pair_key,
        'week_number': record.week_number,
        'pair_round_number': record.pair_round_number,
        'actor_id': record.actor_id,
        'actor_name': record.actor_name,
        'actor_role': record.actor_role,
    }

    for metric_name in successful_metric_names:
      metric_result = metric_results_by_turn_id[metric_name][record.turn_id]
      turn_metrics[metric_name] = metric_result
      judge_scores_by_metric[metric_name].append(metric_result['global_verdict'])
      judge_scores_by_metric_and_actor[metric_name][record.actor_id].append(
          metric_result['global_verdict']
      )

    prior_records = prior_records_by_actor_pair[(record.actor_id, record.pair_key)]
    is_repeated = _is_repeated_turn(
        record,
        prior_records,
        similarity_threshold=similarity_threshold,
        similarity_fn=similarity_fn,
        dense_embedding_model=dense_embedding_model,
    )
    turn_metrics['repetition_rate'] = {
        'is_repeated': is_repeated,
    }
    prior_records.append(record)
    repeated_turns_total += is_repeated
    total_turns += 1
    repetition_rates_by_actor[record.actor_id].append(is_repeated)

    per_turn_metrics.append(turn_metrics)
    agent_metrics.setdefault('_turn_count', 0)
    agent_metrics['_turn_count'] += 1

  overall_metrics: dict[str, Any] = {}
  for metric_name in successful_metric_names:
    actor_means = [
        sum(scores) / len(scores)
        for scores in judge_scores_by_metric_and_actor[metric_name].values()
        if scores
    ]
    overall_metrics[metric_name] = {
        'micro_average': (
            sum(judge_scores_by_metric[metric_name])
            / len(judge_scores_by_metric[metric_name])
            if judge_scores_by_metric[metric_name]
            else 0.0
        ),
        'macro_average': (
            sum(actor_means) / len(actor_means) if actor_means else 0.0
        ),
    }
  for metric_name, error_message in metric_errors.items():
    overall_metrics[metric_name] = {
        'status': 'error',
        'error_message': error_message,
    }

  repetition_actor_rates: list[float] = []
  for actor_id, agent_metrics in per_agent_metrics.items():
    for metric_name in successful_metric_names:
      scores = judge_scores_by_metric_and_actor[metric_name].get(actor_id, [])
      agent_metrics[metric_name] = {
          'mean_verdict': (sum(scores) / len(scores)) if scores else 0.0,
          'num_turns': len(scores),
      }
    repeated_turns = sum(repetition_rates_by_actor.get(actor_id, ()))
    repetition_total = len(repetition_rates_by_actor.get(actor_id, ()))
    repetition_rate = (
        repeated_turns / repetition_total if repetition_total else 0.0
    )
    repetition_actor_rates.append(repetition_rate)
    agent_metrics['repetition_rate'] = {
        'repeated_turns': repeated_turns,
        'total_turns': repetition_total,
        'rate': repetition_rate,
    }
    agent_metrics.pop('_turn_count', None)

  overall_metrics['repetition_rate'] = {
      'micro_average': (
          repeated_turns_total / total_turns if total_turns else 0.0
      ),
      'macro_average': (
          sum(repetition_actor_rates) / len(repetition_actor_rates)
          if repetition_actor_rates
          else 0.0
      ),
  }

  return {
      'overall_metrics': overall_metrics,
      'per_agent_metrics': per_agent_metrics,
      'per_turn_metrics': per_turn_metrics,
      'metric_errors': metric_errors,
  }
