import copy
import json
import re
import unittest
from unittest import mock

from concordia.language_model import no_language_model
from concordia.prefabs.game_master.negotiation.components import (
    hdb_negotiation,
    hdb_negotiation_evaluation,
    hdb_negotiation_helpers,
)


def _buyer_participant_spec() -> dict[str, object]:
  return {
      'id': 'buyer_001',
      'name': 'Buyer 1',
      'role': 'buyer',
      'description': 'Cautious buyer profile.',
      'preferences': {
          'preferences': [
              {
                  'category': 'flat_type',
                  'description': '3-Room',
                  'strength': 1.0,
              },
              {
                  'category': 'town',
                  'description': 'Choa Chu Kang',
                  'strength': 0.8,
              },
          ],
      },
      'budget': {
          'min_price': 450000.0,
          'max_price': 650000.0,
      },
  }


def _seller_participant_spec() -> dict[str, object]:
  return {
      'id': 'seller_001',
      'name': 'Seller 1',
      'role': 'seller',
      'description': 'Patient seller profile.',
      'flat': {
          'flat_type': '3-Room',
          'address': 'Blk 123 Test Avenue Singapore 680123',
          'description': 'Test flat.',
          'town': 'Choa Chu Kang',
          'storey_range': '05 to 08',
          'remaining_lease': 78.0,
          'contra': False,
          'extension_of_stay': False,
          'ethnic_eligibility': 'No quota limit',
          'spr_eligibility': 'True',
          'floor_area_sqm': 68.0,
      },
      'expectations': {
          'min_price': 500000.0,
          'max_price': 530000.0,
      },
  }


class _BoundCanonicalEntity:

  def __init__(self, *, name: str, player_id: str):
    self.name = name
    self._hdb_player_id = player_id


class _FakeJudgeModel(no_language_model.NoLanguageModel):

  def __init__(self):
    self.prompts: list[str] = []
    self.batch_calls: list[dict[str, object]] = []

  @staticmethod
  def _claims_from_evaluation_prompt(prompt: str) -> list[str]:
    match = re.search(
        r'## Claims To Evaluate\s+```json\s+(.*?)\s+```',
        prompt,
        re.DOTALL,
    )
    if not match:
      return []
    try:
      payload = json.loads(match.group(1))
    except json.JSONDecodeError:
      return []
    if not isinstance(payload, list):
      return []
    claim_texts: list[str] = []
    for item in payload:
      if isinstance(item, dict) and 'claim_text' in item:
        claim_texts.append(str(item['claim_text']))
    return claim_texts

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    self.prompts.append(prompt)
    metric_match = re.search(r'Metric name: ([a-z_]+)', prompt)
    metric_name = metric_match.group(1) if metric_match else ''
    if '# Binary Coherence Judge' in prompt:
      return json.dumps({
          'metric_name': metric_name,
          'verdict': 1,
          'evidence_text': 'Binary evidence grounded in the provided turn.',
      })
    if '# Internal Coherence Claim Decomposition' in prompt:
      return json.dumps({
          'metric_name': metric_name,
          'claims': [{
              'claim_text': f'{metric_name} claim',
          }],
      })
    claim_matches = self._claims_from_evaluation_prompt(prompt)
    return json.dumps({
        'metric_name': metric_name,
        'claims': [{
            'claim_text': (
                claim_matches[0]
                if claim_matches
                else f'{metric_name} claim'
            ),
            'verdict': 1,
            'evidence_text': 'Evidence grounded in the provided turn.',
        }],
    })

  def sample_text_batch(self, prompts, **kwargs):
    prompt_list = list(prompts)
    self.batch_calls.append({
        'prompts': prompt_list,
        'kwargs': dict(kwargs),
    })
    outputs: list[str] = []
    for index, prompt in enumerate(prompt_list):
      self.prompts.append(prompt)
      metric_match = re.search(r'Metric name: ([a-z_]+)', prompt)
      metric_name = metric_match.group(1) if metric_match else ''
      if '# Binary Coherence Judge' in prompt:
        outputs.append(json.dumps({
            'metric_name': metric_name,
            'verdict': 1,
            'evidence_text': 'Binary evidence grounded in the provided turn.',
        }))
        continue
      if '# Internal Coherence Claim Decomposition' in prompt:
        outputs.append(json.dumps({
            'metric_name': metric_name,
            'claims': [{
                'claim_text': f'{metric_name} claim {index}',
            }],
        }))
        continue
      claim_matches = self._claims_from_evaluation_prompt(prompt)
      verdict = 1
      if (
          metric_name == 'action_type_verbal_explanation_coherence'
          and index == 1
      ):
        verdict = 0
      outputs.append(json.dumps({
          'metric_name': metric_name,
          'claims': [{
              'claim_text': (
                  claim_matches[0]
                  if claim_matches
                  else f'{metric_name} claim {index}'
              ),
              'verdict': verdict,
              'evidence_text': 'Evidence grounded in the provided turn.',
          }],
      }))
    return outputs


class _EmptyClaimsJudgeModel(no_language_model.NoLanguageModel):

  def sample_text(self, prompt: str, **kwargs) -> str:
    del prompt, kwargs
    return json.dumps({
        'metric_name': 'action_type_verbal_explanation_coherence',
        'claims': [],
    })


class _ClaimRewritingJudgeModel(no_language_model.NoLanguageModel):

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    metric_match = re.search(r'Metric name: ([a-z_]+)', prompt)
    metric_name = metric_match.group(1) if metric_match else ''
    if '# Internal Coherence Claim Decomposition' in prompt:
      return json.dumps({
          'metric_name': metric_name,
          'claims': [{
              'claim_text': 'Original decomposed claim.',
          }],
      })
    return json.dumps({
        'metric_name': metric_name,
        'claims': [{
            'claim_text': 'Rewritten claim text.',
            'verdict': 1,
            'evidence_text': 'Evidence grounded in the provided turn.',
        }],
    })


class _MalformedEvaluationJudgeModel(no_language_model.NoLanguageModel):

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    metric_match = re.search(r'Metric name: ([a-z_]+)', prompt)
    metric_name = metric_match.group(1) if metric_match else ''
    if '# Internal Coherence Claim Decomposition' in prompt:
      return json.dumps({
          'metric_name': metric_name,
          'claims': [{
              'claim_text': 'The verbal explanation should match the action type.',
          }],
      })
    return (
        '{"metric_name":"action_type_verbal_explanation_coherence",'
        '"claims":[{"claim_text":"The verbal explanation should match the '
        'action type.","verdict":1,"evidence_text":"The question format is '
        'consistent with the action type.'
    )


class _MalformedDecompositionJudgeModel(no_language_model.NoLanguageModel):

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    metric_match = re.search(r'Metric name: ([a-z_]+)', prompt)
    metric_name = metric_match.group(1) if metric_match else ''
    if '# Internal Coherence Claim Decomposition' in prompt:
      return (
          '{"metric_name":"action_type_verbal_explanation_coherence",'
          '"claims":[{"claim_text":"The verbal explanation should match the '
          'action type."}'
      )
    return json.dumps({
        'metric_name': metric_name,
        'claims': [{
            'claim_text': 'The verbal explanation should match the action type.',
            'verdict': 1,
            'evidence_text': 'The question format is consistent with the action type.',
        }],
    })


class _SingleMetricFailingJudgeModel(_FakeJudgeModel):

  def sample_text(self, prompt: str, **kwargs) -> str:
    if 'Metric name: action_type_verbal_explanation_coherence' in prompt:
      raise ValueError('Synthetic per-metric judge failure.')
    return super().sample_text(prompt, **kwargs)


class NegotiationEvaluationRecordTest(unittest.TestCase):

  def _make_module(self) -> hdb_negotiation.NegotiationModule:
    participant_specs = {
        'buyer_001': _buyer_participant_spec(),
        'seller_001': _seller_participant_spec(),
    }
    return hdb_negotiation.NegotiationModule(
        entities=(
            _BoundCanonicalEntity(name='Buyer 1', player_id='buyer_001'),
            _BoundCanonicalEntity(name='Seller 1', player_id='seller_001'),
        ),
        participant_specs=participant_specs,
        negotiation_pairs=(('buyer_001', 'seller_001'),),
        enabled=True,
    )

  def test_run_week_emits_schema_grounded_evaluation_records(self):
    module = self._make_module()
    buyer_event = (
        'Buyer 1: '
        '{"type":"QUESTION_BUYER",'
        '"question_details":"Can you share your timeline?",'
        '"internal_reasoning":"Need timing context."}'
    )
    seller_event = (
        'Seller 1: '
        '{"type":"NORMAL_ANSWER",'
        '"answer_details":"I can move in about six weeks.",'
        '"internal_reasoning":"Answer the timeline question without revealing price pressure."}'
    )

    def _record_execute_turn_stage(turn_specs):
      player_ids = [str(turn_spec['player_id']) for turn_spec in turn_specs]
      if player_ids == ['buyer_001']:
        return [{
            'player_id': 'buyer_001',
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'event': buyer_event,
            'decision_rationale': 'Need timing context before discussing price.',
            'force_close': False,
        }]
      if player_ids == ['seller_001']:
        return [{
            'player_id': 'seller_001',
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'event': seller_event,
            'decision_rationale': 'Answer directly to move the conversation forward.',
            'force_close': False,
        }]
      self.fail(f'Unexpected turn specs: {player_ids}')

    with mock.patch.object(
        module,
        '_execute_turn_stage',
        side_effect=_record_execute_turn_stage,
    ), mock.patch.object(
        module,
        '_observe_event',
    ), mock.patch.object(
        module,
        '_flush_pending_observation_updates',
    ):
      outcome = module.run_week(week_number=1)

    self.assertEqual(len(outcome['evaluation_records']), 2)
    buyer_record, seller_record = outcome['evaluation_records']

    self.assertEqual(buyer_record['action_type'], 'QUESTION_BUYER')
    self.assertEqual(
        buyer_record['public_action_payload'],
        {
            'type': 'QUESTION_BUYER',
            'question_details': 'Can you share your timeline?',
        },
    )
    self.assertEqual(
        buyer_record['public_verbal_text'],
        'Can you share your timeline?',
    )
    self.assertEqual(
        buyer_record['internal_reasoning'],
        'Need timing context.',
    )
    self.assertEqual(
        buyer_record['decision_rationale'],
        'Need timing context before discussing price.',
    )
    self.assertEqual(buyer_record['numeric_action_signature'], {})

    self.assertEqual(seller_record['action_type'], 'NORMAL_ANSWER')
    self.assertEqual(
        seller_record['public_action_payload'],
        {
            'type': 'NORMAL_ANSWER',
            'answer_details': 'I can move in about six weeks.',
        },
    )
    self.assertEqual(
        seller_record['public_verbal_text'],
        'I can move in about six weeks.',
    )

  def test_closed_pair_archive_record_carries_evaluation_records(self):
    module = self._make_module()
    pair_key = hdb_negotiation_helpers.pair_key('buyer_001', 'seller_001')
    module._pair_evaluation_records[pair_key] = [{
        'turn_id': f'{pair_key}:1:1:buyer_001:1',
        'pair_key': pair_key,
        'week_number': 1,
        'pair_round_number': 1,
        'actor_id': 'buyer_001',
        'actor_name': 'Buyer 1',
        'actor_role': 'buyer',
        'agent_profile_text': 'profile',
        'action_type': 'QUESTION_BUYER',
        'public_action_payload': {
            'type': 'QUESTION_BUYER',
            'question_details': 'Can you share your timeline?',
        },
        'public_verbal_text': 'Can you share your timeline?',
        'internal_reasoning': 'Need timing context.',
        'decision_rationale': 'Need timing context before discussing price.',
        'numeric_action_signature': {},
    }]

    archive_record = module._build_closed_pair_archive_record(
        buyer_id='buyer_001',
        seller_id='seller_001',
        week_number=1,
        outcome='SUCCESS',
    )

    self.assertEqual(
        archive_record['evaluation_records'],
        module._pair_evaluation_records[pair_key],
    )


class InternalCoherenceEvaluatorTest(unittest.TestCase):

  def _sample_record(self, *, turn_id: str, public_text: str) -> dict[str, object]:
    return {
        'turn_id': turn_id,
        'pair_key': 'buyer_001__seller_001',
        'week_number': 1,
        'pair_round_number': 1 if turn_id.endswith(':1') else 2,
        'actor_id': 'buyer_001',
        'actor_name': 'Buyer 1',
        'actor_role': 'buyer',
        'agent_profile_text': 'Buyer profile text.',
        'action_type': 'QUESTION_BUYER',
        'public_action_payload': {
            'type': 'QUESTION_BUYER',
            'question_details': public_text,
        },
        'public_verbal_text': public_text,
        'internal_reasoning': 'SECRET_RESERVATION_123',
        'decision_rationale': 'Clarify the timeline first.',
        'numeric_action_signature': {},
    }

  def test_collect_evaluation_records_dedupes_weekly_and_archived(self):
    record = self._sample_record(
        turn_id='buyer_001__seller_001:1:1:buyer_001:1',
        public_text='Can you share your timeline?',
    )
    records = hdb_negotiation_evaluation.collect_evaluation_records(
        week_summaries=[{
            'negotiation': {
                'evaluation_records': [record],
            },
        }],
        archived_pair_records=[{
            'evaluation_records': [copy.deepcopy(record)],
        }],
    )
    self.assertEqual(len(records), 1)
    self.assertEqual(records[0]['turn_id'], record['turn_id'])

  def test_evaluate_internal_coherence_computes_global_verdict_and_repetition(self):
    judge_model = _FakeJudgeModel()
    record_one = self._sample_record(
        turn_id='buyer_001__seller_001:1:1:buyer_001:1',
        public_text='Can you share your timeline?',
    )
    record_two = self._sample_record(
        turn_id='buyer_001__seller_001:1:2:buyer_001:2',
        public_text='Can you share your timeline?',
    )

    result = hdb_negotiation_evaluation.evaluate_internal_coherence(
        model=judge_model,
        week_summaries=[{
            'negotiation': {
                'evaluation_records': [record_one, record_two],
            },
        }],
    )

    self.assertEqual(
        result['per_turn_metrics'][0]['repetition_rate']['is_repeated'],
        0,
    )
    self.assertEqual(
        result['per_turn_metrics'][1]['repetition_rate']['is_repeated'],
        1,
    )
    self.assertEqual(
        result['per_turn_metrics'][1]['action_type_verbal_explanation_coherence']['global_verdict'],
        0,
    )
    self.assertEqual(
        result['overall_metrics']['repetition_rate']['micro_average'],
        0.5,
    )
    self.assertTrue(judge_model.batch_calls)
    self.assertTrue(
        any(len(call['prompts']) > 1 for call in judge_model.batch_calls)
    )

    revelation_prompts = [
        prompt
        for prompt in judge_model.prompts
        if 'Metric name: revelation_coherence' in prompt
    ]
    self.assertTrue(revelation_prompts)
    decomposition_prompts = [
        prompt
        for prompt in revelation_prompts
        if '# Internal Coherence Claim Decomposition' in prompt
    ]
    evaluation_prompts = [
        prompt
        for prompt in revelation_prompts
        if '# Internal Coherence Claim Evaluation' in prompt
    ]
    self.assertTrue(decomposition_prompts)
    self.assertTrue(evaluation_prompts)
    for prompt in revelation_prompts:
      self.assertNotIn('SECRET_RESERVATION_123', prompt)
      self.assertIn('## Few-Shot Examples', prompt)
      self.assertIn('### Example 1:', prompt)
      self.assertIn('```json', prompt)

    direct_prompts = [
        prompt
        for prompt in judge_model.prompts
        if 'Metric name: decision_rationale_context_coherence' in prompt
    ]
    self.assertTrue(direct_prompts)
    self.assertTrue(
        any('# Binary Coherence Judge' in prompt for prompt in direct_prompts)
    )

  def test_judge_metric_rejects_empty_claims(self):
    record = hdb_negotiation_evaluation.TurnEvaluationRecord.model_validate(
        self._sample_record(
            turn_id='buyer_001__seller_001:1:1:buyer_001:1',
            public_text='Can you share your timeline?',
        )
    )
    with self.assertRaises(ValueError):
      hdb_negotiation_evaluation._judge_metric_for_record(
          model=_EmptyClaimsJudgeModel(),
          metric_name='action_type_verbal_explanation_coherence',
          record=record,
      )

  def test_judge_metric_accepts_rewritten_claims_and_preserves_original_claims(self):
    record = hdb_negotiation_evaluation.TurnEvaluationRecord.model_validate(
        self._sample_record(
            turn_id='buyer_001__seller_001:1:1:buyer_001:1',
            public_text='Can you share your timeline?',
        )
    )
    result = hdb_negotiation_evaluation._judge_metric_for_record(
        model=_ClaimRewritingJudgeModel(),
        metric_name='action_type_verbal_explanation_coherence',
        record=record,
    )
    self.assertEqual(result['global_verdict'], 1)
    self.assertEqual(len(result['claims']), 1)
    self.assertEqual(result['claims'][0]['claim_text'], 'Original decomposed claim.')
    self.assertEqual(result['claims'][0]['evidence_text'], 'Evidence grounded in the provided turn.')

  def test_judge_metric_recovers_malformed_claim_evaluation_json(self):
    record = hdb_negotiation_evaluation.TurnEvaluationRecord.model_validate(
        self._sample_record(
            turn_id='buyer_001__seller_001:1:1:buyer_001:1',
            public_text='Can you share your timeline?',
        )
    )
    result = hdb_negotiation_evaluation._judge_metric_for_record(
        model=_MalformedEvaluationJudgeModel(),
        metric_name='action_type_verbal_explanation_coherence',
        record=record,
    )

    self.assertEqual(result['global_verdict'], 1)
    self.assertEqual(result['claims'][0]['verdict'], 1)
    self.assertIn(
        'Recovered verdict from malformed structured judge output.',
        result['claims'][0]['evidence_text'],
    )

  def test_judge_metric_recovers_malformed_decomposition_json(self):
    record = hdb_negotiation_evaluation.TurnEvaluationRecord.model_validate(
        self._sample_record(
            turn_id='buyer_001__seller_001:1:1:buyer_001:1',
            public_text='Can you share your timeline?',
        )
    )
    result = hdb_negotiation_evaluation._judge_metric_for_record(
        model=_MalformedDecompositionJudgeModel(),
        metric_name='action_type_verbal_explanation_coherence',
        record=record,
    )

    self.assertEqual(result['global_verdict'], 1)
    self.assertEqual(len(result['claims']), 1)
    self.assertEqual(
        result['claims'][0]['claim_text'],
        'The verbal explanation should match the action type.',
    )

  def test_claim_prompts_use_markdown_and_metric_specific_examples(self):
    record = hdb_negotiation_evaluation.TurnEvaluationRecord.model_validate(
        self._sample_record(
            turn_id='buyer_001__seller_001:1:1:buyer_001:1',
            public_text='Can you share your timeline?',
        )
    )

    decomposition_prompt = (
        hdb_negotiation_evaluation._claim_decomposition_prompt_for_record(
            metric_name='action_type_verbal_explanation_coherence',
            record=record,
        )
    )
    evaluation_prompt = hdb_negotiation_evaluation._claim_evaluation_prompt_for_record(
        metric_name='action_type_verbal_explanation_coherence',
        record=record,
        claims=[
            'The public action should directly seek timeline-related information.'
        ],
    )

    self.assertIn('# Internal Coherence Claim Decomposition', decomposition_prompt)
    self.assertIn('## Output Contract', decomposition_prompt)
    self.assertIn('## Few-Shot Examples', decomposition_prompt)
    self.assertIn(
        '### Example 1: Action type grounds verbal explanation',
        decomposition_prompt,
    )
    self.assertIn('## Decompose This Turn', decomposition_prompt)
    self.assertIn('**Verbal explanation**', decomposition_prompt)

    self.assertIn('# Internal Coherence Claim Evaluation', evaluation_prompt)
    self.assertIn('## Claims To Evaluate', evaluation_prompt)
    self.assertIn('## Evaluate This Turn', evaluation_prompt)
    self.assertIn(
        'The public action should directly seek timeline-related information.',
        evaluation_prompt,
    )

  def test_evaluate_internal_coherence_continues_when_one_metric_fails(self):
    judge_model = _SingleMetricFailingJudgeModel()
    record = self._sample_record(
        turn_id='buyer_001__seller_001:1:1:buyer_001:1',
        public_text='Can you share your timeline?',
    )

    result = hdb_negotiation_evaluation.evaluate_internal_coherence(
        model=judge_model,
        week_summaries=[{
            'negotiation': {
                'evaluation_records': [record],
            },
        }],
    )

    self.assertIn('metric_errors', result)
    self.assertIn(
        'action_type_verbal_explanation_coherence',
        result['metric_errors'],
    )
    self.assertIn(
        'action_type_verbal_explanation_coherence',
        result['overall_metrics'],
    )
    self.assertEqual(
        result['overall_metrics']['action_type_verbal_explanation_coherence']['status'],
        'error',
    )
    self.assertIn(
        'decision_rationale_context_coherence',
        result['per_turn_metrics'][0],
    )
    self.assertNotIn(
        'action_type_verbal_explanation_coherence',
        result['per_turn_metrics'][0],
    )


if __name__ == '__main__':
  unittest.main()
