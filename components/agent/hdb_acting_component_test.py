from absl.testing import absltest
from concordia.components.agent import hdb_acting_component
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.language_model import no_language_model
from concordia.typing import entity as entity_lib


class _FakeEntity:

  def __init__(self, name: str):
    self.name = name


class _FakeBatchModel(no_language_model.NoLanguageModel):

  def __init__(self):
    self.batch_calls = []

  def sample_text_batch(self, prompts, **kwargs):
    self.batch_calls.append({
        'prompts': list(prompts),
        'kwargs': dict(kwargs),
    })
    return [
        (
            '{"type":"QUESTION_BUYER","question":"Can you share your timeline?",'
            '"internal_reasoning":"need timing"}'
        ),
        (
            '{"type":"QUESTION_BUYER","question":"Are you ready to proceed soon?",'
            '"internal_reasoning":"check urgency"}'
        ),
    ]


class HDBActingComponentBatchTest(absltest.TestCase):

  def test_execute_action_attempt_requests_batches_same_schema(self):
    model = _FakeBatchModel()
    component = hdb_acting_component.HDBStructuredActComponent(
        model=model,
        role=common_schemas.RoleType.BUYER,
    )
    component.set_entity(_FakeEntity('Buyer One'))
    action_spec = entity_lib.choice_action_spec(
        call_to_action='What should {name} do next?',
        options=('QUESTION_BUYER',),
    )
    contexts = {
        'action_decisions': (
            '{"preferred_action_type":"QUESTION_BUYER",'
            '"decision_rationale":"Need to understand timeline."}'
        ),
        'NegotiationStrategy': 'Ask about timing before discussing price.',
        'recent_memory': 'Seller mentioned flexibility on move-in date.',
    }

    request_one = component.build_action_attempt_request(contexts, action_spec)
    request_two = component.build_action_attempt_request(
        dict(contexts, recent_memory='Seller asked to move quickly.'),
        action_spec,
    )

    outputs = hdb_acting_component.HDBStructuredActComponent.execute_action_attempt_requests(
        [request_one, request_two]
    )

    self.assertLen(model.batch_calls, 1)
    self.assertLen(model.batch_calls[0]['prompts'], 2)
    self.assertEqual(
        model.batch_calls[0]['kwargs']['json_schema'],
        hdb_acting_component.negotiation_schemas.BuyerQuestion.model_json_schema(),
    )
    self.assertEqual(
        outputs,
        [
            (
                '{"type": "QUESTION_BUYER", "question": "Can you share your timeline?", '
                '"internal_reasoning": "need timing"}'
            ),
            (
                '{"type": "QUESTION_BUYER", "question": "Are you ready to proceed soon?", '
                '"internal_reasoning": "check urgency"}'
            ),
        ],
    )


if __name__ == '__main__':
  absltest.main()
