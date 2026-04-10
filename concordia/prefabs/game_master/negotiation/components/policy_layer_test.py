import tempfile
import textwrap
import unittest
from pathlib import Path

from concordia.prefabs.game_master.negotiation.components import policy_layer


class _FakePolicyOverwriteModel:

  def __init__(self, response: str):
    self._response = response
    self.prompts: list[str] = []
    self.sample_kwargs: list[dict[str, object]] = []

  def sample_text(self, prompt: str, **kwargs) -> str:
    self.prompts.append(prompt)
    self.sample_kwargs.append(dict(kwargs))
    return self._response


class _LegacyPolicyOverwriteModel:

  def __init__(self, response: str):
    self._response = response
    self.prompts: list[str] = []
    self.call_count = 0

  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int,
      terminators=(),
      temperature=0.5,
      top_p=0.95,
      top_k=64,
      timeout=60,
      seed=None,
  ) -> str:
    del max_tokens, terminators, temperature, top_p, top_k, timeout, seed
    self.call_count += 1
    self.prompts.append(prompt)
    return self._response


class PolicyLayerComponentTest(unittest.TestCase):

  def _write_policy_yaml(self, content: str) -> str:
    temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(temp_dir.cleanup)
    path = Path(temp_dir.name) / 'policy.yaml'
    path.write_text(textwrap.dedent(content).strip() + '\n', encoding='utf-8')
    return str(path)

  def test_only_announces_when_policy_changes(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Financing and Affordability Rules
            policy_text: Baseline MSR cap remains in force.
        policies:
          - week: 2
            policies:
              - policy_type: Grants and Subsidies
                policy_text: New first-time buyer grant is now available.
        """
    )
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        enabled=True,
    )

    week_1 = component.announce_policies_for_week(
        week_number=1,
        active_player_ids=['buyer_1', 'seller_1'],
    )
    self.assertEqual(week_1, {})
    self.assertIn(
        'Baseline MSR cap remains in force.',
        component.get_current_policy_prompt(),
    )

    week_2 = component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1', 'seller_1'],
    )
    week_2_message = week_2['seller_1'][0]
    self.assertIn('Week 2 policy announcement.', week_2_message)
    self.assertIn('New policy updates taking effect this week:', week_2_message)
    self.assertIn('New first-time buyer grant is now available.', week_2_message)
    self.assertIn('Baseline MSR cap remains in force.', week_2_message)

    self.assertEqual(
        component.announce_policies_for_week(
            week_number=2,
            active_player_ids=['buyer_1', 'seller_1'],
        ),
        {},
    )

  def test_includes_sources_for_changed_policies_only(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Financing and Affordability Rules
            policy_text: Baseline MSR cap remains in force.
        policies:
          - week: 2
            policies:
              - policy_type: Grants and Subsidies
                policy_text: New first-time buyer grant is now available.
                sources:
                  - data/policies/2022/page/key-statistics.extracted.md
        """
    )
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        enabled=True,
    )

    week_1 = component.announce_policies_for_week(
        week_number=1,
        active_player_ids=['buyer_1'],
    )
    self.assertEqual(week_1, {})

    week_2 = component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1'],
    )
    week_2_message = week_2['buyer_1'][0]
    self.assertIn(
        'data/policies/2022/page/key-statistics.extracted.md',
        week_2_message,
    )
    self.assertEqual(
        component.get_active_source_paths(),
        ['data/policies/2022/page/key-statistics.extracted.md'],
    )

  def test_state_round_trip_restores_current_policies(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Transaction Rules/Processes
            policy_text: Baseline approval workflow applies.
        policies:
          - week: 2
            policies:
              - policy_type: Grants and Subsidies
                policy_text: Week-two grant becomes active.
          - week: 3
            policies:
              - policy_type: Financing and Affordability Rules
                policy_text: Week-three affordability rule becomes active.
        """
    )
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        enabled=True,
    )
    component.announce_policies_for_week(
        week_number=1,
        active_player_ids=['buyer_1'],
    )
    component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1'],
    )

    restored_component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        enabled=True,
    )
    restored_component.set_state(component.get_state())
    week_3 = restored_component.announce_policies_for_week(
        week_number=3,
        active_player_ids=['buyer_1'],
    )
    week_3_message = week_3['buyer_1'][0]

    self.assertIn('Baseline approval workflow applies.', week_3_message)
    self.assertIn('Week-two grant becomes active.', week_3_message)
    self.assertIn('Week-three affordability rule becomes active.', week_3_message)

  def test_overwrite_reassesses_affected_categories(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Grants and Subsidies
            policy_text: Legacy grant amount remains in force.
          - policy_type: Transaction Rules/Processes
            policy_text: Existing OTP workflow remains in force.
        policies:
          - week: 2
            overwrite: true
            policies:
              - policy_type: Grants and Subsidies
                policy_text: New grant amount replaces the legacy grant.
        """
    )
    model = _FakePolicyOverwriteModel(
        """
        {
          "policies": [
            {
              "policy_type": "Grants and Subsidies",
              "policy_text": "New grant amount replaces the legacy grant."
            }
          ]
        }
        """
    )
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        model=model,
        enabled=True,
    )

    week_2 = component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1'],
    )
    week_2_message = week_2['buyer_1'][0]

    self.assertIn('New grant amount replaces the legacy grant.', week_2_message)
    self.assertNotIn('Legacy grant amount remains in force.', week_2_message)
    self.assertIn('Existing OTP workflow remains in force.', week_2_message)
    self.assertEqual(len(model.prompts), 1)
    self.assertIn('Grants and Subsidies', model.prompts[0])
    self.assertNotIn('Transaction Rules/Processes', model.prompts[0])
    self.assertIn('json_schema', model.sample_kwargs[0])

  def test_overwrite_falls_back_to_injected_policies_when_model_output_invalid(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Grants and Subsidies
            policy_text: Legacy grant amount remains in force.
          - policy_type: Financing and Affordability Rules
            policy_text: Existing loan cap remains in force.
        policies:
          - week: 2
            overwrite: true
            policies:
              - policy_type: Grants and Subsidies
                policy_text: Replacement grant becomes active this week.
        """
    )
    model = _FakePolicyOverwriteModel("not json")
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        model=model,
        enabled=True,
    )

    week_2 = component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1'],
    )
    week_2_message = week_2['buyer_1'][0]

    self.assertIn('Replacement grant becomes active this week.', week_2_message)
    self.assertNotIn('Legacy grant amount remains in force.', week_2_message)
    self.assertIn('Existing loan cap remains in force.', week_2_message)

  def test_overwrite_reassessment_retries_without_json_schema_for_legacy_models(self):
    policy_yaml_path = self._write_policy_yaml(
        """
        initial_state:
          - policy_type: Grants and Subsidies
            policy_text: Legacy grant amount remains in force.
          - policy_type: Transaction Rules/Processes
            policy_text: Existing OTP workflow remains in force.
        policies:
          - week: 2
            overwrite: true
            policies:
              - policy_type: Grants and Subsidies
                policy_text: New grant amount replaces the legacy grant.
        """
    )
    model = _LegacyPolicyOverwriteModel(
        """
        {
          "policies": [
            {
              "policy_type": "Grants and Subsidies",
              "policy_text": "New grant amount replaces the legacy grant."
            }
          ]
        }
        """
    )
    component = policy_layer.PolicyLayerComponent(
        policy_yaml_path=policy_yaml_path,
        model=model,
        enabled=True,
    )

    week_2 = component.announce_policies_for_week(
        week_number=2,
        active_player_ids=['buyer_1'],
    )
    week_2_message = week_2['buyer_1'][0]

    self.assertIn('New grant amount replaces the legacy grant.', week_2_message)
    self.assertNotIn('Legacy grant amount remains in force.', week_2_message)
    self.assertIn('Existing OTP workflow remains in force.', week_2_message)
    self.assertEqual(model.call_count, 1)


if __name__ == '__main__':
  unittest.main()
