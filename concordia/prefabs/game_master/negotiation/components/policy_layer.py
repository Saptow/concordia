"""Coordinator-side policy announcement layer for the HDB workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas.policy.schema import (
    PolicyAnnouncement,
    PolicyAnnouncementConfig,
)
from concordia.typing import entity_component
from pydantic import ValidationError
import yaml


class PolicyLayerComponent(action_spec_ignored.ActionSpecIgnored):
  """Announces scheduled policies to active agents before each weekly turn."""

  def __init__(
      self,
      *,
      policy_yaml_path: str,
      enabled: bool = True,
      pre_act_label: str = 'Policy layer',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._policy_yaml_path = str(policy_yaml_path).strip()
    self._enabled = bool(enabled and self._policy_yaml_path)
    self._policies = self._load_policies(self._policy_yaml_path) if self._enabled else []
    self._announced_policy_keys: set[str] = set()
    self._last_announcements: list[dict[str, Any]] = []

  @staticmethod
  def _policy_key(policy: PolicyAnnouncement) -> str:
    return (
        f'{int(policy.week_to_announce_policy)}::'
        f'{policy.policy_type.strip()}::'
        f'{policy.policy_text.strip()}'
    )

  @staticmethod
  def _normalize_yaml_payload(raw_payload: object) -> dict[str, Any]:
    if raw_payload is None:
      return {'policies': []}
    if isinstance(raw_payload, list):
      return {'policies': raw_payload}
    if isinstance(raw_payload, Mapping):
      payload = dict(raw_payload)
      if 'policies' not in payload:
        return {'policies': [payload]}
      return payload
    raise ValueError(
        'policy.yaml must contain either a top-level `policies:` list or a '
        'single policy mapping.'
    )

  @classmethod
  def _load_policies(cls, policy_yaml_path: str) -> list[PolicyAnnouncement]:
    path = Path(policy_yaml_path)
    if not path.exists():
      raise FileNotFoundError(f'Policy YAML file not found: {path}')
    raw_payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    normalized_payload = cls._normalize_yaml_payload(raw_payload)
    try:
      config = PolicyAnnouncementConfig.model_validate(normalized_payload)
    except ValidationError as error:
      raise ValueError(
          f'Invalid policy YAML structure in {path}: {error}'
      ) from error
    return list(config.policies)

  @staticmethod
  def _format_policy_observation(
      *,
      week_number: int,
      policy: PolicyAnnouncement,
  ) -> str:
    return (
        f'Week {week_number} policy announcement.\n'
        f'Policy type: {policy.policy_type.strip()}\n'
        f'Policy details: {policy.policy_text.strip()}'
    )

  def _make_pre_act_value(self) -> str:
    pending_weeks = sorted(
        {
            int(policy.week_to_announce_policy)
            for policy in self._policies
            if self._policy_key(policy) not in self._announced_policy_keys
        }
    )
    return json.dumps(
        {
            'enabled': self._enabled,
            'policy_yaml_path': self._policy_yaml_path,
            'loaded_policy_count': len(self._policies),
            'announced_policy_count': len(self._announced_policy_keys),
            'pending_announcement_weeks': pending_weeks,
            'last_announcements': list(self._last_announcements),
        },
        ensure_ascii=False,
    )

  def is_enabled(self) -> bool:
    return self._enabled

  def announce_policies_for_week(
      self,
      *,
      week_number: int,
      active_player_ids: Sequence[str],
  ) -> dict[str, list[str]]:
    """Return policy observations to deliver immediately to active agents."""
    if not self._enabled:
      self._last_announcements = []
      return {}

    normalized_player_ids = [
        str(player_id).strip() for player_id in active_player_ids if str(player_id).strip()
    ]
    if not normalized_player_ids:
      self._last_announcements = []
      return {}

    due_policies = [
        policy
        for policy in self._policies
        if int(policy.week_to_announce_policy) == int(week_number)
        and self._policy_key(policy) not in self._announced_policy_keys
    ]
    if not due_policies:
      self._last_announcements = []
      return {}

    observations_by_player_id = {
        player_id: [] for player_id in normalized_player_ids
    }
    announcement_log: list[dict[str, Any]] = []
    for policy in due_policies:
      observation = self._format_policy_observation(
          week_number=int(week_number),
          policy=policy,
      )
      for player_id in normalized_player_ids:
        observations_by_player_id[player_id].append(observation)
      policy_key = self._policy_key(policy)
      self._announced_policy_keys.add(policy_key)
      announcement_log.append(
          {
              'policy_key': policy_key,
              'week_number': int(week_number),
              'policy_type': policy.policy_type,
              'audience_player_ids': list(normalized_player_ids),
          }
      )

    self._last_announcements = announcement_log
    return observations_by_player_id

  def get_state(self) -> entity_component.ComponentState:
    return {
        'policy_yaml_path': self._policy_yaml_path,
        'enabled': int(self._enabled),
        'policies': [policy.model_dump() for policy in self._policies],
        'announced_policy_keys': sorted(self._announced_policy_keys),
        'last_announcements': list(self._last_announcements),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'policy_yaml_path' in state:
      self._policy_yaml_path = str(state.get('policy_yaml_path', '')).strip()
    self._enabled = bool(state.get('enabled', 1))
    if 'policies' in state:
      self._policies = [
          PolicyAnnouncement.model_validate(policy)
          for policy in state.get('policies', [])
      ]
    elif self._enabled and self._policy_yaml_path:
      self._policies = self._load_policies(self._policy_yaml_path)
    self._announced_policy_keys = {
        str(value) for value in state.get('announced_policy_keys', [])
    }
    self._last_announcements = [
        dict(item) for item in state.get('last_announcements', [])
        if isinstance(item, Mapping)
    ]
