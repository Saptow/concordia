"""Coordinator-side policy announcement layer for the HDB workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas.policy.schema import (
    PolicyAnnouncementConfig,
    PolicyStateEntry,
    PolicyWeekSchedule,
)
from concordia.language_model import language_model
from concordia.typing import entity_component
from pydantic import ValidationError
import yaml


ACTIVE_POLICY_SOURCES_BLOCK_PREFIX = '[[POLICY_ACTIVE_SOURCES_JSON]]'
ACTIVE_POLICY_SOURCES_BLOCK_SUFFIX = '[[/POLICY_ACTIVE_SOURCES_JSON]]'


class PolicyLayerComponent(action_spec_ignored.ActionSpecIgnored):
  """Tracks current policy state and active policy sources week by week."""

  def __init__(
      self,
      *,
      policy_yaml_path: str,
      model: language_model.LanguageModel | None = None,
      updates_enabled: bool = True,
      enabled: bool = True,
      pre_act_label: str = 'Policy layer',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._policy_yaml_path = str(policy_yaml_path).strip()
    self._model = model
    self._updates_enabled = bool(updates_enabled)
    self._enabled = bool(enabled and self._policy_yaml_path)
    config = (
        self._load_policy_config(self._policy_yaml_path)
        if self._enabled else PolicyAnnouncementConfig()
    )
    self._initial_state = self._clone_policies(config.initial_state)
    self._scheduled_policy_changes = list(config.policies)
    self._current_policies = self._clone_policies(self._initial_state)
    self._applied_policy_weeks: set[int] = set()
    self._announced_weeks: set[int] = set()
    self._last_announcements: list[dict[str, Any]] = []
    self._active_source_counts: dict[str, int] = {}
    self._active_source_order: list[str] = []
    self._rebuild_active_sources()

  @staticmethod
  def _policy_key(policy: PolicyStateEntry) -> str:
    return f'{policy.policy_type.strip()}::{policy.policy_text.strip()}'

  @staticmethod
  def _normalize_policy_type(policy_type: str) -> str:
    return str(policy_type).strip()

  @staticmethod
  def _normalize_markdown_path(path: str) -> str:
    return str(path).strip().replace('\\', '/')

  @classmethod
  def _normalize_policy_sources(
      cls,
      raw_sources: object,
      *,
      context: str,
  ) -> list[str]:
    if raw_sources is None:
      return []
    if isinstance(raw_sources, str):
      raw_values = [raw_sources]
    elif isinstance(raw_sources, Sequence):
      raw_values = list(raw_sources)
    else:
      raise ValueError(
          f'{context} must be a markdown path or list of markdown paths.'
      )

    normalized_sources: list[str] = []
    seen_sources: set[str] = set()
    for index, raw_source in enumerate(raw_values):
      normalized = cls._normalize_markdown_path(str(raw_source))
      if not normalized:
        continue
      if Path(normalized).suffix.lower() not in {'.md', '.markdown', '.txt'}:
        raise ValueError(
            f'{context}[{index}] must point to a markdown source file.'
        )
      if normalized in seen_sources:
        continue
      seen_sources.add(normalized)
      normalized_sources.append(normalized)
    return normalized_sources

  @staticmethod
  def _clone_policies(
      policies: Sequence[PolicyStateEntry],
  ) -> list[PolicyStateEntry]:
    return [
        PolicyStateEntry.model_validate(policy.model_dump())
        for policy in policies
    ]

  @classmethod
  def _normalize_policy_entries(
      cls,
      raw_policies: object,
      *,
      context: str,
  ) -> list[dict[str, Any]]:
    if raw_policies is None:
      return []
    if isinstance(raw_policies, Mapping):
      raise ValueError(f'{context} must be a list of policy mappings.')
    if not isinstance(raw_policies, Sequence) or isinstance(raw_policies, str):
      raise ValueError(f'{context} must be a list of policy mappings.')

    normalized_policies: list[dict[str, Any]] = []
    for index, raw_policy in enumerate(raw_policies):
      if not isinstance(raw_policy, Mapping):
        raise ValueError(
            f'{context}[{index}] must be a mapping with '
            '`policy_type`, `policy_text`, and optional `sources`.'
        )
      normalized_policies.append({
          'policy_type': raw_policy.get('policy_type'),
          'policy_text': raw_policy.get('policy_text'),
          'sources': cls._normalize_policy_sources(
              raw_policy.get('sources', []),
              context=f'{context}[{index}].sources',
          ),
      })
    return normalized_policies

  @classmethod
  def _normalize_schedule_entry(cls, raw_schedule: object) -> dict[str, Any]:
    if not isinstance(raw_schedule, Mapping):
      raise ValueError(
          'Each scheduled policy entry in `policy.yaml` must be a mapping.'
      )
    if 'week' not in raw_schedule or 'policies' not in raw_schedule:
      raise ValueError(
          'Each scheduled policy entry must contain `week` and `policies`.'
      )
    return {
        'week': raw_schedule.get('week'),
        'policies': cls._normalize_policy_entries(
            raw_schedule.get('policies'),
            context='policies',
        ),
        'overwrite': bool(raw_schedule.get('overwrite', False)),
    }

  @classmethod
  def _normalize_yaml_payload(cls, raw_payload: object) -> dict[str, Any]:
    if raw_payload is None:
      return {'initial_state': [], 'policies': []}
    if isinstance(raw_payload, Mapping):
      payload = dict(raw_payload)
      if 'initial_state' in payload or 'policies' in payload:
        raw_schedules = payload.get('policies', [])
        if isinstance(raw_schedules, Mapping):
          raw_schedules = [raw_schedules]
        if raw_schedules is None:
          raw_schedules = []
        if not isinstance(raw_schedules, Sequence) or isinstance(
            raw_schedules, str
        ):
          raise ValueError('`policies` must be a list of scheduled updates.')
        return {
            'initial_state': cls._normalize_policy_entries(
                payload.get('initial_state', []),
                context='initial_state',
            ),
            'policies': [
                cls._normalize_schedule_entry(raw_schedule)
                for raw_schedule in raw_schedules
            ],
        }
    raise ValueError(
        'policy.yaml must contain `initial_state:` and/or `policies:`.'
    )

  @classmethod
  def _load_policy_config(cls, policy_yaml_path: str) -> PolicyAnnouncementConfig:
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
    return PolicyAnnouncementConfig(
        initial_state=cls._clone_policies(config.initial_state),
        policies=sorted(config.policies, key=lambda schedule: int(schedule.week)),
    )

  def _active_source_paths(self) -> list[str]:
    return [
        source_path
        for source_path in self._active_source_order
        if self._active_source_counts.get(source_path, 0) > 0
    ]

  def _add_policy_sources(self, policy: PolicyStateEntry) -> None:
    for raw_source in policy.sources:
      source_path = self._normalize_markdown_path(raw_source)
      if not source_path:
        continue
      if self._active_source_counts.get(source_path, 0) == 0:
        self._active_source_order.append(source_path)
      self._active_source_counts[source_path] = (
          self._active_source_counts.get(source_path, 0) + 1
      )

  def _remove_policy_sources(self, policy: PolicyStateEntry) -> None:
    for raw_source in policy.sources:
      source_path = self._normalize_markdown_path(raw_source)
      current_count = self._active_source_counts.get(source_path, 0)
      if current_count <= 1:
        self._active_source_counts.pop(source_path, None)
      elif current_count > 1:
        self._active_source_counts[source_path] = current_count - 1

  def _rebuild_active_sources(self) -> None:
    self._active_source_counts = {}
    self._active_source_order = []
    for policy in self._current_policies:
      self._add_policy_sources(policy)

  def _append_policy(self, policy: PolicyStateEntry) -> None:
    cloned_policy = PolicyStateEntry.model_validate(policy.model_dump())
    self._current_policies.append(cloned_policy)
    self._add_policy_sources(cloned_policy)

  def _replace_policies_for_categories(
      self,
      *,
      categories: set[str],
      replacement_policies: Sequence[PolicyStateEntry],
  ) -> None:
    normalized_categories = {
        self._normalize_policy_type(category)
        for category in categories
        if self._normalize_policy_type(category)
    }
    retained_policies: list[PolicyStateEntry] = []
    for policy in self._current_policies:
      if self._normalize_policy_type(policy.policy_type) in normalized_categories:
        self._remove_policy_sources(policy)
        continue
      retained_policies.append(policy)
    self._current_policies = retained_policies

    replacement_keys: set[str] = set()
    for policy in replacement_policies:
      policy_key = self._policy_key(policy)
      if policy_key in replacement_keys:
        continue
      replacement_keys.add(policy_key)
      self._append_policy(policy)

  def _fallback_overwrite_policies(
      self,
      *,
      schedule: PolicyWeekSchedule,
  ) -> list[PolicyStateEntry]:
    replacement_policies: list[PolicyStateEntry] = []
    replacement_keys: set[str] = set()
    for policy in schedule.policies:
      policy_key = self._policy_key(policy)
      if policy_key in replacement_keys:
        continue
      replacement_keys.add(policy_key)
      replacement_policies.append(
          PolicyStateEntry.model_validate(policy.model_dump())
      )
    return replacement_policies

  def _reassess_overwritten_categories(
      self,
      *,
      schedule: PolicyWeekSchedule,
  ) -> list[PolicyStateEntry]:
    if not schedule.policies:
      return []
    if self._model is None:
      return self._fallback_overwrite_policies(schedule=schedule)

    affected_categories = sorted({
        self._normalize_policy_type(policy.policy_type)
        for policy in schedule.policies
        if self._normalize_policy_type(policy.policy_type)
    })
    existing_category_policies = [
        policy.model_dump()
        for policy in self._current_policies
        if self._normalize_policy_type(policy.policy_type) in affected_categories
    ]
    injected_policies = [policy.model_dump() for policy in schedule.policies]
    prompt = (
        'You are reconciling HDB resale policy state for a simulation.\n'
        'A weekly update marked `overwrite: true` means the active policy '
        'state for the affected categories must be reassessed.\n'
        'Return ONLY valid JSON with this schema:\n'
        '{"policies":[{"policy_type":"string","policy_text":"string",'
        '"sources":["path.md"]}]}\n'
        'Rules:\n'
        '- Consider only the affected categories listed below.\n'
        '- Remove stale, superseded, or conflicting older policies in those '
        'categories.\n'
        '- Keep still-valid older policies in those categories when they do '
        'not conflict with the new update.\n'
        '- Prefer the new injected policies when they conflict with older '
        'policies.\n'
        '- Preserve the correct markdown `sources` for each surviving policy.\n'
        '- Do not include categories outside the affected set.\n'
        '- Do not include explanations or markdown.\n\n'
        f'Affected categories: {json.dumps(affected_categories, ensure_ascii=False)}\n'
        'Existing active policies in affected categories:\n'
        f'{json.dumps(existing_category_policies, ensure_ascii=False, indent=2)}\n\n'
        'Newly injected policies for this week:\n'
        f'{json.dumps(injected_policies, ensure_ascii=False, indent=2)}\n'
    )
    try:
      response = self._model.sample_text(prompt, max_tokens=1600)
      parsed_response = json.loads(response)
      normalized_policies: list[PolicyStateEntry] = []
      for raw_policy in parsed_response.get('policies', []):
        policy = PolicyStateEntry.model_validate(raw_policy)
        if self._normalize_policy_type(policy.policy_type) not in affected_categories:
          continue
        normalized_policies.append(policy)
      if normalized_policies:
        return normalized_policies
      logging.warning(
          'Policy layer overwrite reassessment returned no valid policies; '
          'falling back to injected policies.'
      )
    except (json.JSONDecodeError, ValidationError, AttributeError, TypeError) as error:
      logging.warning(
          'Policy layer overwrite reassessment failed; '
          'falling back to injected policies. Error: %s',
          error,
      )
    return self._fallback_overwrite_policies(schedule=schedule)

  def _apply_weekly_policy_updates(
      self,
      *,
      week_number: int,
  ) -> list[PolicyWeekSchedule]:
    if not self._updates_enabled:
      return []
    if int(week_number) in self._applied_policy_weeks:
      return []
    due_schedules = [
        schedule
        for schedule in self._scheduled_policy_changes
        if int(schedule.week) == int(week_number)
    ]
    if not due_schedules:
      return []

    existing_policy_keys = {
        self._policy_key(policy) for policy in self._current_policies
    }
    for schedule in due_schedules:
      if schedule.overwrite:
        affected_categories = {
            self._normalize_policy_type(policy.policy_type)
            for policy in schedule.policies
            if self._normalize_policy_type(policy.policy_type)
        }
        self._replace_policies_for_categories(
            categories=affected_categories,
            replacement_policies=self._reassess_overwritten_categories(
                schedule=schedule
            ),
        )
        existing_policy_keys = {
            self._policy_key(policy) for policy in self._current_policies
        }
        continue
      for policy in schedule.policies:
        policy_key = self._policy_key(policy)
        if policy_key in existing_policy_keys:
          continue
        self._append_policy(policy)
        existing_policy_keys.add(policy_key)
    self._applied_policy_weeks.add(int(week_number))
    return due_schedules

  @classmethod
  def _active_sources_block(cls, source_paths: Sequence[str]) -> str:
    return (
        f'{ACTIVE_POLICY_SOURCES_BLOCK_PREFIX}\n'
        f'{json.dumps(list(source_paths), ensure_ascii=False)}\n'
        f'{ACTIVE_POLICY_SOURCES_BLOCK_SUFFIX}'
    )

  @classmethod
  def extract_active_source_paths(cls, text: str) -> list[str] | None:
    raw_text = str(text or '')
    start = raw_text.rfind(ACTIVE_POLICY_SOURCES_BLOCK_PREFIX)
    if start == -1:
      return None
    start += len(ACTIVE_POLICY_SOURCES_BLOCK_PREFIX)
    end = raw_text.find(ACTIVE_POLICY_SOURCES_BLOCK_SUFFIX, start)
    if end == -1:
      return None
    try:
      payload = json.loads(raw_text[start:end].strip())
    except json.JSONDecodeError:
      return None
    if not isinstance(payload, list):
      return None
    normalized_sources: list[str] = []
    seen_sources: set[str] = set()
    for item in payload:
      normalized = cls._normalize_markdown_path(str(item))
      if not normalized or normalized in seen_sources:
        continue
      seen_sources.add(normalized)
      normalized_sources.append(normalized)
    return normalized_sources

  @classmethod
  def _format_policy_observation(
      cls,
      *,
      week_number: int,
      current_policies: Sequence[PolicyStateEntry],
      active_source_paths: Sequence[str],
      newly_applied_schedules: Sequence[PolicyWeekSchedule],
  ) -> str:
    lines = [f'Week {week_number} policy announcement.']

    newly_applied_policies = [
        policy
        for schedule in newly_applied_schedules
        for policy in schedule.policies
    ]
    if newly_applied_policies:
      lines.append('New policy updates taking effect this week:')
      lines.extend(
          f'- {policy.policy_type.strip()}: {policy.policy_text.strip()}'
          for policy in newly_applied_policies
      )
    else:
      lines.append('No new policy changes took effect this week.')

    if current_policies:
      lines.append('Current policy state in effect:')
      lines.extend(
          f'{index}. {policy.policy_type.strip()}: {policy.policy_text.strip()}'
          for index, policy in enumerate(current_policies, start=1)
      )
    else:
      lines.append('No simulation-specific policies are currently in effect.')

    if active_source_paths:
      lines.append(
          f'Policy retrieval source files currently active: '
          f'{len(active_source_paths)}.'
      )
    else:
      lines.append('No policy retrieval source files are currently active.')
    lines.append(cls._active_sources_block(active_source_paths))
    return '\n'.join(lines)

  def _make_pre_act_value(self) -> str:
    pending_weeks = sorted({
        int(schedule.week)
        for schedule in self._scheduled_policy_changes
        if int(schedule.week) not in self._applied_policy_weeks
    })
    return json.dumps(
        {
            'enabled': self._enabled,
            'policy_yaml_path': self._policy_yaml_path,
            'initial_policy_count': len(self._initial_state),
            'scheduled_policy_weeks': pending_weeks if self._updates_enabled else [],
            'current_policy_count': len(self._current_policies),
            'active_source_count': len(self._active_source_paths()),
            'updates_enabled': self._updates_enabled,
            'announced_weeks': sorted(self._announced_weeks),
            'last_announcements': list(self._last_announcements),
        },
        ensure_ascii=False,
    )

  def is_enabled(self) -> bool:
    return self._enabled

  def get_active_source_paths(self) -> list[str]:
    return list(self._active_source_paths())

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
        str(player_id).strip()
        for player_id in active_player_ids
        if str(player_id).strip()
    ]
    if not normalized_player_ids:
      self._last_announcements = []
      return {}

    if int(week_number) in self._announced_weeks:
      self._last_announcements = []
      return {}

    due_schedules = self._apply_weekly_policy_updates(week_number=int(week_number))
    active_source_paths = self._active_source_paths()
    observation = self._format_policy_observation(
        week_number=int(week_number),
        current_policies=self._current_policies,
        active_source_paths=active_source_paths,
        newly_applied_schedules=due_schedules,
    )
    self._announced_weeks.add(int(week_number))
    self._last_announcements = [{
        'week_number': int(week_number),
        'audience_player_ids': list(normalized_player_ids),
        'new_policy_count': sum(
            len(schedule.policies) for schedule in due_schedules
        ),
        'current_policy_count': len(self._current_policies),
        'active_source_count': len(active_source_paths),
    }]
    return {
        player_id: [observation]
        for player_id in normalized_player_ids
    }

  def get_state(self) -> entity_component.ComponentState:
    return {
        'policy_yaml_path': self._policy_yaml_path,
        'enabled': int(self._enabled),
        'updates_enabled': int(self._updates_enabled),
        'initial_state': [
            policy.model_dump() for policy in self._initial_state
        ],
        'policies': [
            schedule.model_dump() for schedule in self._scheduled_policy_changes
        ],
        'current_policies': [
            policy.model_dump() for policy in self._current_policies
        ],
        'applied_policy_weeks': sorted(self._applied_policy_weeks),
        'announced_weeks': sorted(self._announced_weeks),
        'last_announcements': list(self._last_announcements),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'policy_yaml_path' in state:
      self._policy_yaml_path = str(state.get('policy_yaml_path', '')).strip()
    self._enabled = bool(state.get('enabled', 1))
    self._updates_enabled = bool(state.get('updates_enabled', 1))
    if 'initial_state' in state or 'policies' in state:
      raw_payload = {
          'initial_state': state.get('initial_state', []),
          'policies': state.get('policies', []),
      }
      config = PolicyAnnouncementConfig.model_validate(
          self._normalize_yaml_payload(raw_payload)
      )
      self._initial_state = self._clone_policies(config.initial_state)
      self._scheduled_policy_changes = list(config.policies)
    elif self._enabled and self._policy_yaml_path:
      config = self._load_policy_config(self._policy_yaml_path)
      self._initial_state = self._clone_policies(config.initial_state)
      self._scheduled_policy_changes = list(config.policies)
    else:
      self._initial_state = []
      self._scheduled_policy_changes = []

    self._applied_policy_weeks = {
        int(value)
        for value in state.get('applied_policy_weeks', [])
    }
    self._announced_weeks = {
        int(value)
        for value in state.get('announced_weeks', self._applied_policy_weeks)
    }
    if 'current_policies' in state:
      self._current_policies = [
          PolicyStateEntry.model_validate(policy)
          for policy in state.get('current_policies', [])
      ]
    else:
      self._current_policies = self._clone_policies(self._initial_state)
      replayed_weeks = sorted(self._applied_policy_weeks)
      self._applied_policy_weeks = set()
      self._rebuild_active_sources()
      for week_number in replayed_weeks:
        self._apply_weekly_policy_updates(week_number=week_number)
    self._rebuild_active_sources()
    self._last_announcements = [
        dict(item) for item in state.get('last_announcements', [])
        if isinstance(item, Mapping)
    ]
