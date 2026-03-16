"""Concurrent weekly engine for HDB listing and negotiation modules."""

from collections.abc import Callable, Mapping, Sequence
import functools
import json
from typing import Any

from absl import logging
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.environment import engine as engine_lib
from concordia.environment import step_controller as step_controller_lib
from concordia.prefabs.game_master.negotiation.components import hdb_coordinator_helper
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
import termcolor


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION
)

EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'


class HDBSimulationEngine(engine_lib.Engine):
  """Runs listing and negotiation concurrently within each weekly step."""

  def __init__(
      self,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
  ):
    self._call_to_make_observation = call_to_make_observation

  def make_observation(
      self,
      game_master: entity_lib.Entity,
      entity: entity_lib.Entity,
  ) -> str:
    return game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_make_observation.format(name=entity.name),
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )

  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
  ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
    if not entities:
      logging.error('HDBSimulationEngine.next_acting called with no entities.')
      return game_master, entity_lib.skip_this_step_action_spec()
    return entities[0], entity_lib.skip_this_step_action_spec()

  def resolve(
      self,
      game_master: entity_lib.Entity,
      event: str,
  ) -> None:
    game_master.observe(observation=f'{EVENT_TAG} {event}')

  def terminate(
      self,
      game_master: entity_lib.Entity,
  ) -> bool:
    coordinator = self._get_coordinator(game_master)
    return coordinator.should_terminate()

  def next_game_master(
      self,
      game_master: entity_lib.Entity,
      game_masters: Sequence[entity_lib.Entity],
  ) -> entity_lib.Entity:
    if not game_masters:
      logging.error(
          'HDBSimulationEngine.next_game_master called with no game masters.'
      )
      return game_master
    return game_masters[0]

  # Logging helpers
  @staticmethod
  def _has_meaningful_log_value(value: Any) -> bool:
    """Returns whether a log payload contains any non-empty values."""
    if value is None:
      return False
    if isinstance(value, str):
      return bool(value.strip())
    if isinstance(value, Mapping):
      return any(
          HDBSimulationEngine._has_meaningful_log_value(item)
          for item in value.values()
      )
    if isinstance(value, Sequence) and not isinstance(value, str):
      return any(
          HDBSimulationEngine._has_meaningful_log_value(item) for item in value
      )
    return True

  def _collect_entity_logs(
      self,
      *,
      coordinator: hdb_coordinator_helper.WeeklyCoordinator,
      entities: Sequence[entity_lib.Entity],
      active_negotiation_player_ids: Sequence[str],
      listing_player_ids: Sequence[str],
  ) -> dict[str, Mapping[str, Any]]:
    """Collects logs from module-owned agents before falling back to outer ones."""
    collected_logs: dict[str, Mapping[str, Any]] = {}
    negotiation_module = coordinator.get_negotiation_module()
    listing_module = coordinator.get_listing_module()
    negotiation_player_ids = (
        tuple(str(player_id) for player_id in active_negotiation_player_ids)
        if getattr(negotiation_module, 'is_enabled', lambda: True)()
        else ()
    )
    listing_active_player_ids = (
        tuple(str(player_id) for player_id in listing_player_ids)
        if getattr(listing_module, 'is_enabled', lambda: True)()
        else ()
    )
    active_player_ids = set(negotiation_player_ids) | set(listing_active_player_ids)
    active_entity_names = {
        coordinator.get_player_name(player_id) for player_id in active_player_ids
    }
    if not active_entity_names:
      return collected_logs

    module_requests = (
        (
            negotiation_module,
            negotiation_player_ids,
        ),
        (
            listing_module,
            listing_active_player_ids,
        ),
    )
    for module, player_ids in module_requests:
      if not player_ids or not hasattr(module, 'get_entity_log_snapshots'):
        continue
      try:
        snapshots = module.get_entity_log_snapshots(player_ids)
      except Exception as error:  # pylint: disable=broad-exception-caught
        logging.warning('Failed to collect module entity logs: %s', error)
        continue
      if not isinstance(snapshots, Mapping):
        continue
      for entity_name, snapshot in snapshots.items():
        if not self._has_meaningful_log_value(snapshot):
          continue
        collected_logs[str(entity_name)] = snapshot

    for entity in entities:
      if (
          entity.name not in active_entity_names
          or entity.name in collected_logs
          or not hasattr(entity, 'get_last_log')
      ):
        continue
      snapshot = entity.get_last_log()
      if not self._has_meaningful_log_value(snapshot):
        continue
      collected_logs[entity.name] = snapshot

    return collected_logs

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity],
      entities: Sequence[entity_lib.Entity],
      premise: str = '',
      max_steps: int = 100,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
      step_controller: step_controller_lib.StepController | None = None,
      step_callback: (
          Callable[[step_controller_lib.StepData], None] | None
      ) = None,
  ):
    if not game_masters:
      logging.error('HDBSimulationEngine.run_loop called with no game masters.')
      return

    game_master = game_masters[0]
    steps = 0
    # if premise: # Game master has no generative elements 
    #   game_master.observe(f'{EVENT_TAG} {premise}')

    while not self.terminate(game_master) and steps < max_steps:
      if step_controller is not None:
        if not step_controller.wait_for_step_permission():
          break

      summary, active_negotiation_player_ids, listing_player_ids = self._run_week(
          game_master,
          verbose=verbose,
      )
      self._deliver_pending_observations(
          game_master=game_master,
          entities=entities,
          active_player_ids=active_negotiation_player_ids,
          verbose=verbose,
      )

      steps += 1
      coordinator = self._get_coordinator(game_master)
      entity_logs = self._collect_entity_logs(
          coordinator=coordinator,
          entities=entities,
          active_negotiation_player_ids=active_negotiation_player_ids,
          listing_player_ids=listing_player_ids,
      )
      if log is not None:
        log_entry: dict[str, Any] = {
            'Step': steps,
            game_master.name: {'week_summary': summary},
            'Summary': f'Week {summary["week_number"]} {game_master.name}',
        }
        for entity_name, entity_log in entity_logs.items():
          log_entry[f'Entity [{entity_name}]'] = entity_log
        log.append(log_entry)

      if checkpoint_callback is not None:
        checkpoint_callback(steps)

      if step_callback is not None:
        step_callback(
            self._build_step_data(
                steps=steps,
                game_master=game_master,
                summary=summary,
                entity_logs=entity_logs,
            )
        )

  def _get_coordinator(
      self,
      game_master: entity_lib.Entity,
  ) -> hdb_coordinator_helper.WeeklyCoordinator:
    return game_master.get_component(
        'weekly_coordinator',
        type_=hdb_coordinator_helper.WeeklyCoordinator,
    )

# 
  def _deliver_pending_observations(
      self,
      *,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      active_player_ids: Sequence[str],
      verbose: bool,
  ) -> None:
    if not active_player_ids:
      return

    coordinator = self._get_coordinator(game_master)
    entity_by_id = {
        str(entity._hdb_player_id): entity
        for entity in entities
        if getattr(entity, '_hdb_player_id', '')
    }

    def _observe_entity(entity: entity_lib.Entity) -> None:
      observation = self.make_observation(game_master, entity)
      if not observation or not observation.strip():
        return
      if verbose:
        print(
            termcolor.colored(
                f'Entity {entity.name} observed: {observation}',
                _PRINT_COLOR,
            )
        )
      entity.observe(observation)

    tasks = {
        coordinator.get_player_name(player_id): functools.partial(
            _observe_entity,
            entity_by_id[player_id],
        )
        for player_id in active_player_ids
        if player_id in entity_by_id
    }
    for player_id in active_player_ids:
      if player_id in entity_by_id:
        continue
      logging.warning(
          'No registered entity found for negotiating player %s (%s).',
          coordinator.get_player_name(player_id),
          player_id,
      )
    if not tasks:
      return
    concurrency.run_tasks(tasks)

  def _run_week(
      self,
      game_master: entity_lib.Entity,
      *,
      verbose: bool,
  ) -> tuple[dict[str, Any], list[str], list[str]]:
    coordinator = self._get_coordinator(game_master)
    week_context = coordinator.prepare_week()
    listing_module = coordinator.get_listing_module()
    negotiation_module = coordinator.get_negotiation_module()
    active_negotiation_player_ids = sorted(
        {
            str(player_id)
            for pair in week_context['open_negotiation_pairs']
            for player_id in pair
        }
    )

    tasks: dict[str, Callable[[], Any]] = {}
    # TODO: add more modules if needed. 
    if week_context['listing_enabled']:
      tasks['listing'] = functools.partial(
          listing_module.run_week,
          week_number=week_context['week_number'],
          assigned_player_ids=week_context['listing_player_ids'],
      )
    if week_context['negotiation_enabled']:
      tasks['negotiation'] = functools.partial(
          negotiation_module.run_week,
          week_number=week_context['week_number'],
          new_negotiation_pairs=week_context['new_negotiation_pairs'],
      )

    results, errors = (
        concurrency.run_tasks_in_background(tasks) if tasks else ({}, {})
    )
    for module_name, error in errors.items():
      logging.error(
          'HDB weekly module "%s" failed; continuing with partial results: %s',
          module_name,
          error,
      )
    summary = coordinator.complete_week(
        listing_outcome=results.get('listing'),
        negotiation_outcome=results.get('negotiation'),
    )
    self.resolve(game_master, json.dumps(summary))

    if verbose:
      print(
          termcolor.colored(
              (
                  f'Completed week {summary["week_number"]}:\n'
                  f'{json.dumps(summary, indent=2, ensure_ascii=False)}'
              ),
              _PRINT_COLOR,
          )
      )
    return (
        summary,
        active_negotiation_player_ids,
        list(week_context['listing_player_ids']),
    )

  def _build_step_data(
      self,
      *,
      steps: int,
      game_master: entity_lib.Entity,
      summary: Mapping[str, Any],
      entity_logs: Mapping[str, Mapping[str, Any]],
  ) -> step_controller_lib.StepData:
    """Builds step data for the real-time debug UI after each weekly step."""
    week_number = summary.get('week_number', steps)
    entity_actions = {}
    for entity_name, entity_log in entity_logs.items():
      value = entity_log.get('Value')
      if value is None:
        continue
      entity_actions[entity_name] = str(value)

    action = json.dumps(summary, ensure_ascii=False, indent=2)
    return step_controller_lib.StepData(
        step=steps,
        acting_entity=game_master.name,
        action=f'Completed week {week_number}:\n{action}',
        entity_actions=entity_actions,
        entity_logs={name: dict(log) for name, log in entity_logs.items()},
        game_master=game_master.name,
    )
