# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequential (turn-based) action engine.
"""

from collections.abc import Mapping, Sequence
import functools
import json
from typing import Any, Callable

from absl import logging
from concordia.components.game_master import event_resolution as event_resolution_components
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_components
from concordia.components.game_master import next_game_master as next_game_master_components
from concordia.components.game_master import switch_act as switch_act_component
from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from concordia.utils import concurrency
import termcolor


DEFAULT_CALL_TO_MAKE_OBSERVATION = (
    make_observation_component.DEFAULT_CALL_TO_MAKE_OBSERVATION)
DEFAULT_CALL_TO_NEXT_ACTING = next_acting_components.DEFAULT_CALL_TO_NEXT_ACTING
DEFAULT_CALL_TO_NEXT_ACTION_SPEC = (
    next_acting_components.DEFAULT_CALL_TO_NEXT_ACTION_SPEC)
DEFAULT_CALL_TO_RESOLVE = 'Because of all that came before, what happens next?'
DEFAULT_CALL_TO_CHECK_TERMINATION = 'Is the game/simulation finished?'
DEFAULT_CALL_TO_NEXT_GAME_MASTER = (
    next_game_master_components.DEFAULT_CALL_TO_NEXT_GAME_MASTER)

DEFAULT_ACT_COMPONENT_KEY = switch_act_component.DEFAULT_ACT_COMPONENT_KEY

PUTATIVE_EVENT_TAG = event_resolution_components.PUTATIVE_EVENT_TAG
EVENT_TAG = event_resolution_components.EVENT_TAG

_PRINT_COLOR = 'cyan'


def _get_empty_log_entry():
  """Returns a dictionary to store a single log entry."""
  return {
      'terminate': {},
      'next_game_master': {},
      'make_observation': {},
      'next_acting': {},
      'next_action_spec': {},
      'resolve': {},
  }


class Sequential(engine_lib.Engine):
  """Sequential action (turn-based) engine.

  When this engine is used, one entity is acting at a time. The game master
  decides which entity to ask for an action on each step. The entity then
  decides what to do next, which is passed to the game master for resolution.
  The game master prepares observations for all entities in parallel.
  """

  def __init__(
      self,
      call_to_make_observation: str = DEFAULT_CALL_TO_MAKE_OBSERVATION,
      call_to_next_acting: str = DEFAULT_CALL_TO_NEXT_ACTING,
      call_to_next_action_spec: str = DEFAULT_CALL_TO_NEXT_ACTION_SPEC,
      call_to_resolve: str = DEFAULT_CALL_TO_RESOLVE,
      call_to_check_termination: str = DEFAULT_CALL_TO_CHECK_TERMINATION,
      call_to_next_game_master: str = DEFAULT_CALL_TO_NEXT_GAME_MASTER,
  ):
    """Sequential engine constructor."""
    self._call_to_make_observation = call_to_make_observation
    self._call_to_next_acting = call_to_next_acting
    self._call_to_next_action_spec = call_to_next_action_spec
    self._call_to_resolve = call_to_resolve
    self._call_to_check_termination = call_to_check_termination
    self._call_to_next_game_master = call_to_next_game_master

  def make_observation(self,
                       game_master: entity_lib.Entity,
                       entity: entity_lib.Entity) -> str:
    """Make an observation for a game object."""
    observation = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_make_observation.format(
                name=entity.name),
            output_type=entity_lib.OutputType.MAKE_OBSERVATION,
        )
    )
    return observation

  def next_acting(
      self,
      game_master: entity_lib.Entity,
      entities: Sequence[entity_lib.Entity],
      log_entry: Mapping[str, Any] | None = None,
      log: list[Mapping[str, Any]] | None = None,
  ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
    """Return the next entity or entities to act."""
    entities_by_name = {
        entity.name: entity for entity in entities
    }
    next_object_name = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_acting,
            output_type=entity_lib.OutputType.NEXT_ACTING,
            options=tuple(entities_by_name.keys()),
        )
    )
    if log is not None and hasattr(game_master, 'get_last_log'):
      assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
      log_entry['next_acting'] = game_master.get_last_log()
    next_action_spec_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_action_spec.format(
                name=next_object_name),
            output_type=entity_lib.OutputType.NEXT_ACTION_SPEC,
        )
    )
    if log is not None and hasattr(game_master, 'get_last_log'):
      assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
      log_entry['next_action_spec'] = game_master.get_last_log()
    next_action_spec = engine_lib.action_spec_parser(next_action_spec_string)
    return (entities_by_name[next_object_name], next_action_spec)

  def resolve(self,
              game_master: entity_lib.Entity,
              putative_event: str,
              verbose: bool = False) -> None:
    """Resolve an event."""
    game_master.observe(observation=f'{PUTATIVE_EVENT_TAG} {putative_event}')
    result = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_resolve,
            output_type=entity_lib.OutputType.RESOLVE,
        )
    )
    game_master.observe(observation=f'{EVENT_TAG} {result}')

  def terminate(self,
                game_master: entity_lib.Entity,
                verbose: bool = False) -> bool:
    """Decide if the episode should terminate."""
    should_terminate_string = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_check_termination,
            output_type=entity_lib.OutputType.TERMINATE,
            options=tuple(entity_lib.BINARY_OPTIONS.values()),
        )
    )
    if verbose:
      print(termcolor.colored(
          f'Terminate? {should_terminate_string}', _PRINT_COLOR))
    return should_terminate_string == entity_lib.BINARY_OPTIONS['affirmative']

  def next_game_master(self,
                       game_master: entity_lib.Entity,
                       game_masters: Sequence[entity_lib.Entity],
                       verbose: bool = False) -> entity_lib.Entity:
    """Select which game master to use for the next step."""
    if len(game_masters) == 1:
      if verbose:
        print(termcolor.colored(
            (f'Only one game master available ({game_masters[0].name}), '
             'skipping the call to `next_game_master`.'),
            _PRINT_COLOR))
      return game_masters[0]
    game_masters_by_name = {
        game_master_.name: game_master_ for game_master_ in game_masters
    }
    next_game_master_name = game_master.act(
        action_spec=entity_lib.ActionSpec(
            call_to_action=self._call_to_next_game_master,
            output_type=entity_lib.OutputType.NEXT_GAME_MASTER,
            options=tuple(game_masters_by_name.keys()),
        )
    )
    if verbose:
      print(termcolor.colored(
          f'Game master: {next_game_master_name}', _PRINT_COLOR))
    if next_game_master_name not in game_masters_by_name:
      raise ValueError(
          f'Selected game master "{next_game_master_name}" not found in:'
          f' {game_masters_by_name.keys()}'
      )
    return game_masters_by_name[next_game_master_name]

  def _pair_round_snapshot(
      self, game_master: entity_lib.Entity
  ) -> tuple[list[tuple[str, str]], list[int]] | None:
    """Best-effort snapshot of pair labels and pair-local round numbers."""
    if not hasattr(game_master, 'get_component'):
      return None
    try:
      scheduler = game_master.get_component(
          next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    if not hasattr(scheduler, 'get_state'):
      return None
    try:
      state = scheduler.get_state()
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    pair_round_numbers = state.get('pair_round_numbers')
    pair_queue = state.get('pair_queue')
    if not isinstance(pair_round_numbers, list) or not isinstance(pair_queue, list):
      return None

    pairs: list[tuple[str, str]] = []
    for pair in pair_queue:
      if isinstance(pair, (list, tuple)) and len(pair) == 2:
        pairs.append((str(pair[0]), str(pair[1])))
      else:
        pairs.append(('<unknown>', '<unknown>'))

    rounds: list[int] = []
    for value in pair_round_numbers:
      try:
        rounds.append(int(value))
      except (TypeError, ValueError):
        rounds.append(0)

    return pairs, rounds

  def _global_round_snapshot(self, game_master: entity_lib.Entity) -> int | None:
    """Best-effort snapshot of scheduler global round number."""
    if not hasattr(game_master, 'get_component'):
      return None
    try:
      scheduler = game_master.get_component(
          next_acting_components.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    if not hasattr(scheduler, 'get_state'):
      return None
    try:
      state = scheduler.get_state()
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    round_number = state.get('round_number')
    try:
      return int(round_number)
    except (TypeError, ValueError):
      return None

  @staticmethod
  def _format_action_for_display(action: str) -> str:
    """Format `[ACTED]` JSON payload for readable logs.

    - Keeps `[ACTED]` prefix.
    - Pretty-prints JSON with one field per line.
    - Reorders object keys with `type` first.
    """
    actor, sep, payload = action.partition(':')
    if not sep:
      return action
    payload = payload.strip()
    acted_prefix = '[ACTED]'
    if payload.startswith(acted_prefix):
      payload_json = payload[len(acted_prefix):].strip()
      display_prefix = acted_prefix
    else:
      payload_json = payload
      display_prefix = ''

    try:
      parsed = json.loads(payload_json)
    except json.JSONDecodeError:
      return action

    if isinstance(parsed, dict):
      ordered: dict[str, Any] = {}
      if 'type' in parsed:
        ordered['type'] = parsed['type']
      for key, value in parsed.items():
        if key != 'type':
          ordered[key] = value
      pretty_payload = json.dumps(ordered, ensure_ascii=False, indent=2)
    else:
      pretty_payload = json.dumps(parsed, ensure_ascii=False, indent=2)

    prefix = f'{display_prefix} ' if display_prefix else ''
    return f'{actor}: {prefix}{pretty_payload}'

  def run_loop(
      self,
      game_masters: Sequence[entity_lib.Entity | entity_lib.EntityWithLogging],
      entities: Sequence[entity_lib.Entity | entity_lib.EntityWithLogging],
      premise: str = '',
      max_steps: int = 100,
      verbose: bool = False,
      log: list[Mapping[str, Any]] | None = None,
      checkpoint_callback: Callable[[int], None] | None = None,
  ):
    """Run a game loop."""
    if not game_masters:
      raise ValueError('No game masters provided.')

    log_entry = _get_empty_log_entry()
    steps = 0
    game_master = game_masters[0]
    global_round_by_game_master: dict[str, int] = {}
    pair_rounds_by_game_master: dict[str, list[int]] = {}
    if verbose:
      for gm in game_masters:
        round_number = self._global_round_snapshot(gm)
        if round_number is not None:
          global_round_by_game_master[gm.name] = round_number
        pair_snapshot = self._pair_round_snapshot(gm)
        if pair_snapshot is not None:
          _, rounds = pair_snapshot
          pair_rounds_by_game_master[gm.name] = list(rounds)
    if premise:
      premise = f'{EVENT_TAG} {premise}'
      game_master.observe(premise)
    while not self.terminate(game_master, verbose) and steps < max_steps:
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['terminate'] = game_master.get_last_log()

      game_master = self.next_game_master(game_master, game_masters, verbose)
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['next_game_master'] = game_master.get_last_log()

      # Define a function to make an entity's observation and send it to them.
      def _entity_observation(entity: entity_lib.Entity) -> None:
        observation = self.make_observation(game_master, entity)
        # Only observe if the observation is not an empty or whitespace string
        if observation and observation.strip():
          tagged_observation = observation.strip()
          if not tagged_observation.startswith('[OBSERVED]'):
            tagged_observation = f'[OBSERVED] {tagged_observation}'
          entity.observe(tagged_observation)

      tasks = {
          entity.name: functools.partial(_entity_observation, entity)
          for entity in entities
      }
      concurrency.run_tasks(tasks, max_workers=1)

      next_entity, entity_spec_to_use = self.next_acting(
          game_master, entities, log_entry=log_entry, log=log)
      if verbose:
        current_round = self._global_round_snapshot(game_master)
        previous_round = global_round_by_game_master.get(game_master.name)
        if (
            current_round is not None
            and previous_round is not None
            and current_round != previous_round
        ):
          print(termcolor.colored(f'Week: {current_round}', _PRINT_COLOR))
        if current_round is not None:
          global_round_by_game_master[game_master.name] = current_round

        pair_snapshot = self._pair_round_snapshot(game_master)
        if pair_snapshot is not None:
          _, current_pair_rounds = pair_snapshot
          pair_rounds_by_game_master[game_master.name] = list(current_pair_rounds)

      if entity_spec_to_use.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
        # It is often useful to have a game master that does not allow players
        # to take actions. For example, the game master may
        # initialize other players and game masters. In this case, we skip the
        # current step and continue to the next step.
        if verbose:
          print(termcolor.colored(
              '\nSkipping the action phase for the current time step.\n'))
        if checkpoint_callback is not None:
          logging.debug('Calling checkpoint callback at step %s', steps)
          checkpoint_callback(steps)
        steps += 1
        continue

      if verbose:
        choices_text = ', '.join(entity_spec_to_use.options) if entity_spec_to_use.options else ''
        print(termcolor.colored(
            f'Entity {next_entity.name} is next to act. They must respond'
            f' with the choices: "{choices_text}".', _PRINT_COLOR))
      raw_action = next_entity.act(entity_spec_to_use)
      actor_prefix = f'{next_entity.name}:'
      stripped_action = raw_action.strip()
      if stripped_action.startswith(actor_prefix):
        _, _, action_payload = stripped_action.partition(':')
        action_payload = action_payload.strip()
      else:
        action_payload = stripped_action
      if action_payload.startswith('[ACTED]'):
        action = f'{next_entity.name}: {action_payload}'
      else:
        action = f'{next_entity.name}: [ACTED] {action_payload}'
      if verbose:
        display_action = self._format_action_for_display(action)
        print(termcolor.colored(
            display_action, _PRINT_COLOR))

      self.resolve(game_master=game_master,
                   putative_event=action,
                   verbose=verbose)

      steps += 1
      if log is not None and hasattr(game_master, 'get_last_log'):
        assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
        log_entry['resolve'] = game_master.get_last_log()
        next_entity_log = {}
        game_master_key = game_master.name
        entity_key = 'Entity'
        if hasattr(next_entity, 'get_last_log'):
          assert hasattr(game_master, 'get_last_log')  # Assertion for pytype
          next_entity_log = next_entity.get_last_log()
          entity_key = f'{entity_key} [{next_entity.name}]'
        if DEFAULT_ACT_COMPONENT_KEY in log_entry['resolve']:
          event_to_log = log_entry['resolve'][
              DEFAULT_ACT_COMPONENT_KEY
          ]['Value']
          game_master_key = f'{game_master_key} --- {event_to_log}'
        self._log(
            log=log,
            steps=steps,
            entity_key=entity_key,
            entity_log=next_entity_log,
            game_master_key=game_master_key,
            game_master_log=log_entry,
        )
        log_entry = _get_empty_log_entry()

      if checkpoint_callback is not None:
        checkpoint_callback(steps)

  def _log(
      self,
      log: list[Mapping[str, Any]],
      steps: int,
      entity_key: str,
      entity_log: Mapping[str, Any],
      game_master_key: str,
      game_master_log: Mapping[str, Any],
  ):
    """Modify log in place to append a new entry."""
    game_master_finalized_log = {}
    for segment_key, segment_log in game_master_log.items():
      game_master_finalized_log[segment_key] = {}
      for component_key, component_value in segment_log.items():
        if component_value:
          tmp_log_dict = {
              key: value for key, value in component_value.items() if value
          }
          if len(tmp_log_dict) > 1:
            # Only log if component logged more than just a key.
            game_master_finalized_log[segment_key][component_key] = tmp_log_dict

    log.append({
        'Step': steps,
        entity_key: entity_log,
        game_master_key: game_master_finalized_log,
        'Summary': f'Step {steps} {game_master_key}',
    })
