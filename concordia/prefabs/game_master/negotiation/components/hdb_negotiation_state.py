"""Lightweight turn-order components for HDB negotiation game masters."""

from collections.abc import Mapping, Sequence

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import event_resolution as event_resolution_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_component
from concordia.environment import engine as engine_lib
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class PairRoundRobinNextActing(next_acting_component.NextActing):
  """Deterministic next-actor scheduler over negotiation pairs.

  Scheduling policy:
  - A round is complete when every pair in the queue has acted once by both
    participants.
  - Within a pair, actors alternate in fixed order: first participant, then
    second participant.
  - Pair queue order is fixed and repeated every round.

  Pairs can be expressed either with names or IDs. If `player_ids` are supplied,
  the scheduler stores and indexes pairs by ID internally and maps to names when
  returning the next active entity.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      negotiation_pairs: Sequence[Sequence[str] | Mapping[str, str]] | None = None,
      player_ids: Sequence[str] | None = None,
      max_rounds: int | None = None,
      pre_act_label: str = next_acting_component.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    super().__init__(
        model=model,
        player_names=player_names,
        components=(),
        pre_act_label=pre_act_label,
    )
    if not player_names:
      raise ValueError('`player_names` must not be empty.')

    self._max_rounds = max_rounds if max_rounds and max_rounds > 0 else None
    self._player_ids = tuple(player_ids) if player_ids else tuple(player_names)
    if len(self._player_ids) != len(player_names):
      raise ValueError('`player_ids` must have the same length as `player_names`.')
    if len(set(self._player_ids)) != len(self._player_ids):
      raise ValueError('`player_ids` must be unique.')

    self._id_to_name = dict(zip(self._player_ids, player_names))
    self._name_to_id = dict(zip(player_names, self._player_ids))
    self._pair_queue = self._build_pair_queue(negotiation_pairs)

    self._round_number = 1
    self._pair_index = 0
    self._turn_in_pair = 0
    self._total_turns = 0
    self._currently_active_player = None

  def _to_player_id(self, token: str) -> str:
    if token in self._id_to_name:
      return token
    if token in self._name_to_id:
      return self._name_to_id[token]
    raise ValueError(f'Unknown player token: {token}')

  def _parse_pair(self, pair: Sequence[str] | Mapping[str, str]) -> tuple[str, str]:
    if isinstance(pair, Mapping):
      first = (
          pair.get('first')
          or pair.get('left')
          or pair.get('buyer')
          or pair.get('buyer_id')
          or pair.get('a')
      )
      second = (
          pair.get('second')
          or pair.get('right')
          or pair.get('seller')
          or pair.get('seller_id')
          or pair.get('b')
      )
      if first is None or second is None:
        raise ValueError(
            'Pair mapping must contain two participants, for example '
            '`{"first": "...", "second": "..."}`.'
        )
      return self._to_player_id(str(first)), self._to_player_id(str(second))

    if len(pair) != 2:
      raise ValueError(f'Pair entries must have exactly 2 items: {pair}')
    return self._to_player_id(str(pair[0])), self._to_player_id(str(pair[1]))

  def _build_pair_queue(
      self,
      negotiation_pairs: Sequence[Sequence[str] | Mapping[str, str]] | None,
  ) -> list[tuple[str, str]]:
    if negotiation_pairs:
      pairs = [self._parse_pair(pair) for pair in negotiation_pairs]
      if not pairs:
        raise ValueError('`negotiation_pairs` cannot be empty.')
      return pairs

    if len(self._player_ids) % 2 != 0:
      raise ValueError(
          'Automatic pair creation requires an even number of players. '
          'Provide `negotiation_pairs` explicitly for odd counts.'
      )
    return [
        (self._player_ids[i], self._player_ids[i + 1])
        for i in range(0, len(self._player_ids), 2)
    ]

  def _peek_next_actor_id(self) -> str:
    active_pair = self._pair_queue[self._pair_index]
    return active_pair[self._turn_in_pair]

  def _advance(self) -> None:
    self._total_turns += 1
    if self._turn_in_pair == 0:
      self._turn_in_pair = 1
      return

    self._turn_in_pair = 0
    self._pair_index += 1
    if self._pair_index >= len(self._pair_queue):
      self._pair_index = 0
      self._round_number += 1

  def get_pair_queue_names(self) -> list[tuple[str, str]]:
    return [
        (self._id_to_name[first], self._id_to_name[second])
        for first, second in self._pair_queue
    ]

  def get_scheduler_snapshot(self) -> dict[str, int | str]:
    active_pair = self._pair_queue[self._pair_index]
    next_actor_id = self._peek_next_actor_id()
    max_rounds = self._max_rounds if self._max_rounds is not None else 0
    return {
        'round_number': self._round_number,
        'pair_index': self._pair_index,
        'turn_in_pair': self._turn_in_pair,
        'next_actor_name': self._id_to_name[next_actor_id],
        'active_pair_first_name': self._id_to_name[active_pair[0]],
        'active_pair_second_name': self._id_to_name[active_pair[1]],
        'total_turns': self._total_turns,
        'max_rounds': max_rounds,
    }

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''

    next_actor_id = self._peek_next_actor_id()
    self._currently_active_player = self._id_to_name[next_actor_id]
    self._advance()
    return self._currently_active_player

  def get_currently_active_player(self) -> str | None:
    return self._currently_active_player

  def get_state(self) -> entity_component.ComponentState:
    return {
        'currently_active_player': self._currently_active_player,
        'round_number': self._round_number,
        'pair_index': self._pair_index,
        'turn_in_pair': self._turn_in_pair,
        'total_turns': self._total_turns,
        'pair_queue': [list(pair) for pair in self._pair_queue],
        'player_ids': list(self._player_ids),
        'max_rounds': self._max_rounds if self._max_rounds is not None else 0,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._currently_active_player = state.get('currently_active_player')
    self._round_number = int(state.get('round_number', 1))
    self._pair_index = int(state.get('pair_index', 0))
    self._turn_in_pair = int(state.get('turn_in_pair', 0))
    self._total_turns = int(state.get('total_turns', 0))

    pair_queue_state = state.get('pair_queue')
    if pair_queue_state:
      self._pair_queue = [
          (str(pair[0]), str(pair[1])) for pair in pair_queue_state  # type: ignore[index]
      ]

    max_rounds = int(state.get('max_rounds', 0))
    self._max_rounds = max_rounds if max_rounds > 0 else None


class TurnOrderStateTracker(action_spec_ignored.ActionSpecIgnored):
  """Action-spec-ignored component exposing scheduler state as prompt context."""

  def __init__(
      self,
      scheduler_component_key: str = (
          next_acting_component.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
      pre_act_label: str = 'Negotiation turn scheduler',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._scheduler_component_key = scheduler_component_key

  def _make_pre_act_value(self) -> str:
    scheduler = self.get_entity().get_component(
        self._scheduler_component_key, type_=PairRoundRobinNextActing
    )
    snapshot = scheduler.get_scheduler_snapshot()
    queue = scheduler.get_pair_queue_names()
    queue_str = ', '.join([f'({a} <-> {b})' for a, b in queue])

    return (
        f"Round: {snapshot['round_number']}\n"
        f"Active pair index: {snapshot['pair_index']}\n"
        f"Active pair: {snapshot['active_pair_first_name']} <-> "
        f"{snapshot['active_pair_second_name']}\n"
        f"Current turn in pair: {snapshot['turn_in_pair']}\n"
        f"Next actor: {snapshot['next_actor_name']}\n"
        f"Total actor turns dispatched: {snapshot['total_turns']}\n"
        f'Pair queue: {queue_str}'
    )

  def get_state(self) -> entity_component.ComponentState:
    return {'scheduler_component_key': self._scheduler_component_key}

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'scheduler_component_key' in state:
      self._scheduler_component_key = str(state['scheduler_component_key'])


class FixedNextActionSpec(entity_component.ContextComponent):
  """Returns a deterministic next action spec for the currently active player."""

  def __init__(
      self,
      action_mode: str = 'free',
      call_to_action: str = 'What should {name} do next?',
      choice_options: Sequence[str] = (),
      next_acting_component_key: str = (
          next_acting_component.DEFAULT_NEXT_ACTING_COMPONENT_KEY
      ),
  ):
    super().__init__()
    normalized_mode = action_mode.strip().lower()
    if normalized_mode not in ('free', 'choice'):
      raise ValueError('`action_mode` must be either "free" or "choice".')
    if normalized_mode == 'choice' and not choice_options:
      raise ValueError('`choice_options` must be provided when action_mode=choice.')

    self._action_mode = normalized_mode
    self._call_to_action = call_to_action
    self._choice_options = tuple(choice_options)
    self._next_acting_component_key = next_acting_component_key

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''

    scheduler = self.get_entity().get_component(
        self._next_acting_component_key, type_=PairRoundRobinNextActing
    )
    active_player = scheduler.get_currently_active_player() or '{name}'
    prompt = self._call_to_action.format(name=active_player)

    if self._action_mode == 'choice':
      next_spec = entity_lib.choice_action_spec(
          call_to_action=prompt,
          options=self._choice_options,
      )
    else:
      next_spec = entity_lib.free_action_spec(call_to_action=prompt)

    return engine_lib.action_spec_to_string(next_spec)

  def get_state(self) -> entity_component.ComponentState:
    return {
        'action_mode': self._action_mode,
        'call_to_action': self._call_to_action,
        'choice_options': list(self._choice_options),
        'next_acting_component_key': self._next_acting_component_key,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'action_mode' in state:
      self._action_mode = str(state['action_mode'])
    if 'call_to_action' in state:
      self._call_to_action = str(state['call_to_action'])
    if 'choice_options' in state:
      self._choice_options = tuple(state['choice_options'])  # type: ignore[arg-type]
    if 'next_acting_component_key' in state:
      self._next_acting_component_key = str(state['next_acting_component_key'])


class PassthroughResolution(entity_component.ContextComponent):
  """Resolves the latest putative event without additional LLM processing."""

  def __init__(
      self,
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      notify_players: bool = True,
  ):
    super().__init__()
    self._memory_component_key = memory_component_key
    self._make_observation_component_key = make_observation_component_key
    self._notify_players = notify_players

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    suggestions = memory.scan(
        selector_fn=lambda x: event_resolution_component.PUTATIVE_EVENT_TAG in x
    )
    if not suggestions:
      return ''

    latest = suggestions[-1]
    marker = event_resolution_component.PUTATIVE_EVENT_TAG
    event = latest[latest.find(marker) + len(marker) :].strip()

    if self._notify_players and event:
      make_observation = self.get_entity().get_component(
          self._make_observation_component_key,
          type_=make_observation_component.MakeObservation,
      )
      make_observation.add_to_queue('all', event)
    return event

  def get_state(self) -> entity_component.ComponentState:
    return {
        'memory_component_key': self._memory_component_key,
        'make_observation_component_key': self._make_observation_component_key,
        'notify_players': int(self._notify_players),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'memory_component_key' in state:
      self._memory_component_key = str(state['memory_component_key'])
    if 'make_observation_component_key' in state:
      self._make_observation_component_key = str(
          state['make_observation_component_key']
      )
    if 'notify_players' in state:
      self._notify_players = bool(state['notify_players'])
