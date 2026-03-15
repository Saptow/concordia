"""HDB negotiation module with pair-level weekly execution."""

from collections.abc import Mapping, Sequence
import functools
import json
from typing import Any

from absl import logging
from concordia.associative_memory import basic_associative_memory
from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import make_observation as make_observation_component
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import uncertain_negotiator
from concordia.prefabs.game_master.negotiation.components import (
    hdb_negotiation_helpers,
)
from concordia.utils import concurrency
from pydantic import ValidationError
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


def _empty_outcome() -> dict[str, Any]:
  return {
      'number_of_pairs_negotiated': 0,
      'events': [],
      'closed_pairs': [],
      'successful_pairs': [],
      'failed_pairs': [],
  }


class NegotiationModule(action_spec_ignored.ActionSpecIgnored):
  """Runs HDB negotiations with ids as primary keys and names as display labels."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
      participant_specs: Mapping[str, Any] | str,
      negotiation_pairs: Sequence[Mapping[str, str]] | None = None,
      action_prompt: str = 'What should {name} do next?',
      max_rounds: int = 0,
      enabled: bool = True,
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = 'Negotiation module',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._enabled = bool(enabled)
    if not self._enabled:
      return # skip setup if module is disabled; allows for dynamic enabling later with less overhead
    self._model = model
    self._memory_bank = memory_bank
    self._participant_specs = self._normalize_participant_specs(participant_specs)
    if not self._participant_specs:
      logging.error('NegotiationModule requires at least one participant spec.')
      self._enabled = False
      return

    self._player_ids = tuple(self._participant_specs.keys())
    self._id_to_name = {
        player_id: str(spec['name']) for player_id, spec in self._participant_specs.items()
    }
    self._player_names = tuple(self._id_to_name[player_id] for player_id in self._player_ids)

    self._action_prompt = action_prompt
    self._make_observation_component_key = make_observation_component_key

    self._entities_by_id: dict[str, Any] = {}

    self._scheduler = hdb_negotiation_helpers.NegotiationScheduler(
        player_names=self._player_names,
        negotiation_pairs=negotiation_pairs,
        player_ids=self._player_ids,
        max_rounds=max_rounds if max_rounds > 0 else None,
    )
    self._offer_tracker = hdb_negotiation_helpers.ActiveOfferTracker(self._scheduler)
    self._initialize_entities_for_pairs(
        hdb_negotiation_helpers.pair_mappings_from_pair_ids(
            self._scheduler.get_state().get('pair_queue', [])
        )
    )

  # Participant normalization
  @staticmethod
  def _normalize_participant_specs(
      participant_specs: Mapping[str, Any] | str,
  ) -> dict[str, dict[str, Any]]:
    """Validates participant configs through negotiation buyer/seller schemas."""
    if isinstance(participant_specs, str):
      parsed = json.loads(participant_specs) if participant_specs.strip() else {}
    else:
      parsed = dict(participant_specs)

    normalized: dict[str, dict[str, Any]] = {}
    for player_id, raw_spec in parsed.items():
      if not isinstance(raw_spec, Mapping):
        logging.error('Participant spec for %s must be a mapping.', player_id)
        continue
      normalized_player_id = str(player_id)
      spec = {'id': normalized_player_id, **dict(raw_spec)}
      role = str(spec.get('role', '')).strip().lower()
      schema_model: type[
          negotiation_schemas.NegotiationBuyer
          | negotiation_schemas.NegotiationSeller
      ] | None = None

      if role == negotiation_schemas.RoleType.BUYER.value:
        schema_model = negotiation_schemas.NegotiationBuyer
      elif role == negotiation_schemas.RoleType.SELLER.value:
        schema_model = negotiation_schemas.NegotiationSeller
      else:
        modules = str(spec.get('modules', ''))
        if 'uncertain_buyer' in modules:
          schema_model = negotiation_schemas.NegotiationBuyer
        elif 'uncertain_seller' in modules:
          schema_model = negotiation_schemas.NegotiationSeller

      if schema_model is None:
        logging.error(
            'Participant spec for %s must map to NegotiationBuyer or '
            'NegotiationSeller.',
            normalized_player_id,
        )
        continue

      try:
        validated_spec = schema_model.model_validate(spec)
      except ValidationError as error:
        logging.error(
            'Participant spec for %s failed %s validation: %s',
            normalized_player_id,
            schema_model.__name__,
            error,
        )
        continue

      normalized[normalized_player_id] = validated_spec.model_dump(mode='json')
    return normalized

  # Entity and pair initialization
  def _make_stage_memory_bank(self) -> basic_associative_memory.AssociativeMemoryBank:
    return basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=self._memory_bank._embedder,
        allow_duplicates=False,
    )

  def _initialize_entity(self, player_id: str) -> bool:
    """Builds the negotiation entity for a participant id on first use."""
    if player_id in self._entities_by_id:
      return True
    spec = self._participant_specs.get(player_id)
    if spec is None:
      logging.error('No participant spec found for player id: %s', player_id)
      return False

    entity = uncertain_negotiator.Entity(params=spec).build(
        model=self._model,
        memory_bank=self._make_stage_memory_bank(),
    )
    self._entities_by_id[player_id] = entity
    return True

  def _pair_exists(self, buyer_id: str, seller_id: str) -> bool:
    pair_queue = self._scheduler.get_state().get('pair_queue', [])
    normalized = (buyer_id, seller_id)
    return any(
        tuple(str(token) for token in pair) == normalized for pair in pair_queue
    )

  def _register_pair(self, buyer_id: str, seller_id: str) -> None:
    """Registers a new pair with the scheduler and offer tracker."""
    if self._pair_exists(buyer_id, seller_id):
      return
    self._scheduler.append_pair(buyer_id, seller_id)
    self._offer_tracker.register_pair(buyer_id, seller_id)

  def _initialize_entities_for_pairs(
      self,
      new_negotiation_pairs: Sequence[Mapping[str, str]],
  ) -> list[tuple[str, str]]:
    """Initializes both participants for each valid negotiation pair."""
    normalized_pairs: list[tuple[str, str]] = []
    for pair in new_negotiation_pairs:
      buyer_id, seller_id = hdb_negotiation_helpers.normalize_negotiation_pair(pair)
      if not buyer_id or not seller_id:
        continue
      if buyer_id not in self._participant_specs or seller_id not in self._participant_specs:
        logging.warning(
            'Skipping negotiation pair with unknown participant ids: %s',
            (buyer_id, seller_id),
        )
        continue
      buyer_initialized = self._initialize_entity(buyer_id)
      seller_initialized = self._initialize_entity(seller_id)
      if not buyer_initialized or not seller_initialized:
        logging.warning(
            'Skipping negotiation pair due to missing entity initialization: %s',
            (buyer_id, seller_id),
        )
        continue
      self._register_pair(buyer_id, seller_id)
      normalized_pairs.append((buyer_id, seller_id))
    return normalized_pairs

  # Module state
  def set_enabled(self, enabled: bool) -> None:
    self._enabled = bool(enabled)

  def is_enabled(self) -> bool:
    return self._enabled

  def is_finished(self) -> bool:
    return not self._enabled or self._scheduler.all_pairs_closed()

  # Display and observation helpers
  def _get_player_name(self, player_id: str) -> str:
    return self._id_to_name.get(player_id, player_id)

  def _get_make_observation(self) -> make_observation_component.MakeObservation:
    return self.get_entity().get_component(
        self._make_observation_component_key,
        type_=make_observation_component.MakeObservation,
    )

  def get_open_pairs(self) -> list[tuple[str, str]]:
    """Returns currently open pairs as `(buyer_id, seller_id)` tuples."""
    if not self._enabled:
      return []
    self._offer_tracker._ensure_initialized()
    open_pairs: list[tuple[str, str]] = []
    for pair_key in self._offer_tracker._pair_order:
      if pair_key in self._offer_tracker._closed_pairs:
        continue
      buyer_id, seller_id = self._offer_tracker._pair_members[pair_key]
      open_pairs.append((buyer_id, seller_id))
    return open_pairs

  @staticmethod
  def _format_entity_action(entity_name: str, raw_action: str) -> str:
    """Normalizes raw entity output into the shared event transcript format."""
    stripped_action = raw_action.strip()
    actor_prefix = f'{entity_name}:'
    if stripped_action.startswith(actor_prefix):
      _, _, payload = stripped_action.partition(':')
      payload = payload.strip()
    else:
      payload = stripped_action
    if payload.startswith('[ACTED]'):
      return f'{entity_name}: {payload}'
    return f'{entity_name}: [ACTED] {payload}'

  @staticmethod
  def _sanitize_event_for_counterparty(event: str) -> str:
    """Removes private reasoning before the event is observed by the pair."""
    actor, sep, payload = event.partition(':')
    if not sep:
      return event
    payload_json = hdb_negotiation_helpers.ActiveOfferTracker._extract_json_object(
        payload
    )
    if not payload_json:
      return event
    try:
      action = json.loads(payload_json)
    except json.JSONDecodeError:
      return event
    if not isinstance(action, dict):
      return event
    action.pop('internal_reasoning', None)
    sanitized_json = json.dumps(action, ensure_ascii=False)
    start = payload.find(payload_json)
    if start < 0:
      return event
    end = start + len(payload_json)
    return f'{actor}{sep}{payload[:start]}{sanitized_json}{payload[end:]}'

  def _queue_event_to_pair(self, event: str, actor_id: str) -> None:
    """Queues a sanitized event to both members of the actor's pair."""
    pair_members = self._offer_tracker.get_pair_members_for_player(actor_id)
    if not pair_members:
      return
    observed_event = self._sanitize_event_for_counterparty(event)
    make_observation = self._get_make_observation()
    make_observation.add_to_queue(self._get_player_name(pair_members[0]), observed_event)
    make_observation.add_to_queue(self._get_player_name(pair_members[1]), observed_event)

  def _closed_pair_records(
      self,
      pair_keys: Sequence[str],
  ) -> list[dict[str, str]]:
    """Builds summary records for pairs that closed this week."""
    records: list[dict[str, str]] = []
    for pair_key in pair_keys:
      buyer_id, seller_id = self._offer_tracker._pair_members[pair_key]
      outcome = self._offer_tracker._closed_pair_outcomes.get(
          pair_key,
          negotiation_schemas.NegotiationOutcome.CLOSED,
      )
      records.append({
          'buyer_id': buyer_id,
          'buyer_name': self._get_player_name(buyer_id),
          'seller_id': seller_id,
          'seller_name': self._get_player_name(seller_id),
          'outcome': outcome.value,
        })
    return records

  # Pair execution
  def _execute_player_turn(
      self,
      player_id: str,
      *,
      has_active_offer: bool,
  ) -> tuple[str | None, bool]:
    """Runs one player turn and returns `(event, should_force_close_pair)`."""
    action_spec = self._build_action_spec_for_pair_state(
        player_id,
        has_active_offer=has_active_offer,
    )
    if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
      return None, False

    entity = self._entities_by_id.get(player_id)
    if entity is None:
      logging.error(
          'No negotiation entity initialized for %s (%s). Closing pair and continuing.',
          self._get_player_name(player_id),
          player_id,
      )
      return None, True

    raw_action = entity.act(action_spec)
    event = self._format_entity_action(self._get_player_name(player_id), raw_action)
    return event, False

  def _build_action_spec_for_pair_state(
      self,
      player_id: str,
      *,
      has_active_offer: bool,
  ) -> entity_lib.ActionSpec:
    """Builds the action spec from pair-local offer state."""
    pair_members = self._offer_tracker.get_pair_members_for_player(player_id)
    if not pair_members:
      return entity_lib.ActionSpec(
          call_to_action='',
          output_type=entity_lib.OutputType.SKIP_THIS_STEP,
      )
    role = (
        negotiation_schemas.RoleType.BUYER
        if pair_members[0] == player_id
        else negotiation_schemas.RoleType.SELLER
    )
    allowed_actions = tuple(
        negotiation_schemas.get_allowed_action_types(role, has_active_offer)
    )
    if not allowed_actions:
      return entity_lib.ActionSpec(
          call_to_action='',
          output_type=entity_lib.OutputType.SKIP_THIS_STEP,
      )
    prompt = self._action_prompt.format(name=self._get_player_name(player_id))
    return entity_lib.choice_action_spec(
        call_to_action=prompt,
        options=allowed_actions,
    )

  @staticmethod
  def _advance_pair_local_state(
      *,
      actor_id: str,
      buyer_id: str,
      event: str,
      has_active_offer: bool,
      is_closed: bool,
  ) -> tuple[bool, bool]:
    """Updates local pair state from one event without mutating shared trackers."""
    if is_closed:
      return has_active_offer, is_closed
    _, sep, payload = event.partition(':')
    if not sep:
      return has_active_offer, is_closed
    payload_json = hdb_negotiation_helpers.ActiveOfferTracker._extract_json_object(
        payload
    )
    if not payload_json:
      return has_active_offer, is_closed
    try:
      action = json.loads(payload_json)
    except json.JSONDecodeError:
      return has_active_offer, is_closed
    action_type = str(action.get('type', '')).strip().upper()
    if action_type in ('MAKE_OFFER', 'MAKE_COUNTEROFFER'):
      return True, False
    if action_type == 'REJECT_OFFER':
      return False, False
    if action_type == 'ACCEPT_OFFER':
      return False, True
    if action_type == 'WALK_AWAY' and actor_id == buyer_id:
      return False, True
    return has_active_offer, is_closed

  def _run_pair_task(
      self,
      buyer_id: str,
      seller_id: str,
      *,
      has_active_offer: bool,
      is_closed: bool,
  ) -> dict[str, Any]:
    """Runs one weekly pair slice: buyer once, then seller once if still open."""
    if is_closed:
      return {
          'buyer_id': buyer_id,
          'seller_id': seller_id,
          'events': [],
      }

    pair_events: list[dict[str, str]] = []
    local_has_active_offer = has_active_offer
    local_is_closed = is_closed
    force_close = False

    buyer_event, should_close_pair = self._execute_player_turn(
        buyer_id,
        has_active_offer=local_has_active_offer,
    )
    force_close = force_close or should_close_pair
    if buyer_event is not None:
      pair_events.append({'actor_id': buyer_id, 'event': buyer_event})
      local_has_active_offer, local_is_closed = self._advance_pair_local_state(
          actor_id=buyer_id,
          buyer_id=buyer_id,
          event=buyer_event,
          has_active_offer=local_has_active_offer,
          is_closed=local_is_closed,
      )

    if not local_is_closed and not force_close:
      seller_event, should_close_pair = self._execute_player_turn(
          seller_id,
          has_active_offer=local_has_active_offer,
      )
      force_close = force_close or should_close_pair
      if seller_event is not None:
        pair_events.append({'actor_id': seller_id, 'event': seller_event})

    return {
        'buyer_id': buyer_id,
        'seller_id': seller_id,
        'force_close': force_close,
        'events': pair_events,
    }

  # Weekly execution
  def run_week(
      self,
      *,
      week_number: int,
      new_negotiation_pairs: Sequence[Mapping[str, str]] = (),
  ) -> dict[str, Any]:
    """Runs one weekly negotiation step across all open pairs in parallel."""
    if not self._enabled:
      return _empty_outcome()

    if new_negotiation_pairs:
      self._initialize_entities_for_pairs(new_negotiation_pairs)

    if self._scheduler.all_pairs_closed():
      return _empty_outcome()

    self._offer_tracker._ensure_initialized()
    closed_before = set(self._offer_tracker._closed_pairs)
    open_pairs = self._scheduler.get_open_pair_queue_ids()
    negotiated_pairs: list[tuple[str, str]] = []
    number_of_pairs_negotiated = 0
    events: list[str] = []

    pair_tasks = {
        f'{buyer_id}|||{seller_id}': functools.partial(
            self._run_pair_task,
            buyer_id,
            seller_id,
            has_active_offer=self._offer_tracker.has_active_offer_for_player(
                buyer_id
            ),
            is_closed=self._scheduler.is_pair_closed(buyer_id, seller_id),
        )
        for buyer_id, seller_id in open_pairs
    }
    results, errors = (
        concurrency.run_tasks_in_background(pair_tasks) if pair_tasks else ({}, {})
    )
    for pair_key, error in errors.items():
      logging.error(
          'Failed to run negotiation pair %s for week %s: %s',
          pair_key,
          week_number,
          error,
      )

    for buyer_id, seller_id in open_pairs:
      pair_key = hdb_negotiation_helpers.pair_key(buyer_id, seller_id)
      pair_result = results.get(pair_key)
      if pair_result is None:
        continue
      if pair_result.get('force_close', False):
        self._offer_tracker.close_pair(
            buyer_id,
            seller_id,
            outcome=negotiation_schemas.NegotiationOutcome.CLOSED,
        )
        continue
      pair_events = pair_result.get('events', [])
      if not pair_events:
        continue
      number_of_pairs_negotiated += 1
      negotiated_pairs.append((buyer_id, seller_id))
      for pair_event in pair_events:
        actor_id = str(pair_event['actor_id'])
        event = str(pair_event['event'])
        self._offer_tracker.record_resolved_event(event, actor_id=actor_id)
        self._queue_event_to_pair(event, actor_id=actor_id)
        events.append(event)

    self._scheduler.advance_week(negotiated_pairs)

    newly_closed = [
        pair_key
        for pair_key in self._offer_tracker._pair_order
        if pair_key in self._offer_tracker._closed_pairs and pair_key not in closed_before
    ]
    closed_records = self._closed_pair_records(newly_closed)
    successful_pairs = [
        record
        for record in closed_records
        if record['outcome'] == negotiation_schemas.NegotiationOutcome.SUCCESS.value
    ]
    failed_pairs = [
        record
        for record in closed_records
        if record['outcome'] != negotiation_schemas.NegotiationOutcome.SUCCESS.value
    ]

    return {
        'week_number': week_number,
        'number_of_pairs_negotiated': number_of_pairs_negotiated,
        'events': events,
        'closed_pairs': closed_records,
        'successful_pairs': successful_pairs,
        'failed_pairs': failed_pairs,
    }

  # Serialization
  def _make_pre_act_value(self) -> str:
    snapshot = {
        'enabled': self._enabled,
        'open_pairs': [list(pair) for pair in self.get_open_pairs()],
        'initialized_entities': sorted(self._entities_by_id.keys()),
        'scheduler_state': self._scheduler.get_state(),
        'offer_state': self._offer_tracker.get_state(),
    }
    return json.dumps(snapshot)

  def get_state(self) -> entity_component.ComponentState:
    """Serializes module state, entity state, scheduler state, and offer state."""
    entity_states = {
        player_id: entity.get_state()
        for player_id, entity in self._entities_by_id.items()
    }
    return {
        'participant_specs': self._participant_specs,
        'entity_states': entity_states,
        'scheduler_state': self._scheduler.get_state(),
        'offer_state': self._offer_tracker.get_state(),
        'action_prompt': self._action_prompt,
        'make_observation_component_key': self._make_observation_component_key,
        'enabled': int(self._enabled),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores module state and re-initializes entities for tracked pairs."""
    if 'participant_specs' in state:
      self._participant_specs = self._normalize_participant_specs(
          state['participant_specs']  # type: ignore[arg-type]
      )
      self._player_ids = tuple(self._participant_specs.keys())
      self._id_to_name = {
          player_id: str(spec['name'])
          for player_id, spec in self._participant_specs.items()
      }
      self._player_names = tuple(
          self._id_to_name[player_id] for player_id in self._player_ids
      )
      self._scheduler.set_player_names(self._id_to_name)
    if 'scheduler_state' in state:
      self._scheduler.set_state(state['scheduler_state'])
    if 'offer_state' in state:
      self._offer_tracker.set_state(state['offer_state'])
    if 'entity_states' in state:
      pair_queue = self._scheduler.get_state().get('pair_queue', [])
      self._initialize_entities_for_pairs(
          hdb_negotiation_helpers.pair_mappings_from_pair_ids(pair_queue)
      )
      entity_states = state['entity_states']  # type: ignore[assignment]
      for player_id, entity_state in entity_states.items():  # type: ignore[union-attr]
        normalized_player_id = str(player_id)
        if normalized_player_id not in self._entities_by_id:
          self._initialize_entity(normalized_player_id)
        self._entities_by_id[normalized_player_id].set_state(entity_state)
    if 'action_prompt' in state:
      self._action_prompt = str(state['action_prompt'])
    if 'make_observation_component_key' in state:
      self._make_observation_component_key = str(
          state['make_observation_component_key']
      )
    self._enabled = bool(state.get('enabled', 1))
