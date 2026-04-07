"""HDB negotiation module with pair-level weekly execution."""

from collections.abc import Mapping, Sequence
import functools
import json
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.game_master import make_observation as make_observation_component
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.prefabs.entity.negotiation.uncertain_negotiator import (
    update_agent_from_listing,
)
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
      entities: Sequence[entity_component.EntityWithComponents] = (),
      participant_specs: Mapping[str, Any] | str,
      negotiation_pairs: Sequence[Sequence[str]] | None = None,
      action_prompt: str = 'What should {name} do next?',
      max_rounds: int = 0,
      enabled: bool = True,
      make_observation_component_key: str = (
          make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
      ),
      pre_act_label: str = 'Negotiation module',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._action_prompt = action_prompt
    self._make_observation_component_key = make_observation_component_key
    self._enabled = bool(enabled)
    self._participant_specs: dict[str, dict[str, Any]] = {}
    self._player_ids: tuple[str, ...] = ()
    self._player_names: tuple[str, ...] = ()
    self._id_to_name: dict[str, str] = {}
    self._entities_by_id: dict[str, Any] = {}
    self._canonical_entities: tuple[entity_component.EntityWithComponents, ...] = ()
    self._canonical_entities_by_name: dict[str, entity_component.EntityWithComponents] = {}
    self._pair_start_weeks: dict[str, int] = {}
    self._conversation_replays: dict[str, dict[str, Any]] = {}
    self._scheduler = hdb_negotiation_helpers.NegotiationScheduler(
        player_names=(),
        negotiation_pairs=None,
        player_ids=(),
        max_rounds=max_rounds if max_rounds > 0 else None,
        allow_empty_players=True,
    )
    self._offer_tracker = hdb_negotiation_helpers.ActiveOfferTracker(self._scheduler)

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

    self._scheduler = hdb_negotiation_helpers.NegotiationScheduler(
        player_names=self._player_names,
        negotiation_pairs=negotiation_pairs,
        player_ids=self._player_ids,
        max_rounds=max_rounds if max_rounds > 0 else None,
    )
    self._offer_tracker = hdb_negotiation_helpers.ActiveOfferTracker(self._scheduler)
    self.set_canonical_entities(entities)
    if not self._enabled:
      return
    self._ensure_entities_bound()

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
      spec = {'id': player_id, **dict(raw_spec)}
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
            player_id,
        )
        continue

      try:
        validated_spec = schema_model.model_validate(spec)
      except ValidationError as error:
        logging.error(
            'Participant spec for %s failed %s validation: %s',
            player_id,
            schema_model.__name__,
            error,
        )
        continue

      normalized[player_id] = validated_spec.model_dump(mode='json')
    return normalized

  # Entity and pair binding
  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    self._bind_known_entities()

  def set_canonical_entities(
      self,
      entities: Sequence[entity_component.EntityWithComponents],
  ) -> None:
    """Registers the simulation-owned entities available to negotiation."""
    self._canonical_entities = tuple(entities)
    self._canonical_entities_by_name = {
        entity.name: entity for entity in self._canonical_entities
    }
    self._bind_known_entities()

  def _bind_known_entities(self) -> None:
    """Binds canonical entities to participant ids using id first, then name."""
    if not self._participant_specs or not self._canonical_entities:
      return
    for player_id in self._player_ids:
      self._bind_entity_for_player(player_id)

  def _bind_entity_for_player(self, player_id: str) -> bool:
    """Resolves a participant id to its canonical simulation entity."""
    if player_id in self._entities_by_id:
      return True
    spec = self._participant_specs.get(player_id)
    if spec is None:
      logging.error('No participant spec found for player id: %s', player_id)
      return False

    for entity in self._canonical_entities:
      entity_player_id = str(getattr(entity, '_hdb_player_id', '')).strip()
      if entity_player_id == player_id:
        self._entities_by_id[player_id] = entity
        return True

    expected_name = str(spec.get('name', '')).strip()
    if expected_name:
      matching_entity = self._canonical_entities_by_name.get(expected_name)
      if matching_entity is not None:
        current_player_id = str(getattr(matching_entity, '_hdb_player_id', '')).strip()
        if current_player_id and current_player_id != player_id:
          logging.error(
              'Canonical entity %s is already bound to %s, cannot bind to %s.',
              matching_entity.name,
              current_player_id,
              player_id,
          )
          return False
        matching_entity._hdb_player_id = player_id
        self._entities_by_id[player_id] = matching_entity
        return True

    logging.error(
        'Unable to bind canonical negotiation entity for %s (%s).',
        player_id,
        expected_name or 'unnamed participant',
    )
    return False

  def _bind_entity(self, player_id: str) -> bool:
    """Binds the canonical simulation entity for a participant id on first use."""
    return self._bind_entity_for_player(player_id)

  def _pair_exists(self, buyer_id: str, seller_id: str) -> bool:
    pair_queue = self._scheduler.get_state().get('pair_queue', [])
    normalized = (buyer_id, seller_id)
    return any(
        tuple(str(token) for token in pair) == normalized for pair in pair_queue
    )

  def _register_pair(self, buyer_id: str, seller_id: str) -> None:
    """Registers a new pair with the scheduler and offer tracker."""
    if not self._pair_exists(buyer_id, seller_id):
      self._scheduler.append_pair(buyer_id, seller_id)
    self._offer_tracker.register_pair(buyer_id, seller_id)

  def _parse_listing_transfer_payload(
      self,
      pair_payload: Mapping[str, Any] | negotiation_schemas.ListingNegotiationTransferPayload,
  ) -> negotiation_schemas.ListingNegotiationTransferPayload | None:
    if isinstance(pair_payload, negotiation_schemas.ListingNegotiationTransferPayload):
      return pair_payload
    if not isinstance(pair_payload, Mapping):
      return None
    transfer_payload_keys = {
        'match_id',
        'week_matched',
        'listing_record',
        'buyer_state',
        'seller_state',
    }
    if not transfer_payload_keys.issubset(pair_payload.keys()):
      return None
    try:
      return negotiation_schemas.ListingNegotiationTransferPayload.model_validate(
          pair_payload
      )
    except ValidationError as error:
      logging.warning('Skipping invalid listing transfer payload: %s', error)
      return None

  def _apply_listing_transfer_payload(
      self,
      pair_payload: negotiation_schemas.ListingNegotiationTransferPayload,
      *,
      buyer_id: str,
      seller_id: str,
  ) -> None:
    """Applies listing-to-negotiation context once per pair."""
    listing_record = pair_payload.listing_record
    pair_key = hdb_negotiation_helpers.pair_key(buyer_id, seller_id)
    self._pair_start_weeks.setdefault(pair_key, int(pair_payload.week_matched))
    buyer_name = pair_payload.buyer_state.name
    seller_name = pair_payload.seller_state.name
    buyer_observation = (
        f"{buyer_name}, you submitted a negotiation request for {seller_name}'s flat, "
        f"and {seller_name} accepted it. You are now negotiating directly as the buyer.\n\n"
        f"{listing_record.listing_summary}"
    )
    seller_observation = (
        f"{seller_name}, you accepted {buyer_name}'s negotiation request for your flat. "
        f"You are now negotiating directly as the seller."
    )
    buyer_entity = self._entities_by_id.get(buyer_id)
    if buyer_entity is not None:
      update_agent_from_listing(buyer_entity, pair_payload)
      buyer_entity.observe(buyer_observation)
    seller_entity = self._entities_by_id.get(seller_id)
    if seller_entity is not None:
      update_agent_from_listing(seller_entity, pair_payload)
      seller_entity.observe(seller_observation)

  def _bind_entities_for_pairs(
      self,
      new_negotiation_pairs: Sequence[
          Mapping[str, Any] | negotiation_schemas.ListingNegotiationTransferPayload
      ],
  ) -> list[tuple[str, str]]:
    """Binds both participants for each valid negotiation pair."""
    normalized_pairs: list[tuple[str, str]] = []
    for pair in new_negotiation_pairs:
      transfer_payload = self._parse_listing_transfer_payload(pair)
      if transfer_payload is not None:
        buyer_id = str(transfer_payload.buyer_state.id)
        seller_id = str(transfer_payload.seller_state.id)
      else:
        buyer_id, seller_id = hdb_negotiation_helpers.normalize_negotiation_pair(pair)
      if not buyer_id or not seller_id:
        continue
      if buyer_id not in self._participant_specs or seller_id not in self._participant_specs:
        logging.warning(
            'Skipping negotiation pair with unknown participant ids: %s',
            (buyer_id, seller_id),
        )
        continue
      buyer_bound = self._bind_entity(buyer_id)
      seller_bound = self._bind_entity(seller_id)
      if not buyer_bound or not seller_bound:
        logging.warning(
            'Skipping negotiation pair due to missing canonical entity binding: %s',
            (buyer_id, seller_id),
        )
        continue
      pair_already_exists = self._pair_exists(buyer_id, seller_id)
      self._register_pair(buyer_id, seller_id)
      if not pair_already_exists and transfer_payload is not None:
        self._apply_listing_transfer_payload(
            transfer_payload,
            buyer_id=buyer_id,
            seller_id=seller_id,
        )
      normalized_pairs.append((buyer_id, seller_id))
    return normalized_pairs

  def _ensure_entities_bound(self) -> None:
    pair_queue = self._scheduler.get_state().get('pair_queue', [])
    self._bind_entities_for_pairs(
        hdb_negotiation_helpers.pair_mappings_from_pair_ids(pair_queue)
    )

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

  # Logging Methods
  @staticmethod
  def _has_meaningful_log_value(value: Any) -> bool:
    """Returns whether a log payload contains any non-empty values."""
    if value is None:
      return False
    if isinstance(value, str):
      return bool(value.strip())
    if isinstance(value, Mapping):
      return any(
          NegotiationModule._has_meaningful_log_value(item)
          for item in value.values()
      )
    if isinstance(value, Sequence) and not isinstance(value, str):
      return any(
          NegotiationModule._has_meaningful_log_value(item)
          for item in value
      )
    return True

  def get_entity_log_snapshots(
      self,
      player_ids: Sequence[str] | None = None,
  ) -> dict[str, dict[str, Any]]:
    """Returns the latest internal negotiator logs keyed by display name."""
    requested_ids = (
        tuple(str(player_id) for player_id in player_ids)
        if player_ids is not None
        else tuple(self._entities_by_id.keys())
    )
    snapshots: dict[str, dict[str, Any]] = {}
    for player_id in requested_ids:
      entity = self._entities_by_id.get(player_id)
      if entity is None or not hasattr(entity, 'get_last_log'):
        continue
      snapshot = entity.get_last_log()
      if not isinstance(snapshot, Mapping):
        continue
      normalized_snapshot = dict(snapshot)
      if not self._has_meaningful_log_value(normalized_snapshot):
        continue
      snapshots[self._get_player_name(player_id)] = normalized_snapshot
    return snapshots

  def get_entity_memories(
      self,
      player_ids: Sequence[str] | None = None,
  ) -> dict[str, list[str]]:
    """Returns internal negotiator memories keyed by display name."""
    requested_ids = (
        tuple(str(player_id) for player_id in player_ids)
        if player_ids is not None
        else tuple(self._entities_by_id.keys())
    )
    memories: dict[str, list[str]] = {}
    for player_id in requested_ids:
      entity = self._entities_by_id.get(player_id)
      if entity is None:
        continue
      try:
        entity_memory = entity.get_component(
            memory_component.DEFAULT_MEMORY_COMPONENT_KEY
        )
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      if (
          entity_memory is None
          or not hasattr(entity_memory, 'get_all_memories_as_text')
      ):
        continue
      memory_text = list(entity_memory.get_all_memories_as_text())
      if memory_text:
        memories[self._get_player_name(player_id)] = memory_text
    return memories

  def get_pair_state_snapshots(
      self,
      pair_ids: Sequence[Sequence[str]] | None = None,
  ) -> list[dict[str, Any]]:
    """Returns current pair-level negotiation state for weekly HTML logging."""
    self._offer_tracker._ensure_initialized()

    requested_pairs = (
        [
            (str(pair[0]), str(pair[1]))
            for pair in pair_ids
            if len(pair) == 2
        ]
        if pair_ids is not None
        else self._scheduler.get_pair_queue_ids()
    )

    snapshots: list[dict[str, Any]] = []
    for buyer_id, seller_id in requested_pairs:
      pair_key = hdb_negotiation_helpers.pair_key(buyer_id, seller_id)
      if pair_key not in self._offer_tracker._pair_members:
        continue

      buyer_entity = self._entities_by_id.get(buyer_id)
      seller_entity = self._entities_by_id.get(seller_id)
      buyer_log = (
          dict(buyer_entity.get_last_log())
          if buyer_entity is not None and hasattr(buyer_entity, 'get_last_log')
          and isinstance(buyer_entity.get_last_log(), Mapping)
          else {}
      )
      seller_log = (
          dict(seller_entity.get_last_log())
          if seller_entity is not None and hasattr(seller_entity, 'get_last_log')
          and isinstance(seller_entity.get_last_log(), Mapping)
          else {}
      )

      snapshots.append({
          'pair_key': pair_key,
          'buyer_id': buyer_id,
          'buyer_name': self._get_player_name(buyer_id),
          'seller_id': seller_id,
          'seller_name': self._get_player_name(seller_id),
          'pair_round_number': self._scheduler.get_pair_round_number(
              buyer_id,
              seller_id,
          ),
          'closed': pair_key in self._offer_tracker._closed_pairs,
          'outcome': self._offer_tracker._closed_pair_outcomes.get(
              pair_key,
              negotiation_schemas.NegotiationOutcome.CLOSED,
          ).value
          if pair_key in self._offer_tracker._closed_pairs
          else 'OPEN',
          'has_active_offer': self._offer_tracker._active_offers.get(pair_key)
          is not None,
          'active_offer': self._offer_tracker._active_offers.get(pair_key),
          'offer_history': list(
              self._offer_tracker.get_offer_history_for_pair(buyer_id, seller_id)
          ),
          'turn_count': int(self._offer_tracker._turn_counts.get(pair_key, 0)),
          'buyer_log': buyer_log,
          'seller_log': seller_log,
      })
    return snapshots

  def _effective_reservation_distribution_for_listing(
      self,
      player_id: str,
      component_name: str,
  ) -> Any | None:
    entity = self._entities_by_id.get(player_id)
    if entity is None:
      return None
    try:
      uncertainty_component = entity.get_component(component_name)
    except Exception:  # pylint: disable=broad-exception-caught
      return None
    getter = getattr(
        uncertainty_component,
        'get_effective_reservation_distribution',
        None,
    )
    if not callable(getter):
      return None
    return getter()

  def build_relisting_transfer_payloads(
      self,
      pair_records: Sequence[Mapping[str, Any]],
      *,
      week_number: int,
  ) -> list[dict[str, Any]]:
    """Builds negotiation-to-listing return payloads."""
    payloads: list[dict[str, Any]] = []
    for pair_record in pair_records:
      buyer_id = str(pair_record.get('buyer_id', '')).strip()
      seller_id = str(pair_record.get('seller_id', '')).strip()
      if not buyer_id or not seller_id:
        continue
      pair_key = hdb_negotiation_helpers.pair_key(buyer_id, seller_id)
      buyer_effective_reservation = (
          self._effective_reservation_distribution_for_listing(
              buyer_id,
              'uncertain_buyer',
          )
      )
      seller_effective_reservation = (
          self._effective_reservation_distribution_for_listing(
              seller_id,
              'uncertain_seller',
          )
      )
      if (
          buyer_effective_reservation is None
          or seller_effective_reservation is None
      ):
        logging.warning(
            'Skipping negotiation-to-listing payload for %s because reservation state is unavailable.',
            pair_key,
        )
        continue
      payload = negotiation_schemas.NegotiationToListingPayload(
          negotiation_history=negotiation_schemas.NegotiationHistoryRecord(
              buyer_id=buyer_id,
              seller_id=seller_id,
              start_week=int(self._pair_start_weeks.get(pair_key, week_number)),
              end_week=int(week_number),
              offer_history=[
                  negotiation_schemas.OfferHistory.model_validate(offer)
                  for offer in self._offer_tracker.get_offer_history_for_pair(
                      buyer_id,
                      seller_id,
                  )
              ],
          ),
          buyer_state=negotiation_schemas.NegotiationBuyerHandOffPayload(
              buyer_id=buyer_id,
              effective_reservation=buyer_effective_reservation,
          ),
          seller_state=negotiation_schemas.NegotiationSellerHandOffPayload(
              seller_id=seller_id,
              effective_reservation=seller_effective_reservation,
          ),
      )
      payloads.append(payload.model_dump(mode='json'))
    return payloads

  # Observation helpers
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
    action.pop('decision_rationale', None)
    sanitized_json = json.dumps(action, ensure_ascii=False)
    start = payload.find(payload_json)
    if start < 0:
      return event
    end = start + len(payload_json)
    return f'{actor}{sep}{payload[:start]}{sanitized_json}{payload[end:]}'

  def _observe_event(self, observer_id: str, event: str) -> None:
    """Delivers a sanitized event directly to one negotiating entity by id."""
    entity = self._entities_by_id.get(observer_id)
    if entity is None:
      logging.warning(
          'Unable to deliver negotiation observation to %s (%s).',
          self._get_player_name(observer_id),
          observer_id,
      )
      return
    entity.observe(self._sanitize_event_for_counterparty(event))

  def _advance_pair_round_for_entity(self, player_id: str) -> None:
    """Advance pair-local elapsed time once for the entity after a full week."""
    entity = self._entities_by_id.get(player_id)
    if entity is None:
      return
    try:
      strategy = entity.get_component('NegotiationStrategy')
    except Exception:  # pylint: disable=broad-exception-caught
      return
    advance_pair_round = getattr(strategy, 'advance_pair_round', None)
    if callable(advance_pair_round):
      advance_pair_round()

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

  @staticmethod
  def _extract_public_action_from_event(event: str) -> dict[str, Any] | None:
    """Parses the action payload while stripping private reasoning fields."""
    _, sep, payload = event.partition(':')
    if not sep:
      return None
    payload_json = hdb_negotiation_helpers.ActiveOfferTracker._extract_json_object(
        payload
    )
    if not payload_json:
      return None
    try:
      action = json.loads(payload_json)
    except json.JSONDecodeError:
      return None
    if not isinstance(action, dict):
      return None
    action.pop('internal_reasoning', None)
    action.pop('decision_rationale', None)
    return action

  def _ensure_pair_replay_record(
      self,
      *,
      pair_key: str,
      buyer_id: str,
      seller_id: str,
      week_number: int,
  ) -> dict[str, Any]:
    """Creates the replay record for a pair if it does not exist yet."""
    record = self._conversation_replays.get(pair_key)
    if record is None:
      record = {
          'pair_key': pair_key,
          'buyer_id': buyer_id,
          'buyer_name': self._get_player_name(buyer_id),
          'seller_id': seller_id,
          'seller_name': self._get_player_name(seller_id),
          'start_week': int(self._pair_start_weeks.get(pair_key, week_number)),
          'end_week': None,
          'closed': False,
          'outcome': 'OPEN',
          'events': [],
      }
      self._conversation_replays[pair_key] = record
    return record

  def _append_pair_replay_events(
      self,
      *,
      pair_key: str,
      buyer_id: str,
      seller_id: str,
      week_number: int,
      pair_round_number: int,
      pair_events: Sequence[Mapping[str, str]],
  ) -> None:
    """Appends sanitized public events for frontend replay."""
    if not pair_events:
      return
    record = self._ensure_pair_replay_record(
        pair_key=pair_key,
        buyer_id=buyer_id,
        seller_id=seller_id,
        week_number=week_number,
    )
    replay_events = record.setdefault('events', [])
    for pair_event in pair_events:
      actor_id = str(pair_event.get('actor_id', ''))
      raw_event = str(pair_event.get('event', ''))
      public_event = self._sanitize_event_for_counterparty(raw_event)
      replay_events.append({
          'sequence': len(replay_events) + 1,
          'week_number': int(week_number),
          'pair_round_number': int(pair_round_number),
          'actor_id': actor_id,
          'actor_name': self._get_player_name(actor_id),
          'actor_role': (
              negotiation_schemas.RoleType.BUYER.value
              if actor_id == buyer_id
              else negotiation_schemas.RoleType.SELLER.value
          ),
          'event': public_event,
          'action': self._extract_public_action_from_event(public_event),
      })

  def _update_pair_replay_outcome(
      self,
      *,
      pair_key: str,
      buyer_id: str,
      seller_id: str,
      week_number: int,
  ) -> None:
    """Keeps replay metadata aligned with the latest pair status."""
    record = self._ensure_pair_replay_record(
        pair_key=pair_key,
        buyer_id=buyer_id,
        seller_id=seller_id,
        week_number=week_number,
    )
    is_closed = pair_key in self._offer_tracker._closed_pairs
    record['closed'] = is_closed
    record['outcome'] = (
        self._offer_tracker._closed_pair_outcomes.get(
            pair_key,
            negotiation_schemas.NegotiationOutcome.CLOSED,
        ).value
        if is_closed
        else 'OPEN'
    )
    record['end_week'] = int(week_number) if is_closed else None

  # Pair execution
  def _execute_player_turn(
      self,
      player_id: str,
      *,
      buyer_id: str,
      seller_id: str,
      has_active_offer: bool,
  ) -> tuple[str | None, bool]:
    """Runs one player turn and returns `(event, should_force_close_pair)`."""
    action_spec = self._build_action_spec_for_pair_state(
        player_id,
        buyer_id=buyer_id,
        seller_id=seller_id,
        has_active_offer=has_active_offer,
    )
    if action_spec.output_type == entity_lib.OutputType.SKIP_THIS_STEP:
      return None, False

    entity = self._entities_by_id.get(player_id)
    if entity is None:
      logging.error(
          'No canonical negotiation entity bound for %s (%s). Closing pair and continuing.',
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
      buyer_id: str,
      seller_id: str,
      has_active_offer: bool,
  ) -> entity_lib.ActionSpec:
    """Builds the action spec from pair-local offer state."""
    role = (
        negotiation_schemas.RoleType.BUYER
        if buyer_id == player_id
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
        buyer_id=buyer_id,
        seller_id=seller_id,
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
        self._observe_event(seller_id, buyer_event)

    if not local_is_closed and not force_close:
      seller_event, should_close_pair = self._execute_player_turn(
          seller_id,
          buyer_id=buyer_id,
          seller_id=seller_id,
          has_active_offer=local_has_active_offer,
      )
      force_close = force_close or should_close_pair
      if seller_event is not None:
        pair_events.append({'actor_id': seller_id, 'event': seller_event})
        self._observe_event(buyer_id, seller_event)

    if pair_events:
      self._advance_pair_round_for_entity(buyer_id)
      self._advance_pair_round_for_entity(seller_id)

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

    self._ensure_entities_bound()

    if new_negotiation_pairs:
      self._bind_entities_for_pairs(new_negotiation_pairs)

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
            has_active_offer=self._offer_tracker.has_active_offer_for_pair(
                buyer_id,
                seller_id,
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
        self._update_pair_replay_outcome(
            pair_key=pair_key,
            buyer_id=buyer_id,
            seller_id=seller_id,
            week_number=week_number,
        )
        continue
      pair_events = pair_result.get('events', [])
      if not pair_events:
        continue
      self._append_pair_replay_events(
          pair_key=pair_key,
          buyer_id=buyer_id,
          seller_id=seller_id,
          week_number=week_number,
          pair_round_number=self._scheduler.get_pair_round_number(
              buyer_id,
              seller_id,
          ),
          pair_events=pair_events,
      )
      number_of_pairs_negotiated += 1
      negotiated_pairs.append((buyer_id, seller_id))
      for pair_event in pair_events:
        actor_id = str(pair_event['actor_id'])
        event = str(pair_event['event'])
        actor_role = (
            negotiation_schemas.RoleType.BUYER
            if actor_id == buyer_id
            else negotiation_schemas.RoleType.SELLER
        )
        self._offer_tracker.record_pair_event(
            pair_key,
            actor_role=actor_role,
            event=event,
            week_number=week_number,
        )
        events.append(event)
      self._update_pair_replay_outcome(
          pair_key=pair_key,
          buyer_id=buyer_id,
          seller_id=seller_id,
          week_number=week_number,
      )

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
        'bound_entities': sorted(self._entities_by_id.keys()),
        'canonical_entity_names': sorted(self._canonical_entities_by_name.keys()),
        'scheduler_state': self._scheduler.get_state(),
        'offer_state': self._offer_tracker.get_state(),
    }
    return json.dumps(snapshot)

  def get_state(self) -> entity_component.ComponentState:
    """Serializes module-owned negotiation state only.

    Canonical entity state is owned and checkpointed by the simulation layer,
    not by this module.
    """
    return {
        'participant_specs': self._participant_specs,
        'scheduler_state': self._scheduler.get_state(),
        'offer_state': self._offer_tracker.get_state(),
        'pair_start_weeks': dict(self._pair_start_weeks),
        'conversation_replays': {
            key: dict(value, events=list(value.get('events', ())))
            for key, value in self._conversation_replays.items()
        },
        'action_prompt': self._action_prompt,
        'make_observation_component_key': self._make_observation_component_key,
        'enabled': int(self._enabled),
    }

  def get_dynamic_state(self) -> entity_component.ComponentState:
    return {
        'action_prompt': self._action_prompt,
        'enabled': int(self._enabled),
    }

  def get_conversation_replay_records(self) -> list[dict[str, Any]]:
    """Returns sanitized per-pair transcripts for external replay consumers."""
    records: list[dict[str, Any]] = []
    for pair_key in sorted(self._conversation_replays):
      value = self._conversation_replays[pair_key]
      records.append(dict(value, events=list(value.get('events', ()))))
    return records

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores module state and re-binds canonical entities for tracked pairs."""
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
    if 'pair_start_weeks' in state:
      self._pair_start_weeks = {
          str(key): int(value)
          for key, value in state['pair_start_weeks'].items()
      }
    if 'conversation_replays' in state:
      restored_replays: dict[str, dict[str, Any]] = {}
      for key, value in state['conversation_replays'].items():
        if not isinstance(value, Mapping):
          continue
        restored_replays[str(key)] = dict(
            value,
            events=list(value.get('events', ())),
        )
      self._conversation_replays = restored_replays
    else:
      self._conversation_replays = {}
    if 'action_prompt' in state:
      self._action_prompt = str(state['action_prompt'])
    if 'make_observation_component_key' in state:
      self._make_observation_component_key = str(
          state['make_observation_component_key']
      )
    self._enabled = bool(state.get('enabled', 1))
    self._entities_by_id = {}
    self._bind_known_entities()
    self._ensure_entities_bound()
