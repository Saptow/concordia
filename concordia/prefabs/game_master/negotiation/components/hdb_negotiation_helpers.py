"""Helper classes for HDB negotiation scheduling and pair offer state."""

from collections.abc import Mapping, Sequence
import json

from absl import logging
from concordia.hdb_simulation.models import schemas as hdb_schemas
from concordia.typing import entity_component

# Shared pair types
PairIds = tuple[str, str]
PairMapping = Mapping[str, str]
SerializedPairMapping = dict[str, str]


# Pair normalization helpers
def normalize_negotiation_pair(pair: PairMapping) -> PairIds:
  """Normalizes a negotiation pair mapping with explicit buyer/seller ids."""
  buyer_id = str(pair.get('buyer_id', '')).strip()
  seller_id = str(pair.get('seller_id', '')).strip()
  if not buyer_id or not seller_id:
    logging.warning('Malformed negotiation pair received: %s', pair)
    return '', ''
  return buyer_id, seller_id


def pair_mapping_from_ids(pair: Sequence[str]) -> SerializedPairMapping | None:
  """Converts an internal 2-item pair into the standard buyer/seller mapping."""
  if len(pair) != 2:
    logging.warning('Skipping pair with invalid length during restore: %s', pair)
    return None
  buyer_id = str(pair[0]).strip()
  seller_id = str(pair[1]).strip()
  if not buyer_id or not seller_id:
    logging.warning('Skipping pair with empty ids during restore: %s', pair)
    return None
  return {
      'buyer_id': buyer_id,
      'seller_id': seller_id,
  }


def pair_mappings_from_pair_ids(
    negotiation_pair_ids: Sequence[Sequence[str]],
) -> list[SerializedPairMapping]:
  """Converts restored scheduler pairs into the standard mapping format."""
  normalized_pairs: list[SerializedPairMapping] = []
  for pair in negotiation_pair_ids:
    pair_mapping = pair_mapping_from_ids(pair)
    if pair_mapping is not None:
      normalized_pairs.append(pair_mapping)
  return normalized_pairs


def pair_key(buyer_id: str, seller_id: str) -> str:
  """Builds the canonical tracker key for one buyer/seller pair."""
  return f'{buyer_id}|||{seller_id}'


def parse_outcome(value: object) -> hdb_schemas.NegotiationOutcome | None:
  """Parses a serialized negotiation outcome enum value."""
  try:
    return hdb_schemas.NegotiationOutcome(str(value))
  except ValueError:
    logging.warning('Skipping invalid negotiation outcome value: %s', value)
    return None


class NegotiationScheduler:
  """Tracks pair-level negotiation progress across weekly rounds."""

  def __init__(
      self,
      *,
      player_names: Sequence[str],
      negotiation_pairs: Sequence[Sequence[str]] | None = None,
      player_ids: Sequence[str] | None = None,
      max_rounds: int | None = None,
      allow_empty_players: bool = False,
  ):
    if not player_names:
      if not allow_empty_players:
        logging.error('No player names provided to the HDB negotiation scheduler.')
      player_names = ()

    self._max_rounds = max_rounds if max_rounds and max_rounds > 0 else None
    self._player_ids = tuple(player_ids) if player_ids else tuple(player_names)
    if len(self._player_ids) != len(player_names):
      logging.error('Negotiation scheduler player_ids do not align with names.')
      self._player_ids = tuple(player_names)
    if len(set(self._player_ids)) != len(self._player_ids):
      logging.error('Negotiation scheduler player_ids are not unique.')
      deduped_pairs: list[tuple[str, str]] = []
      seen_ids: set[str] = set()
      for player_id, player_name in zip(self._player_ids, player_names, strict=False):
        normalized_id = str(player_id)
        if normalized_id in seen_ids:
          logging.warning(
              'Dropping duplicate negotiation scheduler player id: %s',
              normalized_id,
          )
          continue
        seen_ids.add(normalized_id)
        deduped_pairs.append((normalized_id, str(player_name)))
      self._player_ids = tuple(player_id for player_id, _ in deduped_pairs)
      player_names = tuple(player_name for _, player_name in deduped_pairs)

    self._id_to_name = {
        str(player_id): str(player_name)
        for player_id, player_name in zip(
            self._player_ids, player_names, strict=False
        )
    }
    self._pair_queue = self._build_pair_queue(negotiation_pairs)
    self._player_id_to_pair_index: dict[str, int] = {}
    self._rebuild_player_index()
    self._pair_round_numbers = [1 for _ in self._pair_queue]
    self._closed_pair_indices: set[int] = set()
    self._global_round_number = 1

  # Pair registry
  def _parse_pair(self, pair: Sequence[str]) -> PairIds:
    """Validates an internal 2-item pair and resolves it to canonical ids."""
    pair_mapping = pair_mapping_from_ids(pair)
    if pair_mapping is None:
      return '', ''
    buyer_id, seller_id = normalize_negotiation_pair(pair_mapping)
    if not buyer_id or not seller_id:
      return '', ''
    if buyer_id not in self._id_to_name or seller_id not in self._id_to_name:
      logging.warning(
          'Skipping negotiation pair with unknown participant ids: %s',
          (buyer_id, seller_id),
      )
      return '', ''
    return buyer_id, seller_id

  def _build_pair_queue(
      self,
      negotiation_pairs: Sequence[Sequence[str]] | None,
  ) -> list[PairIds]:
    """Builds the initial pair queue, defaulting to adjacent buyer/seller ids."""
    if negotiation_pairs:
      pairs: list[PairIds] = []
      for pair in negotiation_pairs:
        parsed_pair = self._parse_pair(pair)
        if not all(parsed_pair):
          logging.warning('Skipping invalid negotiation pair %s', pair)
          continue
        pairs.append(parsed_pair)
      if not pairs:
        logging.warning('No valid explicit negotiation pairs were provided.')
      return pairs

    if len(self._player_ids) % 2 != 0:
      logging.warning(
          'Automatic negotiation pair creation has an odd number of players; '
          'the last player will be left unmatched.'
      )
    return [
        (self._player_ids[i], self._player_ids[i + 1])
        for i in range(0, len(self._player_ids) - 1, 2)
    ]

  def _pair_index_for_player_id(self, player_id: str) -> int | None:
    return self._player_id_to_pair_index.get(player_id)

  def _pair_index_for_pair(
      self,
      buyer_id: str,
      seller_id: str,
  ) -> int | None:
    """Returns the stored queue index for a pair, if present."""
    candidate = (str(buyer_id), str(seller_id))
    for index, pair in enumerate(self._pair_queue):
      if pair == candidate:
        return index
    return None

  def _rebuild_player_index(self) -> None:
    """Rebuilds the reverse lookup from player id to pair index."""
    self._player_id_to_pair_index = {}
    for idx, pair in enumerate(self._pair_queue):
      first, second = pair
      self._player_id_to_pair_index[first] = idx
      self._player_id_to_pair_index[second] = idx

  def close_pair_for_player(self, player_id: str) -> None:
    """Closes the pair that contains the given participant id."""
    player_id = str(player_id)
    if not player_id:
      return
    pair_index = self._pair_index_for_player_id(player_id)
    if pair_index is None:
      return
    self._closed_pair_indices.add(pair_index)

  def close_pair(self, buyer_id: str, seller_id: str) -> None:
    """Closes a pair directly by buyer/seller ids."""
    pair_index = self._pair_index_for_pair(buyer_id, seller_id)
    if pair_index is None:
      return
    self._closed_pair_indices.add(pair_index)

  def append_pair(self, buyer_id: str, seller_id: str) -> None:
    """Adds a new pair to the scheduler if it is not already tracked."""
    pair = (str(buyer_id), str(seller_id))
    if pair in self._pair_queue:
      return
    self._pair_queue.append(pair)
    self._pair_round_numbers.append(1)
    self._rebuild_player_index()

  def all_pairs_closed(self) -> bool:
    return not self._pair_queue or len(self._closed_pair_indices) >= len(
        self._pair_queue
    )

  # Weekly progress tracking
  def is_pair_closed(self, buyer_id: str, seller_id: str) -> bool:
    """Returns whether a pair is already closed."""
    pair_index = self._pair_index_for_pair(buyer_id, seller_id)
    if pair_index is None:
      return True
    return pair_index in self._closed_pair_indices

  def get_open_pair_queue_ids(self) -> list[PairIds]:
    """Returns open pairs in deterministic queue order."""
    return [
        pair
        for index, pair in enumerate(self._pair_queue)
        if index not in self._closed_pair_indices
    ]

  def get_pair_round_number(self, buyer_id: str, seller_id: str) -> int:
    """Returns the current pair-local round number."""
    pair_index = self._pair_index_for_pair(buyer_id, seller_id)
    if pair_index is None:
      return 0
    return self._pair_round_numbers[pair_index]

  def advance_week(self, negotiated_pairs: Sequence[tuple[str, str]]) -> None:
    """Advances scheduler state after one weekly pass across all open pairs."""
    negotiated_pair_indices = {
        pair_index
        for buyer_id, seller_id in negotiated_pairs
        if (
            pair_index := self._pair_index_for_pair(buyer_id, seller_id)
        ) is not None
        and pair_index not in self._closed_pair_indices
    }
    for pair_index in negotiated_pair_indices:
      self._pair_round_numbers[pair_index] += 1
    self._global_round_number += 1
    if self._max_rounds is not None and self._global_round_number > self._max_rounds:
      self._closed_pair_indices = set(range(len(self._pair_queue)))

  # Display helpers
  def get_pair_queue_ids(self) -> list[PairIds]:
    return list(self._pair_queue)

  def get_pair_queue_names(self) -> list[tuple[str, str]]:
    """Returns tracked pairs using display names for logging/debugging."""
    return [
        (
            self._id_to_name.get(first, first),
            self._id_to_name.get(second, second),
        )
        for first, second in self._pair_queue
    ]

  def get_player_name(self, player_id: str) -> str:
    return self._id_to_name.get(player_id, player_id)

  def set_player_names(self, player_names_by_id: Mapping[str, str]) -> None:
    """Refreshes display names without changing canonical participant ids."""
    self._id_to_name = {
        player_id: str(
            player_names_by_id.get(
                player_id, self._id_to_name.get(player_id, player_id)
            )
        )
        for player_id in self._player_ids
    }

  def get_scheduler_snapshot(self) -> dict[str, object]:
    """Builds a human-readable snapshot of scheduler progress."""
    max_rounds = self._max_rounds if self._max_rounds is not None else 0
    return {
        'global_round_number': self._global_round_number,
        'open_pairs': [
            {
                'buyer_id': buyer_id,
                'buyer_name': self.get_player_name(buyer_id),
                'seller_id': seller_id,
                'seller_name': self.get_player_name(seller_id),
                'pair_round_number': self.get_pair_round_number(buyer_id, seller_id),
            }
            for buyer_id, seller_id in self.get_open_pair_queue_ids()
        ],
        'closed_pair_count': len(self._closed_pair_indices),
        'max_rounds': max_rounds,
    }

  # Serialization
  def get_state(self) -> entity_component.ComponentState:
    """Serializes scheduler queue, per-pair rounds, and closure state."""
    return {
        'global_round_number': self._global_round_number,
        'pair_queue': [list(pair) for pair in self._pair_queue],
        'pair_round_numbers': list(self._pair_round_numbers),
        'closed_pair_indices': sorted(self._closed_pair_indices),
        'player_ids': list(self._player_ids),
        'max_rounds': self._max_rounds if self._max_rounds is not None else 0,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores scheduler queue and weekly progress state."""
    self._global_round_number = int(state.get('global_round_number', 1))

    pair_queue_state = state.get('pair_queue')
    if pair_queue_state is not None:
      restored_pairs = pair_mappings_from_pair_ids(
          pair_queue_state  # type: ignore[arg-type]
      )
      self._pair_queue = [
          normalize_negotiation_pair(pair_mapping) for pair_mapping in restored_pairs
      ]
    self._rebuild_player_index()

    pair_round_numbers_state = state.get('pair_round_numbers')
    if pair_round_numbers_state:
      restored_pair_rounds = [
          int(x) for x in pair_round_numbers_state  # type: ignore[index]
      ]
      expected = len(self._pair_queue)
      if len(restored_pair_rounds) < expected:
        restored_pair_rounds.extend([1] * (expected - len(restored_pair_rounds)))
      self._pair_round_numbers = restored_pair_rounds[:expected]
    else:
      self._pair_round_numbers = [1 for _ in self._pair_queue]

    closed_pair_indices_state = state.get('closed_pair_indices')
    if closed_pair_indices_state:
      self._closed_pair_indices = {
          int(x) for x in closed_pair_indices_state  # type: ignore[index]
      }
    else:
      self._closed_pair_indices = set()

    max_rounds = int(state.get('max_rounds', 0))
    self._max_rounds = max_rounds if max_rounds > 0 else None


class ActiveOfferTracker:
  """Tracks pair-local offer state and closure outcomes by participant id."""

  def __init__(self, scheduler: NegotiationScheduler):
    self._scheduler = scheduler
    self._player_to_pair: dict[str, str] = {}
    self._pair_members: dict[str, tuple[str, str]] = {}
    self._pair_order: list[str] = []
    self._active_offers: dict[str, dict[str, object] | None] = {}
    self._closed_pairs: set[str] = set()
    self._closed_pair_outcomes: dict[str, hdb_schemas.NegotiationOutcome] = {}

  # Pair-key helpers
  @staticmethod
  def _pair_key(buyer_id: str, seller_id: str) -> str:
    return pair_key(buyer_id, seller_id)

  @staticmethod
  def _extract_json_object(text: str) -> str | None:
    """Extracts the first balanced JSON object embedded in an event string."""
    start = text.find('{')
    if start < 0:
      return None
    candidate = text[start:]
    depth = 0
    for idx, ch in enumerate(candidate):
      if ch == '{':
        depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0:
          return candidate[: idx + 1]
    return None

  # Pair registration and closure
  def _ensure_initialized(self) -> None:
    """Lazily initializes tracker state from the scheduler's pair queue."""
    if self._pair_order:
      return
    for buyer_id, seller_id in self._scheduler.get_pair_queue_ids():
      key = self._pair_key(buyer_id, seller_id)
      self._pair_order.append(key)
      self._pair_members[key] = (buyer_id, seller_id)
      self._player_to_pair[buyer_id] = key
      self._player_to_pair[seller_id] = key
      self._active_offers.setdefault(key, None)

  def register_pair(self, buyer_id: str, seller_id: str) -> None:
    """Registers a newly added pair with empty offer state."""
    self._ensure_initialized()
    current_pair_key = self._pair_key(buyer_id, seller_id)
    if current_pair_key in self._pair_members:
      return
    self._pair_order.append(current_pair_key)
    self._pair_members[current_pair_key] = (buyer_id, seller_id)
    self._player_to_pair[buyer_id] = current_pair_key
    self._player_to_pair[seller_id] = current_pair_key
    self._active_offers[current_pair_key] = None

  def close_pair(
      self,
      buyer_id: str,
      seller_id: str,
      *,
      outcome: hdb_schemas.NegotiationOutcome = hdb_schemas.NegotiationOutcome.CLOSED,
  ) -> None:
    """Marks a pair closed and clears any active offer for it."""
    self._ensure_initialized()
    pair_key = self._pair_key(buyer_id, seller_id)
    if pair_key not in self._pair_members:
      return
    self._active_offers[pair_key] = None
    self._closed_pairs.add(pair_key)
    self._closed_pair_outcomes[pair_key] = outcome
    self._scheduler.close_pair(buyer_id, seller_id)

  # Offer-state transitions
  def _role_for_player(self, player_id: str, pair_key: str) -> hdb_schemas.RoleType:
    """Returns the buyer/seller role for a participant within a pair."""
    buyer_id, seller_id = self._pair_members[pair_key]
    if player_id == buyer_id:
      return hdb_schemas.RoleType.BUYER
    if player_id == seller_id:
      return hdb_schemas.RoleType.SELLER
    logging.error(
        'Player %s (%s) is not in negotiation pair %s.',
        self._scheduler.get_player_name(player_id),
        player_id,
        pair_key,
    )
    return hdb_schemas.RoleType.PLACEHOLDER

  def record_resolved_event(self, event: str, *, actor_id: str) -> None:
    """Applies one resolved event to pair offer state and closure state."""
    self._ensure_initialized()
    _, sep, payload = event.partition(':')
    if not sep:
      return
    pair_key = self._player_to_pair.get(actor_id)
    if not pair_key:
      logging.warning(
          'Unable to resolve negotiation pair for actor %s (%s).',
          self._scheduler.get_player_name(actor_id),
          actor_id,
      )
      return

    payload_json = self._extract_json_object(payload)
    if not payload_json:
      return

    try:
      action = json.loads(payload_json)
    except json.JSONDecodeError:
      return

    action_type = str(action.get('type', '')).strip().upper()
    if not action_type:
      return

    if action_type in ('MAKE_OFFER', 'MAKE_COUNTEROFFER'):
      if pair_key in self._closed_pairs:
        return
      self._active_offers[pair_key] = {
          'offerer_id': actor_id,
          'offerer_name': self._scheduler.get_player_name(actor_id),
          'action_type': action_type,
          'payload': action,
      }
      return

    if action_type == 'REJECT_OFFER':
      self._active_offers[pair_key] = None
      return

    if action_type == 'ACCEPT_OFFER':
      self._active_offers[pair_key] = None
      self._closed_pairs.add(pair_key)
      self._closed_pair_outcomes[pair_key] = hdb_schemas.NegotiationOutcome.SUCCESS
      self._scheduler.close_pair_for_player(actor_id)
      return

    if action_type == 'WALK_AWAY':
      role = self._role_for_player(actor_id, pair_key)
      if role != hdb_schemas.RoleType.BUYER:
        return
      self._active_offers[pair_key] = None
      self._closed_pairs.add(pair_key)
      self._closed_pair_outcomes[pair_key] = (
          hdb_schemas.NegotiationOutcome.CLOSED_WITHOUT_SUCCESS
      )
      self._scheduler.close_pair_for_player(actor_id)

  # Query helpers
  def has_active_offer_for_player(self, player_id: str) -> bool:
    """Returns whether the player's pair currently has an active offer."""
    self._ensure_initialized()
    pair_key = self._player_to_pair.get(player_id)
    if not pair_key:
      return False
    return self._active_offers.get(pair_key) is not None

  def get_action_policy_for_player(self, player_id: str) -> dict[str, object]:
    """Returns the allowed action policy for a player from pair-local state."""
    self._ensure_initialized()
    pair_key = self._player_to_pair.get(player_id)
    if not pair_key:
      return {
          'role': hdb_schemas.RoleType.PLACEHOLDER.value,
          'has_active_offer': False,
          'allowed_action_types': [],
          'pair': '',
      }

    role = self._role_for_player(player_id, pair_key)
    if pair_key in self._closed_pairs:
      buyer_id, seller_id = self._pair_members[pair_key]
      return {
          'role': role.value,
          'has_active_offer': False,
          'allowed_action_types': [],
          'pair': (
              f'{self._scheduler.get_player_name(buyer_id)} '
              f'<-> {self._scheduler.get_player_name(seller_id)}'
          ),
          'closed': True,
          'close_outcome': self._closed_pair_outcomes.get(
              pair_key,
              hdb_schemas.NegotiationOutcome.CLOSED,
          ).value,
      }

    has_active_offer = self._active_offers.get(pair_key) is not None
    allowed_actions = hdb_schemas.get_allowed_action_types(role, has_active_offer)
    buyer_id, seller_id = self._pair_members[pair_key]
    return {
        'role': role.value,
        'has_active_offer': has_active_offer,
        'allowed_action_types': list(allowed_actions),
        'pair': (
            f'{self._scheduler.get_player_name(buyer_id)} '
            f'<-> {self._scheduler.get_player_name(seller_id)}'
        ),
        'closed': False,
    }

  def get_pair_members_for_player(self, player_id: str) -> tuple[str, str] | None:
    """Returns the `(buyer_id, seller_id)` pair for a participant id."""
    self._ensure_initialized()
    pair_key = self._player_to_pair.get(player_id)
    if not pair_key:
      return None
    return self._pair_members.get(pair_key)

  def all_pairs_closed(self) -> bool:
    self._ensure_initialized()
    return not self._pair_order or len(self._closed_pairs) >= len(
        self._pair_order
    )

  # Serialization
  def get_state(self) -> entity_component.ComponentState:
    """Serializes offer, closure, and pair membership state."""
    return {
        'player_to_pair': dict(self._player_to_pair),
        'pair_members': {k: list(v) for k, v in self._pair_members.items()},
        'pair_order': list(self._pair_order),
        'active_offers': dict(self._active_offers),
        'closed_pairs': sorted(self._closed_pairs),
        'closed_pair_outcomes': {
            key: value.value for key, value in self._closed_pair_outcomes.items()
        },
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores offer tracker state from serialized data."""
    if 'player_to_pair' in state:
      self._player_to_pair = dict(state['player_to_pair'])  # type: ignore[arg-type]
    if 'pair_members' in state:
      pair_members = state['pair_members']  # type: ignore[assignment]
      self._pair_members = {
          str(k): (str(v[0]), str(v[1]))  # type: ignore[index]
          for k, v in pair_members.items()  # type: ignore[union-attr]
      }
    if 'pair_order' in state:
      self._pair_order = [str(x) for x in state['pair_order']]  # type: ignore[index]
    if 'active_offers' in state:
      self._active_offers = dict(state['active_offers'])  # type: ignore[arg-type]
    if 'closed_pairs' in state:
      self._closed_pairs = {str(x) for x in state['closed_pairs']}  # type: ignore[index]
    if 'closed_pair_outcomes' in state:
      closed_pair_outcomes: dict[str, hdb_schemas.NegotiationOutcome] = {}
      for key, value in state['closed_pair_outcomes'].items():  # type: ignore[union-attr]
        outcome = parse_outcome(value)
        if outcome is None:
          continue
        closed_pair_outcomes[str(key)] = outcome
      self._closed_pair_outcomes = closed_pair_outcomes
    else:
      self._closed_pair_outcomes = {}
