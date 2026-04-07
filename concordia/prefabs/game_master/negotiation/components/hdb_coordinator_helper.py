from collections.abc import Mapping, Sequence
import json
from typing import TYPE_CHECKING, Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.typing import entity_component

if TYPE_CHECKING:
  from concordia.prefabs.game_master.negotiation.components import hdb_listing
  from concordia.prefabs.game_master.negotiation.components import hdb_negotiation


class WeeklyCoordinator(action_spec_ignored.ActionSpecIgnored):
  """Owns shared weekly state across listing and negotiation modules.

  Responsibilities:
  - track the global week counter
  - keep canonical player id/name mappings for logging and engine use
  - decide which players stay in listing versus negotiation each week
  - stage listing matches as pending negotiation transfers
  - emit a durable week summary for logging, checkpoints, and restore.

  Only cross-module state owned by WeeklyCoordinator.
  """

  def __init__(
      self,
      *,
      player_ids: Sequence[str] = (),
      player_names: Sequence[str] = (),
      listing_component_key: str = 'listing_module',
      negotiation_component_key: str = 'negotiation_module',
      pre_act_label: str = 'Weekly coordinator state',
  ):
    super().__init__(pre_act_label=pre_act_label)
    normalized_player_ids = tuple(str(player_id) for player_id in player_ids)
    normalized_player_names = tuple(str(player_name) for player_name in player_names)
    if (
        normalized_player_ids
        and normalized_player_names
        and len(normalized_player_ids) != len(normalized_player_names)
    ):
      logging.error(
          'WeeklyCoordinator player_ids must align with player_names. '
          'Falling back to names as ids.'
      )
      normalized_player_ids = normalized_player_names
    elif not normalized_player_ids:
      normalized_player_ids = normalized_player_names
    self._player_ids = normalized_player_ids
    self._id_to_name = {
        player_id: normalized_player_names[index]
        if index < len(normalized_player_names)
        else player_id
        for index, player_id in enumerate(self._player_ids)
    }
    self._listing_component_key = listing_component_key
    self._negotiation_component_key = negotiation_component_key
    self._week_number = 1
    # Matches created by listing this week are handed to negotiation next week.
    self._pending_matches: list[dict[str, Any]] = []
    self._module_assignments: dict[str, Any] = {
        'listing': [],
        'negotiation': [],
    }
    self._last_week_summary: dict[str, Any] = {}

  # Module accessors.
  def get_listing_module(self) -> 'hdb_listing.ListingModule':
    from concordia.prefabs.game_master.negotiation.components import hdb_listing

    return self.get_entity().get_component(
        self._listing_component_key,
        type_=hdb_listing.ListingModule,
    )

  def get_negotiation_module(self) -> 'hdb_negotiation.NegotiationModule':
    from concordia.prefabs.game_master.negotiation.components import hdb_negotiation

    return self.get_entity().get_component(
        self._negotiation_component_key,
        type_=hdb_negotiation.NegotiationModule,
    )

  def set_module_enabled(self, module_name: str, enabled: bool) -> None:
    normalized = str(module_name).strip().lower()
    if normalized == 'listing':
      self.get_listing_module().set_enabled(enabled)
      return
    if normalized == 'negotiation':
      self.get_negotiation_module().set_enabled(enabled)
      return
    logging.error('Unknown coordinator module name: %s', module_name)

  def get_registered_player_ids(self) -> tuple[str, ...]:
    return self._player_ids

  def get_player_name(self, player_id: str) -> str:
    normalized_player_id = str(player_id)
    return self._id_to_name.get(normalized_player_id, normalized_player_id)

  # Week preparation helpers.
  def _compute_assignments(
      self,
      new_negotiation_pairs: Sequence[Mapping[str, Any]] = (),
  ) -> tuple[set[str], list[tuple[str, str]]]:
    """Computes this week's listing participants and negotiation pairs.

    Args:
      new_negotiation_pairs: Buyer/seller pairs transferred in from listing,
        typically matches produced during the previous completed week.

    Returns:
      A tuple of:
      - listing player ids that remain open and are not already negotiating, and
      - the full set of open negotiation pairs for the week.
    """
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()

    # Negotiation keeps its currently open pairs and absorbs any new transfers
    # from the previous listing week.
    negotiation_pairs = (
        negotiation.get_open_pairs() if negotiation.is_enabled() else []
    )
    for raw_pair in new_negotiation_pairs:
      buyer_id = str(raw_pair.get('buyer_id', '')).strip()
      seller_id = str(raw_pair.get('seller_id', '')).strip()
      if not buyer_id:
        buyer_state = raw_pair.get('buyer_state', {})
        if isinstance(buyer_state, Mapping):
          buyer_id = str(buyer_state.get('id', '')).strip()
      if not seller_id:
        seller_state = raw_pair.get('seller_state', {})
        if isinstance(seller_state, Mapping):
          seller_id = str(seller_state.get('id', '')).strip()
      if not buyer_id or not seller_id:
        logging.warning('Skipping malformed negotiation transfer pair: %s', raw_pair)
        continue
      candidate = (buyer_id, seller_id)
      if candidate not in negotiation_pairs:
        negotiation_pairs.append(candidate)
    negotiation_ids = {
        player_id
        for pair in negotiation_pairs
        for player_id in pair
    }
    # Listing only receives players who are still open and not already engaged
    # in negotiation this week.
    listing_ids = listing.get_open_player_ids() if listing.is_enabled() else set()
    listing_ids -= negotiation_ids

    self._module_assignments = {
        'listing': sorted(listing_ids),
        'negotiation': [list(pair) for pair in negotiation_pairs],
    }
    return listing_ids, negotiation_pairs


  # Weekly lifecycle methods called by the engine.
  def prepare_week(self) -> dict[str, Any]:
    """Builds the engine-facing context for the next weekly step.

    This method snapshots the coordinator-owned scheduling state before either
    module runs. In particular, any pending listing matches are exposed as
    `new_negotiation_pairs` so negotiation can absorb them during this week.
    """
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()
    current_week = self._week_number

    if listing.is_enabled() and hasattr(listing, 'prepare_weekly_market'):
      listing.prepare_weekly_market(week_number=current_week)

    # Pending matches come from listing output of the previous completed week.
    new_negotiation_pairs = list(self._pending_matches)
    listing_ids, negotiation_pairs = self._compute_assignments(new_negotiation_pairs)

    return {
        'week_number': current_week,
        'assignments': dict(self._module_assignments),
        'listing_player_ids': sorted(listing_ids),
        'new_negotiation_pairs': [dict(pair) for pair in new_negotiation_pairs],
        'open_negotiation_pairs': [list(pair) for pair in negotiation_pairs],
        'listing_enabled': listing.is_enabled(),
        'negotiation_enabled': negotiation.is_enabled(),
    }

  @staticmethod
  def _count_listing_matches(listing_outcome: Any | None) -> int:
    if listing_outcome is None:
      return 0
    matched_pairs = getattr(listing_outcome, 'matched_pairs', ())
    return len(matched_pairs)

  @staticmethod
  def _count_negotiation_pairs(
      negotiation_outcome: Mapping[str, Any] | None,
      key: str,
  ) -> int:
    if negotiation_outcome is None:
      return 0
    value = negotiation_outcome.get(key, ())
    return len(value) if isinstance(value, Sequence) and not isinstance(value, str) else 0

  @staticmethod
  def _sequence_or_empty(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str):
      return list(value)
    return []

  @staticmethod
  def _failed_pairs_to_relist(
      negotiation_outcome: Mapping[str, Any] | None,
  ) -> list[dict[str, Any]]:
    """Extracts failed negotiation pairs that should return to listing."""
    if negotiation_outcome is None:
      return []
    closed_pairs = negotiation_outcome.get('closed_pairs', ())
    if not isinstance(closed_pairs, Sequence) or isinstance(closed_pairs, str):
      return []

    reopened_pairs: list[dict[str, Any]] = []
    for record in closed_pairs:
      if not isinstance(record, Mapping):
        continue
      outcome = str(record.get('outcome', '')).strip().upper()
      if outcome != negotiation_schemas.NegotiationOutcome.CLOSED_WITHOUT_SUCCESS:
        continue
      buyer_id = str(record.get('buyer_id', '')).strip()
      seller_id = str(record.get('seller_id', '')).strip()
      if not buyer_id or not seller_id:
        continue
      reopened_pairs.append({
          **dict(record),
          'buyer_id': buyer_id,
          'seller_id': seller_id,
      })
    return reopened_pairs

  def _format_listing_summary(self, listing_outcome: Any | None) -> list[str]:
    if listing_outcome is None:
      return ['Listing', '  status: disabled']
    matched_pairs = list(getattr(listing_outcome, 'matched_pairs', ()))
    lines = [
        'Listing',
        (
            '  Summary: '
            f'Assigned={len(self._module_assignments.get("listing", ()))} '
            f'Matches={len(matched_pairs)}'
        ),
    ]
    for match in matched_pairs:
      lines.append(
          f'  Match: {match.buyer_name} ({match.buyer_id}) '
          f'<-> {match.seller_name} ({match.seller_id})'
      )
    return lines

  def _format_negotiation_summary(
      self,
      negotiation_outcome: Mapping[str, Any] | None,
  ) -> list[str]:
    if negotiation_outcome is None:
      return ['Negotiation', '  Status: disabled']
    event_lines = self._sequence_or_empty(negotiation_outcome.get('events', ()))
    closed_pairs = self._sequence_or_empty(negotiation_outcome.get('closed_pairs', ()))
    lines = [
        'Negotiation',
        (
            '  Summary: '
            f'{len(self._module_assignments.get("negotiation", ()))} '
            f'Active_pairs={negotiation_outcome.get("number_of_pairs_negotiated", 0)} '
            f'Closed={len(closed_pairs)}'
        )
    ]
    if event_lines:
      lines.append('  Events:')
      for event in event_lines:
        lines.append(f'    - {event}')
    else:
      lines.append('  Events: none')
    if closed_pairs:
      lines.append('  Outcomes:')
      for record in closed_pairs:
        if not isinstance(record, Mapping):
          continue
        lines.append(
            '    - '
            f'{record.get("buyer_name", record.get("buyer_id", "?"))} '
            f'<-> {record.get("seller_name", record.get("seller_id", "?"))}: '
            f'{record.get("outcome", "UNKNOWN")}'
        )
    return lines

  @staticmethod
  def _format_action_event_for_log(event: str) -> dict[str, Any] | str:
    """Converts `[ACTED]` event strings into readable structured payloads."""
    actor, sep, payload = event.partition(':')
    if not sep:
      return event

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
      return event

    if isinstance(parsed, dict):
      ordered: dict[str, Any] = {}
      if 'type' in parsed:
        ordered['type'] = parsed['type']
      for key, value in parsed.items():
        if key != 'type':
          ordered[key] = value
      parsed = ordered

    formatted: dict[str, Any] = {
        'actor': actor,
        'payload': parsed,
    }
    if display_prefix:
      formatted['prefix'] = display_prefix
    return formatted

  def _format_week_summary_for_log(self, summary: Mapping[str, Any]) -> str:
    """Builds a readable JSON block for the completed-week log line."""
    display_summary = dict(summary)
    negotiation = display_summary.get('negotiation')
    if isinstance(negotiation, Mapping):
      display_negotiation = dict(negotiation)
      events = display_negotiation.get('events')
      if isinstance(events, Sequence) and not isinstance(events, str):
        display_negotiation['events'] = [
            self._format_action_event_for_log(str(event))
            for event in events
        ]
      display_summary['negotiation'] = display_negotiation
    return json.dumps(display_summary, ensure_ascii=False, indent=2)

  def _log_week_summary(
      self,
      *,
      summary: Mapping[str, Any],
      week_number: int,
      listing_outcome: Any | None,
      negotiation_outcome: Mapping[str, Any] | None,
  ) -> None:
    successful = self._count_negotiation_pairs(negotiation_outcome, 'successful_pairs')
    failed = self._count_negotiation_pairs(negotiation_outcome, 'failed_pairs')
    lines = [
        '+' + '-' * 54 + '+',
        f'| Week {week_number:<2} Summary{" " * 38}|',
        '+' + '-' * 54 + '+',
        (
            'Overview'
        ),
        (
            '  Listing matches staged for next week='
            f'{len(self._pending_matches)} '
            f'successful negotiations={successful} '
            f'failed negotiations={failed}'
        ),
        *self._format_listing_summary(listing_outcome),
        *self._format_negotiation_summary(negotiation_outcome),
        '',
        f'Completed week {week_number}:',
        self._format_week_summary_for_log(summary),
        '+' + '-' * 54 + '+',
    ]
    rendered = '\n'.join(lines)
    print(rendered, flush=True)
    logging.info('Weekly coordinator summary:\n%s', rendered)

  def complete_week(
      self,
      *,
      listing_outcome: Any | None = None,
      negotiation_outcome: Mapping[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Finalizes the current week and stages cross-module handoff state.

    Listing output is converted into `pending_matches` for the next week, while
    negotiation output is recorded only for summary/logging purposes.

    Args:
      listing_outcome: Result returned by `ListingModule.run_week`.
      negotiation_outcome: Result returned by `NegotiationModule.run_week`.

    Returns:
      The persisted week summary captured before advancing the week counter.
    """
    # Get respective modules for state and method access
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()

    current_week = self._week_number

    # Prepare listing -> negotiation handoff.
    next_pending_matches = []
    if listing._enabled and listing_outcome is not None:
      next_pending_matches = listing.build_negotiation_transfer_payloads(
          listing_outcome.matched_pairs
      )

    if negotiation._enabled and negotiation_outcome is not None:
      relisting_pair_payloads = negotiation.build_relisting_transfer_payloads(
          self._failed_pairs_to_relist(negotiation_outcome),
          week_number=current_week,
      )
    else:
      relisting_pair_payloads = []

    if listing._enabled:
        reopened_listing_pairs = listing.reopen_failed_negotiation_pairs(
            relisting_pair_payloads
        )
    else:
        reopened_listing_pairs = []

    self._pending_matches = next_pending_matches

    listing_snapshot = listing.get_market_snapshot(
        self._module_assignments.get('listing', ())
    )
    listing_summary = (
        listing_outcome.model_dump(mode='json')
        if listing_outcome is not None
        else {'week_number': current_week}
    )
    listing_summary['buyer_states'] = listing_snapshot.get('buyers', [])
    listing_summary['listed_sellers'] = listing_snapshot.get('listed_sellers', [])
    listing_summary['released_seller_ids'] = listing_snapshot.get(
        'released_seller_ids',
        [],
    )
    listing_summary['inactive_seller_ids'] = listing_snapshot.get(
        'inactive_seller_ids',
        [],
    )
    listing_summary['active_seller_ids'] = listing_snapshot.get(
        'active_seller_ids',
        [],
    )

    negotiation_summary = (
        dict(negotiation_outcome) if negotiation_outcome is not None else {}
    )
    negotiation_summary['pair_states'] = negotiation.get_pair_state_snapshots(
        self._module_assignments.get('negotiation', ())
    )

    # Logging
    self._last_week_summary = {
        'week_number': current_week,
        'assignments': dict(self._module_assignments),
        'listing_enabled': listing.is_enabled(),
        'negotiation_enabled': negotiation.is_enabled(),
        'listing': listing_summary,
        'negotiation': negotiation_summary,
        'pending_matches_for_next_week': list(self._pending_matches),
        'reopened_listing_pairs': reopened_listing_pairs,
    }
    self._log_week_summary(
        summary=self._last_week_summary,
        week_number=current_week,
        listing_outcome=listing_outcome,
        negotiation_outcome=negotiation_outcome,
    )
    # Increment week number
    self._week_number += 1
    return dict(self._last_week_summary)

  # Snapshot and persistence helpers.
  def get_week_snapshot(self) -> dict[str, Any]:
    """Returns the latest completed-week summary or the current live snapshot."""
    if self._last_week_summary:
      return dict(self._last_week_summary)
    return {
        'week_number': self._week_number,
        'assignments': dict(self._module_assignments),
        'pending_matches_for_next_week': list(self._pending_matches),
    }

  def should_terminate(self) -> bool:
    """Returns whether the weekly engine can stop advancing the simulation.

    Termination waits for both enabled modules to finish and also blocks if the
    coordinator is still holding pending listing matches that negotiation has
    not yet consumed.
    """
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()
    enabled_modules = [listing.is_enabled(), negotiation.is_enabled()]
    if not any(enabled_modules):
      return True
    pending_matches_blocking = negotiation.is_enabled() and bool(self._pending_matches)
    return (
        not pending_matches_blocking
        and (not listing.is_enabled() or listing.is_finished())
        and (not negotiation.is_enabled() or negotiation.is_finished())
    )

  def _make_pre_act_value(self) -> str:
    return json.dumps(self.get_week_snapshot())

  def get_state(self) -> entity_component.ComponentState:
    """Serializes coordinator-owned cross-module scheduling state."""
    return {
        'player_ids': list(self._player_ids),
        'id_to_name': dict(self._id_to_name),
        'listing_component_key': self._listing_component_key,
        'negotiation_component_key': self._negotiation_component_key,
        'week_number': self._week_number,
        'pending_matches': list(self._pending_matches),
        'module_assignments': dict(self._module_assignments),
        'last_week_summary': dict(self._last_week_summary),
    }

  def get_dynamic_state(self) -> entity_component.ComponentState:
    """Serializes the lightweight mutable state needed for live inspection."""
    return {
        'week_number': self._week_number,
        'pending_matches': list(self._pending_matches),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores coordinator-owned state without touching module internals."""
    # Restore only the coordinator-owned global state. Module internals are
    # restored by their own components.
    if 'player_ids' in state:
      self._player_ids = tuple(str(player_id) for player_id in state['player_ids'])
    if 'id_to_name' in state:
      self._id_to_name = {
          str(player_id): str(player_name)
          for player_id, player_name in state['id_to_name'].items()
      }
    if 'listing_component_key' in state:
      self._listing_component_key = str(state['listing_component_key'])
    if 'negotiation_component_key' in state:
      self._negotiation_component_key = str(state['negotiation_component_key'])
    self._week_number = int(state.get('week_number', 1))
    self._pending_matches = [dict(item) for item in state.get('pending_matches', [])]
    module_assignments = state.get('module_assignments', {})
    if isinstance(module_assignments, Mapping):
      listing_assignments = module_assignments.get('listing', [])
      negotiation_assignments = module_assignments.get('negotiation', [])
      self._module_assignments = {
          'listing': [str(value) for value in listing_assignments],
          'negotiation': [
              [str(token) for token in pair]
              for pair in negotiation_assignments
              if isinstance(pair, Sequence) and not isinstance(pair, str)
          ],
      }
    if 'last_week_summary' in state:
      self._last_week_summary = dict(state['last_week_summary'])
