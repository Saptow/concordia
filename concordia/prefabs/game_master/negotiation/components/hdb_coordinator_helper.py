"""Shared weekly coordinator state for the HDB market game master."""

from collections.abc import Mapping, Sequence
import json
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.prefabs.game_master.negotiation.components import hdb_listing
from concordia.prefabs.game_master.negotiation.components import hdb_negotiation
from concordia.typing import entity_component


class WeeklyCoordinator(action_spec_ignored.ActionSpecIgnored):
  """Owns shared weekly state across listing and negotiation modules."""

  def __init__(
      self,
      *,
      listing_component_key: str = 'listing_module',
      negotiation_component_key: str = 'negotiation_module',
      pre_act_label: str = 'Weekly coordinator state',
  ):
    super().__init__(pre_act_label=pre_act_label)
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
  def get_listing_module(self) -> hdb_listing.ListingModule:
    return self.get_entity().get_component(
        self._listing_component_key,
        type_=hdb_listing.ListingModule,
    )

  def get_negotiation_module(self) -> hdb_negotiation.NegotiationModule:
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
    return self.get_listing_module().get_player_ids()

  def get_player_name(self, player_id: str) -> str:
    return self.get_listing_module().get_player_name(player_id)


  # Week preparation helpers.
  def _compute_assignments(
      self,
      new_negotiation_pairs: Sequence[Mapping[str, Any]] = (),
  ) -> tuple[set[str], list[tuple[str, str]]]:
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
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()
    current_week = self._week_number

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

  def complete_week(
      self,
      *,
      listing_outcome: Any | None = None,
      negotiation_outcome: Mapping[str, Any] | None = None,
  ) -> dict[str, Any]:
    listing = self.get_listing_module()
    negotiation = self.get_negotiation_module()
    current_week = self._week_number

    next_pending_matches = []
    # Listing matches are staged for the next week's negotiation intake.
    if listing_outcome is not None:
      next_pending_matches = [
          match.model_dump() for match in listing_outcome.matched_pairs
      ]

    self._pending_matches = next_pending_matches

    self._last_week_summary = {
        'week_number': current_week,
        'assignments': dict(self._module_assignments),
        'listing_enabled': listing.is_enabled(),
        'negotiation_enabled': negotiation.is_enabled(),
        'listing': listing_outcome.model_dump() if listing_outcome else None,
        'negotiation': dict(negotiation_outcome) if negotiation_outcome else None,
        'pending_matches_for_next_week': list(self._pending_matches),
    }
    self._week_number += 1
    return dict(self._last_week_summary)

  # Snapshot and persistence helpers.
  def get_week_snapshot(self) -> dict[str, Any]:
    if self._last_week_summary:
      return dict(self._last_week_summary)
    return {
        'week_number': self._week_number,
        'assignments': dict(self._module_assignments),
        'pending_matches_for_next_week': list(self._pending_matches),
    }

  def should_terminate(self) -> bool:
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
    return {
        'listing_component_key': self._listing_component_key,
        'negotiation_component_key': self._negotiation_component_key,
        'week_number': self._week_number,
        'pending_matches': list(self._pending_matches),
        'module_assignments': dict(self._module_assignments),
        'last_week_summary': dict(self._last_week_summary),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    # Restore only the coordinator-owned global state. Module internals are
    # restored by their own components.
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
