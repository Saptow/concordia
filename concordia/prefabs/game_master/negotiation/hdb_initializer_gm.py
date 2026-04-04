"""Transaction-conditioned initializer GM for the HDB market workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any

from absl import logging
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import game_master as gm_components
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.name_utils import resolve_profile_name
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import prefab as prefab_lib


def _unique_market_display_name(
    *,
    participant_id: str,
    requested_name: str,
    role_label: str,
    seen_names: set[str],
) -> str:
  """Returns a display name that is unique across the whole market."""
  candidate = str(requested_name).strip() or str(participant_id).strip()
  if candidate not in seen_names:
    seen_names.add(candidate)
    return candidate

  suffix = participant_id.rsplit('_', 1)[-1]
  disambiguated = f'{candidate} ({role_label} {suffix})'
  if disambiguated not in seen_names:
    logging.warning(
        'Duplicate participant display name %r detected; using %r instead.',
        candidate,
        disambiguated,
    )
    seen_names.add(disambiguated)
    return disambiguated

  counter = 2
  while True:
    numbered = f'{disambiguated} #{counter}'
    if numbered not in seen_names:
      logging.warning(
          'Duplicate participant display name %r detected repeatedly; using %r instead.',
          candidate,
          numbered,
      )
      seen_names.add(numbered)
      return numbered
    counter += 1


def build_market_profiles(
    bundle: Mapping[str, Any],
    *,
    town: str,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
  """Build listing/negotiation profiles from a market-segment bundle."""
  buyer_profiles: dict[str, dict[str, object]] = {}
  seen_names: set[str] = set()
  for buyer in bundle.get('buyers_retained', ()):
    buyer_id = str(buyer['buyer_id'])
    buyer_name = resolve_profile_name(
        buyer,
        fallback_name=f"{town} Buyer {buyer_id.rsplit('_', 1)[-1]}",
    )
    buyer_name = _unique_market_display_name(
        participant_id=buyer_id,
        requested_name=buyer_name,
        role_label='Buyer',
        seen_names=seen_names,
    )
    description_parts = [
        (
            f"{buyer['age']}-year-old {buyer['occupation_category']} looking "
            f"for an HDB resale flat in {town}."
        ),
    ]
    if str(buyer.get('general_persona', '')).strip():
      description_parts.append(str(buyer['general_persona']).strip())
    preference_summary = common_schemas.summarize_buyer_features(
        buyer.get('preferences')
    )
    if preference_summary:
      description_parts.append(
          'Housing priorities: ' + preference_summary
      )
    buyer_profiles[buyer_id] = {
        'name': buyer_name,
        'description': ' '.join(description_parts),
        'preferences': buyer['preferences'],
        'budget': buyer['budget'],
    }

  seller_profiles: dict[str, dict[str, object]] = {}
  for seller in bundle.get('sellers', ()):
    seller_id = str(seller['seller_id'])
    seller_name = resolve_profile_name(
        seller,
        fallback_name=f"{town} Seller {seller_id.rsplit('_', 1)[-1]}",
    )
    seller_name = _unique_market_display_name(
        participant_id=seller_id,
        requested_name=seller_name,
        role_label='Seller',
        seen_names=seen_names,
    )
    flat = seller['flat']
    motivation_summary = str(
        seller.get('seller_motivations', {}).get('motivation_summary', '')
    ).strip()
    description_parts = [
        (
            f"{seller['age']}-year-old {seller['occupation_category']} listing "
            f"a {flat['flat_type']} HDB flat in {flat['town']}."
        ),
    ]
    if str(seller.get('general_persona', '')).strip():
      description_parts.append(str(seller['general_persona']).strip())
    if motivation_summary:
      description_parts.append(f'Selling context: {motivation_summary}')
    seller_profiles[seller_id] = {
        'name': seller_name,
        'description': ' '.join(description_parts),
        'flat': flat,
        'expectations': seller['expectations'],
        'initial_market_state': seller.get('initial_market_state', ''),
        'initialization_order': int(seller.get('initialization_order', 0) or 0),
    }

  return buyer_profiles, seller_profiles


def build_entity_params(
    buyer_profiles: Mapping[str, Mapping[str, Any]],
    seller_profiles: Mapping[str, Mapping[str, Any]],
) -> tuple[list[prefab_lib.InstanceConfig], dict[str, dict[str, object]]]:
  """Build participant specs and entity configs for negotiation agents."""
  instance_configs: list[prefab_lib.InstanceConfig] = []
  participant_specs: dict[str, dict[str, object]] = {}

  for buyer_id, payload in buyer_profiles.items():
    buyer_params = {
        'id': buyer_id,
        'role': 'buyer',
        'name': payload['name'],
        'description': payload['description'],
        'preferences': payload['preferences'],
        'budget': payload['budget'],
        'negotiation_config': {
            'preferences': payload['preferences'],
            'own_confidence': 0.75,
            'counterpart_confidence': 0.5,
            'own_reservation_': payload['budget']['max_price'],
            'own_reservation_std': 1000,
            'cp_reservation_': payload['budget']['max_price'] * 0.95,
            'lambda_': 1.0,
            'a': 5.0,
            'b': 100,
            'reservation_value': str(payload['budget']['max_price']),
            'flat_listing': '{}',
            'initial_observations': [],
        },
    }
    participant_specs[buyer_id] = buyer_params
    instance_configs.append(
        prefab_lib.InstanceConfig(
            prefab='negotiation__uncertain_negotiator__Entity',
            role=prefab_lib.Role.ENTITY,
            params=buyer_params,
        )
    )

  for seller_id, payload in seller_profiles.items():
    seller_params = {
        'id': seller_id,
        'role': 'seller',
        'name': payload['name'],
        'description': payload['description'],
        'flat': payload['flat'],
        'expectations': payload['expectations'],
        'negotiation_config': {
            'own_confidence': 1.0,
            'counterpart_confidence': 0.5,
            'own_reservation_': payload['expectations']['min_price'],
            'own_reservation_std': 1000,
            'cp_reservation_': payload['expectations']['max_price'] * 0.95,
            'lambda_': 1.0,
            'a': 5.0,
            'b': 100,
            'reservation_value': str(payload['expectations']['min_price']),
            'flat_listing': json.dumps(payload['flat'], ensure_ascii=False),
        },
    }
    participant_specs[seller_id] = seller_params
    instance_configs.append(
        prefab_lib.InstanceConfig(
            prefab='negotiation__uncertain_negotiator__Entity',
            role=prefab_lib.Role.ENTITY,
            params=seller_params,
        )
    )

  return instance_configs, participant_specs


@dataclasses.dataclass(frozen=True)
class MarketInitializationSummary:
  week_number: int
  initial_negotiation_pairs: tuple[tuple[str, str], ...] = ()
  active_listed_seller_ids: tuple[str, ...] = ()
  inactive_seller_ids: tuple[str, ...] = ()


class HDBMarketInitialiser(action_spec_ignored.ActionSpecIgnored):
  """One-shot initializer that seeds listing and negotiation state for week 1."""

  def __init__(
      self,
      *,
      bundle: Mapping[str, Any],
      town: str,
      next_game_master_name: str,
      week_number: int = 1,
      pre_act_label: str = 'HDB market initializer',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._bundle = dict(bundle)
    self._town = str(town)
    self._next_game_master_name = str(next_game_master_name)
    self._week_number = int(week_number)
    self._buyers_raw = {
        str(buyer['buyer_id']): dict(buyer)
        for buyer in self._bundle.get('buyers_retained', ())
    }
    self._sellers_raw = {
        str(seller['seller_id']): dict(seller)
        for seller in self._bundle.get('sellers', ())
    }
    self._initialized = False
    self._summary = MarketInitializationSummary(week_number=self._week_number)

  @staticmethod
  def _normalize_market_state(value: object) -> str:
    return str(value or '').strip().casefold().replace('-', '_').replace(' ', '_')

  def _sellers_in_state(self, state: str) -> list[str]:
    target = self._normalize_market_state(state)
    seller_ids: list[str] = []
    for seller_id, seller in self._sellers_raw.items():
      if self._normalize_market_state(seller.get('initial_market_state')) == target:
        seller_ids.append(seller_id)
    return seller_ids

  @staticmethod
  def _listing_price_for_seller(seller: listing_schemas.PortalSeller) -> float:
    return max(
        float(seller.expectations.min_price),
        float(seller.expectations.max_price),
    )

  def _candidate_buyer_ids_for_seller(
      self,
      seller_id: str,
      *,
      buyer_profiles: Mapping[str, Mapping[str, Any]],
      seller_profiles: Mapping[str, Mapping[str, Any]],
      used_buyer_ids: set[str],
  ) -> list[str]:
    seller_raw = self._sellers_raw[seller_id]
    seller_profile = seller_profiles[seller_id]
    flat = seller_profile['flat']
    listing_price = float(seller_profile['expectations']['max_price'])
    linked_flat_id = str(seller_raw.get('linked_flat_id', '')).strip()
    seeded_buyer_id = str(seller_raw.get('seeded_buyer_id', '')).strip()
    preferred_buyer_ids = [
        str(buyer_id).strip()
        for buyer_id in seller_raw.get('potential_buyer_ids', ())
        if str(buyer_id).strip()
    ]
    if seeded_buyer_id:
      preferred_buyer_ids = [seeded_buyer_id] + [
          buyer_id for buyer_id in preferred_buyer_ids if buyer_id != seeded_buyer_id
      ]

    if preferred_buyer_ids:
      ranked_preferred_buyer_ids: list[str] = []
      for buyer_id in preferred_buyer_ids:
        if buyer_id in used_buyer_ids:
          continue
        buyer_profile = buyer_profiles.get(buyer_id)
        if buyer_profile is None:
          continue
        budget_max = float(buyer_profile['budget']['max_price'])
        if budget_max < listing_price:
          continue
        ranked_preferred_buyer_ids.append(buyer_id)
      if ranked_preferred_buyer_ids:
        return ranked_preferred_buyer_ids

    ranked: list[tuple[int, float, str]] = []
    for buyer_id, buyer_raw in self._buyers_raw.items():
      if buyer_id in used_buyer_ids:
        continue
      buyer_profile = buyer_profiles.get(buyer_id)
      if buyer_profile is None:
        continue
      budget_max = float(buyer_profile['budget']['max_price'])
      if budget_max < listing_price:
        continue

      preferences = common_schemas.coerce_buyer_preferences(
          buyer_profile.get('preferences')
      )
      feasible_flat_ids = {
          str(flat_id).strip()
          for flat_id in buyer_raw.get('feasible_flat_ids', ())
          if str(flat_id).strip()
      }
      score = 0
      if linked_flat_id and linked_flat_id in feasible_flat_ids:
        score += 10
      if preferences and flat['flat_type'] in preferences.values_for('flat_type'):
        score += 3
      if preferences and flat['town'] in preferences.values_for('town'):
        score += 2
      price_gap = abs(budget_max - listing_price)
      ranked.append((score, -price_gap, buyer_id))

    ranked.sort(reverse=True)
    return [buyer_id for _, _, buyer_id in ranked]

  def _seed_initial_negotiations(
      self,
      *,
      listing_module: Any,
      buyer_profiles: Mapping[str, Mapping[str, Any]],
      seller_profiles: Mapping[str, Mapping[str, Any]],
      negotiating_seller_ids: Sequence[str],
  ) -> list[listing_schemas.NegotiationMatch]:
    portal = listing_module._ensure_portal()
    matched_pairs: list[listing_schemas.NegotiationMatch] = []
    used_buyer_ids: set[str] = set()

    for seller_id in negotiating_seller_ids:
      seller = listing_module._sellers.get(seller_id)
      if seller is None or portal.is_player_closed(seller_id):
        continue

      portal.list_flat(
          seller,
          week=self._week_number,
          listing_price=self._listing_price_for_seller(seller),
      )
      candidate_buyer_ids = self._candidate_buyer_ids_for_seller(
          seller_id,
          buyer_profiles=buyer_profiles,
          seller_profiles=seller_profiles,
          used_buyer_ids=used_buyer_ids,
      )

      matched = None
      for buyer_id in candidate_buyer_ids:
        buyer = listing_module._buyers.get(buyer_id)
        if buyer is None or portal.is_player_closed(buyer_id):
          continue
        portal.search_and_request(buyer, week=self._week_number)
        open_requests = portal.requests_by_seller.get(seller_id, ())
        request_seeded = any(request.buyer_id == buyer_id for request in open_requests)
        if not request_seeded:
          seeded_request = portal.submit_negotiation_request(
              buyer,
              seller_id=seller_id,
              week=self._week_number,
              market_valuation_notes=(
                  'Seeded directly by the market initializer because retrieval '
                  'did not surface the seller listing during week-1 bootstrap.'
              ),
          )
          request_seeded = seeded_request is not None
          if request_seeded:
            logging.info(
                'Seeded initial negotiation request directly for buyer %s and seller %s.',
                buyer_id,
                seller_id,
            )
        if not request_seeded:
          continue
        matched = portal.review_requests_and_start_negotiation(
            seller,
            week=self._week_number,
        )
        if matched is not None:
          matched_pairs.append(matched)
          used_buyer_ids.add(matched.buyer_id)
          break

      if matched is None:
        logging.warning(
            'Unable to seed initial negotiation for seller %s; leaving listing active.',
            seller_id,
        )

    return matched_pairs

  def _activate_listed_sellers(
      self,
      *,
      listing_module: Any,
      listed_seller_ids: Sequence[str],
  ) -> list[str]:
    portal = listing_module._ensure_portal()
    active_seller_ids: list[str] = []
    for seller_id in listed_seller_ids:
      seller = listing_module._sellers.get(seller_id)
      if seller is None or portal.is_player_closed(seller_id):
        continue
      portal.list_flat(
          seller,
          week=self._week_number,
          listing_price=self._listing_price_for_seller(seller),
      )
      active_seller_ids.append(seller_id)
    return active_seller_ids

  def _make_pre_act_value(self) -> str:
    return json.dumps(
        {
            'town': self._town,
            'next_game_master_name': self._next_game_master_name,
            'week_number': self._week_number,
            'initialized': self._initialized,
            'summary': dataclasses.asdict(self._summary),
        },
        ensure_ascii=False,
    )

  def initialize(self, coordinator: entity_component.EntityWithComponents) -> None:
    """Prime the coordinator's market modules exactly once."""
    if self._initialized:
      return
    if coordinator.name != self._next_game_master_name:
      logging.warning(
          'Initializer expected next GM %s but received %s.',
          self._next_game_master_name,
          coordinator.name,
      )

    buyer_profiles, seller_profiles = build_market_profiles(
        self._bundle,
        town=self._town,
    )
    listing_module = coordinator.get_component('listing_module')
    weekly_coordinator = coordinator.get_component('weekly_coordinator')

    negotiating_seller_ids = self._sellers_in_state('negotiating')
    listed_seller_ids = self._sellers_in_state('listed')
    inactive_seller_ids = self._sellers_in_state('not_yet_listed')

    matched_pairs = self._seed_initial_negotiations(
        listing_module=listing_module,
        buyer_profiles=buyer_profiles,
        seller_profiles=seller_profiles,
        negotiating_seller_ids=negotiating_seller_ids,
    )
    active_listed_seller_ids = self._activate_listed_sellers(
        listing_module=listing_module,
        listed_seller_ids=listed_seller_ids,
    )

    pending_matches = listing_module.build_negotiation_transfer_payloads(
        matched_pairs
    )
    coordinator_state = weekly_coordinator.get_state()
    coordinator_state['week_number'] = self._week_number
    coordinator_state['pending_matches'] = pending_matches
    coordinator_state['module_assignments'] = {
        'listing': [],
        'negotiation': [],
    }
    coordinator_state['last_week_summary'] = {}
    weekly_coordinator.set_state(coordinator_state)

    self._summary = MarketInitializationSummary(
        week_number=self._week_number,
        initial_negotiation_pairs=tuple(
            (match.buyer_id, match.seller_id) for match in matched_pairs
        ),
        active_listed_seller_ids=tuple(active_listed_seller_ids),
        inactive_seller_ids=tuple(inactive_seller_ids),
    )
    self._initialized = True

  def is_initialized(self) -> bool:
    return self._initialized

  def get_summary(self) -> MarketInitializationSummary:
    return self._summary

  def get_state(self) -> entity_component.ComponentState:
    return {
        'town': self._town,
        'next_game_master_name': self._next_game_master_name,
        'week_number': self._week_number,
        'initialized': int(self._initialized),
        'summary': dataclasses.asdict(self._summary),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._town = str(state.get('town', self._town))
    self._next_game_master_name = str(
        state.get('next_game_master_name', self._next_game_master_name)
    )
    self._week_number = int(state.get('week_number', self._week_number))
    self._initialized = bool(state.get('initialized', 0))
    summary = state.get('summary', {})
    if isinstance(summary, Mapping):
      self._summary = MarketInitializationSummary(
          week_number=int(summary.get('week_number', self._week_number)),
          initial_negotiation_pairs=tuple(
              tuple(str(token) for token in pair)
              for pair in summary.get('initial_negotiation_pairs', ())
              if isinstance(pair, Sequence) and not isinstance(pair, str)
          ),
          active_listed_seller_ids=tuple(
              str(value) for value in summary.get('active_listed_seller_ids', ())
          ),
          inactive_seller_ids=tuple(
              str(value) for value in summary.get('inactive_seller_ids', ())
          ),
      )


@dataclasses.dataclass
class InitialiserGameMaster(prefab_lib.Prefab):
  """Prefab for the one-shot transaction-conditioned HDB initializer GM."""

  description: str = (
      'A one-shot initializer GM that seeds listing and negotiation state from '
      'transaction-conditioned HDB market outputs before the coordinator starts.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Market_Initialiser',
          'next_game_master_name': 'Market_Coordinator',
          'town': '',
          'bundle': {},
          'week_number': 1,
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    del memory_bank
    name = str(self.params.get('name', 'Market_Initialiser'))
    player_names = [entity.name for entity in self.entities]
    initializer_key = 'market_initializer'
    initializer = HDBMarketInitialiser(
        bundle=self.params.get('bundle', {}),
        town=str(self.params.get('town', '')),
        next_game_master_name=str(
            self.params.get('next_game_master_name', 'Market_Coordinator')
        ),
        week_number=int(self.params.get('week_number', 1)),
    )
    components = {
        initializer_key: initializer,
    }
    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=list(components.keys()),
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components,
    )


GameMaster = InitialiserGameMaster
