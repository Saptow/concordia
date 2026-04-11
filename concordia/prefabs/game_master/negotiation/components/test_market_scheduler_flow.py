import copy
import unittest

from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.pipeline import flat_embedding
from concordia.prefabs.game_master.negotiation.components import (
    hdb_coordinator_helper,
    hdb_listing,
    hdb_negotiation,
    hdb_negotiation_helpers,
)
from concordia.prefabs.game_master.negotiation import hdb_initializer_gm


def _buyer_profile(
    *,
    name: str = 'Buyer 1',
    min_price: float = 450000.0,
    max_price: float = 650000.0,
) -> dict[str, object]:
  return {
      'name': name,
      'description': 'Test buyer profile.',
      'preferences': {
          'preferences': [
              {
                  'category': 'flat_type',
                  'description': '3-Room',
                  'strength': 1.0,
              },
              {
                  'category': 'town',
                  'description': 'Choa Chu Kang',
                  'strength': 0.8,
              },
          ],
      },
      'budget': {
          'min_price': min_price,
          'max_price': max_price,
      },
  }


def _seller_profile(
    *,
    name: str = 'Seller 1',
    flat_type: str = '3-Room',
    town: str = 'Choa Chu Kang',
    min_price: float = 500000.0,
    max_price: float = 530000.0,
) -> dict[str, object]:
  return {
      'name': name,
      'description': 'Test seller profile.',
      'flat': {
          'flat_type': flat_type,
          'address': 'Blk 123 Test Avenue Singapore 680123',
          'description': 'Test flat.',
          'town': town,
          'storey_range': '05 to 08',
          'remaining_lease': 78.0,
          'contra': False,
          'extension_of_stay': False,
          'ethnic_eligibility': 'No quota limit',
          'spr_eligibility': 'True',
          'floor_area_sqm': 68.0,
      },
      'expectations': {
          'min_price': min_price,
          'max_price': max_price,
      },
  }


class _FakeListingModule:

  def __init__(self, open_ids):
    self._open_ids = set(open_ids)
    self.prepare_calls = 0
    self.enabled = True
    self._enabled = True
    self.transfer_payloads = []
    self.reopened_payloads = []
    self.market_snapshot = {
        'buyers': [],
        'listed_sellers': [],
        'released_seller_ids': [],
        'inactive_seller_ids': [],
        'active_seller_ids': sorted(open_ids),
    }

  def set_enabled(self, enabled: bool) -> None:
    self.enabled = enabled
    self._enabled = enabled

  def is_enabled(self) -> bool:
    return self.enabled

  def get_open_player_ids(self) -> set[str]:
    return set(self._open_ids)

  def prepare_weekly_market(self, *, week_number: int | None = None) -> list[str]:
    del week_number
    self.prepare_calls += 1
    return []

  def build_negotiation_transfer_payloads(self, matched_pairs):
    del matched_pairs
    return list(self.transfer_payloads)

  def reopen_failed_negotiation_pairs(self, payloads):
    self.reopened_payloads.append(list(payloads))
    return list(payloads)

  def get_market_snapshot(self, player_ids=None):
    del player_ids
    return dict(self.market_snapshot)

  def is_finished(self) -> bool:
    return not self._open_ids


class _FakeNegotiationModule:

  def __init__(self, open_pairs):
    self._open_pairs = list(open_pairs)
    self.enabled = True
    self._enabled = True
    self.relisting_payloads = []
    self.pair_states = []
    self.last_relisting_pair_records = None

  def set_enabled(self, enabled: bool) -> None:
    self.enabled = enabled
    self._enabled = enabled

  def is_enabled(self) -> bool:
    return self.enabled

  def get_open_pairs(self) -> list[tuple[str, str]]:
    return list(self._open_pairs)

  def build_relisting_transfer_payloads(self, pair_records, *, week_number: int):
    del week_number
    normalized_pair_records = list(pair_records)
    self.last_relisting_pair_records = normalized_pair_records
    if not normalized_pair_records:
      return []
    return list(self.relisting_payloads)

  def get_pair_state_snapshots(self, pair_ids=None):
    del pair_ids
    return list(self.pair_states)

  def is_finished(self) -> bool:
    return not self._open_pairs


class _FakeEntity:

  def __init__(self, components):
    self._components = dict(components)

  def get_component(self, key, type_=None):
    del type_
    return self._components[key]


class _ObservedEntity:

  def __init__(self, *, name: str, player_id: str):
    self.name = name
    self._hdb_player_id = player_id
    self.observations: list[str] = []

  def observe(self, observation: str) -> None:
    self.observations.append(observation)


class _FakeNameModel:

  def __init__(self, response: str):
    self._response = response
    self.prompts: list[str] = []

  def sample_text(self, prompt: str, **kwargs) -> str:
    del kwargs
    self.prompts.append(prompt)
    return self._response


class BuildMarketProfilesTest(unittest.TestCase):

  def test_preserves_initial_market_state_metadata(self):
    bundle = {
        'buyers_retained': [],
        'sellers': [{
            'seller_id': 'seller_999',
            'name': 'Seller 999',
            'age': 55,
            'occupation_category': 'Retired',
            'flat': copy.deepcopy(_seller_profile()['flat']),
            'expectations': copy.deepcopy(_seller_profile()['expectations']),
            'seller_motivations': {'motivation_summary': 'Downsizing.'},
            'initial_market_state': 'not_yet_listed',
            'initialization_order': 7,
            'initial_window_position': 3,
            'initial_window_size': 12,
        }],
    }

    _, seller_profiles = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
    )

    self.assertEqual(
        seller_profiles['seller_999']['initial_market_state'],
        'not_yet_listed',
    )
    self.assertEqual(
        seller_profiles['seller_999']['initialization_order'],
        7,
    )
    self.assertEqual(
        seller_profiles['seller_999']['initial_window_position'],
        3,
    )
    self.assertEqual(
        seller_profiles['seller_999']['initial_window_size'],
        12,
    )

  def test_generates_name_from_persona_with_model_when_name_missing(self):
    bundle = {
        'buyers_retained': [{
            'buyer_id': 'buyer_2023_00563',
            'age': 31,
            'occupation_category': 'Teacher',
            'general_persona': (
                'A careful and community-minded teacher who prefers a calm '
                'home base near family and values reliable transit.'
            ),
            'preferences': copy.deepcopy(_buyer_profile()['preferences']),
            'budget': copy.deepcopy(_buyer_profile()['budget']),
        }],
        'sellers': [],
    }
    model = _FakeNameModel('Nur Aisyah Rahman')

    buyer_profiles, _ = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
        model=model,
    )

    self.assertEqual(
        buyer_profiles['buyer_2023_00563']['name'],
        'Nur Aisyah Rahman',
    )
    self.assertLen(model.prompts, 1)
    self.assertIn('teacher', model.prompts[0].lower())

  def test_name_generation_prompt_uses_few_shot_extract_or_invent_examples(self):
    bundle = {
        'buyers_retained': [{
            'buyer_id': 'buyer_2023_00563',
            'age': 25,
            'occupation_category': 'Associate Professional or Technician',
            'general_persona': (
                'Ivan blends a curiosity-driven pragmatism with quiet '
                'compassion, tunes A.R. Rahman\'s soundtracks while perfecting '
                'his masala dosa, and cannot resist adding a new cricket '
                'figurine to his shelf.'
            ),
            'preferences': copy.deepcopy(_buyer_profile()['preferences']),
            'budget': copy.deepcopy(_buyer_profile()['budget']),
        }],
        'sellers': [],
    }
    model = _FakeNameModel('Ivan')

    buyer_profiles, _ = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
        model=model,
    )

    self.assertEqual(
        buyer_profiles['buyer_2023_00563']['name'],
        'Ivan',
    )
    self.assertLen(model.prompts, 1)
    self.assertIn('if the persona already contains a plausible personal name', model.prompts[0].lower())
    self.assertIn('name: ivan', model.prompts[0].lower())
    self.assertIn('name: nur aisyah rahman', model.prompts[0].lower())

  def test_generates_stable_fallback_name_from_persona_without_model(self):
    bundle = {
        'buyers_retained': [{
            'buyer_id': 'buyer_2023_00853',
            'age': 34,
            'occupation_category': 'Teacher',
            'general_persona': (
                'A careful and community-minded teacher who prefers a calm '
                'home base near family and values reliable transit.'
            ),
            'preferences': copy.deepcopy(_buyer_profile()['preferences']),
            'budget': copy.deepcopy(_buyer_profile()['budget']),
        }],
        'sellers': [],
    }

    buyer_profiles_first, _ = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
        model=None,
    )
    buyer_profiles_second, _ = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
        model=None,
    )

    generated_name = buyer_profiles_first['buyer_2023_00853']['name']
    self.assertTrue(generated_name)
    self.assertNotEqual(generated_name, 'buyer_2023_00853')
    self.assertEqual(
        generated_name,
        buyer_profiles_second['buyer_2023_00853']['name'],
    )

  def test_duplicate_single_names_gain_persona_based_surname(self):
    bundle = {
        'buyers_retained': [
            {
                'buyer_id': 'buyer_2023_00853',
                'age': 34,
                'occupation_category': 'Teacher',
                'general_persona': (
                    'A careful and community-minded teacher who prefers a calm '
                    'home base near family and values reliable transit.'
                ),
                'preferences': copy.deepcopy(_buyer_profile()['preferences']),
                'budget': copy.deepcopy(_buyer_profile()['budget']),
            },
            {
                'buyer_id': 'buyer_2023_00854',
                'age': 29,
                'occupation_category': 'Designer',
                'general_persona': (
                    'A design-conscious and quietly ambitious professional who '
                    'wants a bright flat near transit and weekend cafes.'
                ),
                'preferences': copy.deepcopy(_buyer_profile()['preferences']),
                'budget': copy.deepcopy(_buyer_profile()['budget']),
            },
        ],
        'sellers': [],
    }
    model = _FakeNameModel('Ivan')

    buyer_profiles, _ = hdb_initializer_gm.build_market_profiles(
        bundle,
        town='Choa Chu Kang',
        model=model,
    )

    self.assertEqual(buyer_profiles['buyer_2023_00853']['name'], 'Ivan')
    self.assertNotEqual(
        buyer_profiles['buyer_2023_00854']['name'],
        'Ivan',
    )
    self.assertNotIn(
        '(Buyer',
        buyer_profiles['buyer_2023_00854']['name'],
    )

class BuildEntityParamsTest(unittest.TestCase):

  def test_seller_initial_window_metadata_is_passed_to_negotiation_config(self):
    early_seller_profile = _seller_profile(name='Early Seller')
    early_seller_profile['initial_window_position'] = 1
    early_seller_profile['initial_window_size'] = 8

    _, participant_specs = hdb_initializer_gm.build_entity_params(
        buyer_profiles={},
        seller_profiles={
            'seller_early': early_seller_profile,
        },
    )

    negotiation_config = participant_specs['seller_early']['negotiation_config']
    self.assertEqual(negotiation_config['initial_window_position'], 1)
    self.assertEqual(negotiation_config['initial_window_size'], 8)
    self.assertNotIn('seller_exploration_threshold', negotiation_config)


class NegotiationListingObservationTest(unittest.TestCase):

  def test_listing_handoff_uses_structured_listing_format(self):
    buyer_profile = _buyer_profile(name='Jensen')
    seller_profile = _seller_profile(name='Zulkifli Khair Juwahir', flat_type='4-Room')
    raw_flat_payload = {
        'flat_type': '4-Room',
        'address': '809A CHOA CHU KANG AVE 1',
        'town': 'Choa Chu Kang',
        'floor_range': '10 TO 12',
        'remaining_lease_years': 93.25,
        'floor_area_sqm': 92.0,
        'amenities': {
            'mrt': {'station_names': ['Keat Hong', 'Teck Whye', 'South View']},
            'primary_schools': {
                'school_names': ['CHUA CHU KANG PRIMARY SCHOOL'],
            },
            'hawker_centres': {'hawker_names': []},
            'malls': {
                'mall_names': ['Keat Hong Shopping Centre', 'Sunshine Place'],
            },
        },
        'past_price_trends': {
            'transactions_6m': 275,
            'min_price_6m': 388000.0,
            'max_price_6m': 600000.0,
        },
    }
    seller_profile['flat'].update({
        'address': '809A CHOA CHU KANG AVE 1',
        'description': (
            'This 4-Room HDB flat is at 809A CHOA CHU KANG AVE 1 in Choa Chu Kang. '
            'It offers about 92 sqm of space, in the 10 TO 12 storey range, and '
            'with about 93.2 years of lease remaining.'
        ),
        'storey_range': '10 TO 12',
        'remaining_lease': 93.25,
        'floor_area_sqm': 92.0,
        'nearby_amenities': [
            {'name': 'Keat Hong', 'type': 'MRT', 'radius': 'Within 1km'},
            {'name': 'Teck Whye', 'type': 'MRT', 'radius': 'Within 1km'},
            {'name': 'South View', 'type': 'MRT', 'radius': 'Within 1km'},
            {
                'name': 'CHUA CHU KANG PRIMARY SCHOOL',
                'type': 'Primary School',
                'radius': 'Within 2km',
            },
            {
                'name': 'Keat Hong Shopping Centre',
                'type': 'Shopping Mall',
                'radius': 'Within 1km',
            },
            {
                'name': 'Sunshine Place',
                'type': 'Shopping Mall',
                'radius': 'Within 1km',
            },
        ],
        'past_price_trends': {
            'transactions_6m': 275,
            'min_price_6m': 388000.0,
            'max_price_6m': 600000.0,
        },
    })
    listing_record = negotiation_schemas.ListingRecord(
        listing_id='listing::seller_2023_00006',
        seller_id='seller_2023_00006',
        seller_name='Zulkifli Khair Juwahir',
        listing_price=553356.98,
        listing_summary=flat_embedding.build_listing_summary(
            raw_flat_payload,
            listing_price=553356.98,
            extension_of_stay=False,
        ),
        flat=common_schemas.Flat.model_validate(seller_profile['flat']),
        listed_week=1,
        active=False,
    )
    buyer_state = negotiation_schemas.ListingBuyerState(
        id='buyer_2023_00001',
        name='Jensen',
        description='Test buyer profile.',
        preferences=buyer_profile['preferences'],
        budget=buyer_profile['budget'],
        effective_reservation=common_schemas.NormalDistribution(
            name='buyer_effective_reservation',
            mean=540000.0,
            std=10000.0,
            confidence=0.6,
        ),
        latest_search_results=[],
        latest_market_feedback='No market feedback yet.',
    )
    seller_state = negotiation_schemas.ListingSellerState(
        id='seller_2023_00006',
        name='Zulkifli Khair Juwahir',
        description='Test seller profile.',
        flat=seller_profile['flat'],
        expectations=seller_profile['expectations'],
        effective_reservation=common_schemas.NormalDistribution(
            name='seller_effective_reservation',
            mean=525000.0,
            std=10000.0,
            confidence=0.6,
        ),
        listed=False,
        current_listing_id='listing::seller_2023_00006',
        current_listing_price=553356.98,
        open_requests=0,
    )
    payload = negotiation_schemas.ListingNegotiationTransferPayload(
        match_id='match::buyer_2023_00001::seller_2023_00006',
        week_matched=1,
        listing_record=listing_record,
        buyer_state=buyer_state,
        seller_state=seller_state,
    )
    buyer_entity = _ObservedEntity(name='Jensen', player_id='buyer_2023_00001')
    seller_entity = _ObservedEntity(
        name='Zulkifli Khair Juwahir',
        player_id='seller_2023_00006',
    )
    participant_specs = {
        'buyer_2023_00001': {
            'id': 'buyer_2023_00001',
            'name': 'Jensen',
            'role': 'buyer',
            'description': 'Test buyer profile.',
            'preferences': buyer_profile['preferences'],
            'budget': buyer_profile['budget'],
        },
        'seller_2023_00006': {
            'id': 'seller_2023_00006',
            'name': 'Zulkifli Khair Juwahir',
            'role': 'seller',
            'description': 'Test seller profile.',
            'flat': seller_profile['flat'],
            'expectations': seller_profile['expectations'],
        },
    }
    module = hdb_negotiation.NegotiationModule(
        entities=(buyer_entity, seller_entity),
        participant_specs=participant_specs,
        negotiation_pairs=(('buyer_2023_00001', 'seller_2023_00006'),),
        enabled=True,
    )
    original_batch_update = hdb_negotiation.batch_update_agents_from_listings
    hdb_negotiation.batch_update_agents_from_listings = lambda pairs: None
    try:
      module._bind_entities_for_pairs([payload])
    finally:
      hdb_negotiation.batch_update_agents_from_listings = original_batch_update

    self.assertLen(buyer_entity.observations, 1)
    observation = buyer_entity.observations[0]
    self.assertIn(
        'Listing handoff context for this negotiation:',
        observation,
    )
    _, _, listing_summary = observation.partition(
        'Listing handoff context for this negotiation:\n'
    )
    self.assertEqual(listing_summary, listing_record.listing_summary)


class ListingReleaseTest(unittest.TestCase):

  def _build_listing_module(
      self,
      seller_specs: tuple[tuple[str, str, str, int, float, float], ...] | None = None,
  ) -> hdb_listing.ListingModule:
    buyer_profile = _buyer_profile(name='Buyer 1')
    seller_specs = seller_specs or (
        ('seller_001', 'Seller 1', 'listed', 1, 500000.0, 530000.0),
        ('seller_002', 'Seller 2', 'not_yet_listed', 2, 620000.0, 660000.0),
    )
    seller_profiles: dict[str, dict[str, object]] = {}
    player_ids = ['buyer_001']
    player_names = [str(buyer_profile['name'])]
    for seller_id, seller_name, market_state, order, min_price, max_price in seller_specs:
      seller_profile = _seller_profile(
          name=seller_name,
          flat_type='4-Room' if seller_id.endswith('2') else '3-Room',
          min_price=min_price,
          max_price=max_price,
      )
      seller_profile['initial_market_state'] = market_state
      seller_profile['initialization_order'] = order
      seller_profiles[seller_id] = seller_profile
      player_ids.append(seller_id)
      player_names.append(seller_name)

    return hdb_listing.ListingModule(
        player_names=tuple(player_names),
        player_ids=tuple(player_ids),
        buyer_profiles={'buyer_001': buyer_profile},
        seller_profiles=seller_profiles,
        enabled=True,
    )

  def test_not_yet_listed_seller_is_withheld_until_later_release(self):
    listing_module = self._build_listing_module()
    initial_open_ids = listing_module.get_open_player_ids()
    self.assertIn('buyer_001', initial_open_ids)
    self.assertIn('seller_001', initial_open_ids)
    self.assertNotIn('seller_002', initial_open_ids)

    portal = listing_module._ensure_portal()
    portal.closed_sellers.add('seller_001')

    released = listing_module.prepare_weekly_market(week_number=2)
    released_open_ids = listing_module.get_open_player_ids()

    self.assertEqual(released, ['seller_002'])
    self.assertIn('seller_002', released_open_ids)
    self.assertNotIn('seller_001', released_open_ids)

  def test_releases_sellers_in_initialization_order(self):
    listing_module = self._build_listing_module((
        ('seller_001', 'Seller 1', 'listed', 1, 500000.0, 530000.0),
        ('seller_003', 'Seller 3', 'not_yet_listed', 3, 640000.0, 680000.0),
        ('seller_002', 'Seller 2', 'not_yet_listed', 2, 620000.0, 660000.0),
    ))
    portal = listing_module._ensure_portal()
    portal.closed_sellers.add('seller_001')

    first_release = listing_module.prepare_weekly_market(week_number=2)

    self.assertEqual(first_release, ['seller_002'])
    self.assertIn('seller_002', listing_module.get_open_player_ids())
    self.assertNotIn('seller_003', listing_module.get_open_player_ids())

    portal.closed_sellers.add('seller_002')
    second_release = listing_module.prepare_weekly_market(week_number=3)

    self.assertEqual(second_release, ['seller_003'])
    self.assertIn('seller_003', listing_module.get_open_player_ids())

  def test_does_not_release_when_active_seller_capacity_is_full(self):
    listing_module = self._build_listing_module((
        ('seller_001', 'Seller 1', 'listed', 1, 500000.0, 530000.0),
        ('seller_002', 'Seller 2', 'listed', 2, 620000.0, 660000.0),
        ('seller_003', 'Seller 3', 'not_yet_listed', 3, 640000.0, 680000.0),
    ))

    released = listing_module.prepare_weekly_market(week_number=2)

    self.assertEqual(released, [])
    self.assertIn('seller_001', listing_module.get_open_player_ids())
    self.assertIn('seller_002', listing_module.get_open_player_ids())
    self.assertNotIn('seller_003', listing_module.get_open_player_ids())

  def test_state_round_trip_preserves_delayed_release_queue(self):
    seller_specs = (
        ('seller_001', 'Seller 1', 'listed', 1, 500000.0, 530000.0),
        ('seller_002', 'Seller 2', 'not_yet_listed', 2, 620000.0, 660000.0),
        ('seller_003', 'Seller 3', 'not_yet_listed', 3, 640000.0, 680000.0),
    )
    listing_module = self._build_listing_module(seller_specs)
    portal = listing_module._ensure_portal()
    portal.closed_sellers.add('seller_001')
    self.assertEqual(
        listing_module.prepare_weekly_market(week_number=2),
        ['seller_002'],
    )

    saved_state = listing_module.get_state()
    restored_module = self._build_listing_module(seller_specs)
    restored_module.set_state(saved_state)

    self.assertIn('seller_002', restored_module.get_open_player_ids())
    self.assertNotIn('seller_003', restored_module.get_open_player_ids())

    restored_portal = restored_module._ensure_portal()
    restored_portal.closed_sellers.add('seller_002')
    self.assertEqual(
        restored_module.prepare_weekly_market(week_number=3),
        ['seller_003'],
    )
    self.assertIn('seller_003', restored_module.get_open_player_ids())


class WeeklyCoordinatorSchedulingTest(unittest.TestCase):

  def test_prepare_week_excludes_negotiating_players_from_listing(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001', 'buyer_002', 'seller_002'},
    )
    negotiation_module = _FakeNegotiationModule(
        open_pairs=[('buyer_001', 'seller_001')],
    )
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))

    week_context = coordinator.prepare_week()

    self.assertEqual(listing_module.prepare_calls, 1)
    self.assertEqual(
        sorted(week_context['listing_player_ids']),
        ['buyer_002', 'seller_002'],
    )
    self.assertEqual(
        week_context['open_negotiation_pairs'],
        [['buyer_001', 'seller_001']],
    )

  def test_prepare_week_deduplicates_pending_pairs_against_open_pairs(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001', 'buyer_002', 'seller_002'},
    )
    negotiation_module = _FakeNegotiationModule(
        open_pairs=[('buyer_001', 'seller_001')],
    )
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))
    state = coordinator.get_state()
    state['pending_matches'] = [{
        'buyer_id': 'buyer_001',
        'seller_id': 'seller_001',
        'buyer_state': {'id': 'buyer_001'},
        'seller_state': {'id': 'seller_001'},
    }]
    coordinator.set_state(state)

    week_context = coordinator.prepare_week()

    self.assertEqual(
        week_context['new_negotiation_pairs'],
        state['pending_matches'],
    )
    self.assertEqual(
        week_context['open_negotiation_pairs'],
        [['buyer_001', 'seller_001']],
    )
    self.assertEqual(
        sorted(week_context['listing_player_ids']),
        ['buyer_002', 'seller_002'],
    )

  def test_listing_match_hands_off_to_negotiation_next_week(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001', 'buyer_002', 'seller_002'},
    )
    listing_module.transfer_payloads = [{
        'buyer_id': 'buyer_002',
        'seller_id': 'seller_002',
        'buyer_state': {'id': 'buyer_002'},
        'seller_state': {'id': 'seller_002'},
    }]
    negotiation_module = _FakeNegotiationModule(open_pairs=[])
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))

    listing_outcome = type('ListingOutcome', (), {
        'matched_pairs': [],
        'model_dump': lambda self, mode='json': {
            'week_number': 1,
            'matched_pairs': [],
        },
    })()
    summary = coordinator.complete_week(
        listing_outcome=listing_outcome,
        negotiation_outcome=None,
    )
    next_week = coordinator.prepare_week()

    self.assertEqual(
        summary['pending_matches_for_next_week'],
        listing_module.transfer_payloads,
    )
    self.assertEqual(
        next_week['new_negotiation_pairs'],
        listing_module.transfer_payloads,
    )
    self.assertEqual(
        next_week['open_negotiation_pairs'],
        [['buyer_002', 'seller_002']],
    )
    self.assertNotIn('buyer_002', next_week['listing_player_ids'])
    self.assertNotIn('seller_002', next_week['listing_player_ids'])

  def test_failed_negotiation_reopens_into_listing(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001'},
    )
    negotiation_module = _FakeNegotiationModule(
        open_pairs=[('buyer_001', 'seller_001')],
    )
    negotiation_module.relisting_payloads = [{
        'buyer_id': 'buyer_001',
        'seller_id': 'seller_001',
        'buyer_state': {'buyer_id': 'buyer_001'},
        'seller_state': {'seller_id': 'seller_001'},
        'negotiation_history': {
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'start_week': 1,
            'end_week': 1,
            'offer_history': [],
        },
    }]
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001'),
        player_names=('Buyer 1', 'Seller 1'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))

    negotiation_outcome = {
        'week_number': 1,
        'number_of_pairs_negotiated': 1,
        'events': [],
        'closed_pairs': [{
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'buyer_name': 'Buyer 1',
            'seller_name': 'Seller 1',
            'outcome': 'CLOSED_WITHOUT_SUCCESS',
        }],
        'successful_pairs': [],
        'failed_pairs': [{
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'buyer_name': 'Buyer 1',
            'seller_name': 'Seller 1',
            'outcome': 'CLOSED_WITHOUT_SUCCESS',
        }],
    }

    summary = coordinator.complete_week(
        listing_outcome=None,
        negotiation_outcome=negotiation_outcome,
    )

    self.assertEqual(
        listing_module.reopened_payloads,
        [negotiation_module.relisting_payloads],
    )
    self.assertEqual(
        summary['reopened_listing_pairs'],
        negotiation_module.relisting_payloads,
    )

  def test_successful_negotiation_does_not_reopen_into_listing(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001'},
    )
    negotiation_module = _FakeNegotiationModule(
        open_pairs=[('buyer_001', 'seller_001')],
    )
    negotiation_module.pair_states = [{
        'buyer_id': 'buyer_001',
        'seller_id': 'seller_001',
        'buyer_name': 'Buyer 1',
        'seller_name': 'Seller 1',
        'closed': True,
        'outcome': 'SUCCESS',
    }]
    negotiation_module.relisting_payloads = [{
        'buyer_id': 'buyer_001',
        'seller_id': 'seller_001',
    }]
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001'),
        player_names=('Buyer 1', 'Seller 1'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))

    negotiation_outcome = {
        'week_number': 1,
        'number_of_pairs_negotiated': 1,
        'events': [],
        'closed_pairs': [ {
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'buyer_name': 'Buyer 1',
            'seller_name': 'Seller 1',
            'outcome': 'SUCCESS',
        }],
        'successful_pairs': [ {
            'buyer_id': 'buyer_001',
            'seller_id': 'seller_001',
            'buyer_name': 'Buyer 1',
            'seller_name': 'Seller 1',
            'outcome': 'SUCCESS',
        }],
        'failed_pairs': [],
    }

    summary = coordinator.complete_week(
        listing_outcome=None,
        negotiation_outcome=negotiation_outcome,
    )

    self.assertEqual(listing_module.reopened_payloads, [[]])
    self.assertEqual(summary['reopened_listing_pairs'], [])
    self.assertEqual(summary['negotiation']['pair_states'], [])

  def test_state_round_trip_preserves_pending_matches_for_next_week(self):
    listing_module = _FakeListingModule(
        open_ids={'buyer_001', 'seller_001', 'buyer_002', 'seller_002'},
    )
    listing_module.transfer_payloads = [{
        'buyer_id': 'buyer_002',
        'seller_id': 'seller_002',
        'buyer_state': {'id': 'buyer_002'},
        'seller_state': {'id': 'seller_002'},
    }]
    negotiation_module = _FakeNegotiationModule(open_pairs=[])
    coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
    )
    coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))
    listing_outcome = type('ListingOutcome', (), {
        'matched_pairs': [],
        'model_dump': lambda self, mode='json': {
            'week_number': 1,
            'matched_pairs': [],
        },
    })()
    coordinator.complete_week(
        listing_outcome=listing_outcome,
        negotiation_outcome=None,
    )

    restored_coordinator = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
    )
    restored_coordinator.set_entity(_FakeEntity({
        'listing_module': listing_module,
        'negotiation_module': negotiation_module,
    }))
    restored_coordinator.set_state(coordinator.get_state())

    next_week = restored_coordinator.prepare_week()

    self.assertEqual(next_week['week_number'], 2)
    self.assertEqual(
        next_week['new_negotiation_pairs'],
        listing_module.transfer_payloads,
    )
    self.assertEqual(
        next_week['open_negotiation_pairs'],
        [['buyer_002', 'seller_002']],
    )


class NegotiationSchedulerTest(unittest.TestCase):

  def test_advances_only_pairs_that_negotiated(self):
    scheduler = hdb_negotiation_helpers.NegotiationScheduler(
        player_names=('Buyer 1', 'Seller 1', 'Buyer 2', 'Seller 2'),
        player_ids=('buyer_001', 'seller_001', 'buyer_002', 'seller_002'),
        negotiation_pairs=(
            ('buyer_001', 'seller_001'),
            ('buyer_002', 'seller_002'),
        ),
    )

    scheduler.advance_week([('buyer_002', 'seller_002')])

    self.assertEqual(
        scheduler.get_pair_round_number('buyer_001', 'seller_001'),
        1,
    )
    self.assertEqual(
        scheduler.get_pair_round_number('buyer_002', 'seller_002'),
        2,
    )


if __name__ == '__main__':
  unittest.main()
