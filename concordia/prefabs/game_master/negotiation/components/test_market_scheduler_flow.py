import copy
import unittest

from concordia.prefabs.game_master.negotiation.components import (
    hdb_coordinator_helper,
    hdb_listing,
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
