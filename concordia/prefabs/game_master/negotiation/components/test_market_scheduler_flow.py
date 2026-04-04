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

  def set_enabled(self, enabled: bool) -> None:
    self.enabled = enabled
    self._enabled = enabled

  def is_enabled(self) -> bool:
    return self.enabled

  def get_open_pairs(self) -> list[tuple[str, str]]:
    return list(self._open_pairs)

  def build_relisting_transfer_payloads(self, pair_records, *, week_number: int):
    del pair_records, week_number
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


class ListingReleaseTest(unittest.TestCase):

  def _build_listing_module(self) -> hdb_listing.ListingModule:
    buyer_profile = _buyer_profile(name='Buyer 1')
    seller_one = _seller_profile(name='Seller 1')
    seller_two = _seller_profile(
        name='Seller 2',
        flat_type='4-Room',
        min_price=620000.0,
        max_price=660000.0,
    )
    seller_one['initial_market_state'] = 'listed'
    seller_one['initialization_order'] = 1
    seller_two['initial_market_state'] = 'not_yet_listed'
    seller_two['initialization_order'] = 2

    return hdb_listing.ListingModule(
        player_names=(
            buyer_profile['name'],
            seller_one['name'],
            seller_two['name'],
        ),
        player_ids=('buyer_001', 'seller_001', 'seller_002'),
        buyer_profiles={'buyer_001': buyer_profile},
        seller_profiles={
            'seller_001': seller_one,
            'seller_002': seller_two,
        },
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
