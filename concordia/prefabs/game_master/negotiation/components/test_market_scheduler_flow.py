import copy
import unittest

from concordia.concordia.prefabs.game_master.negotiation.components import (
    hdb_coordinator_helper,
    hdb_listing,
    hdb_negotiation_helpers,
)
from concordia.concordia.prefabs.game_master.negotiation import hdb_initializer_gm
from concordia.hdb_simulation.models.buyer_data import BUYER_DATA
from concordia.hdb_simulation.models.seller_data import SELLER_DATA


class _FakeListingModule:

  def __init__(self, open_ids):
    self._open_ids = set(open_ids)
    self.prepare_calls = 0
    self.enabled = True

  def set_enabled(self, enabled: bool) -> None:
    self.enabled = enabled

  def is_enabled(self) -> bool:
    return self.enabled

  def get_open_player_ids(self) -> set[str]:
    return set(self._open_ids)

  def prepare_weekly_market(self, *, week_number: int | None = None) -> list[str]:
    del week_number
    self.prepare_calls += 1
    return []


class _FakeNegotiationModule:

  def __init__(self, open_pairs):
    self._open_pairs = list(open_pairs)
    self.enabled = True

  def set_enabled(self, enabled: bool) -> None:
    self.enabled = enabled

  def is_enabled(self) -> bool:
    return self.enabled

  def get_open_pairs(self) -> list[tuple[str, str]]:
    return list(self._open_pairs)


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
            'flat': copy.deepcopy(SELLER_DATA['seller_001']['flat']),
            'expectations': copy.deepcopy(SELLER_DATA['seller_001']['expectations']),
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
    buyer_profile = copy.deepcopy(BUYER_DATA['buyer_001'])
    seller_one = copy.deepcopy(SELLER_DATA['seller_001'])
    seller_two = copy.deepcopy(SELLER_DATA['seller_002'])
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
