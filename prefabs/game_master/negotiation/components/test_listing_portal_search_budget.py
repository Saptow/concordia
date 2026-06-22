import unittest

from concordia.hdb_simulation import listing_portal
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas


class _StubRetriever:

  def __init__(self, results):
    self._results = list(results)
    self.last_max_budget = None

  def search(self, query: str, *, max_budget: float | None = None, limit: int = 5):
    del query, limit
    self.last_max_budget = max_budget
    return [
        result
        for result in self._results
        if max_budget is None or float(result.listing_price) <= float(max_budget)
    ]

  def get_listing_record(self, seller_id: str):
    for result in self._results:
      if result.seller_id == seller_id:
        return qdrant_schemas.ListingRecord.model_validate(
            result.model_dump(mode='python')
        )
    return None

  def update_listing_payload(self, record):
    del record

  def deactivate_listing(self, seller_id: str):
    del seller_id


class ListingPortalSearchBudgetTest(unittest.TestCase):

  def test_search_uses_budget_max_not_effective_reservation(self):
    flat = common_schemas.Flat(
        flat_type=common_schemas.FlatType.FOUR_ROOM,
        address='123 Example Street',
        description='Near MRT and mall',
        town='Toa Payoh',
        storey_range='10 TO 12',
        remaining_lease=78.0,
        contra=False,
        extension_of_stay=False,
        ethnic_eligibility='All',
        spr_eligibility='All',
        floor_area_sqm=92.0,
        nearby_amenities=[],
    )
    buyer = listing_schemas.PortalBuyer(
        id='buyer_001',
        name='Buyer One',
        preferences=common_schemas.BuyerPreferenceProfile(
            preferences=[
                common_schemas.BuyerPreferenceItem(
                    category='flat_type',
                    description=str(common_schemas.FlatType.FOUR_ROOM),
                    strength=1.0,
                ),
                common_schemas.BuyerPreferenceItem(
                    category='town',
                    description='Toa Payoh',
                    strength=1.0,
                ),
            ]
        ),
        budget=common_schemas.BuyerBudgetRange(
            min_price=400000.0,
            max_price=600000.0,
        ),
        reservation_price_prior=450000.0,
    )
    result = listing_schemas.PortalSearchResult(
        listing_id='listing::seller_001',
        seller_id='seller_001',
        seller_name='Seller One',
        listing_price=500000.0,
        listing_summary='4-room flat in Toa Payoh',
        flat=flat,
        listed_week=1,
        active=True,
        score=0.9,
    )
    retriever = _StubRetriever([result])
    portal = listing_portal.ListingPortal(retriever=retriever)

    search_result = portal.search_and_request(buyer, week=1)

    self.assertEqual(retriever.last_max_budget, 600000.0)
    self.assertEqual(len(search_result.results), 1)
    self.assertEqual(len(portal.requests_by_seller.get('seller_001', [])), 1)


if __name__ == '__main__':
  unittest.main()
