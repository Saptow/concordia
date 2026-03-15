from collections.abc import Mapping, Sequence
import functools
import json
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation import listing_portal as listing_portal_lib
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.typing import entity_component
from concordia.utils import concurrency


def _dedupe_strings(values: Sequence[str]) -> list[str]:
  """Returns unique non-empty strings in first-seen order."""
  seen: set[str] = set()
  ordered: list[str] = []
  for value in values:
    key = str(value).strip()
    if not key or key in seen:
      continue
    seen.add(key)
    ordered.append(key)
  return ordered

def _listing_price_for_seller(seller: listing_schemas.PortalSeller) -> float:
  """Chooses the current listing price from the seller expectations."""
  min_price = float(seller.expectations.min_price)
  max_price = float(seller.expectations.max_price)
  return max(min_price, max_price)


def execute_listing_week(
    *,
    portal: listing_portal_lib.ListingPortal,
    buyers: Mapping[str, listing_schemas.PortalBuyer],
    sellers: Mapping[str, listing_schemas.PortalSeller],
    week_number: int,
    active_player_ids: Sequence[str],
    active_player_names: Sequence[str],
) -> listing_schemas.ListingWeeklyBatchOutcome:
  """Executes one scripted listing week for the provided active participants.

  Phase order is fixed across the week:
  1. Unlisted active sellers activate their listing.
  2. Active buyers search and submit requests.
  3. Previously listed sellers review requests and create matches.

  Each phase runs participant-local work concurrently, then collates the
  results into a single weekly outcome.
  """
  if not active_player_names:
    return listing_schemas.ListingWeeklyBatchOutcome(
        week_number=week_number,
        active_player_names=[],
    )

  assigned_ids = {str(player_id) for player_id in active_player_ids}
  seller_listing_status_at_week_start = {
      seller_id: portal.is_seller_listed(seller_id)
      for seller_id in sellers
      if seller_id in assigned_ids and not portal.is_player_closed(seller_id)
  }

  newly_listed_listing_ids: list[str] = []
  buyers_processed: list[str] = []
  sellers_reviewed: list[str] = []
  matched_pairs: list[listing_schemas.NegotiationMatch] = []
  closed_player_names: list[str] = []

  eligible_sellers_to_list = [
      (seller_id, seller)
      for seller_id, seller in sellers.items()
      if (
          seller_id in assigned_ids
          and not portal.is_player_closed(seller_id)
          and not seller_listing_status_at_week_start.get(seller_id, False)
      )
  ]
  seller_listing_tasks = {
      seller_id: functools.partial(
          portal.list_flat,
          seller,
          listing_price=_listing_price_for_seller(seller),
          week=week_number,
      )
      for seller_id, seller in eligible_sellers_to_list
  }
  listed_results, listing_errors = (
      concurrency.run_tasks_in_background(seller_listing_tasks)
      if seller_listing_tasks
      else ({}, {})
  )
  for seller_id, error in listing_errors.items():
    logging.error('Failed to list flat for seller %s: %s', seller_id, error)
  for seller_id, _ in eligible_sellers_to_list:
    listing_id = listed_results.get(seller_id)
    if listing_id is None:
      continue
    newly_listed_listing_ids.append(listing_id)

  eligible_buyers = [
      (buyer_id, buyer)
      for buyer_id, buyer in buyers.items()
      if buyer_id in assigned_ids and not portal.is_player_closed(buyer_id)
  ]
  buyer_tasks = {
      buyer_id: functools.partial(
          portal.search_and_request,
          buyer,
          week=week_number,
      )
      for buyer_id, buyer in eligible_buyers
  }
  buyer_results, buyer_errors = (
      concurrency.run_tasks_in_background(buyer_tasks) if buyer_tasks else ({}, {})
  )
  for buyer_id, error in buyer_errors.items():
    logging.error('Failed listing search/request for buyer %s: %s', buyer_id, error)

  for buyer_id, buyer in eligible_buyers:
    if buyer_results.get(buyer_id) is None:
      continue
    buyers_processed.append(buyer.name)

  eligible_sellers_to_review = [
      (seller_id, seller)
      for seller_id, seller in sellers.items()
      if (
          seller_id in assigned_ids
          and not portal.is_player_closed(seller_id)
          and seller_listing_status_at_week_start.get(seller_id, False)
      )
  ]
  seller_review_tasks = {
      seller_id: functools.partial(
          portal.review_requests_and_start_negotiation,
          seller,
          week=week_number,
      )
      for seller_id, seller in eligible_sellers_to_review
  }
  review_results, review_errors = (
      concurrency.run_tasks_in_background(seller_review_tasks)
      if seller_review_tasks
      else ({}, {})
  )
  for seller_id, error in review_errors.items():
    logging.error('Failed to review listing requests for seller %s: %s', seller_id, error)
  for seller_id, seller in eligible_sellers_to_review:
    sellers_reviewed.append(seller.name)
    result = review_results.get(seller_id)
    if result is None:
      continue
    matched_pairs.append(result)
    closed_player_names.extend((result.buyer_name, result.seller_name))

  return listing_schemas.ListingWeeklyBatchOutcome(
      week_number=week_number,
      active_player_names=list(active_player_names),
      newly_listed_listing_ids=newly_listed_listing_ids,
      buyers_processed=buyers_processed,
      sellers_reviewed=sellers_reviewed,
      matched_pairs=matched_pairs,
      closed_player_names=_dedupe_strings(closed_player_names),
  )


class ListingModule(action_spec_ignored.ActionSpecIgnored):
  """
  ListingModule manages the HDB listing portal simulation. 

  Responsibilities:
  - owns buyer/seller portal profiles
  - owns portal state across weeks
  - runs one listing week when asked by the coordinator
  - exposes a compact snapshot for inspection/serialization

  run_week runs one weekly listing-market step in three concurrent phases:
  1. Eligible sellers that have yet to activate their listings activate them in the portal.
  2. Buyers search the portal and submit at most one request.
  3. Sellers review their own request queues and start negotiation matches.

  run_week returns ListingWeeklyBatchOutcome.
  """

  def __init__(
      self,
      *,
      player_names: Sequence[str],
      player_ids: Sequence[str] | None = None,
      buyer_profiles: Mapping[str, Mapping[str, Any]] | str = (),
      seller_profiles: Mapping[str, Mapping[str, Any]] | str = (),
      client: Any | None = None,
      dense_embedding_model: Any | None = None,
      random_seed: int = 0,
      max_rounds: int | None = None,
      enabled: bool = True,
      pre_act_label: str = 'Listing module',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._enabled = bool(enabled)
    self._client = client
    self._dense_embedding_model = dense_embedding_model
    self._random_seed = random_seed
    self._player_names = tuple(player_names)
    self._player_ids = tuple(player_ids) if player_ids else tuple(player_names)
    if len(self._player_names) != len(self._player_ids):
      logging.error('ListingModule player_ids must align with player_names.')
      self._player_ids = tuple(player_names)

    self._id_to_name = dict(zip(self._player_ids, self._player_names))
    self._max_rounds = max_rounds if max_rounds and max_rounds > 0 else None
    self._completed_weeks = 0
    self._last_run_week = 0
    self._stage_exhausted = False
    self._last_outcome = listing_schemas.ListingWeeklyBatchOutcome(week_number=1)

    if isinstance(buyer_profiles, str):
      buyer_profiles = json.loads(buyer_profiles) if buyer_profiles else {}
    if isinstance(seller_profiles, str):
      seller_profiles = json.loads(seller_profiles) if seller_profiles else {}

    self._buyers = {
        buyer_id: listing_schemas.PortalBuyer.model_validate(
            {'id': buyer_id, **payload}
        )
        for buyer_id, payload in dict(buyer_profiles).items()
    }
    self._sellers = {
        seller_id: listing_schemas.PortalSeller.model_validate(
            {'id': seller_id, **payload}
        )
        for seller_id, payload in dict(seller_profiles).items()
    }
    self._portal_state: dict[str, Any] = {}
    self._portal: listing_portal_lib.ListingPortal | None = None
    if not self._enabled:
      return
    self._ensure_portal()

  def _ensure_portal(self) -> listing_portal_lib.ListingPortal:
    if self._portal is None:
      retriever = None
      if self._client is not None or self._dense_embedding_model is not None:
        retriever = listing_portal_lib.ListingPortalRetriever(
            client=self._client,
            dense_embedding_model=self._dense_embedding_model,
        )
      if self._portal_state:
        self._portal = listing_portal_lib.ListingPortal.from_state(
            self._portal_state,
            retriever=retriever,
            random_seed=self._random_seed,
        )
      else:
        self._portal = listing_portal_lib.ListingPortal(
            retriever=retriever,
            random_seed=self._random_seed,
        )
    return self._portal

  def set_enabled(self, enabled: bool) -> None:
    self._enabled = bool(enabled)

  def is_enabled(self) -> bool:
    return self._enabled

  def is_finished(self) -> bool:
    return not self._enabled or self._stage_exhausted or not self.get_open_player_ids()

  def get_open_player_ids(self) -> set[str]:
    if not self._enabled or self._stage_exhausted:
      return set()
    portal = self._ensure_portal()
    open_ids: set[str] = set()
    for buyer_id in self._buyers:
      if not portal.is_player_closed(buyer_id):
        open_ids.add(buyer_id)
    for seller_id in self._sellers:
      if not portal.is_player_closed(seller_id):
        open_ids.add(seller_id)
    return open_ids

  def get_player_ids(self) -> tuple[str, ...]:
    return self._player_ids

  def get_player_name(self, player_id: str) -> str:
    return self._id_to_name.get(str(player_id), str(player_id))

  def get_last_outcome(self) -> listing_schemas.ListingWeeklyBatchOutcome:
    return self._last_outcome

  def _empty_outcome(
      self,
      week_number: int,
      active_player_names: Sequence[str] = (),
  ) -> listing_schemas.ListingWeeklyBatchOutcome:
    outcome = listing_schemas.ListingWeeklyBatchOutcome(
        week_number=week_number,
        active_player_names=list(active_player_names),
    )
    self._last_outcome = outcome
    return outcome

  def run_week(
      self,
      *,
      week_number: int,
      assigned_player_ids: Sequence[str] = (),
  ) -> listing_schemas.ListingWeeklyBatchOutcome:
    """Runs one listing week for the assigned open players."""
    if not self._enabled:
      return self._empty_outcome(week_number)
    if self._max_rounds is not None and week_number > self._max_rounds:
      self._stage_exhausted = True
      return self._empty_outcome(week_number)
    portal = self._ensure_portal()

    open_player_ids = self.get_open_player_ids()
    assigned_ids = (
        set(str(player_id) for player_id in assigned_player_ids)
        if assigned_player_ids
        else set(open_player_ids)
    )
    active_player_ids = [
        player_id
        for player_id in self._player_ids
        if player_id in assigned_ids and player_id in open_player_ids
    ]
    active_player_names = [
        self._id_to_name.get(player_id, player_id) for player_id in active_player_ids
    ]
    if not active_player_names:
      return self._empty_outcome(week_number)
    outcome = execute_listing_week(
        portal=portal,
        buyers=self._buyers,
        sellers=self._sellers,
        week_number=week_number,
        active_player_ids=active_player_ids,
        active_player_names=active_player_names,
    )
    self._completed_weeks += 1
    self._last_run_week = week_number
    if self._max_rounds is not None and week_number >= self._max_rounds:
      self._stage_exhausted = True
    self._last_outcome = outcome
    return outcome

  def _buyer_state(self, player_id: str) -> listing_schemas.ListingBuyerState:
    buyer = self._buyers[player_id]
    portal = self._ensure_portal()
    return listing_schemas.ListingBuyerState(
        player_id=player_id,
        player_name=buyer.name,
        budget_min_price=buyer.budget.min_price,
        budget_max_price=buyer.budget.max_price,
        effective_reservation_price=portal.effective_reservation_price_for_buyer(
            buyer
        ),
        preferred_flat_types=list(buyer.preferences.flat_type),
        preferred_towns=list(buyer.preferences.towns),
        latest_search_results=list(portal.search_results_by_buyer.get(player_id, [])),
        latest_market_feedback=portal.market_feedback_by_buyer.get(
            player_id,
            'No market feedback yet.',
        ),
    )

  def _seller_state(self, player_id: str) -> listing_schemas.ListingSellerState:
    seller = self._sellers[player_id]
    portal = self._ensure_portal()
    listing_id = portal.listing_id_for_seller(player_id)
    listing = portal.get_listing_record(player_id)
    return listing_schemas.ListingSellerState(
        player_id=player_id,
        player_name=seller.name,
        listed=portal.is_seller_listed(player_id),
        current_listing_id=listing_id if listing is not None else None,
        current_listing_price=float(listing.listing_price) if listing is not None else None,
        open_requests=portal.pending_request_count(player_id),
        flat_type=str(seller.flat.flat_type),
        town=seller.flat.town,
    )

  def _make_pre_act_value(self) -> str:
    if self._portal is None:
      snapshot = listing_schemas.ListingPortalSnapshot(
          week_number=max(1, self._last_run_week or 1),
      )
      return snapshot.model_dump_json()

    snapshot = listing_schemas.ListingPortalSnapshot(
        week_number=max(1, self._last_run_week or 1),
        buyers=[
            self._buyer_state(buyer_id)
            for buyer_id in self._buyers
            if not self._portal.is_player_closed(buyer_id)
        ],
        sellers=[
            self._seller_state(seller_id)
            for seller_id in self._sellers
            if not self._portal.is_player_closed(seller_id)
        ],
        matched_pairs=list(self._portal.matched_pairs),
    )
    return snapshot.model_dump_json()

  def get_state(self) -> entity_component.ComponentState:
    return {
        'buyers': {
            buyer_id: buyer.model_dump()
            for buyer_id, buyer in self._buyers.items()
        },
        'sellers': {
            seller_id: seller.model_dump()
            for seller_id, seller in self._sellers.items()
        },
        'portal_state': (
            self._portal.export_state()
            if self._portal is not None
            else dict(self._portal_state)
        ),
        'max_rounds': self._max_rounds or 0,
        'enabled': int(self._enabled),
        'completed_weeks': self._completed_weeks,
        'last_run_week': self._last_run_week,
        'stage_exhausted': int(self._stage_exhausted),
        'last_outcome': self._last_outcome.model_dump(),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'portal_state' in state:
      self._portal_state = dict(state.get('portal_state', {}))
      self._portal = None
    if 'buyers' in state:
      self._buyers = {
          str(buyer_id): listing_schemas.PortalBuyer.model_validate(payload)
          for buyer_id, payload in state['buyers'].items()
      }
    if 'sellers' in state:
      self._sellers = {
          str(seller_id): listing_schemas.PortalSeller.model_validate(payload)
          for seller_id, payload in state['sellers'].items()
      }
    max_rounds = int(state.get('max_rounds', 0))
    self._max_rounds = max_rounds if max_rounds > 0 else None
    self._enabled = bool(state.get('enabled', 1))
    self._completed_weeks = int(state.get('completed_weeks', 0))
    self._last_run_week = int(state.get('last_run_week', 0))
    self._stage_exhausted = bool(state.get('stage_exhausted', 0))
    if 'last_outcome' in state:
      self._last_outcome = listing_schemas.ListingWeeklyBatchOutcome.model_validate(
          state['last_outcome']
      )
