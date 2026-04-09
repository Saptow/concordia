from collections.abc import Mapping, Sequence
import functools
import json
from typing import Any

from absl import logging
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.hdb_simulation import listing_portal as listing_portal_lib
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas
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


def _normalize_max_workers(value: object) -> int | None:
  """Parse worker counts while allowing `None` to mean executor default."""
  if value is None:
    return None
  try:
    parsed = int(value)
  except (TypeError, ValueError):
    return 1
  if parsed <= 0:
    return None
  return parsed


def _derive_failed_negotiation_learning_signal(
    *,
    payload: negotiation_schemas.NegotiationToListingPayload,
) -> tuple[float, float]:
  """Translate a failed negotiation into flat-specific learning evidence."""
  buyer_belief = payload.buyer_state.effective_reservation
  observation = max(0.0, float(buyer_belief.mean))
  confidence = max(0.05, min(1.0, float(buyer_belief.confidence)))
  evidence_count = max(0, int(buyer_belief.evidence_count))
  experience_factor = min(1.0, 0.5 + (0.1 * evidence_count))
  reliability = max(
      0.0,
      min(0.9, confidence * experience_factor),
  )
  return observation, reliability

def execute_listing_week(
    *,
    portal: listing_portal_lib.ListingPortal,
    buyers: Mapping[str, listing_schemas.PortalBuyer],
    sellers: Mapping[str, listing_schemas.PortalSeller],
    week_number: int,
    active_player_ids: Sequence[str],
    active_player_names: Sequence[str],
    seller_listing_max_workers: int | None = 1,
    buyer_search_max_workers: int | None = 1,
    seller_review_max_workers: int | None = 1,
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
      concurrency.run_tasks_in_background(
          seller_listing_tasks,
          max_workers=seller_listing_max_workers,
      )
      if seller_listing_tasks
      else ({}, {})
  )
  for seller_id, error in listing_errors.items():
    logging.error(
        'Listing week %s: failed to activate listing for seller %s: %s',
        week_number,
        seller_id,
        error,
    )
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
      concurrency.run_tasks_in_background(
          buyer_tasks,
          max_workers=buyer_search_max_workers,
      )
      if buyer_tasks
      else ({}, {})
  )
  for buyer_id, error in buyer_errors.items():
    logging.error(
        'Listing week %s: failed buyer %s portal search/request: %s',
        week_number,
        buyer_id,
        error,
    )

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
      concurrency.run_tasks_in_background(
          seller_review_tasks,
          max_workers=seller_review_max_workers,
      )
      if seller_review_tasks
      else ({}, {})
  )
  for seller_id, error in review_errors.items():
    logging.error(
        'Listing week %s: failed seller %s request review/start-negotiation: %s',
        week_number,
        seller_id,
        error,
    )
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
      sellers_without_match_count=max(
          0,
          len(eligible_sellers_to_review) - len(matched_pairs),
      ),
      closed_player_names=_dedupe_strings(closed_player_names),
  )


class ListingModule(action_spec_ignored.ActionSpecIgnored):
  """
  ListingModule manages the HDB listing portal simulation. 

  Responsibilities:
  - owns buyer/seller portal profiles
  - owns portal state across weeks
  - runs one listing week when asked by the coordinator
  - provides listing-to-negotiation transfer payloads for matched pairs
  - provides a method to reopen failed negotiation pairs back into the listing workflow
  - exposes a compact snapshot for inspection/serialization

  run_week runs one weekly listing-market step in three concurrent phases:
  1. Eligible sellers that have yet to activate their listings activate them in the portal.
  2. Buyers search the portal and submit at most one request.
  3. Sellers review their own request queues and start negotiation matches.

  run_week returns ListingWeeklyBatchOutcome.
  """

  @staticmethod
  def _normalize_market_state(value: object) -> str:
    return str(value or '').strip().casefold().replace('-', '_').replace(' ', '_')

  def __init__(
      self,
      *,
      player_names: Sequence[str],
      player_ids: Sequence[str] | None = None,
      buyer_profiles: Mapping[str, Mapping[str, Any]] | str = (),
      seller_profiles: Mapping[str, Mapping[str, Any]] | str = (),
      client: Any | None = None,
      dense_embedding_model: Any | None = None,
      sparse_embedding_model: Any | None = None,
      collection_name: str | None = None,
      db_path: str | None = None,
      random_seed: int = 0,
      max_rounds: int | None = None,
      seller_listing_max_workers: int | None = 1,
      buyer_search_max_workers: int | None = 1,
      seller_review_max_workers: int | None = 1,
      enabled: bool = True,
      pre_act_label: str = 'Listing module',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._enabled = bool(enabled)
    self._client = client
    self._dense_embedding_model = dense_embedding_model
    self._sparse_embedding_model = sparse_embedding_model
    self._collection_name = str(collection_name).strip() or qdrant_schemas.DEFAULT_COLLECTION_NAME
    self._db_path = str(db_path).strip() or qdrant_schemas.DEFAULT_DB_PATH
    self._random_seed = random_seed
    self._player_names = tuple(player_names)
    self._player_ids = tuple(player_ids) if player_ids else tuple(player_names)
    if len(self._player_names) != len(self._player_ids):
      logging.error('ListingModule player_ids must align with player_names.')
      self._player_ids = tuple(player_names)

    self._id_to_name = dict(zip(self._player_ids, self._player_names))
    self._max_rounds = max_rounds if max_rounds and max_rounds > 0 else None
    self._seller_listing_max_workers = _normalize_max_workers(
        seller_listing_max_workers
    )
    self._buyer_search_max_workers = _normalize_max_workers(
        buyer_search_max_workers
    )
    self._seller_review_max_workers = _normalize_max_workers(
        seller_review_max_workers
    )
    self._completed_weeks = 0
    self._last_run_week = 0
    self._stage_exhausted = False
    self._total_sellers_without_match = 0
    self._last_outcome = listing_schemas.ListingWeeklyBatchOutcome(week_number=1)
    self._entities_by_id: dict[str, Any] = {}
    self._canonical_entities: tuple[entity_component.EntityWithComponents, ...] = ()
    self._canonical_entities_by_name: dict[str, entity_component.EntityWithComponents] = {}
    self._active_seller_ids: set[str] = set()
    self._inactive_seller_queue: list[str] = []
    self._seller_release_week_by_id: dict[str, int] = {}
    self._target_active_seller_count = 0
    self._last_released_seller_ids: list[str] = []

    if isinstance(buyer_profiles, str):
      buyer_profiles = json.loads(buyer_profiles) if buyer_profiles else {}
    if isinstance(seller_profiles, str):
      seller_profiles = json.loads(seller_profiles) if seller_profiles else {}

    normalized_seller_profiles = {
        str(seller_id): dict(payload)
        for seller_id, payload in dict(seller_profiles).items()
    }
    self._seller_release_week_by_id = {
        seller_id: max(1, int(payload.get('listing_release_week', 1) or 1))
        for seller_id, payload in normalized_seller_profiles.items()
    }
    initially_listed_seller_ids = {
        seller_id
        for seller_id, payload in normalized_seller_profiles.items()
        if self._normalize_market_state(payload.get('initial_market_state'))
        == 'listed'
    }
    inactive_seller_records = sorted(
        (
            (
                self._seller_release_week_by_id.get(seller_id, 1),
                int(payload.get('initialization_order', 0) or 0),
                seller_id,
            )
            for seller_id, payload in normalized_seller_profiles.items()
            if self._normalize_market_state(payload.get('initial_market_state'))
            == 'not_yet_listed'
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )

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
        for seller_id, payload in normalized_seller_profiles.items()
    }
    self._inactive_seller_queue = [
        seller_id for _, _, seller_id in inactive_seller_records
    ]
    self._active_seller_ids = set(self._sellers) - set(self._inactive_seller_queue)
    self._target_active_seller_count = len(initially_listed_seller_ids)
    self._portal_state: dict[str, Any] = {}
    self._portal: listing_portal_lib.ListingPortal | None = None
    if not self._enabled:
      return
    self._ensure_portal()

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    super().set_entity(entity)
    self._bind_known_entities()

  def set_canonical_entities(
      self,
      entities: Sequence[entity_component.EntityWithComponents],
  ) -> None:
    """Registers the simulation-owned entities available to listing."""
    self._canonical_entities = tuple(entities)
    self._canonical_entities_by_name = {
        entity.name: entity for entity in self._canonical_entities
    }
    self._bind_known_entities()

  def _bind_known_entities(self) -> None:
    """Binds canonical entities to player ids using id first, then name."""
    if not self._canonical_entities:
      return
    for player_id in self._player_ids:
      self._bind_entity_for_player(player_id)

  def _bind_entity_for_player(self, player_id: str) -> bool:
    """Resolves a player id to its canonical simulation entity."""
    if player_id in self._entities_by_id:
      return True

    for entity in self._canonical_entities:
      entity_player_id = str(getattr(entity, '_hdb_player_id', '')).strip()
      if entity_player_id == player_id:
        self._entities_by_id[player_id] = entity
        return True

    expected_name = self._id_to_name.get(str(player_id), '')
    if expected_name:
      matching_entity = self._canonical_entities_by_name.get(expected_name)
      if matching_entity is not None:
        current_player_id = str(getattr(matching_entity, '_hdb_player_id', '')).strip()
        if current_player_id and current_player_id != player_id:
          logging.error(
              'Canonical entity %s is already bound to %s, cannot bind to %s.',
              matching_entity.name,
              current_player_id,
              player_id,
          )
          return False
        matching_entity._hdb_player_id = player_id
        self._entities_by_id[player_id] = matching_entity
        return True

    return False

  def _remember_listing_event(self, player_id: str, memory_text: str) -> None:
    """Adds a listing-stage memory entry for an active participant."""
    entity = self._entities_by_id.get(str(player_id))
    if entity is None and not self._bind_entity_for_player(str(player_id)):
      return
    entity = self._entities_by_id.get(str(player_id))
    if entity is None:
      return
    try:
      memory = entity.get_component(
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
          type_=memory_component.Memory,
      )
    except Exception:
      return
    memory.add(memory_text)

  def _buyer_listing_memory(
      self,
      *,
      buyer: listing_schemas.PortalBuyer,
      buyer_id: str,
      week_number: int,
      portal: listing_portal_lib.ListingPortal,
  ) -> str:
    results = list(portal.search_results_by_buyer.get(buyer_id, ()))
    top_result = results[0] if results else None
    requested_seller_id = str(top_result.seller_id) if top_result is not None else ''
    requested_seller_name = (
        self._id_to_name.get(requested_seller_id, requested_seller_id)
        if requested_seller_id
        else 'none'
    )
    listing_price = (
        f'SGD {float(top_result.listing_price):.2f}'
        if top_result is not None
        else 'NA'
    )
    return (
        f'[listing_action] Week {int(week_number)}: {buyer.name} searched the '
        f'listing portal as a buyer. Search results considered: {len(results)}. '
        f'Request submitted to seller: {requested_seller_name}. '
        f'Top matched listing price: {listing_price}.'
    )

  def _seller_listing_memory(
      self,
      *,
      seller: listing_schemas.PortalSeller,
      seller_id: str,
      week_number: int,
      portal: listing_portal_lib.ListingPortal,
      listed_this_week: bool,
      reviewed_this_week: bool,
  ) -> str:
    listing_record = portal.get_listing_record(seller_id)
    listing_price = (
        f'SGD {float(listing_record.listing_price):.2f}'
        if listing_record is not None
        else 'NA'
    )
    open_requests = portal.pending_request_count(seller_id)
    if listed_this_week:
      return (
          f'[listing_action] Week {int(week_number)}: {seller.name} listed their '
          f'flat on the portal as a seller. Listing price: {listing_price}. '
          f'Open buyer requests after listing: {open_requests}.'
      )
    if reviewed_this_week:
      return (
          f'[listing_action] Week {int(week_number)}: {seller.name} kept their '
          f'flat listed on the portal and reviewed buyer requests as a seller. '
          f'Current listing price: {listing_price}. '
          f'Remaining open requests after review: {open_requests}.'
      )
    return (
        f'[listing_action] Week {int(week_number)}: {seller.name} remained active '
        f'in the listing portal as a seller. Current listing price: {listing_price}.'
    )

  def _ensure_portal(self) -> listing_portal_lib.ListingPortal:
    """Lazily constructs or restores the backing `ListingPortal` instance."""
    if self._portal is None:
      retriever = None
      if (
          self._client is not None
          or self._dense_embedding_model is not None
          or self._sparse_embedding_model is not None
      ):
        retriever = listing_portal_lib.ListingPortalRetriever(
            client=self._client,
            dense_embedding_model=self._dense_embedding_model,
            sparse_embedding_model=self._sparse_embedding_model,
            collection_name=(
                self._collection_name
                or listing_portal_lib.qdrant_schemas.DEFAULT_COLLECTION_NAME
            ),
            db_path=self._db_path or listing_portal_lib.qdrant_schemas.DEFAULT_DB_PATH,
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

  def _release_inactive_sellers_for_week(self, week_number: int) -> list[str]:
    """Release every seller batch whose configured month has entered the window."""
    if not self._inactive_seller_queue:
      self._last_released_seller_ids = []
      return []

    released: list[str] = []
    while self._inactive_seller_queue:
      seller_id = self._inactive_seller_queue[0]
      release_week = self._seller_release_week_by_id.get(seller_id, 1)
      if release_week > week_number:
        break
      self._inactive_seller_queue.pop(0)
      self._active_seller_ids.add(seller_id)
      released.append(seller_id)
    self._last_released_seller_ids = released
    return released

  def prepare_weekly_market(self, *, week_number: int | None = None) -> list[str]:
    """Refresh listing availability before assignments are computed."""
    if not self._enabled:
      self._last_released_seller_ids = []
      return []
    normalized_week = max(1, int(week_number or 1))
    return self._release_inactive_sellers_for_week(normalized_week)

  def get_open_player_ids(self) -> set[str]:
    if not self._enabled or self._stage_exhausted:
      return set()
    portal = self._ensure_portal()
    open_ids: set[str] = set()
    for buyer_id in self._buyers:
      if not portal.is_player_closed(buyer_id):
        open_ids.add(buyer_id)
    for seller_id in self._active_seller_ids:
      if not portal.is_player_closed(seller_id):
        open_ids.add(seller_id)
    return open_ids

  def get_player_ids(self) -> tuple[str, ...]:
    return self._player_ids

  def get_player_name(self, player_id: str) -> str:
    return self._id_to_name.get(str(player_id), str(player_id))

  def get_last_outcome(self) -> listing_schemas.ListingWeeklyBatchOutcome:
    return self._last_outcome


  def build_negotiation_transfer_payloads(
      self,
      matches: Sequence[listing_schemas.NegotiationMatch],
  ) -> list[dict[str, Any]]:
    """Builds listing-to-negotiation payloads for newly matched pairs."""
    payloads: list[dict[str, Any]] = []
    for match in matches:
      buyer_id = str(match.buyer_id)
      seller_id = str(match.seller_id)
      listing_record = self._ensure_portal().get_listing_record(seller_id)
      if listing_record is None:
        logging.warning(
            'Skipping listing-to-negotiation transfer for %s because seller %s has no listing record in the portal.',
            match.match_id,
            seller_id,
        )
        continue
      payload = negotiation_schemas.ListingNegotiationTransferPayload(
          match_id=match.match_id,
          week_matched=match.week_matched,
          listing_record=listing_record,
          buyer_state=self._buyer_state(buyer_id),
          seller_state=self._seller_state(seller_id),
      )
      payloads.append(payload.model_dump(mode='json'))
    return payloads

  def reopen_failed_negotiation_pairs(
      self,
      pair_records: Sequence[Mapping[str, Any]],
  ) -> list[dict[str, Any]]:
    """Reopens failed negotiation pairs back into the listing workflow.

    Each reopened pair removes the buyer and seller from the portal's closed
    participant sets. The seller's listing stays inactive until a later
    listing week relists it through the normal `list_flat` flow.

    Args:
      pair_records: Closed-pair summary records, typically from negotiation.

    Returns:
      Normalized relisting payloads for pairs that were successfully reopened.
    """
    portal = self._ensure_portal()
    reopened_pairs: list[dict[str, Any]] = []
    for pair_record in pair_records:
      try:
        payload = negotiation_schemas.NegotiationToListingPayload.model_validate(
            pair_record
        )
      except Exception as error:  # pylint: disable=broad-exception-caught
        logging.warning(
            'Skipping invalid negotiation-to-listing payload %s: %s',
            pair_record,
            error,
        )
        continue
      buyer_id = str(payload.negotiation_history.buyer_id).strip()
      seller_id = str(payload.negotiation_history.seller_id).strip()
      if not buyer_id or not seller_id:
        logging.warning(
            'Skipping failed negotiation reopen with invalid ids: %s',
            pair_record,
        )
        continue
      buyer = self._buyers.get(buyer_id)
      seller = self._sellers.get(seller_id)
      if buyer is None or seller is None:
        logging.warning(
            'Skipping failed negotiation reopen with unknown ids: %s',
            payload.negotiation_history,
        )
        continue
      buyer.negotiation_history.append(payload.negotiation_history.model_copy(deep=True))
      seller.negotiation_history.append(payload.negotiation_history.model_copy(deep=True))
      buyer_market_state = portal._buyer_market_state(buyer)
      observation, reliability = (
          _derive_failed_negotiation_learning_signal(
              payload=payload,
          )
      )
      if reliability > 0.0:
        buyer_market_state.effective_reservation.update_with_evidence(
            observation,
            reliability=reliability,
        )
        buyer_market_state.effective_reservation.mean = max(
            float(buyer.budget.min_price),
            min(
                float(buyer.budget.max_price),
                float(buyer_market_state.effective_reservation.mean),
            ),
        )
        buyer_market_state.latest_market_feedback = (
            'Updated reservation after a failed negotiation using '
            f'flat-specific evidence near SGD {observation:.0f}.'
        )
        if (
            not buyer_market_state.feedback_history
            or buyer_market_state.feedback_history[-1]
            != buyer_market_state.latest_market_feedback
        ):
          buyer_market_state.feedback_history.append(
              buyer_market_state.latest_market_feedback
          )
      portal._seller_market_state(seller).effective_reservation = (
          payload.seller_state.effective_reservation.model_copy(deep=True)
      )
      portal.closed_buyers.discard(buyer_id)
      portal.closed_sellers.discard(seller_id)
      reopened_pairs.append(payload.model_dump(mode='json'))
    return reopened_pairs

  def _empty_outcome(
      self,
      week_number: int,
      active_player_names: Sequence[str] = (),
  ) -> listing_schemas.ListingWeeklyBatchOutcome:
    avg_sellers_without_match_per_week = (
        self._total_sellers_without_match / self._completed_weeks
        if self._completed_weeks > 0
        else 0.0
    )
    outcome = listing_schemas.ListingWeeklyBatchOutcome(
        week_number=week_number,
        active_player_names=list(active_player_names),
        avg_sellers_without_match_per_week=avg_sellers_without_match_per_week,
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
        seller_listing_max_workers=self._seller_listing_max_workers,
        buyer_search_max_workers=self._buyer_search_max_workers,
        seller_review_max_workers=self._seller_review_max_workers,
    )
    newly_listed_ids = set(outcome.newly_listed_listing_ids)
    reviewed_seller_names = set(outcome.sellers_reviewed)
    processed_buyer_names = set(outcome.buyers_processed)
    for player_id in active_player_ids:
      if player_id in self._buyers:
        buyer = self._buyers[player_id]
        if buyer.name in processed_buyer_names:
          self._remember_listing_event(
              player_id,
              self._buyer_listing_memory(
                  buyer=buyer,
                  buyer_id=player_id,
                  week_number=week_number,
                  portal=portal,
              ),
          )
        continue
      if player_id in self._sellers:
        seller = self._sellers[player_id]
        listing_id = portal.listing_id_for_seller(player_id)
        self._remember_listing_event(
            player_id,
            self._seller_listing_memory(
                seller=seller,
                seller_id=player_id,
                week_number=week_number,
                portal=portal,
                listed_this_week=bool(listing_id and listing_id in newly_listed_ids),
                reviewed_this_week=seller.name in reviewed_seller_names,
            ),
        )
    self._completed_weeks += 1
    self._total_sellers_without_match += int(outcome.sellers_without_match_count)
    avg_sellers_without_match_per_week = (
        self._total_sellers_without_match / self._completed_weeks
        if self._completed_weeks > 0
        else 0.0
    )
    self._last_run_week = week_number
    if self._max_rounds is not None and week_number >= self._max_rounds:
      self._stage_exhausted = True
    outcome = outcome.model_copy(
        update={
            'avg_sellers_without_match_per_week': (
                avg_sellers_without_match_per_week
            ),
        }
    )
    self._last_outcome = outcome
    return outcome

  def _buyer_state(self, player_id: str) -> negotiation_schemas.ListingBuyerState:
    """Builds a runtime listing snapshot for one buyer."""
    buyer = self._buyers[player_id]
    portal = self._ensure_portal()
    market_state = portal._buyer_market_state(buyer)
    return negotiation_schemas.ListingBuyerState(
        id=player_id,
        name=buyer.name,
        role=buyer.role,
        description=buyer.description,
        budget=buyer.budget.model_copy(deep=True),
        preferences=buyer.preferences.model_copy(deep=True),
        negotiation_history=[
            record.model_copy(deep=True) for record in buyer.negotiation_history
        ],
        effective_reservation=market_state.effective_reservation,
        latest_search_results=list(portal.search_results_by_buyer.get(player_id, [])),
        latest_market_feedback=portal.market_feedback_by_buyer.get(
            player_id,
            'No market feedback yet.',
        ),
    )

  def _seller_state(self, player_id: str) -> negotiation_schemas.ListingSellerState:
    """Builds a runtime listing snapshot for one seller."""
    seller = self._sellers[player_id]
    portal = self._ensure_portal()
    market_state = portal._seller_market_state(seller)
    listing_id = portal.listing_id_for_seller(player_id)
    listing = portal.get_listing_record(player_id)
    return negotiation_schemas.ListingSellerState(
        id=player_id,
        name=seller.name,
        role=seller.role,
        description=seller.description,
        flat=seller.flat.model_copy(deep=True),
        expectations=seller.expectations.model_copy(deep=True),
        negotiation_history=[
            record.model_copy(deep=True) for record in seller.negotiation_history
        ],
        effective_reservation=market_state.effective_reservation.model_copy(deep=True),
        listed=portal.is_seller_listed(player_id),
        current_listing_id=listing_id if listing is not None else None,
        current_listing_price=float(listing.listing_price) if listing is not None else None,
        open_requests=portal.pending_request_count(player_id),
    )

  def _make_pre_act_value(self) -> str:
    """Returns a compact JSON snapshot for GM inspection and logging."""
    if self._portal is None:
      snapshot = negotiation_schemas.ListingPortalSnapshot(
          week_number=max(1, self._last_run_week or 1),
      )
      return snapshot.model_dump_json()

    snapshot = negotiation_schemas.ListingPortalSnapshot(
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

  def get_market_snapshot(
      self,
      player_ids: Sequence[str] | None = None,
  ) -> dict[str, Any]:
    """Returns listing-state data formatted for weekly HTML logging."""
    week_number = max(1, self._last_run_week or 1)
    if not self._enabled:
      return {
          'week_number': week_number,
          'buyers': [],
          'listed_sellers': [],
      }

    portal = self._ensure_portal()
    requested_ids = (
        {str(player_id) for player_id in player_ids}
        if player_ids is not None
        else None
    )

    buyers: list[dict[str, Any]] = []
    for buyer_id in self._buyers:
      if requested_ids is not None and buyer_id not in requested_ids:
        continue
      if portal.is_player_closed(buyer_id):
        continue
      buyers.append(self._buyer_state(buyer_id).model_dump(mode='json'))

    listed_sellers: list[dict[str, Any]] = []
    for seller_id in self._sellers:
      if requested_ids is not None and seller_id not in requested_ids:
        continue
      if portal.is_player_closed(seller_id):
        continue
      seller_state = self._seller_state(seller_id)
      if seller_state.listed:
        listed_sellers.append(seller_state.model_dump(mode='json'))

    return {
        'week_number': week_number,
        'buyers': buyers,
        'listed_sellers': listed_sellers,
        'released_seller_ids': list(self._last_released_seller_ids),
        'inactive_seller_ids': list(self._inactive_seller_queue),
        'active_seller_ids': sorted(self._active_seller_ids),
        'sellers_without_match_count': int(
            self._last_outcome.sellers_without_match_count
        ),
        'avg_sellers_without_match_per_week': float(
            self._last_outcome.avg_sellers_without_match_per_week
        ),
    }

  def get_state(self) -> entity_component.ComponentState:
    """Serializes listing-owned profiles, portal state, and progress counters."""
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
        'collection_name': self._collection_name or '',
        'db_path': self._db_path or '',
        'max_rounds': self._max_rounds or 0,
        'enabled': int(self._enabled),
        'completed_weeks': self._completed_weeks,
        'last_run_week': self._last_run_week,
        'stage_exhausted': int(self._stage_exhausted),
        'total_sellers_without_match': self._total_sellers_without_match,
        'last_outcome': self._last_outcome.model_dump(),
        'active_seller_ids': sorted(self._active_seller_ids),
        'inactive_seller_queue': list(self._inactive_seller_queue),
        'seller_release_week_by_id': dict(self._seller_release_week_by_id),
        'target_active_seller_count': self._target_active_seller_count,
        'last_released_seller_ids': list(self._last_released_seller_ids),
    }

  def get_dynamic_state(self) -> entity_component.ComponentState:
    """Serializes only the lightweight mutable progress fields."""
    return {
        'enabled': int(self._enabled),
        'collection_name': self._collection_name or '',
        'db_path': self._db_path or '',
        'max_rounds': self._max_rounds or 0,
        'completed_weeks': self._completed_weeks,
        'last_run_week': self._last_run_week,
        'stage_exhausted': int(self._stage_exhausted),
        'active_seller_count': len(self._active_seller_ids),
        'inactive_seller_count': len(self._inactive_seller_queue),
        'target_active_seller_count': self._target_active_seller_count,
        'total_sellers_without_match': self._total_sellers_without_match,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Restores listing-owned state and defers portal reconstruction lazily."""
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
    if 'collection_name' in state:
      self._collection_name = str(state.get('collection_name', '')).strip() or None
    if 'db_path' in state:
      self._db_path = str(state.get('db_path', '')).strip() or None
    max_rounds = int(state.get('max_rounds', 0))
    self._max_rounds = max_rounds if max_rounds > 0 else None
    self._enabled = bool(state.get('enabled', 1))
    self._completed_weeks = int(state.get('completed_weeks', 0))
    self._last_run_week = int(state.get('last_run_week', 0))
    self._stage_exhausted = bool(state.get('stage_exhausted', 0))
    self._total_sellers_without_match = int(
        state.get('total_sellers_without_match', 0)
    )
    self._active_seller_ids = {
        str(seller_id) for seller_id in state.get('active_seller_ids', ())
    } or set(self._sellers)
    self._inactive_seller_queue = [
        str(seller_id) for seller_id in state.get('inactive_seller_queue', ())
    ]
    self._seller_release_week_by_id = {
        str(seller_id): max(1, int(release_week or 1))
        for seller_id, release_week in dict(
            state.get('seller_release_week_by_id', {})
        ).items()
    }
    self._target_active_seller_count = int(
        state.get('target_active_seller_count', self._target_active_seller_count)
    )
    self._last_released_seller_ids = [
        str(seller_id) for seller_id in state.get('last_released_seller_ids', ())
    ]
    if 'last_outcome' in state:
      self._last_outcome = listing_schemas.ListingWeeklyBatchOutcome.model_validate(
          state['last_outcome']
      )
