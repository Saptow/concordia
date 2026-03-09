"""Game-master components for the HDB listing portal stage."""

from collections.abc import Mapping, Sequence
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.components.game_master import make_observation as make_observation_component
from concordia.components.game_master import next_acting as next_acting_component
from concordia.environment import engine as engine_lib
from concordia.hdb_simulation import listing_portal as listing_portal_lib
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

_WEEKLY_BATCH_ACTION = 'PROCESS_WEEKLY_LISTING_BATCH'


class ListingBatchScheduler(next_acting_component.NextActing):
  """Scheduler that activates every open listing participant in the same week."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      player_ids: Sequence[str] | None = None,
      max_rounds: int | None = None,
      pre_act_label: str = next_acting_component.DEFAULT_NEXT_ACTING_PRE_ACT_LABEL,
  ):
    super().__init__(
        model=model,
        player_names=player_names,
        components=(),
        pre_act_label=pre_act_label,
    )
    if not player_names:
      raise ValueError('No player names provided to the listing scheduler.')

    self._player_names = tuple(player_names)
    self._player_ids = tuple(player_ids) if player_ids else tuple(player_names)
    if len(self._player_ids) != len(self._player_names):
      raise ValueError('player_ids must align with player_names.')

    self._id_to_name = dict(zip(self._player_ids, self._player_names))
    self._name_to_id = dict(zip(self._player_names, self._player_ids))
    self._max_rounds = max_rounds if max_rounds and max_rounds > 0 else None

    self._week_number = 1
    self._completed_weeks = 0
    self._currently_active_player_ids: tuple[str, ...] = ()
    self._closed_player_ids: set[str] = set()

  def _to_player_id(self, player_token: str) -> str:
    if player_token in self._id_to_name:
      return player_token
    if player_token in self._name_to_id:
      return self._name_to_id[player_token]
    raise ValueError(f'Unknown player token: {player_token}')

  def is_player_closed(self, player_token: str) -> bool:
    try:
      player_id = self._to_player_id(player_token)
    except ValueError:
      return True
    return player_id in self._closed_player_ids

  def close_players(self, player_tokens: Sequence[str]) -> None:
    for player_token in player_tokens:
      try:
        player_id = self._to_player_id(player_token)
      except ValueError:
        continue
      self._closed_player_ids.add(player_id)

  def _close_all_remaining_players(self) -> None:
    self._closed_player_ids.update(self._player_ids)

  def all_players_closed(self) -> bool:
    return len(self._closed_player_ids) >= len(self._player_ids)

  def get_open_player_names(self) -> list[str]:
    return [
        self._id_to_name[player_id]
        for player_id in self._player_ids
        if player_id not in self._closed_player_ids
    ]

  def get_currently_active_players(self) -> list[str]:
    return [
        self._id_to_name[player_id]
        for player_id in self._currently_active_player_ids
        if player_id in self._id_to_name
    ]

  def get_current_week_number(self) -> int:
    return self._week_number

  def get_scheduler_snapshot(self) -> listing_schemas.ListingSchedulerSnapshot:
    return listing_schemas.ListingSchedulerSnapshot(
        week_number=self._week_number,
        active_player_names=self.get_currently_active_players(),
        completed_weeks=self._completed_weeks,
        closed_player_count=len(self._closed_player_ids),
        open_player_count=len(self._player_ids) - len(self._closed_player_ids),
        max_rounds=self._max_rounds or 0,
    )

  def mark_week_complete(self, players_to_close: Sequence[str] = ()) -> None:
    self.close_players(players_to_close)
    self._completed_weeks += 1
    self._currently_active_player_ids = ()
    if self.all_players_closed():
      return

    next_week_number = self._week_number + 1
    if self._max_rounds is not None and next_week_number > self._max_rounds:
      self._close_all_remaining_players()
      return
    self._week_number = next_week_number

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTING:
      return ''
    if self.all_players_closed():
      self._currently_active_player_ids = ()
      return ''

    open_player_ids = tuple(
        player_id
        for player_id in self._player_ids
        if player_id not in self._closed_player_ids
    )
    self._currently_active_player_ids = open_player_ids
    return ','.join(self._id_to_name[player_id] for player_id in open_player_ids)

  def get_state(self) -> entity_component.ComponentState:
    return {
        'player_names': list(self._player_names),
        'player_ids': list(self._player_ids),
        'week_number': self._week_number,
        'completed_weeks': self._completed_weeks,
        'currently_active_player_ids': list(self._currently_active_player_ids),
        'closed_player_ids': sorted(self._closed_player_ids),
        'max_rounds': self._max_rounds or 0,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    self._week_number = int(state.get('week_number', 1))
    self._completed_weeks = int(state.get('completed_weeks', 0))
    self._currently_active_player_ids = tuple(
        str(player_id) for player_id in state.get('currently_active_player_ids', [])
    )
    self._closed_player_ids = {
        str(player_id) for player_id in state.get('closed_player_ids', [])
    }
    max_rounds = int(state.get('max_rounds', 0))
    self._max_rounds = max_rounds if max_rounds > 0 else None


class PortalWeekStateTracker(action_spec_ignored.ActionSpecIgnored):
  """Prompt-context view over the weekly listing scheduler."""

  def __init__(
      self,
      scheduler_component_key: str = next_acting_component.DEFAULT_NEXT_ACTING_COMPONENT_KEY,
      pre_act_label: str = 'Listing portal weekly scheduler',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._scheduler_component_key = scheduler_component_key
    self._scheduler: ListingBatchScheduler | None = None

  def _get_scheduler(self) -> ListingBatchScheduler:
    if self._scheduler is None:
      self._scheduler = self.get_entity().get_component(
          self._scheduler_component_key,
          type_=ListingBatchScheduler,
      )
    return self._scheduler

  def _make_pre_act_value(self) -> str:
    return self._get_scheduler().get_scheduler_snapshot().model_dump_json()


class ListingPortalTracker(action_spec_ignored.ActionSpecIgnored):
  """Tracks listing-portal state and executes one deterministic week at a time."""

  def __init__(
      self,
      buyer_profiles: Mapping[str, Mapping[str, Any]],
      seller_profiles: Mapping[str, Mapping[str, Any]],
      scheduler_component_key: str = next_acting_component.DEFAULT_NEXT_ACTING_COMPONENT_KEY,
      random_seed: int = 0,
      pre_act_label: str = 'Listing portal state',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._scheduler_component_key = scheduler_component_key
    self._scheduler: ListingBatchScheduler | None = None
    self._buyers = {
        buyer_id: listing_schemas.PortalBuyer.model_validate(
            {'id': buyer_id, **payload}
        )
        for buyer_id, payload in buyer_profiles.items()
    }
    self._sellers = {
        seller_id: listing_schemas.PortalSeller.model_validate(
            {'id': seller_id, **payload}
        )
        for seller_id, payload in seller_profiles.items()
    }
    self._portal = listing_portal_lib.ListingPortal(random_seed=random_seed)

  def _get_scheduler(self) -> ListingBatchScheduler:
    if self._scheduler is None:
      self._scheduler = self.get_entity().get_component(
          self._scheduler_component_key,
          type_=ListingBatchScheduler,
      )
    return self._scheduler

  def _current_week(self) -> int:
    return self._get_scheduler().get_current_week_number()

  def _is_buyer_closed(self, buyer_id: str) -> bool:
    return self._portal.is_player_closed(buyer_id) or self._get_scheduler().is_player_closed(
        buyer_id
    )

  def _is_seller_closed(self, seller_id: str) -> bool:
    return self._portal.is_player_closed(seller_id) or self._get_scheduler().is_player_closed(
        seller_id
    )

  @staticmethod
  def _notification_models(
      notifications: Sequence[tuple[str, str]],
  ) -> list[listing_schemas.PortalNotification]:
    return [
        listing_schemas.PortalNotification(player_name=player_name, message=message)
        for player_name, message in notifications
    ]

  @staticmethod
  def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
      key = str(value).strip()
      if not key or key in seen:
        continue
      seen.add(key)
      ordered.append(key)
    return ordered

  @staticmethod
  def _listing_summary_for_seller(seller: listing_schemas.PortalSeller) -> str:
    return (seller.description or seller.flat.description or '').strip()

  @staticmethod
  def _listing_price_for_seller(seller: listing_schemas.PortalSeller) -> float:
    min_price = float(seller.expectations.min_price)
    max_price = float(seller.expectations.max_price)
    return max(min_price, max_price)

  def _search_results_for_buyer(
      self,
      buyer: listing_schemas.PortalBuyer,
  ) -> list[listing_schemas.PortalSearchResult]:
    search_query = self._portal._derive_query_from_preferences(buyer, '')
    return self._portal.retriever.search(
        search_query,
        preferred_flat_types=buyer.preferences.flat_type,
        preferred_towns=buyer.preferences.towns,
        max_budget=buyer.budget.max_price,
        limit=5,
    )

  def _requested_listing_ids_for_buyer(
      self,
      buyer: listing_schemas.PortalBuyer,
      results: Sequence[listing_schemas.PortalSearchResult],
  ) -> list[str]:
    requested_ids: list[str] = []
    max_budget = float(buyer.budget.max_price)
    for result in results:
      if result.listing_price > max_budget:
        continue
      if result.score <= 0.0:
        continue
      requested_ids.append(result.listing_id)
      if len(requested_ids) >= 3:
        break
    return requested_ids

  def execute_week(self) -> listing_schemas.ListingWeeklyBatchOutcome:
    scheduler = self._get_scheduler()
    week_number = self._current_week()
    active_player_names = scheduler.get_open_player_names()
    if not active_player_names:
      return listing_schemas.ListingWeeklyBatchOutcome(
          week_number=week_number,
          active_player_names=[],
      )

    seller_listing_status_at_week_start = {
        seller_id: self._portal.is_seller_listed(seller_id)
        for seller_id in self._sellers
        if not self._is_seller_closed(seller_id)
    }

    notifications: list[listing_schemas.PortalNotification] = []
    newly_listed_listing_ids: list[str] = []
    buyers_processed: list[str] = []
    sellers_reviewed: list[str] = []
    matched_pairs: list[listing_schemas.NegotiationMatch] = []
    closed_player_names: list[str] = []

    for seller_id, seller in self._sellers.items():
      if self._is_seller_closed(seller_id):
        continue
      if seller_listing_status_at_week_start.get(seller_id, False):
        continue
      result = self._portal.list_flat(
          seller,
          listing_price=self._listing_price_for_seller(seller),
          listing_summary=self._listing_summary_for_seller(seller),
          week=week_number,
      )
      newly_listed_listing_ids.append(result.listing_id)
      notifications.extend(self._notification_models(result.notifications))

    for buyer_id, buyer in self._buyers.items():
      if self._is_buyer_closed(buyer_id):
        continue
      preview_results = self._search_results_for_buyer(buyer)
      requested_listing_ids = self._requested_listing_ids_for_buyer(
          buyer,
          preview_results,
      )
      result = self._portal.search_and_request(
          buyer,
          search_query='',
          requested_listing_ids=requested_listing_ids,
          market_valuation_notes='',
          week=week_number,
      )
      buyers_processed.append(buyer.name)
      notifications.extend(self._notification_models(result.notifications))

    for seller_id, seller in self._sellers.items():
      if self._is_seller_closed(seller_id):
        continue
      if not seller_listing_status_at_week_start.get(seller_id, False):
        continue
      result = self._portal.review_requests_and_start_negotiation(
          seller,
          buyer_registry=self._buyers,
          week=week_number,
      )
      sellers_reviewed.append(seller.name)
      notifications.extend(self._notification_models(result.notifications))
      if result.match is None:
        continue
      matched_pairs.append(result.match)
      closed_player_names.extend((result.match.buyer_name, result.match.seller_name))

    closed_player_names = self._dedupe(closed_player_names)
    scheduler.mark_week_complete(players_to_close=closed_player_names)

    return listing_schemas.ListingWeeklyBatchOutcome(
      week_number=week_number,
      active_player_names=active_player_names,
      newly_listed_listing_ids=newly_listed_listing_ids,
      buyers_processed=buyers_processed,
      sellers_reviewed=sellers_reviewed,
      matched_pairs=matched_pairs,
      closed_player_names=closed_player_names,
      notifications=notifications,
    )

  def all_players_closed(self) -> bool:
    return self._get_scheduler().all_players_closed()

  def get_portal_pairs(self) -> list[dict[str, Any]]:
    return [match.model_dump() for match in self._portal.matched_pairs]

  def _buyer_state(self, player_id: str) -> listing_schemas.ListingBuyerState:
    buyer = self._buyers[player_id]
    return listing_schemas.ListingBuyerState(
        player_id=player_id,
        player_name=buyer.name,
        budget_min_price=buyer.budget.min_price,
        budget_max_price=buyer.budget.max_price,
        preferred_flat_types=list(buyer.preferences.flat_type),
        preferred_towns=list(buyer.preferences.towns),
        latest_search_results=list(
            self._portal.search_results_by_buyer.get(player_id, [])
        ),
        latest_market_feedback=self._portal.market_feedback_by_buyer.get(
            player_id,
            'No market feedback yet.',
        ),
    )

  def _seller_state(self, player_id: str) -> listing_schemas.ListingSellerState:
    seller = self._sellers[player_id]
    listing_id = self._portal.listing_id_for_seller(player_id)
    listing = self._portal.listings.get(listing_id)
    return listing_schemas.ListingSellerState(
        player_id=player_id,
        player_name=seller.name,
        listed=self._portal.is_seller_listed(player_id),
        current_listing_id=listing_id if listing is not None else None,
        current_listing_price=(
            float(listing.listing_price) if listing is not None else None
        ),
        open_requests=self._portal.pending_request_count(player_id),
        flat_type=str(seller.flat.flat_type),
        town=seller.flat.town,
    )

  def _make_pre_act_value(self) -> str:
    snapshot = listing_schemas.ListingPortalSnapshot(
        week_number=self._current_week(),
        buyers=[
            self._buyer_state(buyer_id)
            for buyer_id in self._buyers
            if not self._is_buyer_closed(buyer_id)
        ],
        sellers=[
            self._seller_state(seller_id)
            for seller_id in self._sellers
            if not self._is_seller_closed(seller_id)
        ],
        matched_pairs=list(self._portal.matched_pairs),
    )
    return snapshot.model_dump_json()

  def get_state(self) -> entity_component.ComponentState:
    return {
        'scheduler_component_key': self._scheduler_component_key,
        'buyers': {
            buyer_id: buyer.model_dump()
            for buyer_id, buyer in self._buyers.items()
        },
        'sellers': {
            seller_id: seller.model_dump()
            for seller_id, seller in self._sellers.items()
        },
        'portal_state': self._portal.export_state(),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'scheduler_component_key' in state:
      self._scheduler_component_key = str(state['scheduler_component_key'])
      self._scheduler = None
    if 'portal_state' in state:
      self._portal = listing_portal_lib.ListingPortal.from_state(
          state['portal_state'],
      )


class PortalBatchActionSpec(entity_component.ContextComponent):
  """Single fixed action spec used to trigger a weekly listing batch."""

  def __init__(
      self,
      call_to_action: str = 'Acknowledge the weekly listing-portal batch step.',
  ):
    super().__init__()
    self._call_to_action = call_to_action

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.NEXT_ACTION_SPEC:
      return ''
    next_spec = entity_lib.choice_action_spec(
        call_to_action=self._call_to_action,
        options=(_WEEKLY_BATCH_ACTION,),
    )
    return engine_lib.action_spec_to_string(next_spec)


class TerminateWhenPortalClosed(entity_component.ContextComponent):
  """Terminates once the weekly listing workflow has no open participants."""

  def __init__(
      self,
      portal_tracker_component_key: str = 'listing_portal_state',
      scheduler_component_key: str = next_acting_component.DEFAULT_NEXT_ACTING_COMPONENT_KEY,
  ):
    super().__init__()
    self._portal_tracker_component_key = portal_tracker_component_key
    self._scheduler_component_key = scheduler_component_key

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.TERMINATE:
      return ''
    tracker = self.get_entity().get_component(
        self._portal_tracker_component_key,
        type_=ListingPortalTracker,
    )
    scheduler = self.get_entity().get_component(
        self._scheduler_component_key,
        type_=ListingBatchScheduler,
    )
    should_terminate = tracker.all_players_closed() or scheduler.all_players_closed()
    return 'Yes' if should_terminate else 'No'


class PortalBatchResolution(entity_component.ContextComponent):
  """Executes one listing week and routes resulting notifications."""

  def __init__(
      self,
      make_observation_component_key: str = make_observation_component.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY,
      portal_tracker_component_key: str = 'listing_portal_state',
  ):
    super().__init__()
    self._make_observation_component_key = make_observation_component_key
    self._portal_tracker_component_key = portal_tracker_component_key

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if action_spec.output_type != entity_lib.OutputType.RESOLVE:
      return ''
    tracker = self.get_entity().get_component(
        self._portal_tracker_component_key,
        type_=ListingPortalTracker,
    )
    outcome = tracker.execute_week()
    make_observation = self.get_entity().get_component(
        self._make_observation_component_key,
        type_=make_observation_component.MakeObservation,
    )
    for notification in outcome.notifications:
      make_observation.add_to_queue(
          notification.player_name,
          notification.message,
      )
    return outcome.model_dump_json()
