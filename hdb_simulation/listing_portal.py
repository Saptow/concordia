"""Listing-portal state and retrieval utilities for the HDB simulation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
from pathlib import Path
import random
import re
import numpy as np
from typing import Any
from concordia.hdb_simulation.models.schemas.listing.schema import PortalSearchResult
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas.common import Flat
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas


@dataclasses.dataclass
class ListFlatResult:
    listing_id: str
    notifications: list[tuple[str, str]]


@dataclasses.dataclass
class SearchAndRequestResult:
    results: list[listing_schemas.PortalSearchResult]
    market_feedback: str
    notifications: list[tuple[str, str]]


@dataclasses.dataclass
class ReviewRequestsResult:
    match: listing_schemas.NegotiationMatch | None
    notifications: list[tuple[str, str]]


class ListingPortalRetriever:
    # TODO: implement embed_text for upserting 
    """
    Listing Portal Helper Class
    This class helps with: 
    - Upserting listings
    - Deactivating listings
    - Searching listings with BM25 and dense retrieval.

    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        dense_embedding_model: SentenceTransformer | None = None,
        collection_name: str = qdrant_schemas.DEFAULT_COLLECTION_NAME,
    ):
        self._dense_embedder=dense_embedding_model
        self._collection_name = collection_name
        self._client = client
        self._records: dict[str, listing_schemas.ListingRecord] = {}
        self._rrf_weights = [1.0, 1.5]  # Relative Weights for BM25 and dense retrieval in RRF scoring
        # Ensure collection exists
        if not self._client.collection_exists(collection_name):
                raise NotImplementedError("Qdrant collection setup is not implemented yet.")
        
    @staticmethod
    def _embed_text(self, text: str, prefix: str) -> np.ndarray:
            if self._dense_embedder is None:
                  raise ValueError("Dense embedder is not initialized.")
            return self._dense_embedder.encode(f"{prefix}: {text}")

    def upsert_listing(self, record: listing_schemas.ListingRecord) -> None:
        """Insert or replace a seller listing in the portal index."""
        document = record.to_document()
        embedding = self._embed_text(document)
        self._client.upsert(
            collection_name=self._collection_name,
            points=[record.to_qdrant_point(embedding)],
        )

    def deactivate_listing(self, listing_id: str) -> None:
        record = self._records.get(listing_id)
        if record is not None:
                record.active = False
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qdrant_models.PointIdsList(points=[listing_id]),
            )

    def search(
        self,
        # TODO: construct a default query based on preferences, price range, etc
        query: str,
        *,
        k: int = 5,
    ) -> list[listing_schemas.PortalSearchResult]:
        """
        Search active listings using Qdrant vector retrieval plus local BM25.
        This method assumes that vector indices for both sparse and dense embeddings are implemented already.
        """

        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                qdrant_models.Prefetch(
                    query=qdrant_models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using=qdrant_schemas.SPARSE_EMBEDDINGS_KEY,
                    limit=2 * k,
                ),
                qdrant_models.Prefetch(
                    query=query,
                    using=qdrant_schemas.DENSE_EMBEDDINGS_KEY,
                    limit=2 * k,
                ),
            ],
            query=qdrant_models.RrfQuery(
                rrf=qdrant_models.Rrf(weights=self._rrf_weights)  # BM25 weighted 2x over dense
            ),
            limit=k,
            with_payload=True,
        )

        # Parse results and map back to PortalSearchResult
        query_score_payloads = [[r.score, r.payload] for r in results.points]

        # TODO: Make sure the payload is coherent
        res = [PortalSearchResult(
            listing_id=payload['listing_id'],
            seller_id=payload['seller_id'],
            seller_name=payload['seller_name'],
            score=score,
            listing_price=payload['listing_price'],
            flat_type=payload['flat_type'],
            town=payload['town'],
            summary=payload['listing_summary'],
        ) for score, payload in query_score_payloads]

        return res


class ListingPortal:
    """
    Listing Portal for HDB.
    
    This class has the following features:
    - Allows sellers to list flats with details and prices.
    - Allows buyers to search for listings based on preferences and budget.
    - Facilitates negotiation requests from buyers to sellers based on listings.
    - Tracks active listings, pending requests, and matched pairs.
    - Provides market feedback to buyers based on search results.
    - Supports exporting and restoring state for persistence.
    """

    def __init__(
        self,
        *,
        retriever: ListingPortalRetriever | None = None,
        random_seed: int = 0,
    ):
        self.retriever = retriever or ListingPortalRetriever()
        self._rng = random.Random(random_seed)

        self.listings: dict[str, listing_schemas.ListingRecord] = {}
        self.requests_by_listing: dict[str, list[listing_schemas.NegotiationRequest]] = {}
        self.search_results_by_buyer: dict[str, list[listing_schemas.PortalSearchResult]] = {}
        self.private_buyer_market_states: dict[str, listing_schemas.BuyerMarketBeliefState] = {}
        self.market_feedback_by_buyer: dict[str, str] = {}
        self.matched_pairs: list[listing_schemas.NegotiationMatch] = []
        # Closed participants are temporarily inactive in the portal workflow.
        # They can be reopened later, e.g. after a failed bilateral negotiation.
        self.closed_buyers: set[str] = set()
        self.closed_sellers: set[str] = set()

    @staticmethod
    def listing_id_for_seller(seller_id: str) -> str:
        return f'listing::{seller_id}'

    @staticmethod
    def _request_id(buyer_id: str, listing_id: str, week: int) -> str:
        return f'request::{buyer_id}::{listing_id}::{week}'

    @staticmethod
    def _match_id(buyer_id: str, seller_id: str, week: int) -> str:
        return f'match::{buyer_id}::{seller_id}::{week}'

    def is_seller_listed(self, seller_id: str) -> bool:
        listing = self.listings.get(self.listing_id_for_seller(seller_id))
        return bool(listing and listing.active)

    def is_player_closed(self, player_id: str) -> bool:
        return player_id in self.closed_buyers or player_id in self.closed_sellers

    def pending_request_count(self, seller_id: str) -> int:
        listing_id = self.listing_id_for_seller(seller_id)
        requests = self.requests_by_listing.get(listing_id, [])
        return sum(
            1
            for request in requests
            if request.buyer_id not in self.closed_buyers
        )

    def _buyer_market_state(
        self,
        buyer: listing_schemas.PortalBuyer,
    ) -> listing_schemas.BuyerMarketBeliefState:
        state = self.private_buyer_market_states.get(buyer.id)
        if state is None:
            base_reservation_price = float(buyer.budget.max_price)
            state = listing_schemas.BuyerMarketBeliefState(
                buyer_id=buyer.id,
                base_reservation_price=base_reservation_price,
                effective_reservation_price=base_reservation_price,
            )
            self.private_buyer_market_states[buyer.id] = state
        return state

    def effective_reservation_price_for_buyer(
        self,
        buyer: listing_schemas.PortalBuyer,
    ) -> float:
        return float(self._buyer_market_state(buyer).effective_reservation_price)

    @staticmethod
    def _derive_query_from_preferences(
        buyer: listing_schemas.PortalBuyer,
        supplied_query: str,
    ) -> str:
        features = buyer.preferences.features.strip()
        if supplied_query.strip():
            return supplied_query.strip()
        return (
            f"{', '.join(buyer.preferences.flat_type)} flat in "
            f"{', '.join(buyer.preferences.towns)}. {features}"
        ).strip()

    @staticmethod
    def _format_results_for_notification(
        results: Sequence[listing_schemas.PortalSearchResult],
    ) -> str:
        if not results:
            return 'No matching listings were found this week.'
        lines = []
        for result in results[:3]:
            lines.append(
                f"- {result.listing_id}: {result.flat_type} in {result.town}, "
                f"SGD {result.listing_price:.0f}, score={result.score:.2f}"
            )
        return '\n'.join(lines)

    @staticmethod
    def _market_feedback(
        effective_reservation_price: float,
        results: Sequence[listing_schemas.PortalSearchResult],
    ) -> str:
        if not results:
            return (
                'Current portal search suggests supply is thin for your preferences. '
                'Broaden town or flat-type filters if urgency rises.'
            )

        prices = [float(result.listing_price) for result in results]
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        if avg_price <= effective_reservation_price:
            affordability = 'Most matching listings remain within your upper budget bound.'
        else:
            affordability = (
                'Most matching listings are above your upper budget bound, so valuation '
                'pressure is increasing.'
            )
        return (
            f"Observed portal valuation band this week: SGD {min_price:.0f} to "
            f"SGD {max_price:.0f}, average SGD {avg_price:.0f}. {affordability}"
        )

    def _update_buyer_market_state(
        self,
        buyer: listing_schemas.PortalBuyer,
        results: Sequence[listing_schemas.PortalSearchResult],
        feedback: str,
    ) -> listing_schemas.BuyerMarketBeliefState:
        state = self._buyer_market_state(buyer)
        state.latest_market_feedback = feedback
        state.feedback_history.append(feedback)

        if not results:
            return state

        prices = [float(result.listing_price) for result in results]
        observed_min_price = min(prices)
        observed_avg_price = sum(prices) / len(prices)
        observed_max_price = max(prices)

        state.latest_observed_min_price = observed_min_price
        state.latest_observed_avg_price = observed_avg_price
        state.latest_observed_max_price = observed_max_price

        current_effective = float(state.effective_reservation_price)
        target_price = observed_avg_price
        learning_rate = 0.35
        upward_cap = max(float(buyer.budget.max_price), current_effective) * 1.25
        updated_effective = current_effective + (
            learning_rate * (target_price - current_effective)
        )
        state.effective_reservation_price = max(
            float(buyer.budget.min_price),
            min(updated_effective, upward_cap),
        )
        return state

    def reopen_buyer(self, buyer_id: str) -> None:
        self.closed_buyers.discard(buyer_id)

    def reopen_seller(self, seller_id: str) -> None:
        self.closed_sellers.discard(seller_id)

    def reopen_after_failed_negotiation(
        self,
        *,
        buyer_id: str,
        seller: listing_schemas.PortalSeller,
        week: int,
        relist_price: float | None = None,
        listing_summary: str | None = None,
        clear_stale_requests: bool = True,
    ) -> ListFlatResult:
        self.reopen_buyer(buyer_id)
        self.reopen_seller(seller.id)
        listing_id = self.listing_id_for_seller(seller.id)
        record = self.listings.get(listing_id)
        if record is None:
            return self.list_flat(
                seller,
                listing_price=relist_price or seller.expectations.max_price,
                listing_summary=listing_summary or seller.flat.description,
                week=week,
            )

        if clear_stale_requests:
            self.requests_by_listing[listing_id] = []
        record.active = True
        record.listed_week = week
        if relist_price is not None:
            record.listing_price = float(relist_price)
        if listing_summary is not None:
            record.listing_summary = listing_summary.strip() or record.listing_summary
        self.retriever.upsert_listing(record)
        return ListFlatResult(
            listing_id=listing_id,
            notifications=[
                (
                    seller.name,
                    (
                        f"[portal] Week {week}: Your listing {listing_id} has been "
                        'reactivated after a failed negotiation.'
                    ),
                ),
            ],
        )

    def list_flat(
        self,
        seller: listing_schemas.PortalSeller,
        *,
        listing_price: float,
        listing_summary: str,
        week: int,
    ) -> ListFlatResult:
        seller_id = seller.id
        listing_id = self.listing_id_for_seller(seller_id)
        record = listing_schemas.ListingRecord(
            listing_id=listing_id,
            seller_id=seller_id,
            seller_name=seller.name,
            listing_price=float(listing_price),
            listing_summary=listing_summary.strip() or seller.flat.description,
            flat=seller.flat,
            listed_week=week,
        )
        self.listings[listing_id] = record
        self.requests_by_listing.setdefault(listing_id, [])
        self.retriever.upsert_listing(record)
        return ListFlatResult(
            listing_id=listing_id,
            notifications=[
                (
                    seller.name,
                    (
                        f"[portal] Week {week}: Your flat is now listed as "
                        f"{listing_id} at SGD {record.listing_price:.0f}."
                    ),
                ),
            ],
        )

    def search_and_request(
        self,
        buyer: listing_schemas.PortalBuyer,
        *,
        search_query: str,
        requested_listing_ids: Sequence[str],
        market_valuation_notes: str,
        week: int,
    ) -> SearchAndRequestResult:
        buyer_id = buyer.id
        effective_reservation_price = self.effective_reservation_price_for_buyer(buyer)
        effective_query = self._derive_query_from_preferences(buyer, search_query)
        results = self.retriever.search(
            effective_query,
            preferred_flat_types=buyer.preferences.flat_type,
            preferred_towns=buyer.preferences.towns,
            max_budget=effective_reservation_price,
            limit=5,
        )
        self.search_results_by_buyer[buyer_id] = results
        feedback = self._market_feedback(effective_reservation_price, results)
        self._update_buyer_market_state(buyer, results, feedback)
        self.market_feedback_by_buyer[buyer_id] = feedback
        notifications = [
            (
                buyer.name,
                (
                    f"[portal] Week {week}: Search results for "
                    f"\"{effective_query}\".\n"
                    f"{self._format_results_for_notification(results)}\n"
                    f"{feedback}"
                ),
            ),
        ]

        valid_listing_ids = {
            result.listing_id for result in results
        } | {
            listing_id
            for listing_id, record in self.listings.items()
            if record.active
        }

        for listing_id in requested_listing_ids:
            if listing_id not in valid_listing_ids:
                continue
            listing = self.listings.get(listing_id)
            if listing is None or not listing.active:
                continue
            seller_id = listing.seller_id
            if seller_id in self.closed_sellers:
                continue
            existing_requests = self.requests_by_listing.setdefault(listing_id, [])
            already_requested = any(
                request.buyer_id == buyer_id
                and request.buyer_id not in self.closed_buyers
                for request in existing_requests
            )
            if already_requested:
                continue
            request = listing_schemas.NegotiationRequest(
                request_id=self._request_id(buyer_id, listing_id, week),
                buyer_id=buyer_id,
                buyer_name=buyer.name,
                listing_id=listing_id,
                seller_id=seller_id,
                week_submitted=week,
                message=market_valuation_notes.strip(),
                market_valuation_notes=market_valuation_notes.strip(),
            )
            existing_requests.append(request)
            notifications.append((
                listing.seller_name,
                (
                    f"[portal] Week {week}: New negotiation request on {listing_id} "
                    f"from {buyer.name}. Total open requests: "
                    f"{self.pending_request_count(seller_id)}."
                ),
            ))
            notifications.append((
                buyer.name,
                (
                    f"[portal] Week {week}: Your request for {listing_id} has been "
                    f"sent to {listing.seller_name}."
                ),
            ))
        return SearchAndRequestResult(
            results=results,
            market_feedback=feedback,
            notifications=notifications,
        )

    def review_requests_and_start_negotiation(
        self,
        seller: listing_schemas.PortalSeller,
        buyer_registry: Mapping[str, listing_schemas.PortalBuyer],
        *,
        week: int,
    ) -> ReviewRequestsResult:
        seller_id = seller.id
        listing_id = self.listing_id_for_seller(seller_id)
        listing = self.listings.get(listing_id)
        if listing is None or not listing.active:
            return ReviewRequestsResult(
                match=None,
                notifications=[
                    (
                        seller.name,
                        f'[portal] Week {week}: Your flat is not currently listed.',
                    ),
                ],
            )

        open_requests = [
            request
            for request in self.requests_by_listing.get(listing_id, [])
            if request.buyer_id not in self.closed_buyers
        ]
        if not open_requests:
            return ReviewRequestsResult(
                match=None,
                notifications=[
                    (
                        seller.name,
                        (
                            f"[portal] Week {week}: No open requests yet for "
                            f"{listing_id}. Your flat remains listed."
                        ),
                    ),
                ],
            )

        chosen_request = self._rng.choice(open_requests)
        buyer = buyer_registry[chosen_request.buyer_id]
        match = listing_schemas.NegotiationMatch(
            match_id=self._match_id(chosen_request.buyer_id, seller_id, week),
            buyer_id=chosen_request.buyer_id,
            buyer_name=buyer.name,
            seller_id=seller_id,
            seller_name=seller.name,
            listing_id=listing_id,
            week_matched=week,
        )
        self.matched_pairs.append(match)
        self.closed_buyers.add(chosen_request.buyer_id)
        self.closed_sellers.add(seller_id)
        listing.active = False
        self.retriever.deactivate_listing(listing_id)
        return ReviewRequestsResult(
            match=match,
            notifications=[
                (
                    seller.name,
                    (
                        f"[portal] Week {week}: A negotiation handoff has started "
                        f"with {buyer.name} for {listing_id}."
                    ),
                ),
                (
                    buyer.name,
                    (
                        f"[portal] Week {week}: {seller.name} accepted your "
                        f"portal request. Bilateral negotiation is now open for "
                        f"{listing_id}."
                    ),
                ),
            ],
        )

    def export_state(self) -> dict[str, Any]:
        return {
            'listings': {
                listing_id: {
                    'seller_id': record.seller_id,
                    'seller_name': record.seller_name,
                    'listing_price': record.listing_price,
                    'listing_summary': record.listing_summary,
                    'flat': record.flat.model_dump(),
                    'listed_week': record.listed_week,
                    'active': int(record.active),
                }
                for listing_id, record in self.listings.items()
            },
            'requests_by_listing': {
                listing_id: [request.model_dump() for request in requests]
                for listing_id, requests in self.requests_by_listing.items()
            },
            'search_results_by_buyer': {
                buyer_id: [result.model_dump() for result in results]
                for buyer_id, results in self.search_results_by_buyer.items()
            },
            'private_buyer_market_states': {
                buyer_id: state.model_dump()
                for buyer_id, state in self.private_buyer_market_states.items()
            },
            'market_feedback_by_buyer': dict(self.market_feedback_by_buyer),
            'matched_pairs': [match.model_dump() for match in self.matched_pairs],
            'closed_buyers': sorted(self.closed_buyers),
            'closed_sellers': sorted(self.closed_sellers),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        retriever: ListingPortalRetriever | None = None,
        random_seed: int = 0,
    ) -> 'ListingPortal':
        restored = cls(
            retriever=retriever,
            random_seed=random_seed,
        )
        for listing_id, payload in dict(state.get('listings', {})).items():
            record = listing_schemas.ListingRecord(
                listing_id=str(listing_id),
                seller_id=str(payload['seller_id']),
                seller_name=str(payload['seller_name']),
                listing_price=float(payload['listing_price']),
                listing_summary=str(payload['listing_summary']),
                flat=Flat.model_validate(payload['flat']),
                listed_week=int(payload['listed_week']),
                active=bool(payload.get('active', 1)),
            )
            restored.listings[listing_id] = record
            restored.requests_by_listing.setdefault(listing_id, [])
            restored.retriever.upsert_listing(record)
            if not record.active:
                restored.retriever.deactivate_listing(listing_id)

        restored.requests_by_listing = {
            listing_id: [
                listing_schemas.NegotiationRequest.model_validate(request)
                for request in requests
            ]
            for listing_id, requests in dict(
                state.get('requests_by_listing', {})
            ).items()
        }
        restored.search_results_by_buyer = {
            buyer_id: [
                listing_schemas.PortalSearchResult.model_validate(result)
                for result in results
            ]
            for buyer_id, results in dict(
                state.get('search_results_by_buyer', {})
            ).items()
        }
        restored.private_buyer_market_states = {
            buyer_id: listing_schemas.BuyerMarketBeliefState.model_validate(payload)
            for buyer_id, payload in dict(
                state.get('private_buyer_market_states', {})
            ).items()
        }
        restored.market_feedback_by_buyer = dict(
            state.get('market_feedback_by_buyer', {})
        )
        restored.matched_pairs = [
            listing_schemas.NegotiationMatch.model_validate(match)
            for match in list(state.get('matched_pairs', []))
        ]
        restored.closed_buyers = {
            str(player_id) for player_id in list(state.get('closed_buyers', []))
        }
        restored.closed_sellers = {
            str(player_id) for player_id in list(state.get('closed_sellers', []))
        }
        return restored
