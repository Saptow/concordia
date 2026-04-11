"""Listing-portal state and retrieval utilities for the HDB simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import random
import threading
import numpy as np
from typing import Any
from absl import logging
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qdrant_models

from sentence_transformers import SentenceTransformer

from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas


PORTAL_SEARCH_LIMIT = 10
PORTAL_REQUEST_LIMIT_PER_WEEK = 5


@dataclasses.dataclass
class SearchAndRequestResult:
    """Return payload for a buyer's weekly portal search step."""

    results: list[listing_schemas.PortalSearchResult]
    market_feedback: str


class ListingPortalRetriever:
    """
    ListingPortalRetriever manages Qdrant access for the shared listing portal.

    The retriever is responsible for:
    - Upserting listings
    - Deactivating listings
    - Updating listing payloads without re-embedding (e.g. for price changes or active status updates)
    - Searching listings with BM25 and dense retrieval.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        dense_embedding_model: SentenceTransformer | None = None,
        sparse_embedding_model: SparseTextEmbedding | None = None,
        collection_name: str = qdrant_schemas.DEFAULT_COLLECTION_NAME,
        db_path: str = qdrant_schemas.DEFAULT_DB_PATH,
    ):
        if client is None:
            client = qdrant_schemas.make_qdrant_client(db_path)

        self._dense_embedder = dense_embedding_model
        self._sparse_embedder = sparse_embedding_model
        self._collection_name = collection_name
        self._client = client
        self._client_lock = threading.RLock()
        # Slightly favor sparse retrieval when fusing ranked candidate lists.
        self._rrf_weights = [1.0, 1.5]
        with self._client_lock:
            collection_exists = self._client.collection_exists(collection_name)
        if not collection_exists:
            raise ValueError(
                f"Qdrant collection '{collection_name}' does not exist. "
                'Create or preload it before using ListingPortalRetriever.'
            )

    def _embed_dense_texts(self, texts: Sequence[str]) -> list[np.ndarray]:
        if self._dense_embedder is None:
            raise RuntimeError(
                'Attempted to embed listing-portal text without a dense '
                'embedding model.'
            )
        if not texts:
            return []
        embeddings = self._dense_embedder.encode(
            list(texts),
            show_progress_bar=False,
        )
        return [np.asarray(embedding) for embedding in embeddings]

    def _embed_dense_text(self, text: str) -> np.ndarray:
        return self._embed_dense_texts([text])[0]

    def _embed_sparse_texts(self, texts: Sequence[str]) -> list[Any | None]:
        if not texts:
            return []
        if self._sparse_embedder is None:
            return [None] * len(texts)
        embeddings = list(self._sparse_embedder.embed(list(texts)))
        if len(embeddings) != len(texts):
            logging.warning(
                'Sparse embedder returned %s embeddings for %s query texts.',
                len(embeddings),
                len(texts),
            )
            embeddings = embeddings[: len(texts)]
            embeddings.extend([None] * (len(texts) - len(embeddings)))
        return embeddings

    def _embed_sparse_text(self, text: str) -> Any | None:
        return self._embed_sparse_texts([text])[0]

    def _rrf_ranker(self) -> qdrant_models.Rrf:
        """Build an RRF ranker compatible with the installed qdrant-client version."""
        model_fields = getattr(qdrant_models.Rrf, 'model_fields', {}) or {}
        if 'weights' in model_fields:
            return qdrant_models.Rrf(weights=self._rrf_weights)
        return qdrant_models.Rrf()

    def upsert_listing(self, record: qdrant_schemas.ListingRecord) -> None:
        """Insert or replace a seller listing in the portal index."""
        document = record.to_document()
        embedding = self._embed_dense_text(document)
        sparse_embedding = self._embed_sparse_text(document)
        with self._client_lock:
            self._client.upsert(
                collection_name=self._collection_name,
                points=[record.to_qdrant_point(embedding, sparse_embedding=sparse_embedding)],
            )

    def update_listing_payload(self, record: qdrant_schemas.ListingRecord) -> None:
        """Updates stored payload fields without recomputing embeddings."""
        with self._client_lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload=record.qdrant_payload(),
                points=qdrant_schemas.seller_filter(record.seller_id),
            )

    def get_listing_record(
        self,
        seller_id: str,
    ) -> qdrant_schemas.ListingRecord | None:
        """Fetches a seller listing record directly from Qdrant payload."""
        with self._client_lock:
            points, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=qdrant_schemas.seller_filter(seller_id),
                with_payload=True,
                with_vectors=False,
                limit=1,
            )
        if not points:
            return None
        payload = points[0].payload or {}
        return qdrant_schemas.ListingRecord.from_qdrant_payload(payload)

    def deactivate_listing(self, seller_id: str) -> None:
        """Marks a seller listing inactive without re-embedding."""
        with self._client_lock:
            self._client.set_payload(
                collection_name=self._collection_name,
                payload={'active': False},
                points=qdrant_schemas.seller_filter(seller_id),
            )

    def search(
        self,
        query: str,
        *,
        max_budget: float | None = None,
        limit: int = 5,
    ) -> list[listing_schemas.PortalSearchResult]:
        """
        Search active listings and apply portal-specific post-filters.

        Retrieval now filters to active listings in Qdrant so inactive or
        currently negotiating sellers do not consume raw top-k slots. Budget
        filtering still stays local so the portal can evolve those rules
        without rebuilding the index.
        """
        dense_query = self._embed_dense_text(query)
        sparse_query = self._embed_sparse_text(query)
        return self._search_with_embeddings(
            dense_query=dense_query,
            sparse_query=sparse_query,
            max_budget=max_budget,
            limit=limit,
        )

    def _search_with_embeddings(
        self,
        *,
        dense_query: np.ndarray,
        sparse_query: Any | None,
        max_budget: float | None,
        limit: int,
    ) -> list[listing_schemas.PortalSearchResult]:
        search_limit = max(10, 3 * limit)
        active_filter = qdrant_schemas.active_listing_filter()
        with self._client_lock:
            if sparse_query is None:
                results = self._client.query_points(
                    collection_name=self._collection_name,
                    query=dense_query,
                    using=qdrant_schemas.DENSE_EMBEDDINGS_KEY,
                    query_filter=active_filter,
                    limit=search_limit,
                    with_payload=True,
                )
            else:
                results = self._client.query_points(
                    collection_name=self._collection_name,
                    prefetch=[
                        qdrant_models.Prefetch(
                            query=qdrant_schemas.sparse_embedding_to_vector(sparse_query),
                            using=qdrant_schemas.SPARSE_EMBEDDINGS_KEY,
                            filter=active_filter,
                            limit=2 * limit,
                        ),
                        qdrant_models.Prefetch(
                            query=dense_query,
                            using=qdrant_schemas.DENSE_EMBEDDINGS_KEY,
                            filter=active_filter,
                            limit=2 * limit,
                        ),
                    ],
                    query=qdrant_models.RrfQuery(
                        rrf=self._rrf_ranker()
                    ),
                    query_filter=active_filter,
                    limit=search_limit,
                    with_payload=True,
                )

        filtered_results: list[listing_schemas.PortalSearchResult] = []
        for point in results.points:
            payload = point.payload or {}
            if not bool(payload.get('active', False)):
                continue
            record = qdrant_schemas.ListingRecord.from_qdrant_payload(payload)
            listing_price = float(record.listing_price)
            if max_budget is not None and listing_price > float(max_budget):
                continue
            filtered_results.append(record.to_search_result(score=float(point.score)))
            if len(filtered_results) >= limit:
                break

        return filtered_results

    def search_many(
        self,
        queries: Sequence[str],
        *,
        max_budgets: Sequence[float | None] | None = None,
        limit: int = 5,
    ) -> list[list[listing_schemas.PortalSearchResult]]:
        """Search many buyer queries with one batched embedding pass."""
        if not queries:
            return []
        dense_queries = self._embed_dense_texts(queries)
        sparse_queries = self._embed_sparse_texts(queries)
        if max_budgets is None:
            normalized_budgets = [None] * len(queries)
        else:
            normalized_budgets = [
                float(budget) if budget is not None else None
                for budget in max_budgets
            ]
            if len(normalized_budgets) != len(queries):
                raise ValueError(
                    'max_budgets must align with queries when provided.'
                )

        return [
            self._search_with_embeddings(
                dense_query=dense_query,
                sparse_query=sparse_query,
                max_budget=max_budget,
                limit=limit,
            )
            for dense_query, sparse_query, max_budget in zip(
                dense_queries,
                sparse_queries,
                normalized_budgets,
                strict=True,
            )
        ]

  
class ListingPortal:
    """
    Shared listing-portal state for the weekly market workflow.

    Buyers search listings, receive lightweight market feedback, and submit at
    most one request at a time. Sellers activate listings and choose one inbound
    request to move into bilateral negotiation. The portal keeps the transient
    state needed to bridge those weekly listing and negotiation phases.
    """

    def __init__(
        self,
        *,
        retriever: ListingPortalRetriever | None = None,
        random_seed: int = 0,
    ):
        self.retriever = retriever or ListingPortalRetriever()
        self._rng = random.Random(random_seed)
        self._state_lock = threading.RLock()

        self.requests_by_seller: dict[str, list[listing_schemas.NegotiationRequest]] = {}
        self.search_results_by_buyer: dict[str, list[listing_schemas.PortalSearchResult]] = {}
        self.private_buyer_market_states: dict[str, negotiation_schemas.BuyerMarketBeliefState] = {}
        self.private_seller_market_states: dict[str, negotiation_schemas.SellerMarketBeliefState] = {}
        self.market_feedback_by_buyer: dict[str, str] = {}
        self.matched_pairs: list[listing_schemas.NegotiationMatch] = []
        # Closed participants are temporarily inactive in the portal workflow.
        # They can be reopened later, e.g. after a failed bilateral negotiation.
        self.closed_buyers: set[str] = set()
        self.closed_sellers: set[str] = set()

    @staticmethod
    def listing_id_for_seller(seller_id: str) -> str:
        return qdrant_schemas.listing_id_for_seller(seller_id)

    @staticmethod
    def _request_id(buyer_id: str, listing_id: str, week: int) -> str:
        return f'request::{buyer_id}::{listing_id}::{week}'

    @staticmethod
    def _match_id(buyer_id: str, seller_id: str, week: int) -> str:
        return f'match::{buyer_id}::{seller_id}::{week}'

    def is_seller_listed(self, seller_id: str) -> bool:
        listing = self.retriever.get_listing_record(seller_id)
        return bool(listing and listing.active)

    def is_player_closed(self, player_id: str) -> bool:
        with self._state_lock:
            return (
                player_id in self.closed_buyers
                or player_id in self.closed_sellers
            )

    def pending_request_count(self, seller_id: str) -> int:
        with self._state_lock:
            requests = self.requests_by_seller.get(seller_id, [])
            return sum(
                1
                for request in requests
                if request.buyer_id not in self.closed_buyers
            )

    def get_listing_record(
        self,
        seller_id: str,
    ) -> qdrant_schemas.ListingRecord | None:
        return self.retriever.get_listing_record(seller_id)

    def has_buyer_negotiated_with_seller(
        self,
        buyer_id: str,
        seller_id: str,
    ) -> bool:
        """Returns whether the buyer and seller have previously entered negotiation."""
        normalized_buyer_id = str(buyer_id).strip()
        normalized_seller_id = str(seller_id).strip()
        if not normalized_buyer_id or not normalized_seller_id:
            return False
        with self._state_lock:
            return any(
                match.buyer_id == normalized_buyer_id
                and match.seller_id == normalized_seller_id
                for match in self.matched_pairs
            )

    def _top_valid_listing_ids_for_buyer(
        self,
        buyer: listing_schemas.PortalBuyer,
        results: Sequence[listing_schemas.PortalSearchResult],
    ) -> list[str]:
        """Returns up to the top-scoring valid listings for the buyer this week."""
        buyer_id = str(buyer.id).strip()
        max_budget = float(buyer.budget.max_price)
        requested_listing_ids: list[str] = []
        for result in results:
            if result.listing_price > max_budget or result.score <= 0.0:
                continue
            if self.has_buyer_negotiated_with_seller(buyer_id, result.seller_id):
                continue
            requested_listing_ids.append(result.listing_id)
            if len(requested_listing_ids) >= PORTAL_REQUEST_LIMIT_PER_WEEK:
                break
        return requested_listing_ids

    def _buyer_market_state(
        self,
        buyer: listing_schemas.PortalBuyer,
    ) -> negotiation_schemas.BuyerMarketBeliefState:
        with self._state_lock:
            state = self.private_buyer_market_states.get(buyer.id)
            if state is None:
                reservation_price_prior = (
                    float(buyer.reservation_price_prior)
                    if buyer.reservation_price_prior is not None
                    else float(buyer.budget.max_price)
                )
                base_reservation_price = max(
                    float(buyer.budget.min_price),
                    min(float(buyer.budget.max_price), reservation_price_prior),
                )
                state = negotiation_schemas.BuyerMarketBeliefState(
                    buyer_id=buyer.id,
                    base_reservation_price=base_reservation_price,
                    effective_reservation=uncertain_helper.NormalDistribution(
                        name='Effective reservation price',
                        mean=base_reservation_price,
                        std=max(1000.0, 0.05 * base_reservation_price),
                        confidence=0.5,
                    ),
                )
                self.private_buyer_market_states[buyer.id] = state
            return state

    def _seller_market_state(
        self,
        seller: listing_schemas.PortalSeller,
    ) -> negotiation_schemas.SellerMarketBeliefState:
        with self._state_lock:
            state = self.private_seller_market_states.get(seller.id)
            if state is None:
                base_reservation_price = float(seller.expectations.min_price)
                state = negotiation_schemas.SellerMarketBeliefState(
                    seller_id=seller.id,
                    base_reservation_price=base_reservation_price,
                    effective_reservation=uncertain_helper.NormalDistribution(
                        name='Effective reservation price',
                        mean=base_reservation_price,
                        std=max(1000.0, 0.05 * base_reservation_price),
                        confidence=0.5,
                    ),
                )
                self.private_seller_market_states[seller.id] = state
            return state

    def effective_reservation_price_for_buyer(
        self,
        buyer: listing_schemas.PortalBuyer,
    ) -> float:
        return float(
            self._buyer_market_state(buyer).effective_reservation.mean
        )

    @staticmethod
    def _derive_query_from_preferences(
        buyer: listing_schemas.PortalBuyer,
        supplied_query: str,
    ) -> str:
        flat_types = buyer.preferences.values_for('flat_type')
        towns = buyer.preferences.values_for('town')
        features = buyer.preferences.feature_summary()
        if supplied_query.strip():
            return supplied_query.strip()
        return (
            f"{', '.join(flat_types)} flat in "
            f"{', '.join(towns)}. {features}"
        ).strip()

    @staticmethod
    def _listing_match_scores(
        buyer: listing_schemas.PortalBuyer,
        results: Sequence[listing_schemas.PortalSearchResult],
    ) -> list[float]:
        return [
            common_schemas.build_buyer_flat_preference_match_score(
                buyer.preferences,
                result.flat,
            )
            for result in results
        ]

    @staticmethod
    def _relevant_market_observations(
        results: Sequence[listing_schemas.PortalSearchResult],
        match_scores: Sequence[float],
    ) -> tuple[list[tuple[float, float]], float, list[float]]:
        relevant_pairs: list[tuple[float, float]] = [
            (float(result.listing_price), float(match_score))
            for result, match_score in zip(results, match_scores, strict=True)
            if float(match_score) > 0.0
        ]
        if not relevant_pairs:
            return ([], 0.0, [])
        relevant_prices = [price for price, _ in relevant_pairs]
        return (
            relevant_pairs,
            ListingPortal._market_signal_reliability(relevant_prices),
            relevant_prices,
        )

    @staticmethod
    def _market_feedback(
        effective_reservation_price: float,
        relevant_prices: Sequence[float],
    ) -> str:
        if not relevant_prices:
            return (
                'Current portal search surfaced few listings that fit your '
                'preferences closely enough to update your willingness-to-pay.'
            )

        avg_price = sum(relevant_prices) / len(relevant_prices)
        min_price = min(relevant_prices)
        max_price = max(relevant_prices)
        if avg_price <= effective_reservation_price:
            affordability = (
                'Most preference-relevant listings remain within your effective '
                'reservation bound.'
            )
        else:
            affordability = (
                'Most preference-relevant listings are above your effective '
                'reservation bound, so valuation pressure is increasing.'
            )
        return (
            f"Observed preference-relevant portal valuation band this week: SGD "
            f"{min_price:.0f} to "
            f"SGD {max_price:.0f}, average SGD {avg_price:.0f}. {affordability}"
        )

    @staticmethod
    def _market_signal_reliability(
        prices: Sequence[float],
    ) -> float:
        if not prices:
            return 0.0
        average_price = max(sum(prices) / len(prices), 1.0)
        relative_spread = (max(prices) - min(prices)) / average_price
        sample_factor = min(1.0, len(prices) / 5.0)
        spread_factor = max(0.25, 1.0 - min(relative_spread, 0.75))
        return max(0.15, min(0.9, 0.15 + (0.75 * sample_factor * spread_factor)))

    def _update_buyer_market_state(
        self,
        buyer: listing_schemas.PortalBuyer,
        relevant_observations: Sequence[tuple[float, float]],
        market_reliability: float,
        relevant_prices: Sequence[float],
        feedback: str,
    ) -> negotiation_schemas.BuyerMarketBeliefState:
        state = self._buyer_market_state(buyer)
        state.latest_market_feedback = feedback
        if not state.feedback_history or state.feedback_history[-1] != feedback:
            state.feedback_history.append(feedback)
        if not relevant_observations or market_reliability <= 0.0:
            return state

        state.latest_observed_min_price = min(relevant_prices)
        state.latest_observed_avg_price = sum(relevant_prices) / len(relevant_prices)
        state.latest_observed_max_price = max(relevant_prices)

        belief = state.effective_reservation
        for observed_price, match_score in relevant_observations:
            reliability = max(
                0.0,
                min(1.0, float(market_reliability) * float(match_score)),
            )
            if reliability <= 0.0:
                continue
            belief.update_with_evidence(
                observed_price,
                reliability=reliability,
            )
        belief.mean = max(
            float(buyer.budget.min_price),
            min(float(buyer.budget.max_price), belief.mean),
        )
        return state

    def search_and_request(
        self,
        buyer: listing_schemas.PortalBuyer,
        *,
        week: int,
        ) -> SearchAndRequestResult:
        """
        Buyer method to search for listings and send negotiation requests to sellers.
         - Search results are based on the buyer's preferences and affordability.
        """
        max_budget = float(buyer.budget.max_price)
        effective_query = self._derive_query_from_preferences(buyer, '')
        results = self.retriever.search(
            effective_query,
            max_budget=max_budget,
            # Keep a slightly broader candidate pool than the single request we
            # currently send so market feedback can reflect nearby alternatives.
            limit=PORTAL_SEARCH_LIMIT,
        )
        match_scores = self._listing_match_scores(
            buyer,
            results,
        )
        relevant_observations, market_reliability, relevant_prices = (
            self._relevant_market_observations(results, match_scores)
        )
        feedback = self._market_feedback(
            self.effective_reservation_price_for_buyer(buyer),
            relevant_prices,
        )
        requested_listing_ids = self._top_valid_listing_ids_for_buyer(
            buyer,
            results,
        )
        buyer_id = buyer.id
        with self._state_lock:
            self.search_results_by_buyer[buyer_id] = list(results)
            self._update_buyer_market_state(
                buyer,
                relevant_observations,
                market_reliability,
                relevant_prices,
                feedback,
            )
            self.market_feedback_by_buyer[buyer_id] = feedback

        results_by_listing_id = {
            result.listing_id: result for result in results
        }

        for listing_id in requested_listing_ids:
            result = results_by_listing_id.get(listing_id)
            if result is None:
                continue
            self.submit_negotiation_request(
                buyer,
                seller_id=result.seller_id,
                week=week,
                listing_id=listing_id,
            )
        return SearchAndRequestResult(
            results=list(results),
            market_feedback=feedback,
        )

    def search_and_request_many(
        self,
        buyers: Sequence[listing_schemas.PortalBuyer],
        *,
        week: int,
    ) -> dict[str, SearchAndRequestResult]:
        """Run buyer portal search/request with batched query embeddings."""
        eligible_buyers = [
            buyer
            for buyer in buyers
            if str(buyer.id).strip() and not self.is_player_closed(buyer.id)
        ]
        if not eligible_buyers:
            return {}

        queries = [
            self._derive_query_from_preferences(buyer, '')
            for buyer in eligible_buyers
        ]
        budgets = [float(buyer.budget.max_price) for buyer in eligible_buyers]
        search_results = self.retriever.search_many(
            queries,
            max_budgets=budgets,
            limit=PORTAL_SEARCH_LIMIT,
        )

        results_by_buyer: dict[str, SearchAndRequestResult] = {}
        for buyer, results in zip(eligible_buyers, search_results, strict=True):
            match_scores = self._listing_match_scores(
                buyer,
                results,
            )
            relevant_observations, market_reliability, relevant_prices = (
                self._relevant_market_observations(results, match_scores)
            )
            feedback = self._market_feedback(
                self.effective_reservation_price_for_buyer(buyer),
                relevant_prices,
            )
            requested_listing_ids = self._top_valid_listing_ids_for_buyer(
                buyer,
                results,
            )
            buyer_id = buyer.id
            with self._state_lock:
                self.search_results_by_buyer[buyer_id] = list(results)
                self._update_buyer_market_state(
                    buyer,
                    relevant_observations,
                    market_reliability,
                    relevant_prices,
                    feedback,
                )
                self.market_feedback_by_buyer[buyer_id] = feedback

            results_by_listing_id = {
                result.listing_id: result for result in results
            }
            for listing_id in requested_listing_ids:
                result = results_by_listing_id.get(listing_id)
                if result is None:
                    continue
                self.submit_negotiation_request(
                    buyer,
                    seller_id=result.seller_id,
                    week=week,
                    listing_id=listing_id,
                )

            results_by_buyer[str(buyer_id)] = SearchAndRequestResult(
                results=list(results),
                market_feedback=feedback,
            )

        return results_by_buyer

    def submit_negotiation_request(
        self,
        buyer: listing_schemas.PortalBuyer,
        *,
        seller_id: str,
        week: int,
        listing_id: str | None = None,
        message: str = '',
        market_valuation_notes: str = '',
    ) -> listing_schemas.NegotiationRequest | None:
        """Creates one validated negotiation request for a specific seller."""
        buyer_id = str(buyer.id).strip()
        normalized_seller_id = str(seller_id).strip()
        if not buyer_id or not normalized_seller_id:
            return None
        if buyer_id in self.closed_buyers or normalized_seller_id in self.closed_sellers:
            return None
        if self.has_buyer_negotiated_with_seller(buyer_id, normalized_seller_id):
            return None

        listing = self.retriever.get_listing_record(normalized_seller_id)
        if listing is None or not listing.active:
            return None
        if float(listing.listing_price) > float(buyer.budget.max_price):
            return None

        effective_listing_id = (
            str(listing_id).strip() if listing_id is not None else listing.listing_id
        )
        if effective_listing_id != listing.listing_id:
            return None

        with self._state_lock:
            if (
                buyer_id in self.closed_buyers
                or normalized_seller_id in self.closed_sellers
            ):
                return None
            if self.has_buyer_negotiated_with_seller(
                buyer_id,
                normalized_seller_id,
            ):
                return None
            existing_requests = self.requests_by_seller.setdefault(
                normalized_seller_id,
                [],
            )
            already_requested = any(
                request.buyer_id == buyer_id
                and request.buyer_id not in self.closed_buyers
                for request in existing_requests
            )
            if already_requested:
                return None

            request = listing_schemas.NegotiationRequest(
                request_id=self._request_id(buyer_id, effective_listing_id, week),
                buyer_id=buyer_id,
                buyer_name=buyer.name,
                listing_id=effective_listing_id,
                seller_id=normalized_seller_id,
                week_submitted=week,
                message=message,
                market_valuation_notes=market_valuation_notes,
            )
            existing_requests.append(request)
            return request

    def list_flat(
        self,
        seller: listing_schemas.PortalSeller,
        *,
        week: int,
        listing_price = None,
    ) -> str:
        seller_id = seller.id
        listing_id = self.listing_id_for_seller(seller_id)
        record = self.retriever.get_listing_record(seller_id)
        if record is None:
            logging.warning(
                'Skipping listing activation for seller %s because listing %s was not preloaded.',
                seller_id,
                listing_id,
            )
            return listing_id
        if listing_price is not None:
            record.listing_price = float(listing_price)
        record.active = True
        with self._state_lock:
            self.requests_by_seller.setdefault(seller_id, [])
        self.retriever.update_listing_payload(record)
        return listing_id

    def review_requests_and_start_negotiation(
        self,
        seller: listing_schemas.PortalSeller,
        *,
        week: int,
    ) -> listing_schemas.NegotiationMatch | None:
        seller_id = seller.id
        listing_id = self.listing_id_for_seller(seller_id)

        with self._state_lock:
            open_requests = [
                request
                for request in self.requests_by_seller.get(seller_id, [])
                if request.buyer_id not in self.closed_buyers
            ]
            if not open_requests:
                return None

            # Seller-side request triage is intentionally simple for now; this
            # keeps the portal deterministic apart from the configured RNG seed.
            chosen_request = self._rng.choice(open_requests)
            match = listing_schemas.NegotiationMatch(
                match_id=self._match_id(chosen_request.buyer_id, seller_id, week),
                buyer_id=chosen_request.buyer_id,
                buyer_name=chosen_request.buyer_name,
                seller_id=seller_id,
                seller_name=seller.name,
                listing_id=listing_id,
                week_matched=week,
            )
            self.matched_pairs.append(match)
            self.closed_buyers.add(chosen_request.buyer_id)
            self.closed_sellers.add(seller_id)
            self.requests_by_seller[seller_id] = []
        self.retriever.deactivate_listing(seller_id)
        return match

    def export_state(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                'requests_by_seller': {
                    seller_id: [request.model_dump() for request in requests]
                    for seller_id, requests in self.requests_by_seller.items()
                },
                'search_results_by_buyer': {
                    buyer_id: [result.model_dump() for result in results]
                    for buyer_id, results in self.search_results_by_buyer.items()
                },
                'private_buyer_market_states': {
                    buyer_id: state.model_dump()
                    for buyer_id, state in self.private_buyer_market_states.items()
                },
                'private_seller_market_states': {
                    seller_id: state.model_dump()
                    for seller_id, state in self.private_seller_market_states.items()
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
        restored.requests_by_seller = {
            seller_id: [
                listing_schemas.NegotiationRequest.model_validate(request)
                for request in requests
            ]
            for seller_id, requests in dict(
                state.get('requests_by_seller', {})
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
            buyer_id: negotiation_schemas.BuyerMarketBeliefState.model_validate(payload)
            for buyer_id, payload in dict(
                state.get('private_buyer_market_states', {})
            ).items()
        }
        restored.private_seller_market_states = {
            seller_id: negotiation_schemas.SellerMarketBeliefState.model_validate(payload)
            for seller_id, payload in dict(
                state.get('private_seller_market_states', {})
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
