"""Listing-portal state and retrieval utilities for the HDB simulation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
import random
import re
import numpy as np
from typing import Any
from absl import logging
from qdrant_client import QdrantClient, models as qdrant_models

from sentence_transformers import SentenceTransformer

from concordia.prefabs.entity.negotiation.components import uncertain_helper
from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas

@dataclasses.dataclass
class SearchAndRequestResult:
    results: list[listing_schemas.PortalSearchResult]
    market_feedback: str

class ListingPortalRetriever:
    # TODO: implement embed_text for upserting 
    """
    ListingPortalRetriever manages interactions with Qdrant vector database for ListingPortal. 
    This class helps with: 
    - Upserting listings
    - Deactivating listings
    - Updating listing payloads without re-embedding (e.g. for price changes or active status updates)
    - Searching listings with BM25 and dense retrieval.

    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        dense_embedding_model: SentenceTransformer | None = None,
        collection_name: str = qdrant_schemas.DEFAULT_COLLECTION_NAME,
        db_path: str = qdrant_schemas.DEFAULT_DB_PATH,
    ):
        if client is None:
            client = qdrant_schemas.make_qdrant_client(db_path)
        
        self._dense_embedder=dense_embedding_model
        self._collection_name = collection_name
        self._client = client
        self._rrf_weights = [1.0, 1.5]  # Relative Weights for BM25 and dense retrieval in RRF scoring
        # Ensure collection exists
        if not self._client.collection_exists(collection_name):
            logging.exception(f"Qdrant collection '{collection_name}' does not exist. Please create it before using the retriever.")

    def _embed_dense_text(self, text: str) -> np.ndarray:
            if self._dense_embedder is None:
                logging.exception("Attempted to embed text but no dense embedding model was provided.")
            return self._dense_embedder.encode(text)

    def _rrf_ranker(self) -> qdrant_models.Rrf:
        """Build an RRF ranker compatible with the installed qdrant-client version."""
        model_fields = getattr(qdrant_models.Rrf, 'model_fields', {}) or {}
        if 'weights' in model_fields:
            return qdrant_models.Rrf(weights=self._rrf_weights)
        return qdrant_models.Rrf()

    def upsert_listing(self, record: listing_schemas.ListingRecord) -> None:
        """Insert or replace a seller listing in the portal index."""
        document = record.to_document()
        embedding = self._embed_dense_text(document)
        self._client.upsert(
            collection_name=self._collection_name,
            points=[record.to_qdrant_point(embedding)],
        )

    def update_listing_payload(self, record: listing_schemas.ListingRecord) -> None:
        """Updates stored payload fields without recomputing embeddings."""
        self._client.set_payload(
            collection_name=self._collection_name,
            payload=record.qdrant_payload(),
            points=qdrant_schemas.seller_filter(record.seller_id),
        )

    def get_listing_record(
        self,
        seller_id: str,
    ) -> listing_schemas.ListingRecord | None:
        """Fetches a seller listing record directly from Qdrant payload."""
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
        return listing_schemas.ListingRecord.from_qdrant_payload(payload)

    def deactivate_listing(self, seller_id: str) -> None:
        """Marks a seller listing inactive without re-embedding."""
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
        Search active listings using Qdrant vector retrieval plus local BM25.
        This method assumes that vector indices for both sparse and dense embeddings are implemented already.
        """
        dense_query=self._embed_dense_text(query)
        results = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                qdrant_models.Prefetch(
                    query=qdrant_models.Document(
                        text=query,
                        model="Qdrant/bm25",
                    ),
                    using=qdrant_schemas.SPARSE_EMBEDDINGS_KEY,
                    limit=2 * limit,
                    weight=self._rrf_weights[0],
                ),
                qdrant_models.Prefetch(
                    query=dense_query,
                    using=qdrant_schemas.DENSE_EMBEDDINGS_KEY,
                    limit=2 * limit,
                    weight=self._rrf_weights[1],
                ),
            ],
            query=qdrant_models.RrfQuery(
                rrf=self._rrf_ranker()
            ),
            limit=max(10, 3 * limit),
            with_payload=True,
        )


        filtered_results: list[listing_schemas.PortalSearchResult] = []
        for point in results.points:
            payload = point.payload or {}
            if not bool(payload.get('active', False)):
                continue
            record = listing_schemas.ListingRecord.from_qdrant_payload(payload)
            listing_price = float(record.listing_price)
            if max_budget is not None and listing_price > float(max_budget):
                continue
            filtered_results.append(record.to_search_result(score=float(point.score)))
            if len(filtered_results) >= limit:
                break

        return filtered_results

  
class ListingPortal:
    """
    Listing Portal for HDB.
    
    This class implements the action space for both buyers and sellers within the listing portal workflow. 
    The state of the players are also maintained within this class.
    For buyers, 
    1. They can search for listings based on their preferences and send negotiation requests to sellers. 
    2. They can update their effective reservation price for their desired preferences to guide their search and request strategy in subsequent weeks.
        - Based on market feedback from search results
        - TODO: based on policy shocks injected throughout the simulation. 

    For sellers, 
    1. They can list their flats as active in the market. (to induct them into the market)
    2. They review incoming negotiation requests and select one to start a bilateral negotiation, which temporarily deactivates the listing and other pending requests.
    3. TODO: They can receive market feedback based on search results and policies to adjust their listing price throughout the simulation.
    4. [NOT IMPLEMENTED FOR FYP] They can delist their listing to withdraw from the market. 
    """

    def __init__(
        self,
        *,
        retriever: ListingPortalRetriever | None = None,
        random_seed: int = 0,
    ):
        self.retriever = retriever or ListingPortalRetriever()
        self._rng = random.Random(random_seed)

        self.requests_by_seller: dict[str, list[listing_schemas.NegotiationRequest]] = {}
        self.search_results_by_buyer: dict[str, list[listing_schemas.PortalSearchResult]] = {}
        self.private_buyer_market_states: dict[str, listing_schemas.BuyerMarketBeliefState] = {}
        self.private_seller_market_states: dict[str, listing_schemas.SellerMarketBeliefState] = {}
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

    # Shared portal state helpers.
    def is_seller_listed(self, seller_id: str) -> bool:
        listing = self.retriever.get_listing_record(seller_id)
        return bool(listing and listing.active)

    def is_player_closed(self, player_id: str) -> bool:
        return player_id in self.closed_buyers or player_id in self.closed_sellers

    def pending_request_count(self, seller_id: str) -> int:
        requests = self.requests_by_seller.get(seller_id, [])
        return sum(
            1
            for request in requests
            if request.buyer_id not in self.closed_buyers
        )

    def get_listing_record(
        self,
        seller_id: str,
    ) -> listing_schemas.ListingRecord | None:
        return self.retriever.get_listing_record(seller_id)

    # Buyer-side helpers and actions.
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
        """Returns the top-scoring valid listing for the buyer, if any."""
        buyer_id = str(buyer.id).strip()
        max_budget = float(buyer.budget.max_price)
        for result in results:
            if result.listing_price > max_budget or result.score <= 0.0:
                continue
            if self.has_buyer_negotiated_with_seller(buyer_id, result.seller_id):
                continue
            return [result.listing_id]
        return []

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
    ) -> listing_schemas.SellerMarketBeliefState:
        state = self.private_seller_market_states.get(seller.id)
        if state is None:
            base_reservation_price = float(seller.expectations.min_price)
            state = listing_schemas.SellerMarketBeliefState(
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
        features = buyer.preferences.features.strip()
        if supplied_query.strip():
            return supplied_query.strip()
        return (
            f"{', '.join(buyer.preferences.flat_type)} flat in "
            f"{', '.join(buyer.preferences.towns)}. {features}"
        ).strip()

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
            affordability = 'Most matching listings remain within your effective reservation bound.'
        else:
            affordability = (
                'Most matching listings are above your effective reservation bound, '
                'so valuation pressure is increasing.'
            )
        return (
            f"Observed portal valuation band this week: SGD {min_price:.0f} to "
            f"SGD {max_price:.0f}, average SGD {avg_price:.0f}. {affordability}"
        )

    @staticmethod
    def _market_signal_reliability(
        results: Sequence[listing_schemas.PortalSearchResult],
    ) -> float:
        if not results:
            return 0.0
        prices = [float(result.listing_price) for result in results]
        average_price = max(sum(prices) / len(prices), 1.0)
        relative_spread = (max(prices) - min(prices)) / average_price
        sample_factor = min(1.0, len(prices) / 5.0)
        spread_factor = max(0.25, 1.0 - min(relative_spread, 0.75))
        return max(0.15, min(0.9, 0.15 + (0.75 * sample_factor * spread_factor)))

    def _update_buyer_market_state(
        self,
        buyer: listing_schemas.PortalBuyer,
        results: Sequence[listing_schemas.PortalSearchResult],
        feedback: str,
    ) -> listing_schemas.BuyerMarketBeliefState:
        state = self._buyer_market_state(buyer)
        state.latest_market_feedback = feedback
        if not state.feedback_history or state.feedback_history[-1] != feedback:
            state.feedback_history.append(feedback)
        if not results:
            return state

        prices = [float(result.listing_price) for result in results]
        state.latest_observed_min_price = min(prices)
        state.latest_observed_avg_price = sum(prices) / len(prices)
        state.latest_observed_max_price = max(prices)

        belief = state.effective_reservation
        belief.update_with_evidence(
            state.latest_observed_avg_price,
            reliability=self._market_signal_reliability(results),
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
         - Search results are based on the buyer's preferences and effective reservation price.
        """
        max_budget = min(
            float(buyer.budget.max_price),
            self.effective_reservation_price_for_buyer(buyer),
        )
        effective_query = self._derive_query_from_preferences(buyer, '')
        results = self.retriever.search(
            effective_query,
            max_budget=max_budget,
            limit=10,  # TODO: revise this as needed
        )
        feedback = self._market_feedback(
            self.effective_reservation_price_for_buyer(buyer),
            results,
        )
        requested_listing_ids = self._top_valid_listing_ids_for_buyer(
            buyer,
            results,
        )
        buyer_id = buyer.id
        self.search_results_by_buyer[buyer_id] = list(results)
        self._update_buyer_market_state(buyer, results, feedback)
        self.market_feedback_by_buyer[buyer_id] = feedback

        results_by_listing_id = {
            result.listing_id: result for result in results
        }

        for listing_id in requested_listing_ids:
            result = results_by_listing_id.get(listing_id)
            if result is None:
                continue
            seller_id = result.seller_id
            listing = self.retriever.get_listing_record(seller_id)
            if listing is None or not listing.active:
                continue
            if seller_id in self.closed_sellers:
                continue
            if self.has_buyer_negotiated_with_seller(buyer_id, seller_id):
                continue
            existing_requests = self.requests_by_seller.setdefault(seller_id, [])
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
                message='',
                market_valuation_notes='',
            )
            existing_requests.append(request)
        return SearchAndRequestResult(
            results=list(results),
            market_feedback=feedback,
        )

    # Seller-side actions.
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

        open_requests = [
            request
            for request in self.requests_by_seller.get(seller_id, [])
            if request.buyer_id not in self.closed_buyers
        ]
        if not open_requests:
            return None

        chosen_request = self._rng.choice(open_requests) # For now, randomly select a request to review.
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

    # Persistence helpers.
    def export_state(self) -> dict[str, Any]:
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
            buyer_id: listing_schemas.BuyerMarketBeliefState.model_validate(payload)
            for buyer_id, payload in dict(
                state.get('private_buyer_market_states', {})
            ).items()
        }
        restored.private_seller_market_states = {
            seller_id: listing_schemas.SellerMarketBeliefState.model_validate(payload)
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
