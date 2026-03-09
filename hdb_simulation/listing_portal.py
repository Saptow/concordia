"""Listing-portal state and retrieval utilities for the HDB simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import math
import random
from typing import Any
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from pydantic import BaseModel

from concordia.hdb_simulation.models.schemas import listing as listing_schemas
from concordia.hdb_simulation.models.schemas.common import Flat

class ListingRecord(BaseModel):
  """Internal representation of a flat listed on the portal."""

  listing_id: str
  seller_id: str
  seller_name: str
  listing_price: float
  listing_summary: str
  flat: Flat
  listed_week: int
  active: bool = True

  @staticmethod
  def _format_field_name(field_name: str) -> str:
    return field_name.replace('_', ' ').strip().title()

  def flat_metadata(self) -> dict[str, Any]:
    metadata = self.flat.model_dump(mode='python')
    metadata['flat_type'] = str(self.flat.flat_type)
    return metadata

  def to_document(self) -> str:
    lines = [
        f'Listing ID: {self.listing_id}',
        f'Asking Price: SGD {self.listing_price:.0f}',
    ]
    for key, value in self.flat_metadata().items():
      if isinstance(value, list):
        rendered = ', '.join(str(item) for item in value) if value else 'None listed'
      elif value is None:
        rendered = 'None'
      else:
        rendered = str(value)
      lines.append(f'{self._format_field_name(key)}: {rendered}')
    lines.append(f'Summary: {self.listing_summary}')
    return '\n'.join(lines)

  def to_search_result(self, score: float) -> listing_schemas.PortalSearchResult:
    return listing_schemas.PortalSearchResult(
        listing_id=self.listing_id,
        seller_id=self.seller_id,
        seller_name=self.seller_name,
        score=float(score),
        listing_price=float(self.listing_price),
        flat_type=str(self.flat.flat_type),
        town=self.flat.town,
        summary=self.listing_summary,
    )


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
  """Hybrid listing retrieval backed by a mandatory local ChromaDB index."""

  def __init__(
      self,
      embedding_model_name: str = './models/intfloat-e5-base-v2',
      local_files_only: bool = True,
      collection_name: str = 'hdb_listing_portal',
  ):
    self._embedding_model_name = embedding_model_name
    self._collection_name = collection_name
    self._records: dict[str, ListingRecord] = {}
    self._embeddings: dict[str, np.ndarray] = {}

    try:
      self._embedder = SentenceTransformer(
          embedding_model_name,
          local_files_only=local_files_only,
      )
    except Exception as exc:
      raise RuntimeError(
          'Failed to load the local embedding model for the listing portal.'
      ) from exc

    try:
      client = chromadb.Client()
      self._chroma_collection = client.get_or_create_collection(
          name=collection_name,
          metadata={'hnsw:space': 'cosine'},
      )
    except Exception as exc:
      raise RuntimeError(
          'Failed to initialize the local ChromaDB collection for the listing portal.'
      ) from exc

  def _embed_text(self, text: str, *, prefix: str) -> np.ndarray | None:
    if self._embedder is None:
      return None
    try:
      vector = self._embedder.encode(
          f'{prefix}: {text}',
          show_progress_bar=False,
      )
    except Exception:
      return None
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
      return None
    return arr / norm

  @staticmethod
  def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
      return 0.0
    return float(np.dot(left, right))

  @staticmethod
  def _keyword_overlap(query: str, document: str) -> float:
    query_tokens = {
        token.strip(' ,.;:').lower()
        for token in query.split()
        if token.strip(' ,.;:')
    }
    doc_tokens = {
        token.strip(' ,.;:').lower()
        for token in document.split()
        if token.strip(' ,.;:')
    }
    if not query_tokens or not doc_tokens:
      return 0.0
    return len(query_tokens & doc_tokens) / max(1, len(query_tokens))

  def upsert_listing(self, record: ListingRecord) -> None:
    """Insert or replace a seller listing in the portal index."""
    self._records[record.listing_id] = record
    document = record.to_document()
    embedding = self._embed_text(document, prefix='passage')
    if embedding is not None:
      self._embeddings[record.listing_id] = embedding

    if self._chroma_collection is None or embedding is None:
      return

    metadata = {
        'seller_id': record.seller_id,
        'seller_name': record.seller_name,
        'listing_price': float(record.listing_price),
        **record.flat_metadata(),
    }
    try:
      self._chroma_collection.upsert(
          ids=[record.listing_id],
          documents=[document],
          embeddings=[embedding.tolist()],
          metadatas=[metadata],
      )
    except Exception:
      self._chroma_collection = None

  def deactivate_listing(self, listing_id: str) -> None:
    record = self._records.get(listing_id)
    if record is not None:
      record.active = False

  def _hybrid_score(
      self,
      record: ListingRecord,
      query: str,
      query_embedding: np.ndarray | None,
      preferred_flat_types: Sequence[str],
      preferred_towns: Sequence[str],
      max_budget: float | None,
  ) -> float:
    score = 0.0
    record_document = record.to_document()
    score += 0.35 * self._cosine(
        query_embedding,
        self._embeddings.get(record.listing_id),
    )
    score += 0.20 * self._keyword_overlap(query, record_document)

    if preferred_flat_types and str(record.flat.flat_type) in set(preferred_flat_types):
      score += 0.20
    if preferred_towns and record.flat.town in set(preferred_towns):
      score += 0.15
    if max_budget is not None:
      if record.listing_price <= max_budget:
        score += 0.15
      else:
        overshoot = max(0.0, record.listing_price - max_budget)
        score -= min(0.20, overshoot / max(max_budget, 1.0))
    return score

  def _candidate_records(
      self,
      query_embedding: np.ndarray,
      limit: int,
  ) -> list[ListingRecord]:
    response = self._chroma_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=max(limit * 5, limit, 10),
    )
    candidate_ids = response.get('ids', [[]])
    ids = [str(record_id) for record_id in candidate_ids[0]] if candidate_ids else []
    candidates = [
        self._records[record_id]
        for record_id in ids
        if record_id in self._records and self._records[record_id].active
    ]
    if candidates:
      return candidates
    return [record for record in self._records.values() if record.active]

  def search(
      self,
      query: str,
      *,
      preferred_flat_types: Sequence[str] = (),
      preferred_towns: Sequence[str] = (),
      max_budget: float | None = None,
      limit: int = 5,
  ) -> list[listing_schemas.PortalSearchResult]:
    """Search active listings using hybrid vector, lexical, and metadata scoring."""
    query_embedding = self._embed_text(query, prefix='query')
    if query_embedding is None:
      raise RuntimeError('Failed to embed listing-portal query.')
    scored: list[tuple[float, ListingRecord]] = []
    for record in self._candidate_records(query_embedding, max(1, int(limit))):
      score = self._hybrid_score(
          record=record,
          query=query,
          query_embedding=query_embedding,
          preferred_flat_types=preferred_flat_types,
          preferred_towns=preferred_towns,
          max_budget=max_budget,
      )
      scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        record.to_search_result(score=score)
        for score, record in scored[: max(1, int(limit))]
    ]


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

    self.listings: dict[str, ListingRecord] = {}
    self.requests_by_listing: dict[str, list[listing_schemas.NegotiationRequest]] = {}
    self.search_results_by_buyer: dict[str, list[listing_schemas.PortalSearchResult]] = {}
    self.market_feedback_by_buyer: dict[str, str] = {}
    self.matched_pairs: list[listing_schemas.NegotiationMatch] = []
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
      buyer: listing_schemas.PortalBuyer,
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
    if avg_price <= buyer.budget.max_price:
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
    record = ListingRecord(
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
    effective_query = self._derive_query_from_preferences(buyer, search_query)
    results = self.retriever.search(
        effective_query,
        preferred_flat_types=buyer.preferences.flat_type,
        preferred_towns=buyer.preferences.towns,
        max_budget=buyer.budget.max_price,
        limit=5,
    )
    self.search_results_by_buyer[buyer_id] = results
    feedback = self._market_feedback(buyer, results)
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
          market_valuation_notes=feedback,
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
      record = ListingRecord(
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
