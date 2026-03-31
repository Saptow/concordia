"""Qdrant-facing listing record schemas."""

from collections.abc import Sequence
from typing import Any
import uuid

from configs import QdrantConfig
from pydantic import BaseModel
from qdrant_client import models

from concordia.hdb_simulation.models.schemas.common import Flat

# Constants for Qdrant vector database
DENSE_EMBEDDINGS_KEY = QdrantConfig.DENSE_EMBEDDINGS_KEY
SPARSE_EMBEDDINGS_KEY = QdrantConfig.SPARSE_EMBEDDINGS_KEY
DEFAULT_COLLECTION_NAME = QdrantConfig.DEFAULT_COLLECTION_NAME
DEFAULT_DB_PATH = QdrantConfig.DEFAULT_DB_PATH


class ListingRecord(BaseModel):
    """Canonical listing metadata record stored in the portal index."""

    listing_id: str
    seller_id: str
    seller_name: str
    listing_price: float
    listing_summary: str # use this for vector and keyword indexing
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

    def qdrant_payload(self) -> dict[str, Any]:
      
        return {
            'listing_id': self.listing_id,
            'seller_id': self.seller_id,
            'seller_name': self.seller_name,
            'listing_price': float(self.listing_price),
            'listing_summary': self.listing_summary,
            'listed_week': int(self.listed_week),
            'active': bool(self.active),
            'flat_metadata': self.flat_metadata(),
        }

    def qdrant_point_id(self) -> str:
        """Return a deterministic UUID string accepted by Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, self.listing_id))

    def to_document(self) -> str:
        '''
        Render the listing record into a string for embedding and indexing.
        '''   
        lines=[] # no need to include prices
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

    def to_qdrant_point(
        self,
        embedding: Sequence[float],
    ) -> models.PointStruct:
        return models.PointStruct(
            id=self.qdrant_point_id(),
            vector={DENSE_EMBEDDINGS_KEY: [float(value) for value in embedding]},
            payload=self.qdrant_payload(),
        )

    def to_search_result(self, score: float) -> "PortalSearchResult":
        from concordia.hdb_simulation.models.schemas.listing.schema import PortalSearchResult

        return PortalSearchResult(
            **self.model_dump(mode='json'),
            score=float(score),
        )
