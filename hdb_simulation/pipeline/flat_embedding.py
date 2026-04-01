import json
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

from configs import QdrantConfig
from concordia.hdb_simulation.models.schemas.common import (
    Amenity,
    AmenityType,
    Flat,
    PriceTrend,
)
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas


def build_listing_summary(
    flat_payload: dict[str, Any],
    *,
    listing_price: float,
    extension_of_stay: bool = False,
) -> str:
  amenities = flat_payload.get('amenities', {})
  amenities = amenities if isinstance(amenities, dict) else {}
  mrt_names = [
      str(name).strip()
      for name in amenities.get('mrt', {}).get('station_names', [])
      if str(name).strip()
  ]
  school_names = [
      str(name).strip()
      for name in amenities.get('primary_schools', {}).get('school_names', [])
      if str(name).strip()
  ]
  hawker_names = [
      str(name).strip()
      for name in amenities.get('hawker_centres', {}).get('hawker_names', [])
      if str(name).strip()
  ]
  mall_names = [
      str(name).strip()
      for name in amenities.get('malls', {}).get('mall_names', [])
      if str(name).strip()
  ]

  address = str(flat_payload.get('address', '')).strip() or 'Not shown'
  flat_type = str(flat_payload.get('flat_type', '')).strip() or 'Not shown'
  storey_range = str(flat_payload.get('floor_range', '')).strip() or 'Not shown'
  floor_area = (
      f'{float(flat_payload["floor_area_sqm"]):.0f} sqm'
      if flat_payload.get('floor_area_sqm') not in (None, '')
      else 'Not shown'
  )

  lease_parts: list[str] = []
  if flat_payload.get('lease_commencement_year') not in (None, ''):
    lease_parts.append(str(flat_payload['lease_commencement_year']))
  if flat_payload.get('remaining_lease_years') not in (None, ''):
    lease_parts.append(
        f'{float(flat_payload["remaining_lease_years"]):.1f} years remaining'
    )
  lease_line = ' / '.join(lease_parts) if lease_parts else 'Not shown'

  town = str(flat_payload.get('town', '')).strip()
  seller_description = f'{flat_type} HDB flat at {address}'
  if town:
    seller_description += f' in {town}'
  detail_bits: list[str] = []
  if flat_payload.get('floor_area_sqm') not in (None, ''):
    detail_bits.append(f'about {float(flat_payload["floor_area_sqm"]):.0f} sqm')
  if str(flat_payload.get('floor_range', '')).strip():
    detail_bits.append(
        f'in the {str(flat_payload["floor_range"]).strip()} storey range'
    )
  if flat_payload.get('remaining_lease_years') not in (None, ''):
    detail_bits.append(
        'with roughly '
        f'{float(flat_payload["remaining_lease_years"]):.1f} years of lease remaining'
    )
  if detail_bits:
    seller_description += ', offering ' + ' and '.join(detail_bits)
  seller_description += '.'
  if mrt_names:
    seller_description += f' MRT access within 1km includes {", ".join(mrt_names)}.'
  if mall_names:
    seller_description += (
        f' Nearby shopping malls within 1km include {", ".join(mall_names)}.'
    )
  if hawker_names:
    seller_description += (
        f' Nearby hawker centres within 1km include {", ".join(hawker_names)}.'
    )
  if school_names:
    seller_description += (
        ' For families with young children, primary schools within 2km include '
        f'{", ".join(school_names)}.'
    )

  past_price_trends = flat_payload.get('past_price_trends', {})
  past_price_trends = past_price_trends if isinstance(past_price_trends, dict) else {}
  transactions_6m = (
      str(int(past_price_trends['transactions_6m']))
      if past_price_trends.get('transactions_6m') not in (None, '')
      else 'Not shown'
  )
  if (
      past_price_trends.get('min_price_6m') is not None
      and past_price_trends.get('max_price_6m') is not None
  ):
    price_range = (
        f'${float(past_price_trends["min_price_6m"]):,.0f} to '
        f'${float(past_price_trends["max_price_6m"]):,.0f}'
    )
  else:
    price_range = 'Not shown'

  return '\n'.join([
      f'**Listing Price:** ${float(listing_price):,.0f}',
      '',
      '## Key Details',
      f'- Block / Address: {address}',
      f'- Flat Type: {flat_type}',
      f'- Storey Range: {storey_range}',
      f'- Floor Area: {floor_area}',
      f'- Lease Commencement / Remaining Lease: {lease_line}',
      '',
      '## Seller Description',
      seller_description,
      '',
      '## Transaction / Price Information',
      (
          'Number of transactions of same flat type around here past 6 months: '
          f'{transactions_6m}'
      ),
      f'- Price Range: {price_range}',
      '',
      '## Nearby Amenities',
      '- MRT / LRT: ' + (', '.join(mrt_names) if mrt_names else 'None listed within 1km'),
      '- Primary Schools: '
      + (', '.join(school_names) if school_names else 'None listed within 2km'),
      '- Hawker centres: '
      + (', '.join(hawker_names) if hawker_names else 'None listed within 1km'),
      '- Malls: ' + (', '.join(mall_names) if mall_names else 'None listed within 1km'),
      '',
      '## Other Listing Flags',
      f'- Temporary Extension of Stay: {"Yes" if extension_of_stay else "No"}',
  ])


def build_listing_record(
    flat_payload: dict[str, Any],
    *,
    seller_record: dict[str, Any] | None = None,
    listed_week: int = 0,
    active: bool = False,
) -> qdrant_schemas.ListingRecord:
  seller_record = seller_record or {}
  seller_id = str(
      seller_record.get('seller_id') or f"seller::{flat_payload['flat_id']}"
  ).strip()
  seller_name = str(
      seller_record.get('seller_name')
      or seller_record.get('name')
      or seller_id
  ).strip() or seller_id
  extension_of_stay = bool(
      (seller_record.get('flat') or {}).get('extension_of_stay', False)
  )
  expectations = seller_record.get('expectations', {})
  if isinstance(expectations, dict) and expectations.get('max_price') is not None:
    listing_price = float(expectations['max_price'])
  else:
    listing_price = float(flat_payload.get('observed_resale_price', 0.0) or 0.0)

  listing_summary = build_listing_summary(
      flat_payload,
      listing_price=listing_price,
      extension_of_stay=extension_of_stay,
  )

  amenities = flat_payload.get('amenities', {})
  amenities = amenities if isinstance(amenities, dict) else {}
  nearby_amenities: list[Amenity] = []
  for name in amenities.get('mrt', {}).get('station_names', []):
    text = str(name).strip()
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.MRT, radius='Within 1km')
      )
  for name in amenities.get('primary_schools', {}).get('school_names', []):
    text = str(name).strip()
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.SCHOOL, radius='Within 2km')
      )
  for name in amenities.get('hawker_centres', {}).get('hawker_names', []):
    text = str(name).strip()
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.HAWKER, radius='Within 1km')
      )
  for name in amenities.get('malls', {}).get('mall_names', []):
    text = str(name).strip()
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.MALL, radius='Within 1km')
      )

  past_price_trends = flat_payload.get('past_price_trends', {})
  past_price_trends = (
      PriceTrend.model_validate(past_price_trends)
      if isinstance(past_price_trends, dict) and past_price_trends
      else None
  )

  return qdrant_schemas.ListingRecord(
      listing_id=qdrant_schemas.listing_id_for_seller(seller_id),
      seller_id=seller_id,
      seller_name=seller_name,
      listing_price=listing_price,
      listing_summary=listing_summary,
      flat=Flat(
          flat_type=str(flat_payload['flat_type']),
          address=str(flat_payload.get('address', '')).strip(),
          description=listing_summary.split(
              '## Seller Description\n',
              1,
          )[1].split('\n\n## Transaction', 1)[0],
          town=str(flat_payload.get('town', '')).strip(),
          storey_range=str(flat_payload.get('floor_range', '')).strip(),
          remaining_lease=float(flat_payload.get('remaining_lease_years', 0.0) or 0.0),
          contra=False,
          extension_of_stay=extension_of_stay,
          minimum_occupancy_period_completed=True,
          ethnic_eligibility='Unknown',
          spr_eligibility='Unknown',
          floor_area_sqm=float(flat_payload.get('floor_area_sqm', 0.0) or 0.0),
          nearby_amenities=nearby_amenities,
          past_price_trends=past_price_trends,
      ),
      listed_week=int(listed_week),
      active=bool(active),
  )


def ensure_listing_collection(
    *,
    client: QdrantClient,
    dense_embedder: SentenceTransformer,
    collection_name: str = QdrantConfig.DEFAULT_COLLECTION_NAME,
) -> None:
  if client.collection_exists(collection_name):
    return
  client.create_collection(
      collection_name=collection_name,
      vectors_config={
          qdrant_schemas.DENSE_EMBEDDINGS_KEY: qdrant_models.VectorParams(
              size=dense_embedder.get_sentence_embedding_dimension(),
              distance=qdrant_models.Distance.COSINE,
          ),
      },
      sparse_vectors_config={
          qdrant_schemas.SPARSE_EMBEDDINGS_KEY: qdrant_models.SparseVectorParams(),
      },
  )


def index_market_segment_flats(
    *,
    flat_data_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    dense_embedder: SentenceTransformer,
    client: QdrantClient | None = None,
    seller_data_path: str | Path | None = None,
    collection_name: str = QdrantConfig.DEFAULT_COLLECTION_NAME,
    db_path: str = QdrantConfig.DEFAULT_DB_PATH,
    persist_db_path: str | None = None,
    listed_week: int = 0,
    active: bool = False,
) -> list[qdrant_schemas.ListingRecord]:
  """Index market-segment flats after `market_segment_processing.py` has run."""
  if manifest_path is not None:
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    flat_data_path = manifest.get('flat_units_path') or manifest.get('flat_data_path')
    if seller_data_path is None:
      seller_data_path = manifest.get('sellers_path')
  if flat_data_path is None:
    raise ValueError('Provide either flat_data_path or manifest_path.')

  flat_path = Path(flat_data_path)
  if seller_data_path is None:
    candidate = flat_path.with_name('sellers.jsonl')
    seller_path = candidate if candidate.exists() else None
  else:
    seller_path = Path(seller_data_path)

  flat_rows = [
      json.loads(line)
      for line in flat_path.read_text(encoding='utf-8').splitlines()
      if line.strip()
  ]

  sellers_by_flat_id: dict[str, dict[str, Any]] = {}
  if seller_path is not None and seller_path.exists():
    for line in seller_path.read_text(encoding='utf-8').splitlines():
      if not line.strip():
        continue
      seller = json.loads(line)
      flat_id = str(seller.get('linked_flat_id', '')).strip()
      if flat_id:
        sellers_by_flat_id[flat_id] = seller

  records = [
      build_listing_record(
          flat_row,
          seller_record=sellers_by_flat_id.get(
              str(flat_row.get('flat_id', '')).strip()
          ),
          listed_week=listed_week,
          active=active,
      )
      for flat_row in flat_rows
  ]

  dense_vectors = dense_embedder.encode(
      [record.listing_summary for record in records],
      show_progress_bar=False,
  )
  points = [
      record.to_qdrant_point(
          dense_vector.tolist() if hasattr(dense_vector, 'tolist') else dense_vector
      )
      for record, dense_vector in zip(records, dense_vectors, strict=True)
  ]
  client = client or qdrant_schemas.make_qdrant_client(db_path)
  ensure_listing_collection(
      client=client,
      dense_embedder=dense_embedder,
      collection_name=collection_name,
  )
  client.upsert(
      collection_name=collection_name,
      points=points,
  )
  persist_target = str(persist_db_path or '').strip()
  if persist_target and persist_target != str(db_path).strip():
      persistent_client = qdrant_schemas.make_qdrant_client(persist_target)
      if persistent_client.collection_exists(collection_name):
          persistent_client.delete_collection(collection_name)
      ensure_listing_collection(
          client=persistent_client,
          dense_embedder=dense_embedder,
          collection_name=collection_name,
      )
      persistent_client.upsert(
          collection_name=collection_name,
          points=points,
      )
  return records
