import json
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

from configs import QdrantConfig
from concordia.hdb_simulation.name_utils import resolve_profile_name
from concordia.hdb_simulation.models.schemas.common import (
    Amenity,
    AmenityType,
    Flat,
    PriceTrend,
)
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas


def _clean_amenity_name(value: Any) -> str:
  if value is None:
    return ''
  if isinstance(value, float) and value != value:
    return ''

  text = str(value).strip()
  if text.casefold() in {'', 'nan', 'none', 'null', '[]'}:
    return ''
  return text


def _join_natural(values: list[str]) -> str:
  if not values:
    return ''
  if len(values) == 1:
    return values[0]
  if len(values) == 2:
    return f'{values[0]} and {values[1]}'
  return f"{', '.join(values[:-1])}, and {values[-1]}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
  seen: set[str] = set()
  deduped: list[str] = []
  for value in values:
    if value in seen:
      continue
    seen.add(value)
    deduped.append(value)
  return deduped


def _extract_amenity_names(
    flat_payload: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
  amenities = flat_payload.get('amenities', {})
  amenities = amenities if isinstance(amenities, dict) else {}
  mrt_names = [
      cleaned
      for name in amenities.get('mrt', {}).get('station_names', [])
      if (cleaned := _clean_amenity_name(name))
  ]
  school_names = [
      cleaned
      for name in amenities.get('primary_schools', {}).get('school_names', [])
      if (cleaned := _clean_amenity_name(name))
  ]
  hawker_names = [
      cleaned
      for name in amenities.get('hawker_centres', {}).get('hawker_names', [])
      if (cleaned := _clean_amenity_name(name))
  ]
  mall_names = [
      cleaned
      for name in amenities.get('malls', {}).get('mall_names', [])
      if (cleaned := _clean_amenity_name(name))
  ]
  return (
      _dedupe_preserve_order(mrt_names),
      _dedupe_preserve_order(school_names),
      _dedupe_preserve_order(hawker_names),
      _dedupe_preserve_order(mall_names),
  )


def _build_seller_description(
    *,
    address: str,
    flat_type: str,
    town: str,
    storey_range: str,
    floor_area_sqm: Any,
    remaining_lease_years: Any,
    mrt_names: list[str],
    school_names: list[str],
    hawker_names: list[str],
    mall_names: list[str],
) -> str:
  intro = f'This {flat_type} HDB flat is at {address}'
  if town and town != 'Not shown':
    intro += f' in {town}'
  intro += '.'

  detail_bits: list[str] = []
  if floor_area_sqm not in (None, ''):
    detail_bits.append(f'about {float(floor_area_sqm):.0f} sqm of space')
  if storey_range and storey_range != 'Not shown':
    detail_bits.append(f'in the {storey_range} storey range')
  if remaining_lease_years not in (None, ''):
    detail_bits.append(
        f'with about {float(remaining_lease_years):.1f} years of lease remaining'
    )

  sentences = [intro]
  if detail_bits:
    sentences.append(f'It offers {_join_natural(detail_bits)}.')
  sentences.append(
      'Neighbourhood details are limited to amenities explicitly named in the '
      'listing payload. If something is not listed here, treat it as unknown '
      'rather than absent.'
  )

  if mrt_names:
    sentences.append(
        'The listing explicitly names these MRT/LRT stations within 1 km: '
        f'{_join_natural(mrt_names)}.'
    )
  else:
    sentences.append('The listing does not name any MRT/LRT stations within 1 km.')

  if mall_names:
    sentences.append(
        'The listing explicitly names these shopping malls within 1 km: '
        f'{_join_natural(mall_names)}.'
    )
  else:
    sentences.append('The listing does not name any shopping malls within 1 km.')

  if hawker_names:
    sentences.append(
        'The listing explicitly names these hawker centres within 1 km: '
        f'{_join_natural(hawker_names)}.'
    )
  else:
    sentences.append('The listing does not name any hawker centres within 1 km.')

  if school_names:
    sentences.append(
        'The listing explicitly names these primary schools within 2 km: '
        f'{_join_natural(school_names)}.'
    )
  else:
    sentences.append('The listing does not name any primary schools within 2 km.')

  return ' '.join(sentences)


def build_listing_summary(
    flat_payload: dict[str, Any],
    *,
    listing_price: float,
    extension_of_stay: bool = False,
) -> str:
  mrt_names, school_names, hawker_names, mall_names = _extract_amenity_names(
      flat_payload
  )

  address = str(flat_payload.get('address', '')).strip() or 'Not shown'
  flat_type = str(flat_payload.get('flat_type', '')).strip() or 'Not shown'
  storey_range = str(flat_payload.get('floor_range', '')).strip() or 'Not shown'
  town = str(flat_payload.get('town', '')).strip()
  seller_description = _build_seller_description(
      address=address,
      flat_type=flat_type,
      town=town,
      storey_range=storey_range,
      floor_area_sqm=flat_payload.get('floor_area_sqm'),
      remaining_lease_years=flat_payload.get('remaining_lease_years'),
      mrt_names=mrt_names,
      school_names=school_names,
      hawker_names=hawker_names,
      mall_names=mall_names,
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

  summary_sentences = [
      f'Listing price is SGD {float(listing_price):,.0f}.',
      seller_description,
  ]
  if transactions_6m != 'Not shown' and price_range != 'Not shown':
    summary_sentences.append(
        'Over the past 6 months, '
        f'{transactions_6m} similar nearby transactions were recorded, with prices '
        f'ranging from {price_range}.'
    )
  elif transactions_6m != 'Not shown':
    summary_sentences.append(
        'Over the past 6 months, '
        f'{transactions_6m} similar nearby transactions were recorded.'
    )
  elif price_range != 'Not shown':
    summary_sentences.append(
        f'Recent similar nearby transactions ranged from {price_range}.'
    )
  summary_sentences.append(
      'Temporary extension of stay is '
      f'{"available" if extension_of_stay else "not requested"}.'
  )
  return ' '.join(summary_sentences)


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
  seller_name = resolve_profile_name(seller_record, fallback_name=seller_id)
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
  mrt_names, school_names, hawker_names, mall_names = _extract_amenity_names(
      flat_payload
  )
  seller_description = _build_seller_description(
      address=str(flat_payload.get('address', '')).strip() or 'Not shown',
      flat_type=str(flat_payload.get('flat_type', '')).strip() or 'Not shown',
      town=str(flat_payload.get('town', '')).strip(),
      storey_range=str(flat_payload.get('floor_range', '')).strip() or 'Not shown',
      floor_area_sqm=flat_payload.get('floor_area_sqm'),
      remaining_lease_years=flat_payload.get('remaining_lease_years'),
      mrt_names=mrt_names,
      school_names=school_names,
      hawker_names=hawker_names,
      mall_names=mall_names,
  )

  amenities = flat_payload.get('amenities', {})
  amenities = amenities if isinstance(amenities, dict) else {}
  nearby_amenities: list[Amenity] = []
  for name in amenities.get('mrt', {}).get('station_names', []):
    text = _clean_amenity_name(name)
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.MRT, radius='Within 1km')
      )
  for name in amenities.get('primary_schools', {}).get('school_names', []):
    text = _clean_amenity_name(name)
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.SCHOOL, radius='Within 2km')
      )
  for name in amenities.get('hawker_centres', {}).get('hawker_names', []):
    text = _clean_amenity_name(name)
    if text:
      nearby_amenities.append(
          Amenity(name=text, type=AmenityType.HAWKER, radius='Within 1km')
      )
  for name in amenities.get('malls', {}).get('mall_names', []):
    text = _clean_amenity_name(name)
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
          description=seller_description,
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
    sparse_embedder: SparseTextEmbedding | None = None,
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

  documents = [record.to_document() for record in records]
  dense_vectors = dense_embedder.encode(documents, show_progress_bar=False)
  sparse_vectors = (
      list(sparse_embedder.embed(documents))
      if sparse_embedder is not None
      else [None] * len(records)
  )
  points = [
      record.to_qdrant_point(
          dense_vector.tolist() if hasattr(dense_vector, 'tolist') else dense_vector,
          sparse_embedding=sparse_vector,
      )
      for record, dense_vector, sparse_vector in zip(
          records,
          dense_vectors,
          sparse_vectors,
          strict=True,
      )
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
