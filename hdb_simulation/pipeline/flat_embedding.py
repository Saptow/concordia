import json
from pathlib import Path
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models as qdrant_models
from sentence_transformers import SentenceTransformer

from configs import QdrantConfig
from concordia.language_model import language_model
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


def _clean_text(value: Any) -> str:
  if value is None:
    return ''
  return str(value).strip()


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


def _value_or_unknown(
    value: Any,
    *,
    unknown_text: str = 'None',
) -> str:
  text = _clean_text(value)
  return text if text else unknown_text


def _format_currency(value: float | None) -> str:
  if value is None:
    return 'Price not specified.'
  return f'${float(value):,.0f}'


def _format_floor_area(
    value: Any,
    *,
    unknown_text: str = 'None',
) -> str:
  if value in (None, ''):
    return unknown_text
  return f'{float(value):.0f} sqm'


def _format_remaining_lease(
    value: Any,
    *,
    unknown_text: str = 'None',
) -> str:
  if value in (None, ''):
    return unknown_text
  return f'{float(value):.1f} years remaining'


def _format_joined_list(values: list[str], *, empty_text: str) -> str:
  return ', '.join(values) if values else empty_text


def _extract_price_trends(flat_payload: dict[str, Any]) -> PriceTrend | None:
  past_price_trends = flat_payload.get('past_price_trends', {})
  if not isinstance(past_price_trends, dict) or not past_price_trends:
    return None
  return PriceTrend.model_validate(past_price_trends)


def _build_nearby_amenities(flat_payload: dict[str, Any]) -> list[Amenity]:
  amenities = flat_payload.get('amenities', {})
  amenities = amenities if isinstance(amenities, dict) else {}
  nearby_amenities: list[Amenity] = []

  amenity_specs = (
      ('mrt', 'station_names', AmenityType.MRT, 'Within 1km'),
      ('primary_schools', 'school_names', AmenityType.SCHOOL, 'Within 2km'),
      ('hawker_centres', 'hawker_names', AmenityType.HAWKER, 'Within 1km'),
      ('malls', 'mall_names', AmenityType.MALL, 'Within 1km'),
  )
  for bucket_key, names_key, amenity_type, radius in amenity_specs:
    for name in amenities.get(bucket_key, {}).get(names_key, []):
      text = _clean_amenity_name(name)
      if text:
        nearby_amenities.append(
            Amenity(name=text, type=amenity_type, radius=radius)
        )

  return nearby_amenities


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
  clean_flat_type = _clean_text(flat_type)
  clean_address = _clean_text(address)
  clean_town = _clean_text(town)
  clean_storey_range = _clean_text(storey_range)

  if clean_flat_type:
    intro = f'This {clean_flat_type} HDB flat'
  else:
    intro = 'This HDB flat'
  if clean_address:
    intro += f' is at {clean_address}'
  if clean_town:
    intro += f' in {clean_town}'
  intro += '.'

  detail_bits: list[str] = []
  if floor_area_sqm not in (None, ''):
    detail_bits.append(f'about {_format_floor_area(floor_area_sqm)} of space')
  if clean_storey_range:
    detail_bits.append(f'in the {clean_storey_range} storey range')
  if remaining_lease_years not in (None, ''):
    detail_bits.append(f'with about {_format_remaining_lease(remaining_lease_years)}')

  sentences = [intro]
  if detail_bits:
    sentences.append(f'It offers {_join_natural(detail_bits)}.')
  else:
    sentences.append(
        'The listing does not clearly specify the floor area, storey range, or '
        'remaining lease.'
    )

  if mrt_names:
    sentences.append(
        f'Nearby MRT/LRT access includes {_join_natural(mrt_names)}.'
    )
  else:
    sentences.append('Nearby MRT/LRT options are not specified in the listing.')

  if mall_names:
    sentences.append(
        f'Nearby shopping options include {_join_natural(mall_names)}.'
    )
  else:
    sentences.append('Nearby shopping options are not specified in the listing.')

  if hawker_names:
    sentences.append(
        f'Nearby hawker centres include {_join_natural(hawker_names)}.'
    )
  else:
    sentences.append('Nearby hawker centres are not specified in the listing.')

  if school_names:
    sentences.append(
        'For families with young children, primary schools within 2 km include '
        f'{_join_natural(school_names)}.'
    )
  else:
    sentences.append(
        'Nearby primary schools for families with young children are not '
        'specified in the listing.'
    )

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

  address = _clean_text(flat_payload.get('address'))
  flat_type = _clean_text(flat_payload.get('flat_type'))
  storey_range = _clean_text(flat_payload.get('floor_range'))
  town = _clean_text(flat_payload.get('town'))
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

  past_price_trends = _extract_price_trends(flat_payload)
  transactions_6m = (
      str(int(past_price_trends.transactions_6m))
      if past_price_trends is not None
      else 'None'
  )
  price_range = 'None'
  if past_price_trends is not None:
    price_range = (
        f'{_format_currency(past_price_trends.min_price_6m)} to '
        f'{_format_currency(past_price_trends.max_price_6m)}'
    )

  lines = [
      f'**Listing Price:** {_format_currency(listing_price)}',
      '',
      '## Key Details',
      f'- Block / Address: {_value_or_unknown(address)}',
      (
          '- Flat Type: '
          f'{_value_or_unknown(flat_type, unknown_text="Flat type is not specified in the listing.")}'
      ),
      f'- Storey Range: {_value_or_unknown(storey_range)}',
      f'- Floor Area: {_format_floor_area(flat_payload.get("floor_area_sqm"))}',
      (
          '- Lease Commencement / Remaining Lease: '
          f'{_format_remaining_lease(flat_payload.get("remaining_lease_years"))}'
      ),
      '',
      '## Seller Description',
      seller_description,
      '',
      '## Transaction / Price Information',
      f'- Nearby transactions in past 6 months: {transactions_6m}',
      '',
      '## Nearby Amenities',
      (
          '- MRT / LRT: '
          f'{_format_joined_list(mrt_names, empty_text="None")}'
      ),
      (
          '- Primary Schools: '
          f'{_format_joined_list(school_names, empty_text="None")}'
      ),
      (
          '- Hawker centres: '
          f'{_format_joined_list(hawker_names, empty_text="None")}'
      ),
      (
          '- Malls: '
          f'{_format_joined_list(mall_names, empty_text="None")}'
      ),
      '',
      '## Price Trends',
      f'Price Range: {price_range}',
      '',
      '## Other Listing Flags',
      f'- Temporary Extension of Stay: {"Yes" if extension_of_stay else "No"}',
  ]
  return '\n'.join(lines)


def build_listing_record(
    flat_payload: dict[str, Any],
    *,
    seller_record: dict[str, Any] | None = None,
    model: language_model.LanguageModel | None = None,
    listed_week: int = 0,
    active: bool = False,
) -> qdrant_schemas.ListingRecord:
  seller_record = seller_record or {}
  seller_id = str(
      seller_record.get('seller_id') or f"seller::{flat_payload['flat_id']}"
  ).strip()
  seller_name = resolve_profile_name(
      seller_record,
      fallback_name=seller_id,
      model=model,
      role_label='seller',
  )
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
      address=_clean_text(flat_payload.get('address')),
      flat_type=_clean_text(flat_payload.get('flat_type')),
      town=_clean_text(flat_payload.get('town')),
      storey_range=_clean_text(flat_payload.get('floor_range')),
      floor_area_sqm=flat_payload.get('floor_area_sqm'),
      remaining_lease_years=flat_payload.get('remaining_lease_years'),
      mrt_names=mrt_names,
      school_names=school_names,
      hawker_names=hawker_names,
      mall_names=mall_names,
  )
  nearby_amenities = _build_nearby_amenities(flat_payload)
  past_price_trends = _extract_price_trends(flat_payload)

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
    model: language_model.LanguageModel | None = None,
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
          model=model,
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
      try:
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
      finally:
          close_fn = getattr(persistent_client, 'close', None)
          if callable(close_fn):
              close_fn()
  return records
