"""Helpers for building or reusing preprocessed market segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from absl import logging
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs import QdrantConfig, SegmentConfig
from concordia.concordia.contrib.language_models.vllm.vllm_model import (
    VLLMLanguageModel,
)
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas
from concordia.hdb_simulation.pipeline import flat_embedding
from concordia.hdb_simulation.pipeline import market_segment_processing


MANIFEST_BUNDLE_PATH_KEYS = (
    'flat_units_path',
    'sellers_path',
    'buyers_broad_path',
    'buyers_retained_path',
)
MANIFEST_OPTIONAL_PATH_KEYS = ('qdrant_db_path',)
MANIFEST_OPTIONAL_SCALAR_KEYS = (
    'qdrant_collection_name',
    'market_segment_name',
    'town',
)


def _read_jsonl_records(path: str | Path) -> list[dict[str, object]]:
    file_path = Path(path)
    return [
        json.loads(line)
        for line in file_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _resolve_manifest_artifact_path(
    manifest_file: Path,
    raw_path: str | Path,
) -> str:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((manifest_file.parent / candidate).resolve())


def _resolve_market_manifest(
    manifest_path: str | Path,
) -> dict[str, object]:
    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))

    missing_keys = [
        key
        for key in MANIFEST_BUNDLE_PATH_KEYS
        if not str(manifest.get(key, '')).strip()
    ]
    if missing_keys:
        raise ValueError(
            f'Manifest {manifest_file} is missing required path keys: '
            f'{missing_keys}'
        )

    resolved_manifest: dict[str, object] = {
        key: _resolve_manifest_artifact_path(manifest_file, manifest[key])
        for key in MANIFEST_BUNDLE_PATH_KEYS
    }
    for key in MANIFEST_OPTIONAL_PATH_KEYS:
        raw_value = str(manifest.get(key, '')).strip()
        if raw_value:
            resolved_manifest[key] = _resolve_manifest_artifact_path(
                manifest_file,
                raw_value,
            )
    for key in MANIFEST_OPTIONAL_SCALAR_KEYS:
        raw_value = manifest.get(key)
        if str(raw_value or '').strip():
            resolved_manifest[key] = str(raw_value).strip()

    planning_areas = manifest.get('planning_areas')
    if isinstance(planning_areas, list):
        resolved_manifest['planning_areas'] = [
            str(value).strip()
            for value in planning_areas
            if str(value).strip()
        ]

    return resolved_manifest


def _resolve_listing_index_paths(
    *,
    segment_config: SegmentConfig,
    manifest: dict[str, object],
    listing_index_path: str | Path | None,
) -> tuple[str, str, str]:
    market_segment_name = (
        str(manifest.get('market_segment_name', '')).strip()
        or segment_config.market_segment_name
    )
    collection_name = (
        str(manifest.get('qdrant_collection_name', '')).strip()
        or QdrantConfig.market_collection_name(
            market_segment_name=market_segment_name
        )
    )
    requested_db_path = (
        str(listing_index_path or '').strip()
        or str(manifest.get('qdrant_db_path', '')).strip()
    )
    persisted_db_path = (
        str(Path(requested_db_path).expanduser().resolve())
        if requested_db_path
        else QdrantConfig.market_db_path(
            market_segment_name=market_segment_name,
        )
    )
    return market_segment_name, collection_name, persisted_db_path


def load_bundle_from_manifest(
    manifest_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    resolved_manifest = _resolve_market_manifest(manifest_path)

    bundle = {
        'flats': _read_jsonl_records(resolved_manifest['flat_units_path']),
        'sellers': _read_jsonl_records(resolved_manifest['sellers_path']),
        'buyers_broad': _read_jsonl_records(resolved_manifest['buyers_broad_path']),
        'buyers_retained': _read_jsonl_records(
            resolved_manifest['buyers_retained_path']
        ),
    }
    return bundle, resolved_manifest


def build_or_load_market_segment(
    *,
    segment_config: SegmentConfig,
    model: VLLMLanguageModel | None = None,
    market_manifest_path: str | Path | None = None,
    model_source: str = 'local',
    download_dir: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build a market segment or reuse an existing manifest."""
    if market_manifest_path:
        logging.info(
            'Reusing existing market segment manifest from %s.',
            market_manifest_path,
        )
        return load_bundle_from_manifest(market_manifest_path)

    logging.info(
        'Running market segment preprocessing for %s (%s, segment=%s).',
        segment_config.town,
        segment_config.year,
        segment_config.segment_label,
    )
    bundle = market_segment_processing.build_transaction_conditioned_segment(
        segment_config,
        model=model,
        model_source=model_source,
        download_dir=download_dir,
    )
    market_segment_processing.save_segment_outputs(
        bundle,
        segment_config.output_dir,
    )
    manifest_path = segment_config.output_dir / 'manifest.json'
    _, resolved_manifest = load_bundle_from_manifest(manifest_path)
    return bundle, resolved_manifest


def ensure_market_segment_listing_index(
    *,
    segment_config: SegmentConfig,
    manifest: dict[str, object],
    model: VLLMLanguageModel | None = None,
    dense_embedder: SentenceTransformer,
    sparse_embedder: SparseTextEmbedding | None = None,
    client: Any | None = None,
    rebuild: bool = False,
    listing_index_path: str | Path | None = None,
) -> tuple[dict[str, object], int]:
    """Ensure the market-segment listing index exists and return an enriched manifest."""
    (
        market_segment_name,
        collection_name,
        persisted_qdrant_db_path,
    ) = _resolve_listing_index_paths(
        segment_config=segment_config,
        manifest=manifest,
        listing_index_path=listing_index_path,
    )

    enriched_manifest = dict(manifest)
    enriched_manifest['market_segment_name'] = market_segment_name
    enriched_manifest['qdrant_db_path'] = persisted_qdrant_db_path
    enriched_manifest['qdrant_collection_name'] = collection_name

    if not rebuild:
        existing_client = qdrant_schemas.make_qdrant_client(persisted_qdrant_db_path)
        try:
            if existing_client.collection_exists(collection_name):
                logging.info(
                    'Reusing existing market-segment Qdrant index %s from %s.',
                    collection_name,
                    persisted_qdrant_db_path,
                )
                return enriched_manifest, 0
            logging.info(
                'No existing market-segment Qdrant index %s found at %s; rebuilding.',
                collection_name,
                persisted_qdrant_db_path,
            )
        finally:
            close_fn = getattr(existing_client, 'close', None)
            if callable(close_fn):
                close_fn()

    logging.info(
        'Indexing generated flats into Qdrant collection %s.',
        collection_name,
    )
    runtime_client = client or qdrant_schemas.make_qdrant_client(
        QdrantConfig.DEFAULT_DB_PATH
    )
    if runtime_client.collection_exists(collection_name):
        runtime_client.delete_collection(collection_name)
    records = flat_embedding.index_market_segment_flats(
        flat_data_path=manifest['flat_units_path'],
        seller_data_path=manifest['sellers_path'],
        model=model,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        client=runtime_client,
        collection_name=collection_name,
        db_path=QdrantConfig.DEFAULT_DB_PATH,
        persist_db_path=persisted_qdrant_db_path,
        listed_week=0,
        active=False,
    )
    logging.info(
        'Indexed %s generated flats into Qdrant collection %s and saved the persistent copy to %s.',
        len(records),
        collection_name,
        persisted_qdrant_db_path,
    )
    return enriched_manifest, len(records)
