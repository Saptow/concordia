"""Helpers for building or reusing preprocessed market segments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from absl import logging
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs import QdrantConfig, REPO_ROOT, SegmentConfig
from concordia.concordia.contrib.language_models.vllm.vllm_model import (
    VLLMLanguageModel,
)
from concordia.hdb_simulation.models.schemas.listing import qdrant as qdrant_schemas
from concordia.hdb_simulation.pipeline import flat_embedding
from concordia.hdb_simulation.pipeline import market_segment_processing


def read_jsonl_records(path: str | Path) -> list[dict[str, object]]:
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

    manifest_relative = (manifest_file.parent / candidate).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)

    repo_relative = (REPO_ROOT / candidate).resolve()
    if repo_relative.exists():
        return str(repo_relative)

    return str(manifest_relative)


def load_bundle_from_manifest(
    manifest_path: str | Path,
) -> tuple[dict[str, object], dict[str, str]]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))

    required_keys = (
        'flat_units_path',
        'sellers_path',
        'buyers_broad_path',
        'buyers_retained_path',
    )
    missing_keys = [
        key for key in required_keys if not str(manifest.get(key, '')).strip()
    ]
    if missing_keys:
        raise ValueError(
            f'Manifest {manifest_file} is missing required path keys: '
            f'{missing_keys}'
        )

    resolved_manifest = {
        key: _resolve_manifest_artifact_path(manifest_file, manifest[key])
        for key in required_keys
    }
    optional_path_keys = ('qdrant_db_path',)
    for key in optional_path_keys:
        raw_value = str(manifest.get(key, '')).strip()
        if not raw_value:
            continue
        resolved_manifest[key] = _resolve_manifest_artifact_path(
            manifest_file,
            raw_value,
        )
    optional_scalar_keys = ('qdrant_collection_name', 'market_segment_name')
    for key in optional_scalar_keys:
        raw_value = str(manifest.get(key, '')).strip()
        if raw_value:
            resolved_manifest[key] = raw_value

    bundle = {
        'flats': read_jsonl_records(resolved_manifest['flat_units_path']),
        'sellers': read_jsonl_records(resolved_manifest['sellers_path']),
        'buyers_broad': read_jsonl_records(resolved_manifest['buyers_broad_path']),
        'buyers_retained': read_jsonl_records(
            resolved_manifest['buyers_retained_path']
        ),
    }
    return bundle, resolved_manifest


def build_or_load_market_segment(
    *,
    segment_config: SegmentConfig,
    model: VLLMLanguageModel | None = None,
    market_manifest_path: str | Path | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
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
    dense_embedder: SentenceTransformer,
    sparse_embedder: SparseTextEmbedding | None = None,
    client: Any | None = None,
    rebuild: bool = False,
    listing_index_path: str | Path | None = None,
) -> tuple[dict[str, object], int]:
    """Ensure the market-segment listing index exists and return an enriched manifest."""
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
    requested_qdrant_db_path = str(listing_index_path or '').strip() or str(
        manifest.get('qdrant_db_path', '')
    ).strip()
    if requested_qdrant_db_path:
        persisted_qdrant_db_path = str(
            Path(requested_qdrant_db_path).expanduser().resolve()
        )
    else:
        persisted_qdrant_db_path = QdrantConfig.market_db_path(
            market_segment_name=market_segment_name,
        )

    enriched_manifest = dict(manifest)
    enriched_manifest['market_segment_name'] = market_segment_name
    enriched_manifest['qdrant_db_path'] = persisted_qdrant_db_path
    enriched_manifest['qdrant_collection_name'] = collection_name

    if not rebuild and requested_qdrant_db_path:
        existing_client = qdrant_schemas.make_qdrant_client(persisted_qdrant_db_path)
        if existing_client.collection_exists(collection_name):
            logging.info(
                'Reusing existing market-segment Qdrant index %s from %s.',
                collection_name,
                persisted_qdrant_db_path,
            )
            return enriched_manifest, 0
        logging.warning(
            'Manifest pointed to %s, but collection %s was missing; rebuilding listing index.',
            persisted_qdrant_db_path,
            collection_name,
        )

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
