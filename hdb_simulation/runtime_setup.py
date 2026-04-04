"""Shared runtime setup helpers for HDB scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from absl import logging
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs import DenseEmbedderConfig
from configs import LLMConfig
from configs import SparseEmbedderConfig
from concordia.concordia.contrib.language_models.vllm.vllm_model import (
    VLLMLanguageModel,
)


def configure_logging() -> None:
    """Configure ABSL logging for CLI scripts."""
    logging.use_absl_handler()
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold('fatal')


def initialise_model() -> VLLMLanguageModel:
    """Initialise the local vLLM model used across HDB workflows."""
    logging.info('Initialising model...')
    model = VLLMLanguageModel(
        model_name=LLMConfig.MODEL_NAME,
        trust_remote_code=LLMConfig.TRUST_REMOTE_CODE,
        gpu_memory_utilization=LLMConfig.GPU_MEMORY_UTILIZATION,
        max_model_len=LLMConfig.MAX_MODEL_LEN,
        max_num_seqs=LLMConfig.MAX_NUM_SEQS,
        limit_mm_per_prompt=LLMConfig.LIMIT_MM_PER_PROMPT,
        tensor_parallel_size=LLMConfig.TENSOR_PARALLEL_SIZE,
        disable_custom_all_reduce=LLMConfig.DISABLE_CUSTOM_ALL_REDUCE,
        enforce_eager=LLMConfig.ENFORCE_EAGER,
    )
    logging.info('Model initialised successfully.')
    return model


def initialise_dense_embedding_model() -> SentenceTransformer:
    """Initialise the dense sentence embedder used by the listing portal."""
    model = SentenceTransformer(
        DenseEmbedderConfig.MODEL_NAME,
        device=DenseEmbedderConfig.DEVICE,
        local_files_only=DenseEmbedderConfig.LOCAL_FILES_ONLY,
    )
    logging.info('Dense embedder initialised.')
    return model


def _resolve_hf_hub_cache() -> str | None:
    cache_dir = os.environ.get('HF_HUB_CACHE')
    if cache_dir:
        return cache_dir
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return os.path.join(hf_home, 'hub')
    return None


def _resolve_fastembed_bm25_path(hf_hub_cache: str | None) -> str | None:
    if not hf_hub_cache:
        return None
    repo_dir = Path(hf_hub_cache) / 'models--Qdrant--bm25'
    if not repo_dir.exists():
        return None

    ref_main = repo_dir / 'refs' / 'main'
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding='utf-8').strip()
        if snapshot_name:
            snapshot_dir = repo_dir / 'snapshots' / snapshot_name
            if snapshot_dir.exists():
                return str(snapshot_dir)

    snapshots_dir = repo_dir / 'snapshots'
    if not snapshots_dir.exists():
        return None
    candidates = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not candidates:
        return None
    # Prefer the newest snapshot when refs/main is unavailable.
    latest_snapshot = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(latest_snapshot)


def initialise_sparse_embedding_model() -> SparseTextEmbedding:
    """Initialise the sparse BM25 embedder used by the listing portal."""
    cache_dir = _resolve_hf_hub_cache()
    model_kwargs = {
        'model_name': SparseEmbedderConfig.MODEL_NAME,
        'local_files_only': SparseEmbedderConfig.LOCAL_FILES_ONLY,
    }
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
        logging.info('Initialising sparse embedder from cache root %s.', cache_dir)
    else:
        logging.info('Initialising sparse embedder without explicit HF_HUB_CACHE.')
    specific_model_path = _resolve_fastembed_bm25_path(cache_dir)
    if specific_model_path:
        model_kwargs['specific_model_path'] = specific_model_path
        logging.info(
            'Using local BM25 snapshot at %s for offline sparse retrieval.',
            specific_model_path,
        )
    else:
        logging.warning(
            'Could not resolve a local BM25 snapshot under HF_HUB_CACHE. '
            'FastEmbed will rely on its cache_dir behavior.'
        )
    model = SparseTextEmbedding(**model_kwargs)
    logging.info('Sparse embedder initialised.')
    return model
