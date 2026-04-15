"""Shared runtime setup helpers for HDB scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from absl import logging
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs import DenseEmbedderConfig
from configs import get_active_llm_profile_name
from configs import get_llm_config
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
    llm_config = get_llm_config()
    profile_name = get_active_llm_profile_name()
    logging.info(
        "Initialising model %s with LLM profile %s.",
        llm_config.MODEL_NAME,
        profile_name,
    )

    # Map config attribute name -> VLLMLanguageModel kwarg name
    config_to_model_kwargs = {
        "MODEL_NAME": "model_name",
        "TRUST_REMOTE_CODE": "trust_remote_code",
        "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
        "MAX_MODEL_LEN": "max_model_len",
        "MAX_NUM_BATCHED_TOKENS": "max_num_batched_tokens",
        "MAX_NUM_SEQS": "max_num_seqs",
        "LIMIT_MM_PER_PROMPT": "limit_mm_per_prompt",
        "TENSOR_PARALLEL_SIZE": "tensor_parallel_size",
        "DISABLE_CUSTOM_ALL_REDUCE": "disable_custom_all_reduce",
        "ENFORCE_EAGER": "enforce_eager",
        "PERFORMANCE_MODE": "performance_mode",
        "QUANTIZATION": "quantization",
        "CACHE_DTYPE": "cache_dtype",
    }

    model_kwargs = {}

    for config_name, model_kwarg in config_to_model_kwargs.items():
        if hasattr(llm_config, config_name):
            value = getattr(llm_config, config_name)
            if value is not None:
                model_kwargs[model_kwarg] = value

    # Optional: allow fully custom extra kwargs
    extra_kwargs = getattr(llm_config, "EXTRA_VLLM_KWARGS", {})
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    model = VLLMLanguageModel(**model_kwargs)

    logging.info(
        "Model %s initialised successfully with LLM profile %s.",
        llm_config.MODEL_NAME,
        profile_name,
    )
    return model


def initialise_dense_embedding_model() -> SentenceTransformer:
    """Initialise the dense sentence embedder used by the listing portal."""
    logging.info(
        'Initialising dense embedder %s on device %s.',
        DenseEmbedderConfig.MODEL_NAME,
        DenseEmbedderConfig.DEVICE,
    )
    model = SentenceTransformer(
        DenseEmbedderConfig.MODEL_NAME,
        device=DenseEmbedderConfig.DEVICE,
        local_files_only=DenseEmbedderConfig.LOCAL_FILES_ONLY,
    )
    logging.info(
        'Dense embedder %s initialised on %s.',
        DenseEmbedderConfig.MODEL_NAME,
        DenseEmbedderConfig.DEVICE,
    )
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
    logging.info(
        'Sparse embedder %s initialised.',
        SparseEmbedderConfig.MODEL_NAME,
    )
    return model
