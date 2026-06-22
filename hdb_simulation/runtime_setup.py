"""Shared runtime setup helpers for HDB scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from absl import logging
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

from configs import DenseEmbedderConfig
from configs import get_active_llm_profile_name
from configs import get_llm_config
from configs import REPO_ROOT
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


def _resolve_local_model_name(model_name: str | None, default_model_name: str) -> str:
    resolved_model_name = model_name or default_model_name
    resolved_path = Path(resolved_model_name)
    if resolved_path.is_absolute():
        return str(resolved_path)

    repo_relative_path = REPO_ROOT / resolved_path
    if repo_relative_path.exists():
        return str(resolved_path)

    models_relative_path = REPO_ROOT / "models" / resolved_path
    if models_relative_path.exists():
        return str(Path("models") / resolved_path)

    return str(resolved_model_name)


def _absolute_repo_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _resolve_existing_local_path(
    model_name: str | None,
    default_model_name: str,
) -> str | None:
    resolved_model_name = _resolve_local_model_name(model_name, default_model_name)
    resolved_path = _absolute_repo_path(resolved_model_name)
    if resolved_path.exists():
        return str(resolved_path)
    return None


def initialise_local_model(
    *,
    model_name: str | None = None,
) -> VLLMLanguageModel:
    """Initialise the local vLLM model used across HDB workflows."""
    llm_config = get_llm_config()
    profile_name = get_active_llm_profile_name()
    resolved_model_name = _resolve_local_model_name(model_name, llm_config.MODEL_NAME)
    logging.info(
        "Initialising model %s with LLM profile %s.",
        resolved_model_name,
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

    if model_name is not None:
        model_kwargs["model_name"] = resolved_model_name

    # Optional: allow fully custom extra kwargs
    extra_kwargs = getattr(llm_config, "EXTRA_VLLM_KWARGS", {})
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    model = VLLMLanguageModel(**model_kwargs)

    logging.info(
        "Model %s initialised successfully with LLM profile %s.",
        resolved_model_name,
        profile_name,
    )
    return model


def initialise_download_model(
    *,
    model_name: str,
    download_dir: str | None = None,
) -> VLLMLanguageModel:
    """Initialise a vLLM model that may download its weights on first use."""
    llm_config = get_llm_config()
    profile_name = get_active_llm_profile_name()
    logging.info(
        "Initialising downloadable model %s with LLM profile %s.",
        model_name,
        profile_name,
    )

    config_to_model_kwargs = {
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

    model_kwargs: dict[str, Any] = {"model_name": model_name}
    for config_name, model_kwarg in config_to_model_kwargs.items():
        if hasattr(llm_config, config_name):
            value = getattr(llm_config, config_name)
            if value is not None:
                model_kwargs[model_kwarg] = value

    extra_kwargs = getattr(llm_config, "EXTRA_VLLM_KWARGS", {})
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)
    if download_dir:
        model_kwargs["download_dir"] = download_dir

    model = VLLMLanguageModel(**model_kwargs)
    logging.info(
        "Downloadable model %s initialised successfully with LLM profile %s.",
        model_name,
        profile_name,
    )
    return model


def initialise_model(
    *,
    model_source: str = "local",
    model_name: str | None = None,
    download_dir: str | None = None,
) -> VLLMLanguageModel:
    """Initialise a vLLM model from local weights or by downloading them."""
    normalized_source = str(model_source).strip().casefold()
    if normalized_source == "local":
        return initialise_local_model(model_name=model_name)
    if normalized_source == "download":
        if not model_name:
            raise ValueError(
                "model_name is required when model_source='download'."
            )
        return initialise_download_model(
            model_name=model_name,
            download_dir=download_dir,
        )
    raise ValueError(
        f"Unsupported model_source '{model_source}'. "
        "Expected one of: local, download."
    )


def initialise_dense_embedding_model(
    *,
    model_source: str = "local",
    model_name: str | None = None,
    device: str | None = None,
    local_files_only: bool | None = None,
    download_dir: str | None = None,
) -> SentenceTransformer:
    """Initialise the dense sentence embedder from local weights or downloads."""
    normalized_source = str(model_source).strip().casefold()
    resolved_device = device or DenseEmbedderConfig.DEVICE
    if normalized_source == "local":
        resolved_model_name = _resolve_local_model_name(
            model_name,
            DenseEmbedderConfig.MODEL_NAME,
        )
        resolved_local_files_only = (
            DenseEmbedderConfig.LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
    elif normalized_source == "download":
        resolved_model_name = model_name or DenseEmbedderConfig.REMOTE_MODEL_NAME
        resolved_local_files_only = False if local_files_only is None else local_files_only
    else:
        raise ValueError(
            f"Unsupported model_source '{model_source}' for dense embedder. "
            "Expected one of: local, download."
        )

    logging.info(
        'Initialising dense embedder %s on device %s.',
        resolved_model_name,
        resolved_device,
    )
    model_kwargs: dict[str, Any] = {
        'model_name_or_path': resolved_model_name,
        'device': resolved_device,
        'local_files_only': resolved_local_files_only,
    }
    if download_dir:
        model_kwargs['cache_folder'] = download_dir
    model = SentenceTransformer(**model_kwargs)
    logging.info(
        'Dense embedder %s initialised on %s.',
        resolved_model_name,
        resolved_device,
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


def _resolve_local_sparse_model_path(model_name: str | None = None) -> str | None:
    return _resolve_existing_local_path(
        model_name,
        SparseEmbedderConfig.LOCAL_MODEL_PATH,
    )


def initialise_sparse_embedding_model(
    *,
    model_source: str = "local",
    model_name: str | None = None,
    local_files_only: bool | None = None,
    download_dir: str | None = None,
) -> SparseTextEmbedding:
    """Initialise the sparse BM25 embedder from local weights or downloads."""
    normalized_source = str(model_source).strip().casefold()
    cache_dir = download_dir or _resolve_hf_hub_cache()
    resolved_remote_model_name = model_name or SparseEmbedderConfig.MODEL_NAME
    if normalized_source == "local":
        resolved_local_files_only = (
            SparseEmbedderConfig.LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
        specific_model_path = _resolve_local_sparse_model_path(model_name)
        if specific_model_path is None:
            specific_model_path = _resolve_fastembed_bm25_path(cache_dir)
    elif normalized_source == "download":
        resolved_local_files_only = False if local_files_only is None else local_files_only
        specific_model_path = None
    else:
        raise ValueError(
            f"Unsupported model_source '{model_source}' for sparse embedder. "
            "Expected one of: local, download."
        )

    model_kwargs = {
        'model_name': resolved_remote_model_name,
        'local_files_only': resolved_local_files_only,
    }
    if cache_dir:
        model_kwargs['cache_dir'] = cache_dir
        logging.info('Initialising sparse embedder from cache root %s.', cache_dir)
    else:
        logging.info('Initialising sparse embedder without explicit HF_HUB_CACHE.')
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
        resolved_remote_model_name,
    )
    return model
