"""Shared runtime setup helpers for HDB scripts."""

from __future__ import annotations

import sys

from absl import logging
from sentence_transformers import SentenceTransformer

from configs import DenseEmbedderConfig
from configs import LLMConfig
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
