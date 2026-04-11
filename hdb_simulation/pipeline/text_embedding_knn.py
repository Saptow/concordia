"""Clean a CSV text column, embed it, and compute cosine KNN matches.

This script is designed for a simple CSV input where one column contains free
text. It performs light normalization, optional lemmatization, dense embedding
with `BAAI/bge-large-en-v1.5`, and cosine nearest-neighbour search for
`k=2..5`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 5
WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LemmatizerConfig:
    mode: str
    transform: Callable[[str], str]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clean a CSV text column, embed it with BGE, and compute cosine "
            "nearest neighbours."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help=(
            "Name of the text column. If omitted, the script will infer it "
            "when the CSV has exactly one column."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for outputs. Defaults to a sibling folder named "
            "`<input_stem>_text_embedding_knn`."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model to load.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the embedder, for example `cpu` or `cuda`.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during embedding.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=DEFAULT_MIN_K,
        help="Smallest neighbour count to report.",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
        help="Largest neighbour count to report.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load the model from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--disable-lemmatization",
        action="store_true",
        help="Skip lemmatization and only apply whitespace normalization.",
    )
    return parser


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def _infer_text_column(frame: pd.DataFrame, requested_column: str | None) -> str:
    if requested_column:
        if requested_column not in frame.columns:
            available = ", ".join(str(column) for column in frame.columns)
            raise ValueError(
                f"Column `{requested_column}` was not found. Available columns: {available}"
            )
        return requested_column

    if len(frame.columns) != 1:
        available = ", ".join(str(column) for column in frame.columns)
        raise ValueError(
            "Could not infer the text column because the CSV has multiple "
            f"columns: {available}. Please pass `--text-column`."
        )
    return str(frame.columns[0])


def _normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _build_wordnet_lemmatizer() -> LemmatizerConfig | None:
    try:
        from nltk.corpus import wordnet
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        return None

    try:
        wordnet.ensure_loaded()
    except LookupError:
        return None

    lemmatizer = WordNetLemmatizer()

    def _lemmatize_word(word: str) -> str:
        lowered = word.lower()
        lemma = lemmatizer.lemmatize(lowered, pos="n")
        lemma = lemmatizer.lemmatize(lemma, pos="v")
        lemma = lemmatizer.lemmatize(lemma, pos="a")
        return lemma

    return LemmatizerConfig(mode="nltk_wordnet", transform=_lemmatize_word)


def _heuristic_lemmatize_word(word: str) -> str:
    lowered = word.lower()
    if len(lowered) <= 3:
        return lowered

    irregulars = {
        "children": "child",
        "men": "man",
        "women": "woman",
        "mice": "mouse",
        "geese": "goose",
        "better": "good",
        "best": "good",
    }
    if lowered in irregulars:
        return irregulars[lowered]

    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("xes", "zes", "ches", "shes", "sses")) and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return lowered[:-1]
    if lowered.endswith("ing") and len(lowered) > 5:
        stem = lowered[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("v"):
            return stem + "e"
        return stem
    if lowered.endswith("ed") and len(lowered) > 4:
        stem = lowered[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        if stem.endswith("v"):
            return stem + "e"
        return stem
    return lowered


def _build_lemmatizer(*, disable_lemmatization: bool) -> LemmatizerConfig:
    if disable_lemmatization:
        return LemmatizerConfig(mode="disabled", transform=lambda word: word.lower())

    wordnet_lemmatizer = _build_wordnet_lemmatizer()
    if wordnet_lemmatizer is not None:
        return wordnet_lemmatizer

    LOGGER.warning(
        "NLTK WordNet is unavailable. Falling back to a lightweight heuristic "
        "lemmatizer."
    )
    return LemmatizerConfig(mode="heuristic", transform=_heuristic_lemmatize_word)


def _clean_text(value: object, lemmatizer: LemmatizerConfig) -> str:
    text = _normalize_whitespace(str(value or ""))
    if not text:
        return ""

    cleaned = WORD_RE.sub(
        lambda match: lemmatizer.transform(match.group(0)),
        text.lower(),
    )
    return _normalize_whitespace(cleaned)


def _resolve_output_dir(input_csv: Path, requested_output_dir: str | None) -> Path:
    if requested_output_dir:
        return Path(requested_output_dir)
    return input_csv.parent / f"{input_csv.stem}_text_embedding_knn"


def _validate_k_range(
    row_count: int,
    requested_min_k: int,
    requested_max_k: int,
) -> tuple[int, int]:
    if requested_min_k < 1:
        raise ValueError("`--min-k` must be at least 1.")
    if requested_max_k < requested_min_k:
        raise ValueError("`--max-k` must be greater than or equal to `--min-k`.")
    if row_count < 2:
        raise ValueError("Need at least 2 rows to compute nearest neighbours.")

    max_possible_k = row_count - 1
    resolved_min_k = min(requested_min_k, max_possible_k)
    resolved_max_k = min(requested_max_k, max_possible_k)

    if resolved_max_k < requested_min_k:
        raise ValueError(
            f"Requested `k={requested_min_k}..{requested_max_k}`, but only "
            f"{row_count} row(s) are available."
        )

    if resolved_min_k != requested_min_k or resolved_max_k != requested_max_k:
        LOGGER.warning(
            "Capping requested K range from %s..%s to %s..%s because the input "
            "contains only %s row(s).",
            requested_min_k,
            requested_max_k,
            resolved_min_k,
            resolved_max_k,
            row_count,
        )
    return resolved_min_k, resolved_max_k


def _load_frame(input_csv: Path, text_column: str | None) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(input_csv)
    resolved_text_column = _infer_text_column(frame, text_column)
    frame = frame.copy()
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))
    frame[resolved_text_column] = frame[resolved_text_column].fillna("").astype(str)
    return frame, resolved_text_column


def _encode_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    local_files_only: bool,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading embedding model %s on %s.", model_name, device)
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )
    LOGGER.info("Encoding %s text row(s).", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _compute_knn_rows(
    frame: pd.DataFrame,
    *,
    text_column: str,
    embeddings: np.ndarray,
    min_k: int,
    max_k: int,
) -> pd.DataFrame:
    neighbour_count = min(len(frame), max_k + 1)
    knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=neighbour_count,
    )
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings, return_distance=True)

    rows: list[dict[str, object]] = []
    for source_offset, (row_distances, row_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        source_row_id = int(frame.iloc[source_offset]["row_id"])
        source_text = str(frame.iloc[source_offset][text_column])
        source_cleaned_text = str(frame.iloc[source_offset]["cleaned_text"])

        ranked_neighbours = [
            (int(candidate_index), float(candidate_distance))
            for candidate_index, candidate_distance in zip(
                row_indices,
                row_distances,
                strict=True,
            )
            if int(candidate_index) != source_offset
        ]

        for k in range(min_k, max_k + 1):
            selected_neighbours = ranked_neighbours[:k]
            for rank, (neighbor_offset, cosine_distance) in enumerate(
                selected_neighbours,
                start=1,
            ):
                neighbour_row = frame.iloc[neighbor_offset]
                rows.append(
                    {
                        "source_row_id": source_row_id,
                        "k": k,
                        "rank": rank,
                        "neighbor_row_id": int(neighbour_row["row_id"]),
                        "cosine_similarity": float(1.0 - cosine_distance),
                        "cosine_distance": cosine_distance,
                        "source_text": source_text,
                        "source_cleaned_text": source_cleaned_text,
                        "neighbor_text": str(neighbour_row[text_column]),
                        "neighbor_cleaned_text": str(neighbour_row["cleaned_text"]),
                    }
                )

    return pd.DataFrame(rows)


def _write_outputs(
    *,
    output_dir: Path,
    cleaned_frame: pd.DataFrame,
    embeddings: np.ndarray,
    knn_frame: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "cleaned_texts.csv"
    embeddings_path = output_dir / "embeddings.npy"
    knn_path = output_dir / "knn_neighbors.csv"
    metadata_path = output_dir / "run_metadata.json"

    cleaned_frame.to_csv(cleaned_path, index=False)
    np.save(embeddings_path, embeddings)
    knn_frame.to_csv(knn_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    LOGGER.info("Saved cleaned texts to %s.", cleaned_path)
    LOGGER.info("Saved embeddings to %s.", embeddings_path)
    LOGGER.info("Saved KNN results to %s.", knn_path)
    LOGGER.info("Saved run metadata to %s.", metadata_path)


def main(argv: list[str] | None = None) -> None:
    _configure_logging()
    args = _build_parser().parse_args(argv)

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV was not found: {input_csv}")

    frame, text_column = _load_frame(input_csv, args.text_column)
    min_k, max_k = _validate_k_range(len(frame), args.min_k, args.max_k)
    lemmatizer = _build_lemmatizer(
        disable_lemmatization=args.disable_lemmatization
    )

    LOGGER.info(
        "Cleaning %s row(s) from column `%s` using `%s` lemmatization.",
        len(frame),
        text_column,
        lemmatizer.mode,
    )
    frame["cleaned_text"] = frame[text_column].map(
        lambda value: _clean_text(value, lemmatizer)
    )

    embeddings = _encode_texts(
        frame["cleaned_text"].tolist(),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    knn_frame = _compute_knn_rows(
        frame,
        text_column=text_column,
        embeddings=embeddings,
        min_k=min_k,
        max_k=max_k,
    )

    output_dir = _resolve_output_dir(input_csv, args.output_dir)
    metadata = {
        "input_csv": str(input_csv.resolve()),
        "text_column": text_column,
        "row_count": int(len(frame)),
        "embedding_dimension": int(embeddings.shape[1]),
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "min_k": min_k,
        "max_k": max_k,
        "lemmatizer_mode": lemmatizer.mode,
        "local_files_only": bool(args.local_files_only),
        "output_dir": str(output_dir.resolve()),
    }
    _write_outputs(
        output_dir=output_dir,
        cleaned_frame=frame,
        embeddings=embeddings,
        knn_frame=knn_frame,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
