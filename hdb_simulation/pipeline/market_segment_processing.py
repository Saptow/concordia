"""Build a transaction-conditioned HDB resale market segment (steps 1-10).

This helper module intentionally stops before Concordia initialisation. It
builds the flat universe, seller pool, broad buyer pool, and retained buyer
pool for the transaction-conditioned Choa Chu Kang 2023 resale segment.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from absl import logging
from pydantic import BaseModel, Field

from configs import SegmentConfig
from concordia.concordia.contrib.language_models.vllm.vllm_model import (
    VLLMLanguageModel,
)
from concordia.hdb_simulation.models.schemas.common import (
    Amenity,
    AmenityType,
    BuyerBudgetRange,
    BuyerPreferenceItem,
    BuyerPreferenceProfile,
    Flat,
    SellerExpectationRange,
)
from concordia.hdb_simulation.pipeline.financial_feasibility import (
    INCOME_BANDS,
    compute_buyer_financials,
    resolve_income_band_upper,
)
from concordia.hdb_simulation.pipeline.resident_population_processing import (
    build_planning_area_age_groups,
)


DEFAULT_LLM_RETRIES = 3
MAX_REACHABLE_MARKET_SAMPLE_FLATS = 30
MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS = 5
MAX_BROAD_BUYER_GENERATION_ATTEMPT_FACTOR = 20
MARKET_QUANTILE_DOMINANCE_GRID = (0.2, 0.4, 0.6, 0.8)
MIN_FEASIBLE_RETAINED_BUYERS_PER_SELLER = 5
MIN_BUYER_INCOME_BAND_LOWER = 3000.0
DEFAULT_SURVEY_ARCHETYPES_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "survey"
)
DEFAULT_BUYER_ARCHETYPES_PATH = DEFAULT_SURVEY_ARCHETYPES_DIR / "buyer_archetypes.json"
DEFAULT_SELLER_ARCHETYPES_PATH = (
    DEFAULT_SURVEY_ARCHETYPES_DIR / "seller_archetypes.json"
)
BUYER_AGE_PRIOR_GROUPS = (
    "20 - 24",
    "25 - 29",
    "30 - 34",
    "35 - 39",
    "40 - 44",
    "45 - 49",
    "50 - 54",
    "55 - 59",
    "60 - 64",
    "65 - 69",
    "70 - 74",
    "75 - 79",
)


FLAT_TYPE_LABELS = {
    "1 ROOM": "1-Room",
    "2 ROOM": "2-Room",
    "3 ROOM": "3-Room",
    "4 ROOM": "4-Room",
    "5 ROOM": "5-Room",
    "EXECUTIVE": "Executive",
}

FLAT_TYPE_DWELLING_LABELS = {
    "1-Room": ["1- and 2-Room Flats", "HDB Dwellings"],
    "2-Room": ["1- and 2-Room Flats", "HDB Dwellings"],
    "3-Room": ["3-Room Flats", "HDB Dwellings"],
    "4-Room": ["4-Room Flats", "HDB Dwellings"],
    "5-Room": ["5-Room and Executive Flats", "5-Room Flats", "HDB Dwellings"],
    "Executive": [
        "5-Room and Executive Flats",
        "Executive Flats",
        "HDB Dwellings",
    ],
}

def _clean_amenity_name(value: Any) -> str:
    """Normalize amenity names and drop placeholder/null-like values."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "[]"}:
        return ""
    return text

class SellerMotivationProfile(BaseModel):
    seller_archetype_type: str = ""
    motivation_summary: str = ""
    reasons: list[str] = Field(default_factory=list)


# Generic text / file utilities.
def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _coerce_planning_areas(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        text = str(value or "").strip()
        items = [text] if text else []
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = _normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return tuple(cleaned)


def _planning_area_keys(value: Any) -> set[str]:
    return {_normalize_text(item) for item in _coerce_planning_areas(value)}


def _matches_planning_area(value: Any, planning_areas: Any) -> bool:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return False
    return normalized_value in _planning_area_keys(planning_areas)


def _planning_area_label(value: Any) -> str:
    planning_areas = _coerce_planning_areas(value)
    if not planning_areas:
        return ""
    if len(planning_areas) == 1:
        return planning_areas[0]
    return ", ".join(planning_areas)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_archetype_config(
    path: Path,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} archetype JSON not found at {path}. "
            "Generate it with analysis/generate_survey_archetypes.py or "
            "provide a valid config path."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} archetype config must be a non-empty JSON list.")
    logging.info("Loaded %s %s archetype entries from %s.", len(payload), label, path)
    return payload


def _flat_type_from_row(value: Any) -> str:
    key = str(value or "").strip().upper()
    return FLAT_TYPE_LABELS.get(key, str(value or "").strip())


# Sampling helpers used for demographic generation.
def _parse_age_band(label: str, *, adult_floor: int = 21) -> tuple[int, int]:
    text = str(label or "").strip()
    if not text:
        return adult_floor, max(adult_floor, adult_floor + 4)
    lowered = text.casefold()
    if lowered.startswith("below"):
        match = re.search(r"(\d+)", text)
        upper = int(match.group(1)) - 1 if match else adult_floor + 4
        return adult_floor, max(adult_floor, upper)
    if "over" in lowered:
        match = re.search(r"(\d+)", text)
        lower = int(match.group(1)) if match else adult_floor
        return max(adult_floor, lower), max(adult_floor, lower + 4)
    numbers = [int(part) for part in re.findall(r"\d+", text)]
    if len(numbers) >= 2:
        return max(adult_floor, numbers[0]), max(adult_floor, numbers[1])
    if len(numbers) == 1:
        lower = max(adult_floor, numbers[0])
        return lower, lower + 4
    return adult_floor, adult_floor + 4


def _sample_age_from_band(age_band: str, rng: random.Random) -> int:
    lower, upper = _parse_age_band(age_band)
    return rng.randint(lower, upper)


def _income_age_group_for_band(age_band: str, rng: random.Random) -> str:
    sampled_age = _sample_age_from_band(age_band, rng)
    lower = max(15, (sampled_age // 5) * 5)
    upper = lower + 4
    if lower >= 85:
        return "85 Years & Over"
    return f"{lower} - {upper} Years"


def _canonical_buyer_age_prior_group(value: Any) -> str:
    allowed_labels = {
        _normalize_text(label): label for label in BUYER_AGE_PRIOR_GROUPS
    }
    return allowed_labels.get(_normalize_text(value), "")


def _age_matches_target(value: Any, *, age: int) -> bool:
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value) == age

    text = str(value or "").strip()
    if not text:
        return False
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text)) == age

    lowered = text.casefold()
    lower, upper = _parse_age_band(text, adult_floor=0)
    if "over" in lowered:
        return age >= lower
    return lower <= age <= upper


def _weighted_choice(
    frame: pd.DataFrame, value_col: str, weight_col: str, rng: random.Random
) -> Any:
    if frame.empty:
        raise ValueError("Cannot sample from an empty frame.")
    weights = frame[weight_col].astype(float).tolist()
    values = frame[value_col].tolist()
    return rng.choices(values, weights=weights, k=1)[0]


def _filter_dwelling_distribution(
    frame: pd.DataFrame,
    flat_type_label: str,
) -> pd.DataFrame:
    candidate_labels = FLAT_TYPE_DWELLING_LABELS.get(flat_type_label, ["HDB Dwellings"])
    dwelling_col = frame.columns[0]
    normalized_dwelling_values = frame[dwelling_col].map(_normalize_text)
    for candidate in candidate_labels:
        subset = frame[normalized_dwelling_values == _normalize_text(candidate)]
        if not subset.empty:
            return subset.copy()
    return frame[
        normalized_dwelling_values == _normalize_text("HDB Dwellings")
    ].copy()


def _load_nemotron_pool(nemotron_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(nemotron_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {nemotron_dir}.")

    logging.info(
        "Loading Nemotron donor pool from %s parquet file(s) under %s.",
        len(parquet_files),
        nemotron_dir,
    )
    donors = pd.concat(
        [pd.read_parquet(path) for path in parquet_files],
        ignore_index=True,
    )
    donors = donors.fillna("")
    logging.info("Loaded %s Nemotron donor rows.", len(donors))
    return donors


def _sample_nemotron_donor(
    donors: pd.DataFrame,
    *,
    rng: random.Random,
    planning_area: Any,
    age: int | None = None,
    marital_status: str | None = None,
    education_level: str | None = None,
    occupation: str | None = None,
) -> dict[str, Any]:
    """Sample a donor row, narrowing by planning area and available profile fields."""
    candidates = donors.copy()
    target_planning_areas = _planning_area_keys(planning_area)

    if target_planning_areas:
        subset = candidates[
            candidates["planning_area"].map(_normalize_text).isin(target_planning_areas)
        ]
        if not subset.empty:
            candidates = subset

    if age is not None:
        for age_column in ("age", "age_group"):
            if age_column not in candidates.columns:
                continue
            subset = candidates[
                candidates[age_column].map(
                    lambda value: _age_matches_target(value, age=age)
                )
            ]
            if not subset.empty:
                candidates = subset
                break

    field_pairs = [
        ("marital_status", marital_status),
        ("education_level", education_level),
        ("occupation", occupation),
    ]
    for field_name, target_value in field_pairs:
        if not target_value:
            continue
        subset = candidates[
            candidates[field_name].map(_normalize_text) == _normalize_text(target_value)
        ]
        if not subset.empty:
            candidates = subset

    if candidates.empty:
        candidates = donors

    sampled = candidates.sample(n=1, random_state=rng.randint(0, 10_000_000)).iloc[0]
    return sampled.to_dict()


def _storey_midpoint(storey_range: str) -> float:
    numbers = [int(part) for part in re.findall(r"\d+", str(storey_range))]
    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2.0
    if len(numbers) == 1:
        return float(numbers[0])
    return 0.0


def _hedonic_feature_frame(flats: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "flat_type": flat["flat_type"],
                "flat_model": flat["flat_model"],
                "floor_area_sqm": float(flat["floor_area_sqm"]),
                "remaining_lease_years": float(flat["remaining_lease_years"]),
                "storey_mid": _storey_midpoint(flat["floor_range"]),
                "observed_resale_price": float(flat["observed_resale_price"]),
            }
            for flat in flats
        ]
    )


def _select_hedonic_training_flats(
    training_flats: list[dict[str, Any]],
    *,
    town: Any,
    preferred_flat_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter the fixed pre-window hedonic pool to town, then to preferred flat types if available."""
    target_flat_types = {flat_type for flat_type in (preferred_flat_types or []) if flat_type}
    town_flats = [
        flat for flat in training_flats if _matches_planning_area(flat["town"], town)
    ]
    if not target_flat_types:
        return town_flats

    filtered = [flat for flat in town_flats if flat["flat_type"] in target_flat_types]
    if filtered:
        return filtered
    return town_flats


def _estimate_hedonic_price(
    training_flats: list[dict[str, Any]],
    *,
    target_flat: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Fit a simple hedonic anchor on log prices and return log-space stats."""

    training_frame = _hedonic_feature_frame(training_flats)
    y = training_frame["observed_resale_price"].astype(float).to_numpy()
    log_y = np.log(np.clip(y, a_min=1.0, a_max=None))

    if target_flat is None:
        target_frame = training_frame.iloc[[0]].copy()
        target_frame.loc[:, "floor_area_sqm"] = training_frame["floor_area_sqm"].mean()
        target_frame.loc[:, "remaining_lease_years"] = training_frame[
            "remaining_lease_years"
        ].mean()
        target_frame.loc[:, "storey_mid"] = training_frame["storey_mid"].mean()
        target_frame.loc[:, "flat_type"] = training_frame["flat_type"].mode().iloc[0]
        target_frame.loc[:, "flat_model"] = training_frame["flat_model"].mode().iloc[0]
    else:
        target_frame = _hedonic_feature_frame([target_flat])

    training_design = pd.get_dummies(
        training_frame.drop(columns=["observed_resale_price"]),
        columns=["flat_type", "flat_model"],
        dtype=float,
    )
    target_design = pd.get_dummies(
        target_frame.drop(columns=["observed_resale_price"]),
        columns=["flat_type", "flat_model"],
        dtype=float,
    ).reindex(columns=training_design.columns, fill_value=0.0)

    if len(training_frame) < 8 or training_design.shape[1] >= len(training_frame):
        anchor_log = float(np.median(log_y))
        sigma_log = float(np.std(log_y, ddof=0)) if len(log_y) > 1 else 0.05
        return anchor_log, max(sigma_log, 0.05)

    x_train = np.column_stack(
        [np.ones(len(training_design)), training_design.to_numpy(dtype=float)]
    )
    x_target = np.column_stack(
        [np.ones(len(target_design)), target_design.to_numpy(dtype=float)]
    )

    beta, *_ = np.linalg.lstsq(x_train, log_y, rcond=None)
    fitted = x_train @ beta
    residuals = log_y - fitted
    sigma_log = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    anchor_log = float((x_target @ beta)[0])

    if not np.isfinite(anchor_log):
        anchor_log = float(np.median(log_y))
    sigma_log = max(sigma_log, 0.05)
    return anchor_log, sigma_log


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise json.JSONDecodeError("No JSON object found in model output.", text, 0)


def _call_model_for_json(
    model: VLLMLanguageModel,
    *,
    prompt: str,
    result_model: type[BaseModel],
    max_tokens: int = 800,
    retries: int = DEFAULT_LLM_RETRIES,
) -> BaseModel:
    """Call the model with a JSON schema and retry if parsing fails."""
    last_error: Exception | None = None
    current_prompt = prompt
    for _ in range(retries):
        response = model.sample_text(
            current_prompt,
            max_tokens=max_tokens,
            json_schema=result_model.model_json_schema(),
        )
        try:
            return result_model.model_validate(_extract_json_object(response))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            current_prompt = (
                current_prompt
                + "\n\nReturn JSON only and follow the requested schema exactly."
            )
    raise RuntimeError(
        f"Failed to parse model response as JSON after {retries} attempts: {last_error}"
    )


def _build_seller_motivation_generation_input(
    seller: dict[str, Any],
    seller_archetypes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "seller_profile": {
            "seller_id": seller["seller_id"],
            "age": seller["age"],
            "marital_status": seller["marital_status"],
            "education_level": seller["education_level"],
            "occupation_category": seller["occupation_category"],
            "industry": seller["industry"],
            "general_persona": seller["general_persona"],
            "initial_market_state": seller["initial_market_state"],
            "flat": seller["flat"],
            "expectations": seller["expectations"],
        },
        "seller_archetypes": seller_archetypes,
        "task": (
            "Given the seller profile and the seller archetype dataset, infer the "
            "seller's likely motivations and reasons for selling."
        ),
        "target_fields": {
            "seller_archetype_type": "string",
            "motivation_summary": "string",
            "reasons": ["string"],
        },
    }


def _build_seller_motivation_prompt(payload: dict[str, Any]) -> str:
    return f"""
# Role
You are generating seller motivations for an HDB resale simulation.

## Task
Read the seller profile and seller archetypes, then infer the most plausible
seller motivation profile.

## Input
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## Private Chain-of-Thought Process

Think step by step privately. Do not reveal your reasoning.

1. Read the seller's demographic profile, flat context, and current expectations.
2. Compare the seller against the provided seller archetypes.
3. Choose the most plausible seller archetype type when one is clearly supported.
4. Infer a concise motivation summary and a short list of realistic reasons.
5. Return only the final JSON object.

## Rules

- Ground the answer in the seller profile and flat context.
- Pick one seller archetype type from the provided archetypes when possible.
- Keep the motivation summary concise and realistic.
- Keep `reasons` short, concrete, and behaviorally plausible.
- Return JSON only. Do not wrap the answer in markdown fences.
"""


def _compact_flat_for_preference_summary(flat: dict[str, Any]) -> dict[str, Any]:
    amenities = flat.get("amenities", {})
    past_price_trends = flat.get("past_price_trends", {})
    return {
        "flat_id": flat["flat_id"],
        "town": flat["town"],
        "flat_type": flat["flat_type"],
        "observed_resale_price": round(float(flat["observed_resale_price"]), 2),
        "floor_area_sqm": round(float(flat["floor_area_sqm"]), 2),
        "remaining_lease_years": round(float(flat["remaining_lease_years"]), 2),
        "amenity_counts": {
            "mrt": int(amenities.get("mrt", {}).get("count", 0)),
            "primary_schools": int(amenities.get("primary_schools", {}).get("count", 0)),
            "malls": int(amenities.get("malls", {}).get("count", 0)),
            "hawker_centres": int(amenities.get("hawker_centres", {}).get("count", 0)),
        },
        "past_price_trends": {
            "transactions_6m": int(past_price_trends.get("transactions_6m", 0)),
            "min_price_6m": round(float(past_price_trends.get("min_price_6m", 0.0)), 2),
            "max_price_6m": round(float(past_price_trends.get("max_price_6m", 0.0)), 2),
        },
    }


def _sample_flats_uniformly(
    flats: list[dict[str, Any]],
    *,
    cap: int = MAX_REACHABLE_MARKET_SAMPLE_FLATS,
) -> list[dict[str, Any]]:
    if cap <= 0 or not flats:
        return []
    if len(flats) <= cap:
        return list(flats)

    step = (len(flats) - 1) / float(cap - 1) if cap > 1 else 0.0
    sampled_indices: list[int] = []
    seen_indices: set[int] = set()
    for position in range(cap):
        candidate_index = int(round(position * step)) if cap > 1 else 0
        if candidate_index in seen_indices:
            continue
        sampled_indices.append(candidate_index)
        seen_indices.add(candidate_index)

    if len(sampled_indices) < cap:
        for candidate_index in range(len(flats)):
            if candidate_index in seen_indices:
                continue
            sampled_indices.append(candidate_index)
            seen_indices.add(candidate_index)
            if len(sampled_indices) >= cap:
                break

    sampled_indices.sort()
    return [flats[index] for index in sampled_indices]


def _sample_indices_uniformly(total_count: int, cap: int) -> list[int]:
    if cap <= 0 or total_count <= 0:
        return []
    if total_count <= cap:
        return list(range(total_count))

    step = (total_count - 1) / float(cap - 1) if cap > 1 else 0.0
    sampled_indices: list[int] = []
    seen_indices: set[int] = set()
    for position in range(cap):
        candidate_index = int(round(position * step)) if cap > 1 else 0
        if candidate_index in seen_indices:
            continue
        sampled_indices.append(candidate_index)
        seen_indices.add(candidate_index)

    if len(sampled_indices) < cap:
        for candidate_index in range(total_count):
            if candidate_index in seen_indices:
                continue
            sampled_indices.append(candidate_index)
            seen_indices.add(candidate_index)
            if len(sampled_indices) >= cap:
                break

    sampled_indices.sort()
    return sampled_indices


def _allocate_stratified_counts(
    stratum_sizes: dict[str, int],
    cap: int,
) -> dict[str, int]:
    """Allocate a fixed sample size proportionally across strata."""
    total_count = sum(max(0, int(size)) for size in stratum_sizes.values())
    if cap <= 0 or total_count <= 0:
        return {stratum: 0 for stratum in stratum_sizes}

    capped_total = min(cap, total_count)
    allocated: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, size in stratum_sizes.items():
        normalized_size = max(0, int(size))
        exact_share = (normalized_size / total_count) * capped_total
        floor_share = min(normalized_size, int(math.floor(exact_share)))
        allocated[stratum] = floor_share
        remainders.append((exact_share - floor_share, stratum))

    remaining = capped_total - sum(allocated.values())
    remainders.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    while remaining > 0:
        progressed = False
        for _, stratum in remainders:
            capacity_left = max(0, int(stratum_sizes[stratum])) - allocated[stratum]
            if capacity_left <= 0:
                continue
            allocated[stratum] += 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break

    return allocated


def _build_market_bucket_summary(
    flats: list[dict[str, Any]],
    *,
    bucket_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for flat in flats:
        key = str(flat.get(bucket_key, "")).strip()
        grouped.setdefault(key, []).append(flat)

    summaries: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket_flats = sorted(
            grouped[key],
            key=lambda flat: (
                float(flat["observed_resale_price"]),
                float(flat["floor_area_sqm"]),
                str(flat["flat_id"]),
            ),
        )
        prices = [float(flat["observed_resale_price"]) for flat in bucket_flats]
        floor_areas = [float(flat["floor_area_sqm"]) for flat in bucket_flats]
        lease_years = [float(flat["remaining_lease_years"]) for flat in bucket_flats]
        amenity_counts = {
            "mrt": [int(flat.get("amenities", {}).get("mrt", {}).get("count", 0)) for flat in bucket_flats],
            "primary_schools": [
                int(flat.get("amenities", {}).get("primary_schools", {}).get("count", 0))
                for flat in bucket_flats
            ],
            "malls": [int(flat.get("amenities", {}).get("malls", {}).get("count", 0)) for flat in bucket_flats],
            "hawker_centres": [
                int(flat.get("amenities", {}).get("hawker_centres", {}).get("count", 0))
                for flat in bucket_flats
            ],
        }
        summaries.append(
            {
                bucket_key: key,
                "reachable_flats": len(bucket_flats),
                "price_range": {
                    "min": round(min(prices), 2),
                    "max": round(max(prices), 2),
                },
                "floor_area_sqm_range": {
                    "min": round(min(floor_areas), 2),
                    "max": round(max(floor_areas), 2),
                },
                "remaining_lease_years_range": {
                    "min": round(min(lease_years), 2),
                    "max": round(max(lease_years), 2),
                },
                "amenity_profile": {
                    amenity_name: {
                        "flats_with_access": sum(1 for count in counts if count > 0),
                        "max_count": max(counts, default=0),
                    }
                    for amenity_name, counts in amenity_counts.items()
                },
            }
        )
    return summaries


def _build_reachable_market_summary(
    reachable_flats: list[dict[str, Any]],
) -> dict[str, Any]:
    if not reachable_flats:
        return {
            "total_reachable_flats": 0,
            "reachable_towns": [],
            "reachable_flat_types": [],
            "overall_price_range": None,
            "town_summaries": [],
            "flat_type_summaries": [],
        }

    sorted_flats = sorted(
        reachable_flats,
        key=lambda flat: (
            str(flat.get("town", "")),
            str(flat.get("flat_type", "")),
            float(flat.get("observed_resale_price", 0.0)),
            str(flat.get("flat_id", "")),
        ),
    )
    prices = [float(flat["observed_resale_price"]) for flat in sorted_flats]
    return {
        "total_reachable_flats": len(sorted_flats),
        "reachable_towns": sorted(
            {str(flat.get("town", "")).strip() for flat in sorted_flats if str(flat.get("town", "")).strip()}
        ),
        "reachable_flat_types": sorted(
            {
                str(flat.get("flat_type", "")).strip()
                for flat in sorted_flats
                if str(flat.get("flat_type", "")).strip()
            }
        ),
        "overall_price_range": {
            "min": round(min(prices), 2),
            "max": round(max(prices), 2),
        },
        "town_summaries": _build_market_bucket_summary(sorted_flats, bucket_key="town"),
        "flat_type_summaries": _build_market_bucket_summary(sorted_flats, bucket_key="flat_type"),
    }


def _build_preference_classification_input(
    buyer: dict[str, Any],
    reachable_flats: list[dict[str, Any]],
    archetypes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "buyer_metadata": {
            "buyer_id": buyer["buyer_id"],
            "town": buyer["town"],
            "age": buyer["age"],
            "income_band": buyer["income_band"],
            "marital_status": buyer["marital_status"],
            "education_level": buyer["education_level"],
            "occupation_category": buyer["occupation_category"],
            "industry": buyer["industry"],
            "general_persona": buyer["general_persona"],
        },
        "financials": buyer["financials"],
        "reachable_market_summary": _build_reachable_market_summary(reachable_flats),
        "archetypes": archetypes,
    }


def _build_buyer_preference_prompt(payload: dict[str, Any]) -> str:
    return f"""# Role

You are inferring buyer preference profiles for an HDB resale simulation.

## Task

Read the buyer metadata, the hierarchical reachable-market summary, and the
archetypes. Infer the buyer's most likely preference profile.

## Input

```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## Private Chain-of-Thought Process

Think step by step privately. Do not reveal your reasoning.

1. Review the buyer's profile and financial context.
2. Inspect the reachable-market summary from overall market level, then town
   level, then flat-type level.
3. Compare the buyer against the provided preference archetypes.
4. Infer the most plausible flat-type and town preferences, constrained by the
   reachable market summary.
5. Return a `preferences` list where each item has a `category`, `description`,
   and numeric `strength` between 0 and 1.
6. Include at least one `flat_type` preference and at least one `town`
   preference.
7. Return only the final JSON object.

## Rules

- Use only towns and flat types supported by the reachable market summary.
- Do not invent unreachable towns or flat types.
- Prefer market patterns that appear consistently across the summary buckets.
- Use only the supported categories from the schema.
- Keep each preference description concise but specific.
- Use `strength` to reflect how strongly the language suggests that preference:
  near 1.0 for must-haves, around 0.5 for ordinary preferences, and near 0.0
  for weak nice-to-have signals.
- Do not reveal your private reasoning or the market summary in the final answer.
- Return JSON only. Do not wrap the answer in markdown fences.
"""


def _constrain_buyer_preferences_to_reachable_market(
    buyer: dict[str, Any],
    preferences: BuyerPreferenceProfile,
) -> BuyerPreferenceProfile:
    """Clip flat-type and town preferences to the buyer's reachable market."""
    payload = buyer.get("preference_classification_input", {})
    reachable_market_summary = payload.get("reachable_market_summary", {})
    allowed_flat_types = [
        str(value).strip()
        for value in reachable_market_summary.get("reachable_flat_types", ())
        if str(value).strip()
    ]
    allowed_towns = [
        str(value).strip()
        for value in reachable_market_summary.get("reachable_towns", ())
        if str(value).strip()
    ]
    allowed_town_keys = _planning_area_keys(allowed_towns)

    constrained_items: list[BuyerPreferenceItem] = []
    original_flat_type_items = preferences.items_for("flat_type")
    original_town_items = preferences.items_for("town")
    for item in preferences.preferences:
        if item.category == "flat_type":
            if item.description.strip() in allowed_flat_types:
                constrained_items.append(item)
            continue
        if item.category == "town":
            if _normalize_text(item.description) in allowed_town_keys:
                constrained_items.append(item)
            continue
        constrained_items.append(item)

    if not any(item.category == "flat_type" for item in constrained_items) and allowed_flat_types:
        constrained_items.insert(
            0,
            BuyerPreferenceItem(
                category="flat_type",
                description=allowed_flat_types[0],
                strength=(
                    max(item.strength for item in original_flat_type_items)
                    if original_flat_type_items
                    else 0.75
                ),
            ),
        )
    if not any(item.category == "town" for item in constrained_items) and allowed_towns:
        constrained_items.insert(
            1 if constrained_items and constrained_items[0].category == "flat_type" else 0,
            BuyerPreferenceItem(
                category="town",
                description=allowed_towns[0],
                strength=(
                    max(item.strength for item in original_town_items)
                    if original_town_items
                    else 0.75
                ),
            ),
        )

    return BuyerPreferenceProfile.model_validate(
        {"preferences": [item.model_dump() for item in constrained_items]}
    )


def _load_town_transactions(config: SegmentConfig) -> pd.DataFrame:
    """Load all successful resale transactions for the configured town."""
    frame = pd.read_csv(config.resale_path)
    town_rows = frame[
        frame["town"].map(lambda value: _matches_planning_area(value, config.town))
    ].copy()
    if town_rows.empty:
        raise ValueError(
            f"No resale rows found for planning_area={_planning_area_label(config.town)!r}."
        )

    town_rows["Date"] = pd.to_datetime(town_rows["Date"])
    town_rows["flat_type_label"] = town_rows["flat_type"].map(_flat_type_from_row)
    town_rows["sale_id"] = town_rows["sale_id"].astype(int)
    return town_rows.sort_values(["Date", "sale_id"]).reset_index(drop=True)


def _load_transactions(config: SegmentConfig) -> pd.DataFrame:
    """Load the simulation window for the configured town-year segment."""
    town_rows = _load_town_transactions(config)
    filtered = town_rows[town_rows["year"].astype(int) == int(config.year)].copy()
    filtered = filtered[filtered["Date"].dt.month.isin(config.segment_months)].copy()
    if filtered.empty:
        raise ValueError(
            "No successful resale rows found for "
            f"planning_area={_planning_area_label(config.town)!r} year={config.year} "
            f"segment={config.segment_label!r}."
        )
    return filtered.reset_index(drop=True)


def _restrain_transactions(
    transactions: pd.DataFrame,
    *,
    restrained_seller_count: int | None,
    sampled_flat_ratio: float | None = None,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Downsample transactions while preserving the observed flat-type mix."""
    if sampled_flat_ratio is not None:
        if not (0 < sampled_flat_ratio <= 1):
            raise ValueError("sampled_flat_ratio must be within (0, 1].")
        if sampled_flat_ratio >= 1:
            return transactions.reset_index(drop=True)

        selected_frames: list[pd.DataFrame] = []
        normalized_towns = transactions["town"].map(_normalize_text)
        ordered_towns = list(dict.fromkeys(normalized_towns.tolist()))
        for normalized_town in ordered_towns:
            town_rows = transactions[normalized_towns == normalized_town].copy()
            if town_rows.empty:
                continue
            town_target_count = max(
                1,
                math.ceil(len(town_rows) * sampled_flat_ratio),
            )
            if town_target_count >= len(town_rows):
                selected_frames.append(town_rows)
                continue
            selected_frames.append(
                _restrain_transactions(
                    town_rows,
                    restrained_seller_count=town_target_count,
                    sampled_flat_ratio=None,
                    rng=rng,
                )
            )

        if not selected_frames:
            return transactions.iloc[0:0].copy().reset_index(drop=True)
        return (
            pd.concat(selected_frames, ignore_index=True)
            .sort_values(["Date", "sale_id"])
            .reset_index(drop=True)
        )

    if restrained_seller_count is None:
        return transactions.reset_index(drop=True)
    if restrained_seller_count <= 0:
        raise ValueError("restrained_seller_count must be positive when provided.")
    if len(transactions) <= restrained_seller_count:
        return transactions.reset_index(drop=True)

    flat_type_series = (
        transactions["flat_type_label"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .replace("", "unknown")
    )
    stratum_sizes = {
        str(flat_type): int(count)
        for flat_type, count in flat_type_series.value_counts(sort=False).items()
    }
    allocated_counts = _allocate_stratified_counts(
        stratum_sizes,
        restrained_seller_count,
    )

    selected_indices: list[int] = []
    for flat_type, sample_count in sorted(allocated_counts.items()):
        if sample_count <= 0:
            continue
        stratum_indices = flat_type_series[flat_type_series == flat_type].index.tolist()
        if sample_count >= len(stratum_indices):
            selected_indices.extend(stratum_indices)
            continue
        if rng is None:
            relative_indices = _sample_indices_uniformly(
                total_count=len(stratum_indices),
                cap=sample_count,
            )
            selected_indices.extend(
                stratum_indices[index] for index in relative_indices
            )
        else:
            selected_indices.extend(rng.sample(stratum_indices, k=sample_count))

    selected_indices = sorted(set(selected_indices))
    if len(selected_indices) < restrained_seller_count:
        selected_set = set(selected_indices)
        remaining_indices = [
            index for index in transactions.index.tolist() if index not in selected_set
        ]
        deficit = restrained_seller_count - len(selected_indices)
        if rng is None:
            relative_indices = _sample_indices_uniformly(
                total_count=len(remaining_indices),
                cap=deficit,
            )
            selected_indices.extend(
                remaining_indices[index] for index in relative_indices
            )
        else:
            selected_indices.extend(rng.sample(remaining_indices, k=deficit))
        selected_indices = sorted(selected_indices)

    return transactions.iloc[selected_indices].copy().reset_index(drop=True)


def _build_hedonic_training_flats(
    transactions: pd.DataFrame,
    *,
    town: Any,
    window_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Build the fixed 6-month pre-window pool used for hedonic calibration."""
    cutoff_date = window_start - pd.Timedelta(days=183)
    training_rows = transactions[
        (transactions["town"].map(lambda value: _matches_planning_area(value, town)))
        & (transactions["Date"] < window_start)
        & (transactions["Date"] >= cutoff_date)
    ].copy()

    training_flats: list[dict[str, Any]] = []
    for row in training_rows.itertuples(index=False):
        training_flats.append(
            {
                "town": str(row.town).strip(),
                "transaction_date": pd.Timestamp(row.Date).date().isoformat(),
                "flat_type": str(row.flat_type_label),
                "flat_model": str(row.flat_model).strip(),
                "floor_area_sqm": float(row.floor_area_sqm),
                "floor_range": str(row.storey_range).strip(),
                "remaining_lease_years": round(float(row.remaining_lease) / 12.0, 2),
                "observed_resale_price": float(row.resale_price),
            }
        )
    return training_flats


def _build_past_price_trends(
    transactions: pd.DataFrame,
    *,
    town: str,
    flat_type: str,
    reference_date: pd.Timestamp,
    fallback_price: float,
) -> dict[str, Any]:
    """Summarise same-town, same-flat-type transactions in the prior 6 months."""
    cutoff_date = reference_date - pd.Timedelta(days=183)
    history = transactions[
        (transactions["town"].map(_normalize_text) == _normalize_text(town))
        & (transactions["flat_type_label"] == flat_type)
        & (transactions["Date"] < reference_date)
        & (transactions["Date"] >= cutoff_date)
    ].copy()

    if history.empty:
        return {
            "transactions_6m": 0,
            "min_price_6m": round(float(fallback_price), 2),
            "max_price_6m": round(float(fallback_price), 2),
        }

    prices = history["resale_price"].astype(float)
    return {
        "transactions_6m": int(len(history)),
        "min_price_6m": round(float(prices.min()), 2),
        "max_price_6m": round(float(prices.max()), 2),
    }


def _build_flat_universe(
    transactions: pd.DataFrame,
    town_transactions: pd.DataFrame,
    config: SegmentConfig,
) -> list[dict[str, Any]]:
    """Convert observed successful transactions into the simulated flat universe."""
    date_min = transactions["Date"].min()
    date_max = transactions["Date"].max()
    date_span_days = max(1, int((date_max - date_min).days))
    transaction_months = [
        pd.Timestamp(value).to_period("M")
        for value in transactions["Date"].tolist()
    ]
    ordered_months = sorted(dict.fromkeys(transaction_months))
    initial_window_month_count = min(
        config.initial_window_months,
        len(ordered_months),
    )
    initial_window_months = set(ordered_months[:initial_window_month_count])
    month_index_by_period = {
        month_period: index
        for index, month_period in enumerate(ordered_months)
    }
    initial_window_transaction_count = sum(
        1 for month_period in transaction_months if month_period in initial_window_months
    )
    negotiating_cutoff = initial_window_transaction_count // 2

    flats: list[dict[str, Any]] = []
    initial_window_position = 0
    for order_index, row in enumerate(transactions.itertuples(index=False), start=1):
        transaction_date = pd.Timestamp(row.Date)
        relative_timing = round((transaction_date - date_min).days / date_span_days, 6)
        transaction_month = transaction_date.to_period("M")
        month_index = month_index_by_period[transaction_month]
        if transaction_month in initial_window_months:
            initial_window_position += 1
            current_initial_window_position = initial_window_position
            if initial_window_position <= negotiating_cutoff:
                initial_state = "negotiating"
            else:
                initial_state = "listed"
            listing_release_week = 1
        else:
            current_initial_window_position = 0
            initial_state = "not_yet_listed"
            # Expand the active transaction window by one calendar month every
            # four simulation weeks after the week-1 bootstrap window.
            listing_release_week = (
                1 + (4 * (month_index - initial_window_month_count + 1))
            )

        mall_names: list[str] = []
        mall_value = row.nearby_mall_names
        if isinstance(mall_value, list):
            mall_items = mall_value
        else:
            mall_text = str(mall_value).strip()
            if mall_text and mall_text != "[]":
                try:
                    parsed_malls = ast.literal_eval(mall_text)
                    mall_items = parsed_malls if isinstance(parsed_malls, list) else []
                except (SyntaxError, ValueError):
                    mall_items = []
            else:
                mall_items = []
        for item in mall_items:
            cleaned = _clean_amenity_name(item)
            if cleaned:
                mall_names.append(cleaned)

        mrt_names: list[str] = []
        mrt_value = row.nearby_mrt_names_lines
        if isinstance(mrt_value, list):
            mrt_items = mrt_value
        else:
            mrt_text = str(mrt_value).strip()
            if mrt_text and mrt_text != "[]":
                try:
                    parsed_mrt = ast.literal_eval(mrt_text)
                    mrt_items = parsed_mrt if isinstance(parsed_mrt, list) else []
                except (SyntaxError, ValueError):
                    mrt_items = []
            else:
                mrt_items = []
        for item in mrt_items:
            if isinstance(item, dict):
                station_name = _clean_amenity_name(item.get("station_name", ""))
            else:
                station_name = _clean_amenity_name(item)
            if station_name and station_name not in mrt_names:
                mrt_names.append(station_name)

        school_text = _clean_amenity_name(row.pri_school_names_0_2km)
        school_names = [part.strip() for part in school_text.split("|") if part.strip()]

        hawker_text = _clean_amenity_name(row.hawker_names_0_1km)
        hawker_names = [part.strip() for part in hawker_text.split("|") if part.strip()]
        remaining_lease_years = round(float(row.remaining_lease) / 12.0, 2)
        observed_price = float(row.resale_price)
        past_price_trends = _build_past_price_trends(
            town_transactions,
            town=str(row.town).strip(),
            flat_type=str(row.flat_type_label),
            reference_date=transaction_date,
            fallback_price=observed_price,
        )

        normalized_town = _normalize_text(row.town).replace(" ", "_")
        flat_id_prefix = f"{config.year}_{normalized_town}"
        if not config.is_full_year_segment:
            flat_id_prefix = f"{flat_id_prefix}_{config.segment_label}"
        flat_id = f"{flat_id_prefix}_{order_index:05d}"
        flats.append(
            {
                "flat_id": flat_id,
                "town": str(row.town).strip(),
                "year": int(config.year),
                "segment": config.segment_label,
                "transaction_date": transaction_date.date().isoformat(),
                "transaction_year_month": str(transaction_month),
                "simulated_market_entry_date": (
                    transaction_date
                    - pd.DateOffset(months=config.lead_months)
                ).date().isoformat(),
                "initialization_order": order_index,
                "relative_transaction_timing": relative_timing,
                "initial_market_state": initial_state,
                "initial_window_position": int(current_initial_window_position),
                "initial_window_size": int(initial_window_transaction_count),
                "listing_release_week": int(max(1, listing_release_week)),
                "address": str(row.address).strip(),
                "flat_type": str(row.flat_type_label),
                "floor_range": str(row.storey_range).strip(),
                "floor_area_sqm": float(row.floor_area_sqm),
                "flat_model": str(row.flat_model).strip(),
                "lease_commencement_year": int(row.lease_commence_date),
                "remaining_lease_years": remaining_lease_years,
                "observed_resale_price": observed_price,
                "amenities": {
                    "mrt": {
                        "count": len(mrt_names),
                        "station_names": mrt_names,
                    },
                    "primary_schools": {
                        "count": int(row.num_pri_schools_0_2km),
                        "school_names": school_names,
                    },
                    "malls": {
                        "count": len(mall_names),
                        "mall_names": mall_names,
                    },
                    "hawker_centres": {
                        "count": len(hawker_names),
                        "hawker_names": hawker_names,
                    },
                },
                "past_price_trends": past_price_trends,
            }
        )

    return flats


def _sample_seller_demographics(
    flat_type: str,
    distribution_tables: dict[str, pd.DataFrame],
    rng: random.Random,
) -> dict[str, Any]:
    """Sample seller demographics conditional on the observed flat type."""
    age_frame = _filter_dwelling_distribution(distribution_tables["age"], flat_type)
    marital_frame = _filter_dwelling_distribution(
        distribution_tables["marital"], flat_type
    )
    education_frame = _filter_dwelling_distribution(
        distribution_tables["education"], flat_type
    )
    occupation_frame = _filter_dwelling_distribution(
        distribution_tables["occupation"], flat_type
    )

    age_band = str(_weighted_choice(age_frame, age_frame.columns[1], "count", rng))
    marital_status = str(
        _weighted_choice(marital_frame, marital_frame.columns[1], "count", rng)
    )
    education_level = str(
        _weighted_choice(education_frame, education_frame.columns[1], "count", rng)
    )
    occupation_category = str(
        _weighted_choice(occupation_frame, occupation_frame.columns[1], "count", rng)
    )

    return {
        "age": _sample_age_from_band(age_band, rng),
        "marital_status": marital_status,
        "education_level": education_level,
        "occupation_category": occupation_category,
    }


def _build_seller_flat(flat: dict[str, Any]) -> dict[str, Any]:
    """Map flat-universe fields into the shared Flat schema used by sellers."""
    nearby_amenities = [
        Amenity(name=name, type=AmenityType.MRT, radius="Within 1km").model_dump()
        for raw_name in flat["amenities"]["mrt"]["station_names"]
        if (name := _clean_amenity_name(raw_name))
    ]
    nearby_amenities.extend(
        Amenity(
            name=name, type=AmenityType.SCHOOL, radius="Within 2km"
        ).model_dump()
        for raw_name in flat["amenities"]["primary_schools"]["school_names"]
        if (name := _clean_amenity_name(raw_name))
    )
    nearby_amenities.extend(
        Amenity(name=name, type=AmenityType.MALL, radius="Within 1km").model_dump()
        for raw_name in flat["amenities"]["malls"]["mall_names"]
        if (name := _clean_amenity_name(raw_name))
    )
    nearby_amenities.extend(
        Amenity(
            name=name, type=AmenityType.HAWKER, radius="Within 1km"
        ).model_dump()
        for raw_name in flat["amenities"]["hawker_centres"]["hawker_names"]
        if (name := _clean_amenity_name(raw_name))
    )

    return Flat(
        flat_type=flat["flat_type"],
        address=flat["address"],
        description=(
            f'{flat["flat_type"]} flat in {flat["town"]} with '
            f'{flat["floor_area_sqm"]:.0f} sqm and '
            f'{flat["remaining_lease_years"]:.1f} years remaining lease.'
        ),
        town=flat["town"],
        storey_range=flat["floor_range"],
        remaining_lease=flat["remaining_lease_years"],
        contra=False,
        extension_of_stay=False,
        ethnic_eligibility="Unknown",
        spr_eligibility="Unknown",
        floor_area_sqm=flat["floor_area_sqm"],
        nearby_amenities=nearby_amenities,
    ).model_dump()


def _build_seller_expectations(
    flat: dict[str, Any],
    hedonic_training_flats: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample a seller reservation / asking range around the hedonic anchor."""
    anchor_log_price, sigma_log_price = _estimate_hedonic_price(
        hedonic_training_flats,
        target_flat=flat,
    )
    reservation_price = float(
        np.exp(rng.normalvariate(anchor_log_price, sigma_log_price))
    )
    ask_price = max(
        reservation_price,
        float(np.exp(anchor_log_price + sigma_log_price)),
    )
    return SellerExpectationRange(
        min_price=round(reservation_price, 2),
        max_price=round(ask_price, 2),
    ).model_dump()


def _build_sellers(
    flats: list[dict[str, Any]],
    hedonic_training_flats: list[dict[str, Any]],
    distribution_tables: dict[str, pd.DataFrame],
    donors: pd.DataFrame,
    seller_archetypes: list[dict[str, Any]],
    *,
    config: SegmentConfig,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Create one synthetic seller per observed flat and attach LLM-ready motivation input."""
    sellers: list[dict[str, Any]] = []
    for index, flat in enumerate(flats, start=1):
        demographics = _sample_seller_demographics(
            flat["flat_type"], distribution_tables, rng
        )
        donor = _sample_nemotron_donor(
            donors,
            rng=rng,
            planning_area=config.town,
            age=demographics["age"],
            marital_status=demographics["marital_status"],
            education_level=demographics["education_level"],
            occupation=demographics["occupation_category"],
        )

        seller_record = {
            "seller_id": f"seller_{config.year}_{index:05d}",
            "linked_flat_id": flat["flat_id"],
            "initialization_order": flat["initialization_order"],
            "initial_market_state": flat["initial_market_state"],
            "initial_window_position": int(flat.get("initial_window_position", 0) or 0),
            "initial_window_size": int(flat.get("initial_window_size", 0) or 0),
            "listing_release_week": int(flat.get("listing_release_week", 1) or 1),
            "transaction_year_month": str(flat.get("transaction_year_month", "")).strip(),
            "age": demographics["age"],
            "marital_status": demographics["marital_status"],
            "education_level": demographics["education_level"],
            "occupation_category": demographics["occupation_category"],
            "industry": str(donor.get("industry", "")).strip(),
            "general_persona": str(donor.get("persona", "")).strip(),
            "flat": _build_seller_flat(flat),
            "expectations": _build_seller_expectations(
                flat,
                hedonic_training_flats,
                rng=rng,
            ),
            "seller_motivations": {
                "seller_archetype_type": "",
                "motivation_summary": "",
                "reasons": [],
            },
        }
        seller_record["seller_motivation_generation_input"] = (
            _build_seller_motivation_generation_input(seller_record, seller_archetypes)
        )
        sellers.append(seller_record)
    return sellers


def _load_buyer_age_prior(path: Path, town: str) -> pd.DataFrame:
    """Load the town-level buyer age prior used to seed the broad buyer pool."""
    planning_areas = _coerce_planning_areas(town)
    frame = pd.read_csv(path).fillna("")
    if not {"planning_area", "age_group", "population"}.issubset(frame.columns):
        frame = pd.concat(
            [build_planning_area_age_groups(planning_area) for planning_area in planning_areas],
            ignore_index=True,
        ).fillna("")
    subset = frame[
        frame["planning_area"].map(lambda value: _matches_planning_area(value, town))
    ].copy()
    subset["age_group"] = subset["age_group"].map(_canonical_buyer_age_prior_group)
    subset = subset[subset["age_group"].astype(bool)].copy()
    if subset.empty:
        raise ValueError(
            "No buyer age prior found for planning_area="
            f"{_planning_area_label(town)!r} within the supported age range "
            f"{BUYER_AGE_PRIOR_GROUPS[0]!r} to {BUYER_AGE_PRIOR_GROUPS[-1]!r}."
        )
    age_group_order = {
        label: index for index, label in enumerate(BUYER_AGE_PRIOR_GROUPS)
    }
    grouped = (
        subset.groupby("age_group", as_index=False)["population"]
        .sum()
        .reset_index(drop=True)
    )
    grouped["age_group_order"] = grouped["age_group"].map(age_group_order)
    grouped = grouped.sort_values("age_group_order").reset_index(drop=True)
    grouped = grouped.drop(columns=["age_group_order"])
    grouped["population"] = grouped["population"].astype(float)
    return grouped[grouped["population"] > 0].copy()


def _sample_income_band(
    income_prior: pd.DataFrame, age_band: str, rng: random.Random
) -> str:
    def _income_band_allowed(value: object) -> bool:
        band = INCOME_BANDS.get(str(value).strip())
        if band is None:
            return False
        lower = band.get("lower")
        return lower is not None and float(lower) >= MIN_BUYER_INCOME_BAND_LOWER

    age_group = _income_age_group_for_band(age_band, rng)
    subset = income_prior[
        (income_prior["age_group"].map(_normalize_text) == _normalize_text(age_group))
        & (income_prior["sex"].map(_normalize_text) == "total")
        & (income_prior["income_band"].map(_normalize_text) != "total")
    ].copy()
    if subset.empty:
        subset = income_prior[
            (income_prior["sex"].map(_normalize_text) == "total")
            & (income_prior["income_band"].map(_normalize_text) != "total")
        ].copy()
    subset = subset[subset["income_band"].map(_income_band_allowed)]
    subset["count"] = subset["count"].astype(float)
    subset = subset[subset["count"] > 0]
    if subset.empty:
        raise ValueError(
            "No eligible buyer income bands remain after applying the minimum "
            f"lower-bound filter of {MIN_BUYER_INCOME_BAND_LOWER:.0f}."
        )
    return str(_weighted_choice(subset, "income_band", "count", rng))


def _sample_overall_distribution(
    frame: pd.DataFrame,
    value_column: str,
    rng: random.Random,
) -> str:
    dwelling_col = frame.columns[0]
    subset = frame[
        frame[dwelling_col].map(_normalize_text) == _normalize_text("HDB Dwellings")
    ].copy()
    subset["count"] = subset["count"].astype(float)
    subset = subset[subset["count"] > 0]
    return str(_weighted_choice(subset, value_column, "count", rng))


def _estimate_buyer_hedonic_anchor(
    buyer: dict[str, Any],
    hedonic_training_flats: list[dict[str, Any]],
) -> tuple[float, float]:
    """Estimate the buyer-specific hedonic anchor and spread."""
    preference_payload = buyer.get("preferences", {})
    preferred_flat_types: list[str] = []
    if preference_payload.get("preferences"):
        preferred_flat_types = BuyerPreferenceProfile.model_validate(
            preference_payload
        ).values_for("flat_type")
    training_flats = _select_hedonic_training_flats(
        hedonic_training_flats,
        town=buyer["town"],
        preferred_flat_types=preferred_flat_types,
    )
    return _estimate_hedonic_price(training_flats)


def _draw_buyer_value_prior(
    buyer: dict[str, Any],
    hedonic_training_flats: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> float:
    """Draw a buyer's private valuation prior from the relevant hedonic market."""
    anchor_log_price, sigma_log_price = _estimate_buyer_hedonic_anchor(
        buyer,
        hedonic_training_flats,
    )
    return float(np.exp(rng.normalvariate(anchor_log_price, sigma_log_price)))


def _build_buyer_budget(
    buyer: dict[str, Any],
    hedonic_training_flats: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Build hard buyer budget bounds from financial feasibility."""
    del rng  # Budget bounds are deterministic once buyer features are fixed.
    anchor_log_price, sigma_log_price = _estimate_buyer_hedonic_anchor(
        buyer,
        hedonic_training_flats,
    )
    max_price = float(buyer["financials"]["effective_ceiling"])
    min_price = max(0.0, float(np.exp(anchor_log_price - sigma_log_price)))
    if max_price < min_price:
        min_price = max(0.0, min(max_price, min_price))
    return BuyerBudgetRange(
        min_price=round(min_price, 2),
        max_price=round(max_price, 2),
    ).model_dump()


def _annotate_buyer_market_feasibility(
    buyer: dict[str, Any],
    flats: list[dict[str, Any]],
    price_by_flat_id: dict[str, float],
    archetypes: list[dict[str, Any]] | None = None,
) -> bool:
    """Annotate buyer feasibility against the sampled flat market."""
    flat_index = {flat["flat_id"]: flat for flat in flats}
    effective_ceiling = float(buyer["financials"]["effective_ceiling"])
    feasible_flat_ids = [
        flat["flat_id"]
        for flat in flats
        if float(price_by_flat_id.get(str(flat["flat_id"]), 0.0)) <= effective_ceiling
    ]
    buyer["feasible_flat_ids"] = feasible_flat_ids
    if not feasible_flat_ids:
        buyer["retained"] = False
        return False

    if archetypes is not None:
        reachable_flats = [flat_index[flat_id] for flat_id in feasible_flat_ids]
        buyer["preferences"] = {"preferences": []}
        buyer["preference_classification_input"] = (
            _build_preference_classification_input(
                buyer,
                reachable_flats,
                archetypes,
            )
        )
        buyer["retained"] = True
    return True


def _build_broad_buyers(
    flats: list[dict[str, Any]],
    price_by_flat_id: dict[str, float],
    hedonic_training_flats: list[dict[str, Any]],
    donors: pd.DataFrame,
    archetypes: list[dict[str, Any]],
    *,
    config: SegmentConfig,
    rng: random.Random,
    age_prior: pd.DataFrame,
    income_prior: pd.DataFrame,
    distribution_tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Generate broad buyers conditioned on affordability for the sampled flats."""
    broad_count = max(len(flats), math.ceil(len(flats) * config.buyer_pool_multiplier))
    buyers: list[dict[str, Any]] = []
    max_attempts = max(
        broad_count,
        broad_count * MAX_BROAD_BUYER_GENERATION_ATTEMPT_FACTOR,
    )
    attempts = 0

    while len(buyers) < broad_count and attempts < max_attempts:
        attempts += 1
        index = len(buyers) + 1
        age_band = str(_weighted_choice(age_prior, "age_group", "population", rng))
        income_band = _sample_income_band(income_prior, age_band, rng)
        age = _sample_age_from_band(age_band, rng)

        donor = _sample_nemotron_donor(
            donors,
            rng=rng,
            planning_area=config.town,
            age=age,
        )
        donor_planning_area = str(donor.get("planning_area", "")).strip()
        planning_areas = _coerce_planning_areas(config.town)
        if donor_planning_area and _matches_planning_area(
            donor_planning_area,
            planning_areas,
        ):
            buyer_town = donor_planning_area
        else:
            buyer_town = planning_areas[0] if planning_areas else str(config.town)

        marital_status = (
            str(donor.get("marital_status", "")).strip()
            if str(donor.get("marital_status", "")).strip()
            else _sample_overall_distribution(
                distribution_tables["marital"],
                distribution_tables["marital"].columns[1],
                rng,
            )
        )
        education_level = (
            str(donor.get("education_level", "")).strip()
            if str(donor.get("education_level", "")).strip()
            else _sample_overall_distribution(
                distribution_tables["education"],
                distribution_tables["education"].columns[1],
                rng,
            )
        )
        occupation_category = (
            str(donor.get("occupation", "")).strip()
            if str(donor.get("occupation", "")).strip()
            else _sample_overall_distribution(
                distribution_tables["occupation"],
                distribution_tables["occupation"].columns[1],
                rng,
            )
        )

        monthly_income = resolve_income_band_upper(income_band)
        financials = compute_buyer_financials(
            current_age=age,
            monthly_income=monthly_income,
        ).__dict__

        buyer_record = {
            "buyer_id": f"buyer_{config.year}_{index:05d}",
            "town": buyer_town,
            "age": age,
            "income_band": income_band,
            "marital_status": marital_status,
            "education_level": education_level,
            "occupation_category": occupation_category,
            "industry": str(donor.get("industry", "")).strip(),
            "general_persona": str(donor.get("persona", "")).strip(),
            "financials": financials,
            "retained": False,
            "feasible_flat_ids": [],
            "preferences": {"preferences": []},
        }
        reservation_price_prior = _draw_buyer_value_prior(
            buyer_record,
            hedonic_training_flats,
            rng=rng,
        )
        buyer_record["reservation_price_prior"] = round(reservation_price_prior, 2)
        buyer_record["budget"] = _build_buyer_budget(
            buyer_record,
            hedonic_training_flats,
            rng=rng,
        )
        if not _annotate_buyer_market_feasibility(
            buyer_record,
            flats,
            price_by_flat_id,
            archetypes,
        ):
            continue
        buyers.append(buyer_record)

    if len(buyers) < broad_count:
        logging.warning(
            "Conditioned broad buyer generation only produced %s buyer(s) after %s attempts; broad target=%s.",
            len(buyers),
            attempts,
            broad_count,
        )

    return buyers


def _populate_seller_motivations(
    sellers: list[dict[str, Any]],
    *,
    model: VLLMLanguageModel,
) -> None:
    """Populate seller motivation fields in-place using the configured language model."""
    for seller in sellers:
        payload = seller.get("seller_motivation_generation_input")
        if not payload:
            continue
        result = _call_model_for_json(
            model,
            prompt=_build_seller_motivation_prompt(payload),
            result_model=SellerMotivationProfile,
            max_tokens=500,
        )
        seller["seller_motivations"] = result.model_dump()


def _populate_buyer_preferences(
    buyers: list[dict[str, Any]],
    hedonic_training_flats: list[dict[str, Any]],
    *,
    model: VLLMLanguageModel,
    rng: random.Random,
) -> None:
    """Populate retained-buyer preferences in-place, then refresh their hedonic budgets."""
    for buyer in buyers:
        payload = buyer.get("preference_classification_input")
        if not payload:
            continue
        result = _call_model_for_json(
            model,
            prompt=_build_buyer_preference_prompt(payload),
            result_model=BuyerPreferenceProfile,
            max_tokens=500,
        )
        buyer["preferences"] = _constrain_buyer_preferences_to_reachable_market(
            buyer,
            result,
        ).model_dump()
        reservation_price_prior = _draw_buyer_value_prior(
            buyer,
            hedonic_training_flats,
            rng=rng,
        )
        buyer["reservation_price_prior"] = round(reservation_price_prior, 2)
        buyer["budget"] = _build_buyer_budget(
            buyer,
            hedonic_training_flats,
            rng=rng,
        )


def _retain_feasible_buyers(
    buyers: list[dict[str, Any]],
    flats: list[dict[str, Any]],
    price_by_flat_id: dict[str, float],
    archetypes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only buyers who can financially reach at least one flat in the seller pool."""
    retained: list[dict[str, Any]] = []

    for buyer in buyers:
        if not _annotate_buyer_market_feasibility(
            buyer,
            flats,
            price_by_flat_id,
            archetypes,
        ):
            continue
        retained.append(buyer)

    return retained


def _validate_negotiating_seller_seedability(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
) -> tuple[bool, list[str], dict[str, str]]:
    """Checks whether negotiating sellers admit a one-to-one feasible matching."""
    match_assignments = _match_negotiating_sellers_to_buyers(sellers, buyers)
    negotiating_seller_ids = [
        str(seller.get("seller_id", "")).strip()
        for seller in sellers
        if str(seller.get("initial_market_state", "")).strip().casefold()
        == "negotiating"
    ]
    unmatched_seller_ids = [
        seller_id for seller_id in negotiating_seller_ids if seller_id not in match_assignments
    ]
    return not unmatched_seller_ids, unmatched_seller_ids, match_assignments


def _validate_seller_candidate_coverage(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Checks whether every seller has enough feasible retained buyers."""
    uncovered_seller_ids: list[str] = []
    for seller in sorted(
        sellers,
        key=lambda item: int(item.get("initialization_order", 0)),
    ):
        seller_id = str(seller.get("seller_id", "")).strip()
        if not seller_id:
            continue
        candidate_buyer_ids = _rank_candidate_buyer_ids_for_seller(seller, buyers)
        if len(candidate_buyer_ids) >= MIN_FEASIBLE_RETAINED_BUYERS_PER_SELLER:
            continue
        uncovered_seller_ids.append(seller_id)
    return not uncovered_seller_ids, uncovered_seller_ids


def _validate_market_quantile_dominance(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """Computes market-level quantile-dominance diagnostics for buyer ceilings."""
    listing_prices = np.array(
        [
            float(seller.get("expectations", {}).get("max_price", 0.0))
            for seller in sellers
            if float(seller.get("expectations", {}).get("max_price", 0.0)) > 0.0
        ],
        dtype=float,
    )
    buyer_ceilings = np.array(
        [
            float(buyer.get("financials", {}).get("effective_ceiling", 0.0))
            for buyer in buyers
            if float(buyer.get("financials", {}).get("effective_ceiling", 0.0)) > 0.0
        ],
        dtype=float,
    )
    if listing_prices.size == 0 or buyer_ceilings.size == 0:
        diagnostics = {
            "ok": False,
            "reason": "empty_market_or_buyers",
            "listing_count": int(listing_prices.size),
            "buyer_count": int(buyer_ceilings.size),
        }
        return False, diagnostics

    quantiles = np.array(MARKET_QUANTILE_DOMINANCE_GRID, dtype=float)
    listing_quantiles = np.quantile(listing_prices, quantiles)
    buyer_quantiles = np.quantile(buyer_ceilings, quantiles)
    gaps = buyer_quantiles - listing_quantiles
    top_supported = float(np.max(buyer_ceilings)) >= float(np.max(listing_prices))
    diagnostics = {
        "ok": bool(np.all(gaps >= 0.0) and top_supported),
        "quantiles": quantiles.tolist(),
        "listing_quantiles": np.round(listing_quantiles, 2).tolist(),
        "buyer_quantiles": np.round(buyer_quantiles, 2).tolist(),
        "gaps": np.round(gaps, 2).tolist(),
        "min_gap": round(float(np.min(gaps)), 2),
        "max_listing_price": round(float(np.max(listing_prices)), 2),
        "max_buyer_ceiling": round(float(np.max(buyer_ceilings)), 2),
        "top_supported": bool(top_supported),
        "listing_count": int(listing_prices.size),
        "buyer_count": int(buyer_ceilings.size),
    }
    return bool(diagnostics["ok"]), diagnostics


def _cap_retained_buyers(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
    *,
    cap: int,
) -> list[dict[str, Any]]:
    """Trim retained buyers while preserving negotiating seeds and seller coverage."""
    if len(buyers) <= cap:
        return buyers

    negotiating_seller_order = {
        str(seller.get("seller_id", "")).strip(): index
        for index, seller in enumerate(
            sorted(
                sellers,
                key=lambda item: int(item.get("initialization_order", 0)),
            )
        )
        if str(seller.get("initial_market_state", "")).strip().casefold()
        == "negotiating"
    }
    match_assignments = _match_negotiating_sellers_to_buyers(sellers, buyers)
    buyer_by_id = {
        str(buyer.get("buyer_id", "")).strip(): buyer
        for buyer in buyers
        if str(buyer.get("buyer_id", "")).strip()
    }

    seeded_buyers: list[dict[str, Any]] = []
    for seller_id, buyer_id in sorted(
        match_assignments.items(),
        key=lambda item: negotiating_seller_order.get(item[0], math.inf),
    ):
        buyer = buyer_by_id.get(buyer_id)
        if buyer is not None:
            seeded_buyers.append(buyer)

    if len(seeded_buyers) > cap:
        logging.warning(
            "Retained buyer cap %s is below the %s buyers needed to seed negotiating sellers; truncating seeded buyers.",
            cap,
            len(seeded_buyers),
        )
        seeded_buyers = seeded_buyers[:cap]

    candidate_buyer_ids_by_seller = {
        str(seller.get("seller_id", "")).strip(): _rank_candidate_buyer_ids_for_seller(
            seller,
            buyers,
        )
        for seller in sellers
        if str(seller.get("seller_id", "")).strip()
    }
    selected_ids = {
        str(buyer.get("buyer_id", "")).strip() for buyer in seeded_buyers
    }
    seller_ids_covered_by_selected = {
        seller_id
        for seller_id, candidate_buyer_ids in candidate_buyer_ids_by_seller.items()
        if selected_ids.intersection(candidate_buyer_ids)
    }

    while len(selected_ids) < cap:
        best_buyer = None
        best_buyer_id = ""
        best_newly_covered_seller_ids: set[str] = set()
        for buyer in buyers:
            buyer_id = str(buyer.get("buyer_id", "")).strip()
            if not buyer_id or buyer_id in selected_ids:
                continue
            newly_covered_seller_ids = {
                seller_id
                for seller_id, candidate_buyer_ids in candidate_buyer_ids_by_seller.items()
                if seller_id not in seller_ids_covered_by_selected
                and buyer_id in candidate_buyer_ids
            }
            if not newly_covered_seller_ids:
                continue
            if best_buyer is None:
                best_buyer = buyer
                best_buyer_id = buyer_id
                best_newly_covered_seller_ids = newly_covered_seller_ids
                continue
            best_priority = (
                len(best_newly_covered_seller_ids),
                len(best_buyer.get("feasible_flat_ids", ())),
                float(best_buyer.get("budget", {}).get("max_price", 0.0)),
                best_buyer_id,
            )
            candidate_priority = (
                len(newly_covered_seller_ids),
                len(buyer.get("feasible_flat_ids", ())),
                float(buyer.get("budget", {}).get("max_price", 0.0)),
                buyer_id,
            )
            if candidate_priority > best_priority:
                best_buyer = buyer
                best_buyer_id = buyer_id
                best_newly_covered_seller_ids = newly_covered_seller_ids

        if best_buyer is None:
            break
        selected_ids.add(best_buyer_id)
        seller_ids_covered_by_selected.update(best_newly_covered_seller_ids)

    remaining_candidates = [
        buyer
        for buyer in buyers
        if str(buyer.get("buyer_id", "")).strip() not in selected_ids
    ]
    remaining_candidates.sort(
        key=lambda buyer: (
            len(buyer.get("feasible_flat_ids", ())),
            float(buyer.get("budget", {}).get("max_price", 0.0)),
            str(buyer.get("buyer_id", "")).strip(),
        ),
        reverse=True,
    )

    capped_buyers = [
        buyer
        for buyer in buyers
        if str(buyer.get("buyer_id", "")).strip() in selected_ids
    ]
    remaining_capacity = max(0, cap - len(capped_buyers))
    capped_buyers.extend(remaining_candidates[:remaining_capacity])
    capped_ids = {str(buyer.get("buyer_id", "")).strip() for buyer in capped_buyers}
    uncovered_seller_ids = sorted(
        seller_id
        for seller_id, candidate_buyer_ids in candidate_buyer_ids_by_seller.items()
        if candidate_buyer_ids and not capped_ids.intersection(candidate_buyer_ids)
    )
    if uncovered_seller_ids:
        logging.warning(
            "Retained buyer cap %s left %s seller(s) without any retained candidate buyers after capping: %s",
            cap,
            len(uncovered_seller_ids),
            uncovered_seller_ids[:10],
        )
    return [
        buyer
        for buyer in buyers
        if str(buyer.get("buyer_id", "")).strip() in capped_ids
    ]


def _rank_candidate_buyer_ids_for_seller(
    seller: dict[str, Any],
    buyers: list[dict[str, Any]],
) -> list[str]:
    """Rank feasible buyers for one seller using the initializer's heuristics."""
    listing_price = float(seller["expectations"]["max_price"])
    linked_flat_id = str(seller.get("linked_flat_id", "")).strip()
    flat = seller.get("flat", {})

    ranked: list[tuple[int, float, str]] = []
    for buyer in buyers:
        buyer_id = str(buyer.get("buyer_id", "")).strip()
        if not buyer_id:
            continue

        preference_payload = buyer.get("preferences", {})
        preference_profile = (
            BuyerPreferenceProfile.model_validate(preference_payload)
            if preference_payload.get("preferences")
            else None
        )
        feasible_flat_ids = {
            str(flat_id).strip()
            for flat_id in buyer.get("feasible_flat_ids", ())
            if str(flat_id).strip()
        }
        score = 0
        if linked_flat_id and linked_flat_id in feasible_flat_ids:
            score += 10
        else:
            continue
        if preference_profile and flat.get("flat_type") in preference_profile.values_for("flat_type"):
            score += 3
        if preference_profile and _matches_planning_area(
            flat.get("town"),
            preference_profile.values_for("town"),
        ):
            score += 2
        price_gap = abs(
            float(buyer.get("financials", {}).get("effective_ceiling", 0.0))
            - listing_price
        )
        ranked.append((score, -price_gap, buyer_id))

    ranked.sort(reverse=True)
    return [buyer_id for _, _, buyer_id in ranked]


def _match_negotiating_sellers_to_buyers(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
) -> dict[str, str]:
    """Finds a maximum one-to-one matching for negotiating sellers."""
    ordered_sellers = [
        seller
        for seller in sorted(
            sellers,
            key=lambda item: int(item.get("initialization_order", 0)),
        )
        if str(seller.get("initial_market_state", "")).strip().casefold()
        == "negotiating"
    ]
    if not ordered_sellers:
        return {}

    candidate_buyer_ids_by_seller = {
        str(seller["seller_id"]): _rank_candidate_buyer_ids_for_seller(seller, buyers)
        for seller in ordered_sellers
    }
    matched_seller_by_buyer: dict[str, str] = {}

    def _try_assign(seller_id: str, visited_buyer_ids: set[str]) -> bool:
        for buyer_id in candidate_buyer_ids_by_seller.get(seller_id, ()):
            if buyer_id in visited_buyer_ids:
                continue
            visited_buyer_ids.add(buyer_id)
            current_seller_id = matched_seller_by_buyer.get(buyer_id)
            if current_seller_id is None or _try_assign(
                current_seller_id,
                visited_buyer_ids,
            ):
                matched_seller_by_buyer[buyer_id] = seller_id
                return True
        return False

    for seller in ordered_sellers:
        _try_assign(str(seller["seller_id"]), set())

    return {
        seller_id: buyer_id for buyer_id, seller_id in matched_seller_by_buyer.items()
    }


def _annotate_seller_potential_matches(
    sellers: list[dict[str, Any]],
    buyers: list[dict[str, Any]],
) -> None:
    """Store feasible buyer rankings so the main simulation can reuse them."""
    for seller in sellers:
        seller["potential_buyer_ids"] = _rank_candidate_buyer_ids_for_seller(
            seller,
            buyers,
        )

    match_assignments = _match_negotiating_sellers_to_buyers(sellers, buyers)
    downgraded_seller_ids: list[str] = []
    for seller in sorted(
        sellers,
        key=lambda item: int(item.get("initialization_order", 0)),
    ):
        seller_id = str(seller.get("seller_id", "")).strip()
        intended_state = str(
            seller.get("initial_market_state_intended", seller.get("initial_market_state", ""))
        ).strip()
        seller["initial_market_state_intended"] = intended_state
        if (
            intended_state.casefold()
            != "negotiating"
        ):
            seller["initial_market_state"] = intended_state
            seller["seeded_buyer_id"] = ""
            continue
        seeded_buyer_id = match_assignments.get(seller_id, "")
        seller["seeded_buyer_id"] = seeded_buyer_id
        if seeded_buyer_id:
            seller["initial_market_state"] = "negotiating"
        else:
            seller["initial_market_state"] = "listed"
            downgraded_seller_ids.append(seller_id)

    if downgraded_seller_ids:
        logging.info(
            "Downgraded %s intended negotiating sellers to listed after feasible matching: %s",
            len(downgraded_seller_ids),
            ", ".join(downgraded_seller_ids),
        )


def _build_buyer_pools_with_regeneration(
    sellers: list[dict[str, Any]],
    flats: list[dict[str, Any]],
    hedonic_training_flats: list[dict[str, Any]],
    donors: pd.DataFrame,
    archetypes: list[dict[str, Any]],
    *,
    config: SegmentConfig,
    rng: random.Random,
    age_prior: pd.DataFrame,
    income_prior: pd.DataFrame,
    distribution_tables: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retry buyer generation against a fixed seller market until constraints pass."""
    seller_count = len(flats)
    price_by_flat_id = {
        str(seller.get("linked_flat_id", "")).strip(): float(
            seller.get("expectations", {}).get("max_price", 0.0)
        )
        for seller in sellers
        if str(seller.get("linked_flat_id", "")).strip()
    }
    target_retained_buyer_count = max(
        1,
        math.ceil(len(flats) * config.retained_buyer_pool_multiplier),
    )
    best_broad_buyers: list[dict[str, Any]] = []
    best_retained_buyers: list[dict[str, Any]] = []
    best_matched_count = -1
    best_unmatched_seller_ids: list[str] = []
    best_uncovered_seller_ids: list[str] = []
    last_unmatched_seller_ids: list[str] = []
    last_uncovered_seller_ids: list[str] = []

    for broad_attempt in range(1, MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS + 1):
        broad_buyers = _build_broad_buyers(
            flats,
            price_by_flat_id,
            hedonic_training_flats,
            donors,
            archetypes,
            config=config,
            rng=rng,
            age_prior=age_prior,
            income_prior=income_prior,
            distribution_tables=distribution_tables,
        )
        oversampled_retained_buyers = _retain_feasible_buyers(
            broad_buyers,
            flats,
            price_by_flat_id,
            archetypes,
        )
        logging.info(
            "Oversampled buyer pool %s/%s produced %s feasible buyers for %s sellers before matching; retained buyer target=%s.",
            broad_attempt,
            MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS,
            len(oversampled_retained_buyers),
            seller_count,
            target_retained_buyer_count,
        )

        if len(oversampled_retained_buyers) < target_retained_buyer_count:
            last_unmatched_seller_ids = [
                str(seller.get("seller_id", ""))
                for seller in sellers
                if str(seller.get("initial_market_state", "")).strip().casefold()
                == "negotiating"
            ]
            logging.info(
                "Skipping oversampled buyer pool %s/%s because only %s feasible buyers were retained, below the retained buyer target of %s.",
                broad_attempt,
                MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS,
                len(oversampled_retained_buyers),
                target_retained_buyer_count,
            )
            continue

        (
            market_support_ok,
            market_support_diagnostics,
        ) = _validate_market_quantile_dominance(
            sellers,
            oversampled_retained_buyers,
        )
        if not market_support_ok:
            logging.info(
                "Market quantile dominance diagnostic flagged oversampled buyer pool %s/%s: %s",
                broad_attempt,
                MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS,
                market_support_diagnostics,
            )

        retained_buyers = _cap_retained_buyers(
            sellers,
            oversampled_retained_buyers,
            cap=target_retained_buyer_count,
        )
        (
            all_sellers_covered,
            uncovered_seller_ids,
        ) = _validate_seller_candidate_coverage(
            sellers,
            retained_buyers,
        )
        (
            seedable,
            unmatched_seller_ids,
            match_assignments,
        ) = _validate_negotiating_seller_seedability(
            sellers,
            retained_buyers,
        )
        logging.info(
            "Oversampled buyer pool %s/%s retained %s feasible buyers for %s sellers; retained buyer target=%s; matched negotiating sellers=%s unmatched negotiating sellers=%s uncovered sellers=%s.",
            broad_attempt,
            MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS,
            len(retained_buyers),
            seller_count,
            target_retained_buyer_count,
            len(match_assignments),
            len(unmatched_seller_ids),
            len(uncovered_seller_ids),
        )
        last_unmatched_seller_ids = unmatched_seller_ids
        last_uncovered_seller_ids = uncovered_seller_ids
        matched_count = len(match_assignments)

        if (
            matched_count > best_matched_count
            or (
                matched_count == best_matched_count
                and (
                    not best_retained_buyers
                    or len(uncovered_seller_ids) < len(best_uncovered_seller_ids)
                )
            )
        ):
            best_broad_buyers = broad_buyers
            best_retained_buyers = retained_buyers
            best_matched_count = matched_count
            best_unmatched_seller_ids = unmatched_seller_ids
            best_uncovered_seller_ids = uncovered_seller_ids

        if (
            len(retained_buyers) >= target_retained_buyer_count
            and seedable
            and all_sellers_covered
        ):
            logging.info(
                "Accepted oversampled buyer pool %s/%s for seller market sample: retained buyers=%s target=%s matched negotiating sellers=%s uncovered sellers=%s.",
                broad_attempt,
                MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS,
                len(retained_buyers),
                target_retained_buyer_count,
                len(match_assignments),
                len(uncovered_seller_ids),
            )
            return broad_buyers, retained_buyers

    if best_retained_buyers:
        unmatched_text = (
            ", ".join(best_unmatched_seller_ids)
            if best_unmatched_seller_ids
            else "none"
        )
        uncovered_text = (
            ", ".join(best_uncovered_seller_ids)
            if best_uncovered_seller_ids
            else "none"
        )
        raise ValueError(
            "Unable to build a retained buyer pool that both seeds all "
            "negotiating sellers and covers every seller after "
            f"{MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS} oversampled-pool "
            f"attempts; best pool retained {len(best_retained_buyers)} buyers, "
            f"matched negotiating sellers={best_matched_count}, "
            f"unmatched negotiating sellers={unmatched_text}, "
            f"uncovered sellers={uncovered_text}."
        )

    unmatched_text = (
        ", ".join(last_unmatched_seller_ids) if last_unmatched_seller_ids else "unknown"
    )
    uncovered_text = (
        ", ".join(last_uncovered_seller_ids) if last_uncovered_seller_ids else "unknown"
    )
    raise ValueError(
        "Unable to build a buyer pool with enough retained buyers after "
        f"{MAX_OVERSAMPLED_BUYER_POOL_REGEN_ATTEMPTS} oversampled-pool attempts; "
        f"unmatched negotiating sellers: {unmatched_text}; "
        f"uncovered sellers: {uncovered_text}."
    )


def build_transaction_conditioned_segment(
    config: SegmentConfig,
    model: VLLMLanguageModel | None = None,
) -> dict[str, Any]:
    """Run the end-to-end preprocessing pipeline for one town-year market segment."""
    logging.info(
        "Building transaction-conditioned market segment for planning_area=%s year=%s segment=%s.",
        _planning_area_label(config.town),
        config.year,
        config.segment_label,
    )
    rng = random.Random(config.random_seed)

    distribution_tables = {
        "age": pd.read_csv(config.flat_type_age_path).fillna(""),
        "marital": pd.read_csv(config.flat_type_marital_path).fillna(""),
        "education": pd.read_csv(config.flat_type_education_path).fillna(""),
        "occupation": pd.read_csv(config.flat_type_occupation_path).fillna(""),
    }
    age_prior = _load_buyer_age_prior(config.age_prior_path, config.town)
    income_prior = pd.read_csv(config.income_prior_path).fillna("")
    logging.info(
        "Loaded demographic priors and distribution tables from configured CSV inputs."
    )
    donors = _load_nemotron_pool(config.nemotron_dir)
    buyer_archetypes_path = config.survey_archetypes_path or DEFAULT_BUYER_ARCHETYPES_PATH
    seller_archetypes_path = (
        config.seller_archetypes_path or DEFAULT_SELLER_ARCHETYPES_PATH
    )
    archetypes = _load_archetype_config(
        buyer_archetypes_path,
        label="buyer",
    )
    seller_archetypes = _load_archetype_config(
        seller_archetypes_path,
        label="seller",
    )

    town_transactions = _load_town_transactions(config)
    transactions = _load_transactions(config)
    logging.info(
        "Loaded %s town transactions and %s total transaction rows.",
        len(town_transactions),
        len(transactions),
    )
    window_start = pd.Timestamp(transactions["Date"].min())
    hedonic_training_flats = _build_hedonic_training_flats(
        town_transactions,
        town=config.town,
        window_start=window_start,
    )
    logging.info(
        "Prepared %s hedonic training flats using transaction window ending at %s.",
        len(hedonic_training_flats),
        window_start,
    )
    sampled_transaction_count: int | None = None
    if config.sampled_flat_ratio is not None and config.sampled_flat_ratio < 1:
        planning_area_sample_counts = (
            transactions["town"]
            .map(_normalize_text)
            .value_counts(sort=False)
            .map(lambda count: max(1, math.ceil(int(count) * config.sampled_flat_ratio)))
        )
        sampled_transaction_count = int(planning_area_sample_counts.sum())
        if sampled_transaction_count >= len(transactions):
            sampled_transaction_count = None
    seller_market_attempts = (
        max(1, int(config.seller_segment_regeneration_attempts))
        if sampled_transaction_count is not None
        else 1
    )
    last_market_error: Exception | None = None
    flats: list[dict[str, Any]] = []
    sellers: list[dict[str, Any]] = []
    broad_buyers: list[dict[str, Any]] = []
    retained_buyers: list[dict[str, Any]] = []

    for seller_attempt in range(1, seller_market_attempts + 1):
        restrained_transactions = _restrain_transactions(
            transactions,
            restrained_seller_count=sampled_transaction_count,
            sampled_flat_ratio=config.sampled_flat_ratio if sampled_transaction_count is not None else None,
            rng=rng if sampled_transaction_count is not None else None,
        )
        if sampled_transaction_count is not None:
            logging.info(
                "Seller market sample %s/%s restrained seller pool to %s transaction(s) using per-planning-area sampled_flat_ratio=%s; oversampled buyer pool multiplier=%s and retained buyer pool multiplier=%s.",
                seller_attempt,
                seller_market_attempts,
                len(restrained_transactions),
                config.sampled_flat_ratio,
                config.buyer_pool_multiplier,
                config.retained_buyer_pool_multiplier,
            )
        flats = _build_flat_universe(restrained_transactions, town_transactions, config)
        sellers = _build_sellers(
            flats,
            hedonic_training_flats,
            distribution_tables,
            donors,
            seller_archetypes,
            config=config,
            rng=rng,
        )
        try:
            broad_buyers, retained_buyers = _build_buyer_pools_with_regeneration(
                sellers,
                flats,
                hedonic_training_flats,
                donors,
                archetypes,
                config=config,
                rng=rng,
                age_prior=age_prior,
                income_prior=income_prior,
                distribution_tables=distribution_tables,
            )
            last_market_error = None
            logging.info(
                "Accepted seller market sample %s/%s with %s sellers, %s broad buyers, and %s retained buyers.",
                seller_attempt,
                seller_market_attempts,
                len(sellers),
                len(broad_buyers),
                len(retained_buyers),
            )
            break
        except ValueError as error:
            last_market_error = error
            logging.warning(
                "Rejected seller market sample %s/%s because the retained buyer pool could not satisfy the transacting-segment constraints: %s",
                seller_attempt,
                seller_market_attempts,
                error,
            )
            continue

    if last_market_error is not None:
        raise ValueError(
            "Unable to build a transacting market segment after "
            f"{seller_market_attempts} seller-market sample attempts. "
            f"Last failure: {last_market_error}"
        ) from last_market_error

    if model is not None:
        logging.info(
            "Populating seller motivations and buyer preferences with the configured model."
        )
        _populate_seller_motivations(sellers, model=model)
        _populate_buyer_preferences(
            retained_buyers,
            hedonic_training_flats,
            model=model,
            rng=rng,
        )
    else:
        logging.info(
            "Skipping model-based enrichment because no language model was provided."
        )

    _annotate_seller_potential_matches(sellers, retained_buyers)
    logging.info(
        "Built market segment with %s flats, %s sellers, %s broad buyers, and %s retained buyers.",
        len(flats),
        len(sellers),
        len(broad_buyers),
        len(retained_buyers),
    )

    return {
        "flats": flats,
        "sellers": sellers,
        "buyers_broad": broad_buyers,
        "buyers_retained": retained_buyers,
    }


def save_segment_outputs(bundle: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Persist the generated segment artifacts and return a small path manifest."""
    logging.info("Saving generated market segment outputs to %s.", output_dir)
    flats_path = output_dir / "flat_units.jsonl"
    sellers_path = output_dir / "sellers.jsonl"
    buyers_broad_path = output_dir / "buyers_broad.jsonl"
    buyers_retained_path = output_dir / "buyers_retained.jsonl"
    manifest_path = output_dir / "manifest.json"

    _write_jsonl(flats_path, bundle["flats"])
    _write_jsonl(sellers_path, bundle["sellers"])
    _write_jsonl(buyers_broad_path, bundle["buyers_broad"])
    _write_jsonl(buyers_retained_path, bundle["buyers_retained"])

    manifest = {
        "town": (
            bundle["flats"][0]["town"]
            if len({str(flat.get("town", "")).strip() for flat in bundle["flats"]}) <= 1
            else _planning_area_label(
                sorted(
                    {
                        str(flat.get("town", "")).strip()
                        for flat in bundle["flats"]
                        if str(flat.get("town", "")).strip()
                    }
                )
            )
        ) if bundle["flats"] else "",
        "planning_areas": sorted(
            {
                str(flat.get("town", "")).strip()
                for flat in bundle["flats"]
                if str(flat.get("town", "")).strip()
            }
        ),
        "year": bundle["flats"][0]["year"] if bundle["flats"] else "",
        "segment": bundle["flats"][0]["segment"] if bundle["flats"] else "full_year",
        "market_segment_name": output_dir.name,
        "flat_units_path": flats_path.name,
        "sellers_path": sellers_path.name,
        "buyers_broad_path": buyers_broad_path.name,
        "buyers_retained_path": buyers_retained_path.name,
    }
    _write_json(manifest_path, manifest)
    logging.info("Saved market segment manifest to %s.", manifest_path)
    return manifest
