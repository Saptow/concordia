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
    BuyerPreferenceProfile,
    Flat,
    SellerExpectationRange,
)
from concordia.hdb_simulation.pipeline.financial_feasibility import (
    compute_buyer_financials,
    resolve_income_band_upper,
)


DEFAULT_LLM_RETRIES = 3
MAX_REACHABLE_MARKET_SAMPLE_FLATS = 50


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

DEFAULT_BUYER_ARCHETYPES = [ # Dummy ones for now until the real data is plugged in
    {
        "archetype_type": "family",
        "description": "Values space, school access, and practical family-oriented amenities.",
        "preferences": ["larger flat types", "good school access", "nearby hawker centres"],
    },
    {
        "archetype_type": "convenience_seeker",
        "description": "Prioritises MRT access, town convenience, and day-to-day transport.",
        "preferences": ["near MRT", "near malls", "convenient commute"],
    },
    {
        "archetype_type": "value_guard",
        "description": "Emphasises affordability, nearby comparables, and avoiding overpaying.",
        "preferences": ["lower price", "strong comparables", "budget discipline"],
    },
    {
        "archetype_type": "investor",
        "description": "Prefers stronger remaining lease and larger usable floor area.",
        "preferences": ["longer remaining lease", "more floor area", "overall livability"],
    },
]

DEFAULT_SELLER_ARCHETYPES = [ # Dummy ones for now until the real data is plugged in
    {
        "archetype_type": "upgrader",
        "description": "Selling to move to a larger or better-located home.",
        "reasons": ["needs more space", "wants a better location", "timing an upgrade"],
    },
    {
        "archetype_type": "rightsizer",
        "description": "Selling to move into a smaller and more manageable home.",
        "reasons": ["children moved out", "lower upkeep", "simpler living arrangement"],
    },
    {
        "archetype_type": "relocator",
        "description": "Selling because of a relocation in work, family, or caregiving needs.",
        "reasons": ["job relocation", "caregiving responsibilities", "closer to family"],
    },
    {
        "archetype_type": "liquidity_seeker",
        "description": "Selling to unlock liquidity or improve household finances.",
        "reasons": ["free up cash", "reduce financial pressure", "rebalance household finances"],
    },
]

class SellerMotivationProfile(BaseModel):
    seller_archetype_type: str = ""
    motivation_summary: str = ""
    reasons: list[str] = Field(default_factory=list)


# Generic text / file utilities.
def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()


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


def _flat_type_from_row(value: Any) -> str:
    key = str(value or "").strip().upper()
    return FLAT_TYPE_LABELS.get(key, str(value or "").strip())


# Sampling helpers used for demographic generation.
def _parse_age_band(label: str, *, adult_floor: int = 20) -> tuple[int, int]:
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
    planning_area: str,
    age: int | None = None,
    marital_status: str | None = None,
    education_level: str | None = None,
    occupation: str | None = None,
) -> dict[str, Any]:
    """Sample a donor row, narrowing by planning area and available profile fields."""
    candidates = donors.copy()
    target_planning_area = _normalize_text(planning_area)

    if target_planning_area:
        subset = candidates[
            candidates["planning_area"].map(_normalize_text) == target_planning_area
        ]
        if not subset.empty:
            candidates = subset

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
    town: str,
    preferred_flat_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Filter the fixed pre-window hedonic pool to town, then to preferred flat types if available."""
    target_flat_types = {flat_type for flat_type in (preferred_flat_types or []) if flat_type}
    town_flats = [
        flat for flat in training_flats if _normalize_text(flat["town"]) == _normalize_text(town)
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
    """Fit a simple hedonic anchor and return (predicted price, residual scale)."""

    training_frame = _hedonic_feature_frame(training_flats)
    y = training_frame["observed_resale_price"].astype(float).to_numpy()

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
        anchor = float(np.median(y))
        sigma = float(np.std(y, ddof=0)) if len(y) > 1 else max(anchor * 0.05, 1.0)
        return anchor, max(sigma, max(anchor * 0.05, 1.0))

    x_train = np.column_stack(
        [np.ones(len(training_design)), training_design.to_numpy(dtype=float)]
    )
    x_target = np.column_stack(
        [np.ones(len(target_design)), target_design.to_numpy(dtype=float)]
    )

    beta, *_ = np.linalg.lstsq(x_train, y, rcond=None)
    fitted = x_train @ beta
    residuals = y - fitted
    sigma = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
    anchor = float((x_target @ beta)[0])

    if not np.isfinite(anchor) or anchor <= 0:
        anchor = float(np.median(y))
    sigma = max(sigma, max(anchor * 0.05, 1.0))
    return anchor, sigma


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
                "representative_flats": [
                    _compact_flat_for_preference_summary(flat)
                    for flat in _sample_flats_uniformly(bucket_flats)
                ],
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
            "global_representative_flats": [],
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
        "global_representative_flats": [
            _compact_flat_for_preference_summary(flat)
            for flat in _sample_flats_uniformly(sorted_flats)
        ],
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
3. Use the representative flats only as concrete examples within those summary
   buckets, not as the full market.
4. Compare the buyer against the provided preference archetypes.
5. Infer the most plausible flat-type and town preferences, constrained by the
   reachable market summary.
6. Write a concise `features` summary describing the buyer's likely housing
   priorities.
7. Return only the final JSON object.

## Rules

- Use only towns and flat types supported by the reachable market summary.
- Do not invent unreachable towns or flat types.
- Prefer market patterns that appear consistently across the summary buckets.
- Keep `features` concise but specific.
- Return JSON only. Do not wrap the answer in markdown fences.
"""


def _load_town_transactions(config: SegmentConfig) -> pd.DataFrame:
    """Load all successful resale transactions for the configured town."""
    frame = pd.read_csv(config.resale_path)
    town_rows = frame[
        frame["town"].map(_normalize_text) == _normalize_text(config.town)
    ].copy()
    if town_rows.empty:
        raise ValueError(f"No resale rows found for town={config.town!r}.")

    town_rows["Date"] = pd.to_datetime(town_rows["Date"])
    town_rows["flat_type_label"] = town_rows["flat_type"].map(_flat_type_from_row)
    town_rows["sale_id"] = town_rows["sale_id"].astype(int)
    return town_rows.sort_values(["Date", "sale_id"]).reset_index(drop=True)


def _load_transactions(config: SegmentConfig) -> pd.DataFrame:
    """Load the simulation window: successful transactions for the configured town-year."""
    town_rows = _load_town_transactions(config)
    filtered = town_rows[town_rows["year"].astype(int) == int(config.year)].copy()
    if filtered.empty:
        raise ValueError(
            f"No successful resale rows found for town={config.town!r} year={config.year}."
        )
    return filtered.reset_index(drop=True)


def _build_hedonic_training_flats(
    transactions: pd.DataFrame,
    *,
    town: str,
    window_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Build the fixed 6-month pre-window pool used for hedonic calibration."""
    cutoff_date = window_start - pd.Timedelta(days=183)
    training_rows = transactions[
        (transactions["town"].map(_normalize_text) == _normalize_text(town))
        & (transactions["Date"] < window_start)
        & (transactions["Date"] >= cutoff_date)
    ].copy()

    training_flats: list[dict[str, Any]] = []
    for row in training_rows.itertuples(index=False):
        training_flats.append(
            {
                "town": town,
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

    flats: list[dict[str, Any]] = []
    for order_index, row in enumerate(transactions.itertuples(index=False), start=1):
        transaction_date = pd.Timestamp(row.Date)
        relative_timing = round((transaction_date - date_min).days / date_span_days, 6)
        if relative_timing < 0.4:
            initial_state = "negotiating"
        elif relative_timing < 0.8:
            initial_state = "listed"
        else:
            initial_state = "not_yet_listed"

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
            cleaned = str(item).strip()
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
                station_name = str(item.get("station_name", "")).strip()
            else:
                station_name = str(item).strip()
            if station_name and station_name not in mrt_names:
                mrt_names.append(station_name)

        school_text = str(row.pri_school_names_0_2km or "").strip()
        school_names = [part.strip() for part in school_text.split("|") if part.strip()]

        hawker_text = str(row.hawker_names_0_1km or "").strip()
        hawker_names = [part.strip() for part in hawker_text.split("|") if part.strip()]
        remaining_lease_years = round(float(row.remaining_lease) / 12.0, 2)
        observed_price = float(row.resale_price)
        past_price_trends = _build_past_price_trends(
            town_transactions,
            town=config.town,
            flat_type=str(row.flat_type_label),
            reference_date=transaction_date,
            fallback_price=observed_price,
        )

        normalized_town = _normalize_text(config.town).replace(" ", "_")
        flat_id = f"{config.year}_{normalized_town}_{order_index:05d}"
        flats.append(
            {
                "flat_id": flat_id,
                "town": config.town,
                "year": int(config.year),
                "transaction_date": transaction_date.date().isoformat(),
                "simulated_market_entry_date": (
                    transaction_date - pd.DateOffset(months=config.lead_months)
                ).date().isoformat(),
                "initialization_order": order_index,
                "relative_transaction_timing": relative_timing,
                "initial_market_state": initial_state,
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
        for name in flat["amenities"]["mrt"]["station_names"]
    ]
    nearby_amenities.extend(
        Amenity(
            name=name, type=AmenityType.SCHOOL, radius="Within 2km"
        ).model_dump()
        for name in flat["amenities"]["primary_schools"]["school_names"]
    )
    nearby_amenities.extend(
        Amenity(name=name, type=AmenityType.MALL, radius="Within 1km").model_dump()
        for name in flat["amenities"]["malls"]["mall_names"]
    )
    nearby_amenities.extend(
        Amenity(
            name=name, type=AmenityType.HAWKER, radius="Within 1km"
        ).model_dump()
        for name in flat["amenities"]["hawker_centres"]["hawker_names"]
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
    anchor_price, sigma_price = _estimate_hedonic_price(
        hedonic_training_flats,
        target_flat=flat,
    )
    reservation_price = max(0.0, rng.normalvariate(anchor_price, sigma_price))
    ask_price = max(reservation_price, anchor_price + sigma_price)
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
    frame = pd.read_csv(path).fillna("")
    subset = frame[
        (frame["planning_area"].map(_normalize_text) == _normalize_text(town))
        & (frame["age_group"].map(_normalize_text) != "total")
    ].copy()
    if subset.empty:
        raise ValueError(f"No buyer age prior found for planning area={town!r}.")
    grouped = (
        subset.groupby("age_group", as_index=False)["population"]
        .sum()
        .sort_values("age_group")
        .reset_index(drop=True)
    )
    grouped["population"] = grouped["population"].astype(float)
    return grouped[grouped["population"] > 0].copy()


def _sample_income_band(
    income_prior: pd.DataFrame, age_band: str, rng: random.Random
) -> str:
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
    subset["count"] = subset["count"].astype(float)
    subset = subset[subset["count"] > 0]
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


def _build_buyer_budget(
    buyer: dict[str, Any],
    hedonic_training_flats: list[dict[str, Any]],
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Build a buyer budget using a hedonic value draw capped by financial feasibility."""
    preferred_flat_types = buyer["preferences"].get("flat_type", [])
    training_flats = _select_hedonic_training_flats(
        hedonic_training_flats,
        town=buyer["town"],
        preferred_flat_types=preferred_flat_types,
    )
    anchor_price, sigma_price = _estimate_hedonic_price(training_flats)
    buyer_value_draw = max(0.0, rng.normalvariate(anchor_price, sigma_price))
    max_price = min(float(buyer["financials"]["effective_ceiling"]), buyer_value_draw)
    min_price = max(0.0, anchor_price - sigma_price)
    if max_price < min_price:
        min_price = max(0.0, min(max_price, min_price))
    return BuyerBudgetRange(
        min_price=round(min_price, 2),
        max_price=round(max_price, 2),
    ).model_dump()


def _build_broad_buyers(
    flats: list[dict[str, Any]],
    hedonic_training_flats: list[dict[str, Any]],
    donors: pd.DataFrame,
    *,
    config: SegmentConfig,
    rng: random.Random,
    age_prior: pd.DataFrame,
    income_prior: pd.DataFrame,
    distribution_tables: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    """Generate the broad buyer pool before feasibility filtering."""
    broad_count = max(len(flats), math.ceil(len(flats) * config.buyer_pool_multiplier))
    buyers: list[dict[str, Any]] = []

    for index in range(1, broad_count + 1):
        age_band = str(_weighted_choice(age_prior, "age_group", "population", rng))
        income_band = _sample_income_band(income_prior, age_band, rng)
        age = _sample_age_from_band(age_band, rng)

        donor = _sample_nemotron_donor(
            donors,
            rng=rng,
            planning_area=config.town,
            age=age,
        )

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
            forced_max_cov=monthly_income,
        ).__dict__

        buyer_record = {
            "buyer_id": f"buyer_{config.year}_{index:05d}",
            "town": config.town,
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
            "preferences": BuyerPreferenceProfile().model_dump(),
        }
        buyer_record["budget"] = _build_buyer_budget(
            buyer_record,
            hedonic_training_flats,
            rng=rng,
        )
        buyers.append(buyer_record)

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
        buyer["preferences"] = result.model_dump()
        buyer["budget"] = _build_buyer_budget(
            buyer,
            hedonic_training_flats,
            rng=rng,
        )


def _retain_feasible_buyers(
    buyers: list[dict[str, Any]],
    flats: list[dict[str, Any]],
    archetypes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only buyers who can financially reach at least one flat in the seller pool."""
    flat_index = {flat["flat_id"]: flat for flat in flats}
    retained: list[dict[str, Any]] = []

    for buyer in buyers:
        effective_ceiling = float(buyer["financials"]["effective_ceiling"])
        feasible_flat_ids = [
            flat["flat_id"]
            for flat in flats
            if float(flat["observed_resale_price"]) <= effective_ceiling
        ]
        buyer["feasible_flat_ids"] = feasible_flat_ids
        if not feasible_flat_ids:
            continue

        reachable_flats = [flat_index[flat_id] for flat_id in feasible_flat_ids]
        buyer["preferences"] = BuyerPreferenceProfile().model_dump()
        buyer["preference_classification_input"] = _build_preference_classification_input(
            buyer,
            reachable_flats,
            archetypes,
        )
        buyer["retained"] = True
        retained.append(buyer)

    return retained


def build_transaction_conditioned_segment(
    config: SegmentConfig,
    model: VLLMLanguageModel | None = None,
) -> dict[str, Any]:
    """Run the end-to-end preprocessing pipeline for one town-year market segment."""
    logging.info(
        "Building transaction-conditioned market segment for town=%s year=%s.",
        config.town,
        config.year,
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
    if config.survey_archetypes_path is None:
        archetypes = DEFAULT_BUYER_ARCHETYPES
        logging.info(
            "Using default buyer archetypes because no survey archetypes path was provided."
        )
    else:
        archetypes = json.loads(config.survey_archetypes_path.read_text(encoding="utf-8"))
        if not isinstance(archetypes, list) or not archetypes:
            raise ValueError("Buyer archetype config must be a non-empty JSON list.")
        logging.info(
            "Loaded %s buyer archetype entries from %s.",
            len(archetypes),
            config.survey_archetypes_path,
        )

    if config.seller_archetypes_path is None:
        seller_archetypes = DEFAULT_SELLER_ARCHETYPES
        logging.info(
            "Using default seller archetypes because no seller archetypes path was provided."
        )
    else:
        seller_archetypes = json.loads(
            config.seller_archetypes_path.read_text(encoding="utf-8")
        )
        if not isinstance(seller_archetypes, list) or not seller_archetypes:
            raise ValueError("Seller archetype config must be a non-empty JSON list.")
        logging.info(
            "Loaded %s seller archetype entries from %s.",
            len(seller_archetypes),
            config.seller_archetypes_path,
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
    flats = _build_flat_universe(transactions, town_transactions, config)
    sellers = _build_sellers(
        flats,
        hedonic_training_flats,
        distribution_tables,
        donors,
        seller_archetypes,
        config=config,
        rng=rng,
    )
    broad_buyers = _build_broad_buyers(
        flats,
        hedonic_training_flats,
        donors,
        config=config,
        rng=rng,
        age_prior=age_prior,
        income_prior=income_prior,
        distribution_tables=distribution_tables,
    )
    retained_buyers = _retain_feasible_buyers(broad_buyers, flats, archetypes)
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
        "flat_units_path": str(flats_path),
        "sellers_path": str(sellers_path),
        "buyers_broad_path": str(buyers_broad_path),
        "buyers_retained_path": str(buyers_retained_path),
    }
    _write_json(manifest_path, manifest)
    logging.info("Saved market segment manifest to %s.", manifest_path)
    return manifest
