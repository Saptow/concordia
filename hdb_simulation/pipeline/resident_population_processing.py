"""Clean resident population census data into planning-area age-group priors."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from configs import PROCESSED_DIR
from configs import REPO_ROOT


RAW_RESIDENT_POPULATION_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "ResidentPopulationbyPlanningAreaSubzoneofResidenceAgeGroupandSexCensusofPopulation2020.csv"
)
OUTPUT_SUFFIX = "_age_groups.csv"
AGE_GROUPS = (
    ("20_24", "20 - 24"),
    ("25_29", "25 - 29"),
    ("30_34", "30 - 34"),
    ("35_39", "35 - 39"),
    ("40_44", "40 - 44"),
    ("45_49", "45 - 49"),
    ("50_54", "50 - 54"),
    ("55_59", "55 - 59"),
    ("60_64", "60 - 64"),
    ("65_69", "65 - 69"),
    ("70_74", "70 - 74"),
    ("75_79", "75 - 79"),
)


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _resolve_output_path(planning_area: str) -> Path:
    return PROCESSED_DIR / f"{_slugify(planning_area)}{OUTPUT_SUFFIX}"


def _load_raw_resident_population() -> pd.DataFrame:
    return pd.read_csv(RAW_RESIDENT_POPULATION_PATH).fillna("")


def _select_planning_area_total_row(
    frame: pd.DataFrame,
    planning_area: str,
) -> pd.Series:
    normalized_area = _normalize_text(planning_area)
    number_col = frame["Number"].map(_normalize_text)
    total_label = _normalize_text(f"{planning_area} - Total")

    exact_total_matches = frame[number_col == total_label]
    if not exact_total_matches.empty:
        return exact_total_matches.iloc[0]

    exact_matches = frame[number_col == normalized_area]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    raise ValueError(
        f"Could not find a planning-area total row for planning_area={planning_area!r}."
    )


def build_planning_area_age_groups(planning_area: str) -> pd.DataFrame:
    """Return processed age-group totals for one planning area."""
    raw_frame = _load_raw_resident_population()
    row = _select_planning_area_total_row(raw_frame, planning_area)

    records = []
    for raw_age_group, output_age_group in AGE_GROUPS:
        value = pd.to_numeric(row[f"Total_{raw_age_group}"], errors="coerce")
        if pd.isna(value):
            raise ValueError(
                f"Missing total population for age_group={output_age_group!r} "
                f"and planning_area={planning_area!r}."
            )
        records.append(
            {
                "planning_area": planning_area,
                "sex": "Total",
                "age_group": output_age_group,
                "population": int(value),
            }
        )

    return pd.DataFrame(
        records,
        columns=["planning_area", "sex", "age_group", "population"],
    )


def write_planning_area_age_groups(
    planning_area: str,
    *,
    output_path: Path | None = None,
) -> Path:
    """Write one planning area's processed age-group CSV and return the path."""
    resolved_output_path = output_path or _resolve_output_path(planning_area)
    frame = build_planning_area_age_groups(planning_area)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved_output_path, index=False)
    return resolved_output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert resident population census totals into age-group priors."
    )
    parser.add_argument(
        "--planning-area",
        required=True,
        help="Planning area name, for example 'Choa Chu Kang'.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to data/processed/<planning_area>_age_groups.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_path = write_planning_area_age_groups(
        args.planning_area,
        output_path=args.output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
