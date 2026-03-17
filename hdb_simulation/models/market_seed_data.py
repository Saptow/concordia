"""Lightweight market seeding helpers for HDB listing and negotiation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import random
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PERSONA_DATA_DIR = Path('data') / 'nemotron'
PERSONA_COLUMNS = (
    'persona',
    'cultural_background',
    'skills_and_expertise',
    'hobbies_and_interests',
    'age',
    'occupation',
    'planning_area',
)
SUPPORTED_FLAT_TYPES = {
    '1 ROOM': '1-Room',
    '2 ROOM': '2-Room',
    '3 ROOM': '3-Room',
    '4 ROOM': '4-Room',
    '5 ROOM': '5-Room',
    'EXECUTIVE': 'Executive',
}


@dataclass(frozen=True)
class MarketSeedPair:
    buyer_id: str
    buyer_profile: dict[str, Any]
    seller_id: str
    seller_profile: dict[str, Any]


def _normalize_text(value: str) -> str:
    return str(value).strip().casefold()


def _title_case_words(value: str) -> str:
    return ' '.join(part.capitalize() for part in str(value).strip().split())


def _coerce_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_remaining_lease_years(value: str, *, sale_year: int, lease_commence_date: int) -> float:
    numeric_value = _coerce_float(value)
    if numeric_value is not None:
        return max(0.0, numeric_value)
    text = str(value or '').strip().lower()
    if text:
        years_match = re.search(r'(\d+)\s+year', text)
        months_match = re.search(r'(\d+)\s+month', text)
        years = float(years_match.group(1)) if years_match else 0.0
        months = float(months_match.group(1)) if months_match else 0.0
        if years or months:
            return years + (months / 12.0)
    remaining_years = 99 - max(0, sale_year - int(lease_commence_date))
    return max(0.0, float(remaining_years))


def _load_resale_rows(
    *,
    resale_csv_path: str | Path,
    planning_area: str,
    year: int,
    max_pairs: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    csv_path = Path(resale_csv_path)
    target_area = _normalize_text(planning_area)
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            month = str(row.get('month', '')).strip()
            if not month.startswith(f'{int(year):04d}-'):
                continue
            if _normalize_text(row.get('town', '')) != target_area:
                continue
            if str(row.get('flat_type', '')).strip().upper() not in SUPPORTED_FLAT_TYPES:
                continue
            rows.append(row)
            if len(rows) >= max_pairs:
                break
    if not rows:
        raise ValueError(
            f'No resale rows found for planning area={planning_area!r} year={year}.'
        )
    return rows


def _load_persona_rows(
    *,
    planning_area: str,
    count: int,
    persona_data_dir: str | Path,
) -> list[dict[str, Any]]:
    target_area = _normalize_text(planning_area)
    matches: list[dict[str, Any]] = []
    parquet_files = sorted(Path(persona_data_dir).glob('*.parquet'))
    if not parquet_files:
        raise ValueError(
            f'No Nemotron parquet files found under {Path(persona_data_dir)!s}.'
        )
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file, columns=list(PERSONA_COLUMNS))
        filtered = frame[
            frame['planning_area'].fillna('').map(_normalize_text) == target_area
        ]
        if filtered.empty:
            continue
        matches.extend(filtered.head(count - len(matches)).to_dict(orient='records'))
        if len(matches) >= count:
            break
    if len(matches) < count:
        raise ValueError(
            f'Only found {len(matches)} personas for planning area={planning_area!r}; '
            f'need {count}.'
        )
    return matches


def _build_profile_description(persona_row: dict[str, Any], *, role: str) -> str:
    age = int(persona_row.get('age', 0) or 0)
    occupation = str(persona_row.get('occupation', 'resident')).strip()
    planning_area = str(persona_row.get('planning_area', 'Singapore')).strip()
    persona = str(persona_row.get('persona', '')).strip()
    cultural_background = str(persona_row.get('cultural_background', '')).strip()
    skills = str(persona_row.get('skills_and_expertise', '')).strip()
    hobbies = str(persona_row.get('hobbies_and_interests', '')).strip()
    summary = f'{age}-year-old {occupation} from {planning_area}. {persona}'.strip()
    extras = ' '.join(
        part for part in (cultural_background, skills, hobbies) if part
    ).strip()
    if role == 'buyer':
        return (
            f'{summary} This buyer is participating in an HDB resale search and '
            f'negotiation. {extras}'
        ).strip()
    return (
        f'{summary} This seller is listing an HDB resale flat and may negotiate on '
        f'price and terms. {extras}'
    ).strip()


def _build_flat_description(
    *,
    planning_area_name: str,
    flat_type: str,
    floor_area_sqm: float,
    flat_model: str,
    resale_row: dict[str, Any],
    randomizer: random.Random,
) -> str:
    opening_templates = [
        f'Well-kept {flat_type} resale flat in {planning_area_name}',
        f'Practical {flat_type} unit in {planning_area_name}',
        f'Comfortable {flat_type} home in {planning_area_name}',
        f'Functional {flat_type} resale listing in {planning_area_name}',
    ]
    size_templates = [
        f'around {floor_area_sqm:.0f} sqm',
        f'with about {floor_area_sqm:.0f} sqm of space',
        f'offering roughly {floor_area_sqm:.0f} sqm',
    ]
    model_templates = [
        f'configured as a {flat_model} model',
        f'with a {flat_model} layout',
        f'built on the {flat_model} model',
    ]
    description_bits = [
        randomizer.choice(opening_templates),
        randomizer.choice(size_templates),
        randomizer.choice(model_templates),
    ]
    mrt_name = str(resale_row.get('Name', '')).strip()
    mrt_distance_m = _coerce_float(resale_row.get('min_mrt_distance_m'))
    school_distance_m = _coerce_float(resale_row.get('min_school_distance_m'))
    if mrt_name and mrt_distance_m is not None:
        description_bits.append(
            randomizer.choice([
                f'about {mrt_distance_m:.0f}m from {mrt_name}',
                f'with MRT access via {mrt_name} roughly {mrt_distance_m:.0f}m away',
                f'and located around {mrt_distance_m:.0f}m from {mrt_name}',
            ])
        )
    if school_distance_m is not None:
        description_bits.append(
            randomizer.choice([
                f'about {school_distance_m:.0f}m from the nearest school',
                f'with the nearest school roughly {school_distance_m:.0f}m away',
                f'and school access within about {school_distance_m:.0f}m',
            ])
        )
    return ' '.join(description_bits) + '.'


def _build_random_buyer_preferences(
    *,
    planning_area_name: str,
    floor_area_sqm: float,
    resale_row: dict[str, Any],
    randomizer: random.Random,
) -> dict[str, Any]:
    flat_type_choices = list(SUPPORTED_FLAT_TYPES.values())
    flat_type_count = randomizer.randint(1, min(3, len(flat_type_choices)))
    flat_types = randomizer.sample(flat_type_choices, k=flat_type_count)
    if str(resale_row.get('flat_type', '')).strip().upper() in SUPPORTED_FLAT_TYPES:
        actual_flat_type = SUPPORTED_FLAT_TYPES[str(resale_row['flat_type']).strip().upper()]
        if actual_flat_type not in flat_types:
            flat_types[0] = actual_flat_type

    feature_pool = [
        'near MRT access',
        'good primary school access',
        'higher floor preferred',
        'more open views',
        'efficient layout',
        'longer remaining lease',
        'larger floor area',
    ]
    feature_count = randomizer.randint(2, 4)
    selected_features = randomizer.sample(feature_pool, k=feature_count)
    if _coerce_float(resale_row.get('min_mrt_distance_m')) is not None:
        selected_features.append('walkable transit option')
    if _coerce_float(resale_row.get('min_school_distance_m')) is not None:
        selected_features.append('close to schools')

    return {
        'flat_type': flat_types,
        'towns': [planning_area_name],
        'features': (
            f"Prefers flats around {floor_area_sqm:.0f} sqm in {planning_area_name} with "
            + ', '.join(dict.fromkeys(selected_features))
            + '.'
        ),
    }


def build_market_seed_pairs(
    *,
    resale_csv_path: str | Path,
    persona_data_dir: str | Path = DEFAULT_PERSONA_DATA_DIR,
    planning_area: str,
    year: int,
    max_pairs: int = 6,
    random_seed: int = 0,
) -> list[MarketSeedPair]:
    """Build buyer/seller seed pairs from flat transactions plus Singapore personas."""
    if max_pairs <= 0:
        raise ValueError('max_pairs must be greater than zero.')

    resale_rows = _load_resale_rows(
        resale_csv_path=resale_csv_path,
        planning_area=planning_area,
        year=year,
        max_pairs=max_pairs,
    )
    persona_rows = _load_persona_rows(
        planning_area=planning_area,
        count=max_pairs * 2,
        persona_data_dir=persona_data_dir,
    )

    seed_pairs: list[MarketSeedPair] = []
    planning_area_name = _title_case_words(planning_area)
    randomizer = random.Random(random_seed)
    for index, resale_row in enumerate(resale_rows, start=1):
        buyer_persona = persona_rows[(index - 1) * 2]
        seller_persona = persona_rows[(index - 1) * 2 + 1]

        sale_year = int(str(resale_row['month']).split('-', maxsplit=1)[0])
        flat_type = SUPPORTED_FLAT_TYPES[str(resale_row['flat_type']).strip().upper()]
        resale_price = float(resale_row['resale_price'])
        floor_area_sqm = float(resale_row['floor_area_sqm'])
        lease_commence_date = int(resale_row['lease_commence_date'])
        remaining_lease = _parse_remaining_lease_years(
            resale_row.get('remaining_lease', ''),
            sale_year=sale_year,
            lease_commence_date=lease_commence_date,
        )
        address = (
            f"Blk {str(resale_row['block']).strip()} "
            f"{_title_case_words(resale_row['street_name'])}, Singapore"
        )
        flat_description = _build_flat_description(
            planning_area_name=planning_area_name,
            flat_type=flat_type,
            floor_area_sqm=floor_area_sqm,
            flat_model=str(resale_row['flat_model']).strip(),
            resale_row=resale_row,
            randomizer=randomizer,
        )
        seller_id = f'seller_{index:03d}'
        buyer_id = f'buyer_{index:03d}'

        seller_profile = {
            'name': f'{planning_area_name} Seller {index:03d}',
            'age': int(seller_persona.get('age', 0) or 0),
            'occupation': str(seller_persona.get('occupation', 'Resident')).strip(),
            'description': _build_profile_description(seller_persona, role='seller'),
            'flat': {
                'flat_type': flat_type,
                'address': address,
                'description': flat_description,
                'town': planning_area_name,
                'storey_range': str(resale_row['storey_range']).strip(),
                'remaining_lease': remaining_lease,
                'contra': False,
                'extension_of_stay': False,
                'minimum_occupancy_period_completed': True,
                'ethnic_eligibility': 'Unknown',
                'spr_eligibility': 'Unknown',
                'floor_area_sqm': floor_area_sqm,
                'nearby_amenities': [],
            },
            'expectations': {
                'min_price': round(resale_price * 0.98, 2),
                'max_price': round(resale_price * 1.02, 2),
            },
        }
        buyer_profile = {
            'name': f'{planning_area_name} Buyer {index:03d}',
            'age': int(buyer_persona.get('age', 0) or 0),
            'occupation': str(buyer_persona.get('occupation', 'Resident')).strip(),
            'description': _build_profile_description(buyer_persona, role='buyer'),
            'budget': {
                'min_price': round(resale_price * 0.90, 2),
                'max_price': round(resale_price * 1.08, 2),
            },
            'preferences': _build_random_buyer_preferences(
                planning_area_name=planning_area_name,
                floor_area_sqm=floor_area_sqm,
                resale_row=resale_row,
                randomizer=randomizer,
            ),
        }
        seed_pairs.append(
            MarketSeedPair(
                buyer_id=buyer_id,
                buyer_profile=buyer_profile,
                seller_id=seller_id,
                seller_profile=seller_profile,
            )
        )
    return seed_pairs
