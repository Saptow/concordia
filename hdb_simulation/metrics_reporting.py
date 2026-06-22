"""Helpers for simulation metrics, tables, and chart rendering.

This module keeps reporting-specific logic out of ``main.py`` so the entrypoint
can focus on assembling the simulation and saving artifacts. The helpers here
cover three reporting jobs:

1. Compute aggregate transaction metrics for the HTML scoreboard.
2. Build grouped price comparison tables for flat types and towns.
3. Render overall and per-town price trend charts for the summary HTML.
"""

from __future__ import annotations

import base64
import html
import io
import statistics
from datetime import datetime
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from configs import SimulationConfig
from concordia.concordia.utils import structured_logging

# Aggregate weekly price movement into larger windows so the rendered chart
# stays readable even for longer runs.
PRICE_TREND_BUCKET_WEEKS = 4


def _coerce_positive_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and float(value) > 0.0:
        return float(value)
    return None


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_float_metric(value: float | None, *, digits: int) -> str:
    if value is None:
        return 'NA'
    return f'{value:.{digits}f}'


def _iter_record_actions(record: dict[str, Any]):
    for event in record.get('events', ()):
        if not isinstance(event, dict):
            continue
        action = event.get('action')
        if isinstance(action, dict):
            yield action


def _count_sequence_items(value: object) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _extract_week_summary_from_log_entry(
    simulation_log: structured_logging.SimulationLog,
    entry: structured_logging.StructuredLogEntry,
) -> dict[str, Any] | None:
    if entry.entry_type != 'step':
        return None
    payload = simulation_log.reconstruct_value(dict(entry.deduplicated_data))
    if not isinstance(payload, dict):
        return None
    value = payload.get('value')
    if not isinstance(value, dict):
        return None
    week_summary = value.get('week_summary')
    if not isinstance(week_summary, dict):
        return None
    return dict(week_summary)


def _collect_completed_week_summaries(
    simulation_log: structured_logging.SimulationLog,
) -> list[dict[str, Any]]:
    summaries_by_week: dict[int, dict[str, Any]] = {}

    for entry in simulation_log.entries:
        week_summary = _extract_week_summary_from_log_entry(simulation_log, entry)
        if week_summary is None:
            continue
        week_number = week_summary.get('week_number', entry.step)
        normalized_week_number = entry.step
        if isinstance(week_number, (int, float)):
            normalized_week_number = max(int(round(float(week_number))), 1)
        summaries_by_week[normalized_week_number] = week_summary

    return [
        summaries_by_week[week_number]
        for week_number in sorted(summaries_by_week)
    ]


def _compute_weekly_market_activity_metrics(
    simulation_log: structured_logging.SimulationLog,
) -> dict[str, float]:
    """Summarise simple weekly operational metrics from coordinator snapshots."""
    week_summaries = _collect_completed_week_summaries(simulation_log)
    if not week_summaries:
        return {}

    total_weeks = float(len(week_summaries))
    total_closures = 0

    for week_summary in week_summaries:
        negotiation_summary = week_summary.get('negotiation', {})
        if not isinstance(negotiation_summary, dict):
            negotiation_summary = {}
        total_closures += _count_sequence_items(
            negotiation_summary.get('closed_pairs')
        )

    return {
        'Average closed records per week': total_closures / total_weeks,
    }


def _normalize_market_state(value: object) -> str:
    return str(value or '').strip().casefold().replace('-', '_').replace(' ', '_')


def _normalize_flat_type(value: object) -> str:
    flat_type = str(value or '').strip()
    return flat_type or 'Unknown'


def _normalize_town(value: object) -> str:
    town = str(value or '').strip()
    return town or 'Unknown'


def _extract_successful_records(
    replay_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_records = _dedupe_replay_records(replay_records)
    return [
        record
        for record in normalized_records
        if bool(record.get('closed'))
        and str(record.get('outcome', '')).strip().upper() == 'SUCCESS'
    ]


def _max_event_sequence(record: dict[str, Any]) -> int:
    max_sequence = 0
    for event in record.get('events', ()):
        if not isinstance(event, dict):
            continue
        sequence = event.get('sequence')
        if isinstance(sequence, (int, float)):
            max_sequence = max(max_sequence, int(sequence))
    return max_sequence


def _record_identity(record: dict[str, Any]) -> str | None:
    pair_key = str(record.get('pair_key', '')).strip()
    if pair_key:
        return pair_key
    seller_id = str(record.get('seller_id', '')).strip()
    buyer_id = str(record.get('buyer_id', '')).strip()
    if seller_id and buyer_id:
        return f'{buyer_id}|||{seller_id}'
    return seller_id or None


def _record_completeness_key(record: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(bool(record.get('closed'))),
        int(str(record.get('outcome', '')).strip().upper() == 'SUCCESS'),
        len(record.get('events', ())) if isinstance(record.get('events'), list) else 0,
        _max_event_sequence(record),
    )


def _dedupe_replay_records(
    replay_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer the most complete replay snapshot when the same pair appears twice."""
    deduped: dict[str, dict[str, Any]] = {}
    anonymous_records: list[dict[str, Any]] = []

    for raw_record in replay_records:
        if not isinstance(raw_record, dict):
            continue
        record = dict(raw_record, events=list(raw_record.get('events', ())))
        record_id = _record_identity(record)
        if not record_id:
            anonymous_records.append(record)
            continue
        existing = deduped.get(record_id)
        if (
            existing is None
            or _record_completeness_key(record) > _record_completeness_key(existing)
        ):
            deduped[record_id] = record

    return anonymous_records + list(deduped.values())


def _count_listed_successes_exceeding_avg_success_weeks(
    *,
    seller_profiles: dict[str, dict[str, object]],
    successful_records: list[dict[str, Any]],
) -> int:
    """Count listed-at-start sellers whose successful close exceeds the threshold."""
    threshold_weeks = float(SimulationConfig.AVG_SUCCESS_WEEKS)
    exceeding_count = 0

    for record in successful_records:
        seller_id = str(record.get('seller_id', '')).strip()
        if not seller_id:
            continue

        seller_profile = seller_profiles.get(seller_id, {})
        if _normalize_market_state(seller_profile.get('initial_market_state')) != 'listed':
            continue

        listing_start_week = int(seller_profile.get('listing_release_week', 1) or 1)
        end_week = record.get('end_week')
        if not isinstance(end_week, int):
            continue

        weeks_to_close = max(0, end_week - listing_start_week + 1)
        if weeks_to_close > threshold_weeks:
            exceeding_count += 1

    return exceeding_count


def listed_successes_exceeding_avg_success_weeks_label() -> str:
    return (
        'Successful closes from initially listed sellers taking more than '
        f'{SimulationConfig.AVG_SUCCESS_WEEKS} weeks'
    )


def compute_transaction_metrics(
    *,
    seller_profiles: dict[str, dict[str, object]],
    replay_records: list[dict[str, Any]],
    simulation_log: structured_logging.SimulationLog,
) -> dict[str, Any]:
    """Build the compact scoreboard shown above the HTML summary sections.

    The intent here is to keep a small set of high-signal metrics that are easy
    to scan while still grounding them in the replay artifacts produced by the
    simulation.
    """
    total_pairs = max(len(seller_profiles), 0)
    closed_records = [
        record
        for record in replay_records
        if isinstance(record, dict) and bool(record.get('closed'))
    ]
    successful_records = _extract_successful_records(replay_records)
    successful_pairs = len(successful_records)
    closed_pairs = len(closed_records)
    success_rate_pct = (
        (successful_pairs / total_pairs) * 100.0 if total_pairs else 0.0
    )
    listed_successes_exceeding_avg_success_weeks = (
        _count_listed_successes_exceeding_avg_success_weeks(
            seller_profiles=seller_profiles,
            successful_records=successful_records,
        )
    )

    absolute_price_errors: list[float] = []
    negotiation_lengths: list[float] = []
    offer_counts: list[float] = []

    # These per-success aggregates are reused by both the summary scoreboard and
    # the richer breakdown tables below, so they stay close to the replay data.
    for record in successful_records:
        seller_id = str(record.get('seller_id', '')).strip()
        seller_profile = seller_profiles.get(seller_id, {})
        observed_price = _coerce_positive_float(
            seller_profile.get('observed_resale_price')
        )
        agreed_price = _extract_final_agreed_price(record)
        if observed_price is not None and agreed_price is not None:
            absolute_price_errors.append(abs(agreed_price - observed_price))

        start_week = record.get('start_week')
        end_week = record.get('end_week')
        if isinstance(start_week, int) and isinstance(end_week, int):
            negotiation_lengths.append(float(max(0, end_week - start_week + 1)))

        offer_counts.append(float(_extract_offer_count(record)))

    metrics = {
        'Successful Transactions': f'{successful_pairs}/{total_pairs}',
        'Success Rate (%)': f'{success_rate_pct:.1f}',
        'Closed/Successful Pairs': f'{closed_pairs}/{successful_pairs}',
        listed_successes_exceeding_avg_success_weeks_label(): (
            listed_successes_exceeding_avg_success_weeks
        ),
    }
    for label, value in _compute_weekly_market_activity_metrics(simulation_log).items():
        metrics[label] = _format_float_metric(value, digits=2)
    optional_metrics = (
        ('Mean Absolute Price Error (SGD)', _safe_mean(absolute_price_errors), 0),
        ('Average Negotiation Length (weeks)', _safe_mean(negotiation_lengths), 1),
        ('Average Offer Count', _safe_mean(offer_counts), 1),
    )
    for label, value, digits in optional_metrics:
        if value is not None:
            metrics[label] = _format_float_metric(value, digits=digits)
    return metrics


def _extract_offer_price_from_action(action: dict[str, Any]) -> float | None:
    action_type = str(action.get('type', '')).strip().upper()
    if action_type == 'MAKE_OFFER':
        value = action.get('offer_price')
    elif action_type == 'MAKE_COUNTEROFFER':
        value = action.get('counteroffer_price')
    else:
        return None
    return _coerce_positive_float(value)


def _extract_final_agreed_price(record: dict[str, Any]) -> float | None:
    if str(record.get('outcome', '')).strip().upper() != 'SUCCESS':
        return None
    last_offer_price: float | None = None
    for action in _iter_record_actions(record):
        offer_price = _extract_offer_price_from_action(action)
        if offer_price is not None:
            last_offer_price = offer_price
        if str(action.get('type', '')).strip().upper() == 'ACCEPT_OFFER':
            settled_price = _coerce_positive_float(action.get('price_settled'))
            if settled_price is not None:
                return settled_price
            return last_offer_price
    return None


def _extract_offer_count(record: dict[str, Any]) -> int:
    count = 0
    for action in _iter_record_actions(record):
        if _extract_offer_price_from_action(action) is not None:
            count += 1
    return count


def _extract_observed_transaction_week(
    seller_profile: dict[str, object],
) -> int | None:
    transaction_date = str(seller_profile.get('transaction_date', '') or '').strip()
    if transaction_date:
        try:
            observed_date = datetime.fromisoformat(transaction_date).date()
        except ValueError:
            observed_date = None
        if observed_date is not None:
            baseline_date = seller_profile.get('_observed_timeline_start_date')
            if hasattr(baseline_date, 'toordinal'):
                return max(((observed_date - baseline_date).days // 7) + 1, 1)
    initialization_order = seller_profile.get('initialization_order')
    if isinstance(initialization_order, (int, float)):
        return max(int(round(float(initialization_order))), 1)
    return None


def _extract_simulated_transaction_week(
    record: dict[str, Any],
    seller_profile: dict[str, object],
) -> int | None:
    end_week = record.get('end_week')
    if isinstance(end_week, (int, float)):
        return max(int(round(float(end_week))), 1)
    initialization_order = seller_profile.get('initialization_order')
    if isinstance(initialization_order, (int, float)):
        return max(int(round(float(initialization_order))), 1)
    return None


def _build_simulated_close_entries(
    *,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    for record in successful_records:
        seller_id = str(record.get('seller_id', '')).strip()
        seller_profile = seller_profiles.get(seller_id)
        if not isinstance(seller_profile, dict):
            continue
        timing = _extract_simulated_transaction_week(record, seller_profile)
        agreed_price = _extract_final_agreed_price(record)
        if timing is None or agreed_price is None or agreed_price <= 0.0:
            continue
        entries.append({
            'seller_id': seller_id,
            'town': _normalize_town(seller_profile.get('town')),
            'flat_type': _normalize_flat_type(seller_profile.get('flat_type')),
            'price': agreed_price,
            'timing': timing,
        })

    return entries


def _build_observed_price_entries(
    *,
    seller_profiles: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for seller_id, seller_profile in seller_profiles.items():
        if not isinstance(seller_profile, dict):
            continue
        observed_price = _coerce_positive_float(
            seller_profile.get('observed_resale_price')
        )
        if observed_price is None:
            continue
        transaction_date = str(seller_profile.get('transaction_date', '') or '').strip()
        observed_date = None
        if transaction_date:
            try:
                observed_date = datetime.fromisoformat(transaction_date).date()
            except ValueError:
                observed_date = None
        entries.append({
            'seller_id': str(seller_id),
            'town': _normalize_town(seller_profile.get('town')),
            'flat_type': _normalize_flat_type(seller_profile.get('flat_type')),
            'price': observed_price,
            'observed_date': observed_date,
            'initialization_order': seller_profile.get('initialization_order'),
        })
    return entries


def _observed_entry_timing(
    entry: dict[str, object],
    *,
    baseline_date,
) -> int | None:
    observed_date = entry.get('observed_date')
    if hasattr(observed_date, 'toordinal') and hasattr(baseline_date, 'toordinal'):
        return max(((observed_date - baseline_date).days // 7) + 1, 1)
    initialization_order = entry.get('initialization_order')
    if isinstance(initialization_order, (int, float)):
        return max(int(round(float(initialization_order))), 1)
    return None


def _format_currency(value: float | None) -> str:
    if value is None:
        return 'NA'
    return f'${value:,.0f}'


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _format_currency_range(values: list[float]) -> str:
    if not values:
        return 'NA'
    return f'{_format_currency(min(values))} to {_format_currency(max(values))}'


def _group_by_flat_type(sample: dict[str, object]) -> tuple[str, ...]:
    return (_normalize_flat_type(sample.get('flat_type')),)


def _group_by_town(sample: dict[str, object]) -> tuple[str, ...]:
    return (_normalize_town(sample.get('town')),)


def _group_by_town_and_flat_type(sample: dict[str, object]) -> tuple[str, ...]:
    return (
        _normalize_town(sample.get('town')),
        _normalize_flat_type(sample.get('flat_type')),
    )


def _build_grouped_price_breakdown_html(
    *,
    title: str,
    key_headers: tuple[str, ...],
    scope_label: str,
    empty_state_label: str,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
    group_key_fn,
) -> str:
    """Render one comparison table for observed vs simulated prices.

    The grouping function determines whether this becomes a flat-type table, a
    town table, or a combined town x flat-type table.
    """
    observed_samples = _build_observed_price_entries(
        seller_profiles=seller_profiles,
    )
    simulated_samples = _build_simulated_close_entries(
        successful_records=successful_records,
        seller_profiles=seller_profiles,
    )
    seller_counts_by_group: dict[tuple[str, ...], int] = {}
    successful_counts_by_group: dict[tuple[str, ...], int] = {}
    mae_by_group: dict[tuple[str, ...], list[float]] = {}

    for seller_profile in seller_profiles.values():
        if not isinstance(seller_profile, dict):
            continue
        group_key = tuple(str(value) for value in group_key_fn(seller_profile))
        seller_counts_by_group[group_key] = (
            seller_counts_by_group.get(group_key, 0) + 1
        )

    for sample in simulated_samples:
        group_key = tuple(str(value) for value in group_key_fn(sample))
        successful_counts_by_group[group_key] = (
            successful_counts_by_group.get(group_key, 0) + 1
        )

    observed_by_group: dict[tuple[str, ...], list[float]] = {}
    for sample in observed_samples:
        group_key = tuple(str(value) for value in group_key_fn(sample))
        observed_by_group.setdefault(group_key, []).append(float(sample['price']))

    simulated_by_group: dict[tuple[str, ...], list[float]] = {}
    for sample in simulated_samples:
        group_key = tuple(str(value) for value in group_key_fn(sample))
        simulated_by_group.setdefault(group_key, []).append(float(sample['price']))

    successful_by_seller_id = {
        str(sample.get('seller_id', '')): sample for sample in simulated_samples
    }
    for seller_id, seller_profile in seller_profiles.items():
        observed_price = _coerce_positive_float(
            seller_profile.get('observed_resale_price')
        )
        simulated_sample = successful_by_seller_id.get(str(seller_id))
        if observed_price is None or not isinstance(simulated_sample, dict):
            continue
        simulated_price = _coerce_positive_float(simulated_sample.get('price'))
        if simulated_price is None:
            continue
        group_key = tuple(str(value) for value in group_key_fn(seller_profile))
        mae_by_group.setdefault(group_key, []).append(
            abs(simulated_price - observed_price)
        )

    all_group_keys = sorted({
        *seller_counts_by_group.keys(),
        *observed_by_group.keys(),
        *simulated_by_group.keys(),
        *mae_by_group.keys(),
    })

    if not all_group_keys:
        return (
            f'<section class="summary-card"><h2>{html.escape(title)}</h2>'
            f'<div class="empty-state">No successful transactions were available '
            f'for {html.escape(empty_state_label)}.</div></section>'
        )

    rows: list[str] = []
    for group_key in all_group_keys:
        seller_count = seller_counts_by_group.get(group_key, 0)
        successful_count = successful_counts_by_group.get(group_key, 0)
        success_rate = (
            (successful_count / seller_count) * 100.0 if seller_count else None
        )
        observed_prices = observed_by_group.get(group_key, [])
        simulated_prices = simulated_by_group.get(group_key, [])
        mae_values = mae_by_group.get(group_key, [])
        group_key_cells = ''.join(
            (
                '<td style="padding: 8px 10px; text-align: left;">'
                f'{html.escape(str(value))}</td>'
            )
            for value in group_key
        )
        rows.append(
            '<tr>'
            f'{group_key_cells}'
            f'<td style="padding: 8px 10px; text-align: right;">{seller_count}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{successful_count}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_float_metric(success_rate, digits=1) if success_rate is not None else "NA"}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{len(observed_prices)}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency(_safe_mean(observed_prices))}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency_range(observed_prices)}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{len(simulated_prices)}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency(_safe_mean(simulated_prices))}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency_range(simulated_prices)}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency(_safe_median(simulated_prices))}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_currency(_safe_mean(mae_values))}</td>'
            '</tr>'
        )

    successful_with_explicit_price = len(simulated_samples)
    excluded_successful_records = max(
        len(successful_records) - successful_with_explicit_price,
        0,
    )
    note = (
        'Success rate is computed as successful sellers over total sellers in each '
        f'{scope_label}. Simulated prices in this table are drawn only from '
        'successful pairs with an explicit accepted settlement price.'
    )
    if excluded_successful_records:
        note += (
            f' Excluded {excluded_successful_records} successful record'
            f'{"s" if excluded_successful_records != 1 else ""} without a recoverable '
            'accepted price.'
        )
    note += (
        ' Price MAE uses only successful pairs with both a recoverable accepted '
        'price and an observed resale price.'
    )

    header_cells = ''.join(
        (
            '<th style="padding: 8px 10px; text-align: left;">'
            f'{html.escape(header)}</th>'
        )
        for header in key_headers
    )
    return (
        f'<section class="summary-card"><h2>{html.escape(title)}</h2>'
        f'<p style="margin: 0 0 12px 0; color: #52606d;">{html.escape(note)}</p>'
        '<div style="overflow-x: auto;">'
        '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
        '<thead>'
        '<tr style="background: #f4f7fb; border-bottom: 1px solid #d7e0ea;">'
        f'{header_cells}'
        '<th style="padding: 8px 10px; text-align: right;">Seller Count</th>'
        '<th style="padding: 8px 10px; text-align: right;">Successful</th>'
        '<th style="padding: 8px 10px; text-align: right;">Success Rate (%)</th>'
        '<th style="padding: 8px 10px; text-align: right;">Observed Count</th>'
        '<th style="padding: 8px 10px; text-align: right;">Observed Avg</th>'
        '<th style="padding: 8px 10px; text-align: right;">Observed Min-Max</th>'
        '<th style="padding: 8px 10px; text-align: right;">Simulated Count</th>'
        '<th style="padding: 8px 10px; text-align: right;">Simulated Avg</th>'
        '<th style="padding: 8px 10px; text-align: right;">Simulated Min-Max</th>'
        '<th style="padding: 8px 10px; text-align: right;">Simulated Median</th>'
        '<th style="padding: 8px 10px; text-align: right;">Price MAE</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
        + ''.join(rows) +
        '</tbody>'
        '</table>'
        '</div>'
        '</section>'
    )


def build_flat_type_breakdown_html(
    *,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> str:
    return _build_grouped_price_breakdown_html(
        title='Flat Type Breakdown',
        key_headers=('Flat Type',),
        scope_label='flat type',
        empty_state_label='flat-type aggregation',
        successful_records=successful_records,
        seller_profiles=seller_profiles,
        group_key_fn=_group_by_flat_type,
    )


def build_town_breakdown_html(
    *,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> str:
    return _build_grouped_price_breakdown_html(
        title='Town Breakdown',
        key_headers=('Town',),
        scope_label='town',
        empty_state_label='town aggregation',
        successful_records=successful_records,
        seller_profiles=seller_profiles,
        group_key_fn=_group_by_town,
    )


def build_town_flat_type_breakdown_html(
    *,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> str:
    return _build_grouped_price_breakdown_html(
        title='Town x Flat Type Breakdown',
        key_headers=('Town', 'Flat Type'),
        scope_label='town and flat type combination',
        empty_state_label='town and flat-type aggregation',
        successful_records=successful_records,
        seller_profiles=seller_profiles,
        group_key_fn=_group_by_town_and_flat_type,
    )


def _observed_price_points_from_entries(
    observed_entries: list[dict[str, object]],
) -> list[tuple[float, float]]:
    observed_dates = [
        entry['observed_date']
        for entry in observed_entries
        if hasattr(entry.get('observed_date'), 'toordinal')
    ]
    observed_timeline_start_date = min(observed_dates) if observed_dates else None
    observed_points: list[tuple[float, float]] = []
    for entry in observed_entries:
        timing = _observed_entry_timing(
            entry,
            baseline_date=observed_timeline_start_date,
        )
        if timing is None:
            continue
        observed_points.append((timing, float(entry['price'])))
    return observed_points


def _collect_observed_price_trend_points(
    *,
    seller_profiles: dict[str, dict[str, object]],
) -> list[tuple[float, float]]:
    return _observed_price_points_from_entries(
        _build_observed_price_entries(
            seller_profiles=seller_profiles,
        )
    )


def _simulated_price_points_from_entries(
    simulated_entries: list[dict[str, object]],
) -> list[tuple[float, float]]:
    return [
        (int(entry['timing']), float(entry['price']))
        for entry in simulated_entries
    ]


def _collect_simulated_price_trend_points(
    *,
    successful_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> list[tuple[float, float]]:
    return _simulated_price_points_from_entries(
        _build_simulated_close_entries(
            successful_records=successful_records,
            seller_profiles=seller_profiles,
        )
    )


def _bucket_price_trend_points(
    points: list[tuple[float, float]],
    *,
    bucket_weeks: int = PRICE_TREND_BUCKET_WEEKS,
) -> list[tuple[int, float]]:
    bucket_totals: dict[int, float] = {}
    bucket_counts: dict[int, int] = {}

    for timing, price in points:
        ordinal_week = max(int(round(timing)), 1)
        bucket_index = (ordinal_week - 1) // bucket_weeks
        bucket_totals[bucket_index] = bucket_totals.get(bucket_index, 0.0) + price
        bucket_counts[bucket_index] = bucket_counts.get(bucket_index, 0) + 1

    return [
        (bucket_index, bucket_totals[bucket_index] / bucket_counts[bucket_index])
        for bucket_index in sorted(bucket_totals)
        if bucket_counts[bucket_index] > 0
    ]


def _price_trend_bucket_label(
    bucket_index: int,
    *,
    bucket_weeks: int = PRICE_TREND_BUCKET_WEEKS,
) -> str:
    start_week = bucket_index * bucket_weeks + 1
    end_week = start_week + bucket_weeks - 1
    return f'Weeks {start_week}-{end_week}'


def _render_price_trend_chart_section_html(
    *,
    title: str,
    note: str,
    observed_points: list[tuple[float, float]],
    simulated_points: list[tuple[float, float]],
) -> str:
    """Render a single chart section from already-prepared price points."""
    observed_series = _bucket_price_trend_points(observed_points)
    simulated_series = _bucket_price_trend_points(simulated_points)

    if not observed_series and not simulated_series:
        return (
            f'<section class="summary-card"><h2>{html.escape(title)}</h2>'
            '<div class="empty-state">No observed or simulated prices were '
            'available to plot in 4-week ordinal buckets.</div></section>'
        )

    figure, axis = plt.subplots(figsize=(10, 4.8))
    all_bucket_indices = sorted({
        bucket_index for bucket_index, _ in observed_series + simulated_series
    })

    if observed_series:
        axis.plot(
            [bucket_index for bucket_index, _ in observed_series],
            [price for _, price in observed_series],
            color='#2563eb',
            marker='o',
            linewidth=2.4,
            label='Observed',
        )
    if simulated_series:
        axis.plot(
            [bucket_index for bucket_index, _ in simulated_series],
            [price for _, price in simulated_series],
            color='#dc2626',
            marker='o',
            linewidth=2.4,
            label='Simulated',
        )

    axis.set_title(title)
    axis.set_xlabel(
        f'Transaction timeline aggregated every {PRICE_TREND_BUCKET_WEEKS} weeks'
    )
    axis.set_ylabel('Average resale price (SGD)')
    axis.set_xticks(all_bucket_indices)
    axis.set_xticklabels([
        _price_trend_bucket_label(bucket_index)
        for bucket_index in all_bucket_indices
    ])
    axis.tick_params(axis='x', labelrotation=20)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f'${value:,.0f}')
    )
    axis.grid(True, axis='y', alpha=0.25)
    axis.legend()
    figure.tight_layout()

    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', dpi=160, bbox_inches='tight')
    plt.close(figure)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')

    return (
        f'<section class="summary-card"><h2>{html.escape(title)}</h2>'
        f'<p style="margin: 0 0 12px 0; color: #52606d;">{html.escape(note)}</p>'
        f'<img src="data:image/png;base64,{chart_base64}" '
        f'alt="{html.escape(title)}" '
        'style="width: 100%; max-width: 1040px; display: block; margin: 0 auto; '
        'border: 1px solid #d7e0ea; border-radius: 12px; background: #ffffff;" />'
        '</section>'
    )


def build_price_trend_chart_html(
    *,
    replay_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> str:
    """Build the overall observed vs simulated price trend card."""
    successful_records = _extract_successful_records(replay_records)
    observed_points = _collect_observed_price_trend_points(
        seller_profiles=seller_profiles,
    )
    simulated_points = _collect_simulated_price_trend_points(
        successful_records=successful_records,
        seller_profiles=seller_profiles,
    )
    return _render_price_trend_chart_section_html(
        title='Price Trend',
        note=(
            'Observed prices are drawn from the historical seller profiles '
            'included in this simulation sample. Simulated prices are drawn '
            'from successful negotiated closes using the explicit accepted '
            'settlement price when available. Both series are aggregated into '
            f'{PRICE_TREND_BUCKET_WEEKS}-week buckets.'
        ),
        observed_points=observed_points,
        simulated_points=simulated_points,
    )


def build_town_price_trend_html(
    *,
    replay_records: list[dict[str, Any]],
    seller_profiles: dict[str, dict[str, object]],
) -> str:
    """Build one price trend chart per town for the HTML summary."""
    successful_records = _extract_successful_records(replay_records)
    observed_entries = _build_observed_price_entries(
        seller_profiles=seller_profiles,
    )
    simulated_entries = _build_simulated_close_entries(
        successful_records=successful_records,
        seller_profiles=seller_profiles,
    )
    towns = sorted({
        _normalize_town(sample.get('town'))
        for sample in observed_entries + simulated_entries
    })
    if not towns:
        return (
            '<section class="summary-card"><h2>Price Trend by Town</h2>'
            '<div class="empty-state">No observed or simulated prices were '
            'available to plot by town.</div></section>'
        )

    blocks: list[str] = []
    for town in towns:
        observed_points = _observed_price_points_from_entries([
            sample
            for sample in observed_entries
            if _normalize_town(sample.get('town')) == town
        ])
        simulated_points = _simulated_price_points_from_entries([
            sample
            for sample in simulated_entries
            if _normalize_town(sample.get('town')) == town
        ])
        observed_series = _bucket_price_trend_points(observed_points)
        simulated_series = _bucket_price_trend_points(simulated_points)
        if not observed_series and not simulated_series:
            blocks.append(
                '<div style="margin-top: 18px;">'
                f'<h3 style="margin: 0 0 10px 0;">{html.escape(town)}</h3>'
                '<div class="empty-state">No observed or simulated prices were '
                'available to plot for this town.</div>'
                '</div>'
            )
            continue

        figure, axis = plt.subplots(figsize=(9, 4.4))
        all_bucket_indices = sorted({
            bucket_index
            for bucket_index, _ in observed_series + simulated_series
        })

        if observed_series:
            axis.plot(
                [bucket_index for bucket_index, _ in observed_series],
                [price for _, price in observed_series],
                color='#2563eb',
                marker='o',
                linewidth=2.2,
                label='Observed',
            )
        if simulated_series:
            axis.plot(
                [bucket_index for bucket_index, _ in simulated_series],
                [price for _, price in simulated_series],
                color='#dc2626',
                marker='o',
                linewidth=2.2,
                label='Simulated',
            )

        axis.set_title(town)
        axis.set_xlabel(
            f'Transaction timeline aggregated every {PRICE_TREND_BUCKET_WEEKS} weeks'
        )
        axis.set_ylabel('Average resale price (SGD)')
        axis.set_xticks(all_bucket_indices)
        axis.set_xticklabels([
            _price_trend_bucket_label(bucket_index)
            for bucket_index in all_bucket_indices
        ])
        axis.tick_params(axis='x', labelrotation=20)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f'${value:,.0f}')
        )
        axis.grid(True, axis='y', alpha=0.25)
        axis.legend()
        figure.tight_layout()

        buffer = io.BytesIO()
        figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        plt.close(figure)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode('ascii')
        blocks.append(
            '<div style="margin-top: 18px;">'
            f'<h3 style="margin: 0 0 10px 0;">{html.escape(town)}</h3>'
            f'<img src="data:image/png;base64,{chart_base64}" '
            f'alt="{html.escape(town)} price trend" '
            'style="width: 100%; max-width: 980px; display: block; margin: 0 auto; '
            'border: 1px solid #d7e0ea; border-radius: 12px; background: #ffffff;" />'
            '</div>'
        )

    return (
        '<section class="summary-card"><h2>Price Trend by Town</h2>'
        '<p style="margin: 0 0 12px 0; color: #52606d;">Each chart pools flats '
        'within one planning area, then compares observed and simulated prices '
        f'in {PRICE_TREND_BUCKET_WEEKS}-week buckets using the same rules as the '
        'overall chart.</p>'
        + ''.join(blocks) +
        '</section>'
    )


def extract_successful_records(
    replay_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Public wrapper used by ``main.py`` when building section inputs."""
    return _extract_successful_records(replay_records)
