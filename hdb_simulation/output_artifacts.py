"""Helpers for replay and evaluation artifact generation.

This module keeps file-writing and replay/evaluation extraction logic out of
``main.py`` so the entrypoint can stay focused on simulation setup.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from absl import logging

from concordia.concordia.prefabs.game_master.negotiation.components import (
    hdb_negotiation_evaluation,
)
from concordia.concordia.utils import structured_logging

logger = logging


def extract_week_summary_from_log_entry(
    simulation_log: structured_logging.SimulationLog,
    entry: structured_logging.StructuredLogEntry,
) -> dict[str, Any] | None:
    """Pull one weekly coordinator snapshot out of a structured log entry."""
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


def collect_completed_week_summaries(
    simulation_log: structured_logging.SimulationLog,
) -> list[dict[str, Any]]:
    """Collect normalized per-week summaries from the structured simulation log."""
    summaries_by_week: dict[int, dict[str, Any]] = {}

    for entry in simulation_log.entries:
        week_summary = extract_week_summary_from_log_entry(simulation_log, entry)
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


def write_internal_coherence_jsonl(
    *,
    output_dir: str,
    timestamp: str,
    evaluation_result: dict[str, Any] | None,
    error_message: str | None = None,
) -> str:
    """Persist the internal coherence evaluation as a dedicated JSONL artifact."""
    evaluation_path = os.path.join(
        output_dir,
        f'internal_coherence_evaluation_{timestamp}.jsonl',
    )
    with open(evaluation_path, 'w', encoding='utf-8') as handle:
        def _write_jsonl_record(record: dict[str, Any]) -> None:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=str) + '\n'
            )

        metadata_record = {
            'record_type': 'metadata',
            'evaluation_type': 'internal_coherence',
            'status': 'error' if error_message else 'ok',
            'generated_at': datetime.now().isoformat(),
        }
        if error_message:
            metadata_record['error_message'] = error_message
        _write_jsonl_record(metadata_record)
        if error_message or not isinstance(evaluation_result, dict):
            return evaluation_path

        _write_jsonl_record({
            'record_type': 'overall_metrics',
            'payload': evaluation_result.get('overall_metrics', {}),
        })

        per_agent_metrics = evaluation_result.get('per_agent_metrics', {})
        if isinstance(per_agent_metrics, dict):
            for actor_id in sorted(per_agent_metrics):
                _write_jsonl_record({
                    'record_type': 'per_agent_metrics',
                    'actor_id': actor_id,
                    'payload': per_agent_metrics[actor_id],
                })

        per_turn_metrics = evaluation_result.get('per_turn_metrics', ())
        if isinstance(per_turn_metrics, list):
            for turn_metrics in per_turn_metrics:
                _write_jsonl_record({
                    'record_type': 'per_turn_metrics',
                    'payload': turn_metrics,
                })

    return evaluation_path


def compute_internal_coherence_outputs(
    *,
    model,
    week_summaries: list[dict[str, Any]],
    output_dir: str,
    timestamp: str,
    closed_pair_archive_jsonl_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None, str]:
    """Collect, evaluate, and persist internal coherence artifacts."""
    raw_evaluation_records = hdb_negotiation_evaluation.collect_evaluation_records(
        week_summaries=week_summaries,
        archive_jsonl_path=closed_pair_archive_jsonl_path,
    )
    evaluation_result: dict[str, Any] | None = None
    error_message: str | None = None
    try:
        evaluation_result = hdb_negotiation_evaluation.evaluate_internal_coherence(
            model=model,
            week_summaries=week_summaries,
            archive_jsonl_path=closed_pair_archive_jsonl_path,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        error_message = str(error)
        logger.exception('Failed to compute internal coherence evaluation.')

    evaluation_jsonl_path = write_internal_coherence_jsonl(
        output_dir=output_dir,
        timestamp=timestamp,
        evaluation_result=evaluation_result,
        error_message=error_message,
    )
    return (
        raw_evaluation_records,
        evaluation_result,
        error_message,
        evaluation_jsonl_path,
    )


def summarize_internal_coherence_by_pair(
    evaluation_result: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-turn internal coherence verdicts to per-pair summaries."""
    if not isinstance(evaluation_result, dict):
        return {}

    per_pair_scores: dict[str, dict[str, list[float]]] = {}
    for turn_metrics in evaluation_result.get('per_turn_metrics', ()):
        if not isinstance(turn_metrics, dict):
            continue
        pair_key = str(turn_metrics.get('pair_key', '')).strip()
        if not pair_key:
            continue
        pair_bucket = per_pair_scores.setdefault(pair_key, {})
        for metric_name, metric_value in turn_metrics.items():
            if metric_name in {
                'turn_id',
                'pair_key',
                'week_number',
                'pair_round_number',
                'actor_id',
                'actor_name',
                'actor_role',
            }:
                continue
            if metric_name == 'repetition_rate':
                if isinstance(metric_value, dict):
                    pair_bucket.setdefault(metric_name, []).append(
                        float(metric_value.get('is_repeated', 0) or 0)
                    )
                continue
            if isinstance(metric_value, dict) and 'global_verdict' in metric_value:
                pair_bucket.setdefault(metric_name, []).append(
                    float(metric_value.get('global_verdict', 0) or 0)
                )

    per_pair_summaries: dict[str, dict[str, Any]] = {}
    for pair_key, metric_scores in per_pair_scores.items():
        metric_means = {
            metric_name: (sum(scores) / len(scores))
            for metric_name, scores in metric_scores.items()
            if scores and metric_name != 'repetition_rate'
        }
        repetition_scores = metric_scores.get('repetition_rate', [])
        per_pair_summaries[pair_key] = {
            'turn_count': max(
                [len(scores) for scores in metric_scores.values()] or [0]
            ),
            'metric_means': metric_means,
            'repetition_rate': (
                (sum(repetition_scores) / len(repetition_scores))
                if repetition_scores
                else 0.0
            ),
        }
    return per_pair_summaries


def attach_evaluation_outputs_to_replay_records(
    *,
    replay_records: list[dict[str, Any]],
    raw_evaluation_records: list[dict[str, Any]],
    evaluation_result: dict[str, Any] | None,
) -> None:
    """Attach raw and aggregate evaluation outputs to replay records in-place."""
    evaluation_records_by_pair_key: dict[str, list[dict[str, Any]]] = {}
    for record in raw_evaluation_records:
        pair_key = str(record.get('pair_key', '')).strip()
        if not pair_key:
            continue
        evaluation_records_by_pair_key.setdefault(pair_key, []).append(record)

    internal_coherence_by_pair = summarize_internal_coherence_by_pair(
        evaluation_result
    )
    for replay_record in replay_records:
        pair_key = str(replay_record.get('pair_key', '')).strip()
        if not pair_key:
            continue
        if pair_key in evaluation_records_by_pair_key:
            replay_record['evaluation_records'] = [
                dict(record)
                for record in evaluation_records_by_pair_key[pair_key]
            ]
        if pair_key in internal_coherence_by_pair:
            replay_record['internal_coherence_evaluation'] = dict(
                internal_coherence_by_pair[pair_key]
            )


def collect_conversation_replay_records(
    sim,
    *,
    seller_profiles: dict[str, dict[str, object]] | None = None,
    archive_jsonl_path: str | None = None,
) -> list[dict[str, Any]]:
    """Collect replay records from the negotiation module and closed-pair archive."""
    seller_profiles = seller_profiles or {}
    records_by_pair_key: dict[str, dict[str, Any]] = {}
    records_without_pair_key: list[dict[str, Any]] = []

    def _enrich_replay_record(record: dict[str, Any]) -> dict[str, Any]:
        enriched_record = dict(record)
        seller_id = str(enriched_record.get('seller_id', '')).strip()
        seller_profile = seller_profiles.get(seller_id, {})
        if isinstance(seller_profile, dict):
            town = str(seller_profile.get('town', '') or '').strip()
            flat_type = str(seller_profile.get('flat_type', '') or '').strip()
            if town and not str(enriched_record.get('town', '') or '').strip():
                enriched_record['town'] = town
            if flat_type and not str(
                enriched_record.get('flat_type', '') or ''
            ).strip():
                enriched_record['flat_type'] = flat_type
        return enriched_record

    def _append_replay_record(record: dict[str, Any]) -> None:
        enriched_record = _enrich_replay_record(record)
        pair_key = str(enriched_record.get('pair_key', '')).strip()
        if pair_key:
            records_by_pair_key[pair_key] = enriched_record
        else:
            records_without_pair_key.append(enriched_record)

    for gm in sim.get_game_masters():
        if not hasattr(gm, 'get_component'):
            continue
        try:
            negotiation_module = gm.get_component('negotiation_module')
        except Exception:
            continue
        getter = getattr(negotiation_module, 'get_conversation_replay_records', None)
        if not callable(getter):
            continue
        replay_records = getter()
        if not isinstance(replay_records, list):
            continue
        for record in replay_records:
            if isinstance(record, dict):
                _append_replay_record(record)
        if records_by_pair_key or records_without_pair_key:
            break

    if archive_jsonl_path and os.path.exists(archive_jsonl_path):
        with open(archive_jsonl_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        'Skipping malformed closed-pair archive line from %s.',
                        archive_jsonl_path,
                    )
                    continue
                if isinstance(parsed, dict):
                    pair_key = str(parsed.get('pair_key', '')).strip()
                    if pair_key and pair_key in records_by_pair_key:
                        continue
                    _append_replay_record(parsed)

    return list(records_by_pair_key.values()) + records_without_pair_key
