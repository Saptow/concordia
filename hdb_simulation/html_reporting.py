"""HTML rendering helpers for simulation reports.

This module owns the final HTML-facing presentation layer so ``main.py`` can
stay focused on orchestration and artifact wiring. It covers:

1. The internal coherence evaluation summary card.
2. Assembly of the full simulation HTML page from pre-rendered sections.
"""

from __future__ import annotations

import html
from typing import Any

from concordia.concordia.utils import structured_logging

INTERNAL_COHERENCE_METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ('decision_rationale_context_coherence', 'Decision rationale vs context'),
    ('verbal_explanation_context_coherence', 'Verbal explanation vs context'),
    ('action_type_verbal_explanation_coherence', 'Action type vs verbal explanation'),
    ('action_type_decision_rationale_coherence', 'Action type vs decision rationale'),
    (
        'decision_rationale_verbal_explanation_coherence',
        'Decision rationale vs verbal explanation',
    ),
    ('revelation_coherence', 'Hidden-information revelation'),
    ('repetition_rate', 'Repetition rate'),
)


def _format_percentage(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        value = 0.0
    return f'{value * 100.0:.{digits}f}%'


def _format_metric_cell(
    metric_value: dict[str, Any] | None,
    *,
    value_key: str,
    digits: int = 1,
) -> str:
    """Render one metric cell while preserving explicit error states."""
    if not isinstance(metric_value, dict):
        return 'N/A'
    if metric_value.get('status') == 'error':
        return 'ERR'
    raw_value = metric_value.get(value_key)
    if not isinstance(raw_value, (int, float)):
        return 'N/A'
    return _format_percentage(float(raw_value), digits=digits)


def build_internal_coherence_summary_html(
    *,
    evaluation_result: dict[str, Any] | None,
    error_message: str | None = None,
) -> str:
    """Render the post-hoc internal coherence evaluation as one summary card."""
    if error_message:
        return (
            '<section class="summary-card"><h2>Internal Coherence Evaluation</h2>'
            '<div class="empty-state">'
            'Evaluation could not be completed after the run. '
            f'{html.escape(error_message)}'
            '</div></section>'
        )

    if not isinstance(evaluation_result, dict):
        return (
            '<section class="summary-card"><h2>Internal Coherence Evaluation</h2>'
            '<div class="empty-state">No evaluation result was available.</div>'
            '</section>'
        )

    overall_metrics = evaluation_result.get('overall_metrics', {})
    per_agent_metrics = evaluation_result.get('per_agent_metrics', {})
    per_turn_metrics = evaluation_result.get('per_turn_metrics', ())

    overall_rows: list[str] = []
    for metric_name, label in INTERNAL_COHERENCE_METRIC_LABELS:
        metric_value = overall_metrics.get(metric_name, {})
        if not isinstance(metric_value, dict):
            continue
        micro_display = _format_metric_cell(
            metric_value,
            value_key='micro_average',
        )
        macro_display = _format_metric_cell(
            metric_value,
            value_key='macro_average',
        )
        overall_rows.append(
            '<tr>'
            f'<td style="padding: 8px 10px; text-align: left;">{html.escape(label)}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{micro_display}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{macro_display}</td>'
            '</tr>'
        )

    if not overall_rows:
        overall_rows.append(
            '<tr><td colspan="3" style="padding: 10px;">No aggregate metrics were '
            'produced.</td></tr>'
        )

    metric_errors = evaluation_result.get('metric_errors', {})
    error_note = ''
    if isinstance(metric_errors, dict) and metric_errors:
        error_items = ''.join(
            '<li>'
            f'<strong>{html.escape(label)}</strong>: '
            f'{html.escape(str(metric_errors.get(metric_name, "")))}'
            '</li>'
            for metric_name, label in INTERNAL_COHERENCE_METRIC_LABELS
            if metric_name in metric_errors
        )
        if error_items:
            error_note = (
                '<div style="margin: 0 0 12px 0; padding: 10px 12px; '
                'background: #fff7ed; border: 1px solid #fed7aa; '
                'border-radius: 8px; color: #9a3412;">'
                '<strong>Metric evaluation errors</strong>'
                '<ul style="margin: 8px 0 0 18px; padding: 0;">'
                f'{error_items}'
                '</ul>'
                '</div>'
            )

    sorted_agents = sorted(
        (
            value
            for value in per_agent_metrics.values()
            if isinstance(value, dict)
        ),
        key=lambda item: (
            str(item.get('actor_role', '')),
            str(item.get('actor_name', '')),
            str(item.get('actor_id', '')),
        ),
    )
    agent_rows: list[str] = []
    for agent_metrics in sorted_agents:
        repetition_summary = agent_metrics.get('repetition_rate', {})
        turn_count = 0
        if isinstance(repetition_summary, dict):
            repetition_total_turns = repetition_summary.get('total_turns')
            if isinstance(repetition_total_turns, int):
                turn_count = repetition_total_turns
        if not turn_count:
            metric_turns = agent_metrics.get(
                'decision_rationale_context_coherence',
                {},
            )
            if isinstance(metric_turns, dict):
                turn_count = int(metric_turns.get('num_turns', 0) or 0)
        agent_rows.append(
            '<tr>'
            f'<td style="padding: 8px 10px; text-align: left;">{html.escape(str(agent_metrics.get("actor_name", "")))}</td>'
            f'<td style="padding: 8px 10px; text-align: left;">{html.escape(str(agent_metrics.get("actor_role", "")))}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{turn_count}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("decision_rationale_context_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("verbal_explanation_context_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("action_type_verbal_explanation_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("action_type_decision_rationale_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("decision_rationale_verbal_explanation_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_metric_cell(agent_metrics.get("revelation_coherence"), value_key="mean_verdict")}</td>'
            f'<td style="padding: 8px 10px; text-align: right;">{_format_percentage(float(repetition_summary.get("rate", 0.0) or 0.0)) if isinstance(repetition_summary, dict) else "0.0%"}</td>'
            '</tr>'
        )

    if not agent_rows:
        agent_rows.append(
            '<tr><td colspan="10" style="padding: 10px;">No per-agent metrics were '
            'produced.</td></tr>'
        )

    turn_count = len(per_turn_metrics) if isinstance(per_turn_metrics, list) else 0
    return (
        '<section class="summary-card"><h2>Internal Coherence Evaluation</h2>'
        '<p style="margin: 0 0 12px 0; color: #52606d;">'
        'This post-hoc evaluation uses the stored negotiation turn records to '
        'score internal coherence across action choice, rationale, public '
        f'explanation, hidden-information leakage, and repetition. Evaluated turns: {turn_count}.'
        '</p>'
        + error_note +
        '<div style="overflow-x: auto; margin-bottom: 18px;">'
        '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
        '<thead><tr style="background: #f4f7fb; border-bottom: 1px solid #d7e0ea;">'
        '<th style="padding: 8px 10px; text-align: left;">Metric</th>'
        '<th style="padding: 8px 10px; text-align: right;">Micro Avg</th>'
        '<th style="padding: 8px 10px; text-align: right;">Macro Avg</th>'
        '</tr></thead><tbody>'
        + ''.join(overall_rows) +
        '</tbody></table></div>'
        '<div style="overflow-x: auto;">'
        '<table style="width: 100%; border-collapse: collapse; font-size: 14px;">'
        '<thead><tr style="background: #f4f7fb; border-bottom: 1px solid #d7e0ea;">'
        '<th style="padding: 8px 10px; text-align: left;">Agent</th>'
        '<th style="padding: 8px 10px; text-align: left;">Role</th>'
        '<th style="padding: 8px 10px; text-align: right;">Turns</th>'
        '<th style="padding: 8px 10px; text-align: right;">Rationale/Context</th>'
        '<th style="padding: 8px 10px; text-align: right;">Verbal/Context</th>'
        '<th style="padding: 8px 10px; text-align: right;">Action/Verbal</th>'
        '<th style="padding: 8px 10px; text-align: right;">Action/Rationale</th>'
        '<th style="padding: 8px 10px; text-align: right;">Rationale/Verbal</th>'
        '<th style="padding: 8px 10px; text-align: right;">No Leakage</th>'
        '<th style="padding: 8px 10px; text-align: right;">Repetition</th>'
        '</tr></thead><tbody>'
        + ''.join(agent_rows) +
        '</tbody></table></div>'
        '</section>'
    )


def render_simulation_html(
    *,
    simulation_log: structured_logging.SimulationLog,
    player_scores: dict[str, Any],
    summary_sections_html: list[str],
    title: str = 'Simulation Log',
) -> str:
    """Build the final HTML page from the structured log and section fragments."""
    entity_memories_for_html = {
        entity_name: simulation_log.get_entity_memories(entity_name)
        for entity_name in simulation_log.get_entity_names()
        if simulation_log.get_entity_memories(entity_name)
    }
    return structured_logging.render_dynamic_html(
        simulation_log=simulation_log,
        entity_memories=entity_memories_for_html or None,
        game_master_memories=simulation_log.get_game_master_memories() or None,
        player_scores=player_scores,
        summary_sections_html=summary_sections_html,
        title=title,
    )
