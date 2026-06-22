#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTML rendering for structured simulation logs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import html
import json
from typing import Any

from concordia.utils import structured_logging


def _escape(value: object) -> str:
  return html.escape(str(value), quote=True)


def _json_block(value: Any) -> str:
  rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
  return f'<pre class="json-block">{html.escape(rendered)}</pre>'


def _metrics_block(metrics: Mapping[str, Any]) -> str:
  rows = []
  for label, value in metrics.items():
    rows.append(
        '<div class="metric-row">'
        f'<div class="metric-label">{_escape(label)}</div>'
        f'<div class="metric-value">{_escape(value)}</div>'
        '</div>'
    )
  return '<div class="metric-grid">' + ''.join(rows) + '</div>'


def _details(summary: str, body: str, *, open_by_default: bool = False,
             css_class: str = '') -> str:
  open_attr = ' open' if open_by_default else ''
  class_attr = f' class="{css_class}"' if css_class else ''
  return (
      f'<details{class_attr}{open_attr}>'
      f'<summary>{summary}</summary>'
      f'{body}'
      '</details>'
  )


def _render_collection(
    title: str,
    items: Sequence[Mapping[str, Any]],
    summary_builder,
    *,
    empty_message: str,
    open_by_default: bool = True,
) -> str:
  if not items:
    return (
        f'<section class="collection-card"><h4>{_escape(title)}</h4>'
        f'<div class="empty-state">{_escape(empty_message)}</div></section>'
    )

  parts = [
      '<section class="collection-card">',
      f'<h4>{_escape(title)} ({len(items)})</h4>',
  ]
  for item in items:
    parts.append(
        _details(
            _escape(summary_builder(item)),
            _json_block(item),
            open_by_default=False,
            css_class='item-dropdown',
        )
    )
  parts.append('</section>')
  return ''.join(parts)


def _participant_summary(participant: Mapping[str, Any]) -> str:
  name = participant.get('name') or participant.get('id') or 'Participant'
  participant_id = participant.get('id')
  suffix = f' ({participant_id})' if participant_id else ''
  return f'{name}{suffix}'


def _seller_summary(seller: Mapping[str, Any]) -> str:
  name = seller.get('name') or seller.get('id') or 'Seller'
  seller_id = seller.get('id')
  listing_price = seller.get('current_listing_price')
  suffix = f' ({seller_id})' if seller_id else ''
  if listing_price is None:
    return f'{name}{suffix}'
  return f'{name}{suffix} | listed at {listing_price}'


def _pair_summary(pair_state: Mapping[str, Any]) -> str:
  buyer_name = pair_state.get('buyer_name') or pair_state.get('buyer_id') or 'Buyer'
  seller_name = (
      pair_state.get('seller_name') or pair_state.get('seller_id') or 'Seller'
  )
  round_number = pair_state.get('pair_round_number', '?')
  outcome = pair_state.get('outcome', 'OPEN')
  return (
      f'{buyer_name} <-> {seller_name} | {outcome} | '
      f'Pair Negotiation Week {round_number}'
  )


def _matched_pair_summary(match: Mapping[str, Any]) -> str:
  buyer_name = match.get('buyer_name') or match.get('buyer_id') or 'Buyer'
  seller_name = match.get('seller_name') or match.get('seller_id') or 'Seller'
  match_id = match.get('match_id')
  suffix = f' | {match_id}' if match_id else ''
  return f'{buyer_name} <-> {seller_name}{suffix}'


def _extract_payload(
    simulation_log: structured_logging.SimulationLog,
    entry: structured_logging.StructuredLogEntry,
) -> dict[str, Any]:
  reconstructed = simulation_log.reconstruct_value(dict(entry.deduplicated_data))
  return dict(reconstructed) if isinstance(reconstructed, Mapping) else {}


def _extract_week_summary(payload: Mapping[str, Any]) -> dict[str, Any] | None:
  value = payload.get('value')
  if not isinstance(value, Mapping):
    return None
  week_summary = value.get('week_summary')
  if not isinstance(week_summary, Mapping):
    return None
  return dict(week_summary)


def _build_week_views(
    simulation_log: structured_logging.SimulationLog,
) -> list[dict[str, Any]]:
  step_entries: dict[int, list[structured_logging.StructuredLogEntry]] = defaultdict(list)
  for entry in simulation_log.entries:
    step_entries[entry.step].append(entry)

  views: list[dict[str, Any]] = []
  for step in sorted(step_entries):
    entries = step_entries[step]
    week_summary: dict[str, Any] | None = None
    additional_logs: list[dict[str, Any]] = []
    for entry in entries:
      payload = _extract_payload(simulation_log, entry)
      summary_candidate = _extract_week_summary(payload)
      if summary_candidate is not None:
        week_summary = summary_candidate
        continue
      if payload:
        additional_logs.append({
            'entity_name': entry.entity_name,
            'component_name': entry.component_name,
            'entry_type': entry.entry_type,
            'payload': payload,
        })

    views.append({
        'step': step,
        'week_number': (
            int(week_summary.get('week_number', step))
            if isinstance(week_summary, Mapping)
            else step
        ),
        'summary': entries[0].summary if entries else '',
        'week_summary': week_summary or {},
        'additional_logs': additional_logs,
    })
  return views


def _render_memory_sections(
    entity_memories: Mapping[str, Sequence[str]],
    gm_memories: Sequence[str],
) -> str:
  sections: list[str] = []

  if entity_memories:
    parts = ['<section class="memory-section"><h2>Participant Memories</h2>']
    for entity_name, memories in sorted(entity_memories.items()):
      memory_body = ''.join(
          f'<div class="memory-row">{_escape(memory)}</div>' for memory in memories
      )
      parts.append(
          _details(
              _escape(f'{entity_name} ({len(memories)})'),
              memory_body or '<div class="empty-state">No memories.</div>',
              open_by_default=False,
              css_class='memory-dropdown',
          )
      )
    parts.append('</section>')
    sections.append(''.join(parts))

  if gm_memories:
    gm_body = ''.join(
        f'<div class="memory-row">{_escape(memory)}</div>' for memory in gm_memories
    )
    sections.append(
        '<section class="memory-section"><h2>Game Master Memories</h2>'
        + _details(
            _escape(f'Game Master ({len(gm_memories)})'),
            gm_body,
            open_by_default=False,
            css_class='memory-dropdown',
        )
        + '</section>'
    )

  return ''.join(sections)


def _render_week_view(week_view: Mapping[str, Any]) -> str:
  week_summary = dict(week_view.get('week_summary', {}))
  listing = dict(week_summary.get('listing', {}))
  negotiation = dict(week_summary.get('negotiation', {}))

  buyers = [
      item for item in listing.get('buyer_states', ())
      if isinstance(item, Mapping)
  ]
  listed_sellers = [
      item for item in listing.get('listed_sellers', ())
      if isinstance(item, Mapping)
  ]
  released_seller_ids = list(listing.get('released_seller_ids', ()))
  inactive_seller_ids = list(listing.get('inactive_seller_ids', ()))
  matched_pairs = [
      item for item in listing.get('matched_pairs', ())
      if isinstance(item, Mapping)
  ]
  sellers_reviewed = list(listing.get('sellers_reviewed', ()))
  reviewed_seller_count = len(sellers_reviewed)
  matched_seller_count = len(matched_pairs)
  sellers_without_match_count = int(
      listing.get(
          'sellers_without_match_count',
          max(0, reviewed_seller_count - matched_seller_count),
      )
  )
  seller_match_rate = (
      (matched_seller_count / reviewed_seller_count) * 100.0
      if reviewed_seller_count > 0
      else 0.0
  )
  seller_no_match_rate = (
      (sellers_without_match_count / reviewed_seller_count) * 100.0
      if reviewed_seller_count > 0
      else 0.0
  )
  pair_states = [
      item for item in negotiation.get('pair_states', ())
      if isinstance(item, Mapping)
  ]
  closed_pairs = [
      item for item in negotiation.get('closed_pairs', ())
      if isinstance(item, Mapping)
  ]

  additional_logs = [
      item for item in week_view.get('additional_logs', ())
      if isinstance(item, Mapping)
  ]
  additional_logs_html = ''
  if additional_logs:
    additional_logs_html = _render_collection(
        'Additional Logs',
        additional_logs,
        lambda item: (
            f'{item.get("entity_name", "Unknown")} '
            f'[{item.get("component_name", "component")}]'
        ),
        empty_message='No extra logs recorded.',
        open_by_default=False,
    )

  listing_html = ''.join([
      '<section class="module-card">',
      '<h3>Listing</h3>',
      (
          '<div class="module-meta">'
          f'Buyer states: {len(buyers)} | '
          f'Currently listed sellers: {len(listed_sellers)} | '
          f'Released this week: {len(released_seller_ids)} | '
          f'Still inactive: {len(inactive_seller_ids)} | '
          f'Matches this week: {len(matched_pairs)} | '
          f'Seller match rate this week: {seller_match_rate:.1f}% | '
          f'Seller no-match rate this week: {seller_no_match_rate:.1f}%'
          '</div>'
      ),
      _render_collection(
          'Buyer States',
          buyers,
          _participant_summary,
          empty_message='No active buyers in listing this week.',
      ),
      _render_collection(
          'Currently Listed Sellers',
          listed_sellers,
          _seller_summary,
          empty_message='No sellers are currently listed this week.',
      ),
      _render_collection(
          'Matched Pairs',
          matched_pairs,
          _matched_pair_summary,
          empty_message='No new listing matches this week.',
          open_by_default=False,
      ),
      '</section>',
  ])

  negotiation_html = ''.join([
      '<section class="module-card">',
      '<h3>Negotiation</h3>',
      (
          '<div class="module-meta">'
          f'Pair states: {len(pair_states)} | '
          f'Pairs negotiated this week: '
          f'{_escape(negotiation.get("number_of_pairs_negotiated", 0))} | '
          f'Closed this week: {len(closed_pairs)}'
          '</div>'
      ),
      _render_collection(
          'Negotiation Pair States',
          pair_states,
          _pair_summary,
          empty_message='No negotiation pairs tracked this week.',
      ),
      _render_collection(
          'Closed Pairs This Week',
          closed_pairs,
          _pair_summary,
          empty_message='No pairs closed this week.',
          open_by_default=False,
      ),
      '</section>',
  ])

  assignments = dict(week_summary.get('assignments', {}))
  listing_assignments = assignments.get('listing', ())
  negotiation_assignments = assignments.get('negotiation', ())
  overview = (
      '<div class="week-overview">'
      f'Listing assignments: {_escape(len(listing_assignments))} | '
      f'Negotiation assignments: {_escape(len(negotiation_assignments))} | '
      f'Pending matches next week: '
      f'{_escape(len(week_summary.get("pending_matches_for_next_week", ())))}'
      '</div>'
  )

  return ''.join([
      '<section class="week-card">',
      '<div class="week-header">',
      f'<h2>Week {_escape(week_view.get("week_number", week_view.get("step", "?")))}</h2>',
      (
          f'<div class="week-step">Step {_escape(week_view.get("step", "?"))}'
          '</div>'
      ),
      '</div>',
      (
          f'<div class="week-subtitle">{_escape(week_view.get("summary", ""))}'
          '</div>'
          if week_view.get('summary')
          else ''
      ),
      overview,
      '<div class="module-grid">',
      listing_html,
      negotiation_html,
      '</div>',
      additional_logs_html,
      '</section>',
  ])


def render_dynamic_html(
    simulation_log: structured_logging.SimulationLog,
    entity_memories: dict[str, list[str]] | None = None,
    game_master_memories: list[str] | None = None,
    player_scores: dict[str, Any] | None = None,
    summary_sections_html: list[str] | None = None,
    title: str = 'Simulation Log',
) -> str:
  """Render the log into a weekly, single-page HTML view."""
  week_views = _build_week_views(simulation_log)
  entity_memories_data = entity_memories or {}
  gm_memories_data = game_master_memories or []

  html_parts = [
      '<!DOCTYPE html><html><head><meta charset="UTF-8">',
      f'<title>{html.escape(title)}</title>',
      """
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #eef3f8; color: #17212b; }
.container { max-width: 1440px; margin: 0 auto; padding: 24px; }
.page-header { margin-bottom: 24px; }
.page-header h1 { margin: 0 0 8px 0; font-size: 30px; }
.page-header p { margin: 0; color: #52606d; }
.summary-card, .week-card, .memory-section { background: #ffffff; border: 1px solid #d7e0ea; border-radius: 14px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }
.summary-card { padding: 16px 18px; margin-bottom: 20px; }
.week-card { padding: 18px; margin-bottom: 18px; }
.metric-grid { display: grid; gap: 10px; }
.metric-row { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 10px 12px; border: 1px solid #e5ecf3; border-radius: 10px; background: #f8fbfd; }
.metric-label { font-weight: 600; color: #233445; }
.metric-value { color: #102a43; text-align: right; font-variant-numeric: tabular-nums; }
.week-header { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
.week-header h2 { margin: 0; font-size: 24px; }
.week-step { color: #5b6b79; font-size: 14px; }
.week-subtitle { margin-top: 8px; color: #405261; }
.week-overview { margin-top: 12px; padding: 10px 12px; background: #f4f8fc; border-radius: 10px; color: #314353; font-size: 14px; }
.module-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; margin-top: 16px; }
.module-card { border: 1px solid #d7e0ea; border-radius: 12px; padding: 14px; background: #fcfdff; }
.module-card h3 { margin: 0 0 8px 0; font-size: 20px; }
.module-meta { margin-bottom: 12px; color: #516170; font-size: 14px; }
.collection-card { margin-top: 12px; }
.collection-card h4 { margin: 0 0 8px 0; font-size: 15px; color: #233445; }
details { border: 1px solid #d7e0ea; border-radius: 10px; background: #ffffff; }
details + details { margin-top: 8px; }
summary { cursor: pointer; list-style: none; padding: 10px 12px; font-weight: 600; }
summary::-webkit-details-marker { display: none; }
.item-dropdown[open] summary, .memory-dropdown[open] summary { border-bottom: 1px solid #e5ecf3; }
.json-block { margin: 0; padding: 12px; background: #0f172a; color: #e2e8f0; border-radius: 0 0 10px 10px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.45; }
.empty-state { padding: 12px; border: 1px dashed #c5d0db; border-radius: 10px; color: #6a7a88; background: #f8fbfd; }
.memory-section { margin-top: 20px; padding: 18px; }
.memory-section h2 { margin: 0 0 12px 0; font-size: 20px; }
.memory-row { padding: 10px 12px; border-top: 1px solid #e8eef5; background: #fff; }
</style>
</head><body><div class="container">
""",
      '<header class="page-header">',
      f'<h1>{html.escape(title)}</h1>',
      '<p>Weekly listing and negotiation state in one page. Participant-heavy sections use dropdowns instead of tabs.</p>',
      '</header>',
  ]

  if player_scores:
    html_parts.append(
        '<section class="summary-card"><h2>Simulation Summary</h2>'
        + _metrics_block(player_scores)
        + '</section>'
    )
  if summary_sections_html:
    html_parts.extend(section for section in summary_sections_html if section)

  if week_views:
    html_parts.extend(_render_week_view(week_view) for week_view in week_views)
  else:
    html_parts.append(
        '<section class="week-card"><div class="empty-state">No weekly log entries were recorded.</div></section>'
    )

  html_parts.append(
      _render_memory_sections(entity_memories_data, gm_memories_data)
  )
  html_parts.append('</div></body></html>')
  return ''.join(html_parts)
