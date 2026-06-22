"""Shared helpers for component-level parsing and prompt assembly."""

from __future__ import annotations

from typing import Any


def extract_first_json_object(text: str) -> str | None:
  """Return the first balanced JSON object embedded in ``text``.

  This helper tolerates surrounding prose and correctly ignores braces that
  appear inside JSON strings.
  """
  candidate = str(text or '').strip()
  start = candidate.find('{')
  if start < 0:
    return None

  candidate = candidate[start:]
  depth = 0
  in_string = False
  escaped = False
  for idx, ch in enumerate(candidate):
    if in_string:
      if escaped:
        escaped = False
      elif ch == '\\':
        escaped = True
      elif ch == '"':
        in_string = False
      continue

    if ch == '"':
      in_string = True
    elif ch == '{':
      depth += 1
    elif ch == '}':
      depth -= 1
      if depth == 0:
        return candidate[: idx + 1]
  return None


def truncate_text(text: str, *, max_chars: int) -> str:
  """Normalize text and truncate from the end with an ellipsis."""
  normalized = str(text or '').strip()
  if max_chars <= 0 or len(normalized) <= max_chars:
    return normalized
  if max_chars <= 3:
    return normalized[:max_chars]
  return normalized[: max_chars - 3].rstrip() + '...'


def truncate_tail_text(
    text: str,
    *,
    max_chars: int,
    marker: str = '\n\n... [truncated] ...',
) -> str:
  """Normalize text and truncate the tail while keeping a marker."""
  normalized = str(text or '').strip()
  if max_chars <= 0 or len(normalized) <= max_chars:
    return normalized
  if max_chars <= 10:
    return normalized[:max_chars]
  available = max_chars - len(marker)
  if available <= 0:
    return normalized[:max_chars]
  return normalized[:available].rstrip() + marker


def coerce_positive_float_or_none(value: Any) -> float | None:
  """Return a positive finite float, or ``None`` when coercion fails."""
  if value is None or isinstance(value, bool):
    return None
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return None
  if parsed <= 0.0:
    return None
  return parsed
