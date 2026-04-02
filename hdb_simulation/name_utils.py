"""Helpers for resolving participant display names."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_NAME_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'.-]*$")
_NAME_PARTICLES = {
    'bin', 'binti', 'ibn', 'abu',
    'al', 'el',
    'da', 'de', 'del', 'della', 'der', 'di', 'dos', 'du',
    'la', 'le', 'van', 'von', 'st',
}
_PERSONA_FOLLOW_WORDS = {
    'a', 'an', 'the',
    'is', 'was', 'works', 'worked', 'lives', 'lived',
    'values', 'enjoys', 'enjoyed', 'keeps', 'kept',
    'collects', 'collected', 'writes', 'wrote',
    'hums', 'dreams', 'prefers', 'prefers', 'likes',
    'loves', 'carries', 'spends', 'obsesses', 'indulges',
    'balances', 'juggles', 'navigates', 'maintains',
    'supports', 'cares', 'seeks', 'prioritizes', 'prioritises',
    'hopes', 'wants', 'needs', 'plans', 'tries',
    'often', 'still', 'despite', 'while', 'whose', 'who',
    'with',
}
_PERSONA_NAME_PATTERNS = (
    re.compile(
        r"^(?P<name>[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4})\s*,"
    ),
    re.compile(
        r"^(?P<name>[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4})\s+"
        r"(?:is|was|works|worked|lives|lived|values|enjoys|keeps|collects|"
        r"writes|hums|dreams|prefers|likes|loves|carries|spends|obsesses|"
        r"indulges|balances|juggles|navigates|maintains|supports|cares|"
        r"seeks|prioritizes|prioritises|hopes|wants|needs|plans|tries)\b"
    ),
    re.compile(
        r"^(?P<name>[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4})\s+"
        r"(?:a|an|the)\b"
    ),
)


def _is_name_token(token: str) -> bool:
    if not _NAME_TOKEN_RE.match(token):
        return False
    return token[:1].isupper() or token.casefold() in _NAME_PARTICLES


def _extract_name_from_leading_tokens(text: str) -> str:
    words = text.split()
    if not words:
        return ""

    name_tokens: list[str] = []
    for token in words:
        cleaned = token.strip(".,;:!?()[]{}\"'")
        if not cleaned:
            break
        if not name_tokens and cleaned.casefold() in {'mr', 'mrs', 'ms', 'mdm', 'dr'}:
            name_tokens.append(cleaned.rstrip('.'))
            continue
        if _is_name_token(cleaned):
            name_tokens.append(cleaned)
            if len(name_tokens) >= 6:
                break
            continue
        break

    if not name_tokens:
        return ""

    if len(name_tokens) == 1 and name_tokens[0].casefold() not in {'mr', 'mrs', 'ms', 'mdm', 'dr'}:
        return ""

    consumed = len(name_tokens)
    next_word = ""
    if consumed < len(words):
        next_word = words[consumed].strip(".,;:!?()[]{}\"'").casefold()

    if next_word and next_word not in _PERSONA_FOLLOW_WORDS:
        return ""

    return " ".join(name_tokens).strip()


def _extract_name_from_persona(persona: str) -> str:
    text = re.sub(r"\s+", " ", str(persona).strip())
    if not text:
        return ""
    for pattern in _PERSONA_NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            candidate = match.group("name").strip()
            if candidate:
                return candidate
    return _extract_name_from_leading_tokens(text)


def resolve_profile_name(
    record: Mapping[str, Any],
    *,
    fallback_name: str,
) -> str:
    """Resolve a participant name from explicit fields or persona text."""
    explicit_name = str(record.get('name') or record.get('seller_name') or '').strip()
    if explicit_name:
        return explicit_name
    persona = str(record.get('general_persona', '')).strip()
    if persona:
        candidate = _extract_name_from_persona(persona)
        if candidate:
            return candidate
    return fallback_name
