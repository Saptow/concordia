"""Helpers for resolving participant display names."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
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

_FALLBACK_GIVEN_NAMES = (
    'Aarav', 'Adeline', 'Aiden', 'Aisha', 'Amir', 'Aqil', 'Benjamin',
    'Brandon', 'Charlotte', 'Cheryl', 'Daniel', 'Darren', 'Dev', 'Ethan',
    'Farah', 'Felix', 'Grace', 'Hafiz', 'Hannah', 'Haziq', 'Isabel', 'Jia',
    'Jin', 'Joel', 'Kai', 'Keith', 'Marcus', 'Mei', 'Nadia', 'Natasha',
    'Noah', 'Nurul', 'Priya', 'Ravi', 'Ryan', 'Siti', 'Sofia', 'Syafiq',
    'Tania', 'Wei', 'Xavier', 'Ying', 'Yusuf', 'Zara',
)
_FALLBACK_MIDDLE_NAMES = (
    '', '', '', 'Ahmad', 'Aisyah', 'Anand', 'Bin', 'Binti', 'Chandra',
    'Hui', 'Jie', 'Jun', 'Kumar', 'Ling', 'Mei', 'Nair', 'Nur', 'Qistina',
    'Raj', 'Wei', 'Xin',
)
_FALLBACK_SURNAMES = (
    'Abdullah', 'Ahmad', 'Chong', 'Goh', 'Ibrahim', 'Kaur', 'Koh', 'Kumar',
    'Lim', 'Loh', 'Nair', 'Ng', 'Ong', 'Pillai', 'Rahman', 'Sharma', 'Tan',
    'Teo', 'Toh', 'Wong', 'Yap', 'Yeo',
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


def _normalize_generated_name(text: str) -> str:
    candidate = re.sub(r"\s+", " ", str(text).strip())
    if not candidate:
        return ""
    candidate = candidate.splitlines()[0].strip().strip("\"'`")
    if ":" in candidate:
        prefix, _, suffix = candidate.partition(":")
        if prefix.strip().casefold() in {"name", "generated name"}:
            candidate = suffix.strip().strip("\"'`")
    extracted = _extract_name_from_persona(candidate)
    if extracted:
        return extracted

    tokens = [
        token.strip(".,;:!?()[]{}\"'")
        for token in candidate.split()
    ]
    tokens = [token for token in tokens if token]
    if not tokens or len(tokens) > 6:
        return ""
    if all(_is_name_token(token) for token in tokens):
        return " ".join(tokens)
    return ""


def _generate_name_from_persona(
    *,
    model: Any,
    persona: str,
    role_label: str,
 ) -> str:
    sample_text = getattr(model, 'sample_text', None)
    if not callable(sample_text):
        return ""

    prompt = (
        "You are naming an HDB resale market participant in Singapore.\n"
        "If the persona already contains a plausible personal name, return "
        "that exact name.\n"
        "If the persona does not contain a plausible personal name, invent "
        "one that fits the persona and sounds realistic in Singapore.\n\n"
        "Return only the final name.\n\n"
        "Examples:\n"
        "Role: buyer\n"
        "Persona: Ivan blends a curiosity-driven pragmatism with quiet "
        "compassion, tunes A.R. Rahman's soundtracks while perfecting his "
        "masala dosa, and cannot resist adding a new cricket figurine to "
        "his shelf.\n"
        "Name: Ivan\n\n"
        "Role: seller\n"
        "Persona: Lay Fong Ng, 30, is a karaoke-obsessed, detail-driven "
        "event planner who double-checks seating charts at midnight.\n"
        "Name: Lay Fong Ng\n\n"
        "Role: buyer\n"
        "Persona: A careful and community-minded teacher who prefers a calm "
        "home base near family and values reliable transit.\n"
        "Name: Nur Aisyah Rahman\n\n"
        f"Role: {role_label}\n"
        f"Persona: {persona}\n"
        "Name:"
    )
    try:
        response = sample_text(prompt, max_tokens=20)
    except Exception:
        return ""

    normalized = _normalize_generated_name(response)
    return normalized


def _fallback_name_from_persona(
    *,
    persona: str,
    role_label: str,
) -> str:
    """Generate a stable Singapore-style fallback name from persona text."""
    seed_text = f"{role_label}\n{persona}".encode('utf-8', errors='ignore')
    digest = hashlib.sha256(seed_text).digest()
    given_name = _FALLBACK_GIVEN_NAMES[digest[0] % len(_FALLBACK_GIVEN_NAMES)]
    middle_name = _FALLBACK_MIDDLE_NAMES[digest[1] % len(_FALLBACK_MIDDLE_NAMES)]
    surname = _FALLBACK_SURNAMES[digest[2] % len(_FALLBACK_SURNAMES)]
    if middle_name in {'Bin', 'Binti'}:
        connector_name = _FALLBACK_GIVEN_NAMES[digest[3] % len(_FALLBACK_GIVEN_NAMES)]
        return f'{given_name} {middle_name} {connector_name}'
    if middle_name:
        return f'{given_name} {middle_name} {surname}'
    return f'{given_name} {surname}'


def _fallback_surname_from_persona(
    *,
    persona: str,
    role_label: str,
) -> str:
    """Generate a stable surname for duplicate-name disambiguation."""
    seed_text = f"surname\n{role_label}\n{persona}".encode(
        'utf-8',
        errors='ignore',
    )
    digest = hashlib.sha256(seed_text).digest()
    return _FALLBACK_SURNAMES[digest[0] % len(_FALLBACK_SURNAMES)]


def resolve_profile_name(
    record: Mapping[str, Any],
    *,
    fallback_name: str = "",
    model: Any | None = None,
    role_label: str = "participant",
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
        if model is not None:
            generated_name = _generate_name_from_persona(
                model=model,
                persona=persona,
                role_label=role_label,
            )
            if generated_name:
                return generated_name
        fallback_from_persona = _fallback_name_from_persona(
            persona=persona,
            role_label=role_label,
        )
        if fallback_from_persona:
            return fallback_from_persona
    return fallback_name


def disambiguate_profile_name(
    *,
    requested_name: str,
    record: Mapping[str, Any],
    role_label: str,
) -> str:
    """Prefer adding a surname over exposing role/id-style duplicate suffixes."""
    candidate = str(requested_name).strip()
    if not candidate:
        return candidate

    tokens = candidate.split()
    if len(tokens) >= 2:
        return candidate

    persona = str(record.get('general_persona', '')).strip()
    if not persona:
        return candidate

    surname = _fallback_surname_from_persona(
        persona=persona,
        role_label=role_label,
    )
    if not surname:
        return candidate
    if tokens[0] == surname:
        return candidate
    return f'{candidate} {surname}'
