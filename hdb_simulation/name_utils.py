"""Helpers for resolving participant display names.

This module intentionally stays conservative:
1. use explicit name fields when present
2. otherwise try to extract a leading personal name from persona text
3. otherwise ask the language model for a plausible display name
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re

from concordia.language_model import language_model


_NAME_TOKEN_RE = re.compile(r"^[A-Z][A-Za-z'.-]*$")


def _normalize_name_text(text: object) -> str:
    candidate = re.sub(r"\s+", " ", str(text or "").strip())
    if not candidate:
        return ""
    candidate = candidate.splitlines()[0].strip().strip("\"'`")
    if ":" in candidate:
        prefix, _, suffix = candidate.partition(":")
        if prefix.strip().casefold() == "name":
            candidate = suffix.strip().strip("\"'`")
    return candidate


def _extract_name_from_persona(persona: str) -> str:
    text = _normalize_name_text(persona)
    if not text:
        return ""

    name_tokens: list[str] = []
    for token in text.split():
        cleaned = token.strip(".,;:!?()[]{}\"'")
        if not cleaned or not _NAME_TOKEN_RE.match(cleaned):
            break
        name_tokens.append(cleaned)
        if len(name_tokens) >= 5:
            break

    if len(name_tokens) < 2:
        return ""
    return " ".join(name_tokens)


def _name_seed_text(
    record: Mapping[str, object],
    fallback_name: str,
) -> str:
    seed_parts = (
        record.get("name"),
        record.get("buyer_id"),
        record.get("seller_id"),
        record.get("flat_id"),
        record.get("linked_flat_id"),
        record.get("general_persona"),
        fallback_name,
    )
    normalized_parts = [
        str(value).strip() for value in seed_parts if str(value or "").strip()
    ]
    if normalized_parts:
        return "|".join(normalized_parts)
    return "participant"


def _stable_name_seed(record: Mapping[str, object], fallback_name: str) -> int:
    digest = hashlib.sha256(
        _name_seed_text(record, fallback_name).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _clean_generated_name(text: str) -> str:
    cleaned = _normalize_name_text(text)
    if not cleaned:
        return ""
    if ":" in cleaned:
        _, _, suffix = cleaned.partition(":")
        cleaned = suffix.strip()
    cleaned = cleaned.splitlines()[0].strip().strip(".,;:!?()[]{}\"'")
    return re.sub(r"\s+", " ", cleaned)


def _generate_model_name(
    record: Mapping[str, object],
    *,
    model: language_model.LanguageModel,
    fallback_name: str,
    role_label: str,
) -> str:
    persona = str(record.get("general_persona", "")).strip()
    occupation = str(record.get("occupation_category", "")).strip()
    age = str(record.get("age", "")).strip()
    prompt = (
        "Generate one plausible human display name for a Singapore HDB resale "
        "market simulation participant.\n"
        "If the persona already contains a plausible personal name, return that "
        "name exactly.\n"
        "If the persona does not contain a plausible personal name, invent a "
        "realistic personal name and return only the name.\n"
        "Do not include explanations, labels, punctuation-only decorations, or "
        "multiple options.\n\n"
        "Examples:\n"
        "Persona: Ivan blends a curiosity-driven pragmatism with quiet compassion.\n"
        "Name: Ivan\n\n"
        "Persona: A careful and community-minded teacher who prefers a calm home base near family.\n"
        "Name: Nur Aisyah Rahman\n\n"
        f"Role: {role_label}\n"
        f"Age: {age or 'unknown'}\n"
        f"Occupation: {occupation or 'unknown'}\n"
        f"Persona: {persona or 'No persona provided.'}\n"
        "Name:" # no need for structured outputs here; will not help in name generation, use few shot prompting to solve instead
    )
    response = model.sample_text(
        prompt,
        max_tokens=12,
        seed=_stable_name_seed(record, fallback_name),
    )
    return _clean_generated_name(response)


def resolve_profile_name(
    record: Mapping[str, object],
    *,
    fallback_name: str = "",
    model: language_model.LanguageModel | None = None,
    role_label: str = "participant",
) -> str:
    """Resolve a participant name from explicit fields, persona text, or model."""
    explicit_name = str(record.get("name") or "").strip()
    if explicit_name:
        return explicit_name

    persona_name = _extract_name_from_persona(
        str(record.get("general_persona", "")).strip()
    )
    if persona_name:
        return persona_name

    if model is not None:
        generated_name = _generate_model_name(
            record,
            model=model,
            fallback_name=fallback_name,
            role_label=role_label,
        )
        if generated_name:
            return generated_name

    return f"{role_label.title()} {_stable_name_seed(record, fallback_name) % 10000:04d}"


def disambiguate_profile_name(
    *,
    requested_name: str,
    record: Mapping[str, object],
    model: language_model.LanguageModel | None = None,
    role_label: str = "participant",
) -> str:
    """Prefer the fuller persona-extracted name when it improves uniqueness."""
    candidate = str(requested_name).strip()
    if not candidate:
        return candidate

    persona_name = _extract_name_from_persona(
        str(record.get("general_persona", "")).strip()
    )
    if not persona_name or persona_name == candidate:
        return candidate

    candidate_tokens = candidate.split()
    persona_tokens = persona_name.split()
    if len(persona_tokens) <= len(candidate_tokens):
        return candidate
    if len(candidate_tokens) == 1 and persona_tokens[0] == candidate_tokens[0]:
        return persona_name
    if candidate in persona_name:
        return persona_name
    if len(candidate_tokens) == 1:
        expanded_name = resolve_profile_name(
            record,
            fallback_name=candidate,
            model=model,
            role_label=role_label,
        )
        if expanded_name and expanded_name != candidate:
            return expanded_name
    return candidate
