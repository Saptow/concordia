"""Active-policy guidance context for HDB negotiation agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs import NegotiationComponentConfig
from configs import PolicyToolConfig
from configs import REPO_ROOT
from concordia.components.agent import action_spec_ignored
from concordia.hdb_simulation.models.schemas.policy.schema import FullPolicyPage
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyPage
from concordia.hdb_simulation.models.schemas.policy.schema import RetrievedFullPolicyPages
from concordia.typing import entity_component


TEXT_PAGE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
DEFAULT_CURRENT_POLICY_PROMPT = (
    "No simulation-specific policies are currently in effect."
)
POLICY_SUMMARY_MAX_TOKENS = 768


class HDBPolicyToolPrompt(action_spec_ignored.ActionSpecIgnored):
    """Retrieve relevant HDB resale policy context."""

    def __init__(
        self,
        *,
        model: Any,
        policy_jsonl_filenames: tuple[str, ...] = (
            PolicyToolConfig.DEFAULT_POLICY_JSONL_FILENAMES
        ),
        policy_directory: str | Path = PolicyToolConfig.DEFAULT_POLICY_DIRECTORY,
        max_directory_candidates: int = (
            PolicyToolConfig.DEFAULT_MAX_DIRECTORY_CANDIDATES
        ),
        max_page_chars: int = PolicyToolConfig.DEFAULT_MAX_PAGE_CHARS,
        max_prompt_chars: int = PolicyToolConfig.DEFAULT_MAX_PROMPT_CHARS,
        max_component_chars: int = PolicyToolConfig.DEFAULT_MAX_COMPONENT_CHARS,
        tool_call_retries: int = PolicyToolConfig.DEFAULT_TOOL_CALL_RETRIES,
        pre_act_label: str = "# POLICY SEARCH TOOL",
    ):
        super().__init__(pre_act_label=pre_act_label)
        self._model = model
        self._policy_directory = Path(policy_directory)
        self._policy_jsonl_filenames = tuple(
            str(filename).strip()
            for filename in policy_jsonl_filenames
            if str(filename).strip()
        )
        self._policy_index_paths = self._resolve_policy_index_paths()
        self._max_directory_candidates = max(1, int(max_directory_candidates))
        self._max_page_chars = max(1000, max_page_chars) if max_page_chars > 0 else 0
        self._max_prompt_chars = (
            max(2_000, max_prompt_chars) if max_prompt_chars > 0 else 0
        )
        self._max_component_chars = (
            max(1_500, max_component_chars)
            if max_component_chars > 0
            else 0
        )
        self._tool_call_retries = max(1, int(tool_call_retries))
        self._last_cache_key: str | None = None
        self._last_cache_value: str | None = None
        self._policy_pages_cache: list[PolicyPage] | None = None
        self._policy_pages_cache_signature: tuple[tuple[str, int], ...] | None = None
        self._current_policy_prompt = DEFAULT_CURRENT_POLICY_PROMPT
        self._synced_active_source_paths: list[str] | None = None

    @property
    def name(self) -> str:
        return NegotiationComponentConfig.POLICY_TOOL_COMPONENT_KEY

    def _resolve_policy_index_paths(self) -> tuple[Path, ...]:
        resolved_paths: list[Path] = []
        for filename in self._policy_jsonl_filenames:
            candidate = Path(filename)
            if candidate.is_absolute():
                resolved_paths.append(candidate)
            else:
                resolved_paths.append(self._policy_directory / candidate)
        return tuple(resolved_paths)

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        except ValueError:
            return str(path)

    @staticmethod
    def _decode_jsonl_line(raw_line: str) -> list[Any]:
        decoder = json.JSONDecoder()
        records: list[Any] = []
        text = raw_line.strip()
        index = 0

        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            record, index = decoder.raw_decode(text, idx=index)
            records.append(record)
        return records

    @staticmethod
    def _normalize_page_path(path: str) -> str:
        return str(path).strip().replace("\\", "/")

    def _load_policy_directory(self) -> list[PolicyPage]:
        signatures: list[tuple[str, int]] = []
        for policy_index_path in self._policy_index_paths:
            if not policy_index_path.exists():
                continue
            stat = policy_index_path.stat()
            signatures.append((str(policy_index_path), stat.st_mtime_ns))
        if not signatures:
            raise FileNotFoundError(
                "No policy summary JSONL files were found for policy retrieval."
            )
        signature = tuple(signatures)
        if (
            self._policy_pages_cache is not None
            and self._policy_pages_cache_signature == signature
        ):
            return self._policy_pages_cache

        pages_by_path: dict[str, PolicyPage] = {}
        for policy_index_path in self._policy_index_paths:
            if not policy_index_path.exists():
                continue
            with policy_index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    for record in self._decode_jsonl_line(stripped):
                        page = PolicyPage.model_validate(record)
                        pages_by_path[self._normalize_page_path(page.path)] = page

        self._policy_pages_cache = list(pages_by_path.values())
        self._policy_pages_cache_signature = signature
        return self._policy_pages_cache

    def _policy_index_signature(self) -> tuple[tuple[str, int], ...]:
        signatures: list[tuple[str, int]] = []
        for policy_index_path in self._policy_index_paths:
            if not policy_index_path.exists():
                continue
            stat = policy_index_path.stat()
            signatures.append(
                (
                    self._display_path(policy_index_path),
                    stat.st_mtime_ns,
                )
            )
        return tuple(signatures)

    def _filter_pages_to_active_sources(
        self,
        *,
        pages: list[PolicyPage],
        active_source_paths: list[str] | None,
    ) -> list[PolicyPage]:
        if active_source_paths is None:
            return pages
        allowed_paths = {
            self._normalize_page_path(path) for path in active_source_paths
        }
        return [
            page for page in pages
            if self._normalize_page_path(page.path) in allowed_paths
        ]

    def _resolve_local_path(self, record_path: str) -> Path | None:
        path = Path(record_path)
        if path.suffix.lower() not in TEXT_PAGE_SUFFIXES:
            return None
        resolved = path if path.is_absolute() else REPO_ROOT / path
        return resolved if resolved.exists() else None

    def _load_page_text(self, record_path: str) -> str:
        resolved = self._resolve_local_path(record_path)
        if resolved is None:
            return ""
        return resolved.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _maybe_truncate_text(text: str, *, max_chars: int) -> str:
        normalized = str(text or "").strip()
        if max_chars <= 0:
            return normalized
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _truncate_tail_text(text: str, *, max_chars: int) -> str:
        normalized = str(text or "").strip()
        if max_chars <= 0:
            return normalized
        if len(normalized) <= max_chars:
            return normalized
        if max_chars <= 10:
            return normalized[:max_chars]
        marker = "\n\n... [truncated] ..."
        available = max_chars - len(marker)
        if available <= 0:
            return normalized[:max_chars]
        return normalized[:available].rstrip() + marker

    def _fit_policy_summary_prompt(
        self,
        *,
        current_policy_prompt: str,
        full_page_result: str,
    ) -> str:
        prefix = (
            "# Role\n"
            "You are an HDB resale policy analyst preparing static policy guidance "
            "for a simulation state.\n\n"
            "# Task\n"
            "Summarize the grounded active policy source pages for negotiation use "
            "without restating the current policy state verbatim.\n\n"
            "# Inputs\n"
            "## Current Policy State\n"
        )
        between_sections = "\n\n## Active Policy Source Pages\n"
        suffix = (
            "\n\n# Instructions\n"
            "1. Summarize only policies that are currently active.\n"
            "2. Ground every policy summary in the supplied active policy source pages.\n"
            "3. Focus on concrete implications for negotiation behavior and constraints.\n"
            "4. Do not mention inactive or hypothetical policies.\n"
            "5. Do not repeat or paraphrase the full current policy state section; "
            "assume it is already shown separately and use the guidance area for "
            "added value only.\n"
            "6. Keep the guidance concise and implication-focused rather than "
            "relisting policy text.\n\n"
            "# Output Format\n"
            "Return markdown with exactly these sections:\n"
            "- `## Relevant Policy Summaries`\n"
            "- `## Overall Policy Guidance`\n"
        )
        prompt = (
            prefix
            + current_policy_prompt
            + between_sections
            + full_page_result
            + suffix
        )
        if self._max_prompt_chars <= 0 or len(prompt) <= self._max_prompt_chars:
            return prompt

        fixed_without_pages = prefix + current_policy_prompt + between_sections + suffix
        available_for_pages = self._max_prompt_chars - len(fixed_without_pages)
        if available_for_pages > 0:
            prompt = (
                fixed_without_pages.replace(
                    between_sections,
                    between_sections
                    + self._truncate_tail_text(
                        full_page_result,
                        max_chars=available_for_pages,
                    ),
                    1,
                )
            )
            if len(prompt) <= self._max_prompt_chars:
                return prompt

        fixed_without_state = prefix + between_sections + full_page_result + suffix
        available_for_state = self._max_prompt_chars - len(fixed_without_state)
        trimmed_state = self._truncate_tail_text(
            current_policy_prompt,
            max_chars=max(0, available_for_state),
        )
        prompt = prefix + trimmed_state + between_sections + full_page_result + suffix
        if len(prompt) <= self._max_prompt_chars:
            return prompt

        available_for_pages = self._max_prompt_chars - len(
            prefix + trimmed_state + between_sections + suffix
        )
        trimmed_pages = self._truncate_tail_text(
            full_page_result,
            max_chars=max(0, available_for_pages),
        )
        return prefix + trimmed_state + between_sections + trimmed_pages + suffix

    def _fit_pre_act_component_text(
        self,
        *,
        current_policy_prompt: str,
        policy_guidance: str,
    ) -> str:
        prefix = "## Current Policy State\n"
        between_sections = "\n\n## Relevant Policy Guidance\n"
        component_text = (
            prefix + current_policy_prompt + between_sections + policy_guidance
        )
        if (
            self._max_component_chars <= 0
            or len(component_text) <= self._max_component_chars
        ):
            return component_text

        fixed_without_guidance = prefix + current_policy_prompt + between_sections
        available_for_guidance = (
            self._max_component_chars - len(fixed_without_guidance)
        )
        if available_for_guidance > 0:
            trimmed_guidance = self._truncate_tail_text(
                policy_guidance,
                max_chars=available_for_guidance,
            )
            component_text = (
                prefix + current_policy_prompt + between_sections + trimmed_guidance
            )
            if len(component_text) <= self._max_component_chars:
                return component_text

        fixed_without_state = prefix + between_sections + policy_guidance
        available_for_state = self._max_component_chars - len(fixed_without_state)
        trimmed_state = self._truncate_tail_text(
            current_policy_prompt,
            max_chars=max(0, available_for_state),
        )
        component_text = prefix + trimmed_state + between_sections + policy_guidance
        if len(component_text) <= self._max_component_chars:
            return component_text

        available_for_guidance = self._max_component_chars - len(
            prefix + trimmed_state + between_sections
        )
        trimmed_guidance = self._truncate_tail_text(
            policy_guidance,
            max_chars=max(0, available_for_guidance),
        )
        return prefix + trimmed_state + between_sections + trimmed_guidance

    def _run_full_page_retrieval_tool(
        self,
        requested_paths: list[str],
        pages: list[PolicyPage],
    ) -> str:
        path_lookup = {page.path: page for page in pages}
        full_pages: list[FullPolicyPage] = []

        for path in requested_paths[: self._max_directory_candidates]:
            page = path_lookup.get(str(path))
            if page is None:
                continue
            full_text = self._load_page_text(page.path)
            if not full_text.strip():
                continue
            full_pages.append(
                FullPolicyPage(
                    path=page.path,
                    source=page.source,
                    summary=page.summary,
                    tags=page.tags,
                    content=self._maybe_truncate_text(
                        full_text,
                        max_chars=self._max_page_chars,
                    ),
                )
            )

        return RetrievedFullPolicyPages(policy_pages=full_pages).model_dump_json()

    @staticmethod
    def _no_relevant_policy_summary() -> str:
        return "No relevant HDB resale policy summary for the current context."

    def _cache_result(self, *, cache_key: str, result: str) -> str:
        self._last_cache_key = cache_key
        self._last_cache_value = result
        return result

    def set_active_policy_context(
        self,
        *,
        current_policy_prompt: str,
        active_source_paths: list[str] | tuple[str, ...] | None,
    ) -> None:
        normalized_prompt = str(current_policy_prompt or "").strip()
        if active_source_paths is None:
            normalized_active_source_paths = None
        else:
            normalized_paths: list[str] = []
            seen_paths: set[str] = set()
            for raw_path in active_source_paths:
                normalized = self._normalize_page_path(str(raw_path))
                if not normalized or normalized in seen_paths:
                    continue
                seen_paths.add(normalized)
                normalized_paths.append(normalized)
            normalized_active_source_paths = normalized_paths
        if (
            (normalized_prompt or DEFAULT_CURRENT_POLICY_PROMPT)
            == self._current_policy_prompt
            and normalized_active_source_paths == self._synced_active_source_paths
        ):
            return
        self._current_policy_prompt = (
            normalized_prompt or DEFAULT_CURRENT_POLICY_PROMPT
        )
        self._synced_active_source_paths = normalized_active_source_paths
        self._last_cache_key = None
        self._last_cache_value = None

    def _compose_pre_act_value(self, policy_guidance: str) -> str:
        return self._fit_pre_act_component_text(
            current_policy_prompt=self._current_policy_prompt,
            policy_guidance=self._dedupe_policy_guidance(
                policy_guidance or self._no_relevant_policy_summary()
            ),
        )

    @staticmethod
    def _dedupe_policy_guidance(policy_guidance: str) -> str:
        guidance = str(policy_guidance or "").strip()
        if not guidance:
            return ""

        current_state_marker = "## Current Policy State"
        summaries_marker = "## Relevant Policy Summaries"
        overall_marker = "## Overall Policy Guidance"

        current_state_index = guidance.find(current_state_marker)
        if current_state_index < 0:
            return guidance

        summaries_index = guidance.find(summaries_marker)
        overall_index = guidance.find(overall_marker)
        next_section_candidates = [
            index
            for index in (summaries_index, overall_index)
            if index > current_state_index
        ]
        if not next_section_candidates:
            return guidance

        next_section_index = min(next_section_candidates)
        trimmed = (
            guidance[:current_state_index].rstrip()
            + ("\n\n" if guidance[:current_state_index].strip() else "")
            + guidance[next_section_index:].lstrip()
        ).strip()
        return trimmed or guidance

    def _summarize_active_policy_sources(
        self,
        *,
        pages: list[PolicyPage],
        active_source_paths: list[str] | None,
    ) -> str:
        if active_source_paths == []:
            return self._no_relevant_policy_summary()
        if not pages:
            return self._no_relevant_policy_summary()
        chat = getattr(self._model, "chat", None)
        if not callable(chat):
            return self._no_relevant_policy_summary()

        full_page_result = self._run_full_page_retrieval_tool(
            active_source_paths or [page.path for page in pages],
            pages,
        )
        if not full_page_result.strip():
            return self._no_relevant_policy_summary()

        prompt = self._fit_policy_summary_prompt(
            current_policy_prompt=self._current_policy_prompt,
            full_page_result=full_page_result,
        )
        for _ in range(self._tool_call_retries):
            try:
                return chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=POLICY_SUMMARY_MAX_TOKENS,
                ).strip()
            except Exception:
                continue
        return self._no_relevant_policy_summary()

    def _make_pre_act_value(self) -> str:
        active_source_paths = self._synced_active_source_paths
        if active_source_paths == []:
            return self._compose_pre_act_value(self._no_relevant_policy_summary())
        cache_key = json.dumps(
            {
                "current_policy_prompt": self._current_policy_prompt,
                "policy_index_paths": [
                    self._display_path(path) for path in self._policy_index_paths
                ],
                "policy_index_signature": list(self._policy_index_signature()),
                "active_source_paths": active_source_paths,
                "policy_jsonl_filenames": list(self._policy_jsonl_filenames),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key == self._last_cache_key and self._last_cache_value is not None:
            return self._last_cache_value

        try:
            pages = self._filter_pages_to_active_sources(
                pages=self._load_policy_directory(),
                active_source_paths=active_source_paths,
            )
        except Exception:
            return self._cache_result(
                cache_key=cache_key,
                result=self._compose_pre_act_value(
                    self._no_relevant_policy_summary()
                ),
            )

        tool_result = self._summarize_active_policy_sources(
            pages=pages,
            active_source_paths=active_source_paths,
        )
        return self._cache_result(
            cache_key=cache_key,
            result=self._compose_pre_act_value(
                tool_result or self._no_relevant_policy_summary()
            ),
        )

    def update(self) -> None:
        super().update()

    def get_state(self) -> entity_component.ComponentState:
        return {
            "policy_jsonl_filenames": list(self._policy_jsonl_filenames),
            "policy_directory": str(self._policy_directory),
            "max_directory_candidates": self._max_directory_candidates,
            "max_page_chars": self._max_page_chars,
            "max_prompt_chars": self._max_prompt_chars,
            "max_component_chars": self._max_component_chars,
            "tool_call_retries": self._tool_call_retries,
            "current_policy_prompt": self._current_policy_prompt,
            "active_source_paths": (
                list(self._synced_active_source_paths)
                if self._synced_active_source_paths is not None
                else None
            ),
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        if "policy_jsonl_filenames" in state:
            self._policy_jsonl_filenames = tuple(
                str(filename).strip()
                for filename in state["policy_jsonl_filenames"]
                if str(filename).strip()
            )
        if "policy_directory" in state:
            self._policy_directory = Path(str(state["policy_directory"]))
        self._policy_index_paths = self._resolve_policy_index_paths()
        self._policy_pages_cache = None
        self._policy_pages_cache_signature = None
        if "max_directory_candidates" in state:
            self._max_directory_candidates = max(
                1,
                int(state["max_directory_candidates"]),
            )
        if "max_page_chars" in state:
            max_page_chars = int(state["max_page_chars"])
            self._max_page_chars = max(1_000, max_page_chars) if max_page_chars > 0 else 0
        if "max_prompt_chars" in state:
            max_prompt_chars = int(state["max_prompt_chars"])
            self._max_prompt_chars = (
                max(2_000, max_prompt_chars) if max_prompt_chars > 0 else 0
            )
        if "max_component_chars" in state:
            max_component_chars = int(state["max_component_chars"])
            self._max_component_chars = (
                max(1_500, max_component_chars)
                if max_component_chars > 0
                else 0
            )
        if "tool_call_retries" in state:
            self._tool_call_retries = max(1, int(state["tool_call_retries"]))
        if "current_policy_prompt" in state:
            current_policy_prompt = str(state["current_policy_prompt"]).strip()
            self._current_policy_prompt = (
                current_policy_prompt or DEFAULT_CURRENT_POLICY_PROMPT
            )
        if "active_source_paths" in state:
            raw_paths = state["active_source_paths"]
            if raw_paths is None:
                self._synced_active_source_paths = None
            else:
                normalized_paths: list[str] = []
                seen_paths: set[str] = set()
                for raw_path in raw_paths:
                    normalized = self._normalize_page_path(str(raw_path))
                    if not normalized or normalized in seen_paths:
                        continue
                    seen_paths.add(normalized)
                    normalized_paths.append(normalized)
                self._synced_active_source_paths = normalized_paths
        self._last_cache_key = None
        self._last_cache_value = None
