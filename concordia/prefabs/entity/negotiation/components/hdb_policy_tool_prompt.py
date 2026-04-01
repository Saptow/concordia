"""vLLM-compatible HDB policy retrieval context for negotiation agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs import NegotiationComponentConfig
from configs import PolicyToolConfig
from configs import REPO_ROOT
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.hdb_simulation.models.schemas.policy.schema import FullPolicyPage
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyPage
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyPageDirectory
from concordia.hdb_simulation.models.schemas.policy.schema import RelevantPolicyPathSelection
from concordia.hdb_simulation.models.schemas.policy.schema import RetrievedFullPolicyPages
from concordia.typing import entity_component


TEXT_PAGE_SUFFIXES = frozenset({".md", ".markdown", ".txt"})


class HDBPolicyToolPrompt(action_spec_ignored.ActionSpecIgnored):
    """Retrieve relevant HDB resale policy context."""

    def __init__(
        self,
        *,
        model: Any,
        observation_component_key: str = (
            NegotiationComponentConfig.OBSERVATION_COMPONENT_KEY
        ),
        memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
        num_memories_to_retrieve: int = 6,
        policy_jsonl_filename: str = PolicyToolConfig.DEFAULT_POLICY_JSONL_FILENAME,
        policy_directory: str | Path = PolicyToolConfig.DEFAULT_POLICY_DIRECTORY,
        max_directory_candidates: int = (
            PolicyToolConfig.DEFAULT_MAX_DIRECTORY_CANDIDATES
        ),
        max_page_chars: int = PolicyToolConfig.DEFAULT_MAX_PAGE_CHARS,
        tool_call_retries: int = PolicyToolConfig.DEFAULT_TOOL_CALL_RETRIES,
        pre_act_label: str = "# POLICY SEARCH TOOL",
    ):
        super().__init__(pre_act_label=pre_act_label)
        self._model = model
        self._observation_component_key = observation_component_key
        self._memory_component_key = memory_component_key
        self._num_memories_to_retrieve = max(1, int(num_memories_to_retrieve))
        self._policy_directory = Path(policy_directory)
        self._policy_jsonl_filename = str(policy_jsonl_filename)
        self._policy_index_path = self._resolve_policy_index_path()
        self._max_directory_candidates = max(1, int(max_directory_candidates))
        self._max_page_chars = max(1000, max_page_chars) if max_page_chars > 0 else 0
        self._tool_call_retries = max(1, int(tool_call_retries))
        self._last_cache_key: str | None = None
        self._last_cache_value: str | None = None
        self._policy_pages_cache: list[PolicyPage] | None = None
        self._policy_pages_cache_path: Path | None = None
        self._policy_pages_cache_mtime_ns: int | None = None
        self._tool_schemas = self._build_vllm_tools()
        self._tool_schema_by_name = {
            tool["function"]["name"]: tool for tool in self._tool_schemas
        }

    @property
    def name(self) -> str:
        return NegotiationComponentConfig.POLICY_TOOL_COMPONENT_KEY

    def _resolve_policy_index_path(self) -> Path:
        candidate = Path(self._policy_jsonl_filename)
        if candidate.is_absolute():
            return candidate
        return self._policy_directory / candidate

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

    def _load_policy_directory(self) -> list[PolicyPage]:
        policy_index_path = self._policy_index_path
        stat = policy_index_path.stat()
        if (
            self._policy_pages_cache is not None
            and self._policy_pages_cache_path == policy_index_path
            and self._policy_pages_cache_mtime_ns == stat.st_mtime_ns
        ):
            return self._policy_pages_cache

        pages: list[PolicyPage] = []
        with policy_index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                for record in self._decode_jsonl_line(stripped):
                    pages.append(PolicyPage.model_validate(record))

        self._policy_pages_cache = pages
        self._policy_pages_cache_path = policy_index_path
        self._policy_pages_cache_mtime_ns = stat.st_mtime_ns
        return pages

    # Policy records point to extracted markdown pages using a few historical
    # path conventions, so normalize those variants before reading content.
    def _resolve_local_path(self, record_path: str) -> Path | None:
        path = Path(record_path)
        candidate_paths: list[Path] = []

        if "/raw_html/" in record_path:
            candidate_paths.append(
                Path(record_path.replace("/raw_html/", "/page/")).with_suffix(".extracted.md")
            )
        elif "/raw_pdf/" in record_path:
            candidate_paths.append(
                Path(record_path.replace("/raw_pdf/", "/page/")).with_suffix(".extracted.md")
            )
        elif "/page/" in record_path:
            if path.suffix.lower() in TEXT_PAGE_SUFFIXES:
                candidate_paths.append(path)
            else:
                candidate_paths.append(path.with_suffix(".extracted.md"))
        elif path.suffix.lower() in TEXT_PAGE_SUFFIXES:
            candidate_paths.append(path)
        else:
            candidate_paths.append(path.with_suffix(".extracted.md"))

        for path in candidate_paths:
            resolved = path if path.is_absolute() else REPO_ROOT / path
            if resolved.exists():
                return resolved
        return None

    def _load_page_text(self, record_path: str) -> str:
        resolved = self._resolve_local_path(record_path)
        if resolved is None:
            return ""

        if resolved.suffix.lower() in TEXT_PAGE_SUFFIXES:
            return resolved.read_text(encoding="utf-8", errors="ignore")

        return ""

    def _current_observation(self) -> str:
        try:
            observation_value = self.get_named_component_pre_act_value(
                self._observation_component_key
            ).strip()
        except Exception:
            return "None"
        return observation_value or "None"

    def _recent_memories(self) -> list[str]:
        try:
            memory = self.get_entity().get_component(
                self._memory_component_key,
                type_=memory_component.Memory,
            )
            raw_memories = list(
                memory.retrieve_recent(limit=self._num_memories_to_retrieve)
            )
        except Exception:
            return []

        cleaned: list[str] = []
        for memory_text in raw_memories:
            text = str(memory_text).strip()
            if text:
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _maybe_truncate_text(text: str, *, max_chars: int) -> str:
        normalized = str(text or "").strip()
        if max_chars <= 0:
            return normalized
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _build_vllm_tools(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "type": "function",
                "function": {
                    "name": PolicyToolConfig.DIRECTORY_SCREENING_TOOL_NAME,
                    "description": (
                        "Read the configured HDB resale policy directory JSONL and return "
                        "PolicyPageDirectory-style entries that can be screened for relevance."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "policy_jsonl_path": {
                                "type": "string",
                                "description": "Path to the HDB resale policy JSONL directory file.",
                            }
                        },
                        "required": ["policy_jsonl_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": PolicyToolConfig.FULL_PAGE_RETRIEVAL_TOOL_NAME,
                    "description": (
                        "Read exact HDB resale policy pages for the given directory paths and "
                        "return RetrievedFullPolicyPages-style records with grounded full text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Exact policy page paths selected from the policy directory.",
                            }
                        },
                        "required": ["paths"],
                    },
                },
            },
        )

    @staticmethod
    def _parse_tool_calls(response: str) -> list[dict[str, Any]] | None:
        try:
            payload = json.loads(str(response or "").strip())
        except json.JSONDecodeError:
            return None

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return None

        parsed_calls: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                return None
            name = item.get("name") or item.get("tool_name")
            arguments = item.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return None
            parsed_calls.append({"name": name, "arguments": arguments})
        return parsed_calls or None

    def _run_directory_screening_tool(self, pages: list[PolicyPage]) -> str:
        directory = PolicyPageDirectory(
            policy_pages=[
                PolicyPage(
                    path=page.path,
                    source=page.source,
                    summary=str(page.summary or "").strip(),
                    tags=page.tags,
                )
                for page in pages
            ]
        )
        return directory.model_dump_json()

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

    def _execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        pages: list[PolicyPage],
    ) -> str:
        if tool_name == PolicyToolConfig.DIRECTORY_SCREENING_TOOL_NAME:
            return self._run_directory_screening_tool(pages)
        if tool_name == PolicyToolConfig.FULL_PAGE_RETRIEVAL_TOOL_NAME:
            raw_paths = arguments.get("paths", [])
            if not isinstance(raw_paths, list):
                raw_paths = []
            requested_paths = [str(path) for path in raw_paths if str(path).strip()]
            return self._run_full_page_retrieval_tool(requested_paths, pages)
        return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

    def _run_required_tool_call(
        self,
        *,
        prompt: str,
        tool_name: str,
        pages: list[PolicyPage],
    ) -> tuple[str, str]:
        chat = getattr(self._model, "chat", None)
        if not callable(chat):
            return "", ""

        tool_schema = self._tool_schema_by_name[tool_name]
        for _ in range(self._tool_call_retries):
            try:
                response = chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=2_000,
                    tools=[tool_schema],
                )
                tool_calls = self._parse_tool_calls(response)
                if not tool_calls or len(tool_calls) != 1:
                    continue

                tool_call = tool_calls[0]
                if tool_call["name"] != tool_name:
                    continue

                tool_result = self._execute_tool_call(
                    tool_call["name"],
                    tool_call["arguments"],
                    pages,
                )
                return response, tool_result
            except Exception:
                continue

        return "", ""

    @staticmethod
    def _empty_relevant_policy_path_selection(
        retrieval_decision: str = "No relevant HDB resale policies were identified.",
    ) -> RelevantPolicyPathSelection:
        return RelevantPolicyPathSelection(
            relevant_paths=[],
            retrieval_decision=retrieval_decision,
        )

    def _normalize_relevant_paths(
        self,
        relevant_paths: list[str],
    ) -> list[str]:
        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for path in relevant_paths:
            normalized = str(path).strip()
            if normalized and normalized not in seen_paths:
                seen_paths.add(normalized)
                normalized_paths.append(normalized)
        return normalized_paths[: self._max_directory_candidates]

    # The workflow has three stages:
    # 1. Ask the model to call the directory screening tool.
    # 2. Ask the model to choose exact relevant paths from that grounded output.
    # 3. Retrieve full pages for only those paths and summarize them.
    def _screen_relevant_policies_from_directory(
        self,
        *,
        observation: str,
        recent_memories: list[str],
        pages: list[PolicyPage],
    ) -> tuple[RelevantPolicyPathSelection, str, str]:
        directory_prompt = (
            "# Role\n"
            "You are an HDB resale policy directory screener.\n\n"
            "# Task\n"
            "Screen the HDB resale policy directory and identify which exact policy paths are relevant to the current negotiation context.\n\n"
            "# Inputs\n"
            "## Observation\n"
            f"{observation}\n\n"
            "## Retrieved Memories\n"
            f"{json.dumps(recent_memories, ensure_ascii=False)}\n\n"
            "# Private Chain-of-Thought Process\n"
            "Think step by step privately. Do not reveal your reasoning.\n\n"
            "1. Decide whether the observation or recent memories raise HDB resale policy questions.\n"
            "2. Use the directory screening tool to inspect the policy directory.\n"
            "3. After seeing the tool result, identify only the exact relevant policy paths.\n"
            "4. Return no invented paths.\n"
        )
        directory_tool_call, directory_tool_result = self._run_required_tool_call(
            prompt=directory_prompt,
            tool_name=PolicyToolConfig.DIRECTORY_SCREENING_TOOL_NAME,
            pages=pages,
        )
        if not directory_tool_call or not directory_tool_result:
            return (
                self._empty_relevant_policy_path_selection(
                    "The policy directory screening tool did not return a usable result."
                ),
                "",
                "",
            )

        path_selection = self._empty_relevant_policy_path_selection(
            "Unable to parse relevant policy paths from the directory result."
        )
        for _ in range(self._tool_call_retries):
            try:
                selection_response = self._model.sample_text(
                    (
                        "# Role\n"
                        "You are an HDB resale policy relevance analyst.\n\n"
                        "# Task\n"
                        "Read the screened HDB resale policy directory result and identify only the exact policy paths that are relevant to the current negotiation context.\n\n"
                        "# Inputs\n"
                        "## Observation\n"
                        f"{observation}\n\n"
                        "## Retrieved Memories\n"
                        f"{json.dumps(recent_memories, ensure_ascii=False)}\n\n"
                        "## Directory Screening Tool Call\n"
                        f"{directory_tool_call}\n\n"
                        "## Directory Screening Tool Result\n"
                        f"{directory_tool_result}\n\n"
                        "# Private Chain-of-Thought Process\n"
                        "Think step by step privately. Do not reveal your reasoning.\n\n"
                        "1. Review the screened directory entries.\n"
                        "2. Keep only entries that are materially relevant to the observation or recent memories.\n"
                        "3. Return only exact `path` values that appear in the screened directory result.\n"
                        "4. If nothing is relevant, return an empty list.\n"
                    ),
                    json_schema=RelevantPolicyPathSelection.model_json_schema(),
                )
                path_selection = RelevantPolicyPathSelection.model_validate_json(
                    selection_response
                )
                break
            except Exception:
                continue

        path_selection.relevant_paths = self._normalize_relevant_paths(
            path_selection.relevant_paths
        )
        return path_selection, directory_tool_call, directory_tool_result

    def _summarize_relevant_policies_from_full_pages(
        self,
        *,
        observation: str,
        recent_memories: list[str],
        pages: list[PolicyPage],
        path_selection: RelevantPolicyPathSelection,
        directory_tool_call: str,
        directory_tool_result: str,
    ) -> str:
        chat = getattr(self._model, "chat", None)
        if not callable(chat):
            return ""

        relevant_paths = path_selection.relevant_paths
        if not relevant_paths:
            return ""

        full_page_tool_call = ""
        full_page_tool_result = ""
        full_page_prompt = (
            "# Role\n"
            "You are an HDB resale policy page retriever.\n\n"
            "# Task\n"
            "Retrieve the full pages for the relevant HDB resale policy entries.\n\n"
            "# Inputs\n"
            "## Observation\n"
            f"{observation}\n\n"
            "## Retrieved Memories\n"
            f"{json.dumps(recent_memories, ensure_ascii=False)}\n\n"
            "## Directory Screening Tool Call\n"
            f"{directory_tool_call}\n\n"
            "## Directory Screening Tool Result\n"
            f"{directory_tool_result}\n\n"
            "## Relevant Directory Paths\n"
            f"{json.dumps(relevant_paths, ensure_ascii=False)}\n\n"
            "# Private Chain-of-Thought Process\n"
            "Think step by step privately. Do not reveal your reasoning.\n\n"
            "1. Review the relevant directory paths.\n"
            "2. Use the full-page retrieval tool with those exact paths.\n"
            "3. Return only the tool call.\n"
        )
        full_page_tool_call, full_page_tool_result = self._run_required_tool_call(
            prompt=full_page_prompt,
            tool_name=PolicyToolConfig.FULL_PAGE_RETRIEVAL_TOOL_NAME,
            pages=pages,
        )
        if not full_page_tool_call or not full_page_tool_result:
            return ""

        final_prompt = (
            "# Role\n"
            "You are an HDB resale policy analyst helping a negotiation agent.\n\n"
            "# Task\n"
            "Produce the final markdown policy context grounded in the tool results.\n\n"
            "# Inputs\n"
            "## Observation\n"
            f"{observation}\n\n"
            "## Retrieved Memories\n"
            f"{json.dumps(recent_memories, ensure_ascii=False)}\n\n"
            "## Directory Screening Tool Call\n"
            f"{directory_tool_call}\n\n"
            "## Directory Screening Tool Result\n"
            f"{directory_tool_result}\n\n"
            "## Directory Relevance Decision\n"
            f"{path_selection.retrieval_decision}\n\n"
            "## Relevant Directory Paths\n"
            f"{json.dumps(relevant_paths, ensure_ascii=False)}\n\n"
            "## Full Page Retrieval Tool Call\n"
            f"{full_page_tool_call or 'Not used'}\n\n"
            "## Full Page Retrieval Tool Result\n"
            f"{full_page_tool_result or 'No relevant full pages were retrieved.'}\n\n"
            "# Private Chain-of-Thought Process\n"
            "Think step by step privately. Do not reveal your reasoning.\n\n"
            "1. Decide whether the directory screening found relevant policy entries.\n"
            "2. If full pages were retrieved, summarize only grounded policy details from those pages.\n"
            "3. If no relevant policies were found, say so explicitly.\n"
            "4. Tie the policy implications back to the observation and recent memories.\n\n"
            "# Output Format\n"
            "Return markdown with exactly these sections:\n"
            "- `## Tool Contract`\n"
            "- `## Retrieval Decision`\n"
            "- `## Relevant Policy Summaries`\n"
            "- `## Overall Policy Guidance`\n"
        )
        for _ in range(self._tool_call_retries):
            try:
                return (
                    chat(
                        [{"role": "user", "content": final_prompt}],
                        max_tokens=6000,
                    ).strip()
                )
            except Exception:
                continue
        return ""

    def _run_required_hdb_policy_tool_workflow(
        self,
        *,
        observation: str,
        recent_memories: list[str],
        pages: list[PolicyPage],
    ) -> str:
        chat = getattr(self._model, "chat", None)
        if not callable(chat):
            return ""

        path_selection, directory_tool_call, directory_tool_result = (
            self._screen_relevant_policies_from_directory(
                observation=observation,
                recent_memories=recent_memories,
                pages=pages,
            )
        )
        if not directory_tool_call or not directory_tool_result:
            return ""

        if not path_selection.relevant_paths:
            return ""

        return self._summarize_relevant_policies_from_full_pages(
            observation=observation,
            recent_memories=recent_memories,
            pages=pages,
            path_selection=path_selection,
            directory_tool_call=directory_tool_call,
            directory_tool_result=directory_tool_result,
        )

    def _cache_result(self, *, cache_key: str, result: str) -> str:
        self._last_cache_key = cache_key
        self._last_cache_value = result
        return result

    def _make_pre_act_value(self) -> str:
        observation = self._current_observation()
        recent_memories = self._recent_memories()
        cache_key = json.dumps(
            {
                "observation": observation,
                "recent_memories": recent_memories,
                "policy_index_path": self._display_path(self._policy_index_path),
                "policy_jsonl_filename": self._policy_jsonl_filename,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key == self._last_cache_key and self._last_cache_value is not None:
            return self._last_cache_value

        try:
            pages = self._load_policy_directory()
        except Exception:
            return self._cache_result(cache_key=cache_key, result="")

        tool_result = self._run_required_hdb_policy_tool_workflow(
            observation=observation,
            recent_memories=recent_memories,
            pages=pages,
        )
        return self._cache_result(cache_key=cache_key, result=tool_result or "")

    def update(self) -> None:
        super().update()
        self._last_cache_key = None
        self._last_cache_value = None

    def get_state(self) -> entity_component.ComponentState:
        return {
            "observation_component_key": self._observation_component_key,
            "memory_component_key": self._memory_component_key,
            "num_memories_to_retrieve": self._num_memories_to_retrieve,
            "policy_jsonl_filename": self._policy_jsonl_filename,
            "policy_directory": str(self._policy_directory),
            "max_directory_candidates": self._max_directory_candidates,
            "max_page_chars": self._max_page_chars,
            "tool_call_retries": self._tool_call_retries,
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        if "observation_component_key" in state:
            self._observation_component_key = str(state["observation_component_key"])
        if "memory_component_key" in state:
            self._memory_component_key = str(state["memory_component_key"])
        if "num_memories_to_retrieve" in state:
            self._num_memories_to_retrieve = max(
                1,
                int(state["num_memories_to_retrieve"]),
            )
        if "policy_jsonl_filename" in state:
            self._policy_jsonl_filename = str(state["policy_jsonl_filename"])
        if "policy_directory" in state:
            self._policy_directory = Path(str(state["policy_directory"]))
        self._policy_index_path = self._resolve_policy_index_path()
        self._policy_pages_cache = None
        self._policy_pages_cache_path = None
        self._policy_pages_cache_mtime_ns = None
        if "max_directory_candidates" in state:
            self._max_directory_candidates = max(
                1,
                int(state["max_directory_candidates"]),
            )
        if "max_page_chars" in state:
            max_page_chars = int(state["max_page_chars"])
            self._max_page_chars = max(1_000, max_page_chars) if max_page_chars > 0 else 0
        if "tool_call_retries" in state:
            self._tool_call_retries = max(1, int(state["tool_call_retries"]))
        self._last_cache_key = None
        self._last_cache_value = None
