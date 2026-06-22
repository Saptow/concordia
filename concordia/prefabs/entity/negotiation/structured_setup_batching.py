"""Batch helpers for structured setup prompts during negotiation handoff."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from configs import LanguageModelConfig


@dataclasses.dataclass(frozen=True)
class StructuredSetupRequest:
    """One structured setup prompt that can be grouped into a batch."""

    component: Any
    response_key: str
    prompt_text: str
    specific_schema: Any
    max_tokens: int = LanguageModelConfig.DEFAULT_MAX_TOKENS
    terminators: tuple[str, ...] = LanguageModelConfig.DEFAULT_TERMINATORS
    temperature: float = LanguageModelConfig.DEFAULT_TEMPERATURE
    top_p: float = LanguageModelConfig.DEFAULT_TOP_P
    top_k: int = LanguageModelConfig.DEFAULT_TOP_K


def execute_setup_request(request: StructuredSetupRequest) -> str:
    """Execute one structured setup request through the underlying model."""
    return request.component._model.sample_text(
        prompt=request.prompt_text,
        max_tokens=request.max_tokens,
        terminators=request.terminators,
        json_schema=request.specific_schema.model_json_schema(),
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )


def execute_setup_requests(
    requests: Sequence[StructuredSetupRequest],
) -> list[str]:
    """Execute structured setup requests, batching compatible groups."""
    if not requests:
        return []

    outputs = [""] * len(requests)
    grouped_requests: dict[
        tuple[Any, ...],
        list[tuple[int, StructuredSetupRequest]],
    ] = {}
    for index, request in enumerate(requests):
        group_key = (
            id(request.component._model),
            request.specific_schema,
            request.max_tokens,
            request.terminators,
            request.temperature,
            request.top_p,
            request.top_k,
        )
        grouped_requests.setdefault(group_key, []).append((index, request))

    for grouped in grouped_requests.values():
        model = grouped[0][1].component._model
        batch_sampler = getattr(model, "sample_text_batch", None)
        use_batch = callable(batch_sampler) and len(grouped) > 1
        if not use_batch:
            for index, request in grouped:
                outputs[index] = execute_setup_request(request)
            continue
        try:
            raw_responses = batch_sampler(
                [request.prompt_text for _, request in grouped],
                max_tokens=grouped[0][1].max_tokens,
                terminators=grouped[0][1].terminators,
                json_schema=grouped[0][1].specific_schema.model_json_schema(),
                temperature=grouped[0][1].temperature,
                top_p=grouped[0][1].top_p,
                top_k=grouped[0][1].top_k,
            )
        except Exception:
            raw_responses = []
        if len(raw_responses) != len(grouped):
            for index, request in grouped:
                outputs[index] = execute_setup_request(request)
            continue
        for (index, _), raw_response in zip(grouped, raw_responses):
            outputs[index] = raw_response
    return outputs
