# Copyright 2025 DeepMind Technologies Limited.
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

"""Language Model that uses vLLM for local inference.

vLLM is a fast and memory-efficient inference engine for large language models.
This wrapper allows Concordia to use locally hosted models through vLLM.

Example usage:
  model = vllm_model.VLLMLanguageModel(
      model_name="microsoft/DialoGPT-medium",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.8,
  )

  # Only load the base model once, then create multiple LoRA adapters.
  lora1 = vllm_model.VLLMLora(
      model_name="microsoft/DialoGPT-medium",
      lora_path="/path/to/lora1",
      vllm_language_model=model,
  )
  lora2 = vllm_model.VLLMLora(
      model_name="microsoft/DialoGPT-medium",
      lora_path="/path/to/lora2",
      vllm_language_model=model,
  )

  # Load another model with its own LoRA adapter.
  lora3 = vllm_model.VLLMLora(
      model_name="Qwen/Qwen2.5-7B-Instruct",
      lora_path="/path/to/lora3",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.8,
  )
"""

from collections.abc import Collection, Sequence
import threading
from typing import Any, Mapping, override

from configs import VLLMConfig
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from vllm import LLM
from vllm import SamplingParams
from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams


class VLLMLanguageModel(language_model.LanguageModel):
  """Language model wrapper for vLLM local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      tensor_parallel_size: int = VLLMConfig.DEFAULT_TENSOR_PARALLEL_SIZE,
      gpu_memory_utilization: float = VLLMConfig.DEFAULT_GPU_MEMORY_UTILIZATION,
      enable_lora: bool = False,
      max_model_len: int | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      enable_prefix_caching: bool = VLLMConfig.DEFAULT_ENABLE_PREFIX_CACHING,
      max_lora_rank: int = VLLMConfig.DEFAULT_MAX_LORA_RANK,
      structured_outputs_backend: str | None = VLLMConfig.DEFAULT_STRUCTURED_OUTPUTS_BACKEND,
      structured_outputs_disable_fallback: bool = VLLMConfig.DEFAULT_STRUCTURED_OUTPUTS_DISABLE_FALLBACK,
      **kwargs: Any,
  ):
    """Initialize the vLLM language model.

    Args:
      model_name: The name or path of the model to load.
      tensor_parallel_size: Number of GPUs to use for tensor parallelism.
      gpu_memory_utilization: Fraction of GPU memory to use.
      enable_lora: Whether to enable LoRA adapters.
      max_model_len: Maximum model context length.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      enable_prefix_caching: Whether to enable prefix caching in vLLM.
      max_lora_rank: Maximum rank for LoRA adapters.
      structured_outputs_backend: Structured output backend to use for
        deterministic behavior. Set to None to use vLLM default selection.
      structured_outputs_disable_fallback: Whether to disable fallback to other
        structured output backends.
      **kwargs: Additional arguments passed to vLLM LLM constructor.

    Raises:
      ImportError: If vLLM is not installed.
      ValueError: If LoRA is enabled but no path is provided.
    """
    self._model_name = model_name
    self._measurements = measurements
    self._channel = channel
    self._enable_lora = enable_lora
    self._nbr_lora_adapters = 0  # Number of LoRA adapters used with this model.
    # Each adapter needs a unique ID which is why we keep count.
    self._lock = threading.Lock()

    # Initialize vLLM model
    llm_kwargs = {
        'model': model_name,
        'tensor_parallel_size': tensor_parallel_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'enable_lora': enable_lora,
        'enable_prefix_caching': enable_prefix_caching,
        'max_lora_rank': max_lora_rank,
        **kwargs,
    }

    if max_model_len is not None:
      llm_kwargs['max_model_len'] = max_model_len

    if structured_outputs_backend is not None:
      llm_kwargs['structured_outputs_config'] = StructuredOutputsConfig(
          backend=structured_outputs_backend,
          disable_fallback=structured_outputs_disable_fallback,
      )

    self._llm = LLM(**llm_kwargs)

  def _build_sampling_params(
      self,
      *,
      max_tokens: int,
      terminators: Collection[str],
      temperature: float,
      top_p: float,
      top_k: int,
      seed: int | None,
      json_schema: dict[str, Any] | None = None,
  ) -> SamplingParams:
    """Build sampling params shared by single and batched generation."""
    structured_outputs = None
    if json_schema is not None:
      structured_outputs = StructuredOutputsParams(json=json_schema)
    return SamplingParams(
        structured_outputs=structured_outputs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=list(terminators) if terminators else None,
    )

  def increment_lora_adapters(self) -> int:
    """Increment the count of LoRA adapters used."""
    if not self._enable_lora:
      raise ValueError('LoRA is not enabled for this model.')
    self._nbr_lora_adapters += 1
    return self._nbr_lora_adapters

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      lora_request: LoRARequest | None = None,
      json_schema: dict[str, Any] | None = None,
  ) -> str:
    """Sample text from the vLLM model."""
    del timeout  # vLLM doesn't support timeout in SamplingParams
    sampling_params = self._build_sampling_params(
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        json_schema=json_schema,
    )

    # Generate response
    with self._lock:
      outputs = self._llm.generate(
          prompts=[prompt],
          sampling_params=sampling_params,
          lora_request=lora_request,
      )

    # Extract generated text
    generated_text = outputs[0].outputs[0].text

    # Log statistics
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(generated_text)},
      )

    return generated_text

  def sample_text_batch(
      self,
      prompts: Sequence[str],
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      lora_request: LoRARequest | None = None,
      json_schema: dict[str, Any] | None = None,
  ) -> list[str]:
    """Sample batched text from the vLLM model with shared params."""
    del timeout  # vLLM doesn't support timeout in SamplingParams
    if not prompts:
      return []

    sampling_params = self._build_sampling_params(
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        json_schema=json_schema,
    )

    with self._lock:
      outputs = self._llm.generate(
          prompts=list(prompts),
          sampling_params=sampling_params,
          lora_request=lora_request,
      )

    generated_texts = [output.outputs[0].text for output in outputs]

    if self._measurements is not None:
      for generated_text in generated_texts:
        self._measurements.publish_datum(
            self._channel,
            {'raw_text_length': len(generated_text)},
        )

    return generated_texts

  def chat(
      self,
      messages: Sequence[Mapping[str, Any]],
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      lora_request: LoRARequest | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      tool_choice: Any = None,
  ) -> str:
    """Run vLLM chat completion, optionally with tool definitions."""
    del timeout  # vLLM chat does not consume timeout via SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=list(terminators) if terminators else None,
    )

    chat_kwargs: dict[str, Any] = {
        'messages': list(messages),
        'sampling_params': sampling_params,
    }
    if lora_request is not None:
        chat_kwargs['lora_request'] = lora_request
    if tools is not None:
        chat_kwargs['tools'] = list(tools)
    if tool_choice is None and tools is not None:
        chat_kwargs['tool_choice'] = 'required'
    elif tool_choice is not None:
        chat_kwargs['tool_choice'] = tool_choice

    with self._lock:
      outputs = self._llm.chat(**chat_kwargs)

    generated_text = outputs[0].outputs[0].text

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {
              'chat_text_length': len(generated_text),
              'chat_with_tools': bool(tools),
          },
      )

    return generated_text

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      max_tokens: int = 2000,
      *,
      seed: int | None = None,
      lora_request: LoRARequest | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from available responses using guided decoding.

    Falls back to prompt-logprob scoring when guided choice output does not map
    cleanly to one of the provided responses.
    """

    if not responses:
      raise ValueError('No responses provided to choose from.')

    sampling_params = SamplingParams(
        structured_outputs=StructuredOutputsParams(choices=list(responses)),
        temperature=language_model.DEFAULT_TEMPERATURE,
        max_tokens=max_tokens,
        seed=seed,
    )

    with self._lock:
      outputs = self._llm.generate(
          prompts=[prompt],
          sampling_params=sampling_params,
          lora_request=lora_request,
      )

    generated_text = outputs[0].outputs[0].text
    guided_idx = self._match_guided_choice_output(generated_text, responses)
    if guided_idx is not None:
      debug_info = {
          'method': 'guided_choice',
          'guided_output': generated_text,
      }
      if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel,
            {'choice_method': 'guided_choice', 'num_choices': len(responses)},
        )
      return guided_idx, responses[guided_idx], debug_info

    return self._sample_choice_with_logprobs(
        prompt=prompt,
        responses=responses,
        seed=seed,
        lora_request=lora_request,
        guided_output=generated_text,
    )

  @staticmethod
  def _match_guided_choice_output(
      generated_text: str, responses: Sequence[str]
  ) -> int | None:
    """Match guided choice text to one of the candidate responses."""
    if not responses:
      return None

    raw = str(generated_text or '').strip()
    if raw in responses:
      return responses.index(raw)

    normalized = raw.strip().strip('`').strip().strip('"').strip("'").strip()
    if normalized in responses:
      return responses.index(normalized)

    normalized_fold = normalized.casefold()
    exact_fold = [
        idx for idx, response in enumerate(responses)
        if str(response).strip().casefold() == normalized_fold
    ]
    if len(exact_fold) == 1:
      return exact_fold[0]

    starts_with = [
        idx for idx, response in enumerate(responses)
        if normalized_fold.startswith(str(response).strip().casefold())
    ]
    if len(starts_with) == 1:
      return starts_with[0]
    return None

  def _sample_choice_with_logprobs(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None,
      lora_request: LoRARequest | None,
      guided_output: str | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Fallback choice scorer using prompt log probabilities."""
    # We only need prompt logprobs, not generation.
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=0,
        seed=seed,
    )

    prompts = [prompt + response for response in responses]

    with self._lock:
      outputs = self._llm.generate(
          prompts=prompts,
          sampling_params=sampling_params,
          lora_request=lora_request,
      )

      logprobs = []
      for i, output in enumerate(outputs):
        if not output.prompt_logprobs:
          raise ValueError('No prompt logprobs returned by vLLM.')

        # Find tokens corresponding to the response
        tokenizer = self._llm.get_tokenizer()
        prompt_tokens = tokenizer.encode(prompt)
        full_tokens = tokenizer.encode(prompts[i])

        # Response tokens are the difference
        response_start_idx = len(prompt_tokens)

        # Sum logprobs for response tokens
        total_logprob = 0.0
        prompt_logprobs = output.prompt_logprobs

        for j in range(response_start_idx, len(full_tokens)):
          if j < len(prompt_logprobs) and prompt_logprobs[j]:
            # Get the token ID at this position
            token_id = full_tokens[j]
            if token_id in prompt_logprobs[j]:
              total_logprob += prompt_logprobs[j][token_id].logprob

        logprobs.append(total_logprob)

    best_idx = int(max(range(len(logprobs)), key=lambda i: logprobs[i]))

    debug_info = {
        'logprobs': {
            response: logprobs[i] for i, response in enumerate(responses)
        },
        'method': 'logprobs',
    }
    if guided_output is not None:
      debug_info['guided_output'] = guided_output

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {
              'choice_method': 'logprobs',
              'num_choices': len(responses),
              'guided_choice_fallback': guided_output is not None,
          },
      )

    return best_idx, responses[best_idx], debug_info


class VLLMLora(language_model.LanguageModel):
  """Language model wrapper for vLLM local inference with LoRA."""

  def __init__(
      self,
      model_name: str,
      *,
      lora_path: str | None = None,
      vllm_language_model: VLLMLanguageModel | None = None,
      tensor_parallel_size: int = VLLMConfig.DEFAULT_TENSOR_PARALLEL_SIZE,
      gpu_memory_utilization: float = VLLMConfig.DEFAULT_GPU_MEMORY_UTILIZATION,
      enable_lora: bool = False,
      max_model_len: int | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      enable_prefix_caching: bool = VLLMConfig.DEFAULT_ENABLE_PREFIX_CACHING,
      max_lora_rank: int = VLLMConfig.DEFAULT_MAX_LORA_RANK,
      structured_outputs_backend: str | None = VLLMConfig.DEFAULT_STRUCTURED_OUTPUTS_BACKEND,
      structured_outputs_disable_fallback: bool = VLLMConfig.DEFAULT_STRUCTURED_OUTPUTS_DISABLE_FALLBACK,
      **kwargs: Any,
  ):
    """Initialize the vLLM language model with LoRA.

    This is a wrapper around VLLMLanguageModel that passes the LoRA request
    along each sampling call.

    Args:
      model_name: The name or path of the model to load.
      lora_path: Path to LoRA adapter weights (must be provided).
      vllm_language_model: An existing VLLMLanguageModel instance to use. If
        None, a new one is created. If provided, other vLLM parameters are
        ignored.
      tensor_parallel_size: Number of GPUs to use for tensor parallelism.
      gpu_memory_utilization: Fraction of GPU memory to use.
      enable_lora: Whether to enable LoRA adapters.
      max_model_len: Maximum model context length.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
      enable_prefix_caching: Whether to enable prefix caching in vLLM.
      max_lora_rank: Maximum rank for LoRA adapters.
      structured_outputs_backend: Structured output backend to use for
        deterministic behavior. Set to None to use vLLM default selection.
      structured_outputs_disable_fallback: Whether to disable fallback to other
        structured output backends.
      **kwargs: Additional arguments passed to vLLM LLM constructor.
    """

    if vllm_language_model is not None:
      self._vllm_model = vllm_language_model
      adapter_id = self._vllm_model.increment_lora_adapters()
    else:
      self._vllm_model = VLLMLanguageModel(
          model_name=model_name,
          tensor_parallel_size=tensor_parallel_size,
          gpu_memory_utilization=gpu_memory_utilization,
          enable_lora=enable_lora,
          max_model_len=max_model_len,
          measurements=measurements,
          channel=channel,
          enable_prefix_caching=enable_prefix_caching,
          max_lora_rank=max_lora_rank,
          structured_outputs_backend=structured_outputs_backend,
          structured_outputs_disable_fallback=(
              structured_outputs_disable_fallback
          ),
          **kwargs,
      )
      adapter_id = self._vllm_model.increment_lora_adapters()

    if lora_path is None:
      raise ValueError('lora_path must be provided to initialize VLLMLora.')

    # Setup LoRA request
    self._lora_request = LoRARequest(
        f'lora_adapter_{adapter_id}', adapter_id, lora_path
    )

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    """Sample text from the vLLM model with LoRA."""
    return self._vllm_model.sample_text(
        prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        lora_request=self._lora_request,
    )

  def sample_text_batch(
      self,
      prompts: Sequence[str],
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      json_schema: dict[str, Any] | None = None,
  ) -> list[str]:
    """Sample batched text from the vLLM model with the LoRA adapter."""
    return self._vllm_model.sample_text_batch(
        prompts,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        lora_request=self._lora_request,
        json_schema=json_schema,
    )

  def chat(
      self,
      messages: Sequence[Mapping[str, Any]],
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
      tools: Sequence[Mapping[str, Any]] | None = None,
      tool_choice: Any = None,
  ) -> str:
    """Run vLLM chat completion with the configured LoRA adapter."""
    return self._vllm_model.chat(
        messages,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
        seed=seed,
        lora_request=self._lora_request,
        tools=tools,
        tool_choice=tool_choice,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from the available responses using log probabilities."""
    return self._vllm_model.sample_choice(
        prompt,
        responses,
        seed=seed,
        lora_request=self._lora_request,
    )
