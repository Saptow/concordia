"""Education-specific Gemini wrapper with structured output support."""

from collections.abc import Collection, Mapping, Sequence
import copy
from enum import Enum
import json
import os
import re
import time
from typing import Any
from dotenv import load_dotenv
from absl import logging
from education.configs import Configs
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
from concordia.utils import text
from google import genai
from google.genai import types
from pydantic import BaseModel


class GeminiLanguageModel(language_model.LanguageModel):
  """Gemini wrapper with Concordia's LM interface plus structured outputs."""

  def __init__(
      self,
      model_name: str | None = None,
      *,
      api_key: str | None = None,
      project: str | None = None,
      location: str | None = None,
      safety_settings: Sequence[types.SafetySetting] | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str | None = None,
      sleep_periodically: bool | None = None,
  ) -> None:
    load_dotenv()  # Load environment variables from .env file
    cfg = Configs
    model_name = model_name or cfg.model_name
    api_key = api_key if api_key is not None else cfg.api_key
    project = project if project is not None else cfg.project
    location = location if location is not None else cfg.location
    safety_settings = (
        tuple(safety_settings)
        if safety_settings is not None
        else cfg.safety_settings
    )
    channel = channel or cfg.channel
    if sleep_periodically is None:
      sleep_periodically = cfg.sleep_periodically

    self._client = self._build_client(
        api_key=api_key,
        project=project,
        location=location,
    )

    self._model_name = model_name
    self._safety_settings = list(safety_settings)
    self._sleep_periodically = sleep_periodically
    self._measurements = measurements
    self._channel = channel
    self._history = cfg.history
    self._max_multiple_choice_attempts = cfg.max_multiple_choice_attempts
    self._calls_between_sleeping = cfg.calls_between_sleeping
    self._n_calls = 0

  def _build_client(
      self,
      *,
      api_key: str | None,
      project: str | None,
      location: str | None,
  ) -> genai.Client:
    if project and api_key:
      raise ValueError(
          'Provide either api_key (for AI Studio) or project/location '
          '(for Vertex AI), not both.'
      )

    if project:
      if not location:
        raise ValueError(
            'location is required when using Vertex AI (project is set).'
        )
      return genai.Client(vertexai=True, project=project, location=location)

    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
      raise ValueError(
          'GEMINI_API_KEY not found. Please provide it via the api_key '
          'parameter or set the GEMINI_API_KEY environment variable.'
      )
    return genai.Client(api_key=api_key)

  def _before_request(self) -> None:
    self._n_calls += 1
    if self._sleep_periodically and (
        self._n_calls % self._calls_between_sleeping == 0
    ):
      logging.info('Sleeping for 10 seconds...')
      time.sleep(10)

  def _publish_length(self, response_text: str) -> None:
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {'raw_text_length': len(response_text)}
      )

  def _clean_text(self, response_text: str) -> str:
    return re.sub(r'```(?:\w+)?\n?', '', response_text).strip()

  def _make_config(
      self,
      *,
      max_tokens: int,
      temperature: float,
      top_p: float,
      top_k: int,
      seed: int | None,
      response_mime_type: str,
      stop_sequences: Collection[str] = (),
      response_schema: Any | None = None,
      response_json_schema: Mapping[str, Any] | None = None,
  ) -> types.GenerateContentConfig:
    config_kwargs = {
        'temperature': temperature,
        'max_output_tokens': max_tokens,
        'candidate_count': 1,
        'top_p': top_p,
        'top_k': top_k,
        'response_mime_type': response_mime_type,
        'safety_settings': self._safety_settings,
        'seed': seed,
    }
    if stop_sequences:
      config_kwargs['stop_sequences'] = list(stop_sequences)
    if response_schema is not None:
      config_kwargs['response_schema'] = response_schema
    if response_json_schema is not None:
      config_kwargs['response_json_schema'] = dict(response_json_schema)
    return types.GenerateContentConfig(**config_kwargs)

  def _log_response_error(
      self,
      *,
      error: Exception,
      prompt: str,
      response: Any,
  ) -> None:
    logging.error('An error occurred: %s', error)
    logging.debug('prompt: %s', prompt)
    logging.debug('response: %s', response)

  def _default_response_mime_type(self, response_schema: Any) -> str:
    if isinstance(response_schema, type) and issubclass(response_schema, Enum):
      return 'text/x.enum'
    return 'application/json'

  def _parse_fallback_response(
      self,
      response_text: str,
      *,
      response_schema: Any | None,
      response_mime_type: str,
  ) -> Any:
    if (
        response_schema is not None
        and isinstance(response_schema, type)
        and issubclass(response_schema, BaseModel)
    ):
      return response_schema.model_validate_json(response_text)

    if (
        response_schema is not None
        and isinstance(response_schema, type)
        and issubclass(response_schema, Enum)
    ):
      enum_value = (
          json.loads(response_text)
          if response_mime_type == 'application/json'
          else response_text
      )
      return response_schema(enum_value)

    if response_mime_type == 'application/json':
      return json.loads(response_text)

    return response_text

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
    del timeout

    self._before_request()
    config = self._make_config(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        response_mime_type='text/plain',
        stop_sequences=terminators,
    )

    chat = self._client.chats.create(
        model=self._model_name,
        history=copy.deepcopy(self._history),
        config=config,
    )
    sample = chat.send_message(message=prompt)

    try:
      response = sample.candidates[0].content.parts[0].text
    except (ValueError, IndexError, AttributeError) as error:
      self._log_response_error(error=error, prompt=prompt, response=sample)
      response = ''

    response = self._clean_text(response)
    self._publish_length(response)
    return text.truncate(response, delimiters=terminators)

  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    sample = ''
    answer = ''
    for attempts in range(self._max_multiple_choice_attempts):
      temperature = sampling.dynamically_adjust_temperature(
          attempts, self._max_multiple_choice_attempts
      )
      question = (
          'The following is a multiple choice question. Respond '
          + 'with one of the possible choices, such as (a) or (b). '
          + f'Do not include reasoning.\n{prompt}'
      )
      sample = self.sample_text(
          question,
          max_tokens=256,
          temperature=temperature,
          seed=seed,
      )
      answer = sampling.extract_choice_response(sample)
      try:
        idx = responses.index(answer)
      except ValueError:
        logging.debug(
            'Sample choice fail: %s extracted from %s.', answer, sample
        )
        continue

      if self._measurements is not None:
        self._measurements.publish_datum(
            self._channel, {'choices_calls': attempts}
        )
      return idx, responses[idx], {}

    raise language_model.InvalidResponseError(
        'Too many multiple choice attempts.\n'
        f'Last attempt: {sample}, extracted: {answer}'
    )

  def sample_structured(
      self,
      prompt: str,
      *,
      response_schema: Any | None = None,
      response_json_schema: Mapping[str, Any] | None = None,
      response_mime_type: str | None = None,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> Any:
    """Samples a structured response.

    Pass exactly one of:
      - `response_schema` for Pydantic models, enums, or typing aliases such as
        `list[MyModel]`
      - `response_json_schema` for raw JSON Schema
    """
    del timeout

    if (response_schema is None) == (response_json_schema is None):
      raise ValueError(
          'Provide exactly one of response_schema or response_json_schema.'
      )

    self._before_request()
    response_mime_type = response_mime_type or self._default_response_mime_type(
        response_schema
    )

    response = self._client.models.generate_content(
        model=self._model_name,
        contents=prompt,
        config=self._make_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            response_json_schema=response_json_schema,
        ),
    )

    parsed = getattr(response, 'parsed', None)
    response_text = self._clean_text(getattr(response, 'text', '') or '')
    self._publish_length(response_text)

    if parsed is not None:
      return parsed

    response_text = response_text.strip()
    if not response_text:
      logging.debug('prompt: %s', prompt)
      logging.debug('response: %s', response)
      raise ValueError('Gemini returned an empty structured response.')

    return self._parse_fallback_response(
        response_text,
        response_schema=response_schema,
        response_mime_type=response_mime_type,
    )
