# Copyright 2023 DeepMind Technologies Limited.
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

"""Agent component for asking questions about the agent's recent memories."""

from collections.abc import Callable, Collection, Mapping, Sequence
import dataclasses
import datetime
import json
from typing import Any, Literal, override
from pydantic import BaseModel, Field, RootModel, create_model

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


SELF_PERCEPTION_QUESTION = (
    'What kind of person is {agent_name}? Respond using 1-5 sentences.')

SITUATION_PERCEPTION_QUESTION = (
    'What kind of situation is {agent_name} in right now? Respond using 1-5 '
    'sentences.'
)
PERSON_BY_SITUATION_QUESTION = (
    'What would a person like {agent_name} do in a situation like this? '
    'Respond using 1-5 sentences.'
)
AVAILABLE_OPTIONS_QUESTION = (
    'What actions are available to {agent_name} right now?'
)
BEST_OPTION_PERCEPTION_QUESTION = (
    "Which of {agent_name}'s options "
    'has the highest likelihood of causing {agent_name} to achieve '
    'their goal? If multiple options have the same likelihood, select '
    'the option that {agent_name} thinks will most quickly and most '
    'surely achieve their goal.'
)


@dataclasses.dataclass(frozen=True)
class TextPreActRequest:
  """One open-text pre-act request that can be batched."""

  component: Any
  prompt_text: str
  question_text: str
  answer_prefix: str
  max_tokens: int = 1000
  terminators: tuple[str, ...] = ('\n',)
  temperature: float = language_model.DEFAULT_TEMPERATURE
  top_p: float = language_model.DEFAULT_TOP_P
  top_k: int = language_model.DEFAULT_TOP_K


def execute_text_pre_act_request(
    request: TextPreActRequest,
) -> str:
  """Execute one open-text pre-act request through the language model."""
  return request.component._model.sample_text(
      prompt=request.prompt_text,
      max_tokens=request.max_tokens,
      terminators=request.terminators,
      temperature=request.temperature,
      top_p=request.top_p,
      top_k=request.top_k,
  )


def execute_text_pre_act_requests(
    requests: Sequence[TextPreActRequest],
) -> list[str]:
  """Execute open-text pre-act requests, batching compatible groups."""
  if not requests:
    return []

  outputs = [''] * len(requests)
  grouped_requests: dict[
      tuple[Any, ...],
      list[tuple[int, TextPreActRequest]],
  ] = {}
  for index, request in enumerate(requests):
    group_key = (
        id(request.component._model),
        request.max_tokens,
        request.terminators,
        request.temperature,
        request.top_p,
        request.top_k,
    )
    grouped_requests.setdefault(group_key, []).append((index, request))

  for grouped in grouped_requests.values():
    model = grouped[0][1].component._model
    batch_sampler = getattr(model, 'sample_text_batch', None)
    use_batch = callable(batch_sampler) and len(grouped) > 1
    if not use_batch:
      for index, request in grouped:
        outputs[index] = execute_text_pre_act_request(request)
      continue
    try:
      raw_responses = batch_sampler(
          [request.prompt_text for _, request in grouped],
          max_tokens=grouped[0][1].max_tokens,
          terminators=grouped[0][1].terminators,
          temperature=grouped[0][1].temperature,
          top_p=grouped[0][1].top_p,
          top_k=grouped[0][1].top_k,
      )
    except Exception:
      raw_responses = []
    if len(raw_responses) != len(grouped):
      for index, request in grouped:
        outputs[index] = execute_text_pre_act_request(request)
      continue
    for (index, _), raw_response in zip(grouped, raw_responses):
      outputs[index] = raw_response
  return outputs


@dataclasses.dataclass(frozen=True)
class StructuredPreActRequest:
  """One structured phase-1 chooser request that can be batched."""

  component: 'QuestionOfRecentMemoriesStructured'
  prompt_text: str
  prompt_context: str
  question_text: str
  answer_prefix: str
  output_schema: Any
  json_schema: dict[str, Any]
  max_tokens: int = 1000
  terminators: tuple[str, ...] = ('\n',)
  temperature: float = language_model.DEFAULT_TEMPERATURE
  top_p: float = language_model.DEFAULT_TOP_P
  top_k: int = language_model.DEFAULT_TOP_K


class QuestionOfRecentMemories(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """A question that conditions the agent's behavior.

  The default question is 'What would a person like {agent_name} do in a
  situation like this?' and the default answer prefix is '{agent_name} would '.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      pre_act_label: str,
      question: str,
      answer_prefix: str,
      add_to_memory: bool,
      memory_tag: str = '',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      terminators: Collection[str] = ('\n',),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      persist_pre_act_value_across_updates: bool = False,
  ):
    """Initializes the QuestionOfRecentMemories component.

    Args:
      model: The language model to use.
      pre_act_label: Prefix to add to the value of the component when called in
        `pre_act`.
      question: The question to ask.
      answer_prefix: The prefix to add to the answer.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_key: The name of the memory component from which to
        retrieve recent memories.
      components: Keys of components to condition the answer on.
      terminators: strings that must not be present in the model's response. If
        emitted by the model the response will be truncated before them.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
      persist_pre_act_value_across_updates: Whether to keep the generated
        pre-act value across update cycles instead of recomputing it each turn.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._components = tuple(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._question = question
    self._terminators = terminators
    self._answer_prefix = answer_prefix
    self._add_to_memory = add_to_memory
    self._memory_tag = memory_tag
    self._persist_pre_act_value_across_updates = bool(
        persist_pre_act_value_across_updates
    )

  def get_component_pre_act_label(self, component_name: str) -> str:
    """Returns the pre-act label of a named component of the parent entity."""
    return (
        self.get_entity().get_component(
            component_name, type_=action_spec_ignored.ActionSpecIgnored
        ).get_pre_act_label()
    )

  def _component_pre_act_display(self, key: str) -> str:
    """Returns the pre-act label and value of a named component."""
    return (
        f'  {self.get_component_pre_act_label(key)}: '
        f'{self.get_named_component_pre_act_value(key)}')

  def _make_pre_act_value(self) -> str:
    request = self.build_batched_pre_act_request()
    if request is None:
      return self._pre_act_value or ''
    raw_response = execute_text_pre_act_request(request)
    return self.apply_batched_pre_act_response(request, raw_response)

  def build_batched_pre_act_request(
      self,
      action_spec: entity_lib.ActionSpec | None = None,
  ) -> TextPreActRequest | None:
    del action_spec
    with self._lock:
      if (
          self._persist_pre_act_value_across_updates
          and self._pre_act_value is not None
      ):
        return None

    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )

    prompt = interactive_document.InteractiveDocument(self._model)

    component_states = '\n'.join(
        [
            self._component_pre_act_display(key)
            for key in self._components
            if key not in ('situation_perception', 'self_perception')
        ]
    )
    prompt.statement(component_states)

    mems = ''
    if self._num_memories_to_retrieve > 0:
      mems = '\n'.join([
          mem
          for mem in memory.retrieve_recent(
              limit=self._num_memories_to_retrieve
          )
      ])
      prompt.statement(f'Recent observations of {agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    question = self._question.format(agent_name=agent_name)
    answer_prefix = self._answer_prefix.format(agent_name=agent_name)
    prompt_text = (
        f'{prompt.view().text()}Question: {question}\n'
        f'Answer: {answer_prefix}'
    )
    return TextPreActRequest(
        component=self,
        prompt_text=prompt_text,
        question_text=question,
        answer_prefix=answer_prefix,
        max_tokens=1000,
        terminators=tuple(self._terminators),
    )

  def apply_batched_pre_act_response(
      self,
      request: TextPreActRequest,
      raw_response: str,
  ) -> str:
    result = str(raw_response or '').strip()
    if not result:
      result = '[no response generated]'
    result = request.answer_prefix + result

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result}')

    with self._lock:
      self._pre_act_value = result

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': request.question_text,
        'State': result,
        'Chain of thought': request.prompt_text.splitlines() + [''],
    }

    if self._clock_now is not None:
      log['Time'] = self._clock_now()

    self._logging_channel(log)
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Converts the component to JSON data."""
    with self._lock:
      return {
          'question': self._question,
          'answer_prefix': self._answer_prefix,
          'add_to_memory': self._add_to_memory,
          'memory_tag': self._memory_tag,
          'memory_component_key': self._memory_component_key,
          'components': list(self._components),
          'terminators': list(self._terminators),
          'num_memories_to_retrieve': self._num_memories_to_retrieve,
          'persist_pre_act_value_across_updates': (
              self._persist_pre_act_value_across_updates
          ),
          'pre_act_label': self.get_pre_act_label(),
      }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Sets the component state from JSON data."""
    with self._lock:
      if 'question' in state:
        self._question = str(state['question'])
      if 'answer_prefix' in state:
        self._answer_prefix = str(state['answer_prefix'])
      if 'add_to_memory' in state:
        self._add_to_memory = bool(state['add_to_memory'])
      if 'memory_tag' in state:
        self._memory_tag = str(state['memory_tag'])
      if 'memory_component_key' in state:
        self._memory_component_key = str(state['memory_component_key'])
      if 'components' in state:
        self._components = tuple(state['components'])
      if 'terminators' in state:
        self._terminators = tuple(state['terminators'])
      if 'num_memories_to_retrieve' in state:
        self._num_memories_to_retrieve = state['num_memories_to_retrieve']
      if 'persist_pre_act_value_across_updates' in state:
        self._persist_pre_act_value_across_updates = bool(
            state['persist_pre_act_value_across_updates']
        )

  @override
  def update(self) -> None:
    if self._persist_pre_act_value_across_updates:
      return
    super().update()

# QuestionOfRecentMemories component with structured outputs
class QuestionOfRecentMemoriesStructured(
    QuestionOfRecentMemories
):
  """A QuestionOfRecentMemories component with structured outputs.
  """
  def __init__(
      self, 
      model: language_model.LanguageModel,
      pre_act_label: str,
      question: str,
      add_to_memory: bool,
      answer_prefix: str = '',
      memory_tag: str = '',
      memory_component_key: str = (
          memory_component.DEFAULT_MEMORY_COMPONENT_KEY
      ),
      components: Sequence[str] = (),
      terminators: Collection[str] = ('\n',),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      output_schema: RootModel = None,
      choice_responses: Sequence[str] = (),
      choice_response_descriptions: Mapping[str, str] | None = None,
      randomize_choice_responses: bool = False,
    ):
    """Initializes the QuestionOfRecentMemories component.

    Args:
      model: The language model to use.
      pre_act_label: Prefix to add to the value of the component when called in
        `pre_act`.
      question: The question to ask.
      answer_prefix: The prefix to add to the answer.
      add_to_memory: Whether to add the answer to the memory.
      memory_tag: The tag to use when adding the answer to the memory.
      memory_component_key: The name of the memory component from which to
        retrieve recent memories.
      components: Keys of components to condition the answer on.
      terminators: strings that must not be present in the model's response. If
        emitted by the model the response will be truncated before them.
      clock_now: time callback to use.
      num_memories_to_retrieve: The number of recent memories to retrieve.
    """
    super().__init__(model, pre_act_label, question, answer_prefix,
                     add_to_memory, memory_tag, memory_component_key, components,
                     terminators, clock_now, num_memories_to_retrieve)
    self._output_schema = output_schema
    self._choice_responses = tuple(str(x) for x in choice_responses if str(x))
    self._choice_response_descriptions = {
        str(key).strip().upper(): str(value).strip()
        for key, value in dict(choice_response_descriptions or {}).items()
        if str(key).strip() and str(value).strip()
    }
    self._runtime_choice_responses: tuple[str, ...] | None = None
    self._randomize_choice_responses = bool(randomize_choice_responses)
    self._last_prompt_context: str = ''

  def get_last_prompt_context(self) -> str:
    """Returns the latest chooser prompt context without the question text."""
    with self._lock:
      return self._last_prompt_context

  @override
  def pre_act(
      self,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Align choice responses with the current action spec when available."""
    runtime_choices: tuple[str, ...] | None = None
    if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
      runtime_choices = tuple(str(x) for x in action_spec.options if str(x))
    with self._lock:
      self._runtime_choice_responses = runtime_choices
    return super().pre_act(action_spec)

  def _get_active_choice_responses(self) -> tuple[str, ...]:
    return (
        self._runtime_choice_responses
        if self._runtime_choice_responses
        else self._choice_responses
    )

  def _resolve_output_schema(
      self,
      active_choice_responses: Sequence[str],
  ) -> Any:
    output_schema = self._output_schema
    if not active_choice_responses or output_schema is None:
      return output_schema
    if 'chosen_action_type' not in getattr(output_schema, 'model_fields', {}):
      return output_schema
    literal_choices = Literal.__getitem__(tuple(active_choice_responses))
    field_info = output_schema.model_fields['chosen_action_type']
    return create_model(
        f'{output_schema.__name__}RuntimeChoices',
        __base__=output_schema,
        chosen_action_type=(
            literal_choices,
            Field(
                ...,
                description=field_info.description,
            ),
        ),
    )

  def _render_active_choice_descriptions(
      self,
      active_choice_responses: Sequence[str],
  ) -> str:
    if not active_choice_responses or not self._choice_response_descriptions:
      return ''
    lines: list[str] = []
    for choice in active_choice_responses:
      normalized_choice = str(choice).strip().upper()
      if not normalized_choice:
        continue
      description = self._choice_response_descriptions.get(normalized_choice)
      if not description:
        continue
      lines.append(f'- {normalized_choice}: {description}')
    return '\n'.join(lines)

  def build_pre_act_request(
      self,
      action_spec: entity_lib.ActionSpec | None = None,
  ) -> StructuredPreActRequest:
    """Build a batchable structured chooser request for this component."""
    if action_spec is not None:
      runtime_choices = None
      if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
        runtime_choices = tuple(str(x) for x in action_spec.options if str(x))
      with self._lock:
        self._runtime_choice_responses = runtime_choices

    agent_name = self.get_entity().name
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    mems = ''
    if self._num_memories_to_retrieve > 0:
      mems = '\n'.join([
          mem
          for mem in memory.retrieve_recent(limit=self._num_memories_to_retrieve)
      ])
    prompt = interactive_document.InteractiveDocument(self._model)
    component_states = '\n'.join(
        [
            self._component_pre_act_display(key)
            for key in self._components
            if key not in ('situation_perception', 'self_perception')
        ]
    )
    prompt.statement(component_states)
    perception_keys = tuple(
        key
        for key in ('situation_perception', 'self_perception')
        if key in self._components
    )
    if perception_keys:
      for key in perception_keys:
        prompt.statement(
            f"{self.get_component_pre_act_label(key)}:\n"
            f"{self.get_named_component_pre_act_value(key)}\n"
        )

    if mems:
      prompt.statement(f'Recent observations of {agent_name}:\n{mems}')
      prompt.statement('')
    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    prompt_context = prompt.view().text().strip()
    question = self._question.format(agent_name=agent_name)
    answer_prefix = self._answer_prefix.format(agent_name=agent_name)
    active_choice_responses = self._get_active_choice_responses()
    active_choice_descriptions = self._render_active_choice_descriptions(
        active_choice_responses
    )
    output_schema = self._resolve_output_schema(active_choice_responses)
    if active_choice_responses:
      if output_schema is None:
        raise ValueError(
            'QuestionOfRecentMemoriesStructured requires an output schema when '
            'batched phase-1 selection uses structured choices.'
        )
      if active_choice_descriptions:
        question = (
            f'{question}\n'
            'Action type descriptions:\n'
            f'{active_choice_descriptions}'
        )
      question = (
          f'{question}\n'
          'Return a structured response that selects exactly one of the '
          'allowed action types and explains the decision briefly.'
      )
    elif output_schema is None:
      raise ValueError(
          'QuestionOfRecentMemoriesStructured requires either '
          '`choice_responses` or `output_schema`.'
      )

    prompt_text = (
        f'{prompt.view().text()}Question: {question}\n'
        f'Answer: {answer_prefix}'
    )
    return StructuredPreActRequest(
        component=self,
        prompt_text=prompt_text,
        prompt_context=prompt_context,
        question_text=question,
        answer_prefix=answer_prefix,
        output_schema=output_schema,
        json_schema=output_schema.model_json_schema(),
    )

  def _parse_pre_act_response(
      self,
      request: StructuredPreActRequest,
      raw_response: str,
  ) -> str:
    """Validate and normalize one structured chooser response."""
    try:
      parsed_response = request.output_schema.model_validate_json(raw_response)
      response = parsed_response.model_dump_json()
    except Exception:
      response = raw_response
    return f'{request.answer_prefix}{response}'

  def apply_pre_act_response(
      self,
      request: StructuredPreActRequest,
      raw_response: str,
  ) -> str:
    """Persist one structured chooser response into the component cache."""
    result_str = self._parse_pre_act_response(request, raw_response)
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    if self._add_to_memory:
      memory.add(f'{self._memory_tag} {result_str}')

    self._last_prompt_context = request.prompt_context
    self._pre_act_value = result_str

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': request.question_text,
        'State': result_str,
        'Chain of thought': (
            request.prompt_text.splitlines()
            + ['']
            + [f'Answer: {result_str}']
        ),
    }
    if self._clock_now is not None:
      log['Time'] = self._clock_now()

    self._logging_channel(log)
    return result_str

  def execute_pre_act_request(
      self,
      request: StructuredPreActRequest,
  ) -> str:
    """Execute one structured chooser request through the language model."""
    raw_response = self._model.sample_text(
        prompt=request.prompt_text,
        max_tokens=request.max_tokens,
        terminators=request.terminators,
        json_schema=request.json_schema,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
    )
    return self.apply_pre_act_response(request, raw_response)

  @classmethod
  def execute_pre_act_requests(
      cls,
      requests: Sequence[StructuredPreActRequest],
  ) -> list[str]:
    """Execute structured chooser requests, batching compatible groups."""
    if not requests:
      return []

    outputs = [''] * len(requests)
    grouped_requests: dict[
        tuple[Any, ...],
        list[tuple[int, StructuredPreActRequest]],
    ] = {}
    for index, request in enumerate(requests):
      group_key = (
          id(request.component._model),
          json.dumps(request.json_schema, sort_keys=True, ensure_ascii=False),
          request.max_tokens,
          request.terminators,
          request.temperature,
          request.top_p,
          request.top_k,
      )
      grouped_requests.setdefault(group_key, []).append((index, request))

    for grouped in grouped_requests.values():
      model = grouped[0][1].component._model
      batch_sampler = getattr(model, 'sample_text_batch', None)
      use_batch = callable(batch_sampler) and len(grouped) > 1
      if not use_batch:
        for index, request in grouped:
          outputs[index] = request.component.execute_pre_act_request(request)
        continue
      try:
        raw_responses = batch_sampler(
            [request.prompt_text for _, request in grouped],
            max_tokens=grouped[0][1].max_tokens,
            terminators=grouped[0][1].terminators,
            json_schema=grouped[0][1].json_schema,
            temperature=grouped[0][1].temperature,
            top_p=grouped[0][1].top_p,
            top_k=grouped[0][1].top_k,
        )
      except Exception:
        raw_responses = []
      if len(raw_responses) != len(grouped):
        for index, request in grouped:
          outputs[index] = request.component.execute_pre_act_request(request)
        continue
      for (index, request), raw_response in zip(grouped, raw_responses):
        outputs[index] = request.component.apply_pre_act_response(
            request,
            raw_response,
        )
    return outputs

  @override
  def _make_pre_act_value(self) -> str:
    """Returns the answer to the question in a structured format."""
    request = self.build_pre_act_request()
    return self.execute_pre_act_request(request)

  @override
  def update(self) -> None:
    with self._lock:
      self._runtime_choice_responses = None
      self._last_prompt_context = ''
      if not self._persist_pre_act_value_across_updates:
        # Mirror `ActionSpecIgnored.update()` here so we do not reacquire
        # `_lock` through `super().update()`, which would deadlock with a
        # non-reentrant lock.
        self._pre_act_value = None
      
class QuestionOfRecentMemoriesWithoutPreAct(
    action_spec_ignored.ActionSpecIgnored, entity_component.ComponentWithLogging
):
  """QuestionOfRecentMemories component that does not output to pre_act.
  """

  def __init__(self, *args, **kwargs):
    self._component = QuestionOfRecentMemories(*args, **kwargs)

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    self._component.set_entity(entity)

  def _make_pre_act_value(self) -> str:
    return ''

  def get_pre_act_value(self) -> str:
    return self._component.get_pre_act_value()

  def get_pre_act_label(self) -> str:
    return self._component.get_pre_act_label()

  def pre_act(
      self,
      unused_action_spec: entity_lib.ActionSpec,
  ) -> str:
    del unused_action_spec
    return ''

  def update(self) -> None:
    self._component.update()


class SelfPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SelfPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of person is the agent?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SELF_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SELF_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is ',
        add_to_memory=False,
        memory_tag='[self reflection]',
        **kwargs,
    )


class SituationPerception(QuestionOfRecentMemories):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class SituationPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what kind of situation is it?'."""

  def __init__(
      self,
      **kwargs,
  ):
    default_pre_act_label = f'\n{SITUATION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=SITUATION_PERCEPTION_QUESTION,
        answer_prefix='{agent_name} is currently ',
        add_to_memory=False,
        **kwargs,
    )


class PersonBySituation(QuestionOfRecentMemories):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class PersonBySituationWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """What would a person like the agent do in a situation like this?"""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{PERSON_BY_SITUATION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=PERSON_BY_SITUATION_QUESTION,
        answer_prefix='{agent_name} would ',
        add_to_memory=False,
        memory_tag='[intent reflection]',
        **kwargs,
    )


class AvailableOptionsPerception(QuestionOfRecentMemories):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class AvailableOptionsPerceptionsWithoutPreAct(
    QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers the question 'what actions are available to me?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{AVAILABLE_OPTIONS_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=AVAILABLE_OPTIONS_QUESTION,
        terminators=('\n\n',),
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerception(QuestionOfRecentMemories):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )


class CombinedPerception(QuestionOfRecentMemories):
  """This component answers the three key questions in one go."""

  def __init__(self, **kwargs):
    agent_name = '{agent_name}'
    question = f"""
Consider the following questions:
1. {SITUATION_PERCEPTION_QUESTION.format(agent_name=agent_name)}
2. {SELF_PERCEPTION_QUESTION.format(agent_name=agent_name)}
3. {PERSON_BY_SITUATION_QUESTION.format(agent_name=agent_name)}

Provide the answers to these three questions in three separate paragraphs, in order."""
    default_pre_act_label = f'\nCombined Perception for {agent_name}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=question,
        answer_prefix='',
        add_to_memory=False,
        **kwargs,
    )


class BestOptionPerceptionWithoutPreAct(QuestionOfRecentMemoriesWithoutPreAct):
  """This component answers 'which action is best for achieving my goal?'."""

  def __init__(self, **kwargs):
    default_pre_act_label = f'\n{BEST_OPTION_PERCEPTION_QUESTION}'
    if kwargs.get('pre_act_label') is None:
      kwargs['pre_act_label'] = default_pre_act_label
    super().__init__(
        question=BEST_OPTION_PERCEPTION_QUESTION,
        answer_prefix="{agent_name}'s best course of action is ",
        add_to_memory=False,
        **kwargs,
    )
