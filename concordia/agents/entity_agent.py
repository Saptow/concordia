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

"""A modular entity agent using the new component system."""

import dataclasses
from collections.abc import Mapping, Sequence
from concurrent import futures
import functools
import threading
import traceback
import types
from typing import cast, override

from absl import logging
from concordia.components.agent import no_op_context_processor
from concordia.typing import entity
from concordia.typing import entity_component
from concordia.utils import concurrency

# TODO: b/313715068 - remove disable once pytype bug is fixed.
# pytype: disable=override-error


@dataclasses.dataclass(frozen=True)
class PreparedAct:
  """Prepared act state held between PRE_ACT and POST_ACT."""

  action_spec: entity.ActionSpec
  contexts: entity_component.ComponentContextMapping


class EntityAgent(entity_component.EntityWithComponents):
  """An agent that has its functionality defined by components.

  The agent has a set of components that define its functionality. The agent
  must have at least an ActComponent and an ObserveComponent. The agent will
  call the ActComponent's `act` method when it needs to act, and the
  ObservationComponent's `observe` method when they need to process an
  observation.
  """

  def __init__(
      self,
      agent_name: str,
      act_component: entity_component.ActingComponent,
      context_processor: (
          entity_component.ContextProcessorComponent | None
      ) = None,
      context_components: Mapping[str, entity_component.ContextComponent] = (
          types.MappingProxyType({})
      ),
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_processor: The component that will be used to process contexts. If
        None, a NoOpContextProcessor will be used.
      context_components: The ContextComponents that will be used by the agent.
    """
    super().__init__()
    self._agent_name = agent_name
    self._control_lock = threading.Lock()
    self._phase_lock = threading.Lock()
    self._phase = entity_component.Phase.READY

    self._act_component = act_component
    self._act_component.set_entity(self)

    if context_processor is None:
      self._context_processor = no_op_context_processor.NoOpContextProcessor()
    else:
      self._context_processor = context_processor
    self._context_processor.set_entity(self)

    self._context_components = dict(context_components)
    for component in self._context_components.values():
      component.set_entity(self)

  @override
  @functools.cached_property
  def name(self) -> str:
    return self._agent_name

  @override
  def get_phase(self) -> entity_component.Phase:
    with self._phase_lock:
      return self._phase

  def _set_phase(self, phase: entity_component.Phase) -> None:
    with self._phase_lock:
      self._phase.check_successor(phase)
      self._phase = phase

  @override
  def get_component(
      self,
      name: str,
      *,
      type_: type[entity_component.ComponentT] = entity_component.BaseComponent,
  ) -> entity_component.ComponentT:
    component = self._context_components[name]
    return cast(entity_component.ComponentT, component)

  def get_act_component(self) -> entity_component.ActingComponent:
    return self._act_component

  def get_all_context_components(
      self,
  ) -> Mapping[str, entity_component.ContextComponent]:
    return types.MappingProxyType(self._context_components)

  def _parallel_call_(
      self,
      method_name: str,
      *args,
      executor: futures.ThreadPoolExecutor | None = None,
  ) -> entity_component.ComponentContextMapping:
    """Calls the named method in parallel on all components.

    If a component instance is registered under multiple names, its method
    will only be called once. The result of that call will be mapped to all
    names under which it was registered.

    All calls will be issued with the same payloads.

    Args:
      method_name: The name of the method to call.
      *args: The arguments to pass to the method.
      executor: An optional existing ThreadPoolExecutor to use.

    Returns:
      A ComponentsContext, that is, a mapping of component name to the result of
      the method call.
    """
    # 1. Identify unique component instances.
    unique_components = list(set(self._context_components.values()))

    # 2. Create and execute tasks for each unique component instance once.
    tasks_for_unique = {
        str(id(component)): functools.partial(
            getattr(component, method_name), *args
        )
        for component in unique_components
    }
    results_by_component_id = concurrency.run_tasks(
        tasks_for_unique, executor=executor
    )

    # 3. Construct the final results dictionary.
    final_results: dict[str, str] = {}
    for name, component in self._context_components.items():
      final_results[name] = results_by_component_id[str(id(component))]

    return types.MappingProxyType(final_results)

  def _parallel_call_filtered_(
      self,
      method_name: str,
      *args,
      excluded_component_names: Sequence[str] = (),
      executor: futures.ThreadPoolExecutor | None = None,
  ) -> entity_component.ComponentContextMapping:
    """Calls the named method on all non-excluded components in parallel."""
    excluded_names = {str(name) for name in excluded_component_names if str(name)}
    if not excluded_names:
      return self._parallel_call_(method_name, *args, executor=executor)

    filtered_components = {
        name: component
        for name, component in self._context_components.items()
        if name not in excluded_names
    }
    unique_components = list(set(filtered_components.values()))
    tasks_for_unique = {
        str(id(component)): functools.partial(
            getattr(component, method_name), *args
        )
        for component in unique_components
    }
    results_by_component_id = concurrency.run_tasks(
        tasks_for_unique, executor=executor
    )

    final_results: dict[str, str] = {}
    for name, component in filtered_components.items():
      final_results[name] = results_by_component_id[str(id(component))]

    return types.MappingProxyType(final_results)

  @override
  def act(
      self, action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC
  ) -> str:
    prepared_act = self.prepare_act(action_spec)
    return self.finalize_prepared_act(prepared_act)

  def prepare_act(
      self,
      action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC,
  ) -> PreparedAct:
    """Prepare an action and hold the agent in PRE_ACT until finalized."""
    self._control_lock.acquire()
    try:
      self._set_phase(entity_component.Phase.PRE_ACT)
      contexts = self._parallel_call_('pre_act', action_spec)
      contexts = types.MappingProxyType(contexts)
      self._context_processor.pre_act(contexts)
      return PreparedAct(action_spec=action_spec, contexts=contexts)
    except Exception:
      self.set_phase(entity_component.Phase.READY)
      self._control_lock.release()
      raise

  def prepare_act_with_deferred_components(
      self,
      action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC,
      *,
      deferred_component_names: Sequence[str] = (),
  ) -> PreparedAct:
    """Prepare an action while deferring selected context components."""
    deferred_names = tuple(
        str(name) for name in deferred_component_names if str(name)
    )
    if not deferred_names:
      return self.prepare_act(action_spec)

    self._control_lock.acquire()
    try:
      self._set_phase(entity_component.Phase.PRE_ACT)
      contexts = self._parallel_call_filtered_(
          'pre_act',
          action_spec,
          excluded_component_names=deferred_names,
      )
      contexts = types.MappingProxyType(contexts)
      self._context_processor.pre_act(contexts)
      return PreparedAct(action_spec=action_spec, contexts=contexts)
    except Exception:
      self.set_phase(entity_component.Phase.READY)
      self._control_lock.release()
      raise

  def finalize_prepared_act(
      self,
      prepared_act: PreparedAct,
      action_attempt: str | None = None,
  ) -> str:
    """Finalize a prepared action, running POST_ACT and UPDATE."""
    try:
      if action_attempt is None:
        action_attempt = self._act_component.get_action_attempt(
            prepared_act.contexts,
            prepared_act.action_spec,
        )

      self._set_phase(entity_component.Phase.POST_ACT)
      contexts = self._parallel_call_('post_act', action_attempt)
      self._context_processor.post_act(contexts)

      self._set_phase(entity_component.Phase.UPDATE)
      self._parallel_call_('update')

      self._set_phase(entity_component.Phase.READY)
      return action_attempt
    except Exception:
      self.set_phase(entity_component.Phase.READY)
      raise
    finally:
      self._control_lock.release()

  def cancel_prepared_act(self, prepared_act: PreparedAct) -> None:
    """Abort a prepared action and return the agent to READY."""
    del prepared_act
    self.set_phase(entity_component.Phase.READY)
    self._control_lock.release()

  @override
  def observe(self, observation: str) -> None:
    with self._control_lock:
      try:
        self._set_phase(entity_component.Phase.PRE_OBSERVE)
        contexts = self._parallel_call_('pre_observe', observation)
        self._context_processor.pre_observe(contexts)

        self._set_phase(entity_component.Phase.POST_OBSERVE)
        contexts = self._parallel_call_('post_observe')
        self._context_processor.post_observe(contexts)

        self._set_phase(entity_component.Phase.UPDATE)
        self._parallel_call_('update')

        self._set_phase(entity_component.Phase.READY)
      except Exception:
        # Ensure correct error handling in the case of multiple threads
        # using the same entity by setting the phase to ready before raising.
        self.set_phase(entity_component.Phase.READY)
        raise

  def set_state(
      self, entity_components_state: entity_component.EntityState
  ) -> None:
    """Sets the state of the agent."""

    # Restore context components
    context_components_state = entity_components_state.get(
        'context_components', {}
    )
    for component_name, component in self._context_components.items():
      if component_name in context_components_state:
        try:
          component.set_state(context_components_state[component_name])
        except Exception:  # pylint: disable=broad-exception-caught
          logging.error(
              'Error setting state for component %s: %s',
              component_name, traceback.format_exc()
          )

    # Restore act component
    act_state = entity_components_state.get('act_component')
    if act_state:
      try:
        self._act_component.set_state(act_state)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.error(
            'Error setting state for act component: %s', traceback.format_exc()
        )

    # Restore context processor
    proc_state = entity_components_state.get('context_processor')
    if proc_state:
      try:
        self._context_processor.set_state(proc_state)
      except Exception:  # pylint: disable=broad-exception-caught
        logging.error(
            'Error setting state for context processor: %s',
            traceback.format_exc()
        )

  def get_state(self) -> entity_component.EntityState:
    """Returns the state of the agent as a dictionary."""
    return {
        'act_component': self._act_component.get_state(),
        'context_processor': self._context_processor.get_state(),
        'context_components': {
            component_name: component.get_state()
            for component_name, component in self._context_components.items()
        },
    }

  def set_phase(self, phase: entity_component.Phase) -> None:
    with self._phase_lock:
      self._phase = phase

  def stateless_act(
      self,
      action_spec: entity.ActionSpec,
  ) -> str:
    """Helper for single stateless action, used by parallel_stateless_act."""
    logging.warning('stateless_act is deprecated. Please use act instead.')

    if self.get_phase() != entity_component.Phase.PRE_ACT:
      raise RuntimeError('Agent must be in PRE_ACT phase for stateless_act')

    # 1. PRE_ACT to gather context
    executor = futures.ThreadPoolExecutor()
    contexts = self._parallel_call_('pre_act', action_spec, executor=executor)
    executor.shutdown(wait=True)
    self._context_processor.pre_act(types.MappingProxyType(contexts))

    # 2. Get action from ActComponent
    action_attempt = self._act_component.get_action_attempt(
        contexts, action_spec
    )
    return action_attempt
