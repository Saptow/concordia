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

"""A lightweight HDB negotiation game master focused on turn scheduling."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.prefabs.game_master.negotiation.components import hdb_negotiation_state
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """Prefab for a scheduler-first negotiation game master.

  This GM is intentionally minimal:
  - deterministic next-actor scheduling by pair and round
  - fixed next-action-spec generation (free or choice)
  - simple pass-through event resolution
  - compact scheduler state tracker for prompt context
  """

  description: str = (
      'A lightweight game master that enforces negotiation turn order.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'HDB Negotiation Scheduler',
          'instructions': (
              'You are the game master for an HDB negotiation simulation. '
              'Your primary responsibility is to enforce turn order and keep '
              'the simulation flow consistent.'
          ),
          # If omitted, pairs are inferred from player order in consecutive
          # chunks of two: (0,1), (2,3), ...
          'negotiation_pairs': (),
          # Optional IDs aligned with entities; if provided, pair entries can
          # reference IDs instead of names.
          'player_ids': (),
          # Action spec behavior for the active entity.
          'action_mode': 'choice',  # "free" or "choice"
          'action_prompt': 'What should {name} do next?',
          'action_options': (),
          # Kept for compatibility with prior callers.
          'max_rounds': 0,
          'extra_components': {},
          'extra_components_index': {},
      }
  )
  entities: (
      Sequence[entity_agent_with_logging.EntityAgentWithLogging]
  ) = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds the scheduler-first negotiation game master."""
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})
    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as extra_components.'
        )

    name = str(self.params.get('name', 'HDB Negotiation Scheduler'))
    custom_instructions = self.params.get('instructions')
    player_names = [entity.name for entity in self.entities]
    if not player_names:
      raise ValueError('No player entities were provided to the game master.')

    negotiation_pairs = self.params.get('negotiation_pairs') or None
    player_ids = self.params.get('player_ids') or None
    max_rounds = int(self.params.get('max_rounds', 0) or 0)

    action_mode = str(self.params.get('action_mode', 'choice')).strip().lower()
    action_prompt = str(self.params.get('action_prompt', 'What should {name} do next?'))
    action_options = self.params.get('action_options', ())
    if isinstance(action_options, str):
      action_options = [opt.strip() for opt in action_options.split(',') if opt.strip()]

    # Core GM components.
    instructions_key = 'instructions'
    instructions = gm_components.instructions.Instructions()
    if custom_instructions is not None:
      if isinstance(custom_instructions, Mapping):
        instructions.set_state(custom_instructions)
      else:
        instructions.set_state({'state': str(custom_instructions)})

    player_characters_key = 'player_characters'
    player_characters = gm_components.instructions.PlayerCharacters(
        player_characters=player_names,
    )

    memory_component_key = actor_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory_component = actor_components.memory.AssociativeMemory(
        memory_bank=memory_bank
    )

    observation_to_memory_key = 'observation_to_memory'
    observation_to_memory = actor_components.observation.ObservationToMemory()

    observation_component_key = (
        actor_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = actor_components.observation.LastNObservations(history_length=200)

    # Turn scheduler and scheduler-state view.
    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = hdb_negotiation_state.PairRoundRobinNextActing(
        model=model,
        player_names=player_names,
        negotiation_pairs=negotiation_pairs,
        player_ids=player_ids,
        max_rounds=max_rounds if max_rounds > 0 else None,
    )

    scheduler_state_key = 'turn_order_state'
    scheduler_state = hdb_negotiation_state.TurnOrderStateTracker(
        scheduler_component_key=next_actor_key,
    )
    offer_state_key = 'pair_offer_state'
    offer_state = hdb_negotiation_state.PairActiveOfferTracker(
        scheduler_component_key=next_actor_key,
    )

    display_events_key = 'display_events'
    display_events = gm_components.event_resolution.DisplayEvents(
        model=model,
        pre_act_label='Resolved events',
    )

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        # Keep entity-facing observations strictly event-queue driven to avoid
        # leaking scheduler/global state across independent negotiation pairs.
        components=[],
        # Resolution queues exact events to players and disables free-form
        # fallback when queue is empty to preserve pair isolation.
        allow_llm_fallback=False,
    )

    # Deterministic next action spec (no LLM decision needed here).
    next_action_spec_key = gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    next_action_spec = hdb_negotiation_state.FixedNextActionSpec(
        action_mode=action_mode,
        call_to_action=action_prompt,
        choice_options=tuple(action_options),
        next_acting_component_key=next_actor_key,
        offer_tracker_component_key=offer_state_key,
    )

    # Pass-through event resolution to ensure processing of observations and no information leakage across pairs. 
    event_resolution_key = gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    event_resolution = hdb_negotiation_state.PassthroughResolution(
        memory_component_key=memory_component_key,
        make_observation_component_key=make_observation_key,
        offer_tracker_component_key=offer_state_key,
        notify_players=True,
    )

    # Keep termination explicit and controlled by outer loop max_steps or
    # external conditions.
    terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate_component = hdb_negotiation_state.TerminateWhenAllPairsClosed(
        offer_tracker_component_key=offer_state_key,
        scheduler_component_key=next_actor_key,
    )

    components_of_game_master = {
        instructions_key: instructions,
        player_characters_key: player_characters,
        memory_component_key: memory_component,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        scheduler_state_key: scheduler_state,
        offer_state_key: offer_state,
        display_events_key: display_events,
        make_observation_key: make_observation,
        next_actor_key: next_actor,
        next_action_spec_key: next_action_spec,
        event_resolution_key: event_resolution,
        terminate_key: terminate_component,
    }

    component_order = list(components_of_game_master.keys())
    if extra_components:
      components_of_game_master.update(extra_components)
      if extra_components_index:
        for component_name in extra_components:
          component_order.insert(
              extra_components_index[component_name],
              component_name,
          )
      else:
        component_order = list(components_of_game_master.keys())

    act_component = gm_components.switch_act.SwitchAct(
        model=model,
        entity_names=player_names,
        component_order=component_order,
    )
    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=name,
        act_component=act_component,
        context_components=components_of_game_master,
    )
