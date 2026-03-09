"""A lightweight HDB listing-portal game master focused on weekly batch actions."""

from collections.abc import Mapping, Sequence
import dataclasses
import json
from typing import Any

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as actor_components
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.concordia.prefabs.game_master.negotiation.components import hdb_listing_gm
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """Prefab for the HDB listing portal workflow."""

  description: str = (
      'A lightweight game master that manages weekly listing-portal batches. '
      'Use this prefab with Concordia\'s simultaneous engine.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'HDB Listing Portal Scheduler',
          'instructions': (
              'You are the game master for the HDB listing portal stage. '
              'Your primary responsibility is to execute weekly market batches, '
              'track listings, and hand off matched pairs into negotiation. '
              'This workflow assumes all open listing participants act in the '
              'same simulated week.'
          ),
          'player_ids': (),
          'action_mode': 'choice',
          'action_prompt': 'Acknowledge the weekly listing-portal batch step.',
          'buyer_profiles': {},
          'seller_profiles': {},
          'max_rounds': 0,
          'extra_components': {},
          'extra_components_index': {},
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})
    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        raise ValueError(
            'extra_components_index must have the same keys as extra_components.'
        )

    name = str(self.params.get('name', 'HDB Listing Portal Scheduler'))
    custom_instructions = self.params.get('instructions')
    player_names = [entity.name for entity in self.entities]
    if not player_names:
      raise ValueError('No player entities were provided to the game master.')

    player_ids = self.params.get('player_ids') or None
    max_rounds = int(self.params.get('max_rounds', 0) or 0)
    action_prompt = str(
        self.params.get(
            'action_prompt', 'Acknowledge the weekly listing-portal batch step.'
        )
    )
    buyer_profiles = self.params.get('buyer_profiles', {})
    seller_profiles = self.params.get('seller_profiles', {})
    if isinstance(buyer_profiles, str):
      buyer_profiles = json.loads(buyer_profiles) if buyer_profiles else {}
    if isinstance(seller_profiles, str):
      seller_profiles = json.loads(seller_profiles) if seller_profiles else {}

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
        components=[],
        allow_llm_fallback=False,
    )

    next_actor_key = gm_components.next_acting.DEFAULT_NEXT_ACTING_COMPONENT_KEY
    next_actor = hdb_listing_gm.ListingBatchScheduler(
        model=model,
        player_names=player_names,
        player_ids=player_ids,
        max_rounds=max_rounds if max_rounds > 0 else None,
    )

    scheduler_state_key = 'week_state'
    scheduler_state = hdb_listing_gm.PortalWeekStateTracker(
        scheduler_component_key=next_actor_key,
    )

    portal_state_key = 'listing_portal_state'
    portal_state = hdb_listing_gm.ListingPortalTracker(
        buyer_profiles=buyer_profiles,
        seller_profiles=seller_profiles,
        scheduler_component_key=next_actor_key,
    )

    next_action_spec_key = gm_components.next_acting.DEFAULT_NEXT_ACTION_SPEC_COMPONENT_KEY
    next_action_spec = hdb_listing_gm.PortalBatchActionSpec(
        call_to_action=action_prompt,
    )

    event_resolution_key = gm_components.switch_act.DEFAULT_RESOLUTION_COMPONENT_KEY
    event_resolution = hdb_listing_gm.PortalBatchResolution(
        make_observation_component_key=make_observation_key,
        portal_tracker_component_key=portal_state_key,
    )

    terminate_key = gm_components.terminate.DEFAULT_TERMINATE_COMPONENT_KEY
    terminate_component = hdb_listing_gm.TerminateWhenPortalClosed(
        portal_tracker_component_key=portal_state_key,
        scheduler_component_key=next_actor_key,
    )

    components_of_game_master = {
        instructions_key: instructions,
        player_characters_key: player_characters,
        memory_component_key: memory_component,
        observation_to_memory_key: observation_to_memory,
        observation_component_key: observation,
        scheduler_state_key: scheduler_state,
        portal_state_key: portal_state,
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
