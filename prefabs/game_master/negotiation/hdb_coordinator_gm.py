"""Coordinator game master for modular weekly market workflows."""

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from absl import logging
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import game_master as gm_components
from concordia.language_model import language_model
from concordia.prefabs.game_master.negotiation.components import hdb_coordinator_helper
from concordia.prefabs.game_master.negotiation.components import hdb_listing
from concordia.prefabs.game_master.negotiation.components import hdb_negotiation
from concordia.prefabs.game_master.negotiation.components import policy_layer
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class GameMaster(prefab_lib.Prefab):
  """Top-level weekly coordinator GM."""

  description: str = (
      'A coordinator game master that holds shared HDB market state while an '
      'external simulation engine advances concurrent weekly listing and '
      'negotiation.'
  )
  params: Mapping[str, Any] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Market Coordinator',
          'instructions': (
              'You coordinate the weekly HDB market workflow. '
              'Hold shared state across listing and negotiation, '
              'pass transfers between modules, and let the external HDB '
              'simulation engine advance each weekly tick.'
          ),
          'player_ids': (),
          'listing': {
              'buyer_profiles': {},
              'seller_profiles': {},
              'max_rounds': 0,
              'enabled': True,
          },
           'negotiation': {
               'negotiation_pairs': (),
               'action_prompt': 'What should {name} do next?',
               'participant_specs': {},
               'max_rounds': 0,
               'enabled': True,
           },
          'extra_components': {},
          'extra_components_index': {},
          'policy_layer': {
              'enabled': False,
              'policy_yaml_path': '',
              'updates_enabled': True,
          },
      }
  )
  entities: Sequence[entity_agent_with_logging.EntityAgentWithLogging] = ()

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds a minimal deterministic coordinator shell for the HDB modules."""
    extra_components = self.params.get('extra_components', {})
    extra_components_index = self.params.get('extra_components_index', {})
    if extra_components_index and extra_components:
      if extra_components_index.keys() != extra_components.keys():
        logging.error(
            'extra_components_index keys do not match extra_components keys.'
        )
        extra_components_index = {}

    name = str(self.params.get('name', 'Market Coordinator'))
    player_names = [entity.name for entity in self.entities]
    if not player_names:
      logging.error('No player entities were provided to the HDB coordinator.')
      player_names = []

    player_ids = self.params.get('player_ids') or None
    listing_params = dict(self.params.get('listing', {}))
    negotiation_params = dict(self.params.get('negotiation', {}))
    policy_layer_params = dict(self.params.get('policy_layer', {}))

    make_observation_key = (
        gm_components.make_observation.DEFAULT_MAKE_OBSERVATION_COMPONENT_KEY
    )
    make_observation = gm_components.make_observation.MakeObservation(
        model=model,
        player_names=player_names,
        components=[],
        allow_llm_fallback=False,
    )

    listing_module_key = 'listing_module'
    listing_module = hdb_listing.ListingModule(
        player_names=player_names,
        player_ids=player_ids,
        buyer_profiles=listing_params.get('buyer_profiles', {}),
        seller_profiles=listing_params.get('seller_profiles', {}),
        client=listing_params.get('client'),
        dense_embedding_model=listing_params.get('dense_embedding_model'),
        sparse_embedding_model=listing_params.get('sparse_embedding_model'),
        collection_name=listing_params.get('collection_name'),
        db_path=listing_params.get('db_path'),
        random_seed=int(listing_params.get('random_seed', 0) or 0),
        max_rounds=int(listing_params.get('max_rounds', 0) or 0) or None,
        seller_listing_max_workers=int(
            listing_params.get('seller_listing_max_workers', 1) or 1
        ),
        buyer_search_max_workers=int(
            listing_params.get('buyer_search_max_workers', 1) or 1
        ),
        seller_review_max_workers=int(
            listing_params.get('seller_review_max_workers', 1) or 1
        ),
        enabled=bool(listing_params.get('enabled', True)),
    )
    listing_module.set_canonical_entities(self.entities)

    negotiation_module_key = 'negotiation_module'
    negotiation_module = hdb_negotiation.NegotiationModule(
        entities=self.entities,
        participant_specs=negotiation_params.get('participant_specs', {}),
        # Preserve an explicit empty sequence so the scheduler starts with no
        # pairs and waits for initializer/listing handoff via pending matches.
        negotiation_pairs=negotiation_params.get('negotiation_pairs'),
        action_prompt=str(
            negotiation_params.get('action_prompt', 'What should {name} do next?')
        ),
        # The main HDB workflow does not use a scheduler-level negotiation cap.
        # Pairs should exit through explicit outcomes, especially buyer WALK_AWAY.
        max_rounds=0,
        max_weeks_open=int(
            negotiation_params.get('max_weeks_open', 0) or 0
        ),
        pair_max_workers=int(
            negotiation_params.get('pair_max_workers', 1) or 1
        ),
        enabled=bool(negotiation_params.get('enabled', True)),
        make_observation_component_key=make_observation_key,
    )

    coordinator_state_key = 'weekly_coordinator'
    coordinator_state = hdb_coordinator_helper.WeeklyCoordinator(
        player_ids=tuple(player_ids) if player_ids else (),
        player_names=tuple(player_names),
        listing_component_key=listing_module_key,
        negotiation_component_key=negotiation_module_key,
    )

    policy_layer_key = 'policy_layer'
    gm_policy_layer = policy_layer.PolicyLayerComponent(
        policy_yaml_path=str(policy_layer_params.get('policy_yaml_path', '')),
        model=model,
        updates_enabled=bool(policy_layer_params.get('updates_enabled', True)),
        enabled=bool(policy_layer_params.get('enabled', False)),
    )

    components_of_game_master = {
        make_observation_key: make_observation,
        policy_layer_key: gm_policy_layer,
        listing_module_key: listing_module,
        negotiation_module_key: negotiation_module,
        coordinator_state_key: coordinator_state,
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
