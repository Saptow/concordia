import dataclasses
from typing import List, Sequence, Optional, Dict
from concordia.hdb_simulation.entities.agents import BuyerAgent, SellerAgent

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

# Initialiser GameMaster 
class InitaliserGameMaster(
    entity_component.ContextComponent
):
    '''
    Game Master entity for initialising the simulation environment.
    '''
    def __init__(
            self,
            model,
            next_game_master_name: str, 
            buyer_agents = [],
            seller_agents = [],
    ):
        super().__init__()
        self.model = model
        self._next_game_master_name = next_game_master_name
        self._buyer_agents = buyer_agents # List of buyer agents in the simulation
        self._seller_agents = seller_agents # List of seller agents in the simulation
        self._initialized = False
    
    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        # Only respond to NEXT_GAME_MASTER queries
        if action_spec.output_type != entity_lib.OutputType.NEXT_GAME_MASTER:
            return ""
        
        if self._initialized:
            # Hand off to the dialogue/main GM
            return self._next_game_master_name
        
        # Run initialization logic (generate scene, inject observations)
        self._run_initialization()
        self._initialized = True
        
        # Return own name to let other components finish this step
        return self.get_entity().name
    
    # TODO: refine prompt for this
    def _generate_seller_scene_llm(self,seller_info: dict) -> str:
        """Generate a scene description for a seller using LLM."""
        prompt = (
            f"Generate a detailed scene description for the following seller:\n"
            f"Name: {seller_info['name']}\n"
            f"Age: {seller_info['age']}\n"
            f"Occupation: {seller_info['occupation']}\n"
            f"Flat Details: {seller_info['flat']}\n"
            f"Expectations: {seller_info['expectations']}\n"
            f"Description: {seller_info.get('description', 'N/A')}\n"
            f"Provide vivid details about the seller's personality, motivations, "
            f"to sell their flat that **ONLY** they will know. They will be negotiating with potential buyers within their own flat, allowing for showcasing its features.\n"
            f"Return the scene as a single paragraph."
        )
        response = self.model.sample_text(
            prompt=prompt
        )
        return response.text.strip()

    # TODO: refine prompt for this
    def _generate_buyer_scene_llm(self,buyer_info: dict) -> str:
        """Generate a scene description for a buyer using LLM."""
        prompt = (
            f"Generate a detailed scene description for the following buyer:\n"
            f"Name: {buyer_info['name']}\n"
            f"Age: {buyer_info['age']}\n"
            f"Occupation: {buyer_info['occupation']}\n"
            f"Budget: {buyer_info['budget']}\n"
            f"Preferences: {buyer_info['preferences']}\n"
            f"Description: {buyer_info.get('description', 'N/A')}\n"
            f"Provide vivid details about the buyer's personality, motivations, "
            f"and what they are looking for in a flat, some of which to be shared only with sellers. They will be negotiating with sellers within their flat.\n"
            f"Return the scene as a single paragraph."
        )
        response = self.model.sample_text(
            prompt=prompt
        )
        return response.text.strip()
    
    def _run_initialization(self):
        """Generate initial scene and inject into observation queues."""
        make_obs = self.get_entity().get_component("__make_observation__")
        
        # TODO: For now, all buyer and seller data are statically defined in data modules. 
        # should be dynamically generated based on hedonic modelling as well. 
        for seller_id, seller_info in self._seller_agents:
            # TODO: to create info asymmetry, partial info to both sellers and buyers. need to find a way to generate this scene. 
            scene_description = self._generate_seller_scene_llm(seller_info)
            make_obs.add_to_queue(seller_id, f"Scene: {scene_description}" )  
        
        for buyer_id, buyer_info in self._buyer_agents:
            # Buyers get a generic market scene
            scene_description = self._generate_buyer_scene_llm(buyer_info)
            make_obs.add_to_queue(buyer_id, f"Scene: {scene_description}" )
        
# GameMaster entity to manage negotiation sessions
class NegotiationGameMaster(
    entity_component.ContextComponent,
    entity_component.ComponentWithLogging
):
    '''
    Game Master entity for managing negotiation sessions between agents.
    '''
    def __init__(
            self,
            agents: Sequence[List[BuyerAgent, SellerAgent]], # list of [buyer_agent, seller_agent] pairs
            components: Sequence[str] = (),
            pre_act_label: str = "\nNegotiationComponent",
            active_deals: Optional[Dict]=None,
            history: Optional[List[Dict[str,float]]] = None
    ):
        super().__init__()
        self._agents = {
            "buyers": {
                agent.id: agent for agent, _ in agents
            },
            "sellers": {
                agent.id: agent for _, agent in agents
            }
        }
        self.active_deals = active_deals if active_deals is not None else {}
        self._components = components
        self._pre_act_label = pre_act_label
        self._state = {} # TODO: define state structure/schema for retrieval
        self.history = history if history is not None else []
        self.processed_actions= set()

    # TODO: refine logging mechanism (use logging if needed)
    def _log_self(self, event_type: str, details: str = ""):
        """Logs an event related to the experiment component."""
        log_entry = {
            "component": self.__class__.__name__,
            "entity": (
                self.get_entity().name
                if hasattr(self, "get_entity") and self.get_entity()
                else "UnknownEntity"
            ),
            "event": event_type,
            "details": details,
        }
        if hasattr(self, "_logging_channel") and self._logging_channel is not None:
            self._logging_channel(log_entry)
        else:
            print(f"{self.__class__.__name__} LOG: {log_entry}", flush=True)

    def get_pre_act_label(self) -> str:
        return self._pre_act_label
    
    # TODO: use negotiation id to track rounds
    def get_pre_act_value(self, nego_id: str) -> str:
        return f"Current round: {self._state[nego_id].get('current_round', 0)}"
    
    # TODO: implement different make observations for different agents (if need be negotiation stages)
    def _handle_make_observation(self,):
        return
    
    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Handle different output types from the simulation engine."""
        self._log_self("pre_act_called", f"ActionSpecOutputType: {action_spec.output_type}")
        output_type = action_spec.output_type
        
        if output_type == entity_lib.OutputType.MAKE_OBSERVATION:
            # check for the agent within call to action
            buyer_id, seller_id = action_spec.call_to_action  # TODO: assume id is found like this first
            buyer_agent, seller_agent = self._agents['buyers'].get(buyer_id), self._agents['sellers'].get(seller_id)
            
            return self._handle_make_observation(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_ACTION_SPEC:
            return self._handle_next_action_spec(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_ACTING:
            return self._handle_next_acting()
        elif output_type == entity_lib.OutputType.RESOLVE:
            return self._resolve(action_spec)
        elif output_type == entity_lib.OutputType.NEXT_GAME_MASTER:
            return self._handle_next_gm()
        else:
            return ""