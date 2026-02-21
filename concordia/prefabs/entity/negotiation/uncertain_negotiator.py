"""Base negotiation agent prefab with core negotiation capabilities."""

from collections.abc import Mapping
import dataclasses
import json

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.concordia.prefabs.entity.negotiation.components import hdb_acting_component
from concordia.prefabs.entity.negotiation.components import uncertain_buyer, uncertain_seller
from concordia.hdb_simulation.models.schemas import BuyerActions, RoleType, SellerActions
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Import our negotiation components
from concordia.prefabs.entity.negotiation.components import negotiation_memory
from concordia.prefabs.entity.negotiation.components import hdb_negotiation_instructions
from concordia.prefabs.entity.negotiation.components import hdb_negotiation_strategy
DEFAULT_ETHICS = (
    f'HDB RESALE ETHICAL CONSTRAINTS (LAW/REG-ALIGNED)'
    f'- Do not fabricate or misrepresent any material fact (offers, deadlines, valuation/COV, approvals, eligibility, defects, inclusions).'
    f'- You may withhold private limits (true budget/reservation/urgency), but if you answer a factual question, answer truthfully or say you are unsure.'
    f'- Do not propose or encourage terms that contradict HDB’s prescribed OTP/deposit framework or any side payments; keep commitments realistic within HDB timelines.'
    f'- Treat inferences as hypotheses; ask clarifying questions instead of asserting unverified claims.'
    f'- No coercion/harassment or exploitation of vulnerability; keep a clear written record of offers and key terms.'
)
@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
    """
    A custom negotiation agent for HDB resale negotiations.
    """

    description: str = (
        'A negotiation agent with core capabilities for engaging in '
        'value-based negotiations.'
        'Current features include:'
        '- Uncertainty modeling of own and opponent preferences (with information asymmetry embedded in the uncertainty)'
    )

    params: Mapping[str, str] = dataclasses.field(default_factory=lambda: {
        'name': 'Negotiator',
        'description': 'Reach a mutually beneficial agreement',
        'negotiation_style': 'competitive',
        'reservation_value': '0.0',
        'ethical_constraints': DEFAULT_ETHICS,
        'modules': '', # e.g. uncertainty_buyer, uncertainty_seller
        'modules_config': '', # e.g. config parameters for the modules in JSON format
        'extra_components': {},
    })

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the base negotiation agent.

        Args:
            model: Language model for reasoning
            memory_bank: Memory bank for storing experiences

        Returns:
            Configured negotiation agent
        """
        # Extract parameters from params dict
        agent_name = self.params.get('name', 'Negotiator')
        description = self.params.get('description', '')
        style = self.params.get('negotiation_style', 'competitive')
        reservation = float(self.params.get('reservation_value', '0.0'))
        ethics = self.params.get('ethical_constraints', DEFAULT_ETHICS)
        # TODO: revise the ethical constraints based on HDB negotiation context

        # Parse module configurations
        modules_str = self.params.get('modules', '')
        modules = [m.strip() for m in modules_str.strip('[]').split(',')] if modules_str else []

        # Parse module configs from JSON
        module_configs_str = self.params.get('modules_config', '')
        try: 
            module_configs = json.loads(module_configs_str) if module_configs_str else {}
        except json.JSONDecodeError:
            module_configs = {}
        
        if 'uncertain_buyer' in modules:
            role = RoleType.BUYER
        elif 'uncertain_seller' in modules:
            role = RoleType.SELLER

        # Create memory component
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank,
        )

        # Create observation component
        observation_to_memory = agent_components.observation.ObservationToMemory()

        # Create observation retrieval component (pulls most recent N observations)
        observation = agent_components.observation.LastNObservations(
            history_length=10, # TODO: make this adaptive based on personality metadata
            pre_act_label='Recent events in the negotiation:'
        )

        # Create negotiation-specific instructions
        #TODO: add context-specific instructions here based on flat metadata (i.e. constraints what not)
        instructions = hdb_negotiation_instructions.HDBNegotiationInstructions(
            agent_name=agent_name,
            role = role,
            description=description,
            negotiation_style=style,
            reservation_value=reservation,
            ethical_constraints=ethics,
            verbose=True,
        )

        # Create negotiation memory
        neg_memory = negotiation_memory.NegotiationMemory(
            agent_name=agent_name,
            memory_bank=memory_bank,
            verbose=True,
        )
        # Setup uncertainty context if specified (should only be one of buyer/seller)
        # TODO: revise negotiation strategy to include strategy evolution based on past failed negotiations, if any
        uncertain_key, uncertain_context = None, None
        strategy_key = 'NegotiationStrategy'
        if 'uncertain_buyer' in modules:
            uncertain_key = 'uncertain_buyer'
            uncertain_configs = module_configs.get('uncertain_buyer', {})
            uncertain_context = uncertain_buyer.UncertainBuyer(
                model=model,
                confidence =uncertain_configs.get('confidence', 0.5), # TODO: determine based on personality metadata
                risk_tolerance=uncertain_configs.get('risk_tolerance', 0.5), # TODO: determine based on personality metadata
                information_gathering_budget=uncertain_configs.get('information_gathering_budget', 0.1), # TODO: determine based on personality metadata
                preferences=uncertain_configs.get('preferences', {}),
                own_reservation_=uncertain_configs.get('own_reservation_', 0.0),
                own_reservation_std=uncertain_configs.get('own_reservation_std', 1000.0),
                lambda_=uncertain_configs.get('lambda_', 1.0),
                a=uncertain_configs.get('a', 3.0),
                b=uncertain_configs.get('b', 5000.0),
            )
            strategy = hdb_negotiation_strategy.HDBNegotiationStrategy(
                model=model,
                agent_name=agent_name,
                role=role,
                uncertain_context=uncertain_context,
                description=description,
                verbose=True,
            )

        elif 'uncertain_seller' in modules:
            uncertain_key = 'uncertain_seller'
            uncertain_configs = module_configs.get('uncertain_seller', {})
            uncertain_context = uncertain_seller.UncertainSeller(
                model=model,
                confidence=uncertain_configs.get('confidence', 0.5), # TODO: determine based on personality metadata
                risk_tolerance=uncertain_configs.get('risk_tolerance', 0.5),
                information_gathering_budget=uncertain_configs.get('information_gathering_budget', 0.1), # TODO: determine based on personality metadata
                own_reservation_=uncertain_configs.get('own_reservation_', 0.0),
                lambda_=uncertain_configs.get('lambda_', 1.0),
                a=uncertain_configs.get('a', 3.0),
                b=uncertain_configs.get('b', 5000.0),
            )
            strategy = hdb_negotiation_strategy.HDBNegotiationStrategy(
                model=model,
                agent_name=agent_name,
                role=role,
                uncertain_context=uncertain_context,
                description=description,
                verbose=True,
            )
    
        question_about_self = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Self-perception as {role}:',
            question=(f'Agent description: {description}\n'
            f'What kind of {role} is {agent_name}? Respond in 1-5 sentences.'
            ),
            answer_prefix=f'{agent_name} is a {role} who ',
            add_to_memory=False,
            memory_tag='[self perception]',
        )

        # Create question components for context and reasoning
        question_about_situation = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Current negotiation situation:',
            question=f'What is the current negotiation situation that {agent_name} is in? Respond in 1-5 sentences.',
            answer_prefix=f'{agent_name} is currently ',
            add_to_memory=False,
            memory_tag='[situation perception]',
            components = [uncertain_key, neg_memory.name]
        )

        question_about_action = agent_components.question_of_recent_memories.QuestionOfRecentMemoriesStructured(
            model=model,
            pre_act_label=f'Next action',
            question=f'Given the negotiation context, what should {agent_name} do?',
            answer_prefix=f'',
            add_to_memory=False,
            memory_tag='[action reasoning]',
            output_schema=BuyerActions if role == RoleType.BUYER else SellerActions,
            components = ['situation_perception', 'self_perception', strategy_key]
        )

        # # Recent memories for context 
        # #TODO: no need for now, since this agent is purely for negotiation purposes; introduce back if needed
        # recent_memories = agent_components.all_similar_memories.AllSimilarMemories(
        #     model=model,
        #     num_memories_to_retrieve=10,
        # )

        # TODO: look into more refined strategy integration on later stage
        # Assemble all components
        components_of_agent = {
            'observation_to_memory': observation_to_memory,
            'observation': observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            instructions.name: instructions,
            neg_memory.name: neg_memory,
            uncertain_key: uncertain_context,
            strategy_key: strategy,
            'situation_perception': question_about_situation,
            'self_perception': question_about_self,
            'action_reasoning': question_about_action,
        }

        # Add any extra components
        extra_components = self.params.get('extra_components', {})
        if isinstance(extra_components, dict):
            components_of_agent.update(extra_components)

        # Define component order for context building
        component_order = [
            'observation_to_memory',
            'observation',
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            instructions.name,
            neg_memory.name,
            uncertain_key,
            strategy_key,
            'situation_perception',
            'self_perception',
            'action_reasoning',
        ]

        # Add extra component names to order
        if isinstance(extra_components, dict):
            component_order.extend([
                name for name in extra_components.keys()
                if name not in component_order
            ])

        # Custom acting component for uncertain negotiator that can handle structured outputs for actions
        act_component = hdb_acting_component.HDBStructuredActComponent(
            model=model,
            role=role,  # RoleType.BUYER or RoleType.SELLER
            structured_component_key='action_reasoning',
            component_order=component_order,
            fallback_to_llm_for_free=False,
        )
                # Create the agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components_of_agent,
        )

        return agent

def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    params: Mapping[str, str],
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build the uncertain negotiator agent."""
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)