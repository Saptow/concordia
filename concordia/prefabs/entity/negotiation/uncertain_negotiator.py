"""Base negotiation agent prefab with core negotiation capabilities."""

from collections.abc import Mapping
import dataclasses
import json

from configs import NegotiationComponentConfig
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import hdb_acting_component
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.prefabs.entity.negotiation.components import (
    uncertain_buyer,
    uncertain_seller,
)
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Import our negotiation components
from concordia.prefabs.entity.negotiation.components import negotiation_memory
from concordia.prefabs.entity.negotiation.components import hdb_negotiation_instructions
from concordia.prefabs.entity.negotiation.components import hdb_policy_tool_prompt
from concordia.prefabs.entity.negotiation.components import hdb_negotiation_strategy
DEFAULT_ETHICS = ( # TODO: refine this to align more with HDB resale context.
    f'HDB RESALE ETHICAL CONSTRAINTS: \n'
    f'- Do NOT fabricate or misrepresent any material fact (offers, deadlines, valuation/COV, approvals, eligibility, defects, inclusions, nearby amenities etc.).\n'
    f'- If you are unsure about a fact, say you are unsure instead of guessing or making up an answer.\n'
    f'- You may withhold private limits (true budget/reservation/urgency), but if you answer a factual question, answer truthfully or say you are unsure.\n'
    f'- Do not propose or encourage terms that contradict HDB’s prescribed OTP/deposit framework or any side payments; keep commitments realistic within HDB timelines.\n'
    f'- Treat inferences as hypotheses; ask clarifying questions instead of asserting unverified claims.\n'
    f'- No coercion/harassment or exploitation of vulnerability; keep a clear written record of offers and key terms.'
)

HDB_ACTION_CHOICE_GUARDRAILS = (
    "ACTION-CHOICE GUARDRAILS:\n"
    "- Follow **Negotiation Strategy State and Numeric Facts** guidance on action choice.\n"
    "- If **Strategy Summary** indicates information gathering and you have budget for it, but there is an ACTIVE OFFER, DO NOT MAKE_COUNTEROFFER; choose REJECT_OFFER and then proceed with information gathering.\n"
    "- If **Strategy Summary** has [IMPORTANT] tag, priortise following that guidance over ALL other strategy guidance."
)
HDB_CONTEXT_ANCHOR = (
    "NOTE:\n"
    "- You are in an HDB resale negotiation for exactly one flat in Singapore.\n"
    "- Ignore and discard ANY off-domain prior context.\n"
    "- TIME RULE: 1 completed negotiation round (buyer turn + seller turn) = 1 week of in-simulation time.\n"
    "- ALL pricing and monetary references in SGD.\n"
)
ACTION_REASONING_MEMORY_WINDOW = 6


def _escape_format_braces(text: str) -> str:
    """Escape braces so downstream `.format(agent_name=...)` stays safe."""
    return str(text).replace('{', '{{').replace('}', '}}')


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
        'reservation_value': '0.0',
        'flat_listing': '',
        'ethical_constraints': DEFAULT_ETHICS,
        'modules': '', # e.g. uncertainty_buyer, uncertainty_seller
        'modules_config': '', # e.g. config parameters for the modules in JSON format
        'negotiation_config': '', # e.g. config parameters for negotiation strategy and instructions in JSON format
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
        negotiation_config = self.params.get('negotiation_config', {})
        if not isinstance(negotiation_config, Mapping):
            negotiation_config = {}

        agent_name = self.params.get('name', 'Negotiator')
        description = self.params.get('description', '')
        reservation = float(
            negotiation_config.get(
                'reservation_value',
                negotiation_config.get('own_reservation_', '0.0'),
            )
        )
        flat_listing_raw = negotiation_config.get('flat_listing', '')
        ethics = self.params.get('ethical_constraints', DEFAULT_ETHICS)
        # TODO: revise the ethical constraints based on HDB negotiation context
        try:
            flat_listing = json.loads(flat_listing_raw) if flat_listing_raw else {}
            if not isinstance(flat_listing, dict):
                flat_listing = {}
        except json.JSONDecodeError:
            flat_listing = {}

        role_raw = str(self.params.get('role', '')).strip().lower()
        if role_raw == common_schemas.RoleType.BUYER.value:
            role = common_schemas.RoleType.BUYER
        elif role_raw == common_schemas.RoleType.SELLER.value:
            role = common_schemas.RoleType.SELLER
        else:
            logging.error('Unable to determine negotiator role for %s.', agent_name)
            role = common_schemas.RoleType.BUYER
        buyer_preferences = {}
        if role == common_schemas.RoleType.BUYER:
            raw_preferences = self.params.get('preferences', {})
            if isinstance(raw_preferences, dict):
                buyer_preferences = raw_preferences

        # Create memory component
        memory = agent_components.memory.AssociativeMemory(
            memory_bank=memory_bank,
        )

        # Create observation component
        observation_to_memory = agent_components.observation.ObservationToMemory()

        # Only expose observations received since the last action so each step
        # reflects the current week's incoming negotiation signal.
        observation = agent_components.observation.ObservationsSinceLastPreAct(
            pre_act_label='# RECENT OBSERVATIONS',
        )

        # Create negotiation-specific instructions
        #TODO: add context-specific instructions here based on flat metadata (i.e. constraints what not)
        instructions = hdb_negotiation_instructions.HDBNegotiationInstructions(
            agent_name=agent_name,
            role = role,
            description=description,
            flat_listing=flat_listing,
            preferences=buyer_preferences if role == common_schemas.RoleType.BUYER else None,
            reservation_value=reservation,
            ethical_constraints=ethics,
            pre_act_label='# NEGOTIATION INSTRUCTIONS',
            verbose=True,
        )
        policy_tool_prompt = hdb_policy_tool_prompt.HDBPolicyToolPrompt(
            model=model,
            observation_component_key=(
                NegotiationComponentConfig.OBSERVATION_COMPONENT_KEY
            ),
            memory_component_key=agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            num_memories_to_retrieve=ACTION_REASONING_MEMORY_WINDOW,
            policy_jsonl_filename=str(
                negotiation_config.get(
                    'policy_jsonl_filename',
                    'hdb_resale_policy_2022_and_before.jsonl',
                )
            ),
            pre_act_label='# POLICY SEARCH TOOL',
        )

        # Setup uncertainty context if specified (should only be one of buyer/seller)
        # TODO: revise negotiation strategy to include strategy evolution based on past failed negotiations, if any
        uncertain_key, uncertain_context = None, None
        strategy_key = 'NegotiationStrategy'
        if role == common_schemas.RoleType.BUYER:
            uncertain_key = 'uncertain_buyer'
            uncertain_context = uncertain_buyer.UncertainBuyer(
                model=model,
                agent_description=description,
                own_confidence=negotiation_config.get('own_confidence', 0.5),
                counterpart_confidence=negotiation_config.get(
                    'counterpart_confidence',
                    0.5,
                ),
                risk_tolerance=negotiation_config.get('risk_tolerance', 0.5), # TODO: determine based on personality metadata
                preferences=negotiation_config.get('preferences', buyer_preferences),
                flat_listing=flat_listing,
                own_reservation_=negotiation_config.get('own_reservation_', 0.0),
                own_reservation_std=negotiation_config.get('own_reservation_std', 1000.0),
                mu=negotiation_config.get('cp_reservation_', 0.0),
                lambda_=negotiation_config.get('lambda_', 1.0),
                a=negotiation_config.get('a', 3.0),
                b=negotiation_config.get('b', 5000.0),
                emit_pre_act_context=False,
                recent_memory_window=ACTION_REASONING_MEMORY_WINDOW,
            )
            strategy = hdb_negotiation_strategy.HDBNegotiationStrategy(
                model=model,
                agent_name=agent_name,
                role=role,
                uncertain_context=uncertain_context,
                description=description,
                verbose=True,
            )

        elif role == common_schemas.RoleType.SELLER:
            uncertain_key = 'uncertain_seller'
            uncertain_context = uncertain_seller.UncertainSeller(
                model=model,
                agent_description=description,
                own_confidence=negotiation_config.get('own_confidence', 1.0),
                counterpart_confidence=negotiation_config.get(
                    'counterpart_confidence',
                    0.5,
                ),
                risk_tolerance=negotiation_config.get('risk_tolerance', 0.5),
                flat_listing=flat_listing,
                own_reservation_=negotiation_config.get('own_reservation_', 0.0),
                mu=negotiation_config.get('cp_reservation_', 0.0),
                lambda_=negotiation_config.get('lambda_', 1.0),
                a=negotiation_config.get('a', 3.0),
                b=negotiation_config.get('b', 5000.0),
                listing_price_prior_discount=negotiation_config.get(
                    'counterpart_listing_price_discount',
                    0.9,
                ),
                emit_pre_act_context=False,
                recent_memory_window=ACTION_REASONING_MEMORY_WINDOW,
            )
            strategy = hdb_negotiation_strategy.HDBNegotiationStrategy(
                model=model,
                agent_name=agent_name,
                role=role,
                uncertain_context=uncertain_context,
                description=description,
            )
        # Build a formatting-safe self-description prompt block.
        safe_description = _escape_format_braces(description)
        preferences_block = ''
        if role == common_schemas.RoleType.BUYER and buyer_preferences:
            preference_lines = []
            for key, value in buyer_preferences.items():
                label = str(key).replace('_', ' ').strip().title()
                if isinstance(value, list):
                    value_str = ', '.join(str(v) for v in value) if value else 'None'
                else:
                    value_str = str(value)
                preference_lines.append(f'- {label}: {value_str}')
            preferences_block = 'Buyer preferences:\n' + '\n'.join(preference_lines) + '\n'
            preferences_block = _escape_format_braces(preferences_block)
        question_about_self = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Who is {agent_name}?',
            question=(
            f'Given the agent description, what kind of {role} is {agent_name}?\n'
            f'Agent description: {safe_description}\n'
            f'{preferences_block}'
            ),
            answer_prefix=f'{agent_name} is a {role} who',
            add_to_memory=False,
            memory_tag='[self perception]'
        )

        # Create question components for context and reasoning
        question_about_situation = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'What situation is {agent_name} in?',
            question=(
                f'What is the current negotiation situation that {agent_name} is in? '
            ),
            answer_prefix=f'{agent_name} is currently',
            add_to_memory=False,
            memory_tag='[situation perception]',
            components = [uncertain_key]
        )

        has_active_offer = (
            str(strategy.fields.get('hasActiveOffer', 'False')).lower()
            == 'true'
        )
        if role == common_schemas.RoleType.BUYER:
            role_action_types = (
                negotiation_schemas.BUYER_OFFER_ACTIONS
                if has_active_offer
                else negotiation_schemas.BUYER_NON_OFFER_ACTIONS
            )
        else:
            role_action_types = (
                negotiation_schemas.SELLER_OFFER_ACTIONS
                if has_active_offer
                else negotiation_schemas.SELLER_NON_OFFER_ACTIONS
            )
        negotiation_action_type_descriptions = negotiation_schemas.format_action_type_descriptions(
            role_action_types
        )

        if role == common_schemas.RoleType.SELLER:
            question = (
                f'Given the negotiation context, what would be the **MOST** appropriate next action for {agent_name}?\n'
                f'Action type descriptions:\n{negotiation_action_type_descriptions}\n'
                f'{HDB_CONTEXT_ANCHOR}\n'
                f'{HDB_ACTION_CHOICE_GUARDRAILS}'
            )
        else:
            question = (
                f'Given the negotiation context, what would be the **MOST** appropriate next action for {agent_name}?\n'
                f'Action type descriptions:\n{negotiation_action_type_descriptions}\n'
                f'{HDB_CONTEXT_ANCHOR}'
                f'{HDB_ACTION_CHOICE_GUARDRAILS}'
                f'If strategy guidance indicates patience is exceeded and you want to terminate without agreement, ONLY use WALK_AWAY.\n'
            )
        action_components = [
            instructions.name,
            policy_tool_prompt.name,
            strategy_key
        ]
        
        question_about_action = agent_components.question_of_recent_memories.QuestionOfRecentMemoriesStructured(
            model=model,
            pre_act_label=f'Next action choice',
            question=question,
            answer_prefix=f'',
            add_to_memory=False,
            memory_tag='[action choice]',
            components=action_components,
            output_schema=common_schemas.ActionChoiceWithRationale,
            choice_responses=role_action_types,
            num_memories_to_retrieve=ACTION_REASONING_MEMORY_WINDOW # TODO: make this adaptive based on personality metadata and recency of relevant context, note that this is even number because this is in negotiation pairs
        )
        
        # TODO: look into more refined strategy integration on later stage
        # Assemble all components
        components_of_agent = {
            NegotiationComponentConfig.OBSERVATION_TO_MEMORY_COMPONENT_KEY: (
                observation_to_memory
            ),
            NegotiationComponentConfig.OBSERVATION_COMPONENT_KEY: observation,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
            instructions.name: instructions,
            NegotiationComponentConfig.POLICY_TOOL_COMPONENT_KEY: policy_tool_prompt,
            uncertain_key: uncertain_context,
            strategy_key: strategy,
            'situation_perception': question_about_situation,
            'self_perception': question_about_self,
            NegotiationComponentConfig.ACTION_DECISIONS_COMPONENT_KEY: (
                question_about_action
            ),
        }

        # Add any extra components
        extra_components = self.params.get('extra_components', {})
        if isinstance(extra_components, dict):
            components_of_agent.update(extra_components)

        # Define component order for context building
        component_order = [
            NegotiationComponentConfig.OBSERVATION_TO_MEMORY_COMPONENT_KEY,
            NegotiationComponentConfig.OBSERVATION_COMPONENT_KEY,
            agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
            instructions.name,
            NegotiationComponentConfig.POLICY_TOOL_COMPONENT_KEY,
            uncertain_key,
            strategy_key,
            'situation_perception',
            'self_perception',
            NegotiationComponentConfig.ACTION_DECISIONS_COMPONENT_KEY,
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
            role=role,
            structured_component_key=(
                NegotiationComponentConfig.ACTION_DECISIONS_COMPONENT_KEY
            ),
            component_order=component_order,
        )
        # Create the agent
        agent = entity_agent_with_logging.EntityAgentWithLogging(
            agent_name=agent_name,
            act_component=act_component,
            context_components=components_of_agent,
            act_component_log_aliases=('action_reasoning',),
        )
        agent._hdb_player_id = str(self.params.get('id', ''))

        # For initialisation, especially for testing
        for observation_text in negotiation_config.get('initial_observations', []):
            agent.observe(observation_text)

        return agent


def update_agent_from_listing(
    agent: entity_agent_with_logging.EntityAgentWithLogging,
    listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
) -> None:
    for component_name in (
        'NegotiationInstructions',
        'uncertain_buyer',
        'uncertain_seller',
        'NegotiationStrategy',
    ):
        try:
            component = agent.get_component(component_name)
        except Exception:
            continue
        apply_listing_handoff = getattr(component, 'apply_listing_handoff', None)
        if not callable(apply_listing_handoff):
            continue
        apply_listing_handoff(listing_payload)

def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    params: Mapping[str, str],
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build the uncertain negotiator agent."""
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)
