"""Base negotiation agent prefab with core negotiation capabilities."""

from collections.abc import Mapping
import dataclasses
import json
from types import SimpleNamespace

from configs import NegotiationComponentConfig
from configs import PolicyToolConfig
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import hdb_acting_component
from concordia.hdb_simulation.models.schemas import common as common_schemas
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.prefabs.entity.negotiation import structured_setup_batching
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
    "- If **Strategy Summary** has [IMPORTANT] tag, priortise following that guidance over ALL other strategy guidance.\n"
    "- Check recent memory for `[self_action]` entries before deciding, including the stored `decision_rationale` and `internal_reasoning` when present. Avoid repeating your own most recent action unless a new observation, offer-state change, or strategy update clearly justifies repeating it."
)
HDB_CONTEXT_ANCHOR = (
    "NOTE:\n"
    "- You are in an HDB resale negotiation for exactly one flat in Singapore.\n"
    "- Ignore and discard ANY off-domain prior context.\n"
    "- TIME RULE: 1 completed negotiation round (buyer turn + seller turn) = 1 week of in-simulation time.\n"
    "- ALL pricing and monetary references in SGD.\n"
)
ACTION_REASONING_MEMORY_WINDOW = 6
MIN_ACTION_REASONING_MEMORY_WINDOW = 4
MAX_ACTION_REASONING_MEMORY_WINDOW = 12


def _clamp_memory_window(value: object) -> int:
    """Clamp the estimated memory window to the supported range."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = ACTION_REASONING_MEMORY_WINDOW
    return max(
        MIN_ACTION_REASONING_MEMORY_WINDOW,
        min(MAX_ACTION_REASONING_MEMORY_WINDOW, parsed),
    )


def _estimate_action_reasoning_memory_window(
    *,
    model: language_model.LanguageModel,
    agent_name: str,
    description: str,
) -> int:
    """Estimate a persona-specific memory window once at agent initialization."""
    prompt = (
        '# Role\n'
        'You decide how many recent memories a negotiator should review before acting.\n\n'
        '# Task\n'
        'Return `num_memories_to_retrieve` as a single integer between 4 and 12.\n'
        'The choice must be based only on the negotiator persona in the description.\n\n'
        '# Heuristic\n'
        '- Careful, reflective, analytical, detail-heavy, strategic, or cautious personas should get longer memory windows.\n'
        '- Spontaneous, impulsive, present-focused, low-deliberation, or decisive personas should get shorter memory windows.\n'
        '- Balanced or ambiguous personas should stay near 6.\n\n'
        '# Few-shot examples\n'
        'Example 1\n'
        'Description: A careful planner who double-checks details, thinks through trade-offs, remembers prior conversations, and dislikes making rushed decisions.\n'
        'Output: {"num_memories_to_retrieve": 10}\n\n'
        'Example 2\n'
        'Description: Lives in the moment, reacts quickly, dislikes overthinking, and prefers to decide based on the latest signal rather than long context.\n'
        'Output: {"num_memories_to_retrieve": 4}\n\n'
        'Example 3\n'
        'Description: Generally practical and balanced. Reviews some recent context before acting, but does not dwell too long on the past.\n'
        'Output: {"num_memories_to_retrieve": 6}\n\n'
        'Example 4\n'
        'Description: Highly strategic and methodical, tracks patterns across prior exchanges, and adjusts carefully based on accumulated context.\n'
        'Output: {"num_memories_to_retrieve": 11}\n\n'
        '# Rules\n'
        '- Use only the description below.\n'
        '- Return JSON only.\n'
        '- Do not explain your reasoning.\n\n'
        '# Negotiator\n'
        f'Name: {agent_name}\n'
        f'Description: {description or "No description provided."}\n\n'
        '# Output\n'
        'Return a JSON object matching the schema exactly.\n'
    )
    try:
        response = model.sample_text(
            prompt=prompt,
            json_schema=negotiation_schemas.PersonaMemoryWindow.model_json_schema(),
            max_tokens=120,
        )
        parsed = negotiation_schemas.PersonaMemoryWindow.model_validate_json(
            response
        )
        return _clamp_memory_window(parsed.num_memories_to_retrieve)
    except Exception:
        return ACTION_REASONING_MEMORY_WINDOW


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
        action_reasoning_memory_window = _clamp_memory_window(
            negotiation_config.get(
                'action_reasoning_memory_window',
                _estimate_action_reasoning_memory_window(
                    model=model,
                    agent_name=agent_name,
                    description=description,
                ),
            )
        )
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
            num_memories_to_retrieve=action_reasoning_memory_window,
            policy_jsonl_filenames=tuple(
                str(filename)
                for filename in negotiation_config.get(
                    'policy_jsonl_filenames',
                    PolicyToolConfig.DEFAULT_POLICY_JSONL_FILENAMES,
                )
                if str(filename).strip()
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
                recent_memory_window=action_reasoning_memory_window,
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
                recent_memory_window=action_reasoning_memory_window,
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
        question_about_self = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            pre_act_label=f'Who is {agent_name}?',
            question=(
            f'Given the agent description, what kind of {role} is {agent_name}?\n'
            f'Agent description: {safe_description}\n'
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
            'situation_perception',
            'self_perception',
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
            num_memories_to_retrieve=action_reasoning_memory_window
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


def batch_update_agents_from_listings(
    agent_payload_pairs: tuple[
        tuple[
            entity_agent_with_logging.EntityAgentWithLogging,
            negotiation_schemas.ListingNegotiationTransferPayload,
        ],
        ...,
    ]
    | list[
        tuple[
            entity_agent_with_logging.EntityAgentWithLogging,
            negotiation_schemas.ListingNegotiationTransferPayload,
        ]
    ],
) -> None:
    def _buyer_safe_payload(
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            match_id=listing_payload.match_id,
            week_matched=listing_payload.week_matched,
            listing_record=listing_payload.listing_record,
            buyer_state=listing_payload.buyer_state,
            seller_state=SimpleNamespace(
                id=listing_payload.seller_state.id,
                name=listing_payload.seller_state.name,
                role=listing_payload.seller_state.role,
                description=listing_payload.seller_state.description,
                listed=listing_payload.seller_state.listed,
                current_listing_id=listing_payload.seller_state.current_listing_id,
                current_listing_price=listing_payload.seller_state.current_listing_price,
            ),
        )

    def _seller_safe_payload(
        listing_payload: negotiation_schemas.ListingNegotiationTransferPayload,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            match_id=listing_payload.match_id,
            week_matched=listing_payload.week_matched,
            listing_record=listing_payload.listing_record,
            seller_state=listing_payload.seller_state,
            buyer_state=SimpleNamespace(
                id=listing_payload.buyer_state.id,
                name=listing_payload.buyer_state.name,
                role=listing_payload.buyer_state.role,
                description=listing_payload.buyer_state.description,
            ),
        )

    pending_structured_updates: list[
        tuple[
            object,
            object,
            list[structured_setup_batching.StructuredSetupRequest],
        ]
    ] = []
    direct_updates: list[tuple[object, object]] = []
    all_requests: list[structured_setup_batching.StructuredSetupRequest] = []

    for agent, listing_payload in agent_payload_pairs:
        buyer_safe_payload = _buyer_safe_payload(listing_payload)
        seller_safe_payload = _seller_safe_payload(listing_payload)
        instructions_payload = SimpleNamespace(
            listing_record=listing_payload.listing_record
        )

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
            if component_name == 'NegotiationInstructions':
                apply_listing_handoff(instructions_payload)
                continue
            if component_name == 'uncertain_buyer':
                target_payload = buyer_safe_payload
            elif component_name == 'uncertain_seller':
                target_payload = seller_safe_payload
            else:
                component_role = getattr(component, '_role', None)
                if component_role == common_schemas.RoleType.BUYER:
                    target_payload = buyer_safe_payload
                elif component_role == common_schemas.RoleType.SELLER:
                    target_payload = seller_safe_payload
                else:
                    target_payload = instructions_payload

            build_requests = getattr(component, 'build_listing_handoff_requests', None)
            apply_responses = getattr(
                component, 'apply_listing_handoff_responses', None
            )
            if callable(build_requests) and callable(apply_responses):
                component_requests = list(build_requests(target_payload) or ())
                if component_requests:
                    pending_structured_updates.append(
                        (component, target_payload, component_requests)
                    )
                    all_requests.extend(component_requests)
                    continue
            direct_updates.append((component, target_payload))

    raw_responses = structured_setup_batching.execute_setup_requests(all_requests)
    response_index = 0
    for component, target_payload, component_requests in pending_structured_updates:
        response_count = len(component_requests)
        component_responses = raw_responses[
            response_index: response_index + response_count
        ]
        response_index += response_count
        getattr(component, 'apply_listing_handoff_responses')(
            target_payload,
            {
                request.response_key: response
                for request, response in zip(
                    component_requests, component_responses
                )
            },
        )

    for component, target_payload in direct_updates:
        component.apply_listing_handoff(target_payload)

def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    params: Mapping[str, str],
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build the uncertain negotiator agent."""
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)
