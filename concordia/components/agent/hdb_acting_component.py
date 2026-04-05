import json
import re
from collections.abc import Sequence
from typing import Any, override

from absl import logging
from pydantic import BaseModel, RootModel

from concordia.document import interactive_document
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.common import RoleType
from concordia.hdb_simulation.models.schemas.policy.schema import PolicyType
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

ALLOWED_POLICY_CONVERSATION_CATEGORIES = ", ".join(
    policy_type.value for policy_type in PolicyType
)

HDB_FIELD_GENERATION_BASE_GUARDRAILS = (
    "## Core Rules\n"
    "- This is an HDB resale negotiation for exactly one flat in Singapore.\n"
    "- Ignore off-domain context and keep content tied to the flat negotiation only.\n"
    "- If mentioning reports/documents, summarize content instead of promising to share files.\n"
    "- Treat any policy information included in the prompt context as private decision support for yourself, not as content to relay automatically to the counterpart.\n"
    f"- If you mention policy explicitly in any public-facing field, restrict it to these categories only: {ALLOWED_POLICY_CONVERSATION_CATEGORIES}.\n"
    "- NEVER mention your own internal state (reservation value, preferences etc.) under verbal_explanation. Include it only under internal_reasoning.\n"
    "- NEVER reveal hidden numeric thresholds or beliefs in public-facing fields, including your own reservation value, your estimate of the counterpart's reservation value, confidence levels, private preference weights, urgency scores, scenario labels, or internal negotiation counts.\n"
    "- If the counterpart asks about your private limit or hidden strategy, answer without disclosing the number or hidden state directly.\n"
)

HDB_FIELD_GENERATION_MONETARY_GUARDRAILS = (
    "## Monetary Rules\n"
    "- ALL pricing and monetary references must be in SGD.\n"
    "- Use realistic, executable numbers for negotiation (no placeholders).\n"
)

HDB_FIELD_GENERATION_INFO_GUARDRAILS = (
    "## Conversation Rules\n"
    "- Keep inquiry/question/answer content focused on flat condition, lease, location, transaction terms, HDB policies, or timeline.\n"
    "- Do not repeat or lightly rephrase a recent question if you already asked for the same information.\n"
    "- If the same issue was already raised recently, move the conversation forward instead of asking it again.\n"
)


def _log_error(message: str) -> None:
    logging.error(message)

class HDBStructuredActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
    """Acting component that produces executable structured actions.

    Current workflow:
      - Read the chosen action type from ``action_decisions``.
      - Build the final action payload for that fixed action type.
      - Fall back to standard prompting only for non-structured action specs.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        role: RoleType,
        structured_component_key: str = "action_decisions",
        component_order: Sequence[str] | None = None,
        randomize_choices: bool = False,
    ):
        """Initialize the HDB structured acting component.

        Args:
            model: Language model used for fallback prompting.
            role: Negotiator role (buyer or seller). Determines schema used.
            structured_component_key: Context key containing the chosen action.
            component_order: Optional ordering for context assembly.
            randomize_choices: Whether choice options are shuffled for CHOICE.
        """
        super().__init__()
        self._model = model
        self._structured_component_key = structured_component_key
        self._component_order = tuple(component_order) if component_order else None
        self._randomize_choices = randomize_choices
        self._role = role

    def _get_ordered_context_keys(
        self, contexts: entity_component.ComponentContextMapping
    ) -> Sequence[str]:
        """Return context keys in deterministic order for prompt assembly."""
        if self._component_order is None:
            return tuple(contexts.keys())
        return self._component_order + tuple(
            sorted(set(contexts.keys()) - set(self._component_order))
        )

    def _build_action_context(
        self, contexts: entity_component.ComponentContextMapping
    ) -> str:
        """Build prompt context while excluding the action-choice component."""
        lines = []
        for k in self._get_ordered_context_keys(contexts):
            if k == self._structured_component_key:
                continue
            v = contexts.get(k)
            if v:
                lines.append(str(v))
        return "\n".join(lines)

    def _stringify_structured_output(self, value: Any) -> str:
        """Convert structured output value to a string representation."""
        if isinstance(value, (BaseModel, RootModel)):
            return value.model_dump_json()
        if isinstance(value, dict):
            return json.dumps(value)
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                _log_error("Empty structured action.")
                return "{}"
            return txt
        return str(value)

    def _serialize_action_payload(self, value: Any) -> str:
        """Return payload as a normalized JSON string when possible."""
        normalized = self._stringify_structured_output(value)
        try:
            json_str = self._extract_first_json_object(normalized)
            payload = json.loads(json_str)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return normalized

    def _extract_first_json_object(self, text: str) -> str:
        """Extract the first complete JSON object from text."""
        candidate = text.strip()
        start = candidate.find("{")
        if start < 0:
            _log_error("Structured action does not contain a JSON object.")
            return "{}"
        candidate = candidate[start:]
        depth = 0
        for idx, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return candidate[: idx + 1]
        _log_error("Unterminated JSON object in structured action.")
        return "{}"

    @staticmethod
    def _schema_for_action_type(action_type: str) -> type[BaseModel] | None:
        """Return the concrete payload model for a canonical action type."""
        mapping: dict[str, type[BaseModel]] = {
            "MAKE_OFFER": negotiation_schemas.MakeOffer,
            "NORMAL_ANSWER": negotiation_schemas.NormalAnswer,
            "INQUIRE_BUYER": negotiation_schemas.BuyerInquiry,
            "QUESTION_BUYER": negotiation_schemas.BuyerQuestion,
            "INQUIRE_SELLER": negotiation_schemas.SellerInquiry,
            "ACCEPT_OFFER": negotiation_schemas.AcceptOffer,
            "REJECT_OFFER": negotiation_schemas.RejectOffer,
            "MAKE_COUNTEROFFER": negotiation_schemas.MakeCounteroffer,
            "WALK_AWAY": negotiation_schemas.BuyerWalkAway,
        }
        return mapping.get(str(action_type).strip().upper())

    @staticmethod
    def _format_action_type_descriptions(action_types: Sequence[str]) -> str:
        lines = []
        for action_type in action_types:
            key = str(action_type).strip().upper()
            if not key:
                continue
            description = negotiation_schemas.NEGOTIATION_ACTION_TYPE_DESCRIPTIONS.get(
                key, "No description available."
            )
            lines.append(f"- {key}: {description}")
        return "\n".join(lines)

    def _parse_action_type(
        self,
        raw: Any,
        allowed_types: Sequence[str] = (),
    ) -> str | None:
        """Extract action-type hint from chooser output text/JSON."""
        allowed = {str(x).strip().upper() for x in allowed_types if str(x).strip()}
        known_types = {
            str(x).strip().upper()
            for x in (
                negotiation_schemas.BUYER_NON_OFFER_ACTIONS
                + negotiation_schemas.BUYER_OFFER_ACTIONS
                + negotiation_schemas.SELLER_NON_OFFER_ACTIONS
                + negotiation_schemas.SELLER_OFFER_ACTIONS
            )
            if str(x).strip()
        }
        candidate_types = allowed or known_types

        def _compact(value: Any) -> str:
            return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())

        compact_lookup = {
            _compact(action_type): action_type
            for action_type in sorted(candidate_types)
        }

        def _coerce(candidate: Any) -> str | None:
            upper = str(candidate or "").strip().upper()
            if not upper:
                return None
            normalized = re.sub(r"[^A-Z0-9]+", "_", upper).strip("_")
            if not normalized:
                return None
            if normalized in candidate_types:
                return normalized
            compact = _compact(upper)
            if compact in compact_lookup:
                return compact_lookup[compact]
            if allowed:
                return None
            if "_" not in normalized:
                return None
            if self._schema_for_action_type(normalized) is None:
                return None
            return normalized

        normalized_raw = raw
        if isinstance(normalized_raw, (BaseModel, RootModel)):
            normalized_raw = normalized_raw.model_dump()
        if isinstance(normalized_raw, dict):
            for key in (
                "type",
                "action_type",
                "chosen_action_type",
                "preferred_action_type",
                "action",
                "name",
            ):
                hint = _coerce(normalized_raw.get(key))
                if hint:
                    return hint

        text = str(normalized_raw or "").strip()
        if not text:
            return None

        try:
            payload = json.loads(self._extract_first_json_object(text))
            if isinstance(payload, dict):
                for key in (
                    "type",
                    "action_type",
                    "chosen_action_type",
                    "preferred_action_type",
                    "action",
                    "name",
                ):
                    hint = _coerce(payload.get(key))
                    if hint:
                        return hint
        except Exception:
            pass

        upper_text = text.upper()
        compact_text = _compact(upper_text)
        if candidate_types:
            for action_type in sorted(candidate_types, key=len, reverse=True):
                if re.search(rf"\b{re.escape(action_type)}\b", upper_text):
                    return action_type
                parts = [part for part in action_type.split("_") if part]
                if parts:
                    spaced_pattern = r"\b" + r"[\s_-]*".join(
                        re.escape(part) for part in parts
                    ) + r"\b"
                    if re.search(spaced_pattern, upper_text):
                        return action_type
                if _compact(action_type) in compact_text:
                    return action_type

        for token in re.findall(r"[A-Z_]{3,}", upper_text):
            hint = _coerce(token)
            if hint:
                return hint
        return None

    @staticmethod
    def _coerce_mapping_payload(raw: Any) -> dict[str, Any]:
        if isinstance(raw, RootModel):
            raw = raw.root
        elif isinstance(raw, BaseModel):
            raw = raw.model_dump()

        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                return {}
            if isinstance(payload, dict):
                return payload
        return {}

    def _build_decision_brief(
        self,
        raw: Any,
        preferred_action_type: str | None,
    ) -> str:
        payload = self._coerce_mapping_payload(raw)
        if not payload:
            return ""

        chosen_type = (
            self._parse_action_type(payload)
            or str(preferred_action_type or "").strip().upper()
        )
        lines = []
        if chosen_type:
            lines.append(f"ChosenActionType={chosen_type}")
        value = str(payload.get("decision_rationale", "")).strip()
        if value:
            lines.append(f"DecisionRationale={value}")
        if not lines:
            return ""
        return "Decision Brief:\n" + "\n".join(lines)

    def _append_prompt_section(
        self,
        prompt: interactive_document.InteractiveDocument,
        prompt_log_sections: list[str],
        title: str,
        body: str,
    ) -> None:
        """Append a markdown section to the prompt and mirrored prompt log."""
        body = str(body).strip()
        if not body:
            return
        section = f"{title}\n{body}\n"
        prompt.statement(section)
        prompt_log_sections.append(section.strip())

    def _generate_action_payload(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
        allowed_types: Sequence[str] = (),
        preferred_action_type: str | None = None,
        decision_brief: str = "",
    ) -> str:
        """Generate payload fields for one already-chosen action type."""
        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)
        allowed_set = {str(x).strip().upper() for x in allowed_types}
        preferred_type = str(preferred_action_type or "").strip().upper()
        if not preferred_type and len(allowed_set) == 1:
            preferred_type = next(iter(allowed_set))
        if not preferred_type:
            _log_error("Missing chosen action type for structured field generation.")
            return "{}"
        if allowed_set and preferred_type not in allowed_set:
            _log_error(
                f"Chosen action type {preferred_type!r} is not allowed this turn."
            )
            preferred_type = next(iter(allowed_set)) if allowed_set else ""
            if not preferred_type:
                return "{}"
        specific_schema = self._schema_for_action_type(preferred_type)
        if specific_schema is None:
            _log_error(
                f"Unsupported action type for structured generation: {preferred_type!r}."
            )
            return "{}"
        chosen_action_description = (
            self._format_action_type_descriptions((preferred_type,))
            if preferred_type
            else ""
        )
        offer_actions = {"MAKE_OFFER", "MAKE_COUNTEROFFER", "ACCEPT_OFFER"}
        info_actions = {"INQUIRE_BUYER","INQUIRE_SELLER","QUESTION_BUYER","NORMAL_ANSWER"}
        action_specific_guardrails = ""
        if preferred_type in offer_actions:
            action_specific_guardrails += HDB_FIELD_GENERATION_MONETARY_GUARDRAILS
        if preferred_type in info_actions:
            action_specific_guardrails += HDB_FIELD_GENERATION_INFO_GUARDRAILS
        meaningful_counteroffer_rule = (
            "- For MAKE_COUNTEROFFER, avoid tiny changes; make a meaningful price move (at least SGD 5,000 or 5% of the current offer price, whichever is larger).\n"
            if preferred_type == "MAKE_COUNTEROFFER"
            else ""
        )
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt_log_sections = []
        self._append_prompt_section(
            prompt,
            prompt_log_sections,
            "# Current Context",
            self._build_action_context(contexts),
        )
        self._append_prompt_section(
            prompt,
            prompt_log_sections,
            "# Decision Brief",
            decision_brief,
        )
        prompt.statement(HDB_FIELD_GENERATION_BASE_GUARDRAILS)
        prompt_log_sections.append(HDB_FIELD_GENERATION_BASE_GUARDRAILS.strip())
        if action_specific_guardrails:
            prompt.statement(action_specific_guardrails)
            prompt_log_sections.append(action_specific_guardrails.strip())
        self._append_prompt_section(
            prompt,
            prompt_log_sections,
            "## Execution Request",
            call_to_action,
        )
        if chosen_action_description:
            self._append_prompt_section(
                prompt,
                prompt_log_sections,
                "# Fixed Action Type",
                (
                    f"- Chosen action type: {preferred_type}\n"
                    f"- Description:\n{chosen_action_description}"
                ),
            )
        internal_reasoning_instructions = (
            "- Use internal_reasoning to explain why the final wording supports the chosen action.\n"
            "- Do not use internal_reasoning to re-open the action choice unless the context makes the chosen action impossible.\n"
        )
        field_generation_instructions = (
            "# Response Rules\n"
            f"- The chosen action type is fixed: {preferred_type}.\n"
            "- Keep the final wording aligned with the decision rationale.\n"
            f"{internal_reasoning_instructions}"
            "- Return using 1st person perspective (I, me, my, etc.).\n"
            f"- Return only the fields required by {preferred_type}.\n"
            "- Include extra type-specific fields where required.\n"
            "- Any numeric price field must be a positive integer.\n"
            "- Avoid loops: do not repeat the same question, request, or answer wording from recent turns unless there is clear new information or you are asking one narrower follow-up.\n"
            f"{meaningful_counteroffer_rule}"
        )
        prompt.statement(field_generation_instructions)
        prompt_log_sections.append(field_generation_instructions.strip())
        structured_question = (
            "# Output\n"
            f"Generate exactly one JSON object for action type `{preferred_type}`.\n"
            "- Use the context above.\n"
            "- Follow the fixed action type.\n"
            "- Return only the required fields for the schema."
        )
        self._logging_channel({
            "Summary": (
                f"StructuredAct field-generation prompt "
                f"({preferred_type})"
            ),
            "Chain of thought": (
                "\n\n".join(prompt_log_sections).splitlines()
                + ["", "Question:", structured_question]
            ),
        })
        generated = prompt.structured_question(
            question=structured_question,
            output_schema=specific_schema,
            max_tokens=2200,
            terminators=(),
        )
        return self._serialize_action_payload(generated)

    @staticmethod
    def _looks_like_action_type_choice(action_spec: entity_lib.ActionSpec) -> bool:
        """True when CHOICE options look like action-type literals."""
        if action_spec.output_type not in entity_lib.CHOICE_ACTION_TYPES:
            return False
        options = tuple(str(opt).strip() for opt in action_spec.options)
        return bool(options) and all(opt and ("_" in opt) and (opt.upper() == opt) for opt in options)

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Produce an action attempt, enforcing structured schema on FREE types."""
        use_structured = (
            action_spec.output_type in entity_lib.FREE_ACTION_TYPES
            or self._looks_like_action_type_choice(action_spec)
        )
        if use_structured:
            allowed_types = tuple(str(opt).strip().upper() for opt in action_spec.options)
            raw = contexts.get(self._structured_component_key)
            if not raw:
                _log_error(
                    f'Missing action choice in "{self._structured_component_key}".'
                )
                return "{}"
            preferred_action_type = self._parse_action_type(
                raw, allowed_types=allowed_types
            )
            if not preferred_action_type:
                if not allowed_types:
                    _log_error(
                        f'Could not parse chosen action type from "{self._structured_component_key}".'
                    )
                    return "{}"
                preferred_action_type = allowed_types[0]
                self._logging_channel({
                    "Summary": (
                        f'Could not parse chosen action type from "{self._structured_component_key}". '
                        f"Falling back to first allowed option: {preferred_action_type}."
                    ),
                    "Value": str(raw),
                })
            if allowed_types and preferred_action_type not in set(allowed_types):
                _log_error(
                    f"Chosen action type {preferred_action_type!r} not in allowed options {sorted(set(allowed_types))}."
                )
                preferred_action_type = allowed_types[0] if allowed_types else None
                if preferred_action_type is None:
                    return "{}"
            decision_brief = self._build_decision_brief(
                raw,
                preferred_action_type=preferred_action_type,
            )
            out = self._generate_action_payload(
                contexts=contexts,
                action_spec=action_spec,
                allowed_types=allowed_types,
                preferred_action_type=preferred_action_type,
                decision_brief=decision_brief,
            )
            self._logging_channel({
                "Summary": (
                    f"Using chosen action type from {self._structured_component_key} "
                    f"to generate payload fields ({preferred_action_type})"
                ),
                "Value": out,
            })
            return out

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(self._build_action_context(contexts) + "\n")
        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)

        if action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
            idx = prompt.multiple_choice_question(
                question=call_to_action,
                answers=action_spec.options,
                randomize_choices=self._randomize_choices,
            )
            return action_spec.options[idx]

        if action_spec.output_type == entity_lib.OutputType.FLOAT:
            sampled = prompt.open_question(call_to_action, max_tokens=2200, terminators=())
            try:
                return str(float(sampled))
            except ValueError:
                return "nan"

        if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
            return prompt.open_question(call_to_action, max_tokens=2200, terminators=())

        logging.error("Unsupported output type: %s", action_spec.output_type)
        return ""
    
    def get_state(self) -> entity_component.ComponentState:
        """Converts component into a dictionary for logging."""
        return {
            "role": self._role.value,
            "structured_component_key": self._structured_component_key,
            "component_order": list(self._component_order) if self._component_order else None,
            "randomize_choices": self._randomize_choices,
        }
    
    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restores component state from a dictionary."""
        if 'role' in state:
            self._role = RoleType(state['role'])
        if 'structured_component_key' in state:
            self._structured_component_key = state['structured_component_key']
        if 'component_order' in state:
            self._component_order = tuple(state['component_order']) if state['component_order'] else None
        if 'randomize_choices' in state:
            self._randomize_choices = state['randomize_choices']
        
