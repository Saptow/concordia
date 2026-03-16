import json
import re
from collections.abc import Sequence
from typing import Any, override

from absl import logging
from pydantic import BaseModel, RootModel

from concordia.document import interactive_document
from concordia.hdb_simulation.models.schemas import negotiation as negotiation_schemas
from concordia.hdb_simulation.models.schemas.common import RoleType
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

HDB_FIELD_GENERATION_BASE_GUARDRAILS = (
    "FIELD-GENERATION GUARDRAILS (MUST-FOLLOW):\n"
    "- This is an HDB resale negotiation for exactly one flat in Singapore.\n"
    "- Ignore off-domain context and keep content tied to the flat negotiation only.\n"
    "- If mentioning reports/documents, summarize content instead of promising to share files.\n"
    "- NEVER mention your own internal state (reservation value, preferences etc.) under verbal_explanation. Include ONLY under internal_reasoning."
)

HDB_FIELD_GENERATION_MONETARY_GUARDRAILS = (
    "- ALL pricing and monetary references must be in SGD.\n"
    "- Use realistic, executable numbers for negotiation (no placeholders).\n"
)

HDB_FIELD_GENERATION_INFO_GUARDRAILS = (
    "- Keep inquiry/question/answer content focused on flat condition, lease, location, transaction terms, HDB policies, or timeline.\n"
)


def _log_error(message: str) -> None:
    logging.error(message)

class HDBStructuredActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
    """Acting component that produces executable structured actions.

    Purpose:
      - Read action intent from a structured context component
        (default key: ``action_decisions``).
      - In action-choice mode, treat that upstream value as a preselected
        ``action_type`` and generate only the remaining payload fields.
      - Optionally validate the resulting payload against role-constrained
        schemas before returning canonical JSON.

    Behavior:
      - For FREE-like action specs, this component prefers structured output.
      - If structured output is missing and ``fallback_to_llm_for_free`` is
        False, it raises an error.
      - For CHOICE/FLOAT, it behaves like a normal LLM-driven acting component.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        role: RoleType,
        structured_component_key: str = "action_decisions",
        component_order: Sequence[str] | None = None,
        randomize_choices: bool = False,
        fallback_to_llm_for_free: bool = False,
        structured_component_outputs_action_choice: bool = False,
        disable_action_validation: bool = True,
        force_offer_timeout_seconds: float = 20.0,
        force_offer_default_price: float = 500000.0,
    ):
        """Initialize the HDB structured acting component.

        Args:
            model: Language model used for fallback prompting.
            role: Negotiator role (buyer or seller). Determines schema used.
            structured_component_key: Context key containing structured action.
            component_order: Optional ordering for context assembly.
            randomize_choices: Whether choice options are shuffled for CHOICE.
            fallback_to_llm_for_free: If True, FREE action falls back to LLM
                when structured action is missing instead of raising.
            structured_component_outputs_action_choice: If True, treat the
                structured component value as action-type choice only, then
                generate full JSON fields in a second structured step.
            disable_action_validation: If True, bypass schema and policy
                validation and pass through generated JSON payloads. Default is
                True for the current choice-first workflow.
            force_offer_timeout_seconds: Max seconds before forcing a deterministic
                executable offer/counteroffer fallback.
            force_offer_default_price: Default positive fallback price used if
                reservation bounds are unavailable.
        """
        super().__init__()
        self._model = model
        self._structured_component_key = structured_component_key
        self._component_order = tuple(component_order) if component_order else None
        self._randomize_choices = randomize_choices
        self._fallback_to_llm_for_free = fallback_to_llm_for_free
        self._structured_component_outputs_action_choice = bool(
            structured_component_outputs_action_choice
        )
        self._disable_action_validation = bool(disable_action_validation)
        self._role = role
        self._force_offer_timeout_seconds = max(1.0, float(force_offer_timeout_seconds))
        self._force_offer_default_price = max(1.0, float(force_offer_default_price))

    def _ordered_keys(self, contexts: entity_component.ComponentContextMapping) -> Sequence[str]:
        """Return context keys in deterministic order for prompt assembly."""
        if self._component_order is None:
            return tuple(contexts.keys())
        return self._component_order + tuple(
            sorted(set(contexts.keys()) - set(self._component_order))
        )

    def _context_for_action(self, contexts: entity_component.ComponentContextMapping) -> str:
        """Build fallback prompt context while excluding structured action key."""
        lines = []
        for k in self._ordered_keys(contexts):
            if k == self._structured_component_key:
                continue
            v = contexts.get(k)
            if v:
                lines.append(str(v))
        return "\n".join(lines)

    def _normalize_structured_action(self, value: Any) -> str:
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

    def _passthrough_action_payload(self, value: Any) -> str:
        """Return payload as JSON string without applying validation rules."""
        normalized = self._normalize_structured_action(value)
        try:
            json_str = self._extract_json(normalized)
            payload = json.loads(json_str)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return normalized

    def _extract_json(self, text: str) -> str:
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

    def _infer_offer_state_from_options(self, options: Sequence[str]) -> bool | None:
        """Infer whether there is an active offer from action-type options."""
        if not options:
            return None
        normalized = {str(opt).strip().upper() for opt in options if str(opt).strip()}
        if not normalized:
            return None
        if self._role == RoleType.BUYER:
            offer_set = set(negotiation_schemas.BUYER_OFFER_ACTIONS)
            non_offer_set = set(negotiation_schemas.BUYER_NON_OFFER_ACTIONS)
        else:
            offer_set = set(negotiation_schemas.SELLER_OFFER_ACTIONS)
            non_offer_set = set(negotiation_schemas.SELLER_NON_OFFER_ACTIONS)
        if normalized <= offer_set:
            return True
        if normalized <= non_offer_set:
            return False
        return None

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

    def _extract_action_type_hint(
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
            payload = json.loads(self._extract_json(text))
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

    def _extract_decision_brief(
        self,
        raw: Any,
        preferred_action_type: str | None,
    ) -> str:
        payload = self._coerce_mapping_payload(raw)
        if not payload:
            return ""

        chosen_type = (
            self._extract_action_type_hint(payload)
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

    def _schema_for_turn(self, has_active_offer: bool | None) -> type[RootModel]:
        """Pick role schema for this turn; fallback to broad schema if unknown."""
        if has_active_offer is None:
            if self._role == RoleType.BUYER:
                return negotiation_schemas.NegotiationBuyerActions
            return negotiation_schemas.NegotiationSellerActions
        return negotiation_schemas.get_action_model(self._role, has_active_offer)

    def _validate_action_for_turn(
        self,
        raw: Any,
        has_active_offer: bool | None,
        allowed_types: Sequence[str] = (),
        bypass_unified_judge: bool = False,
    ) -> str:
        """Validation is disabled; return payload as-is."""
        del has_active_offer, allowed_types, bypass_unified_judge
        return self._passthrough_action_payload(raw)

    def _regenerate_structured_action(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
        has_active_offer: bool | None,
        allowed_types: Sequence[str] = (),
        preferred_action_type: str | None = None,
        decision_brief: str = "",
    ) -> str:
        """Generate payload fields for one already-chosen action type."""
        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)
        del has_active_offer
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
        info_actions = {
            "INQUIRE_BUYER",
            "INQUIRE_SELLER",
            "QUESTION_BUYER",
            "NORMAL_ANSWER",
        }
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
        prompt.statement(self._context_for_action(contexts) + "\n")
        if decision_brief:
            prompt.statement(decision_brief + "\n")
        prompt.statement(HDB_FIELD_GENERATION_BASE_GUARDRAILS)
        if action_specific_guardrails:
            prompt.statement(action_specific_guardrails)
        prompt.statement(call_to_action)
        if chosen_action_description:
            prompt.statement(f"Chosen action description:\n{chosen_action_description}\n")
        internal_reasoning_instructions = (
            "- Use internal_reasoning to explain why the final wording supports the chosen action.\n"
            "- Do not use internal_reasoning to re-open the action choice unless the context makes the chosen action impossible.\n"
        )
        prompt.statement(
            "Field-generation instructions:\n"
            f"- The chosen action type is fixed: {preferred_type}\n"
            "- Keep the final wording aligned with the decision rationale.\n"
            f"{internal_reasoning_instructions}"
            "- Return using 1st person perspective (I, me, my, etc.).\n"
            f"- Return only the fields required by {preferred_type}.\n"
            "- Include extra type-specific fields where required.\n"
            "- Any numeric price field must be a positive integer.\n"
            "- For regular conversations, avoid repeating the same conversation to avoid looping.\n"
            f"{meaningful_counteroffer_rule}"
        )
        structured_question = (
            "Generate the fields required for exactly one JSON object for the chosen action type "
            f"{preferred_type}, using the context and decision brief above."
        )
        self._logging_channel({
            "Summary": (
                f"StructuredAct field-generation prompt "
                f"({preferred_type})"
            ),
            "Chain of thought": (
                prompt.view().text().splitlines()
                + ["", "Question:", structured_question]
            ),
        })
        generated = prompt.structured_question(
            question=structured_question,
            output_schema=specific_schema,
            max_tokens=2200,
            terminators=(),
        )
        return self._passthrough_action_payload(generated)

    @staticmethod
    def _is_action_type_choice(action_spec: entity_lib.ActionSpec) -> bool:
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
            self._structured_component_outputs_action_choice
            or action_spec.output_type in entity_lib.FREE_ACTION_TYPES
            or self._is_action_type_choice(action_spec)
        )
        if use_structured:
            allowed_types = tuple(str(opt).strip().upper() for opt in action_spec.options)
            has_active_offer = self._infer_offer_state_from_options(allowed_types)
            raw = contexts.get(self._structured_component_key)

            if self._structured_component_outputs_action_choice:
                if not raw:
                    _log_error(
                        f'Missing action-type choice in "{self._structured_component_key}". '
                        "Ensure question_about_action writes the chosen action type to this key."
                    )
                    if allowed_types:
                        raw = {"type": allowed_types[0]}
                    else:
                        return "{}"
                preferred_action_type = self._extract_action_type_hint(
                    raw, allowed_types=allowed_types
                )
                if not preferred_action_type:
                    if allowed_types:
                        preferred_action_type = allowed_types[0]
                        self._logging_channel({
                            "Summary": (
                                f'Could not parse chosen action type from "{self._structured_component_key}". '
                                f"Falling back to first allowed option: {preferred_action_type}."
                            ),
                            "Value": str(raw),
                        })
                    else:
                        _log_error(
                            f'Could not parse chosen action type from "{self._structured_component_key}". '
                            "question_about_action should return exactly one action type token."
                        )
                        return "{}"
                if allowed_types and preferred_action_type not in set(allowed_types):
                    _log_error(
                        f"Chosen action type {preferred_action_type!r} not in allowed options {sorted(set(allowed_types))}."
                    )
                    preferred_action_type = allowed_types[0] if allowed_types else None
                    if preferred_action_type is None:
                        return "{}"
                decision_brief = self._extract_decision_brief(
                    raw,
                    preferred_action_type=preferred_action_type,
                )
                out = self._regenerate_structured_action(
                    contexts=contexts,
                    action_spec=action_spec,
                    has_active_offer=has_active_offer,
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

            if raw:
                preferred_action_type = self._extract_action_type_hint(
                    raw, allowed_types=allowed_types
                )
                try:
                    if self._disable_action_validation:
                        out = self._passthrough_action_payload(raw)
                    else:
                        out = self._validate_action_for_turn(
                            raw,
                            has_active_offer=has_active_offer,
                            allowed_types=allowed_types,
                        )
                except Exception:
                    out = self._regenerate_structured_action(
                        contexts=contexts,
                        action_spec=action_spec,
                        has_active_offer=has_active_offer,
                        allowed_types=allowed_types,
                        preferred_action_type=preferred_action_type,
                        decision_brief="",
                    )
                self._logging_channel({
                    "Summary": f"Using structured output from {self._structured_component_key}",
                    "Value": out,
                })
                return out

            if not self._fallback_to_llm_for_free:
                _log_error(
                    f'Missing structured action in "{self._structured_component_key}".'
                )
                return "{}"
            out = self._regenerate_structured_action(
                contexts=contexts,
                action_spec=action_spec,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
                preferred_action_type=None,
                decision_brief="",
            )
            self._logging_channel({
                "Summary": "Regenerated structured output from action spec context",
                "Value": out,
            })
            return out

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(self._context_for_action(contexts) + "\n")
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
            "fallback_to_llm_for_free": self._fallback_to_llm_for_free,
            "structured_component_outputs_action_choice": self._structured_component_outputs_action_choice,
            "disable_action_validation": self._disable_action_validation,
            "force_offer_timeout_seconds": self._force_offer_timeout_seconds,
            "force_offer_default_price": self._force_offer_default_price,
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
        if 'fallback_to_llm_for_free' in state:
            self._fallback_to_llm_for_free = state['fallback_to_llm_for_free']
        if 'structured_component_outputs_action_choice' in state:
            self._structured_component_outputs_action_choice = bool(
                state['structured_component_outputs_action_choice']
            )
        if 'disable_action_validation' in state:
            self._disable_action_validation = bool(state['disable_action_validation'])
        if 'force_offer_timeout_seconds' in state:
            self._force_offer_timeout_seconds = max(
                1.0, float(state['force_offer_timeout_seconds'])
            )
        if 'force_offer_default_price' in state:
            self._force_offer_default_price = max(
                1.0, float(state['force_offer_default_price'])
            )
        
