import json
from collections.abc import Sequence
from typing import Any, override

from pydantic import BaseModel, RootModel

from concordia.document import interactive_document
from concordia.hdb_simulation.models.schemas import BuyerActions, RoleType, SellerActions
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class HDBStructuredActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
    """Acting component that enforces role-constrained structured actions.

    Purpose:
      - Read the action produced by a structured context component
        (default key: ``action_reasoning``).
      - Parse that value as JSON (even if surrounded by extra text).
      - Validate the action against the correct schema for the role:
        ``BuyerActions`` for buyers and ``SellerActions`` for sellers.
      - Return canonical JSON for downstream game-master processing.

    Behavior:
      - For FREE-like action specs, this component prefers structured output.
      - If structured output is missing and ``fallback_to_llm_for_free`` is
        False, it raises an error.
      - For CHOICE/FLOAT, it behaves like a normal LLM-driven acting component.
    """

    def __init__(
        self,
        model: language_model.LanguageModel,
        role: RoleType | str,
        structured_component_key: str = "action_reasoning",
        component_order: Sequence[str] | None = None,
        randomize_choices: bool = True,
        fallback_to_llm_for_free: bool = False,
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
        """
        super().__init__()
        self._model = model
        self._structured_component_key = structured_component_key
        self._component_order = tuple(component_order) if component_order else None
        self._randomize_choices = randomize_choices
        self._fallback_to_llm_for_free = fallback_to_llm_for_free
        self._role = RoleType(role)

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
                raise ValueError("Empty structured action.")
            return txt
        return str(value)

    def _extract_json(self, text: str) -> str:
        """Extract the first complete JSON object from text."""
        candidate = text.strip()
        start = candidate.find("{")
        if start < 0:
            raise ValueError("Structured action does not contain a JSON object.")
        candidate = candidate[start:]
        depth = 0
        for idx, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return candidate[: idx + 1]
        raise ValueError("Unterminated JSON object in structured action.")

    def _validate_action_for_role(self, raw: Any) -> str:
        """Validate extracted JSON against BuyerActions/SellerActions schema."""
        normalized = self._normalize_structured_action(raw)
        json_str = self._extract_json(normalized)
        if self._role == RoleType.BUYER:
            validated = BuyerActions.model_validate_json(json_str)
        else:
            validated = SellerActions.model_validate_json(json_str)
        return validated.model_dump_json()

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Produce an action attempt, enforcing structured schema on FREE types."""
        if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
            raw = contexts.get(self._structured_component_key)
            if raw:
                out = self._validate_action_for_role(raw)
                self._logging_channel({
                    "Summary": f"Using structured output from {self._structured_component_key}",
                    "Value": out,
                })
                return out

            if not self._fallback_to_llm_for_free:
                raise ValueError(
                    f'Missing structured action in "{self._structured_component_key}".'
                )

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

        raise NotImplementedError(f"Unsupported output type: {action_spec.output_type}")
    
    def get_state(self) -> entity_component.ComponentState:
        """Converts component into a dictionary for logging."""
        return {
            "role": self._role.value,
            "structured_component_key": self._structured_component_key,
            "component_order": list(self._component_order) if self._component_order else None,
            "randomize_choices": self._randomize_choices,
            "fallback_to_llm_for_free": self._fallback_to_llm_for_free,
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
        