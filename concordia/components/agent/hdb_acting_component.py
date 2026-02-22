import json
import re
from collections.abc import Sequence
from typing import Any, Literal, override

from pydantic import BaseModel, RootModel

from concordia.document import interactive_document
from concordia.hdb_simulation.models import schemas as hdb_schemas
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ReasoningIntentJudgement(BaseModel):
    """LLM-as-a-judge output for action/intent consistency checks."""

    intent: Literal["ACCEPT", "REJECT", "COUNTER_OR_OFFER", "OTHER"]
    explanation: str


class OfferIntentJudgement(BaseModel):
    """LLM-as-a-judge output for offer/counteroffer detection in prose."""

    contains_offer_intent: bool
    explanation: str


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
        role: hdb_schemas.RoleType,
        structured_component_key: str = "action_reasoning",
        component_order: Sequence[str] | None = None,
        randomize_choices: bool = False,
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
        self._role = role

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

    def _infer_offer_state_from_options(self, options: Sequence[str]) -> bool | None:
        """Infer whether there is an active offer from action-type options."""
        if not options:
            return None
        normalized = {str(opt).strip().upper() for opt in options if str(opt).strip()}
        if not normalized:
            return None
        if self._role == hdb_schemas.RoleType.BUYER:
            offer_set = set(hdb_schemas.BUYER_OFFER_ACTIONS)
            non_offer_set = set(hdb_schemas.BUYER_NON_OFFER_ACTIONS)
        else:
            offer_set = set(hdb_schemas.SELLER_OFFER_ACTIONS)
            non_offer_set = set(hdb_schemas.SELLER_NON_OFFER_ACTIONS)
        if normalized <= offer_set:
            return True
        if normalized <= non_offer_set:
            return False
        return None

    def _schema_for_turn(self, has_active_offer: bool | None) -> type[RootModel]:
        """Pick role schema for this turn; fallback to broad schema if unknown."""
        if has_active_offer is None:
            if self._role == hdb_schemas.RoleType.BUYER:
                return hdb_schemas.BuyerActions
            return hdb_schemas.SellerActions
        return hdb_schemas.get_action_model(self._role, has_active_offer)

    def _textual_non_offer_action_types(self) -> set[str]:
        """Return non-offer action types that carry free-form prose fields."""
        if self._role == hdb_schemas.RoleType.BUYER:
            return {"QUESTION_BUYER", "INQUIRE_BUYER", "NORMAL_ANSWER"}
        return {"INQUIRE_SELLER", "NORMAL_ANSWER"}

    @staticmethod
    def _collect_text_fields(payload: dict[str, Any]) -> str:
        """Join text-bearing fields used by non-offer actions."""
        keys = ("question_details", "inquiry_details", "answer_details", "reasoning")
        chunks = [str(payload.get(k, "")).strip() for k in keys if str(payload.get(k, "")).strip()]
        return "\n".join(chunks)

    def _contains_offer_intent(self, text: str) -> bool:
        """Use LLM-as-a-judge to detect offer/counteroffer intent in prose."""
        if not text:
            return False
        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(
            "You are a strict negotiation-action judge.\n"
            "Determine whether the text semantically contains an OFFER or COUNTEROFFER intent.\n"
            "OFFER/COUNTEROFFER intent means proposing or suggesting a concrete price level/change,\n"
            "even if currency format is unusual (e.g., 850,000 SGD without '$').\n"
            "Do not mark pure questions, pure inquiries, or pure factual answers as offer intent."
        )
        prompt.statement(f"Text to judge:\n{text}\n")
        verdict = prompt.structured_question(
            question=(
                "Does this text contain offer/counteroffer intent? "
                "Return only the schema."
            ),
            output_schema=OfferIntentJudgement,
            max_tokens=220,
            terminators=(),
        )
        if isinstance(verdict, (BaseModel, RootModel)):
            verdict_payload = verdict.model_dump()
        elif isinstance(verdict, dict):
            verdict_payload = verdict
        elif isinstance(verdict, str):
            verdict_payload = json.loads(self._extract_json(verdict))
        else:
            verdict_payload = {}
        return bool(verdict_payload.get("contains_offer_intent", False))

    @staticmethod
    def _extract_single_offer_price(text: str) -> float | None:
        """Extract one unambiguous candidate offer price from text."""
        matches: list[float] = []
        for match in re.finditer(
            r"\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{5,}(?:\.\d+)?)",
            text,
        ):
            raw = match.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            if value >= 1000:
                matches.append(value)
        unique_prices: list[float] = []
        for value in matches:
            if value not in unique_prices:
                unique_prices.append(value)
        if len(unique_prices) == 1:
            return unique_prices[0]
        return None

    @staticmethod
    def _truncate_text(text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit].rstrip()

    @staticmethod
    def _expected_type_from_intent(
        intent: str,
        has_active_offer: bool | None,
    ) -> str | None:
        """Map judged semantic intent to canonical action type."""
        normalized = str(intent).strip().upper()
        if normalized == "ACCEPT":
            return "ACCEPT_OFFER"
        if normalized == "REJECT":
            return "REJECT_OFFER"
        if normalized == "COUNTER_OR_OFFER":
            if has_active_offer is True:
                return "MAKE_COUNTEROFFER"
            if has_active_offer is False:
                return "MAKE_OFFER"
        return None

    def _judge_reasoning_intent(
        self,
        payload: dict[str, Any],
        has_active_offer: bool | None,
        allowed_types: Sequence[str],
    ) -> None:
        """Use an LLM judge to ensure reasoning semantics match action type."""
        action_type = str(payload.get("type", "")).strip().upper()
        verbal_explanation = str(
            payload.get("verbal_explanation")
            or payload.get("explanation")
            or ""
        ).strip()
        if not action_type or not verbal_explanation:
            return

        # Only enforce for offer-state decision actions where confusion is costly.
        if action_type not in {"ACCEPT_OFFER", "REJECT_OFFER", "MAKE_COUNTEROFFER", "MAKE_OFFER"}:
            return

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(
            "You are a strict negotiation-action consistency judge.\n"
            "Decide the semantic intent of the agent's verbal explanation, independent of the provided action type.\n"
            "Return one intent from: ACCEPT, REJECT, COUNTER_OR_OFFER, OTHER.\n"
            "ACCEPT means the text indicates agreeing to the current offer.\n"
            "REJECT means the text indicates declining the current offer.\n"
            "COUNTER_OR_OFFER means the text indicates proposing a different price.\n"
        )
        prompt.statement(
            f"Current action type: {action_type}\n"
            f"Has active offer: {has_active_offer}\n"
            f"Allowed action types this turn: {', '.join(str(x) for x in allowed_types) or '(unknown)'}\n"
            f"Verbal explanation text:\n{verbal_explanation}\n"
        )
        verdict = prompt.structured_question(
            question=(
                "What is the semantic intent category of the verbal explanation text? "
                "Answer only with the schema."
            ),
            output_schema=ReasoningIntentJudgement,
            max_tokens=300,
            terminators=(),
        )
        if isinstance(verdict, (BaseModel, RootModel)):
            verdict_payload = verdict.model_dump()
        elif isinstance(verdict, dict):
            verdict_payload = verdict
        elif isinstance(verdict, str):
            verdict_payload = json.loads(self._extract_json(verdict))
        else:
            verdict_payload = {}

        expected_type = self._expected_type_from_intent(
            verdict_payload.get("intent", "OTHER"),
            has_active_offer=has_active_offer,
        )
        if not expected_type:
            return

        allowed = {str(x).strip().upper() for x in allowed_types}
        if allowed and expected_type not in allowed:
            return

        if action_type != expected_type:
            explanation = str(verdict_payload.get("explanation", "")).strip()
            raise ValueError(
                "Reasoning/action mismatch detected by LLM judge. "
                f"Action type={action_type}, judged_intent_requires={expected_type}. "
                f"Judge explanation: {explanation}"
            )

    def _preferred_offer_action_type(
        self,
        allowed_types: Sequence[str],
        has_active_offer: bool | None,
    ) -> str | None:
        """Choose the canonical offer action type for this turn."""
        allowed = {str(x).strip().upper() for x in allowed_types}
        if "MAKE_COUNTEROFFER" in allowed:
            return "MAKE_COUNTEROFFER"
        if "MAKE_OFFER" in allowed:
            return "MAKE_OFFER"
        if not allowed:
            if has_active_offer is True:
                return "MAKE_COUNTEROFFER"
            if has_active_offer is False:
                return "MAKE_OFFER"
        return None

    def _coerce_offer_like_payload(
        self,
        payload: dict[str, Any],
        has_active_offer: bool | None,
        allowed_types: Sequence[str],
    ) -> dict[str, Any] | None:
        """Convert prose-heavy non-offer output into canonical offer JSON when safe."""
        details_text = self._collect_text_fields(payload)
        offer_type = self._preferred_offer_action_type(
            allowed_types=allowed_types,
            has_active_offer=has_active_offer,
        )
        if not offer_type:
            return None
        price = self._extract_single_offer_price(details_text)
        if price is None:
            return None
        if offer_type == "MAKE_COUNTEROFFER":
            coerced: dict[str, Any] = {
                "type": "MAKE_COUNTEROFFER",
                "counteroffer_price": price,
            }
            if details_text:
                coerced["reasoning"] = self._truncate_text(details_text)
            return coerced
        coerced: dict[str, Any] = {
            "type": "MAKE_OFFER",
            "offer_price": price,
        }
        if details_text:
            coerced["reasoning"] = self._truncate_text(details_text)
        return coerced

    def _validate_action_for_turn(
        self,
        raw: Any,
        has_active_offer: bool | None,
        allowed_types: Sequence[str] = (),
    ) -> str:
        """Validate extracted JSON against role+offer state (+optional allowed types)."""
        normalized = self._normalize_structured_action(raw)
        json_str = self._extract_json(normalized)
        validated = self._schema_for_turn(has_active_offer).model_validate_json(json_str)
        canonical = validated.model_dump_json()
        payload = json.loads(canonical)
        action_type = str(payload.get("type", "")).strip().upper()
        details_text = self._collect_text_fields(payload)

        # Prevent advice-like prose with embedded prices from passing as non-offer actions.
        if (
            action_type in self._textual_non_offer_action_types()
            and self._contains_offer_intent(details_text)
        ):
            raise ValueError(
                "Detected offer/counteroffer language inside a non-offer action. "
                "Regenerate and emit MAKE_OFFER or MAKE_COUNTEROFFER with a numeric price field."
            )

        self._judge_reasoning_intent(
            payload=payload,
            has_active_offer=has_active_offer,
            allowed_types=allowed_types,
        )

        if not allowed_types:
            return canonical
        allowed = {str(x).strip().upper() for x in allowed_types}
        if action_type not in allowed:
            raise ValueError(
                f"Action type {action_type!r} not in allowed options {sorted(allowed)}."
            )
        return canonical

    def _regenerate_structured_action(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
        has_active_offer: bool | None,
        allowed_types: Sequence[str] = (),
    ) -> str:
        """Generate a repaired structured action under current constraints."""
        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)
        allowed_hint = (
            f"\nAllowed action types: {', '.join(allowed_types)}."
            if allowed_types
            else ""
        )
        allowed_set = {str(x).strip().upper() for x in allowed_types}
        walk_away_rule = (
            "- If you want to terminate this negotiation without agreement, use WALK_AWAY.\n"
            if "WALK_AWAY" in allowed_set
            else ""
        )
        offer_intent_non_offer_error = (
            "Detected offer/counteroffer language inside a non-offer action."
        )
        retry_reason = ""
        for attempt in range(8):
            prompt = interactive_document.InteractiveDocument(self._model)
            prompt.statement(self._context_for_action(contexts) + "\n")
            if retry_reason:
                prompt.statement(
                    "Previous output was invalid. "
                    f"{retry_reason} "
                    "Regenerate with a valid executable action JSON.\n"
                )
            generated = prompt.structured_question(
                question=(
                    f"{call_to_action}{allowed_hint}\n"
                    "Rules:\n"
                    "- Return exactly one executable action JSON object, not advice about what to say.\n"
                    "- If you propose/negotiate any numeric price, the action type must be MAKE_OFFER "
                    "or MAKE_COUNTEROFFER.\n"
                    "- Use QUESTION/INQUIRE/NORMAL_ANSWER only for pure questions/answers with no price proposal.\n"
                    "- If your reasoning says you accept, the action type must be ACCEPT_OFFER.\n"
                    "- If your reasoning says you reject, the action type must be REJECT_OFFER.\n"
                    "- If your reasoning proposes a new price, the action type must be MAKE_COUNTEROFFER (or MAKE_OFFER when no active offer exists).\n"
                    f"{walk_away_rule}"
                    "Return exactly one JSON object matching the expected schema."
                ),
                output_schema=self._schema_for_turn(has_active_offer),
                max_tokens=2200,
                terminators=(),
            )
            try:
                return self._validate_action_for_turn(
                    generated,
                    has_active_offer=has_active_offer,
                    allowed_types=allowed_types,
                )
            except ValueError as error:
                retry_reason = str(error).strip()
                if offer_intent_non_offer_error in retry_reason:
                    return self._regenerate_offer_only_action(
                        contexts=contexts,
                        action_spec=action_spec,
                        has_active_offer=has_active_offer,
                        allowed_types=allowed_types,
                        retry_reason=retry_reason,
                    )
                raise

        raise ValueError(
            "Failed to regenerate a valid structured action after repeated attempts."
        )

    def _regenerate_offer_only_action(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
        has_active_offer: bool | None,
        allowed_types: Sequence[str],
        retry_reason: str,
    ) -> str:
        """Regenerate specifically as an executable offer/counteroffer action."""
        allowed_set = {str(x).strip().upper() for x in allowed_types}
        target_offer_type = "MAKE_OFFER"
        if allowed_set and target_offer_type not in allowed_set:
            raise ValueError(
                "Offer intent was detected, but MAKE_OFFER is not allowed this turn."
            )

        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)
        allowed_hint = (
            f"\nAllowed action types: {', '.join(allowed_types)}."
            if allowed_types
            else ""
        )
        strict_retry_reason = retry_reason
        for _ in range(20):
            prompt = interactive_document.InteractiveDocument(self._model)
            prompt.statement(self._context_for_action(contexts) + "\n")
            prompt.statement(
                "Previous output was invalid. "
                f"{strict_retry_reason} "
                f"You MUST output type {target_offer_type} with its numeric price field.\n"
            )
            generated = prompt.structured_question(
                question=(
                    f"{call_to_action}{allowed_hint}\n"
                    "Rules:\n"
                    f"- Return exactly one executable action JSON with type {target_offer_type}.\n"
                    "- Do not output QUESTION/INQUIRE/NORMAL_ANSWER for this response.\n"
                    "- Include a valid positive numeric price field."
                ),
                output_schema=self._schema_for_turn(has_active_offer),
                max_tokens=2200,
                terminators=(),
            )
            try:
                canonical = self._validate_action_for_turn(
                    generated,
                    has_active_offer=has_active_offer,
                    allowed_types=allowed_types,
                )
                action_type = str(json.loads(canonical).get("type", "")).strip().upper()
                if action_type != target_offer_type:
                    raise ValueError(
                        f"Expected {target_offer_type}, got {action_type or '<missing>'}."
                    )
                return canonical
            except ValueError as error:
                strict_retry_reason = str(error).strip()

        raise ValueError(
            "Failed to regenerate a valid offer/counteroffer action after repeated attempts."
        )

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
            action_spec.output_type in entity_lib.FREE_ACTION_TYPES
            or self._is_action_type_choice(action_spec)
        )
        if use_structured:
            allowed_types = tuple(str(opt).strip().upper() for opt in action_spec.options)
            has_active_offer = self._infer_offer_state_from_options(allowed_types)
            raw = contexts.get(self._structured_component_key)
            if raw:
                try:
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
                    )
                self._logging_channel({
                    "Summary": f"Using structured output from {self._structured_component_key}",
                    "Value": out,
                })
                return out

            if not self._fallback_to_llm_for_free:
                raise ValueError(
                    f'Missing structured action in "{self._structured_component_key}".'
                )
            out = self._regenerate_structured_action(
                contexts=contexts,
                action_spec=action_spec,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
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
            self._role = hdb_schemas.RoleType(state['role'])
        if 'structured_component_key' in state:
            self._structured_component_key = state['structured_component_key']
        if 'component_order' in state:
            self._component_order = tuple(state['component_order']) if state['component_order'] else None
        if 'randomize_choices' in state:
            self._randomize_choices = state['randomize_choices']
        if 'fallback_to_llm_for_free' in state:
            self._fallback_to_llm_for_free = state['fallback_to_llm_for_free']
        
