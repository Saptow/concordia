import json
import math
import time
from collections.abc import Sequence
from typing import Any, Literal, override

from pydantic import BaseModel, RootModel, ValidationError

from concordia.document import interactive_document
from concordia.hdb_simulation.models import schemas as hdb_schemas
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class UnifiedActionJudgement(BaseModel):
    """Single-pass LLM-as-a-judge output for action-level consistency checks."""

    intent: Literal["ACCEPT", "REJECT", "COUNTER_OR_OFFER", "OTHER"]
    contains_offer_intent: bool
    walk_away_intent: Literal["WALK_AWAY", "CONTINUE", "UNCLEAR"]
    proposed_price: int | None = None
    explanation: str


class ForcedOfferReasoning(BaseModel):
    """Schema for fallback reasoning tied to a forced offer action."""

    internal_reasoning: str
    verbal_explanation: str


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

    @staticmethod
    def _walk_away_explanation_from_payload(payload: dict[str, Any]) -> str:
        """Extract best public-facing explanation text for WALK_AWAY."""
        for key in ("verbal_explanation", "answer_details", "inquiry_details", "question_details", "reasoning"):
            value = str(payload.get(key, "")).strip()
            if value:
                return value
        return "I am ending this negotiation without agreement."

    def _strategy_requires_walk_away(self, allowed_types: Sequence[str]) -> bool:
        """Return True when buyer strategy patience has been exceeded."""
        if self._role != hdb_schemas.RoleType.BUYER:
            return False
        allowed = {str(x).strip().upper() for x in allowed_types}
        if allowed and "WALK_AWAY" not in allowed:
            return False
        try:
            strategy_component = self.get_entity().get_component("NegotiationStrategy")
        except Exception:
            return False

        should_walk_away = getattr(strategy_component, "should_walk_away", None)
        if not callable(should_walk_away):
            return False

        try:
            return bool(should_walk_away())
        except Exception:
            return False

    @staticmethod
    def _structured_price_from_payload(payload: dict[str, Any]) -> int | None:
        """Extract canonical structured numeric price from the action payload."""
        action_type = str(payload.get("type", "")).strip().upper()
        field_by_action = {
            "MAKE_OFFER": "offer_price",
            "MAKE_COUNTEROFFER": "counteroffer_price",
            "ACCEPT_OFFER": "price_settled",
        }
        price_field = field_by_action.get(action_type)
        if not price_field:
            return None
        raw_value = payload.get(price_field)
        if isinstance(raw_value, bool) or raw_value is None:
            return None
        if isinstance(raw_value, (int, float)):
            return int(math.floor(float(raw_value) + 0.5))
        try:
            return int(float(str(raw_value).strip()))
        except (TypeError, ValueError):
            return None

    def _judge_action_consistency(
        self,
        payload: dict[str, Any],
        has_active_offer: bool | None,
        allowed_types: Sequence[str],
    ) -> UnifiedActionJudgement:
        """Single judge call for holistic consistency checks."""
        action_type = str(payload.get("type", "")).strip().upper()
        verbal_explanation = str(
            payload.get("verbal_explanation")
            or payload.get("explanation")
            or ""
        ).strip()
        non_verbal_text = self._collect_text_fields(payload)
        internal_reasoning = str(payload.get("internal_reasoning", "")).strip()
        structured_price = self._structured_price_from_payload(payload)

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(
            "You are a strict negotiation-action validator.\n"
            "Think step-by-step internally, but return only the output schema.\n"
            "Validate in this sequence:\n"
            "1) classify semantic intent as ACCEPT / REJECT / COUNTER_OR_OFFER / OTHER.\n"
            "2) decide whether public text contains offer/counteroffer intent.\n"
            "3) decide whether text explicitly ends negotiation now (walk away intent).\n"
            "4) extract the explicit proposed/accepted SGD price from verbal_explanation if clear, else null.\n"
            "Definitions:\n"
            "- WALK_AWAY means explicit immediate termination with no deal.\n"
            "- COUNTER_OR_OFFER means proposing a new/alternative price.\n"
            "One-shot calibration examples:\n"
            "Example A (offer intent TRUE):\n"
            "Action payload JSON: "
            "{\"type\":\"QUESTION_BUYER\",\"question_details\":\"I can offer $475,000. "
            "Would you be willing to proceed at this price?\"}\n"
            "Expected judgement: intent=COUNTER_OR_OFFER, contains_offer_intent=True, "
            "walk_away_intent=CONTINUE, proposed_price=475000.\n"
            "Example B (offer intent FALSE):\n"
            "Action payload JSON: "
            "{\"type\":\"NORMAL_ANSWER\",\"answer_details\":\"Recent nearby transactions were "
            "$620,000 and $630,000. Do you have a price range in mind?\"}\n"
            "Expected judgement: intent=OTHER, contains_offer_intent=False, "
            "walk_away_intent=CONTINUE, proposed_price=null.\n"
            "Example C (accept intent in non-offer type):\n"
            "Action payload JSON: "
            "{\"type\":\"NORMAL_ANSWER\",\"answer_details\":\"I'm happy to accept $804,250. "
            "Shall we proceed with OTP?\"}\n"
            "Expected judgement: intent=ACCEPT, contains_offer_intent=True, "
            "walk_away_intent=CONTINUE, proposed_price=804250."
        )
        prompt.statement(
            f"Current action type: {action_type}\n"
            f"Has active offer: {has_active_offer}\n"
            f"Allowed action types this turn: {', '.join(str(x) for x in allowed_types) or '(unknown)'}\n"
            f"Structured price field (if any): {structured_price if structured_price is not None else 'None'}\n"
            f"Verbal explanation:\n{verbal_explanation or '(empty)'}\n"
            f"Other public text fields:\n{non_verbal_text or '(empty)'}\n"
            f"Internal reasoning:\n{internal_reasoning or '(empty)'}\n"
            f"Action payload JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            verdict = prompt.structured_question(
                question=(
                    "Return unified action consistency judgement using the schema only."
                ),
                output_schema=UnifiedActionJudgement,
                max_tokens=420,
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
            return UnifiedActionJudgement.model_validate(verdict_payload)
        except Exception:
            return UnifiedActionJudgement(
                intent="OTHER",
                contains_offer_intent=False,
                walk_away_intent="UNCLEAR",
                proposed_price=None,
                explanation="Unified judge unavailable.",
            )
    @staticmethod
    def _truncate_text(text: str, limit: int = 1000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit].rstrip()

    def _reservation_bounds_from_components(self) -> tuple[float, float] | None:
        """Fetch reservation bounds from strategy/uncertainty components."""
        entity = self.get_entity()

        try:
            strategy_component = entity.get_component("NegotiationStrategy")
        except Exception:
            strategy_component = None

        if strategy_component is not None:
            strategy_state = getattr(strategy_component, "_state", None)
            current_position = (
                getattr(strategy_state, "current_position", None)
                if strategy_state is not None
                else None
            )
            opponent_position = (
                getattr(strategy_state, "opponent_position", None)
                if strategy_state is not None
                else None
            )
            if isinstance(current_position, (int, float)) and isinstance(opponent_position, (int, float)):
                lower = max(0.0, min(float(current_position), float(opponent_position)))
                upper = max(lower, max(float(current_position), float(opponent_position)))
                return (lower, upper)

        for component_name in ("uncertain_buyer", "uncertain_seller"):
            try:
                uncertain_component = entity.get_component(component_name)
            except Exception:
                continue

            own_value = None
            counterpart_value = None

            beliefs = getattr(uncertain_component, "_beliefs", None)
            if isinstance(beliefs, dict):
                own_belief = beliefs.get("own_reservation")
                counterpart_belief = beliefs.get("counterpart_reservation")
                if own_belief is not None:
                    own_value = getattr(own_belief, "get_expected_mean", None)
                if counterpart_belief is not None:
                    counterpart_value = getattr(counterpart_belief, "get_expected_mean", None)

            explicit_own = getattr(uncertain_component, "_own_reservation", None)
            if isinstance(explicit_own, (int, float)):
                own_value = explicit_own

            if isinstance(own_value, (int, float)) and isinstance(counterpart_value, (int, float)):
                lower = max(0.0, min(float(own_value), float(counterpart_value)))
                upper = max(lower, max(float(own_value), float(counterpart_value)))
                return (lower, upper)

        return None

    def _offer_price_from_reservation_bounds(self, action_type: str) -> float:
        """Generate fallback offer price using reservation bounds."""
        bounds = self._reservation_bounds_from_components()
        if bounds is None:
            return self._force_offer_default_price

        lower, upper = bounds
        spread = max(0.0, upper - lower)
        if spread <= 1e-6:
            return max(1.0, lower)

        normalized_action_type = str(action_type).strip().upper()
        if self._role == hdb_schemas.RoleType.BUYER:
            ratio = 0.40 if normalized_action_type == "MAKE_COUNTEROFFER" else 0.25
        else:
            ratio = 0.60 if normalized_action_type == "MAKE_COUNTEROFFER" else 0.75

        return max(1.0, lower + ratio * spread)

    @staticmethod
    def _default_forced_reasoning(
        action_type: str,
        price: float,
        retry_reason: str,
    ) -> tuple[str, str]:
        """Fallback reasoning text when LLM reasoning generation fails."""
        safe_price = max(1.0, float(price))
        internal = (
            "Forced fallback after repeated invalid offer generation. "
            f"Action={action_type}, Price={safe_price:.2f}. "
            f"Last validation error: {retry_reason or 'unknown'}."
        )
        verbal = f"I propose ${safe_price:,.0f} as my next price."
        return internal, verbal

    def _generate_forced_offer_reasoning(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_type: str,
        price: float,
        retry_reason: str,
    ) -> tuple[str, str]:
        """Return deterministic reasoning for forced fallback offers.

        We avoid another LLM call in this path to prevent inconsistent narratives
        (for example, hallucinated prior offers/prices) when execution has
        already fallen back after repeated validation failures.
        """
        del contexts
        internal_reasoning, verbal_explanation = self._default_forced_reasoning(
            action_type=action_type,
            price=price,
            retry_reason=retry_reason,
        )
        return (
            self._truncate_text(internal_reasoning, limit=1000),
            self._truncate_text(verbal_explanation, limit=1000),
        )

    @staticmethod
    def _forced_offer_payload(
        action_type: str,
        price: float,
        internal_reasoning: str,
        verbal_explanation: str,
    ) -> dict[str, Any]:
        """Build deterministic schema-compliant fallback offer JSON."""
        # Offer schemas require integer prices.
        safe_price = max(1, math.floor(float(price) + 0.5))
        payload: dict[str, Any] = {
            "type": action_type,
            "internal_reasoning": internal_reasoning,
            "verbal_explanation": verbal_explanation,
        }
        if action_type == "MAKE_COUNTEROFFER":
            payload["counteroffer_price"] = safe_price
        else:
            payload["offer_price"] = safe_price
        return payload

    def _deterministic_fallback_payload(
        self,
        has_active_offer: bool | None,
        allowed_types: Sequence[str],
        retry_reason: str,
    ) -> dict[str, Any]:
        """Build a safe deterministic fallback action payload."""
        allowed = {str(x).strip().upper() for x in allowed_types}
        preferred_order = (
            "REJECT_OFFER",
            "MAKE_COUNTEROFFER",
            "MAKE_OFFER",
            "NORMAL_ANSWER",
            "INQUIRE_BUYER",
            "QUESTION_BUYER",
            "INQUIRE_SELLER",
            "ACCEPT_OFFER",
            "WALK_AWAY",
        )

        action_type: str | None = None
        if allowed:
            for candidate in preferred_order:
                if candidate in allowed:
                    # Avoid WALK_AWAY as emergency fallback unless it is the only option.
                    if candidate == "WALK_AWAY" and len(allowed) > 1:
                        continue
                    action_type = candidate
                    break
        else:
            if has_active_offer is True:
                action_type = "REJECT_OFFER"
            elif has_active_offer is False:
                action_type = "MAKE_OFFER"
            else:
                action_type = "NORMAL_ANSWER"

        if not action_type:
            action_type = "NORMAL_ANSWER"

        reason = (
            "Deterministic fallback after repeated invalid structured outputs. "
            f"Last validation error: {retry_reason or 'unknown'}."
        )

        if action_type == "MAKE_OFFER":
            price = self._offer_price_from_reservation_bounds(action_type)
            return {
                "type": action_type,
                "offer_price": int(math.floor(float(price) + 0.5)),
                "internal_reasoning": reason,
                "verbal_explanation": f"I propose ${float(price):,.0f} as my offer.",
            }
        if action_type == "MAKE_COUNTEROFFER":
            price = self._offer_price_from_reservation_bounds(action_type)
            return {
                "type": action_type,
                "counteroffer_price": int(math.floor(float(price) + 0.5)),
                "internal_reasoning": reason,
                "verbal_explanation": f"I propose ${float(price):,.0f} as my counteroffer.",
            }
        if action_type == "ACCEPT_OFFER":
            price = self._offer_price_from_reservation_bounds("MAKE_COUNTEROFFER")
            return {
                "type": action_type,
                "price_settled": int(math.floor(float(price) + 0.5)),
                "internal_reasoning": reason,
                "verbal_explanation": "I accept the current offer.",
            }
        if action_type == "REJECT_OFFER":
            return {
                "type": action_type,
                "internal_reasoning": reason,
                "verbal_explanation": "I reject the current offer.",
            }
        if action_type == "INQUIRE_BUYER":
            return {
                "type": action_type,
                "internal_reasoning": reason,
                "inquiry_details": "Could you share more details that affect your valuation?",
            }
        if action_type == "QUESTION_BUYER":
            return {
                "type": action_type,
                "internal_reasoning": reason,
                "question_details": "What conditions would make this proposal acceptable to you?",
            }
        if action_type == "INQUIRE_SELLER":
            return {
                "type": action_type,
                "internal_reasoning": reason,
                "inquiry_details": "Could you clarify the key terms you need for agreement?",
            }
        if action_type == "WALK_AWAY":
            return {
                "type": action_type,
                "internal_reasoning": reason,
                "verbal_explanation": "I am ending this negotiation without agreement.",
            }
        return {
            "type": "NORMAL_ANSWER",
            "internal_reasoning": reason,
            "answer_details": "I need to clarify details before changing my position.",
        }

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

    def _validate_action_for_turn(
        self,
        raw: Any,
        has_active_offer: bool | None,
        allowed_types: Sequence[str] = (),
        bypass_unified_judge: bool = False,
    ) -> str:
        """Validate extracted JSON against role+offer state (+optional allowed types)."""
        normalized = self._normalize_structured_action(raw)
        json_str = self._extract_json(normalized)
        try:
            validated = self._schema_for_turn(has_active_offer).model_validate_json(json_str)
        except ValidationError as error:
            raise ValueError(f"Schema validation failed: {error}") from error
        canonical = validated.model_dump_json()
        payload = json.loads(canonical)
        action_type = str(payload.get("type", "")).strip().upper()
        details_text = self._collect_text_fields(payload)
        allowed = {str(x).strip().upper() for x in allowed_types}

        # Solves buyer drift once patience horizon exceeded.
        if (
            self._strategy_requires_walk_away(allowed_types)
            and action_type not in {"WALK_AWAY", "ACCEPT_OFFER"}
        ):
            internal_reasoning = str(payload.get("internal_reasoning", "")).strip()
            if not internal_reasoning:
                internal_reasoning = (
                    "Patience horizon exceeded; ending negotiation without agreement."
                )
            coerced_payload = {
                "type": "WALK_AWAY",
                "internal_reasoning": internal_reasoning,
                "verbal_explanation": self._walk_away_explanation_from_payload(payload),
            }
            coerced = self._schema_for_turn(has_active_offer).model_validate(coerced_payload)
            canonical = coerced.model_dump_json()
            payload = json.loads(canonical)
            action_type = "WALK_AWAY"
            details_text = self._collect_text_fields(payload)

        judgement: UnifiedActionJudgement | None = None
        if not bypass_unified_judge:
            judgement = self._judge_action_consistency(
                payload=payload,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
            )

        # Coerce buyer non-offer outputs to WALK_AWAY when intent is explicit.
        if (
            not bypass_unified_judge
            and
            self._role == hdb_schemas.RoleType.BUYER
            and action_type in self._textual_non_offer_action_types()
            and ((not allowed) or ("WALK_AWAY" in allowed))
            and judgement is not None
            and judgement.walk_away_intent == "WALK_AWAY"
        ):
            internal_reasoning = str(payload.get("internal_reasoning", "")).strip()
            if not internal_reasoning:
                internal_reasoning = (
                    "Terminating negotiation without agreement due to lack of conditions "
                    "required to proceed confidently."
                )
            coerced_payload = {
                "type": "WALK_AWAY",
                "internal_reasoning": internal_reasoning,
                "verbal_explanation": self._walk_away_explanation_from_payload(payload),
            }
            coerced = self._schema_for_turn(has_active_offer).model_validate(coerced_payload)
            canonical = coerced.model_dump_json()
            payload = json.loads(canonical)
            action_type = "WALK_AWAY"
            details_text = self._collect_text_fields(payload)

        # WALK_AWAY must be explicit and only after patience horizon is exceeded.
        if action_type == "WALK_AWAY":
            if self._role != hdb_schemas.RoleType.BUYER:
                raise ValueError("WALK_AWAY is only valid for buyers.")
            if not self._strategy_requires_walk_away(allowed_types):
                raise ValueError(
                    "WALK_AWAY chosen before patience horizon was exceeded. "
                    "Continue negotiating unless strategy guidance requires termination."
                )
            if (
                not bypass_unified_judge
                and judgement is not None
                and judgement.walk_away_intent != "WALK_AWAY"
            ):
                raise ValueError(
                    "WALK_AWAY chosen but unified judge did not detect explicit termination intent. "
                    f"Judge explanation: {judgement.explanation}"
                )

        # Coerce offer-related intent to correct action types when mismatches are detected.
        if (
            not bypass_unified_judge
            and
            action_type in self._textual_non_offer_action_types()
            and judgement is not None
            and judgement.contains_offer_intent
        ):
            raise ValueError(
                "Detected offer/counteroffer language inside a non-offer action. "
                "Regenerate and emit MAKE_OFFER or MAKE_COUNTEROFFER with a numeric price field."
            )

        # Verbal intent must align with action type for offer-state decisions.
        if not bypass_unified_judge and judgement is not None:
            expected_type = self._expected_type_from_intent(
                judgement.intent,
                has_active_offer=has_active_offer,
            )
            if expected_type and ((not allowed) or (expected_type in allowed)):
                if action_type != expected_type:
                    raise ValueError(
                        "Reasoning/action mismatch detected by unified judge. "
                        f"Action type={action_type}, judged_intent_requires={expected_type}. "
                        f"Judge explanation: {judgement.explanation}"
                    )

        # Structured price (if any) must align with verbal explanation to prevent price mismatch exploits.
        structured_price = self._structured_price_from_payload(payload)
        if not bypass_unified_judge and judgement is not None:
            judged_price = judgement.proposed_price
            if (
                structured_price is not None
                and judged_price is not None
                and int(structured_price) != int(judged_price)
            ):
                raise ValueError(
                    "Price mismatch between verbal explanation and structured price field. "
                    f"StructuredPrice={structured_price}, VerbalPrice={judged_price}. "
                    f"Judge explanation: {judgement.explanation}"
                )

        if not allowed_types:
            return canonical
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
                    "- Any offer/counteroffer price must be a positive integer.\n"
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
                continue

        fallback_payload = self._deterministic_fallback_payload(
            has_active_offer=has_active_offer,
            allowed_types=allowed_types,
            retry_reason=retry_reason,
        )
        try:
            return self._validate_action_for_turn(
                fallback_payload,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
                bypass_unified_judge=True,
            )
        except ValueError as fallback_error:
            raise ValueError(
                "Failed to regenerate a valid structured action after repeated attempts. "
                f"Deterministic fallback also failed: {fallback_error}"
            ) from fallback_error

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
        target_offer_type = self._preferred_offer_action_type(
            allowed_types=allowed_types,
            has_active_offer=has_active_offer,
        )
        if not target_offer_type:
            raise ValueError(
                "Offer intent was detected, but no offer/counteroffer type is allowed this turn."
            )
        if allowed_set and target_offer_type not in allowed_set:
            raise ValueError(
                f"Offer intent was detected, but {target_offer_type} is not allowed this turn."
            )

        call_to_action = action_spec.call_to_action.replace("{name}", self.get_entity().name)
        allowed_hint = (
            f"\nAllowed action types: {', '.join(allowed_types)}."
            if allowed_types
            else ""
        )
        strict_retry_reason = retry_reason
        deadline = time.monotonic() + self._force_offer_timeout_seconds
        for _ in range(20):
            if time.monotonic() >= deadline:
                break

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
                    "- Include a valid positive integer price field."
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

        forced_price = self._offer_price_from_reservation_bounds(target_offer_type)
        internal_reasoning, verbal_explanation = self._generate_forced_offer_reasoning(
            contexts=contexts,
            action_type=target_offer_type,
            price=forced_price,
            retry_reason=strict_retry_reason,
        )
        forced_payload = self._forced_offer_payload(
            action_type=target_offer_type,
            price=forced_price,
            internal_reasoning=internal_reasoning,
            verbal_explanation=verbal_explanation,
        )
        try:
            return self._validate_action_for_turn(
                forced_payload,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
            )
        except ValueError:
            fallback_internal, fallback_verbal = self._default_forced_reasoning(
                action_type=target_offer_type,
                price=forced_price,
                retry_reason=strict_retry_reason,
            )
            deterministic_payload = self._forced_offer_payload(
                action_type=target_offer_type,
                price=forced_price,
                internal_reasoning=fallback_internal,
                verbal_explanation=fallback_verbal,
            )
            return self._validate_action_for_turn(
                deterministic_payload,
                has_active_offer=has_active_offer,
                allowed_types=allowed_types,
                bypass_unified_judge=True,
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
            "force_offer_timeout_seconds": self._force_offer_timeout_seconds,
            "force_offer_default_price": self._force_offer_default_price,
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
        if 'force_offer_timeout_seconds' in state:
            self._force_offer_timeout_seconds = max(
                1.0, float(state['force_offer_timeout_seconds'])
            )
        if 'force_offer_default_price' in state:
            self._force_offer_default_price = max(
                1.0, float(state['force_offer_default_price'])
            )
        


