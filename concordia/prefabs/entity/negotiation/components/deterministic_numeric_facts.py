"""Deterministic numeric facts for negotiation decision context."""

from __future__ import annotations

import json
import math
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.components.agent import observation as observation_component
from concordia.hdb_simulation.models import schemas as hdb_schemas
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class DeterministicNumericFacts(action_spec_ignored.ActionSpecIgnored):
  """Computes deterministic price facts from structured observation events."""

  def __init__(
      self,
      role: hdb_schemas.RoleType,
      memory_component_key: str = memory_component.DEFAULT_MEMORY_COMPONENT_KEY,
      strategy_component_key: str = 'NegotiationStrategy',
      uncertain_component_key: str | None = None,
      max_observations: int = 80,
      emit_pre_act_context: bool = False,
      pre_act_label: str = 'deterministic_numeric_facts',
  ):
    super().__init__(pre_act_label=pre_act_label)
    self._role = role
    self._memory_component_key = memory_component_key
    self._strategy_component_key = strategy_component_key
    self._uncertain_component_key = uncertain_component_key
    self._max_observations = max(1, int(max_observations))
    self._emit_pre_act_context = bool(emit_pre_act_context)

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    if self._emit_pre_act_context:
      return super().pre_act(action_spec)
    # Keep deterministic facts available to dependent components while avoiding
    # duplicate context in the global action prompt.
    del action_spec
    _ = self.get_pre_act_value()
    return ''

  @staticmethod
  def _extract_first_json_object(text: str) -> str | None:
    start = text.find('{')
    if start < 0:
      return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
      ch = text[idx]
      if in_string:
        if escaped:
          escaped = False
        elif ch == '\\':
          escaped = True
        elif ch == '"':
          in_string = False
        continue

      if ch == '"':
        in_string = True
      elif ch == '{':
        depth += 1
      elif ch == '}':
        depth -= 1
        if depth == 0:
          return text[start: idx + 1]

    return None

  @staticmethod
  def _coerce_positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
      return None
    if isinstance(value, (int, float)):
      parsed = float(value)
      if math.isfinite(parsed) and parsed > 0.0:
        return parsed
      return None
    if isinstance(value, str):
      cleaned = (
          value.strip()
          .replace('$', '')
          .replace('SGD', '')
          .replace('sgd', '')
          .replace(',', '')
      )
      if not cleaned:
        return None
      try:
        parsed = float(cleaned)
      except ValueError:
        return None
      if math.isfinite(parsed) and parsed > 0.0:
        return parsed
    return None

  def _extract_action_price(self, payload: dict[str, Any]) -> float | None:
    for key in ('counteroffer_price', 'offer_price', 'price_settled'):
      parsed = self._coerce_positive_float(payload.get(key))
      if parsed is not None:
        return parsed
    return None

  def _parse_observed_action(self, memory_text: str) -> tuple[str, dict[str, Any]] | None:
    text = memory_text.strip()
    obs_tag = observation_component.OBSERVATION_TAG
    if text.startswith(f'{obs_tag} '):
      text = text[len(obs_tag) + 1:].strip()

    actor, sep, payload = text.partition(':')
    if not sep:
      return None
    payload_json = self._extract_first_json_object(payload)
    if not payload_json:
      return None
    try:
      action = json.loads(payload_json)
    except json.JSONDecodeError:
      return None
    if not isinstance(action, dict):
      return None
    return actor.strip(), action

  def _resolve_own_reservation(self) -> float | None:
    # Primary source: strategy state current_position.
    try:
      strategy_component = self.get_entity().get_component(self._strategy_component_key)
    except Exception:
      strategy_component = None

    if strategy_component is not None:
      strategy_state = getattr(strategy_component, '_state', None)
      current_position = getattr(strategy_state, 'current_position', None)
      parsed = self._coerce_positive_float(current_position)
      if parsed is not None:
        return parsed

    # Fallback source from uncertainty components.
    if not self._uncertain_component_key:
      return None
    try:
      uncertain_component = self.get_entity().get_component(self._uncertain_component_key)
    except Exception:
      return None

    if self._role == hdb_schemas.RoleType.BUYER:
      beliefs = getattr(uncertain_component, '_beliefs', None)
      if isinstance(beliefs, dict):
        own_res = beliefs.get('own_reservation')
        expected_mean = getattr(own_res, 'get_expected_mean', None)
        parsed = self._coerce_positive_float(expected_mean)
        if parsed is not None:
          return parsed
      return None

    own_reservation = getattr(uncertain_component, '_own_reservation', None)
    return self._coerce_positive_float(own_reservation)

  def _resolve_opponent_reservation(self) -> float | None:
    # Primary source: strategy state opponent_position.
    try:
      strategy_component = self.get_entity().get_component(self._strategy_component_key)
    except Exception:
      strategy_component = None

    if strategy_component is not None:
      strategy_state = getattr(strategy_component, '_state', None)
      opponent_position = getattr(strategy_state, 'opponent_position', None)
      parsed = self._coerce_positive_float(opponent_position)
      if parsed is not None:
        return parsed

    # Fallback source from uncertainty components.
    if not self._uncertain_component_key:
      return None
    try:
      uncertain_component = self.get_entity().get_component(self._uncertain_component_key)
    except Exception:
      return None

    beliefs = getattr(uncertain_component, '_beliefs', None)
    if isinstance(beliefs, dict):
      counterpart_res = beliefs.get('counterpart_reservation')
      expected_mean = getattr(counterpart_res, 'get_expected_mean', None)
      parsed = self._coerce_positive_float(expected_mean)
      if parsed is not None:
        return parsed
    return None

  def _format_reservation_comparison(
      self,
      own_reservation: float | None,
      opponent_reservation: float | None,
  ) -> str:
    if own_reservation is None or opponent_reservation is None:
      return 'Unknown'
    diff = own_reservation - opponent_reservation
    if abs(diff) <= 1e-9:
      return 'Equal'
    if diff > 0:
      return f'OwnAboveOpponent(Diff={self._format_money(diff)})'
    return f'OwnBelowOpponent(Diff={self._format_money(diff)})'

  @staticmethod
  def _format_money(value: float | None) -> str:
    if value is None:
      return 'NA'
    if abs(value - round(value)) <= 1e-9:
      return f'{int(round(value))}'
    return f'{value:.2f}'

  def _make_pre_act_value(self) -> str:
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=memory_component.Memory
    )
    recent_memories = memory.retrieve_recent(limit=self._max_observations)

    active_offer_price: float | None = None
    active_offer_type: str | None = None
    last_action_type: str | None = None

    for mem in recent_memories:
      parsed = self._parse_observed_action(mem)
      if not parsed:
        continue
      _, action = parsed
      action_type = str(action.get('type', '')).strip().upper()
      if not action_type:
        continue

      last_action_type = action_type

      if action_type in {'MAKE_OFFER', 'MAKE_COUNTEROFFER'}:
        price = self._extract_action_price(action)
        if price is not None:
          active_offer_price = price
          active_offer_type = action_type
      elif action_type in {'REJECT_OFFER', 'ACCEPT_OFFER', 'WALK_AWAY'}:
        active_offer_price = None
        active_offer_type = None

    own_reservation = self._resolve_own_reservation()
    opponent_reservation = self._resolve_opponent_reservation()
    reservation_comparison = self._format_reservation_comparison(
        own_reservation=own_reservation,
        opponent_reservation=opponent_reservation,
    )

    lines = [
      'NUMERIC FACTS (DETERMINISTIC):',
      f'OwnVsOpponentReservation={reservation_comparison}',
      f'LastObservedActionType={last_action_type or "NA"}',
    ]

    if active_offer_price is None:
      lines.append('HasActiveOffer=False')
      lines.append('ActiveOfferPrice=NA')
    else:
      lines.append('HasActiveOffer=True')
      lines.append(f'ActiveOfferPrice={self._format_money(active_offer_price)}')
      lines.append(f'ActiveOfferType={active_offer_type or "NA"}')
      if own_reservation is not None:
        offer_minus_reservation = active_offer_price - own_reservation
        lines.append(
            'OfferMinusOwnReservation='
            f'{self._format_money(offer_minus_reservation)}'
        )
        if self._role == hdb_schemas.RoleType.BUYER:
          lines.append(
              f'OfferWithinOwnReservation={str(active_offer_price <= own_reservation)}'
          )
        else:
          lines.append(
              f'OfferMeetsOwnReservation={str(active_offer_price >= own_reservation)}'
          )

    lines.append(
        'Use these computed facts as authoritative for any numeric comparison.'
    )
    return '\n'.join(lines)

  def get_state(self) -> entity_component.ComponentState:
    return {
      'role': self._role.value,
      'memory_component_key': self._memory_component_key,
      'strategy_component_key': self._strategy_component_key,
      'uncertain_component_key': self._uncertain_component_key or '',
      'max_observations': self._max_observations,
      'emit_pre_act_context': int(self._emit_pre_act_context),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    if 'role' in state:
      self._role = hdb_schemas.RoleType(str(state['role']))
    if 'memory_component_key' in state:
      self._memory_component_key = str(state['memory_component_key'])
    if 'strategy_component_key' in state:
      self._strategy_component_key = str(state['strategy_component_key'])
    if 'uncertain_component_key' in state:
      key = str(state['uncertain_component_key'])
      self._uncertain_component_key = key if key else None
    if 'max_observations' in state:
      self._max_observations = max(1, int(state['max_observations']))
    if 'emit_pre_act_context' in state:
      self._emit_pre_act_context = bool(state['emit_pre_act_context'])
