"""Concordia context component for surfacing pyBKT state to the agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

from concordia.education.knowledge_tracing.pybkt.adapter import (
    AttemptRecord,
    KnowledgeStateSnapshot,
    PyBKTAdapter,
)


AttemptsProvider = Callable[[], Sequence[AttemptRecord | Mapping[str, Any]]]
StudentIdProvider = str | Callable[[], str]


class KnowledgeStateContextComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Injects a pyBKT knowledge-state summary into the agent prompt."""

  def __init__(
      self,
      *,
      adapter: PyBKTAdapter,
      student_id: StudentIdProvider,
      attempts_getter: AttemptsProvider,
      top_k: int = 3,
      pre_act_label: str = '\nCurrent knowledge state',
  ):
    super().__init__(pre_act_label)
    self._adapter = adapter
    self._student_id = student_id
    self._attempts_getter = attempts_getter
    self._top_k = top_k
    self._last_snapshot: KnowledgeStateSnapshot | None = None

  def get_last_snapshot(self) -> KnowledgeStateSnapshot | None:
    return self._last_snapshot

  def _make_pre_act_value(self) -> str:
    student_id = (
        self._student_id() if callable(self._student_id) else self._student_id
    )
    snapshot = self._adapter.build_snapshot(
        student_id=student_id,
        attempts=self._attempts_getter(),
        top_k=self._top_k,
    )
    self._last_snapshot = snapshot

    log = {
        'Key': self.get_pre_act_label(),
        'Summary': snapshot.summary_text,
        'State': {
            'student_id': snapshot.student_id,
            'last_event_index': snapshot.last_event_index,
            'total_attempts': snapshot.total_attempts,
            'mastery_by_kc': snapshot.mastery_by_kc,
            'next_problem_success_prob': snapshot.next_problem_success_prob,
            'weak_kcs': list(snapshot.weak_kcs),
            'ready_problem_ids': list(snapshot.ready_problem_ids),
        },
    }
    self._logging_channel(log)
    return snapshot.summary_text

  def get_state(self) -> entity_component.ComponentState:
    return {
        'top_k': self._top_k,
        'last_summary_text': (
            self._last_snapshot.summary_text
            if self._last_snapshot is not None
            else None
        ),
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    top_k = state.get('top_k')
    if top_k is not None:
      self._top_k = int(top_k)
