"""Adapter utilities for using pykt-toolkit's simpleKT in Concordia.

This module keeps the KT-specific data wrangling out of the agent prefab.
It converts a chronological list of student attempts into the sequence format
expected by pykt-toolkit and exposes a compact knowledge-state snapshot that
the Concordia student can reason over.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from pykt.models.init_model import init_model
from pykt.models.init_model import load_model

JsonMapping = Mapping[str, Any]
PathLike = str | Path

DEFAULT_MODEL_CONFIG = {
    'd_model': 64,
    'n_blocks': 1,
    'dropout': 0.1,
}


@dataclass(frozen=True, slots=True)
class ProblemRecord:
  """Static metadata for a practice problem."""

  problem_id: str
  title: str
  difficulty: str
  knowledge_components: tuple[str, ...]
  prompt: str

  @classmethod
  def from_mapping(cls, data: JsonMapping) -> 'ProblemRecord':
    return cls(
        problem_id=str(data['problem_id']),
        title=str(data.get('title', '')),
        difficulty=str(data.get('difficulty', '')),
        knowledge_components=tuple(
            str(component)
            for component in data.get('knowledge_components', ())
        ),
        prompt=str(data.get('prompt', '')),
    )


@dataclass(frozen=True, slots=True)
class AttemptRecord:
  """One chronological student interaction."""

  event_index: int
  problem_id: str
  correct: int
  attempt_number: int = 1
  hints_used: int = 0
  time_sec: int = 0

  @classmethod
  def from_mapping(cls, data: JsonMapping) -> 'AttemptRecord':
    return cls(
        event_index=int(data['event_index']),
        problem_id=str(data['problem_id']),
        correct=int(data['correct']),
        attempt_number=int(data.get('attempt_number', 1)),
        hints_used=int(data.get('hints_used', 0)),
        time_sec=int(data.get('time_sec', 0)),
    )


@dataclass(frozen=True, slots=True)
class KnowledgeStateSnapshot:
  """Structured KT state that can be surfaced to a Concordia agent."""

  student_id: str
  last_event_index: int
  total_attempts: int
  mastery_by_kc: dict[str, float]
  next_problem_success_prob: dict[str, float]
  weak_kcs: tuple[str, ...]
  ready_problem_ids: tuple[str, ...]
  summary_text: str


class SimpleKTAdapter:
  """Adapter for using pykt-toolkit's simpleKT model in Concordia."""

  def __init__(
      self,
      problem_bank: Sequence[ProblemRecord] | JsonMapping | PathLike,
      *,
      model: Any | None = None,
      model_config: Mapping[str, Any] | None = None,
      emb_type: str = 'qid',
      checkpoint_path: PathLike | None = None,
      max_seq_len: int = 200,
      multi_kc_strategy: str = 'first',
  ):
    self._problems = self._coerce_problem_bank(problem_bank)
    if not self._problems:
      raise ValueError('Problem bank must contain at least one problem.')

    if multi_kc_strategy != 'first':
      raise ValueError(
          "Only multi_kc_strategy='first' is supported in this first pass."
      )

    self._emb_type = emb_type
    self._max_seq_len = max_seq_len
    self._multi_kc_strategy = multi_kc_strategy
    self._model_config = dict(DEFAULT_MODEL_CONFIG | dict(model_config or {}))

    self.problem_by_id = {
        problem.problem_id: problem for problem in self._problems
    }
    self.problem_to_idx: dict[str, int] = {}
    self.kc_to_idx: dict[str, int] = {}
    self.problem_to_primary_kc: dict[str, str] = {}
    self._build_vocab()
    self._model = model or self._maybe_load_model(checkpoint_path)

  @property
  def has_model(self) -> bool:
    return self._model is not None

  @classmethod
  def from_problem_bank_path(
      cls,
      problem_bank_path: PathLike,
      **kwargs,
  ) -> 'SimpleKTAdapter':
    return cls(problem_bank_path, **kwargs)

  @staticmethod
  def load_student_attempts(
      trace_source: JsonMapping | PathLike,
      student_id: str,
  ) -> list[AttemptRecord]:
    """Loads one student's attempts from the sample trace JSON shape."""
    data = _load_json_mapping(trace_source)
    for student in data.get('students', ()):
      if str(student.get('student_id')) != student_id:
        continue
      attempts = [
          AttemptRecord.from_mapping(attempt)
          for attempt in student.get('attempts', ())
      ]
      attempts.sort(key=lambda attempt: attempt.event_index)
      return attempts
    raise KeyError(f'Unknown student_id: {student_id}')

  def build_snapshot(
      self,
      student_id: str,
      attempts: Sequence[AttemptRecord | JsonMapping],
      *,
      top_k: int = 3,
  ) -> KnowledgeStateSnapshot:
    """Builds a compact KT snapshot for agent reasoning."""
    normalized_attempts = self._coerce_attempts(attempts)
    empirical_mastery = self._estimate_empirical_kc_mastery(normalized_attempts)
    predicted_problem_success = self._estimate_problem_success(
        normalized_attempts
    )

    if predicted_problem_success:
      mastery_by_kc = self._aggregate_problem_probs_by_kc(
          predicted_problem_success
      )
    else:
      mastery_by_kc = empirical_mastery

    weak_kcs = tuple(
        kc
        for kc, probability in sorted(
            mastery_by_kc.items(), key=lambda item: item[1]
        )
        if probability < 0.5
    )[:top_k]
    ready_problem_ids = tuple(
        problem_id
        for problem_id, probability in sorted(
            predicted_problem_success.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if probability >= 0.5
    )[:top_k]

    summary_text = self._build_summary_text(
        student_id=student_id,
        mastery_by_kc=mastery_by_kc,
        predicted_problem_success=predicted_problem_success,
        weak_kcs=weak_kcs,
        ready_problem_ids=ready_problem_ids,
        total_attempts=len(normalized_attempts),
    )

    return KnowledgeStateSnapshot(
        student_id=student_id,
        last_event_index=(
            normalized_attempts[-1].event_index if normalized_attempts else 0
        ),
        total_attempts=len(normalized_attempts),
        mastery_by_kc=mastery_by_kc,
        next_problem_success_prob=predicted_problem_success,
        weak_kcs=weak_kcs,
        ready_problem_ids=ready_problem_ids,
        summary_text=summary_text,
    )

  def predict_next_correctness(
      self,
      attempts: Sequence[AttemptRecord | JsonMapping],
      candidate_problem_id: str,
  ) -> float:
    """Predicts correctness for the next attempted problem."""
    if self._model is None:
      raise RuntimeError(
          'No simpleKT model is loaded. Pass a model or checkpoint_path to '
          'SimpleKTAdapter to enable model-backed inference.'
      )

    batch = self.build_inference_batch(attempts, candidate_problem_id)
    self._model.eval()
    with torch.no_grad():
      predictions = self._model(batch)
    return float(predictions[0, -1].item())

  def build_inference_batch(
      self,
      attempts: Sequence[AttemptRecord | JsonMapping],
      candidate_problem_id: str,
  ) -> dict[str, Any]:
    """Builds the dict shape expected by pykt-toolkit simpleKT."""
    if candidate_problem_id not in self.problem_by_id:
      raise KeyError(f'Unknown problem_id: {candidate_problem_id}')

    normalized_attempts = self._coerce_attempts(attempts)
    if not normalized_attempts:
      raise ValueError(
          'simpleKT next-question inference requires at least one historical '
          'attempt before appending a candidate problem.'
      )

    full_sequence = list(normalized_attempts) + [
        AttemptRecord(
            event_index=normalized_attempts[-1].event_index + 1,
            problem_id=candidate_problem_id,
            correct=0,
        )
    ]
    if len(full_sequence) > self._max_seq_len:
      full_sequence = full_sequence[-self._max_seq_len :]

    question_ids = [self.problem_to_idx[item.problem_id] for item in full_sequence]
    concept_ids = [
        self.kc_to_idx[self.problem_to_primary_kc[item.problem_id]]
        for item in full_sequence
    ]
    responses = [int(item.correct) for item in full_sequence]

    qseqs = question_ids[:-1]
    shft_qseqs = question_ids[1:]
    cseqs = concept_ids[:-1]
    shft_cseqs = concept_ids[1:]
    rseqs = responses[:-1]
    shft_rseqs = responses[1:]

    return {
        'qseqs': torch.tensor([qseqs], dtype=torch.long),
        'cseqs': torch.tensor([cseqs], dtype=torch.long),
        'rseqs': torch.tensor([rseqs], dtype=torch.long),
        'shft_qseqs': torch.tensor([shft_qseqs], dtype=torch.long),
        'shft_cseqs': torch.tensor([shft_cseqs], dtype=torch.long),
        'shft_rseqs': torch.tensor([shft_rseqs], dtype=torch.long),
    }

  def _build_vocab(self) -> None:
    next_problem_idx = 1
    next_kc_idx = 1
    for problem in self._problems:
      self.problem_to_idx[problem.problem_id] = next_problem_idx
      next_problem_idx += 1

      if not problem.knowledge_components:
        raise ValueError(
            f'Problem {problem.problem_id} has no knowledge_components.'
        )
      primary_kc = problem.knowledge_components[0]
      self.problem_to_primary_kc[problem.problem_id] = primary_kc
      if primary_kc not in self.kc_to_idx:
        self.kc_to_idx[primary_kc] = next_kc_idx
        next_kc_idx += 1

  def _build_pykt_data_config(self) -> dict[str, Any]:
    # pyKT models expect padding-friendly integer vocabularies. We reserve
    # index 0 locally, so these counts include one extra slot.
    return {
        'num_c': len(self.kc_to_idx) + 1,
        'num_q': len(self.problem_to_idx) + 1,
        'emb_path': '',
    }

  def _coerce_problem_bank(
      self,
      problem_bank: Sequence[ProblemRecord] | JsonMapping | PathLike,
  ) -> list[ProblemRecord]:
    if isinstance(problem_bank, (str, Path)):
      data = _load_json_mapping(problem_bank)
      return [
          ProblemRecord.from_mapping(problem)
          for problem in data.get('problems', ())
      ]
    if isinstance(problem_bank, Mapping):
      return [
          ProblemRecord.from_mapping(problem)
          for problem in problem_bank.get('problems', ())
      ]
    return [
        problem
        if isinstance(problem, ProblemRecord)
        else ProblemRecord.from_mapping(problem)
        for problem in problem_bank
    ]

  def _coerce_attempts(
      self,
      attempts: Sequence[AttemptRecord | JsonMapping],
  ) -> list[AttemptRecord]:
    normalized = [
        attempt
        if isinstance(attempt, AttemptRecord)
        else AttemptRecord.from_mapping(attempt)
        for attempt in attempts
    ]
    normalized.sort(key=lambda attempt: attempt.event_index)
    for attempt in normalized:
      if attempt.problem_id not in self.problem_by_id:
        raise KeyError(
            f'Unknown problem_id in student trace: {attempt.problem_id}'
        )
    return normalized

  def _estimate_empirical_kc_mastery(
      self,
      attempts: Sequence[AttemptRecord],
  ) -> dict[str, float]:
    correct_by_kc: dict[str, int] = defaultdict(int)
    total_by_kc: dict[str, int] = defaultdict(int)

    for attempt in attempts:
      kc = self.problem_to_primary_kc[attempt.problem_id]
      total_by_kc[kc] += 1
      correct_by_kc[kc] += int(attempt.correct)

    mastery_by_kc: dict[str, float] = {}
    for kc in self.kc_to_idx:
      correct = correct_by_kc.get(kc, 0)
      total = total_by_kc.get(kc, 0)
      mastery_by_kc[kc] = (correct + 1.0) / (total + 2.0)
    return mastery_by_kc

  def _estimate_problem_success(
      self,
      attempts: Sequence[AttemptRecord],
  ) -> dict[str, float]:
    if not attempts:
      return {}
    if self._model is None:
      return self._estimate_problem_success_from_empirical_mastery(attempts)

    return {
        problem_id: self.predict_next_correctness(attempts, problem_id)
        for problem_id in self.problem_by_id
    }

  def _estimate_problem_success_from_empirical_mastery(
      self,
      attempts: Sequence[AttemptRecord],
  ) -> dict[str, float]:
    mastery_by_kc = self._estimate_empirical_kc_mastery(attempts)
    return {
        problem_id: mastery_by_kc[self.problem_to_primary_kc[problem_id]]
        for problem_id in self.problem_by_id
    }

  def _aggregate_problem_probs_by_kc(
      self,
      problem_probs: Mapping[str, float],
  ) -> dict[str, float]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for problem_id, probability in problem_probs.items():
      kc = self.problem_to_primary_kc[problem_id]
      bucket[kc].append(float(probability))
    return {
        kc: sum(probabilities) / len(probabilities)
        for kc, probabilities in bucket.items()
    }

  def _build_summary_text(
      self,
      *,
      student_id: str,
      mastery_by_kc: Mapping[str, float],
      predicted_problem_success: Mapping[str, float],
      weak_kcs: Sequence[str],
      ready_problem_ids: Sequence[str],
      total_attempts: int,
  ) -> str:
    strongest = [
        kc
        for kc, _ in sorted(
            mastery_by_kc.items(), key=lambda item: item[1], reverse=True
        )[:2]
    ]
    growing = [
        kc
        for kc, probability in sorted(
            mastery_by_kc.items(), key=lambda item: item[1], reverse=True
        )
        if 0.5 <= probability < 0.75
    ][:2]
    if predicted_problem_success:
      recommended = [
          f'{problem_id} ({predicted_problem_success[problem_id]:.2f})'
          for problem_id in ready_problem_ids[:3]
      ]
    else:
      recommended = []

    return (
        f'Student {student_id} has {total_attempts} recorded attempts. '
        f'Estimated strengths: {", ".join(strongest) or "none yet"}. '
        f'Currently developing: {", ".join(growing) or "none yet"}. '
        f'Weakest areas: {", ".join(weak_kcs) or "none identified"}. '
        f'Recommended next problems: {", ".join(recommended) or "none yet"}.'
    )

  def _maybe_load_model(self, checkpoint_path: PathLike | None) -> Any | None:
    if checkpoint_path is None:
      return None
    return self._load_model_from_checkpoint(checkpoint_path)

  def _load_model_from_checkpoint(self, checkpoint_path: PathLike) -> Any:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
      return load_model(
          'simplekt',
          self._model_config,
          self._build_pykt_data_config(),
          self._emb_type,
          str(checkpoint),
      )

    model = init_model(
        'simplekt',
        self._model_config,
        self._build_pykt_data_config(),
        self._emb_type,
    )
    state_dict = torch.load(str(checkpoint), map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def _load_json_mapping(source: JsonMapping | PathLike) -> JsonMapping:
  if isinstance(source, Mapping):
    return source
  path = Path(source)
  return json.loads(path.read_text(encoding='utf-8'))
