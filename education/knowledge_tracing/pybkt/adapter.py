"""Adapter utilities for using pyBKT with Concordia education agents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

JsonMapping = Mapping[str, Any]
PathLike = str | Path


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


def _load_json_mapping(source: JsonMapping | PathLike) -> JsonMapping:
  if isinstance(source, Mapping):
    return source
  path = Path(source)
  return json.loads(path.read_text(encoding='utf-8'))


def _patch_pybkt_import_compatibility() -> None:
  import sklearn.metrics._classification as sk_classification

  marker = '_concordia_pybkt_log_loss_patch'
  if getattr(sk_classification._log_loss, marker, False):
    return

  original_log_loss = sk_classification._log_loss

  def _safe_log_loss(*args, **kwargs):
    try:
      return original_log_loss(*args, **kwargs)
    except Exception as exc:
      # pyBKT probes many sklearn metrics at import time and only skips
      # TypeError. Newer sklearn versions raise other exceptions instead.
      raise TypeError(str(exc)) from exc

  setattr(_safe_log_loss, marker, True)
  sk_classification._log_loss = _safe_log_loss


def _patch_pybkt_fit_compatibility() -> None:
  import pyBKT.fit.EM_fit as em_fit

  marker = '_concordia_pybkt_em_patch'
  if getattr(em_fit.run, marker, False):
    return

  def _sequential_run(
      data,
      model,
      trans_softcounts,
      emission_softcounts,
      init_softcounts,
      num_outputs,
      parallel=True,
      fixed={},
  ):
    del trans_softcounts, emission_softcounts, init_softcounts, num_outputs
    del parallel

    alldata = data['data']
    big_t, num_subparts = len(alldata[0]), len(alldata)
    allresources = data['resources']
    starts = data['starts']
    learns = model['learns']
    forgets = model['forgets']
    guesses = model['guesses']
    slips = model['slips']
    lengths = data['lengths']

    prior = model['prior']
    num_sequences = len(starts)
    num_resources = len(learns)
    normalize_lengths = False

    if 'prior' in fixed:
      prior = fixed['prior']
    initial_distn = np.empty((2,), dtype='float')
    initial_distn[0] = 1 - prior
    initial_distn[1] = prior

    if 'learns' in fixed:
      learns = learns * (fixed['learns'] < 0) + fixed['learns'] * (
          fixed['learns'] >= 0
      )
    if 'forgets' in fixed:
      forgets = forgets * (fixed['forgets'] < 0) + fixed['forgets'] * (
          fixed['forgets'] >= 0
      )
    as_matrix = np.empty((2, 2 * num_resources))
    em_fit.interleave(as_matrix[0], 1 - learns, forgets.copy())
    em_fit.interleave(as_matrix[1], learns.copy(), 1 - forgets)

    if 'guesses' in fixed:
      guesses = guesses * (fixed['guesses'] < 0) + fixed['guesses'] * (
          fixed['guesses'] >= 0
      )
    if 'slips' in fixed:
      slips = slips * (fixed['slips'] < 0) + fixed['slips'] * (
          fixed['slips'] >= 0
      )
    b_matrix = np.empty((2, 2 * num_subparts))
    em_fit.interleave(b_matrix[0], 1 - guesses, guesses.copy())
    em_fit.interleave(b_matrix[1], slips.copy(), 1 - slips)

    alpha_out = np.zeros((2, big_t))
    inner_input = {
        'As': as_matrix,
        'Bn': b_matrix,
        'initial_distn': initial_distn,
        'allresources': allresources,
        'starts': starts,
        'lengths': lengths,
        'num_resources': num_resources,
        'num_subparts': num_subparts,
        'alldata': alldata,
        'normalizeLengths': normalize_lengths,
        'alpha_out': alpha_out,
        'sequence_idx_start': 0,
        'sequence_idx_end': num_sequences,
    }

    run_output = em_fit.inner(inner_input)
    all_trans_softcounts = run_output[0]
    all_emission_softcounts = run_output[1]
    all_initial_softcounts = run_output[2]
    total_loglike = float(run_output[3])
    for sequence_start, sequence_length, alpha in run_output[4]:
      alpha_out[:, sequence_start : sequence_start + sequence_length] += alpha

    all_trans_softcounts = all_trans_softcounts.flatten(order='F')
    all_emission_softcounts = all_emission_softcounts.flatten(order='F')
    return {
        'total_loglike': total_loglike,
        'all_trans_softcounts': np.reshape(
            all_trans_softcounts,
            (num_resources, 2, 2),
            order='C',
        ),
        'all_emission_softcounts': np.reshape(
            all_emission_softcounts,
            (num_subparts, 2, 2),
            order='C',
        ),
        'all_initial_softcounts': all_initial_softcounts,
        'alpha_out': alpha_out.flatten(order='F').reshape(
            alpha_out.shape,
            order='C',
        ),
    }

  setattr(_sequential_run, marker, True)
  em_fit.run = _sequential_run


def _import_pybkt_model():
  _patch_pybkt_import_compatibility()
  from pyBKT.models import Model

  _patch_pybkt_fit_compatibility()
  return Model


class PyBKTAdapter:
  """Adapter for using pyBKT model snapshots in Concordia."""

  def __init__(
      self,
      problem_bank: Sequence[ProblemRecord] | JsonMapping | PathLike,
      *,
      trace_source: JsonMapping | PathLike | None = None,
      model: Any | None = None,
      fit_kwargs: Mapping[str, Any] | None = None,
      multi_kc_strategy: str = 'first',
  ):
    self._problems = self._coerce_problem_bank(problem_bank)
    if not self._problems:
      raise ValueError('Problem bank must contain at least one problem.')

    if multi_kc_strategy != 'first':
      raise ValueError(
          "Only multi_kc_strategy='first' is supported in this first pass."
      )

    self._multi_kc_strategy = multi_kc_strategy
    self._fit_kwargs = {
        'seed': 42,
        'num_fits': 1,
        'forgets': True,
        'parallel': False,
    } | dict(fit_kwargs or {})
    self._trace_source = trace_source

    self.problem_by_id = {
        problem.problem_id: problem for problem in self._problems
    }
    self.problem_to_primary_kc: dict[str, str] = {}
    self.kc_to_problem_ids: dict[str, list[str]] = defaultdict(list)
    self._build_vocab()

    self._training_frame: pd.DataFrame | None = None
    self._model = model
    if self._model is None and trace_source is not None:
      self._training_frame = self.build_training_frame(trace_source)
      self._model = self._fit_model(self._training_frame)

  @property
  def has_model(self) -> bool:
    return self._model is not None

  @classmethod
  def from_json_paths(
      cls,
      *,
      problem_bank_path: PathLike,
      trace_path: PathLike,
      **kwargs,
  ) -> 'PyBKTAdapter':
    return cls(problem_bank_path, trace_source=trace_path, **kwargs)

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

  def build_training_frame(
      self,
      trace_source: JsonMapping | PathLike,
  ) -> pd.DataFrame:
    """Converts the JSON student traces into pyBKT's flat dataframe."""
    data = _load_json_mapping(trace_source)
    rows = []
    for student in data.get('students', ()):
      student_id = str(student.get('student_id'))
      attempts = [
          AttemptRecord.from_mapping(attempt)
          for attempt in student.get('attempts', ())
      ]
      attempts.sort(key=lambda attempt: attempt.event_index)
      for attempt in attempts:
        primary_kc = self.problem_to_primary_kc.get(attempt.problem_id)
        if primary_kc is None:
          raise KeyError(
              f'Unknown problem_id in student trace: {attempt.problem_id}'
          )
        rows.append({
            'user_id': student_id,
            'order_id': attempt.event_index,
            'correct': attempt.correct,
            'skill_name': primary_kc,
            'problem_id': attempt.problem_id,
            'attempt_number': attempt.attempt_number,
            'hints_used': attempt.hints_used,
            'time_sec': attempt.time_sec,
        })

    if not rows:
      raise ValueError('Student trace source does not contain any attempts.')

    return pd.DataFrame(rows).sort_values(
        ['user_id', 'order_id'],
        kind='mergesort',
    )

  def build_snapshot(
      self,
      student_id: str,
      attempts: Sequence[AttemptRecord | JsonMapping],
      *,
      top_k: int = 3,
  ) -> KnowledgeStateSnapshot:
    """Builds a compact KT snapshot for agent reasoning."""
    normalized_attempts = self._coerce_attempts(attempts)
    mastery_by_kc = self._estimate_mastery_by_kc(
        student_id=student_id,
        attempts=normalized_attempts,
    )
    predicted_problem_success = self._estimate_problem_success(
        student_id=student_id,
        attempts=normalized_attempts,
    )

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

  def _build_vocab(self) -> None:
    for problem in self._problems:
      if not problem.knowledge_components:
        raise ValueError(
            f'Problem {problem.problem_id} has no knowledge_components.'
        )
      primary_kc = problem.knowledge_components[0]
      self.problem_to_primary_kc[problem.problem_id] = primary_kc
      self.kc_to_problem_ids[primary_kc].append(problem.problem_id)

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

  def _fit_model(self, training_frame: pd.DataFrame) -> Any:
    model_cls = _import_pybkt_model()
    model = model_cls(
        seed=int(self._fit_kwargs.get('seed', 42)),
        num_fits=int(self._fit_kwargs.get('num_fits', 1)),
        parallel=bool(self._fit_kwargs.get('parallel', False)),
    )
    model.fit(
        data=training_frame,
        skills='.*',
        forgets=bool(self._fit_kwargs.get('forgets', True)),
        parallel=bool(self._fit_kwargs.get('parallel', False)),
    )
    return model

  def _build_attempt_frame(
      self,
      student_id: str,
      attempts: Sequence[AttemptRecord],
  ) -> pd.DataFrame:
    rows = []
    for attempt in attempts:
      rows.append({
          'user_id': student_id,
          'order_id': attempt.event_index,
          'correct': attempt.correct,
          'skill_name': self.problem_to_primary_kc[attempt.problem_id],
          'problem_id': attempt.problem_id,
      })
    if not rows:
      return pd.DataFrame(
          columns=['user_id', 'order_id', 'correct', 'skill_name', 'problem_id']
      )
    return pd.DataFrame(rows).sort_values('order_id', kind='mergesort')

  def _candidate_prediction(
      self,
      *,
      student_id: str,
      attempts: Sequence[AttemptRecord],
      problem_id: str,
  ) -> tuple[float, float]:
    if problem_id not in self.problem_by_id:
      raise KeyError(f'Unknown problem_id: {problem_id}')

    if self._model is None:
      kc = self.problem_to_primary_kc[problem_id]
      mastery = self._estimate_empirical_kc_mastery(attempts).get(kc, 0.5)
      return mastery, mastery

    frame = self._build_attempt_frame(student_id, attempts)
    next_order_id = (
        int(frame['order_id'].max()) + 1 if not frame.empty else 1
    )
    candidate_row = pd.DataFrame([{
        'user_id': student_id,
        'order_id': next_order_id,
        'correct': 0,
        'skill_name': self.problem_to_primary_kc[problem_id],
        'problem_id': problem_id,
    }])
    prediction_frame = pd.concat(
        [frame, candidate_row],
        ignore_index=True,
    )
    prediction_frame = prediction_frame.sort_values(
        ['user_id', 'order_id'],
        kind='mergesort',
    )
    predictions = self._model.predict(data=prediction_frame)
    last_row = predictions.iloc[-1]
    return (
        float(last_row['correct_predictions']),
        float(last_row['state_predictions']),
    )

  def _estimate_mastery_by_kc(
      self,
      *,
      student_id: str,
      attempts: Sequence[AttemptRecord],
  ) -> dict[str, float]:
    if self._model is None:
      return self._estimate_empirical_kc_mastery(attempts)

    mastery_by_kc = {}
    for kc, problem_ids in self.kc_to_problem_ids.items():
      representative_problem_id = problem_ids[0]
      _, state_prediction = self._candidate_prediction(
          student_id=student_id,
          attempts=attempts,
          problem_id=representative_problem_id,
      )
      mastery_by_kc[kc] = state_prediction
    return mastery_by_kc

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

    mastery_by_kc = {}
    for kc in self.kc_to_problem_ids:
      correct = correct_by_kc.get(kc, 0)
      total = total_by_kc.get(kc, 0)
      mastery_by_kc[kc] = (correct + 1.0) / (total + 2.0)
    return mastery_by_kc

  def _estimate_problem_success(
      self,
      *,
      student_id: str,
      attempts: Sequence[AttemptRecord],
  ) -> dict[str, float]:
    return {
        problem_id: self._candidate_prediction(
            student_id=student_id,
            attempts=attempts,
            problem_id=problem_id,
        )[0]
        for problem_id in self.problem_by_id
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
    recommended = [
        f'{problem_id} ({predicted_problem_success[problem_id]:.2f})'
        for problem_id in ready_problem_ids[:3]
    ]

    return (
        f'Student {student_id} has {total_attempts} recorded attempts. '
        f'Estimated strengths: {", ".join(strongest) or "none yet"}. '
        f'Currently developing: {", ".join(growing) or "none yet"}. '
        f'Weakest areas: {", ".join(weak_kcs) or "none identified"}. '
        f'Recommended next problems: {", ".join(recommended) or "none yet"}.'
    )
