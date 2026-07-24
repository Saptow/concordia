"""Small smoke test loop for the education student entity.

Run with:
  uv run concordia/education/tests/run_student_entity_loop.py
"""

from pathlib import Path
import sys

import numpy as np

_CONCORDIA_ROOT = Path(__file__).resolve().parents[2]
if str(_CONCORDIA_ROOT) not in sys.path:
  sys.path.insert(0, str(_CONCORDIA_ROOT))

from concordia.associative_memory import basic_associative_memory
from concordia.education.models.entities.student import Student
from concordia.testing.mock_model import MockModel
from concordia.education.utils.language_model import GeminiLanguageModel
from concordia.typing import entity as entity_lib


def _embedder(text: str) -> np.ndarray:
    """Small deterministic embedder for local smoke tests."""
    text_value = sum(ord(char) for char in text)
    return np.array([
        float(len(text) % 17) / 17.0,
        float(text_value % 97) / 97.0,
        1.0,
    ])


def main() -> None:
    model = MockModel(
        response=(
            'I will focus on the lesson, review what I know, and ask for help '
            'if I get stuck.'
        )
    )
    # model = GeminiLanguageModel()

    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=_embedder
    )

    student_config = Student(params={
        'name': 'Ava',
        'goal': 'Keep improving at data structures and algorithms.',
        'knowledge_tracing': {
            'backend': 'pybkt',
            'student_id': 'stu_001',
            'problem_bank_path': 'concordia/education/data/sample_problem_bank.json',
            'trace_path': 'concordia/education/data/sample_student_traces.json',
            'seed': 42,
            'num_fits': 1,
            'forgets': True,
            'parallel': False,
        },
    })
    student = student_config.build(model=model, memory_bank=memory_bank)

    action_spec = entity_lib.DEFAULT_SPEECH_ACTION_SPEC
    observations = [
        'Ava learns about depth-first search and breadth-first search.',
        'Ava notices she is unsure about complexity analysis.',
    ]

    for turn_number, observation in enumerate(observations, start=1):
        student.observe(observation)
        action = student.act(action_spec=action_spec)
        print(f'Turn {turn_number}')
        print(f'Observation: {observation}')
        print(f'Action: {action}')
        print('-' * 60)


if __name__ == '__main__':
    main()
