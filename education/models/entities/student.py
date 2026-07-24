"""Barebones Entity for Student """

from collections.abc import Mapping
import dataclasses
from pathlib import Path

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

_DEFAULT_OBSERVATION_HISTORY_LENGTH = 1_000_000
_DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH = 25
_DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH = 5
_DEFAULT_PROBLEM_BANK_PATH = 'concordia/education/data/sample_problem_bank.json'
_DEFAULT_STUDENT_TRACE_PATH = (
    'concordia/education/data/sample_student_traces.json'
)


class StudentProfile:
    """A class representing a student's profile."""
    
    def __init__(
        self,
        name: str = 'Student',
        age: str = '16',
        grade_level: str = 'Secondary 4',
        school: str = 'Not specified',
        subjects: str = 'General studies',
        persona: str = 'Curious and diligent.',
        strengths: str = 'Asks questions and tries hard.',
        challenges: str = (
            'Can lose confidence when unsure.'
        ),
    ):
        self.name = name
        self.age = age
        self.grade_level = grade_level
        self.school = school
        self.subjects = subjects
        self.persona = persona
        self.strengths = strengths
        self.challenges = challenges
    
    def __str__(self) -> str:

        """Formats the student profile as a string."""
        parts = [
            f"Name: {self.name}",
            f"Age: {self.age}",
            f"Grade level: {self.grade_level}",
            f"School: {self.school}",
            f"Subjects: {self.subjects}",
            f"Learner profile: {self.persona}",
            f"Strengths: {self.strengths}",
            f"Challenges: {self.challenges}",
        ]

        return '\n'.join(parts)


def _student_profile(params: Mapping[str, object]) -> str:
  """Formats the student-facing profile block used in prompts."""
  profile = StudentProfile(
      name=str(params.get('name', 'Student')),
      age=str(params.get('age', '16')),
      grade_level=str(params.get('grade_level', 'Secondary 4')),
      school=str(params.get('school', 'Not specified')),
      subjects=str(params.get('subjects', 'General studies')),
      persona=str(params.get('persona', 'Curious and diligent.')),
      strengths=str(params.get('strengths', 'Asks questions and tries hard.')),
      challenges=str(
          params.get('challenges', 'Can lose confidence when unsure.')
      ),
  )
  return str(profile)

@dataclasses.dataclass
class Student(prefab_lib.Prefab):
  """A basic Concordia entity specialized to play the role of a student."""

  description: str = (
      'A student entity that observes classroom events, reflects on who they '
      'are as a learner, and chooses actions a student like them would take.'
  )
  params: Mapping[str, object] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Ava',
          'age': '16',
          'grade_level': 'Secondary 4',
          'school': '',
          'subjects': 'Mathematics, English, Science',
          'persona': 'Curious, diligent, and respectful.',
          'strengths': 'Pays attention in class and asks clarifying questions.',
          'challenges': (
              'Can become hesitant when unsure and may procrastinate on large'
              ' tasks.'
          ),
          'goal': 'Learn effectively and make steady academic progress.',
          'randomize_choices': True,
          'prefix_entity_name': True,
          'observation_history_length': (
              _DEFAULT_OBSERVATION_HISTORY_LENGTH
          ),
          'situation_perception_history_length': (
              _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH
          ),
          'self_perception_history_length': (
              _DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH
          ),
          'person_by_situation_history_length': (
              _DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH
          ),
          'knowledge_tracing': None,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Builds a student entity using Concordia's basic actor pattern."""

    entity_name = str(self.params.get('name', 'Ava'))
    entity_goal = str(
        self.params.get(
            'goal', 'Learn effectively and make steady academic progress.'
        )
    )
    randomize_choices = bool(self.params.get('randomize_choices', True))
    prefix_entity_name = bool(self.params.get('prefix_entity_name', True))
    observation_history_length = int(
        self.params.get(
            'observation_history_length',
            _DEFAULT_OBSERVATION_HISTORY_LENGTH,
        )
    )
    situation_perception_history_length = int(
        self.params.get(
            'situation_perception_history_length',
            _DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH,
        )
    )
    self_perception_history_length = int(
        self.params.get(
            'self_perception_history_length',
            _DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH,
        )
    )
    person_by_situation_history_length = int(
        self.params.get(
            'person_by_situation_history_length',
            _DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH,
        )
    )

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    instructions_key = 'Instructions'
    instructions = agent_components.instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nInstructions',
    )

    student_profile_key = 'StudentProfile'
    student_profile = agent_components.constant.Constant(
        state=_student_profile(self.params),
        pre_act_label='\nStudent profile',
    )

    goal_key = 'Goal'
    goal = agent_components.constant.Constant(
        state=entity_goal,
        pre_act_label='\nLearning goal',
    )

    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory()

    observation_key = (
        agent_components.observation.DEFAULT_OBSERVATION_COMPONENT_KEY
    )
    observation = agent_components.observation.LastNObservations(
        history_length=observation_history_length,
        pre_act_label=(
            '\nEvents so far (ordered from least recent to most recent)'
        ),
    )

    situation_perception_key = 'SituationPerception'
    situation_perception = (
        agent_components.question_of_recent_memories.SituationPerception(
            model=model,
            num_memories_to_retrieve=situation_perception_history_length,
            components=[
                student_profile_key,
                goal_key,
            ],
            pre_act_label=(
                f'\nQuestion: What learning situation is {entity_name} in '
                'right now?\nAnswer'
            ),
        )
    )

    self_perception_key = 'SelfPerception'
    self_perception = (
        agent_components.question_of_recent_memories.SelfPerception(
            model=model,
            num_memories_to_retrieve=self_perception_history_length,
            components=[
                student_profile_key,
                goal_key,
                situation_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: What kind of learner is {entity_name}?\nAnswer'
            ),
        )
    )

    person_by_situation_key = 'PersonBySituation'
    person_by_situation = (
        agent_components.question_of_recent_memories.PersonBySituation(
            model=model,
            num_memories_to_retrieve=person_by_situation_history_length,
            components=[
                student_profile_key,
                goal_key,
                self_perception_key,
                situation_perception_key,
            ],
            pre_act_label=(
                f'\nQuestion: What would a student like {entity_name} do in '
                'this situation to keep learning well?\nAnswer'
            ),
        )
    )

    knowledge_tracing_component = _build_knowledge_tracing_component(
        params=self.params,
        entity_name=entity_name,
    )

    components_of_agent = {
        instructions_key: instructions,
        student_profile_key: student_profile,
        goal_key: goal,
        observation_to_memory_key: observation_to_memory,
        observation_key: observation,
        situation_perception_key: situation_perception,
        self_perception_key: self_perception,
        person_by_situation_key: person_by_situation,
        memory_key: memory,
    }
    if knowledge_tracing_component is not None:
      components_of_agent['KnowledgeState'] = knowledge_tracing_component
    component_order = [
        instructions_key,
        student_profile_key,
        goal_key,
        *(['KnowledgeState'] if knowledge_tracing_component is not None else []),
        observation_key,
        situation_perception_key,
        self_perception_key,
        person_by_situation_key,
    ]

    act_component = agent_components.concat_act_component.ConcatActComponent(
        model=model,
        component_order=component_order,
        randomize_choices=randomize_choices,
        prefix_entity_name=prefix_entity_name,
    )

    return entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
        measurements=self.params.get('measurements'),
    )


Entity = Student


def _build_knowledge_tracing_component(
    *,
    params: Mapping[str, object],
    entity_name: str,
):
  config = params.get('knowledge_tracing')
  if not isinstance(config, Mapping):
    return None

  backend = str(config.get('backend', '')).lower()
  if backend != 'pybkt':
    raise ValueError(
        "Unsupported knowledge_tracing backend. Only 'pybkt' is supported."
    )

  from concordia.education.knowledge_tracing.pybkt import (
      KnowledgeStateContextComponent,
      PyBKTAdapter,
  )

  problem_bank_path = str(
      config.get('problem_bank_path', _DEFAULT_PROBLEM_BANK_PATH)
  )
  trace_path = str(config.get('trace_path', _DEFAULT_STUDENT_TRACE_PATH))
  student_id = str(config.get('student_id', entity_name))
  top_k = int(config.get('top_k', 3))
  fit_kwargs = {
      'seed': int(config.get('seed', 42)),
      'num_fits': int(config.get('num_fits', 1)),
      'forgets': bool(config.get('forgets', True)),
      'parallel': bool(config.get('parallel', False)),
  }

  adapter = PyBKTAdapter.from_json_paths(
      problem_bank_path=_resolve_repo_path(problem_bank_path),
      trace_path=_resolve_repo_path(trace_path),
      fit_kwargs=fit_kwargs,
  )

  def _attempts_getter() -> list:
    return PyBKTAdapter.load_student_attempts(
        _resolve_repo_path(trace_path),
        student_id,
    )

  return KnowledgeStateContextComponent(
      adapter=adapter,
      student_id=student_id,
      attempts_getter=_attempts_getter,
      top_k=top_k,
  )


def _resolve_repo_path(relative_path: str) -> str:
  root = Path(__file__).resolve().parents[4]
  return str((root / relative_path).resolve())
