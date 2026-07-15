"""Barebones Entity for Student """

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

_DEFAULT_OBSERVATION_HISTORY_LENGTH = 1_000_000
_DEFAULT_SITUATION_PERCEPTION_HISTORY_LENGTH = 25
_DEFAULT_SELF_PERCEPTION_HISTORY_LENGTH = 1_000_000
_DEFAULT_PERSON_BY_SITUATION_HISTORY_LENGTH = 5


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
    component_order = [
        instructions_key,
        student_profile_key,
        goal_key,
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
