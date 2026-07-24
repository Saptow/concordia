from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from concordia.components.agent import action_spec_ignored
from concordia.language_model import language_model
from concordia.typing import entity_component


def _slugify(text: str, *, fallback: str) -> str:
  slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
  return slug or fallback


def _clamp_unit_interval(value: float) -> float:
  return max(0.0, min(1.0, float(value)))


def _strip_code_fences(text: str) -> str:
  cleaned = text.strip()
  cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
  cleaned = re.sub(r'\s*```$', '', cleaned)
  return cleaned.strip()


def _extract_json_payload(text: str) -> str:
  cleaned = _strip_code_fences(text)
  if cleaned.startswith('{') or cleaned.startswith('['):
    return cleaned

  object_match = re.search(r'(\{.*\})', cleaned, flags=re.DOTALL)
  if object_match:
    return object_match.group(1)

  array_match = re.search(r'(\[.*\])', cleaned, flags=re.DOTALL)
  if array_match:
    return array_match.group(1)

  return cleaned


def _coerce_metadata(value: Any) -> dict[str, Any]:
  if isinstance(value, Mapping):
    return dict(value)
  return {}


class CourseMaterial(BaseModel):
  """A piece of authoritative course content owned by the game master."""

  material_id: str
  title: str
  material_type: str = 'course_material'
  text: str
  metadata: dict[str, Any] = Field(default_factory=dict)


class CourseConcept(BaseModel):
  """A concept extracted from the course materials."""

  concept_id: str
  name: str
  description: str
  importance: float = 0.5
  prerequisites: list[str] = Field(default_factory=list)
  evidence: list[str] = Field(default_factory=list)


class CourseConceptDigest(BaseModel):
  """The game master's distilled concept map for a course."""

  course_name: str
  summary: str
  concepts: list[CourseConcept]
  source_material_ids: list[str] = Field(default_factory=list)


class StudentKnowledgeArtifact(BaseModel):
  """Historical evidence about a student's learning."""

  artifact_id: str
  title: str
  artifact_type: str = 'student_material'
  text: str
  outcome: str = ''
  metadata: dict[str, Any] = Field(default_factory=dict)


class StudentConceptMastery(BaseModel):
  """A mastery estimate for one course concept."""

  concept_id: str
  score: float
  reasoning: str
  evidence: list[str] = Field(default_factory=list)


class StudentKnowledgeState(BaseModel):
  """A student's mastery profile against the course concept map."""

  student_id: str
  student_name: str
  summary: str
  concept_mastery: list[StudentConceptMastery]
  source_artifact_ids: list[str] = Field(default_factory=list)


class _CourseDigestResponse(BaseModel):
  summary: str
  concepts: list[CourseConcept]


class _StudentKnowledgeResponse(BaseModel):
  summary: str
  concept_mastery: list[StudentConceptMastery]


CourseMaterialInput = CourseMaterial | Mapping[str, Any] | str
StudentArtifactInput = StudentKnowledgeArtifact | Mapping[str, Any] | str
CourseMaterialsSource = (
    Sequence[CourseMaterialInput]
    | Callable[[], Sequence[CourseMaterialInput]]
)


class CourseKnowledgeContextComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging,
):
  """Game-master context that owns the course concept map and learner states."""

  def __init__(
      self,
      *,
      model: language_model.LanguageModel,
      course_name: str,
      course_materials: CourseMaterialsSource,
      max_concepts: int = 12,
      max_material_chars: int = 3500,
      max_student_artifact_chars: int = 2500,
      student_concepts_to_display: int = 4,
      pre_act_label: str = '\nCourse knowledge state',
  ):
    super().__init__(pre_act_label)
    self._model = model
    self._course_name = course_name
    self._course_materials = course_materials
    self._max_concepts = max_concepts
    self._max_material_chars = max_material_chars
    self._max_student_artifact_chars = max_student_artifact_chars
    self._student_concepts_to_display = student_concepts_to_display
    self._course_digest: CourseConceptDigest | None = None
    self._student_states: dict[str, StudentKnowledgeState] = {}

  def digest_course_materials(
      self,
      *,
      force_refresh: bool = False,
  ) -> CourseConceptDigest:
    """Reads course materials and extracts a canonical concept map."""
    if self._course_digest is not None and not force_refresh:
      return self._course_digest

    materials = self._coerce_course_materials(self._resolve_materials_source())
    if not materials:
      raise ValueError('At least one course material is required.')

    prompt = self._build_course_digest_prompt(materials)
    response = self._sample_structured(
        prompt,
        response_schema=_CourseDigestResponse,
        max_tokens=2400,
    )
    digest = self._normalize_course_digest(
        response=response,
        materials=materials,
    )
    if force_refresh:
      self._student_states = {}
    self._course_digest = digest
    return digest

  def register_student_knowledge(
      self,
      *,
      student_id: str,
      student_materials: Sequence[StudentArtifactInput],
      student_name: str | None = None,
      force_refresh: bool = False,
  ) -> StudentKnowledgeState:
    """Preprocesses a student's history into concept-level mastery scores."""
    if student_id in self._student_states and not force_refresh:
      return self._student_states[student_id]

    digest = self.digest_course_materials()
    artifacts = self._coerce_student_artifacts(student_materials)
    if not artifacts:
      raise ValueError('At least one student material is required.')

    prompt = self._build_student_knowledge_prompt(
        digest=digest,
        student_id=student_id,
        student_name=student_name or student_id,
        artifacts=artifacts,
    )
    response = self._sample_structured(
        prompt,
        response_schema=_StudentKnowledgeResponse,
        max_tokens=2600,
    )
    state = self._normalize_student_knowledge_state(
        student_id=student_id,
        student_name=student_name or student_id,
        artifacts=artifacts,
        digest=digest,
        response=response,
    )
    self._student_states[student_id] = state
    return state

  def register_students(
      self,
      students: Mapping[str, Sequence[StudentArtifactInput]],
  ) -> dict[str, StudentKnowledgeState]:
    """Bulk helper for seeding learner knowledge states."""
    return {
        student_id: self.register_student_knowledge(
            student_id=student_id,
            student_materials=materials,
        )
        for student_id, materials in students.items()
    }

  def get_course_digest(self) -> CourseConceptDigest | None:
    return self._course_digest

  def get_student_knowledge_state(
      self,
      student_id: str,
  ) -> StudentKnowledgeState | None:
    return self._student_states.get(student_id)

  def get_registered_students(self) -> Mapping[str, StudentKnowledgeState]:
    return dict(self._student_states)

  def render_student_knowledge_state(
      self,
      student_id: str,
      *,
      max_concepts: int | None = None,
  ) -> str:
    """Returns a concise text summary that other components can reuse."""
    state = self._student_states.get(student_id)
    if state is None:
      return f'No registered knowledge state for {student_id}.'

    concept_count = max_concepts or self._student_concepts_to_display
    ordered = sorted(state.concept_mastery, key=lambda item: item.score)
    weakest = ordered[:concept_count]
    strongest = list(reversed(ordered[-concept_count:]))
    weakest_text = ', '.join(
        f'{item.concept_id}={item.score:.2f}' for item in weakest
    )
    strongest_text = ', '.join(
        f'{item.concept_id}={item.score:.2f}' for item in strongest
    )
    return (
        f'{state.student_name}: {state.summary}\n'
        f'Weakest concepts: {weakest_text or "None"}.\n'
        f'Strongest concepts: {strongest_text or "None"}.'
    )

  def _make_pre_act_value(self) -> str:
    digest = self.digest_course_materials()
    summary = self._build_pre_act_summary(digest)
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Summary': summary,
        'State': self.get_state(),
    })
    return summary

  def _resolve_materials_source(self) -> Sequence[CourseMaterialInput]:
    if callable(self._course_materials):
      return self._course_materials()
    return self._course_materials

  def _coerce_course_materials(
      self,
      materials: Sequence[CourseMaterialInput],
  ) -> list[CourseMaterial]:
    normalized = []
    for idx, material in enumerate(materials, start=1):
      if isinstance(material, CourseMaterial):
        model = material
      elif isinstance(material, str):
        model = CourseMaterial(
            material_id=f'course_material_{idx}',
            title=f'Course material {idx}',
            text=material,
        )
      else:
        title = str(
            material.get('title')
            or material.get('name')
            or f'Course material {idx}'
        )
        model = CourseMaterial(
            material_id=str(
                material.get('material_id')
                or material.get('id')
                or _slugify(title, fallback=f'course_material_{idx}')
            ),
            title=title,
            material_type=str(
                material.get('material_type')
                or material.get('type')
                or 'course_material'
            ),
            text=str(
                material.get('text')
                or material.get('content')
                or material.get('body')
                or material.get('summary')
                or ''
            ),
            metadata=_coerce_metadata(material.get('metadata', {})),
        )
      if model.text.strip():
        normalized.append(model)
    return normalized

  def _coerce_student_artifacts(
      self,
      artifacts: Sequence[StudentArtifactInput],
  ) -> list[StudentKnowledgeArtifact]:
    normalized = []
    for idx, artifact in enumerate(artifacts, start=1):
      if isinstance(artifact, StudentKnowledgeArtifact):
        model = artifact
      elif isinstance(artifact, str):
        model = StudentKnowledgeArtifact(
            artifact_id=f'student_artifact_{idx}',
            title=f'Student artifact {idx}',
            text=artifact,
        )
      else:
        title = str(
            artifact.get('title')
            or artifact.get('name')
            or f'Student artifact {idx}'
        )
        model = StudentKnowledgeArtifact(
            artifact_id=str(
                artifact.get('artifact_id')
                or artifact.get('id')
                or _slugify(title, fallback=f'student_artifact_{idx}')
            ),
            title=title,
            artifact_type=str(
                artifact.get('artifact_type')
                or artifact.get('type')
                or 'student_material'
            ),
            text=str(
                artifact.get('text')
                or artifact.get('content')
                or artifact.get('body')
                or artifact.get('answer')
                or artifact.get('work')
                or ''
            ),
            outcome=str(
                artifact.get('outcome')
                or artifact.get('result')
                or artifact.get('feedback')
                or ''
            ),
            metadata=_coerce_metadata(artifact.get('metadata', {})),
        )
      if model.text.strip() or model.outcome.strip():
        normalized.append(model)
    return normalized

  def _build_course_digest_prompt(
      self,
      materials: Sequence[CourseMaterial],
  ) -> str:
    materials_text = '\n\n'.join(
        (
            f'[{material.material_id}] {material.title}\n'
            f'Type: {material.material_type}\n'
            f'Text:\n{material.text[: self._max_material_chars]}'
        )
        for material in materials
    )
    schema = json.dumps(_CourseDigestResponse.model_json_schema(), indent=2)
    return (
        'You are helping a simulation game master understand a course.\n'
        f'Course name: {self._course_name}\n\n'
        'Given the course materials, extract the key concepts underlying the '
        'course.\n'
        'Focus on the foundational ideas, principles, skills, and recurring '
        'themes that structure the course content.\n'
        f'Return at most {self._max_concepts} concepts.\n'
        'Each concept should have a stable concept_id, a clear description, '
        'an importance score from 0 to 1, any prerequisite concept ids, and '
        'short evidence snippets naming the supporting materials.\n'
        'Prefer concepts that explain multiple parts of the course over '
        'narrow one-off facts.\n'
        'Ground everything in the supplied materials only.\n\n'
        f'Course materials:\n{materials_text}\n\n'
        'Return JSON only using this schema:\n'
        f'{schema}'
    )

  def _build_student_knowledge_prompt(
      self,
      *,
      digest: CourseConceptDigest,
      student_id: str,
      student_name: str,
      artifacts: Sequence[StudentKnowledgeArtifact],
  ) -> str:
    concepts_text = '\n'.join(
        (
            f'- {concept.concept_id}: {concept.name}\n'
            f'  Description: {concept.description}\n'
            f'  Importance: {concept.importance:.2f}\n'
            f'  Prerequisites: {", ".join(concept.prerequisites) or "None"}'
        )
        for concept in digest.concepts
    )
    artifacts_text = '\n\n'.join(
        (
            f'[{artifact.artifact_id}] {artifact.title}\n'
            f'Type: {artifact.artifact_type}\n'
            f'Work:\n{artifact.text[: self._max_student_artifact_chars]}\n'
            f'Outcome/feedback: {artifact.outcome[:600]}'
        )
        for artifact in artifacts
    )
    schema = json.dumps(_StudentKnowledgeResponse.model_json_schema(), indent=2)
    return (
        'You are estimating a student\'s prior knowledge for a simulation '
        'game master.\n'
        f'Course name: {digest.course_name}\n'
        f'Student id: {student_id}\n'
        f'Student name: {student_name}\n\n'
        'Use the course concept map as the grading rubric. For every concept, '
        'rate the student from 0.0 to 1.0 where:\n'
        '0.0 means no evidence of understanding,\n'
        '0.5 means partial or inconsistent understanding,\n'
        '1.0 means strong and well-supported mastery.\n'
        'Keep reasoning concise, evidence-based, and explicit about '
        'uncertainty.\n'
        'Do not invent concepts outside the concept map.\n\n'
        f'Course concept map:\n{concepts_text}\n\n'
        f'Student materials:\n{artifacts_text}\n\n'
        'Return JSON only using this schema:\n'
        f'{schema}'
    )

  def _sample_structured(
      self,
      prompt: str,
      *,
      response_schema: type[BaseModel],
      max_tokens: int,
  ) -> BaseModel:
    structured_sampler = getattr(self._model, 'sample_structured', None)
    if callable(structured_sampler):
      return cast(
          BaseModel,
          structured_sampler(
              prompt,
              response_schema=response_schema,
              max_tokens=max_tokens,
              temperature=0.1,
          ),
      )

    raw = self._model.sample_text(
        prompt=prompt,
        max_tokens=max_tokens,
        terminators=(),
        temperature=0.1,
    )
    return response_schema.model_validate_json(_extract_json_payload(raw))

  def _normalize_course_digest(
      self,
      *,
      response: BaseModel,
      materials: Sequence[CourseMaterial],
  ) -> CourseConceptDigest:
    parsed = cast(_CourseDigestResponse, response)
    concepts = []
    seen_ids = set()
    for idx, concept in enumerate(parsed.concepts, start=1):
      concept_id = _slugify(
          concept.concept_id or concept.name,
          fallback=f'concept_{idx}',
      )
      if concept_id in seen_ids:
        continue
      seen_ids.add(concept_id)
      concepts.append(
          CourseConcept(
              concept_id=concept_id,
              name=concept.name.strip(),
              description=concept.description.strip(),
              importance=_clamp_unit_interval(concept.importance),
              prerequisites=[
                  _slugify(item, fallback=item)
                  for item in concept.prerequisites
                  if item.strip()
              ],
              evidence=[item.strip() for item in concept.evidence if item.strip()],
          )
      )
    if not concepts:
      raise ValueError('The model did not return any course concepts.')
    return CourseConceptDigest(
        course_name=self._course_name,
        summary=parsed.summary.strip(),
        concepts=concepts[: self._max_concepts],
        source_material_ids=[material.material_id for material in materials],
    )

  def _normalize_student_knowledge_state(
      self,
      *,
      student_id: str,
      student_name: str,
      artifacts: Sequence[StudentKnowledgeArtifact],
      digest: CourseConceptDigest,
      response: BaseModel,
  ) -> StudentKnowledgeState:
    parsed = cast(_StudentKnowledgeResponse, response)
    response_by_id = {}
    response_by_name = {}
    for mastery in parsed.concept_mastery:
      concept_id = _slugify(mastery.concept_id, fallback=mastery.concept_id)
      response_by_id[concept_id] = mastery
      response_by_name[concept_id.replace('_', ' ')] = mastery

    normalized_mastery = []
    for concept in digest.concepts:
      mastery = response_by_id.get(concept.concept_id)
      if mastery is None:
        mastery = response_by_name.get(concept.name.lower())
      if mastery is None:
        normalized_mastery.append(
            StudentConceptMastery(
                concept_id=concept.concept_id,
                score=0.0,
                reasoning='No clear evidence of mastery was found in the provided student materials.',
                evidence=[],
            )
        )
        continue
      normalized_mastery.append(
          StudentConceptMastery(
              concept_id=concept.concept_id,
              score=_clamp_unit_interval(mastery.score),
              reasoning=mastery.reasoning.strip(),
              evidence=[
                  item.strip() for item in mastery.evidence if item.strip()
              ],
          )
      )

    return StudentKnowledgeState(
        student_id=student_id,
        student_name=student_name,
        summary=parsed.summary.strip(),
        concept_mastery=normalized_mastery,
        source_artifact_ids=[artifact.artifact_id for artifact in artifacts],
    )

  def _build_pre_act_summary(self, digest: CourseConceptDigest) -> str:
    concept_lines = '\n'.join(
        (
            f'- {concept.concept_id} ({concept.importance:.2f}): '
            f'{concept.description}'
        )
        for concept in digest.concepts
    )
    if self._student_states:
      student_lines = '\n'.join(
          self.render_student_knowledge_state(student_id)
          for student_id in sorted(self._student_states)
      )
    else:
      student_lines = 'No student knowledge states have been registered yet.'

    return (
        f'Course: {digest.course_name}\n'
        f'Course summary: {digest.summary}\n'
        f'Core concepts:\n{concept_lines or "- None"}\n'
        f'Registered student knowledge states:\n{student_lines}'
    )

  def get_state(self) -> entity_component.ComponentState:
    return {
        'course_name': self._course_name,
        'max_concepts': self._max_concepts,
        'max_material_chars': self._max_material_chars,
        'max_student_artifact_chars': self._max_student_artifact_chars,
        'student_concepts_to_display': self._student_concepts_to_display,
        'course_digest': (
            self._course_digest.model_dump(mode='json')
            if self._course_digest is not None
            else None
        ),
        'student_states': {
            student_id: state.model_dump(mode='json')
            for student_id, state in self._student_states.items()
        },
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    course_digest = state.get('course_digest')
    self._course_digest = (
        CourseConceptDigest.model_validate(course_digest)
        if course_digest is not None
        else None
    )
    student_states = cast(Mapping[str, Mapping[str, Any]], state.get(
        'student_states',
        {},
    ))
    self._student_states = {
        student_id: StudentKnowledgeState.model_validate(student_state)
        for student_id, student_state in student_states.items()
    }
